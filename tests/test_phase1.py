import csv
import io
import sqlite3
import tempfile
import unittest
from contextlib import closing, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src import data_loader
from src.blocking import union
from src.blocking.address_blocking import AddressBlocker
from src.blocking.name_blocking import NameBlocker
from src.blocking.phonetic_blocking import PhoneticBlocker
from src.blocking.tokenset_blocking import TokenSetBlocker
from src.evaluation.recall_scorer import score_candidate_file


class NameBlockingTests(unittest.TestCase):
    """Check normalized-name blocking and country isolation."""

    def test_suffix_variants_match_without_cross_country_pairs(self):
        """Match legal-suffix variants while keeping country keys separate."""
        source1 = pd.DataFrame({
            "entity_id": ["S1-1"],
            "business_name": ["Acme Corporation"],
            "country": ["US"],
        })
        other = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2"],
            "business_name": ["ACME Corp.", "Acme Corporation"],
            "country": ["US", "France"],
        })

        self.assertEqual(NameBlocker().generate(source1, other), {("S1-1", "S2-1")})

    def test_token_and_phonetic_blockers_find_variants(self):
        """Find word-order and phonetic variants through separate strategies."""
        source1 = pd.DataFrame({
            "entity_id": ["S1-1"],
            "business_name": ["Katherine Medical"],
            "country": ["US"],
        })
        other = pd.DataFrame({
            "entity_id": ["S2-1", "S2-2"],
            "business_name": ["Medical Katherine", "Catherine Medical"],
            "country": ["US", "US"],
        })

        self.assertIn(("S1-1", "S2-1"), TokenSetBlocker().generate(source1, other))
        self.assertIn(("S1-1", "S2-2"), PhoneticBlocker().generate(source1, other))

    def test_name_blocker_folds_accents_and_handles_joined_or_typoed_names(self):
        """Match accent, joined-word, and one-character name variations."""
        source1 = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": ["B+ Retail Inc", "Prime Money", "Christ Chapel"],
            "country": ["US", "US", "US"],
        })
        other = pd.DataFrame({
            "entity_id": ["S3-1", "S3-2", "S3-3"],
            "business_name": ["B+ Rétail Incorporated", "@primemoney", "Christ Chape1"],
            "country": ["US", "US", "US"],
        })

        pairs = NameBlocker().generate(source1, other)

        self.assertIn(("S1-1", "S3-1"), pairs)
        self.assertIn(("S1-2", "S3-2"), pairs)
        self.assertIn(("S1-3", "S3-3"), pairs)

    def test_address_blocker_matches_number_and_street_variants(self):
        """Match noisy business names when normalized address keys agree."""
        source1 = pd.DataFrame({
            "entity_id": ["S1-1", "S1-2", "S1-3"],
            "business_name": ["Prime Money", "Moore Bitwise Inc", "Christ Chapel"],
            "business_address": [
                "17560 Ellis Road, Tahlequah, OK",
                "337 Oakland Avenue, Michigan City, IN",
                "2100 Cameron Drive, Unit Apartment G, Dundalk, MD",
            ],
            "country": ["US", "US", "US"],
        })
        other = pd.DataFrame({
            "entity_id": ["S2-1", "S3-1", "S3-2"],
            "business_name": ["@primemoney", "moorebitwise.com", "Ectosyn dba Christ Chapel"],
            "business_address": [
                "TAHLEQUAH, OK, 0017560 ELLIS ROAD",
                "0337 Oakland Ave, Michigan City, Indiana",
                "2100 Cameron Dr, Unit APARTMENT G, Dundalk, MD",
            ],
            "country": ["US", "US", "US"],
        })

        pairs = AddressBlocker().generate(source1, other)

        self.assertIn(("S1-1", "S2-1"), pairs)
        self.assertIn(("S1-2", "S3-1"), pairs)
        self.assertIn(("S1-3", "S3-2"), pairs)

    def test_recall_scorer_counts_true_links_and_singletons(self):
        """Score row-aligned candidate data when empty cells parse as missing values."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            candidate_path = Path(temporary_dir) / "candidate_pairs.tsv"
            truth_path = Path(temporary_dir) / "ground_truth.tsv"
            pd.DataFrame({
                "source1_entity_id": ["S1-2", "S1-1"],
                "candidate_entity_ids": ["", "S2-1"],
            }).to_csv(candidate_path, sep="\t", index=False)
            pd.DataFrame({
                "source1_entity_id": ["S1-1", "S1-2"],
                "matched_entity_ids": ["S2-1,S3-1", ""],
            }).to_csv(truth_path, sep="\t", index=False)

            result = score_candidate_file(candidate_path, truth_path)

        self.assertEqual(result["matched_in_candidates"], 1)
        self.assertEqual(result["total_true_matches"], 2)
        self.assertEqual(result["recall"], 0.5)

    def test_union_writes_candidates_and_scores_full_recall(self):
        """Write a row-aligned candidate file from streamed sources."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            candidate_path = root / "candidate_pairs.tsv"
            truth_path = root / "ground_truth.tsv"
            sources = {
                "source1": pd.DataFrame({
                    "entity_id": ["S1-1", "S1-2"],
                    "business_name": ["Acme Corporation", "Solo Shop"],
                    "business_address": ["1 Main Street", "2 Oak Road"],
                    "country": ["US", "US"],
                }),
                "source2": pd.DataFrame({
                    "entity_id": ["S2-1"],
                    "business_name": ["Acme Corp."],
                    "business_address": ["1 Main St"],
                    "country": ["US"],
                }),
                "source3": pd.DataFrame({
                    "entity_id": ["S3-1"],
                    "business_name": ["Acme Incorporated"],
                    "business_address": ["1 Main Street"],
                    "country": ["US"],
                }),
            }
            pd.DataFrame({
                "source1_entity_id": ["S1-2", "S1-1"],
                "matched_entity_ids": ["", "S2-1,S3-1"],
            }).to_csv(truth_path, sep="\t", index=False)

            def fake_iter_source(split, source, columns=None):
                yield sources[source]

            for source, frame in sources.items():
                frame.to_csv(root / f"train_{source}.tsv", sep="\t", index=False)

            with patch.object(union, "iter_source", side_effect=fake_iter_source), \
                    patch.object(union, "TRAIN_DIR", root), \
                    patch.object(data_loader, "TRAIN_DIR", root), \
                    patch.object(union, "CACHE_DIR", root / "cache"), \
                    patch.object(union, "OUTPUT_DIR", root / "output"), \
                    patch.object(union, "BLOCKING_STRATEGIES", ["name"]):
                result = union.generate_candidates("train", output_path=candidate_path)

            score = score_candidate_file(candidate_path, truth_path)

        self.assertEqual(result["candidate_count"], 2)
        self.assertEqual(score["recall"], 1.0)

    def test_strategy_index_cache_is_reused_for_different_output_paths(self):
        """Reuse an input/strategy index when only the output filename changes."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sources = {
                "source1": pd.DataFrame({
                    "entity_id": ["S1-1", "S1-2"],
                    "business_name": ["Acme Corporation", "Solo Shop"],
                    "business_address": ["1 Main Street", "2 Oak Road"],
                    "country": ["US", "US"],
                }),
                "source2": pd.DataFrame({
                    "entity_id": ["S2-1"],
                    "business_name": ["Acme Corp."],
                    "business_address": ["1 Main St"],
                    "country": ["US"],
                }),
                "source3": pd.DataFrame({
                    "entity_id": ["S3-1"],
                    "business_name": ["Acme Incorporated"],
                    "business_address": ["1 Main Street"],
                    "country": ["US"],
                }),
            }
            for source, frame in sources.items():
                frame.to_csv(root / f"train_{source}.tsv", sep="\t", index=False)

            with patch.object(union, "TRAIN_DIR", root), \
                    patch.object(data_loader, "TRAIN_DIR", root), \
                    patch.object(union, "CACHE_DIR", root / "cache"), \
                    patch.object(union, "OUTPUT_DIR", root / "output"), \
                    patch.object(union, "BLOCKING_STRATEGIES", ["name"]):
                first_log = io.StringIO()
                with redirect_stdout(first_log):
                    first = union.generate_candidates(
                        "train", output_path=root / "output" / "first.tsv"
                    )
                second_log = io.StringIO()
                with redirect_stdout(second_log):
                    second = union.generate_candidates(
                        "train", output_path=root / "output" / "renamed.tsv"
                    )

        self.assertEqual(first["candidate_count"], 2)
        self.assertEqual(second["candidate_count"], 2)
        self.assertIn("[CACHE MISS] Building name index for source2", first_log.getvalue())
        cache_hit_line = next(
            line for line in second_log.getvalue().splitlines()
            if "[CACHE HIT]" in line and "name index for source2" in line
        )
        print(cache_hit_line)

    def test_strategy_index_resumes_after_committed_chunk(self):
        """Resume an interrupted index from its last durable source-row checkpoint."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            sources = {
                "source1": pd.DataFrame({
                    "entity_id": ["S1-1", "S1-2"],
                    "business_name": ["Acme Corporation", "Solo Shop"],
                    "business_address": ["1 Main Street", "2 Oak Road"],
                    "country": ["US", "US"],
                }),
                "source2": pd.DataFrame({
                    "entity_id": ["S2-1"],
                    "business_name": ["Acme Corp."],
                    "business_address": ["1 Main St"],
                    "country": ["US"],
                }),
                "source3": pd.DataFrame({
                    "entity_id": ["S3-1"],
                    "business_name": ["Acme Incorporated"],
                    "business_address": ["1 Main Street"],
                    "country": ["US"],
                }),
            }
            for source, frame in sources.items():
                frame.to_csv(root / f"train_{source}.tsv", sep="\t", index=False)

            real_iter_source = data_loader.iter_source

            def interrupted_iter_source(split, source, columns=None, chunksize=100_000):
                for chunk in real_iter_source(split, source, columns, chunksize):
                    yield chunk
                    if source == "source2":
                        raise RuntimeError("simulated interruption after committed chunk")

            with patch.object(union, "TRAIN_DIR", root), \
                    patch.object(data_loader, "TRAIN_DIR", root), \
                    patch.object(union, "CACHE_DIR", root / "cache"), \
                    patch.object(union, "OUTPUT_DIR", root / "output"), \
                    patch.object(union, "BLOCKING_STRATEGIES", ["name"]), \
                    patch.object(union, "iter_source", side_effect=interrupted_iter_source):
                with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                    union.generate_candidates("train", output_path=root / "output" / "interrupted.tsv")

            resume_log = io.StringIO()
            with patch.object(union, "TRAIN_DIR", root), \
                    patch.object(data_loader, "TRAIN_DIR", root), \
                    patch.object(union, "CACHE_DIR", root / "cache"), \
                    patch.object(union, "OUTPUT_DIR", root / "output"), \
                    patch.object(union, "BLOCKING_STRATEGIES", ["name"]), \
                    redirect_stdout(resume_log):
                result = union.generate_candidates("train", output_path=root / "output" / "resumed.tsv")

        self.assertEqual(result["candidate_count"], 2)
        self.assertIn("[CACHE MISS] Building name index for source2 (resuming at row 1)", resume_log.getvalue())
        self.assertIn("[CACHE BUILT] Saved name index for source2", resume_log.getvalue())

    def test_strategy_index_finalization_is_idempotent(self):
        """Reuse already inserted eligible keys when finalization is retried."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            building_path = root / "building.sqlite"
            cache_path = root / "complete.sqlite"
            connection, _ = union._open_index_builder(building_path, "cache-key")
            connection.execute("INSERT INTO postings VALUES ('us', 'exact:acme', 'S2-1')")
            connection.execute("INSERT INTO eligible_keys VALUES ('us', 'exact:acme')")
            connection.commit()

            union._finalize_strategy_index(
                "name", "source2", connection, 1, "cache-key", building_path, cache_path
            )

            with closing(sqlite3.connect(cache_path)) as completed:
                metadata = completed.execute(
                    "SELECT complete, source_rows FROM cache_metadata"
                ).fetchone()
                eligible_count = completed.execute("SELECT COUNT(*) FROM eligible_keys").fetchone()[0]

        self.assertEqual(metadata, (1, 1))
        self.assertEqual(eligible_count, 1)

    def test_candidate_merge_deduplicates_and_preserves_row_alignment(self):
        """Merge additional candidates by aligned source-1 rows without duplicates."""
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            base_path = root / "base.tsv"
            addition_path = root / "addition.tsv"
            output_path = root / "merged.tsv"
            header = ["source1_entity_id", "candidate_entity_ids"]
            with base_path.open("w", encoding="utf-8", newline="") as base_file:
                writer = csv.writer(base_file, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                writer.writerows([["S1-1", "S2-1,S3-1"], ["S1-2", ""]])
            with addition_path.open("w", encoding="utf-8", newline="") as addition_file:
                writer = csv.writer(addition_file, delimiter="\t", lineterminator="\n")
                writer.writerow(header)
                writer.writerows([["S1-1", "S3-1,S2-2"], ["S1-2", "S3-2"]])

            candidate_count, row_count = union.merge_candidate_files(base_path, addition_path, output_path)
            merged = pd.read_csv(output_path, sep="\t", dtype="string[python]")

        self.assertEqual((candidate_count, row_count), (4, 2))
        self.assertEqual(merged.loc[0, "candidate_entity_ids"], "S2-1,S2-2,S3-1")
        self.assertEqual(merged.loc[1, "candidate_entity_ids"], "S3-2")


if __name__ == "__main__":
    unittest.main()