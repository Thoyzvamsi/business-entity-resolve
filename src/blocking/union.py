"""Combines configured blockers and writes the training candidate set.

This module owns blocker orchestration and output; it must never classify pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

import pandas as pd

from src.blocking.base import normalize_business_name
from src.blocking.address_blocking import AddressBlocker
from src.blocking.name_blocking import NameBlocker
from src.blocking.phonetic_blocking import PhoneticBlocker
from src.blocking.tokenset_blocking import TokenSetBlocker
from src.config import (
    BLOCKING_BATCH_SIZE,
    BLOCKING_CACHE_DIR,
    BLOCKING_CACHE_VERSION,
    BLOCKING_INSERT_BATCH_SIZE,
    BLOCKING_MATCH_BATCH_SIZE,
    BLOCKING_STRATEGIES,
    MAX_BLOCK_POSTINGS,
    MAX_CANDIDATES_PER_ENTITY,
    MIN_NAME_NGRAM_LENGTH,
    MIN_TOKEN_LENGTH,
    NAME_NGRAM_SIZE,
    OUTPUT_DIR,
    TRAIN_CANDIDATE_PATH,
    TRAIN_DIR,
)
from src.data_loader import iter_source
from src.evaluation.recall_scorer import score_candidate_file

BLOCKER_REGISTRY = {
    "address": AddressBlocker,
    "name": NameBlocker,
    "phonetic": PhoneticBlocker,
    "tokenset": TokenSetBlocker,
}
CACHE_DIR = BLOCKING_CACHE_DIR


def _create_database(connection: sqlite3.Connection) -> None:
    """Create bounded-cache tables for source rows, postings, and candidates."""
    connection.execute("PRAGMA journal_mode = OFF")
    connection.execute("PRAGMA synchronous = OFF")
    connection.execute("PRAGMA temp_store = FILE")
    connection.execute("PRAGMA cache_size = -65536")
    connection.execute("CREATE TABLE candidates (source1_id TEXT, entity_id TEXT, PRIMARY KEY (source1_id, entity_id)) WITHOUT ROWID")
    connection.execute("CREATE TABLE batch_keys (source1_id TEXT, country TEXT, strategy TEXT, block_key TEXT, PRIMARY KEY (source1_id, strategy, block_key)) WITHOUT ROWID")


def _strategy_instances(strategy_names: list[str] | None = None) -> list[tuple[str, Any]]:
    """Create blocker instances from the configured strategy registry."""
    names = BLOCKING_STRATEGIES if strategy_names is None else strategy_names
    return [(name, BLOCKER_REGISTRY[name]()) for name in names]


def _index_cache_identity(
    split: str,
    source: str,
    strategy: str,
    blocker: Any,
) -> tuple[Path, str]:
    """Return a cache path keyed by source metadata, strategy, and key settings."""
    source_path = _source_tsv_path(split, source)
    stat = source_path.stat()
    identity = {
        "cache_version": BLOCKING_CACHE_VERSION,
        "source_path": str(source_path.resolve()),
        "source_size": stat.st_size,
        "source_mtime_ns": stat.st_mtime_ns,
        "split": split,
        "source": source,
        "strategy": strategy,
        "blocker": type(blocker).__name__,
        "max_block_postings": MAX_BLOCK_POSTINGS,
        "min_token_length": MIN_TOKEN_LENGTH,
        "name_ngram_size": NAME_NGRAM_SIZE,
        "min_name_ngram_length": MIN_NAME_NGRAM_LENGTH,
    }
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = CACHE_DIR / f"{split}_{source}_{strategy}_{cache_key}.sqlite"
    return cache_path, cache_key


def _cached_index_metadata(cache_path: Path, cache_key: str) -> tuple[str, int] | None:
    """Return valid cache build metadata, or None when the index is unusable."""
    if not cache_path.exists():
        return None
    try:
        with closing(sqlite3.connect(cache_path)) as connection:
            row = connection.execute(
                "SELECT built_at, source_rows FROM cache_metadata WHERE cache_key = ? AND complete = 1",
                (cache_key,),
            ).fetchone()
    except sqlite3.Error:
        return None
    return None if row is None else (str(row[0]), int(row[1]))


def _source_tsv_path(split: str, source: str) -> Path:
    """Return the TSV path for one split/source pair."""
    base = TRAIN_DIR if split == "train" else TRAIN_DIR.parent / "test"
    return base / f"{split}_{source}.tsv"


def _cache_row_count_plausible(source_path: Path, source_rows: int) -> bool:
    """Reject tiny complete indexes keyed to large source files."""
    if source_rows < 0:
        return False
    if source_path.stat().st_size >= 1_000_000 and source_rows < 1_000:
        return False
    return True


def _invalidate_cache_file(path: Path, reason: str) -> None:
    """Delete a poisoned or incomplete cache file and explain why."""
    print(f"[CACHE INVALID] Removing {path.name}: {reason}", flush=True)
    path.unlink(missing_ok=True)


def _open_index_builder(path: Path, cache_key: str) -> tuple[sqlite3.Connection, int]:
    """Open a matching incomplete index or create a resumable builder database."""
    if path.exists():
        try:
            connection = sqlite3.connect(path)
            row = connection.execute(
                "SELECT source_rows FROM cache_metadata "
                "WHERE cache_key = ? AND complete = 0",
                (cache_key,),
            ).fetchone()
            if row is not None:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
                connection.execute("PRAGMA cache_size = -262144")
                return connection, int(row[0])
            connection.close()
        except sqlite3.Error:
            try:
                connection.close()
            except (sqlite3.Error, UnboundLocalError):
                pass
        path.unlink(missing_ok=True)

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    connection.execute("PRAGMA temp_store = MEMORY")
    connection.execute("PRAGMA cache_size = -262144")
    connection.execute("CREATE TABLE postings (country TEXT, block_key TEXT, entity_id TEXT, PRIMARY KEY (country, block_key, entity_id)) WITHOUT ROWID")
    connection.execute("CREATE TABLE eligible_keys (country TEXT, block_key TEXT, PRIMARY KEY (country, block_key)) WITHOUT ROWID")
    connection.execute("CREATE TABLE cache_metadata (cache_key TEXT, built_at TEXT, complete INTEGER, source_rows INTEGER)")
    connection.execute(
        "INSERT INTO cache_metadata VALUES (?, '', 0, 0)",
        (cache_key,),
    )
    connection.commit()
    return connection, 0


def _flush_index_batch(connection: sqlite3.Connection, pending: list[tuple[str, str, str]]) -> None:
    """Persist one bounded batch of strategy posting keys."""
    if pending:
        connection.executemany("INSERT OR IGNORE INTO postings VALUES (?, ?, ?)", pending)
        pending.clear()


def _append_index_chunk(
    chunk: pd.DataFrame,
    row_start: int,
    blockers: list[tuple[str, Any]],
    builders: dict[str, sqlite3.Connection],
    pending: dict[str, list[tuple[str, str, str]]],
    resume_rows: dict[str, int],
) -> int:
    """Append rows after each strategy's last committed source checkpoint."""
    columns = ["entity_id", "business_name", "country"]
    if "business_address" in chunk.columns:
        columns.append("business_address")
    row_count = len(chunk)
    for offset, row in enumerate(chunk[columns].itertuples(index=False, name=None)):
        absolute_row = row_start + offset
        entity_id, name, country, *address_values = row
        if pd.isna(name) or pd.isna(country):
            continue
        normalized_name = normalize_business_name(str(name))
        address = address_values[0] if address_values else ""
        normalized_address = normalize_business_name(str(address)) if pd.notna(address) else ""
        country_key = str(country).casefold().strip()
        for strategy, blocker in blockers:
            if strategy not in builders or absolute_row < resume_rows[strategy]:
                continue
            pending[strategy].extend(
                (country_key, key, str(entity_id))
                for key in set(blocker.blocking_keys(normalized_name, normalized_address))
            )
            if len(pending[strategy]) >= BLOCKING_INSERT_BATCH_SIZE:
                _flush_index_batch(builders[strategy], pending[strategy])
    return row_count


def _checkpoint_index_chunk(
    source_rows: int,
    builders: dict[str, sqlite3.Connection],
    pending: dict[str, list[tuple[str, str, str]]],
    cache_keys: dict[str, str],
) -> None:
    """Commit index keys and their source-row checkpoint atomically."""
    for strategy, connection in builders.items():
        _flush_index_batch(connection, pending[strategy])
        connection.execute(
            "UPDATE cache_metadata SET source_rows = ? WHERE cache_key = ? AND complete = 0",
            (source_rows, cache_keys[strategy]),
        )
        connection.commit()


def _finalize_strategy_index(
    strategy: str,
    source: str,
    connection: sqlite3.Connection,
    source_rows: int,
    cache_key: str,
    building_path: Path,
    cache_path: Path,
) -> None:
    """Finalize an eligible-key index and atomically publish the complete cache."""
    connection.execute(
        "INSERT OR IGNORE INTO eligible_keys SELECT country, block_key FROM postings "
        "GROUP BY country, block_key HAVING COUNT(*) <= ?",
        (MAX_BLOCK_POSTINGS,),
    )
    built_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    connection.execute(
        "UPDATE cache_metadata SET built_at = ?, complete = 1, source_rows = ? WHERE cache_key = ?",
        (built_at, source_rows, cache_key),
    )
    connection.commit()
    connection.close()
    os.replace(building_path, cache_path)
    print(f"[CACHE BUILT] Saved {strategy} index for {source} (built {built_at})", flush=True)


def _ensure_strategy_indexes(
    split: str,
    source: str,
    blockers: list[tuple[str, Any]],
) -> tuple[dict[str, Path], int]:
    """Load valid per-strategy caches or build missing indexes in one source pass."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_paths: dict[str, Path] = {}
    cache_keys: dict[str, str] = {}
    builders: dict[str, sqlite3.Connection] = {}
    building_paths: dict[str, Path] = {}
    pending: dict[str, list[tuple[str, str, str]]] = {}
    resume_rows: dict[str, int] = {}
    source_rows: int | None = None
    source_path = _source_tsv_path(split, source)
    for strategy, blocker in blockers:
        cache_path, cache_key = _index_cache_identity(split, source, strategy, blocker)
        cache_paths[strategy] = cache_path
        cache_keys[strategy] = cache_key
        building_path = cache_path.with_name(f"{cache_path.stem}.building.sqlite")
        metadata = _cached_index_metadata(cache_path, cache_key)
        if metadata is not None:
            built_at, cached_rows = metadata
            if not _cache_row_count_plausible(source_path, cached_rows):
                _invalidate_cache_file(
                    cache_path,
                    f"implausible source_rows={cached_rows} for {source_path.name}",
                )
            else:
                if source_rows is not None and source_rows != cached_rows:
                    raise ValueError(f"Cached source row counts disagree for {split}/{source}")
                source_rows = cached_rows
                print(f"[CACHE HIT] Loaded {strategy} index for {source} (built {built_at})", flush=True)
                continue
        completed_build = _cached_index_metadata(building_path, cache_key)
        if completed_build is not None:
            built_at, cached_rows = completed_build
            if not _cache_row_count_plausible(source_path, cached_rows):
                _invalidate_cache_file(
                    building_path,
                    f"implausible source_rows={cached_rows} for {source_path.name}",
                )
            else:
                os.replace(building_path, cache_path)
                if source_rows is not None and source_rows != cached_rows:
                    raise ValueError(f"Cached source row counts disagree for {split}/{source}")
                source_rows = cached_rows
                print(f"[CACHE HIT] Loaded {strategy} index for {source} (built {built_at})", flush=True)
                continue
        connection, checkpoint = _open_index_builder(building_path, cache_key)
        suffix = f" (resuming at row {checkpoint:,})" if checkpoint else ""
        print(f"[CACHE MISS] Building {strategy} index for {source}{suffix}", flush=True)
        building_paths[strategy] = building_path
        builders[strategy] = connection
        resume_rows[strategy] = checkpoint
        pending[strategy] = []

    if not builders:
        return cache_paths, source_rows or 0

    initial_stat = source_path.stat()
    try:
        checkpoint_values = set(resume_rows.values())
        if len(checkpoint_values) == 1:
            checkpointed_rows = next(iter(checkpoint_values))
            if checkpointed_rows > 0:
                actual_rows = _count_source_rows(split, source)
                if checkpointed_rows > actual_rows:
                    raise ValueError(
                        f"Index checkpoint {checkpointed_rows} exceeds {actual_rows} rows for {split}/{source}"
                    )
                if checkpointed_rows == actual_rows:
                    print(
                        f"{source}: finalizing {len(builders)} strateg"
                        f"{'y' if len(builders) == 1 else 'ies'} "
                        f"from checkpoint ({actual_rows:,} rows)",
                        flush=True,
                    )
                    for strategy, connection in builders.items():
                        _finalize_strategy_index(
                            strategy,
                            source,
                            connection,
                            actual_rows,
                            cache_keys[strategy],
                            building_paths[strategy],
                            cache_paths[strategy],
                        )
                    if source_rows is not None and source_rows != actual_rows:
                        raise ValueError(
                            f"Cached and newly indexed source row counts disagree for {split}/{source}"
                        )
                    return cache_paths, actual_rows

        row_count = 0
        for chunk in iter_source(split, source):
            row_count += _append_index_chunk(
                chunk, row_count, blockers, builders, pending, resume_rows
            )
            _checkpoint_index_chunk(row_count, builders, pending, cache_keys)
            print(f"{source}: indexed {row_count:,} rows", flush=True)

        final_stat = source_path.stat()
        if (initial_stat.st_size, initial_stat.st_mtime_ns) != (final_stat.st_size, final_stat.st_mtime_ns):
            raise RuntimeError(f"Source file changed while indexing: {source_path}")

        for strategy, connection in builders.items():
            _finalize_strategy_index(
                strategy,
                source,
                connection,
                row_count,
                cache_keys[strategy],
                building_paths[strategy],
                cache_paths[strategy],
            )
        if source_rows is not None and source_rows != row_count:
            raise ValueError(f"Cached and newly indexed source row counts disagree for {split}/{source}")
        return cache_paths, row_count
    except BaseException:
        for connection in builders.values():
            try:
                connection.close()
            except sqlite3.Error:
                pass
        raise


def _add_batch_keys(
    connection: sqlite3.Connection,
    chunk: pd.DataFrame,
    blockers: list[tuple[str, Any]],
) -> None:
    """Insert strategy keys for one source-1 batch into SQLite."""
    keys: list[tuple[str, str, str, str]] = []
    columns = ["entity_id", "business_name", "country"]
    if "business_address" in chunk.columns:
        columns.append("business_address")
    for row in chunk[columns].itertuples(index=False, name=None):
        source1_id, name, country, *address_values = row
        if pd.isna(name) or pd.isna(country):
            continue
        normalized_name = normalize_business_name(str(name))
        address = address_values[0] if address_values else ""
        normalized_address = normalize_business_name(str(address)) if pd.notna(address) else ""
        country_key = str(country).casefold().strip()
        for strategy, blocker in blockers:
            keys.extend(
                (str(source1_id), country_key, strategy, block_key)
                for block_key in set(blocker.blocking_keys(normalized_name, normalized_address))
            )
    connection.executemany("INSERT OR IGNORE INTO batch_keys VALUES (?, ?, ?, ?)", keys)


def _attach_strategy_indexes(connection: sqlite3.Connection, cache_paths: dict[str, Path]) -> list[str]:
    """Attach cached strategy databases to the candidate database."""
    aliases = []
    for strategy, path in cache_paths.items():
        alias = f"idx_{strategy}"
        connection.execute(f"ATTACH DATABASE ? AS {alias}", (str(path),))
        aliases.append(alias)
    return aliases


def _match_source1_batches(
    connection: sqlite3.Connection,
    split: str,
    blockers: list[tuple[str, Any]],
) -> int:
    """Match source-1 batches to eligible postings and return the source-1 row count."""
    strategy_sql = {}
    for strategy, _ in blockers:
        alias = f"idx_{strategy}"
        strategy_sql[strategy] = (
            "WITH matched AS ("
            " SELECT DISTINCT k.source1_id, p.entity_id FROM batch_keys k"
            f" JOIN {alias}.eligible_keys e ON e.country = k.country AND e.block_key = k.block_key"
            f" JOIN {alias}.postings p ON p.country = k.country AND p.block_key = k.block_key"
            " WHERE k.strategy = ?"
            "), ranked AS ("
            " SELECT source1_id, entity_id, ROW_NUMBER() OVER ("
            " PARTITION BY source1_id ORDER BY entity_id) AS rank FROM matched"
            ") INSERT OR IGNORE INTO candidates"
            " SELECT source1_id, entity_id FROM ranked WHERE rank <= ?"
        )
    row_count = 0
    for chunk in iter_source(split, "source1"):
        for offset in range(0, len(chunk), BLOCKING_MATCH_BATCH_SIZE):
            batch = chunk.iloc[offset:offset + BLOCKING_MATCH_BATCH_SIZE]
            _add_batch_keys(connection, batch, blockers)
            for strategy, _ in blockers:
                connection.execute(
                    strategy_sql[strategy],
                    (strategy, MAX_CANDIDATES_PER_ENTITY),
                )
            connection.execute("DELETE FROM batch_keys")
            connection.commit()
        row_count += len(chunk)
        print(f"source1: matched {row_count:,} rows", flush=True)
    return row_count


def _write_candidate_file(connection: sqlite3.Connection, split: str, path: Path) -> int:
    """Stream row-aligned candidates to TSV and return the unique pair count."""
    candidate_count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for chunk in iter_source(split, "source1"):
            for offset in range(0, len(chunk), BLOCKING_BATCH_SIZE):
                source1_ids = chunk["entity_id"].iloc[offset:offset + BLOCKING_BATCH_SIZE].astype(str).tolist()
                placeholders = ",".join("?" for _ in source1_ids)
                rows = connection.execute(
                    "SELECT source1_id, GROUP_CONCAT(entity_id, ',') FROM ("
                    "SELECT source1_id, entity_id FROM candidates "
                    f"WHERE source1_id IN ({placeholders}) ORDER BY source1_id, entity_id) "
                    "GROUP BY source1_id",
                    source1_ids,
                )
                candidate_map = dict(rows)
                writer.writerows((source1_id, candidate_map.get(source1_id) or "") for source1_id in source1_ids)
    return int(candidate_count)


def generate_candidates(
    split: str = "train",
    strategies: list[str] | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Generate the configured candidate union for one split using bounded memory."""
    if split != "train":
        raise ValueError("Phase 1 candidate scoring supports only the train split")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    candidate_path = TRAIN_CANDIDATE_PATH if output_path is None else output_path
    candidate_path.parent.mkdir(parents=True, exist_ok=True)
    blockers = _strategy_instances(strategies)
    target_count = 0
    source1_count = 0
    with tempfile.TemporaryDirectory(prefix="ber-blocking-", dir=OUTPUT_DIR) as temporary_dir:
        database_path = Path(temporary_dir) / "blocking.sqlite"
        with closing(sqlite3.connect(database_path)) as connection:
            _create_database(connection)
            for source_name in ("source2", "source3"):
                cache_paths, indexed_rows = _ensure_strategy_indexes(split, source_name, blockers)
                target_count += indexed_rows
                aliases = _attach_strategy_indexes(connection, cache_paths)
                print(f"Matching source1 against {source_name}...", flush=True)
                try:
                    matched_source1_count = _match_source1_batches(connection, split, blockers)
                finally:
                    for alias in aliases:
                        connection.execute(f"DETACH DATABASE {alias}")
                if source1_count and source1_count != matched_source1_count:
                    raise ValueError("source1 row count changed during candidate generation")
                source1_count = matched_source1_count
            candidate_count = _write_candidate_file(connection, split, candidate_path)

    possible_pairs = source1_count * target_count
    return {
        "source1_count": source1_count,
        "target_count": target_count,
        "candidate_count": candidate_count,
        "reduction_ratio": 1 - candidate_count / possible_pairs if possible_pairs else 0.0,
        "candidate_path": candidate_path,
    }


def merge_candidate_files(base_path: Path, addition_path: Path, output_path: Path) -> tuple[int, int]:
    """Merge row-aligned candidate files atomically and return pair and row counts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_count = 0
    source1_count = 0
    with tempfile.TemporaryDirectory(prefix="ber-merge-", dir=output_path.parent) as temporary_dir:
        temporary_path = Path(temporary_dir) / output_path.name
        with base_path.open("r", encoding="utf-8", newline="") as base_file:
            with addition_path.open("r", encoding="utf-8", newline="") as addition_file:
                base_reader = csv.reader(base_file, delimiter="\t")
                addition_reader = csv.reader(addition_file, delimiter="\t")
                base_header = next(base_reader, None)
                addition_header = next(addition_reader, None)
                if base_header != addition_header or base_header != ["source1_entity_id", "candidate_entity_ids"]:
                    raise ValueError("candidate files must have the expected matching headers")
                with temporary_path.open("w", encoding="utf-8", newline="") as output_file:
                    writer = csv.writer(output_file, delimiter="\t", lineterminator="\n")
                    writer.writerow(base_header)
                    for base_row, addition_row in zip_longest(base_reader, addition_reader):
                        if base_row is None or addition_row is None:
                            raise ValueError("candidate files have different source1 row counts")
                        if base_row[0] != addition_row[0]:
                            raise ValueError("candidate files are not aligned by source1_entity_id")
                        candidate_ids = set(filter(None, base_row[1].split(",")))
                        candidate_ids.update(filter(None, addition_row[1].split(",")))
                        writer.writerow([base_row[0], ",".join(sorted(candidate_ids))])
                        candidate_count += len(candidate_ids)
                        source1_count += 1
                        if source1_count % 100_000 == 0:
                            print(f"merged {source1_count:,} source1 rows", flush=True)
        temporary_path.replace(output_path)
    return candidate_count, source1_count


def _count_source_rows(split: str, source: str) -> int:
    """Count rows in one source using selected-ID chunks."""
    return sum(len(chunk) for chunk in iter_source(split, source, columns=["entity_id"]))


def main() -> None:
    """Run training blocking, print reduction statistics, and score blocking recall."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train"], default="train")
    parser.add_argument("--strategies", nargs="+", choices=tuple(BLOCKER_REGISTRY))
    parser.add_argument("--output", type=Path, default=TRAIN_CANDIDATE_PATH)
    parser.add_argument("--merge-candidates", nargs=2, type=Path, metavar=("BASE", "ADDITION"))
    args = parser.parse_args()
    if args.merge_candidates:
        candidate_count, source1_count = merge_candidate_files(
            args.merge_candidates[0],
            args.merge_candidates[1],
            TRAIN_CANDIDATE_PATH,
        )
        target_count = _count_source_rows("train", "source2") + _count_source_rows("train", "source3")
        possible_pairs = source1_count * target_count
        result = {
            "candidate_count": candidate_count,
            "reduction_ratio": 1 - candidate_count / possible_pairs if possible_pairs else 0.0,
            "candidate_path": TRAIN_CANDIDATE_PATH,
        }
    else:
        result = generate_candidates(args.split, args.strategies, args.output)
    print("candidate pairs:", result["candidate_count"])
    print("reduction ratio:", f"{result['reduction_ratio']:.6%}")
    score = score_candidate_file(result["candidate_path"])
    print(f"blocking recall: {score['recall']:.6%} ({score['matched_in_candidates']}/{score['total_true_matches']})")


if __name__ == "__main__":
    main()