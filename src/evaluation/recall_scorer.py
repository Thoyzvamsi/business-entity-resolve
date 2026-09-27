"""Measures candidate-generation recall against training ground truth.

This module owns blocking recall metrics; it must not alter candidates or labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import TRAIN_CANDIDATE_PATH, TRAIN_DIR
from src.data_loader import iter_tsv


def score_candidate_file(
    candidate_path: Path = TRAIN_CANDIDATE_PATH,
    ground_truth_path: Path | None = None,
) -> dict[str, Any]:
    """Compute the fraction of true links present in candidates keyed by source-1 ID."""
    truth_path = ground_truth_path or TRAIN_DIR / "train_ground_truth.tsv"
    total_true_matches = 0
    matched_in_candidates = 0
    total_by_source = {"S2": 0, "S3": 0}
    matched_by_source = {"S2": 0, "S3": 0}
    missed_examples: list[tuple[str, str]] = []
    truth_by_source1: dict[str, set[str]] = {}
    for chunk in iter_tsv(truth_path):
        for source1_id, matched_ids in chunk.itertuples(index=False, name=None):
            if source1_id in truth_by_source1:
                raise ValueError(f"duplicate source1_entity_id in ground truth: {source1_id}")
            truth_set = set(matched_ids.split(",")) if pd.notna(matched_ids) and matched_ids else set()
            truth_by_source1[str(source1_id)] = truth_set
            total_true_matches += len(truth_set)
            for target_id in truth_set:
                target_source = target_id[:2]
                if target_source in total_by_source:
                    total_by_source[target_source] += 1

    candidate_rows = 0
    for chunk in iter_tsv(candidate_path):
        for source1_id, candidate_ids in chunk.itertuples(index=False, name=None):
            source1_id = str(source1_id)
            truth_set = truth_by_source1.pop(source1_id, None)
            if truth_set is None:
                raise ValueError(f"unexpected or duplicate source1_entity_id in candidates: {source1_id}")
            candidate_set = set(candidate_ids.split(",")) if pd.notna(candidate_ids) and candidate_ids else set()
            matched_set = candidate_set & truth_set
            matched_in_candidates += len(matched_set)
            for target_id in matched_set:
                target_source = target_id[:2]
                if target_source in matched_by_source:
                    matched_by_source[target_source] += 1
            if len(missed_examples) < 20:
                missed_examples.extend(
                    (source1_id, target_id)
                    for target_id in sorted(truth_set - candidate_set)
                    if len(missed_examples) < 20
                )
            candidate_rows += 1

    if truth_by_source1:
        raise ValueError(f"candidate file is missing {len(truth_by_source1)} source1 rows")

    recall = matched_in_candidates / total_true_matches if total_true_matches else 0.0
    return {
        "recall": recall,
        "matched_in_candidates": matched_in_candidates,
        "total_true_matches": total_true_matches,
        "candidate_rows": candidate_rows,
        "total_by_source": total_by_source,
        "matched_by_source": matched_by_source,
        "missed_examples": missed_examples,
    }


def main() -> None:
    """Print recall for the configured training candidate file."""
    result = score_candidate_file()
    print(f"blocking recall: {result['recall']:.6%}")
    print(f"matched true links: {result['matched_in_candidates']}/{result['total_true_matches']}")
    for source in ("S2", "S3"):
        total = result["total_by_source"][source]
        matched = result["matched_by_source"][source]
        recall = matched / total if total else 0.0
        print(f"{source} recall: {recall:.6%} ({matched}/{total})")
    print("sample missed links:", result["missed_examples"][:10])


if __name__ == "__main__":
    main()