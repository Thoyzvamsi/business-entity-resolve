"""Performs basic exploratory data analysis for the challenge data.

This module owns descriptive statistics only; it never creates matches or model artifacts.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from src.config import EDA_CHUNK_SIZE, TRAIN_DIR
from src.data_loader import iter_split, iter_tsv


def _source_summary() -> Dict[str, Dict[str, Any]]:
    """Stream source files and accumulate schema, null, country, and name statistics."""
    summary: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "test"):
        split_sources: Dict[str, Dict[str, Any]] = {}
        for source, chunk in iter_split(split):
            stats = split_sources.setdefault(source, {
                "rows": 0,
                "columns": list(chunk.columns),
                "null_counts": Counter(),
                "country_counts": Counter(),
                "name_length_sum": 0,
                "name_length_min": None,
                "name_length_max": 0,
                "name_count": 0,
            })
            stats["rows"] += len(chunk)
            stats["null_counts"].update(chunk.isna().sum().to_dict())
            stats["country_counts"].update(chunk["country"].dropna().astype(str).value_counts().to_dict())
            name_lengths = chunk["business_name"].dropna().str.len()
            if not name_lengths.empty:
                stats["name_length_sum"] += int(name_lengths.sum())
                stats["name_length_min"] = min(stats["name_length_min"] or int(name_lengths.min()), int(name_lengths.min()))
                stats["name_length_max"] = max(stats["name_length_max"], int(name_lengths.max()))
                stats["name_count"] += len(name_lengths)
        summary[split] = split_sources
    return summary


def _ground_truth_singletons() -> tuple[int, int]:
    """Count ground-truth rows and singleton entities using bounded memory."""
    total_rows = 0
    singleton_count = 0
    path = TRAIN_DIR / "train_ground_truth.tsv"
    for chunk in iter_tsv(path, EDA_CHUNK_SIZE):
        matched_ids = chunk["matched_entity_ids"].fillna("").str.strip()
        singleton_count += int(matched_ids.eq("").sum())
        total_rows += len(chunk)
    return total_rows, singleton_count


def run_eda() -> Dict[str, Any]:
    """Return summary statistics for train/test data and singleton rates."""
    sources = _source_summary()
    total_rows, singleton_count = _ground_truth_singletons()
    train_country_counts = Counter()
    test_country_counts = Counter()
    for source_stats in sources["train"].values():
        train_country_counts.update(source_stats["country_counts"])
    for source_stats in sources["test"].values():
        test_country_counts.update(source_stats["country_counts"])

    summary = {
        "train_rows": sum(stats["rows"] for stats in sources["train"].values()),
        "test_rows": sum(stats["rows"] for stats in sources["test"].values()),
        "singleton_rate": singleton_count / total_rows if total_rows else 0.0,
        "france_in_train": "France" in train_country_counts,
        "france_in_test": "France" in test_country_counts,
        "train_country_counts": dict(train_country_counts),
        "test_country_counts": dict(test_country_counts),
        "train_sources": sources["train"],
        "test_sources": sources["test"],
        "ground_truth_size": total_rows,
        "singleton_count": singleton_count,
    }
    return summary


def main() -> None:
    """Print a concise EDA summary for the challenge dataset."""
    summary = run_eda()
    print("train rows:", summary["train_rows"])
    print("test rows:", summary["test_rows"])
    print("singletons:", summary["singleton_count"], "of", summary["ground_truth_size"])
    print("singleton rate:", f"{summary['singleton_rate']:.6%}")
    print("train country counts:", summary["train_country_counts"])
    print("test country counts:", summary["test_country_counts"])
    print("France in train?", summary["france_in_train"])
    print("France in test?", summary["france_in_test"])
    for split in ("train", "test"):
        for source, stats in summary[f"{split}_sources"].items():
            mean_name_length = stats["name_length_sum"] / stats["name_count"] if stats["name_count"] else 0
            print(f"{split} {source}: rows={stats['rows']}, columns={stats['columns']}")
            print(f"{split} {source} null counts:", dict(stats["null_counts"]))
            print(f"{split} {source} name length min/mean/max:", stats["name_length_min"], round(mean_name_length, 2), stats["name_length_max"])


if __name__ == "__main__":
    main()
