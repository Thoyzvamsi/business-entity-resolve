"""Loads the challenge TSV data and ground truth for train/test splits.

This module owns file access only; it must not contain model logic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator

import pandas as pd

from src.config import EDA_CHUNK_SIZE, TEST_DIR, TRAIN_DIR


def _read_tsv(path: Path) -> pd.DataFrame:
    """Read one tab-separated file using Python-backed string columns."""
    return pd.read_csv(path, sep="\t", dtype="string[python]")


def iter_tsv(path: Path, chunksize: int = EDA_CHUNK_SIZE) -> Iterator[pd.DataFrame]:
    """Yield chunks from a TSV without loading the entire file into memory."""
    yield from pd.read_csv(path, sep="\t", dtype="string[python]", chunksize=chunksize)


def iter_source(
    split: str,
    source: str,
    columns: list[str] | None = None,
    chunksize: int = EDA_CHUNK_SIZE,
) -> Iterator[pd.DataFrame]:
    """Yield selected columns from one source file in bounded-size chunks."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if source not in {"source1", "source2", "source3"}:
        raise ValueError("source must be 'source1', 'source2', or 'source3'")
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    path = split_dir / f"{split}_{source}.tsv"
    selected = ["entity_id", "business_name", "country", "business_address"] if columns is None else columns
    yield from pd.read_csv(
        path,
        sep="\t",
        usecols=selected,
        dtype="string[python]",
        chunksize=chunksize,
    )


def load_split(split: str) -> Dict[str, pd.DataFrame]:
    """Load the source1/source2/source3 data for the requested split."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    return {source: load_source(split, source) for source in ("source1", "source2", "source3")}


def load_source(split: str, source: str, columns: list[str] | None = None) -> pd.DataFrame:
    """Return one source file, optionally restricted to the requested columns."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if source not in {"source1", "source2", "source3"}:
        raise ValueError("source must be 'source1', 'source2', or 'source3'")
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    path = split_dir / f"{split}_{source}.tsv"
    selected = ["entity_id", "business_name", "business_address", "country"] if columns is None else columns
    return pd.read_csv(path, sep="\t", usecols=selected, dtype="string[python]")


def load_ground_truth() -> pd.DataFrame:
    """Load the training ground-truth mapping from S1 entity IDs to matched S2/S3 IDs."""
    path = TRAIN_DIR / "train_ground_truth.tsv"
    return pd.read_csv(path, sep="\t", dtype="string[python]")


def iter_split(split: str) -> Iterator[tuple[str, pd.DataFrame]]:
    """Yield each source's train or test records in bounded-size chunks."""
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    split_dir = TRAIN_DIR if split == "train" else TEST_DIR
    for source in ("source1", "source2", "source3"):
        path = split_dir / f"{split}_{source}.tsv"
        yield from ((source, chunk) for chunk in iter_tsv(path))


def load_all_data() -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load train and test data for the entire challenge."""
    return {"train": load_split("train"), "test": load_split("test")}
