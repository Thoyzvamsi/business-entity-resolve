"""Generates candidates from normalized names and leading name tokens.

This module owns name-key blocking; it must never score or classify candidate pairs.
"""

from collections.abc import Iterable

import pandas as pd

from src.blocking.base import BaseBlocker
from src.config import MIN_NAME_NGRAM_LENGTH, NAME_NGRAM_SIZE


class NameBlocker(BaseBlocker):
    """Block records on a normalized full name and its first informative token."""

    def blocking_keys(self, normalized_name: str, normalized_address: str = "") -> Iterable[str]:
        """Return exact-name and compact character n-gram keys."""
        if normalized_name:
            yield f"exact:{normalized_name}"
        compact_name = normalized_name.replace(" ", "")
        if len(compact_name) >= MIN_NAME_NGRAM_LENGTH:
            for offset in range(len(compact_name) - NAME_NGRAM_SIZE + 1):
                yield f"ngram:{compact_name[offset:offset + NAME_NGRAM_SIZE]}"

    def generate(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Return source1/other pairs sharing an eligible normalized-name key."""
        return self._generate_indexed(source1, other)