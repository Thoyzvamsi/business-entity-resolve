"""Generates candidates from informative name-token overlap.

This module owns token-based blocking; it must never score or classify candidate pairs.
"""

from collections.abc import Iterable

import pandas as pd

from src.blocking.base import BaseBlocker
from src.config import MIN_TOKEN_LENGTH

_COMMON_TOKENS = {
    "and", "the", "for", "with", "from", "service", "services", "group",
    "company", "corporation", "international", "global", "store", "shop",
}


class TokenSetBlocker(BaseBlocker):
    """Block records on shared informative tokens, independent of word order."""

    def blocking_keys(self, normalized_name: str, normalized_address: str = "") -> Iterable[str]:
        """Return distinct non-generic name tokens that meet the configured length."""
        for word in set(normalized_name.split()):
            if len(word) >= MIN_TOKEN_LENGTH and word not in _COMMON_TOKENS:
                yield f"token:{word}"

    def generate(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Return source1/other pairs sharing an informative name token."""
        return self._generate_indexed(source1, other)