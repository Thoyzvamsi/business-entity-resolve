"""Generates candidates from Soundex keys for normalized name tokens.

This module owns phonetic blocking; it must never score or classify candidate pairs.
"""

from collections.abc import Iterable

import pandas as pd

from src.blocking.base import BaseBlocker

_SOUNDEX_CODES = {
    **dict.fromkeys("bfpv", "1"),
    **dict.fromkeys("cgjkqsxz", "2"),
    **dict.fromkeys("dt", "3"),
    **dict.fromkeys("l", "4"),
    **dict.fromkeys("mn", "5"),
    **dict.fromkeys("r", "6"),
}


def _soundex(word: str) -> str:
    """Return a four-character Soundex code for one token."""
    if not word:
        return ""
    first = word[0]
    code = _SOUNDEX_CODES.get(first, "")
    digits = [code] if code else []
    previous = code
    for character in word[1:]:
        current = _SOUNDEX_CODES.get(character, "")
        if current and current != previous:
            digits.append(current)
        previous = current
    return (first.upper() + "".join(digits) + "000")[:4]


class PhoneticBlocker(BaseBlocker):
    """Block records when at least one name token has the same Soundex code."""

    def blocking_keys(self, normalized_name: str, normalized_address: str = "") -> Iterable[str]:
        """Return distinct Soundex keys for name tokens of at least three characters."""
        for word in set(normalized_name.split()):
            if len(word) >= 3:
                yield f"soundex:{_soundex(word)}"

    def generate(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Return source1/other pairs sharing an eligible phonetic key."""
        return self._generate_indexed(source1, other)