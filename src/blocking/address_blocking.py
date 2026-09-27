"""Generates candidates from distinctive normalized address tokens.

This module owns address-key blocking; it must never score or classify pairs.
"""

from collections.abc import Iterable

import pandas as pd

from src.blocking.base import BaseBlocker

_ADDRESS_STOP_WORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "drive", "dr", "boulevard",
    "blvd", "lane", "ln", "highway", "hwy", "suite", "ste", "unit", "apartment",
    "apt", "floor", "building", "bldg", "north", "south", "east", "west",
    "county", "city", "usa", "united", "states",
}


class AddressBlocker(BaseBlocker):
    """Block records on normalized house numbers and distinctive address words."""

    def blocking_keys(self, normalized_name: str, normalized_address: str = "") -> Iterable[str]:
        """Return normalized numeric and distinctive address-token keys."""
        for token in set(normalized_address.split()):
            if token.isdigit():
                canonical_number = token.lstrip("0") or "0"
                if len(canonical_number) >= 2:
                    yield f"address-number:{canonical_number}"
            elif len(token) >= 5 and token not in _ADDRESS_STOP_WORDS:
                yield f"address-token:{token}"

    def generate(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Return pairs that share an eligible country-scoped address key."""
        return self._generate_indexed(source1, other)
