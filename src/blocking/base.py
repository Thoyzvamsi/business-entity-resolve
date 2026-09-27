"""Defines the candidate-generation contract and shared blocking utilities.

This module owns pair generation primitives; it must never classify candidates.
"""

from __future__ import annotations

import re
import unicodedata
from abc import ABC, abstractmethod
from typing import Iterable

import pandas as pd

from src.config import MAX_BLOCK_POSTINGS, MAX_CANDIDATES_PER_ENTITY

_NON_ALPHANUMERIC = re.compile(r"[^\w]+", re.UNICODE)
_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd",
    "limited", "llc", "llp", "plc", "pvt", "private",
}


def normalize_business_name(value: str) -> str:
    """Normalize punctuation and trailing legal suffixes in a business name."""
    folded = unicodedata.normalize("NFKD", value.casefold().replace("&", " and "))
    accent_free = "".join(character for character in folded if not unicodedata.combining(character))
    normalized = _NON_ALPHANUMERIC.sub(" ", accent_free)
    words = normalized.split()
    while words and words[-1] in _LEGAL_SUFFIXES:
        words.pop()
    return " ".join(words)


class BaseBlocker(ABC):
    """Owns candidate pair generation for one strategy, never pair classification."""

    @abstractmethod
    def blocking_keys(self, normalized_name: str, normalized_address: str = "") -> Iterable[str]:
        """Return the blocking keys for a business record."""

    @abstractmethod
    def generate(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Return candidate pairs as a set of source1 and other entity IDs."""

    def _generate_indexed(self, source1: pd.DataFrame, other: pd.DataFrame) -> set[tuple[str, str]]:
        """Generate pairs through a bounded posting index over the other source."""
        postings: dict[tuple[str, str], list[str]] = {}
        oversized: set[tuple[str, str]] = set()
        other_columns = ["entity_id", "business_name", "country"]
        if "business_address" in other.columns:
            other_columns.append("business_address")
        other_rows = other[other_columns].itertuples(index=False, name=None)
        for row in other_rows:
            entity_id, name, country, *address_values = row
            if pd.isna(name) or pd.isna(country):
                continue
            address = address_values[0] if address_values else ""
            country_key = str(country).casefold().strip()
            normalized_address = normalize_business_name(str(address)) if pd.notna(address) else ""
            for block_key in set(self.blocking_keys(normalize_business_name(str(name)), normalized_address)):
                key = (country_key, block_key)
                if key in oversized:
                    continue
                bucket = postings.setdefault(key, [])
                bucket.append(str(entity_id))
                if len(bucket) > MAX_BLOCK_POSTINGS:
                    del postings[key]
                    oversized.add(key)

        pairs: set[tuple[str, str]] = set()
        per_source1: dict[str, int] = {}
        source_columns = ["entity_id", "business_name", "country"]
        if "business_address" in source1.columns:
            source_columns.append("business_address")
        source_rows = source1[source_columns].itertuples(index=False, name=None)
        for row in source_rows:
            source1_id, name, country, *address_values = row
            if pd.isna(name) or pd.isna(country):
                continue
            address = address_values[0] if address_values else ""
            country_key = str(country).casefold().strip()
            source1_id = str(source1_id)
            normalized_address = normalize_business_name(str(address)) if pd.notna(address) else ""
            for block_key in set(self.blocking_keys(normalize_business_name(str(name)), normalized_address)):
                for other_id in postings.get((country_key, block_key), ()):
                    if per_source1.get(source1_id, 0) >= MAX_CANDIDATES_PER_ENTITY:
                        break
                    pairs.add((source1_id, other_id))
                    per_source1[source1_id] = per_source1.get(source1_id, 0) + 1
                if per_source1.get(source1_id, 0) >= MAX_CANDIDATES_PER_ENTITY:
                    break
        return pairs