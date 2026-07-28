"""Shared structural and semantic safety rules for entity spans."""

from __future__ import annotations

import re


SUSPICIOUS_GENERIC_SURFACES = frozenset(
    {
        "bệnh",
        "không",
        "khoa",
        "lúc",
        "mang",
        "phẫu",
        "thể",
        "thuật",
        "tin",
        "và",
        "xét",
    }
)

_NO_ALPHANUMERIC = re.compile(r"^[\W_]*$", re.UNICODE)


def is_suspicious_generic_surface(text: str) -> bool:
    """Return whether a normalized surface is a known generic collapse token."""
    return text.strip().casefold() in SUSPICIOUS_GENERIC_SURFACES


def validate_entity_span(
    raw_text: str,
    start: int,
    end: int,
    text: str,
    *,
    max_length: int | None = None,
) -> tuple[str, ...]:
    """Return deterministic violation codes for one proposed entity span."""
    violations: list[str] = []

    if not (0 <= start < end <= len(raw_text)):
        violations.append("invalid_range")
        return tuple(violations)

    if raw_text[start:end] != text:
        violations.append("offset_mismatch")

    if not text.strip():
        violations.append("whitespace_only")
    elif _NO_ALPHANUMERIC.fullmatch(text):
        violations.append("punctuation_only")

    if "\n" in text or "\r" in text:
        violations.append("multiline")

    if (
        start > 0
        and raw_text[start - 1].isalnum()
        and raw_text[start].isalnum()
    ):
        violations.append("left_word_split")

    if (
        end < len(raw_text)
        and raw_text[end - 1].isalnum()
        and raw_text[end].isalnum()
    ):
        violations.append("right_word_split")

    if max_length is not None and end - start > max_length:
        violations.append("too_long")

    return tuple(violations)
