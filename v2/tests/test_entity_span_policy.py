from __future__ import annotations

import pytest

from clinical_nlp_lab.entity_span_policy import (
    is_suspicious_generic_surface,
    validate_entity_span,
)


@pytest.mark.parametrize(
    ("text",),
    [
        ("COVID-19",),
        ("38.5°C",),
        ("SpO2 95%",),
        ("HBsAg (+)",),
        ("T3/T4",),
        ("sốt",),
    ],
)
def test_medical_special_characters_are_allowed(text: str):
    assert validate_entity_span(text, 0, len(text), text) == ()


@pytest.mark.parametrize("text", ["-", "...", "/()", "_", " \t "])
def test_surface_without_alphanumeric_content_is_rejected(text: str):
    violations = validate_entity_span(text, 0, len(text), text)

    assert "punctuation_only" in violations or "whitespace_only" in violations


def test_policy_reports_offset_boundary_and_length_violations():
    raw_text = "abcdef"

    assert "invalid_range" in validate_entity_span(raw_text, 2, 2, "")
    assert "offset_mismatch" in validate_entity_span(raw_text, 0, 2, "zz")
    assert "left_word_split" in validate_entity_span(raw_text, 1, 3, "bc")
    assert "right_word_split" in validate_entity_span(raw_text, 0, 2, "ab")
    assert "too_long" in validate_entity_span(raw_text, 0, 6, raw_text, max_length=5)


def test_policy_reports_multiline_text():
    raw_text = "fever\ncough"

    assert "multiline" in validate_entity_span(raw_text, 0, len(raw_text), raw_text)


def test_generic_surface_helper_normalizes_case_and_whitespace():
    assert is_suspicious_generic_surface(" VÀ ")
    assert not is_suspicious_generic_surface("sốt")
