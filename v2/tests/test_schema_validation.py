from __future__ import annotations

from clinical_nlp_lab.schema import OFFICIAL_SCHEMA_KEYS, validate_submission_payload


def _symptom_type() -> str:
    return next(key for key in OFFICIAL_SCHEMA_KEYS if "TRI" in key)


def _payload(text: str, position: tuple[int, int]) -> list[dict[str, object]]:
    return [
        {
            "text": text,
            "type": _symptom_type(),
            "position": list(position),
            "assertions": [],
        }
    ]


def test_submission_validation_rejects_punctuation_only_entity():
    errors = validate_submission_payload(_payload("-", (5, 6)), "pain - fever")

    assert any("punctuation_only" in error for error in errors)


def test_submission_validation_rejects_multiline_entity():
    raw_text = "fever\ncough"
    errors = validate_submission_payload(_payload(raw_text, (0, len(raw_text))), raw_text)

    assert any("multiline" in error for error in errors)
