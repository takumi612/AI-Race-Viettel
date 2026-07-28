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


def test_submission_schema_does_not_apply_output_quality_policy():
    raw_text = "fever\ncough"
    multiline = validate_submission_payload(
        _payload(raw_text, (0, len(raw_text))), raw_text
    )
    long_text = "x" * 162
    over_output_limit = validate_submission_payload(
        _payload(long_text, (0, len(long_text))), long_text
    )

    assert multiline == []
    assert over_output_limit == []


def test_submission_schema_still_rejects_offset_mismatch():
    errors = validate_submission_payload(_payload("fever", (0, 5)), "cough")

    assert any("Offset mismatch" in error for error in errors)
