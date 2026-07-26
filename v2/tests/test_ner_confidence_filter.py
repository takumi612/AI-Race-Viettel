from __future__ import annotations

import json

import pytest

from clinical_nlp_lab.ner import (
    filter_entities_by_confidence,
    load_ner_confidence_threshold,
)
from clinical_nlp_lab.schema import EntityAnnotation


def _entity(text: str, start: int, confidence: float) -> EntityAnnotation:
    return EntityAnnotation(
        text=text,
        type="DISEASE",
        position=(start, start + len(text)),
        confidence=confidence,
    )


def test_detector_filter_keeps_only_entities_at_or_above_calibrated_threshold():
    entities = [
        _entity("bệnh", 0, 0.899),
        _entity("sốt", 10, 0.90),
        _entity("viêm phổi", 20, 0.97),
    ]

    filtered = filter_entities_by_confidence(entities, 0.90)

    assert [(item.text, item.start, item.end) for item in filtered] == [
        ("sốt", 10, 13),
        ("viêm phổi", 20, 29),
    ]


def test_detector_loads_versioned_calibration_artifact(tmp_path):
    (tmp_path / "ner_calibration.json").write_text(
        json.dumps(
            {
                "schema_id": "clinical_nlp.ner_calibration",
                "schema_version": 1,
                "confidence_threshold": 0.93,
            }
        ),
        encoding="utf-8",
    )

    assert load_ner_confidence_threshold(tmp_path) == 0.93


def test_legacy_checkpoint_uses_conservative_threshold(tmp_path):
    assert load_ner_confidence_threshold(tmp_path) == 0.85


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_id": "wrong", "schema_version": 1, "confidence_threshold": 0.9},
        {
            "schema_id": "clinical_nlp.ner_calibration",
            "schema_version": 2,
            "confidence_threshold": 0.9,
        },
        {
            "schema_id": "clinical_nlp.ner_calibration",
            "schema_version": 1,
            "confidence_threshold": 1.5,
        },
    ],
)
def test_malformed_calibration_fails_closed(tmp_path, payload):
    (tmp_path / "ner_calibration.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="NER calibration"):
        load_ner_confidence_threshold(tmp_path)
