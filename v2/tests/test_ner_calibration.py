from __future__ import annotations

import json

import numpy as np

from clinical_nlp_lab.training import (
    compute_bio_span_metrics,
    write_ner_calibration,
)


def _logits_from_probabilities(rows: list[list[list[float]]]) -> np.ndarray:
    probabilities = np.asarray(rows, dtype=np.float64)
    return np.log(probabilities)


def test_entity_metric_rejects_wrong_boundary_despite_high_token_accuracy():
    # Gold is one two-token DISEASE span. Prediction truncates it to one token.
    labels = np.asarray([[-100, 1, 2, 0, 0, -100]])
    logits = _logits_from_probabilities(
        [[
            [0.99, 0.005, 0.005],
            [0.01, 0.98, 0.01],
            [0.90, 0.05, 0.05],
            [0.98, 0.01, 0.01],
            [0.98, 0.01, 0.01],
            [0.99, 0.005, 0.005],
        ]]
    )

    metrics = compute_bio_span_metrics((logits, labels))

    assert metrics["accuracy"] == 0.75
    assert metrics["entity_precision"] == 0.0
    assert metrics["entity_recall"] == 0.0
    assert metrics["entity_f1"] == 0.0


def test_calibration_removes_low_confidence_false_positive():
    # First row has one correct high-confidence entity. Second row is all O but
    # receives a lower-confidence spurious B prediction.
    labels = np.asarray([[1, 0], [0, 0]])
    logits = _logits_from_probabilities(
        [
            [[0.02, 0.97, 0.01], [0.98, 0.01, 0.01]],
            [[0.39, 0.60, 0.01], [0.98, 0.01, 0.01]],
        ]
    )

    metrics = compute_bio_span_metrics((logits, labels))

    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["entity_f1"] == 1.0
    assert metrics["ner_confidence_threshold"] >= 0.9


def test_entity_metric_ignores_predictions_at_masked_owner_window_positions():
    labels = np.asarray([[1, 0, -100]])
    logits = _logits_from_probabilities(
        [[
            [0.01, 0.98, 0.01],
            [0.98, 0.01, 0.01],
            [0.01, 0.98, 0.01],
        ]]
    )

    metrics = compute_bio_span_metrics((logits, labels))

    assert metrics["entity_precision"] == 1.0
    assert metrics["entity_recall"] == 1.0
    assert metrics["entity_f1"] == 1.0


def test_calibration_artifact_is_versioned_and_keeps_validation_metrics(tmp_path):
    destination = tmp_path / "ner_calibration.json"

    write_ner_calibration(
        destination,
        {
            "entity_precision": 0.8,
            "entity_recall": 0.6,
            "entity_f1": 0.685714,
            "ner_confidence_threshold": 0.9,
        },
    )

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload == {
        "schema_id": "clinical_nlp.ner_calibration",
        "schema_version": 1,
        "objective": "entity_exact_f1_precision_tiebreak",
        "confidence_threshold": 0.9,
        "validation": {
            "entity_precision": 0.8,
            "entity_recall": 0.6,
            "entity_f1": 0.685714,
        },
    }
