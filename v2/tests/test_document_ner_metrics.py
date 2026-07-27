from __future__ import annotations

from clinical_nlp_lab.natural_validation import (
    calibrate_document_entity_threshold,
    compare_document_merge_strategies,
)
from clinical_nlp_lab.schema import EntityAnnotation


def entity(start: int, end: int, entity_type: str, confidence: float = 1.0):
    return EntityAnnotation(
        text="x" * (end - start),
        type=entity_type,
        position=(start, end),
        confidence=confidence,
    )


def test_document_threshold_uses_exact_span_f1_with_precision_tiebreak():
    gold = {"181": [entity(10, 20, "DISEASE")]}
    predicted = {
        "181": [
            entity(10, 20, "DISEASE", confidence=0.95),
            entity(30, 45, "DISEASE", confidence=0.70),
        ]
    }

    report = calibrate_document_entity_threshold(gold, predicted, {"181": "organizer"})

    assert report["confidence_threshold"] == 0.95
    assert report["exact"]["precision"] == 1.0
    assert report["exact"]["recall"] == 1.0
    assert report["length_buckets"]["1-50"]["gold"] == 1


def test_consensus_merge_beats_legacy_union_on_same_chunk_predictions():
    raw = "x" * 80
    gold = {"181": [entity(10, 40, "DISEASE")]}
    chunks = {
        "181": [
            entity(10, 40, "DISEASE", confidence=0.97),
            entity(30, 60, "DISEASE", confidence=0.96),
        ]
    }

    report = compare_document_merge_strategies(
        gold, chunks, {"181": "organizer"}, {"181": raw}
    )

    assert report["legacy_union"]["exact"]["precision"] == 0.0
    assert report["boundary_consensus"]["exact"]["precision"] == 1.0


def test_document_metrics_include_type_overlap_length_and_deterministic_payload():
    gold = {
        "181": [
            entity(0, 60, "DISEASE"),
            entity(100, 205, "DRUG"),
        ]
    }
    predicted = {
        "181": [
            entity(1, 60, "DISEASE", confidence=0.99),
            entity(100, 205, "DRUG", confidence=0.99),
        ]
    }

    first = calibrate_document_entity_threshold(gold, predicted, {"181": "organizer"})
    second = calibrate_document_entity_threshold(gold, predicted, {"181": "organizer"})

    assert first == second
    assert first["by_type"]["DISEASE"]["overlap"]["f1"] == 1.0
    assert first["length_buckets"]["51-100"]["gold"] == 1
    assert first["length_buckets"]["101+"]["gold"] == 1
