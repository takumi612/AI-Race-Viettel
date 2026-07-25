from __future__ import annotations

import pytest

from clinical_nlp_lab.output_quality import (
    OutputQualityError,
    audit_output_documents,
    enforce_output_quality,
)
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation


def _entity(text: str, entity_type: str, start: int, end: int) -> EntityAnnotation:
    return EntityAnnotation(text, entity_type, (start, end))


def test_quality_gate_rejects_pathological_coverage_and_type_collapse():
    text = "x" * 100
    documents = [
        ClinicalDocument(
            str(index),
            text,
            [_entity(text[:83], "LAB_NAME", 0, 83)],
        )
        for index in range(100)
    ]

    report = audit_output_documents(documents)

    assert report["mean_coverage"] == 0.83
    assert report["max_type_share"] == 1.0
    with pytest.raises(OutputQualityError, match="coverage|type_share"):
        enforce_output_quality(report)


def test_quality_gate_accepts_sparse_well_bounded_output():
    text = "patient has fever and takes aspirin"
    documents = [
        ClinicalDocument(
            "1",
            text,
            [
                _entity("fever", "SYMPTOM", 12, 17),
                _entity("aspirin", "DRUG", 28, 35),
            ],
        )
    ]

    report = audit_output_documents(documents)

    enforce_output_quality(report)
    assert report["boundary_error_count"] == 0
