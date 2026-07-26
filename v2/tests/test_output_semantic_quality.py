from __future__ import annotations

import pytest

from clinical_nlp_lab.output_quality import (
    OutputQualityError,
    audit_output_documents,
    enforce_output_quality,
)
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation


def _entity(raw_text: str, text: str, entity_type: str, *, candidates=()):
    start = raw_text.index(text)
    return EntityAnnotation(
        text=text,
        type=entity_type,
        position=(start, start + len(text)),
        candidates=list(candidates),
    )


def test_audit_counts_candidate_coverage_only_for_disease_and_drug():
    raw_text = "tăng huyết áp metformin sốt"
    document = ClinicalDocument(
        "1",
        raw_text,
        [
            _entity(raw_text, "tăng huyết áp", "DISEASE", candidates=("I10",)),
            _entity(raw_text, "metformin", "DRUG"),
            _entity(raw_text, "sốt", "SYMPTOM"),
        ],
    )

    report = audit_output_documents([document])

    assert report["candidate_eligible_count"] == 2
    assert report["candidate_linked_entity_count"] == 1
    assert report["candidate_link_rate"] == 0.5


def test_repeated_generic_single_token_entities_are_rejected_but_fever_is_valid():
    raw_text = "và không lúc sốt"
    collapsed = ClinicalDocument(
        "1",
        raw_text,
        [
            _entity(raw_text, "và", "DISEASE"),
            _entity(raw_text, "không", "DISEASE"),
            _entity(raw_text, "lúc", "DISEASE"),
            _entity(raw_text, "sốt", "SYMPTOM"),
        ],
    )

    report = audit_output_documents([collapsed])

    assert report["single_token_entity_count"] == 4
    assert report["suspicious_generic_span_count"] == 3
    with pytest.raises(OutputQualityError, match="generic_spans=3"):
        enforce_output_quality(report)

    valid = ClinicalDocument(
        "2",
        "Bệnh nhân sốt.",
        [_entity("Bệnh nhân sốt.", "sốt", "SYMPTOM")],
    )
    enforce_output_quality(audit_output_documents([valid]))
