from __future__ import annotations

from clinical_nlp_lab.dataset_quality import audit_training_contract
from clinical_nlp_lab.entity_types import ENTITY_TYPE_TO_ID
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation


def test_training_contract_accepts_all_canonical_types_and_assertion_scope():
    raw_text = "disease drug symptom labname labresult"
    values = [
        ("disease", "DISEASE", ["isHistorical"]),
        ("drug", "DRUG", ["isNegated"]),
        ("symptom", "SYMPTOM", ["isFamily"]),
        ("labname", "LAB_NAME", []),
        ("labresult", "LAB_RESULT", []),
    ]
    entities = []
    for text, entity_type, assertions in values:
        start = raw_text.index(text)
        entities.append(
            EntityAnnotation(
                text,
                entity_type,
                (start, start + len(text)),
                assertions=assertions,
            )
        )

    report = audit_training_contract([ClinicalDocument("1", raw_text, entities)])

    assert report["is_valid"] is True
    assert set(report["type_counts"]) == set(ENTITY_TYPE_TO_ID)
