from __future__ import annotations

import pytest
from clinical_nlp_lab.schema import EntityAnnotation
from clinical_nlp_lab.candidate_training import (
    ScoredCandidate,
    build_candidate_features,
    fit_candidate_calibration,
    generate_hard_negatives,
)


def test_disease_icd10_drug_rxnorm_type_matching():
    disease_mention = EntityAnnotation(text="Cúm A", type="DISEASE", position=(0, 5))
    drug_mention = EntityAnnotation(text="Paracetamol", type="DRUG", position=(0, 11))

    icd_cand = {"code": "J10", "system": "ICD10", "aliases": ["Cúm A"]}
    rx_cand = {"code": "161", "system": "RXNORM", "aliases": ["Paracetamol"]}

    f1 = build_candidate_features(disease_mention, icd_cand)
    f2 = build_candidate_features(disease_mention, rx_cand)
    f3 = build_candidate_features(drug_mention, rx_cand)

    assert f1.type_match is True
    assert f2.type_match is False
    assert f3.type_match is True


def test_generate_hard_negatives_limit_and_system():
    positive = {"code": "J10", "system": "ICD10"}
    retrieved = [
        {"code": "J10", "system": "ICD10"},      # positive -> skip
        {"code": "J11", "system": "ICD10"},      # hard negative
        {"code": "161", "system": "RXNORM"},     # diff system -> skip
        {"code": "J12", "system": "ICD10"},      # hard negative
    ]

    negs = generate_hard_negatives(positive, retrieved, limit=1)
    assert len(negs) == 1
    assert negs[0].code == "J11"
    assert negs[0].system == "ICD10"


def test_insufficient_ranking_labels_fallback():
    scored = [
        ScoredCandidate("Cúm", "DISEASE", "J10", "ICD10", 0.8, True),
    ]
    artifact = fit_candidate_calibration(scored, objective="precision_first", kb_hash="hash1")
    assert artifact.policy == "deterministic_fallback"
    assert artifact.confidence_threshold >= 0.65
