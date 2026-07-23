from __future__ import annotations

from clinical_nlp_lab.candidate_training import (
    ScoredCandidate,
    build_candidate_training_artifact,
)
from clinical_nlp_lab.candidate_policy import CandidatePolicy
from clinical_nlp_lab.pipeline import load_candidate_policy


def test_candidate_training_artifact_binds_hard_negatives_and_calibration():
    artifact = build_candidate_training_artifact(
        positives=(
            {
                "code": "E11",
                "system": "ICD10",
                "aliases": ["diabetes"],
            },
        ),
        retrieved=(
            (
                {"code": "E11", "system": "ICD10", "aliases": ["diabetes"]},
                {"code": "E13", "system": "ICD10", "aliases": ["other diabetes"]},
                {"code": "RX1", "system": "RXNORM", "aliases": ["metformin"]},
            ),
        ),
        scored_examples=(
            ScoredCandidate("diabetes", "DISEASE", "E11", "ICD10", 0.9, True),
            ScoredCandidate("diabetes", "DISEASE", "E13", "ICD10", 0.2, False),
        ),
        kb_hash="k" * 64,
    )

    assert artifact.hard_negative_count == 1
    assert artifact.hard_negatives[0].code == "E13"
    assert artifact.calibration.policy == "deterministic_fallback"
    assert artifact.kb_hash == "k" * 64
    assert len(artifact.artifact_sha256) == 64


def test_candidate_policy_can_bind_calibration_artifact():
    artifact = build_candidate_training_artifact(
        positives=({"code": "E11", "system": "ICD10", "aliases": ["diabetes"]},),
        retrieved=(
            (
                {"code": "E11", "system": "ICD10", "aliases": ["diabetes"]},
                {"code": "E13", "system": "ICD10", "aliases": ["other diabetes"]},
            ),
        ),
        scored_examples=(
            ScoredCandidate("diabetes", "DISEASE", "E11", "ICD10", 0.9, True),
            ScoredCandidate("diabetes", "DISEASE", "E13", "ICD10", 0.2, False),
        ),
        kb_hash="k" * 64,
    )

    policy = CandidatePolicy.from_calibration(artifact.calibration)
    assert policy.min_score == artifact.calibration.confidence_threshold
    assert policy.output_k == 1


def test_candidate_policy_can_load_calibration_mapping():
    policy = CandidatePolicy.from_calibration(
        {
            "objective": "precision_first",
            "confidence_threshold": 0.72,
            "kb_hash": "k" * 64,
        }
    )
    assert policy.min_score == 0.72
    policy.validate_kb_hash("k" * 64)


def test_pipeline_loads_persisted_candidate_calibration(tmp_path):
    (tmp_path / "candidate_calibration.json").write_text(
        '{"objective":"precision_first","confidence_threshold":0.73,"kb_hash":"' + "k" * 64 + '"}',
        encoding="utf-8",
    )
    policy = load_candidate_policy(tmp_path, default_min_score=0.5, default_min_margin=0.05)
    assert policy.min_score == 0.73
    assert policy.calibration_kb_hash == "k" * 64
