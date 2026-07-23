from __future__ import annotations

import pytest
from clinical_nlp_lab.curriculum import (
    CurriculumError,
    plan_curriculum,
    validate_stage_transition,
)


def test_plan_curriculum_modes():
    full_stages = plan_curriculum("full")
    assert len(full_stages) == 4
    assert [s.name for s in full_stages] == ["stage1", "stage2", "stage3", "final_fit"]

    resume_stages = plan_curriculum("resume")
    assert len(resume_stages) == 3
    assert [s.name for s in resume_stages] == ["stage2", "stage3", "final_fit"]

    inf_stages = plan_curriculum("inference_only")
    assert len(inf_stages) == 0


def test_validate_stage_transition_parent_checks():
    stage2 = plan_curriculum("full")[1]  # stage2 expects parent stage1

    # Missing parent manifest
    with pytest.raises(CurriculumError, match="Missing parent manifest"):
        validate_stage_transition(stage2, parent_manifest=None, current_fingerprints={})

    # Wrong parent name
    bad_manifest = {"stage_name": "stage0", "checkpoint_sha256": "abc"}
    with pytest.raises(CurriculumError, match="Parent stage name mismatch"):
        validate_stage_transition(stage2, parent_manifest=bad_manifest, current_fingerprints={})

    # Missing checkpoint sha256
    no_ckpt_manifest = {"stage_name": "stage1"}
    with pytest.raises(CurriculumError, match="missing checkpoint_sha256"):
        validate_stage_transition(stage2, parent_manifest=no_ckpt_manifest, current_fingerprints={})

    # Stale fingerprint
    stale_manifest = {
        "stage_name": "stage1",
        "checkpoint_sha256": "abc1234",
        "fingerprints": {"dataset": "fp_old"},
    }
    with pytest.raises(CurriculumError, match="Stale fingerprint"):
        validate_stage_transition(
            stage2,
            parent_manifest=stale_manifest,
            current_fingerprints={"dataset": "fp_new"},
        )

    # Valid transition
    valid_manifest = {
        "stage_name": "stage1",
        "checkpoint_sha256": "abc1234",
        "fingerprints": {"dataset": "fp_curr"},
    }
    validate_stage_transition(
        stage2,
        parent_manifest=valid_manifest,
        current_fingerprints={"dataset": "fp_curr"},
    )
