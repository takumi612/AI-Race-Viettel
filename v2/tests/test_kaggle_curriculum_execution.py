from __future__ import annotations

from collections import Counter

from clinical_nlp_lab.curriculum import plan_curriculum, resolve_stage_hyperparameters
from clinical_nlp_lab.sampling import select_stage_document_ids


def test_stage_hyperparameters_override_global_training_defaults():
    global_config = {"ner_epochs": 20, "learning_rate": 3e-5, "batch_size": 2}
    stage3 = plan_curriculum("full")[2]

    resolved = resolve_stage_hyperparameters(global_config, stage3, fast_dev_run=False)

    assert resolved == {
        "epochs": 6,
        "learning_rate": 1e-5,
        "batch_size": 2,
    }


def test_stage2_and_stage3_apply_different_source_exposure():
    synthetic = tuple(str(value) for value in range(201, 301))
    organizer = tuple(str(value) for value in range(101, 121))
    stage2, stage3 = plan_curriculum("full")[1:3]

    selected2 = select_stage_document_ids(synthetic, organizer, stage2, seed=42)
    selected3 = select_stage_document_ids(synthetic, organizer, stage3, seed=42)

    assert selected2 != selected3
    counts2 = Counter("organizer" if int(item) <= 200 else "synthetic" for item in selected2)
    counts3 = Counter("organizer" if int(item) <= 200 else "synthetic" for item in selected3)
    assert counts2["organizer"] / len(selected2) == 0.35
    assert counts3["organizer"] / len(selected3) == 0.80
    assert len(selected2) == len(synthetic) + len(organizer)
    assert len(selected3) == len(synthetic) + len(organizer)
    stage2_synthetic = [item for item in selected2 if int(item) > 200]
    stage3_synthetic = [item for item in selected3 if int(item) > 200]
    assert len(stage2_synthetic) == len(set(stage2_synthetic))
    assert len(stage3_synthetic) > len(set(stage3_synthetic))
