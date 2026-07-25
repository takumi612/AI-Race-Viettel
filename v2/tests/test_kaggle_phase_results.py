from __future__ import annotations

import json

from clinical_nlp_lab.kaggle_phases import read_training_stage_result


def test_stage_result_exposes_executable_settings_and_metrics(tmp_path):
    payload = {
        "trained": True,
        "train_chunks": 120,
        "validation_chunks": 30,
        "configured_epochs": 4,
        "learning_rate": 1e-5,
        "batch_size": 2,
        "training_loss": 0.25,
        "best_metric": 0.81,
        "best_checkpoint": "checkpoint-42",
        "removed_checkpoints": [],
        "output_dir": "ner_model",
    }
    (tmp_path / "training_result.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert read_training_stage_result(tmp_path) == payload
