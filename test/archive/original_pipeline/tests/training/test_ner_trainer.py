import json

import numpy as np
import pytest

from src.config import NERConfig, PipelineConfig
from src.ner.model_extractor import ModelNERExtractor, merge_hybrid_entities
from src.training.ner.bio import LABEL2ID
from src.training.ner.config import NERTrainingConfig
from src.training.ner.train import (
    build_compute_metrics,
    build_ner_run_plan,
    main,
)


def _training_config_mapping():
    return {
        "schema_version": 1,
        "base_model": "xlm-roberta-base",
        "dataset_dir": "data/training",
        "output_dir": "artifacts/training/ner",
        "seed": 20260719,
        "max_length": 384,
        "stride": 64,
        "num_train_epochs": 3,
        "learning_rate": 2e-5,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "fp16": True,
        "gradient_checkpointing": True,
        "early_stopping_patience": 2,
        "save_steps": 100,
        "logging_steps": 20,
        "recall_floor": 0.60,
        "local_files_only": False,
    }


def _write_training_dataset(project_root):
    dataset = project_root / "data" / "training"
    (dataset / "ner").mkdir(parents=True)
    (dataset / "manifests").mkdir()
    (dataset / "manifests" / "build.json").write_text(
        json.dumps({"schema_version": 1, "build_id": "build-1"}),
        encoding="utf-8",
    )
    records = [
        {
            "record_id": "s1",
            "text": "Sốt",
            "entities": [],
            "split": "synthetic_train",
        },
        {
            "record_id": "s2",
            "text": "Ho",
            "entities": [],
            "split": "synthetic_validation",
        },
        {
            "record_id": "101",
            "text": "Đau",
            "entities": [],
            "split": "trusted_fold",
            "fold": 0,
        },
        {
            "record_id": "102",
            "text": "Mệt",
            "entities": [],
            "split": "trusted_fold",
            "fold": 1,
        },
        {
            "record_id": "181",
            "text": "holdout",
            "entities": [],
            "split": "holdout",
        },
    ]
    (dataset / "ner" / "records.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_ner_training_config_is_strict_and_colab_sized():
    config = NERTrainingConfig.from_mapping(_training_config_mapping())

    assert config.base_model == "xlm-roberta-base"
    assert config.max_length == 384
    assert config.gradient_accumulation_steps == 4
    assert config.to_mapping()["fp16"] is True

    invalid = _training_config_mapping()
    invalid["unknown"] = True
    with pytest.raises(ValueError, match="unknown"):
        NERTrainingConfig.from_mapping(invalid)


def test_dry_run_plan_selects_only_stage_records(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _write_training_dataset(project_root)
    config = NERTrainingConfig.from_mapping(_training_config_mapping())

    synthetic = build_ner_run_plan(
        config,
        project_root=project_root,
        stage="synthetic",
        fold=None,
    )
    trusted = build_ner_run_plan(
        config,
        project_root=project_root,
        stage="trusted-fold",
        fold=1,
    )

    assert synthetic["train_record_ids"] == ["s1"]
    assert synthetic["eval_record_ids"] == ["s2"]
    assert trusted["train_record_ids"] == ["101"]
    assert trusted["eval_record_ids"] == ["102"]
    assert "181" not in json.dumps((synthetic, trusted))


def test_compute_metrics_uses_exact_absolute_spans():
    feature = {
        "record_id": "101",
        "text": "Sốt",
        "absolute_offsets": [[0, 0], [0, 3], [0, 0]],
    }
    logits = np.zeros((1, 3, len(LABEL2ID)), dtype=np.float32)
    logits[0, 0, LABEL2ID["O"]] = 1
    logits[0, 1, LABEL2ID["B-TRIỆU_CHỨNG"]] = 4
    logits[0, 2, LABEL2ID["O"]] = 1
    labels = np.asarray(
        [[-100, LABEL2ID["B-TRIỆU_CHỨNG"], -100]],
        dtype=np.int64,
    )

    metrics = build_compute_metrics((feature,))((logits, labels))

    assert metrics == {
        "precision": 1.0,
        "recall": 1.0,
        "f0_5": 1.0,
    }


class _Predictor:
    def predict(self, text):
        return [
            {
                "offset_mapping": [(0, 0), (0, 3), (0, 0)],
                "label_ids": [
                    LABEL2ID["O"],
                    LABEL2ID["B-TRIỆU_CHỨNG"],
                    LABEL2ID["O"],
                ],
                "confidences": [1.0, 0.92, 1.0],
            }
        ]


def test_model_extractor_is_injectable_and_hybrid_is_precision_gated():
    extractor = ModelNERExtractor(predictor=_Predictor(), threshold=0.9)
    model_entities = extractor.extract_entities("Sốt")
    rule_entities = [{"text": "Sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3]}]

    merged = merge_hybrid_entities(
        "Sốt",
        rule_entities,
        model_entities,
        default_threshold=0.95,
        per_type_thresholds={},
    )

    assert model_entities[0]["confidence"] == pytest.approx(0.92)
    assert merged == tuple(rule_entities)


def test_pipeline_ner_modes_are_strict_and_default_to_rule():
    assert PipelineConfig().ner.mode == "rule"
    model_config = PipelineConfig.from_mapping(
        {
            "ner": {
                "mode": "hybrid",
                "model_artifact": "artifacts/ner-v1",
            }
        }
    )
    assert model_config.ner.mode == "hybrid"

    with pytest.raises(ValueError, match="model_artifact"):
        NERConfig(mode="model")


def test_ner_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--stage" in output
    assert "--dry-run" in output
