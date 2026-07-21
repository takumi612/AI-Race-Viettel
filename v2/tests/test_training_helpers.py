import importlib.util
import sys
import types
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]


def _load_training():
    package = types.ModuleType("clinical_nlp_lab")
    package.__path__ = [str(ROOT / "clinical_nlp_lab")]
    sys.modules["clinical_nlp_lab"] = package
    schema_spec = importlib.util.spec_from_file_location("clinical_nlp_lab.schema", ROOT / "clinical_nlp_lab" / "schema.py")
    schema_module = importlib.util.module_from_spec(schema_spec)
    sys.modules["clinical_nlp_lab.schema"] = schema_module
    schema_spec.loader.exec_module(schema_module)
    spec = importlib.util.spec_from_file_location("clinical_nlp_lab.training", ROOT / "clinical_nlp_lab" / "training.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clinical_nlp_lab.training"] = module
    spec.loader.exec_module(module)
    return module


def test_compute_non_o_metrics_reports_precision_recall_and_f1():
    training = _load_training()
    logits = np.array([[[5, 0, 0], [0, 5, 0], [5, 0, 0], [0, 0, 5]]])
    labels = np.array([[0, 1, -100, 1]])

    metrics = training.compute_non_o_metrics((logits, labels))

    assert metrics == {"precision": 0.5, "recall": 0.5, "f1": 0.5, "accuracy": 2 / 3}


def test_remove_nested_checkpoints_keeps_final_model(tmp_path):
    training = _load_training()
    (tmp_path / "model.safetensors").write_text("final", encoding="utf-8")
    checkpoint = tmp_path / "checkpoint-10"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_text("duplicate", encoding="utf-8")

    removed = training.remove_nested_checkpoints(tmp_path)

    assert removed == ["checkpoint-10"]
    assert (tmp_path / "model.safetensors").exists()
    assert not checkpoint.exists()
