import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_kaggle_notebook_has_step_logger_and_failure_context():
    notebook = json.loads((ROOT / "medical_information_extraction_kaggle.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "def log_step" in source
    assert "STEP_START" in source
    assert "STEP_END" in source
    assert "STEP_ERROR" in source
    assert all(f"STEP {step}" in source for step in range(1, 10))


def test_ner_config_uses_twenty_epochs():
    config = json.loads((ROOT / "artifacts" / "config.json").read_text(encoding="utf-8"))

    assert config["ner_epochs"] == 20


def test_training_script_logs_epoch_and_summary_context():
    source = (ROOT / "scripts" / "train_ner_subprocess.py").read_text(encoding="utf-8")

    assert "[TRAINING_START]" in source
    assert "[TRAINING_END]" in source
    assert "[TRAINING_ERROR]" in source
