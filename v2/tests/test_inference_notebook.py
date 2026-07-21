import ast
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "medical_information_extraction_inference_kaggle.ipynb"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_inference_notebook_is_clean_and_compiles():
    notebook = _load_notebook()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))


def test_inference_notebook_loads_results_without_training():
    source = "\n".join("".join(cell.get("source", [])) for cell in _load_notebook()["cells"])
    assert "RESULTS_ZIP_OVERRIDE" in source
    assert "training_artifacts/ner_model" in source
    assert "model.safetensors" in source
    assert "run_inference(" in source
    assert "validate_submission_payload" in source
    assert '"training_skipped": True' in source
    assert "train_ner_subprocess.py" not in source
    assert "Trainer(" not in source
    assert ".train()" not in source


def test_inference_notebook_checks_all_direct_bundled_requirements():
    source = "\n".join("".join(cell.get("source", [])) for cell in _load_notebook()["cells"])
    assert '"sentencepiece": "sentencepiece"' in source
    assert '"safetensors": "safetensors"' in source


def test_inference_runbook_covers_complete_kaggle_workflow():
    text = (ROOT / "KAGGLE_INFERENCE_RUNBOOK.md").read_text(encoding="utf-8")
    for phrase in (
        "results.zip",
        "input.zip",
        "medical_information_extraction_inference_kaggle.ipynb",
        "GPU",
        "Internet",
        "Run All",
        "output.zip",
        "training_skipped",
    ):
        assert phrase in text
