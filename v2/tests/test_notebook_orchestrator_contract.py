from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "medical_information_extraction_kaggle.ipynb"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def test_training_notebook_is_thin_orchestrator_api_client():
    source = _source()
    ast.parse(source)
    assert "from clinical_nlp_lab.orchestration import" in source
    assert "execute_run(" in source
    assert "resume_run(" in source
    assert "run_inference_only(" in source
    assert "train_transformer_ner" not in source
    assert "Trainer(" not in source
    assert ".train()" not in source
    assert "PHASES" in source


def test_training_notebook_exposes_one_observable_code_cell_per_phase():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    code_cells = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 15  # setup + 13 phases + finalization
    for index in range(1, 14):
        assert f'PHASE_INDEX = {index}' in code_cells[index]
        assert "run_phase(SESSION, PHASE_NAME)" in code_cells[index]
