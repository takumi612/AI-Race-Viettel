from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "medical_information_extraction_kaggle.ipynb"
GENERATOR = ROOT / "tools" / "build_kaggle_notebook.py"


def _source() -> str:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )


def _generated_setup_source() -> str:
    spec = importlib.util.spec_from_file_location("training_notebook_builder", GENERATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    notebook = module.build_notebook()
    setup_cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code")
    return "".join(setup_cell["source"])


def _data_source_resolver():
    tree = ast.parse(_generated_setup_source())
    resolver_nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {
            "_has_direct_files",
            "_unique_paths",
            "_require_one",
            "_resolve_data_sources",
        }
    ]
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=resolver_nodes, type_ignores=[]), "<resolver>", "exec"), namespace)
    return namespace["_resolve_data_sources"]


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


def test_kaggle_data_resolver_selects_real_files_from_nested_input_layout(tmp_path):
    mount = tmp_path / "kaggle-input"
    data_root = mount / "ai-race-input-v2" / "ai-race-clinical-data"
    training_root = data_root / "synthetic_train_v2"
    inference_root = data_root / "input" / "input"
    (training_root / "input").mkdir(parents=True)
    (training_root / "gt").mkdir()
    inference_root.mkdir(parents=True)
    (training_root / "input" / "001.txt").write_text("training", encoding="utf-8")
    (training_root / "gt" / "001.json").write_text("{}", encoding="utf-8")
    (inference_root / "101.txt").write_text("inference", encoding="utf-8")

    dataset_root, input_source = _data_source_resolver()(
        kaggle_input_root=mount,
        is_kaggle=True,
        dataset_root_override="",
        input_source_override="",
    )

    assert dataset_root == training_root
    assert input_source == inference_root
