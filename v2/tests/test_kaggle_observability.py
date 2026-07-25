import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[1]


def test_kaggle_notebook_has_step_logger_and_failure_context():
    notebook = json.loads((ROOT / "medical_information_extraction_kaggle.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])

    assert "def log_step" in source
    assert "STEP_START" in source
    assert "STEP_END" in source
    assert "STEP_ERROR" in source
    assert all(f"STEP {step}" in source for step in range(1, 10))


def test_training_notebook_exposes_qwen_toggle_and_passes_it_to_pipeline():
    notebook = json.loads((ROOT / "medical_information_extraction_kaggle.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "ENABLE_QWEN_RERANKER" in source
    assert "enable_qwen_reranker=ENABLE_QWEN_RERANKER" in source
    assert "QWEN_GPU_MEMORY_UTILIZATION = 0.50" in source
    assert "qwen_gpu_memory_utilization=QWEN_GPU_MEMORY_UTILIZATION" in source


def test_ner_config_uses_twenty_epochs():
    config = json.loads((ROOT / "artifacts" / "config.json").read_text(encoding="utf-8"))

    assert config["ner_epochs"] == 20


def test_training_script_logs_epoch_and_summary_context():
    source = (ROOT / "scripts" / "train_ner_subprocess.py").read_text(encoding="utf-8")

    assert "[TRAINING_START]" in source
    assert "[TRAINING_END]" in source
    assert "[TRAINING_ERROR]" in source


def test_notebook_finalization_logs_summary_without_status_argument_collision():
    notebook = json.loads((ROOT / "medical_information_extraction_kaggle.ipynb").read_text(encoding="utf-8"))
    final_cell = next(cell for cell in reversed(notebook["cells"]) if cell["cell_type"] == "code")
    final_source = "".join(final_cell["source"])
    events = []

    def log_step(step, status, message, **context):
        events.append((step, status, message, context))

    summary = SimpleNamespace(status="PASS", phase_completed="phase_13_packaging")
    namespace = {
        "SESSION": SimpleNamespace(completed=["phase_13_packaging"]),
        "ACTIVE_PHASES": ("phase_13_packaging",),
        "finish_run": lambda _session: summary,
        "log_step": log_step,
        "json": json,
    }

    exec(compile(final_source, "<finalization>", "exec"), namespace)

    assert events == [
        (
            9,
            "END",
            "Run completed",
            {"run_status": "PASS", "phase": "phase_13_packaging"},
        )
    ]
