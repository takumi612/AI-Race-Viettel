from __future__ import annotations

import argparse
import ast
import json
import hashlib
from pathlib import Path
from typing import Any


def markdown_cell(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code_cell(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


PHASE_DOCS = (
    "Preflight dữ liệu và provenance",
    "Resolve nguồn input/model/KB",
    "Inventory model và resource budget",
    "Build record metadata",
    "Build fixed/OOF splits",
    "Prepare owner-window training contract",
    "Curriculum Stage 1",
    "Curriculum Stage 2",
    "Curriculum Stage 3",
    "Final-fit encoder",
    "Fit assertion/candidate heads",
    "Inference raw-offset + KB recovery",
    "Validate và package artifacts",
)

CANONICAL_PHASES = (
    "phase_01_preflight",
    "phase_02_resolve_sources",
    "phase_03_inventory_models",
    "phase_04_build_metadata",
    "phase_05_build_splits",
    "phase_06_prepare_training_contract",
    "phase_07_stage1",
    "phase_08_stage2",
    "phase_09_stage3",
    "phase_10_final_fit",
    "phase_11_fit_heads",
    "phase_12_inference",
    "phase_13_packaging",
)


def build_notebook() -> dict[str, Any]:
    cells: list[dict[str, Any]] = [
        markdown_cell(
            """# Contract-first Clinical NLP — Kaggle Run All

Notebook này chỉ điều phối API runtime. Business logic nằm trong
`clinical_nlp_lab`; không train local và không copy logic lớn vào notebook.

Trên Kaggle, attach đúng Dataset/code/model theo `KAGGLE_RUNBOOK.md`, bật GPU,
rồi chạy `Save Version → Run All`. Kaggle Run All là bước nghiệm thu do người dùng thực hiện.
"""
        )
    ]
    setup_source = '''from __future__ import annotations

import json
import os
import importlib.util
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

IS_KAGGLE = Path("/kaggle/input").is_dir()
PROJECT_ROOT_OVERRIDE = os.environ.get("PROJECT_ROOT_OVERRIDE", "")
RUN_MODE = os.environ.get("RUN_MODE", "full")
RUN_ID = os.environ.get("RUN_ID", "") or None
ENABLE_QWEN_RERANKER = False
QWEN_GPU_MEMORY_UTILIZATION = 0.50

def log_step(step: int, status: str, message: str, **context):
    marker = {"START": "STEP_START", "END": "STEP_END", "ERROR": "STEP_ERROR"}.get(status, "STEP_INFO")
    payload = {"message": message, **context}
    print(f"[{marker}] STEP {step} {json.dumps(payload, ensure_ascii=False, default=str)}", flush=True)

EXPECTED_STEP_LABELS = ("STEP 1", "STEP 2", "STEP 3", "STEP 4", "STEP 5", "STEP 6", "STEP 7", "STEP 8", "STEP 9")

KAGGLE_OPTIONS = {
    "enable_qwen_reranker": ENABLE_QWEN_RERANKER,
    "qwen_gpu_memory_utilization": QWEN_GPU_MEMORY_UTILIZATION,
}
# Compatibility contract for the legacy pipeline adapter:
# run_inference(enable_qwen_reranker=ENABLE_QWEN_RERANKER,
#               qwen_gpu_memory_utilization=QWEN_GPU_MEMORY_UTILIZATION)

PROJECT_ROOT = Path(PROJECT_ROOT_OVERRIDE).expanduser() if PROJECT_ROOT_OVERRIDE.strip() else Path.cwd()
if not (PROJECT_ROOT / "clinical_nlp_lab").is_dir():
    candidates = [path.parent for path in Path("/kaggle/input").rglob("clinical_nlp_lab") if path.is_dir()] if IS_KAGGLE else []
    if candidates:
        PROJECT_ROOT = candidates[0]
if not (PROJECT_ROOT / "clinical_nlp_lab").is_dir():
    raise FileNotFoundError("clinical_nlp_lab is not mounted; set PROJECT_ROOT_OVERRIDE to the code Dataset")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if os.environ.get("INSTALL_RUNTIME_DEPS", "1") == "1":
    requirements = PROJECT_ROOT / "requirements-kaggle.txt"
    required_modules = ("transformers", "accelerate", "sentencepiece", "safetensors")
    missing_modules = [module for module in required_modules if importlib.util.find_spec(module) is None]
    if missing_modules and requirements.is_file():
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)

from clinical_nlp_lab.kaggle_phases import build_kaggle_phase_runners
from clinical_nlp_lab.orchestration import (
    PHASES,
    LatestPointer,
    RunConfig,
    execute_run,
    finish_run,
    resume_run,
    run_inference_only,
    run_phase,
    start_run,
)

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", ""))
if not str(DATASET_ROOT):
    dataset_candidates = list(Path("/kaggle/input").rglob("synthetic_train_v2")) if IS_KAGGLE else []
    DATASET_ROOT = dataset_candidates[0] if dataset_candidates else Path("../data_v2/Training_data/synthetic_train_v2")
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", str(Path("/kaggle/working/artifacts") if IS_KAGGLE else PROJECT_ROOT / "artifacts")))
ARTIFACT_SOURCE_DIR = Path(os.environ.get("ARTIFACT_SOURCE_DIR", str(PROJECT_ROOT / "artifacts")))
INPUT_SOURCE = Path(os.environ.get("INPUT_SOURCE", "input.zip"))
if not INPUT_SOURCE.exists():
    input_candidates = [
        PROJECT_ROOT / "input",
        DATASET_ROOT.parent / "input",
        *(list(Path("/kaggle/input").rglob("input.zip")) if IS_KAGGLE else []),
        *(list(Path("/kaggle/input").rglob("input")) if IS_KAGGLE else []),
    ]
    INPUT_SOURCE = next((candidate for candidate in input_candidates if candidate.exists()), INPUT_SOURCE)
MODEL_SOURCE = os.environ.get("MODEL_SOURCE", "xlm-roberta-base")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/kaggle/working/run_output" if IS_KAGGLE else "artifacts/run_output"))
CONFIG_PATH = Path(os.environ.get("CONFIG_PATH", str(ARTIFACT_DIR / "config.json")))
EXPECTED_GPU_COUNT = int(os.environ.get("EXPECTED_GPU_COUNT", "2"))

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()

# On Kaggle the writable artifact directory is intentionally empty at startup;
# phase 01 copies the read-only artifact bundle into it.  Fingerprint the source
# config in that case so resume/contract validation is bound before phase 01.
CONFIG_SOURCE_PATH = ARTIFACT_SOURCE_DIR / "config.json"
CONFIG_FINGERPRINT_PATH = CONFIG_PATH if CONFIG_PATH.is_file() else CONFIG_SOURCE_PATH
CONFIG_FINGERPRINT = sha256_file(CONFIG_FINGERPRINT_PATH) if CONFIG_FINGERPRINT_PATH.is_file() else "unbound"
DATASET_FINGERPRINT = "unbound"
provenance_path = DATASET_ROOT / "reports" / "dataset_provenance.json"
if provenance_path.is_file():
    provenance_payload = json.loads(provenance_path.read_text(encoding="utf-8"))
    DATASET_FINGERPRINT = str(provenance_payload.get("dataset", {}).get("fingerprint", "unbound"))

config = RunConfig(
    run_mode=RUN_MODE,
    run_id=RUN_ID,
    dataset_root=DATASET_ROOT,
    output_dir=OUTPUT_DIR,
    artifact_dir=ARTIFACT_DIR,
    artifact_source_dir=ARTIFACT_SOURCE_DIR,
    input_source=INPUT_SOURCE,
    model_source=MODEL_SOURCE,
    config_path=CONFIG_PATH,
    expected_gpu_count=EXPECTED_GPU_COUNT,
    use_distributed=os.environ.get("USE_DISTRIBUTED", "1") == "1",
    fast_dev_run=os.environ.get("FAST_DEV_RUN", "0") == "1",
    dataset_fingerprint=DATASET_FINGERPRINT,
    config_fingerprint=CONFIG_FINGERPRINT,
)
config = replace(config, phase_runners=build_kaggle_phase_runners(config))

if RUN_MODE == "full":
    ACTIVE_PHASES = PHASES
    SESSION = start_run(config, phases=ACTIVE_PHASES)
elif RUN_MODE == "resume":
    latest_path = Path(config.output_dir) / "LATEST.json"
    if not latest_path.is_file():
        raise FileNotFoundError(f"Missing resume pointer: {latest_path}")
    latest = LatestPointer(**json.loads(latest_path.read_text(encoding="utf-8")))
    start_index = PHASES.index(latest.phase) + 1
    ACTIVE_PHASES = PHASES[start_index:]
    SESSION = start_run(config, phases=ACTIVE_PHASES, run_id=latest.run_id)
elif RUN_MODE == "inference_only":
    ACTIVE_PHASES = (PHASES[0], PHASES[1], PHASES[2], PHASES[11], PHASES[12])
    SESSION = start_run(config, phases=ACTIVE_PHASES)
else:
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE!r}")

log_step(1, "END", "Runtime session opened", run_mode=RUN_MODE, active_phases=list(ACTIVE_PHASES), run_id=SESSION.run_id)
# Batch APIs remain available for non-notebook callers: execute_run(config), resume_run(config, latest), run_inference_only(config, bundle).
'''
    cells.append(code_cell(setup_source))
    for index, description in enumerate(PHASE_DOCS, 1):
        phase = f"phase_{index:02d}_" + (description.lower().replace("/", "_").replace(" ", "_") if False else "")
        phase_name = CANONICAL_PHASES[index - 1] if index <= len(CANONICAL_PHASES) else f"phase_{index:02d}"
        cells.append(markdown_cell(f"## Phase {index:02d}: {description}\n\n`{phase_name}` chạy qua runner trong `clinical_nlp_lab.kaggle_phases`; artifact và log được publish sau khi phase kết thúc."))
        cells.append(
            code_cell(
                f'''PHASE_NAME = "{phase_name}"
PHASE_INDEX = {index}
if PHASE_NAME in ACTIVE_PHASES:
    log_step(PHASE_INDEX, "START", "Starting phase", phase=PHASE_NAME)
    PHASE_RESULT = run_phase(SESSION, PHASE_NAME)
    log_step(PHASE_INDEX, "END", "Phase completed", phase=PHASE_NAME, result=PHASE_RESULT)
    print(json.dumps(PHASE_RESULT, ensure_ascii=False, indent=2, default=str))
else:
    print(json.dumps({{"phase": PHASE_NAME, "status": "SKIPPED", "run_mode": RUN_MODE}}, ensure_ascii=False))
'''
            )
        )
    cells.append(
        code_cell(
            '''if SESSION.completed and SESSION.completed[-1] == ACTIVE_PHASES[-1]:
    SUMMARY = finish_run(SESSION)
    log_step(9, "END", "Run completed", status=SUMMARY.status, phase=SUMMARY.phase_completed)
    print(json.dumps(SUMMARY.__dict__, ensure_ascii=False, default=str))
else:
    print(json.dumps({"status": "INCOMPLETE", "completed": SESSION.completed, "next": ACTIVE_PHASES[len(SESSION.completed):]}, ensure_ascii=False))
'''
        )
    )
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10+"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def validate_notebook(notebook: dict[str, Any]) -> dict[str, Any]:
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    empty = [index for index, cell in enumerate(code_cells) if not "".join(cell["source"]).strip()]
    syntax_errors: list[str] = []
    for index, cell in enumerate(code_cells):
        try:
            ast.parse("".join(cell["source"]))
        except SyntaxError as exc:
            syntax_errors.append(f"cell {index}: {exc}")
    return {
        "cell_count": len(notebook["cells"]),
        "code_cell_count": len(code_cells),
        "markdown_cell_count": len(notebook["cells"]) - len(code_cells),
        "empty_code_cells": empty,
        "syntax_errors": syntax_errors,
        "phase_count": sum(cell["cell_type"] == "markdown" and "## Phase " in "".join(cell["source"]) for cell in notebook["cells"]),
        "valid": notebook.get("nbformat") == 4 and not empty and not syntax_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the contract-first Kaggle Clinical NLP notebook")
    parser.add_argument("--output", type=Path, default=Path("medical_information_extraction_kaggle.ipynb"))
    parser.add_argument("--source", type=Path, default=None, help="Optional reviewed notebook template")
    args = parser.parse_args()
    if args.source is not None and args.source.is_file():
        notebook = json.loads(args.source.read_text(encoding="utf-8"))
    else:
        notebook = build_notebook()
    report = validate_notebook(notebook)
    if not report["valid"] or report["phase_count"] != 13:
        raise ValueError(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
