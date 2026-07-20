from __future__ import annotations

import argparse
import json
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


def build_notebook() -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    cells.append(
        markdown_cell(
            """# Clinical NLP Training on Kaggle

Notebook này train XLM-R NER bằng annotation thật, reload checkpoint vừa train,
chạy entity linking/assertion và tạo `/kaggle/working/output.zip`.

Trước khi Run All: attach Dataset có input + annotation, bật GPU, và bật
Internet nếu không attach sẵn code/model. Xem `KAGGLE_RUNBOOK.md`."""
        )
    )
    cells.append(markdown_cell("## 1. Runtime and repository bootstrap"))
    cells.append(
        code_cell(
            '''from pathlib import Path
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys

IS_KAGGLE = Path("/kaggle/input").is_dir()
KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")

GITHUB_REPO_URL = "https://github.com/takumi612/AI-Race-Viettel.git"
GITHUB_BRANCH = "main"
PROJECT_ROOT_OVERRIDE = ""
MODEL_NAME_OR_PATH_OVERRIDE = ""
INPUT_SOURCE_OVERRIDE = ""
TRAIN_SOURCE_OVERRIDE = ""
INSTALL_MISSING_DEPENDENCIES = True
FAST_DEV_RUN = False
REQUIRE_TRAINING_DATA = True
REQUIRE_GPU = True

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")

def _is_project(path: Path) -> bool:
    return (path / "clinical_nlp_lab").is_dir() and (path / "artifacts/config.json").is_file()

project_candidates = []
if PROJECT_ROOT_OVERRIDE.strip():
    project_candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())
project_candidates.append(Path.cwd())
if IS_KAGGLE:
    project_candidates.extend(marker.parent for marker in KAGGLE_INPUT_ROOT.rglob("clinical_nlp_lab") if marker.is_dir())

PROJECT_ROOT = next((path.resolve() for path in project_candidates if _is_project(path)), None)
clone_dir = KAGGLE_WORKING_ROOT / "AI-Race-Viettel"
if PROJECT_ROOT is None and IS_KAGGLE:
    if clone_dir.exists() and not _is_project(clone_dir):
        raise RuntimeError(f"Clone destination exists but is not a valid project: {clone_dir}")
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", GITHUB_BRANCH, GITHUB_REPO_URL, str(clone_dir)],
            check=True,
        )
    PROJECT_ROOT = clone_dir.resolve()
if PROJECT_ROOT is None:
    raise FileNotFoundError(
        "Project code not found. Enable Internet for Git clone or attach a code Dataset and set PROJECT_ROOT_OVERRIDE."
    )

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

required_imports = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "bm25s": "bm25s",
    "faiss-cpu": "faiss",
    "sentence-transformers": "sentence_transformers"
}
missing = [package for package, module in required_imports.items() if importlib.util.find_spec(module) is None]
if IS_KAGGLE and missing and INSTALL_MISSING_DEPENDENCIES:
    requirements = PROJECT_ROOT / "requirements-kaggle.txt"
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
        importlib.invalidate_caches()
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"Could not install {missing}. Enable Internet or attach an environment/model Dataset."
        ) from exc

missing_after = [package for package, module in required_imports.items() if importlib.util.find_spec(module) is None]
DEPENDENCIES_READY = not missing_after
if IS_KAGGLE and missing_after:
    raise RuntimeError(f"Missing training dependencies after setup: {missing_after}")

print({
    "is_kaggle": IS_KAGGLE,
    "project_root": str(PROJECT_ROOT),
    "dependencies_ready": DEPENDENCIES_READY,
    "missing_dependencies": missing_after,
})'''
        )
    )
    cells.append(markdown_cell("## 2. Discover attached input and annotations"))
    cells.append(
        code_cell(
            '''def _has_text_files(path: Path) -> bool:
    return path.is_dir() and any(path.glob("*.txt"))

def _is_archive_path(path: Path, search_root: Path) -> bool:
    try:
        relative_parts = path.relative_to(search_root).parts
    except ValueError:
        relative_parts = path.parts
    return bool({"test", "archive", "runtime_evidence"} & {part.lower() for part in relative_parts})

def _has_training_layout(path: Path) -> bool:
    if not path.is_dir():
        return False
    text_stems = {item.stem for item in path.glob("*.txt")}
    json_stems = {item.stem for item in path.glob("*.json")}
    direct_pairs = bool(text_stems & json_stems)
    split_pairs = _has_text_files(path / "input") and (path / "gt").is_dir() and any((path / "gt").glob("*.json"))
    return direct_pairs or split_pairs

search_roots = [PROJECT_ROOT]
if IS_KAGGLE:
    search_roots = [path for path in KAGGLE_INPUT_ROOT.iterdir() if path.is_dir()] + search_roots

input_candidates = []
if INPUT_SOURCE_OVERRIDE.strip():
    input_candidates.append(Path(INPUT_SOURCE_OVERRIDE).expanduser())
for root in search_roots:
    input_candidates.extend(path for path in root.rglob("input.zip") if not _is_archive_path(path, root))
    input_candidates.extend(
        path for path in root.rglob("input")
        if _has_text_files(path)
        and path.parent.name.lower() not in {"synthetic_train_v1", "train"}
        and not _is_archive_path(path, root)
    )
INPUT_SOURCE = next((path.resolve() for path in input_candidates if path.is_file() or _has_text_files(path)), None)
if INPUT_SOURCE is None:
    raise FileNotFoundError(
        "Real inference input not found. Attach a Dataset containing input.zip or input/<id>.txt."
    )

train_candidates = []
if TRAIN_SOURCE_OVERRIDE.strip():
    train_candidates.append(Path(TRAIN_SOURCE_OVERRIDE).expanduser())
for root in search_roots:
    train_candidates.extend(path for path in root.rglob("train") if not _is_archive_path(path, root))
    train_candidates.extend(path for path in root.rglob("synthetic_train_v1") if not _is_archive_path(path, root))

# Local-only fixture lets the generated notebook be executed as a smoke test.
if not IS_KAGGLE:
    local_fixture_candidates = [
        ancestor / "test/archive/tests/fixtures/paired_annotations"
        for ancestor in (PROJECT_ROOT, *PROJECT_ROOT.parents)
    ]
    local_fixture = next(
        (path for path in local_fixture_candidates if _has_training_layout(path)),
        None,
    )
    if local_fixture is not None:
        train_candidates.insert(0, local_fixture)

TRAIN_SOURCE = next((path.resolve() for path in train_candidates if _has_training_layout(path)), None)
if TRAIN_SOURCE is None and REQUIRE_TRAINING_DATA:
    raise FileNotFoundError(
        "No annotated training data found. Attach train/*.txt + *.json or synthetic_train_v1/input + gt."
    )

if IS_KAGGLE:
    RUN_ROOT = KAGGLE_WORKING_ROOT
else:
    RUN_ROOT = PROJECT_ROOT / "test/runtime_evidence/kaggle_notebook_simulation"
RUN_ROOT.mkdir(parents=True, exist_ok=True)
TRAINING_ROOT = RUN_ROOT / "training_artifacts"
NER_MODEL_DIR = TRAINING_ROOT / "ner_model"
OUTPUT_DIR = RUN_ROOT / "output"
DIAGNOSTICS_DIR = RUN_ROOT / "diagnostics"
OUTPUT_ZIP = RUN_ROOT / "output.zip"
TRAINED_ARTIFACTS_ZIP = RUN_ROOT / "trained_ner_artifacts.zip"

print({
    "input_source": str(INPUT_SOURCE),
    "train_source": str(TRAIN_SOURCE) if TRAIN_SOURCE else None,
    "run_root": str(RUN_ROOT),
    "output_zip": str(OUTPUT_ZIP),
})'''
        )
    )
    cells.append(markdown_cell("## 3. Validate data, split by document, and check GPU"))
    cells.append(
        code_cell(
            '''from clinical_nlp_lab.config import load_config, set_reproducible_seed
from clinical_nlp_lab.data import (
    document_train_validation_split,
    load_annotated_documents,
    load_input_documents,
    validate_documents,
)
from clinical_nlp_lab.schema import write_json

CONFIG = load_config(PROJECT_ROOT / "artifacts/config.json")
SEED_STATUS = set_reproducible_seed(int(CONFIG["seed"]))
INPUT_DOCUMENTS = load_input_documents(INPUT_SOURCE)
ANNOTATED_DOCUMENTS = load_annotated_documents(TRAIN_SOURCE) if TRAIN_SOURCE else []
ANNOTATION_REPORT = validate_documents(ANNOTATED_DOCUMENTS)
if not ANNOTATION_REPORT["is_valid"]:
    raise ValueError(f"Training annotation validation failed: {ANNOTATION_REPORT['errors'][:10]}")
if REQUIRE_TRAINING_DATA and not ANNOTATED_DOCUMENTS:
    raise ValueError("Training is required but no annotated documents were loaded")

TRAIN_DOCUMENTS, VALIDATION_DOCUMENTS = document_train_validation_split(
    ANNOTATED_DOCUMENTS,
    float(CONFIG["validation_fraction"]),
    int(CONFIG["seed"]),
)
if FAST_DEV_RUN:
    TRAIN_DOCUMENTS = TRAIN_DOCUMENTS[: min(16, len(TRAIN_DOCUMENTS))]
    VALIDATION_DOCUMENTS = VALIDATION_DOCUMENTS[: min(4, len(VALIDATION_DOCUMENTS))]

GPU_STATUS = {"available": False, "name": None}
if DEPENDENCIES_READY:
    import torch
    GPU_STATUS = {
        "available": torch.cuda.is_available(),
        "name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
if IS_KAGGLE and REQUIRE_GPU and not GPU_STATUS["available"]:
    raise RuntimeError("GPU is required. Open Kaggle Settings and select a GPU accelerator.")

print({
    "input_documents": len(INPUT_DOCUMENTS),
    "annotated_documents": len(ANNOTATED_DOCUMENTS),
    "train_documents": len(TRAIN_DOCUMENTS),
    "validation_documents": len(VALIDATION_DOCUMENTS),
    "entities": ANNOTATION_REPORT["entity_count"],
    "gpu": GPU_STATUS,
    "seed": SEED_STATUS,
})'''
        )
    )
    cells.append(markdown_cell("## 4. Training configuration"))
    cells.append(
        code_cell(
            '''def _looks_like_xlmr_model(path: Path) -> bool:
    config_path = path / "config.json"
    if not config_path.is_file():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return payload.get("model_type") in {"xlm-roberta", "roberta"} and (
        (path / "tokenizer.json").is_file()
        or (path / "sentencepiece.bpe.model").is_file()
    )

attached_models = []
if IS_KAGGLE:
    attached_models = [path.parent for path in KAGGLE_INPUT_ROOT.rglob("config.json") if _looks_like_xlmr_model(path.parent)]

if MODEL_NAME_OR_PATH_OVERRIDE.strip():
    MODEL_SOURCE = MODEL_NAME_OR_PATH_OVERRIDE
elif attached_models:
    MODEL_SOURCE = str(attached_models[0])
else:
    MODEL_SOURCE = str(CONFIG["ner_model_name"])

NER_EPOCHS = 1 if FAST_DEV_RUN else int(CONFIG["ner_epochs"])
TRAIN_BATCH_SIZE = 2 if FAST_DEV_RUN else int(CONFIG["batch_size"])
LEARNING_RATE = float(CONFIG["learning_rate"])

print({
    "model_source": MODEL_SOURCE,
    "epochs": NER_EPOCHS,
    "batch_size": TRAIN_BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "fast_dev_run": FAST_DEV_RUN,
})'''
        )
    )
    cells.append(markdown_cell("## 5. Train XLM-R token classifier"))
    cells.append(
        code_cell(
            '''from clinical_nlp_lab.training import train_transformer_ner, transformer_training_availability

TRAINING_AVAILABILITY = transformer_training_availability()
if DEPENDENCIES_READY and TRAIN_DOCUMENTS:
    if (NER_MODEL_DIR / "model.safetensors").exists() or (NER_MODEL_DIR / "pytorch_model.bin").exists():
        print(f"[SKIP] Found existing NER weights at {NER_MODEL_DIR}, skipping training.")
        NER_TRAINING_RESULT = {"trained": False, "reason": "checkpoint_exists", "model_dir": str(NER_MODEL_DIR)}
    else:
        NER_TRAINING_RESULT = train_transformer_ner(
            TRAIN_DOCUMENTS,
            VALIDATION_DOCUMENTS,
            NER_MODEL_DIR,
            model_name=MODEL_SOURCE,
            max_length=int(CONFIG["max_length"]),
            stride=int(CONFIG["stride"]),
            learning_rate=LEARNING_RATE,
            epochs=NER_EPOCHS,
            batch_size=TRAIN_BATCH_SIZE,
            seed=int(CONFIG["seed"]),
        )
else:
    NER_TRAINING_RESULT = {
        "trained": False,
        "reason": TRAINING_AVAILABILITY.reason if not DEPENDENCIES_READY else "No training documents",
    }

TRAINING_ROOT.mkdir(parents=True, exist_ok=True)
write_json(TRAINING_ROOT / "training_result.json", NER_TRAINING_RESULT)
if IS_KAGGLE and not NER_TRAINING_RESULT.get("trained") and NER_TRAINING_RESULT.get("reason") != "checkpoint_exists":
    raise RuntimeError(f"NER training did not complete: {NER_TRAINING_RESULT}")

print(NER_TRAINING_RESULT)'''
        )
    )
    cells.append(markdown_cell("## 6. Package the trained checkpoint"))
    cells.append(
        code_cell(
            '''archive_base = TRAINED_ARTIFACTS_ZIP.with_suffix("")
created_archive = shutil.make_archive(str(archive_base), "zip", root_dir=TRAINING_ROOT)
assert Path(created_archive).is_file()
print({"trained_artifacts_zip": created_archive, "bytes": Path(created_archive).stat().st_size})'''
        )
    )
    cells.append(markdown_cell("## 6.5 Garbage collection"))
    cells.append(
        code_cell(
            '''import gc\nimport torch\ngc.collect()\nif torch.cuda.is_available():\n    torch.cuda.empty_cache()\n    print("Đã dọn dẹp xong bộ nhớ GPU.")'''
        )
    )
    cells.append(markdown_cell("## 7. Inference with the newly trained NER model"))
    cells.append(
        code_cell(
            '''from clinical_nlp_lab.pipeline import run_inference

ACTIVE_NER_MODEL = NER_MODEL_DIR if (NER_TRAINING_RESULT.get("trained") or NER_TRAINING_RESULT.get("reason") == "checkpoint_exists") else None
INFERENCE_SUMMARY = run_inference(
    INPUT_SOURCE,
    OUTPUT_DIR,
    PROJECT_ROOT / "artifacts",
    create_zip=True,
    diagnostics_dir=DIAGNOSTICS_DIR,
    zip_path=OUTPUT_ZIP,
    ner_model_dir=ACTIVE_NER_MODEL,
)
print(INFERENCE_SUMMARY)'''
        )
    )
    cells.append(markdown_cell("## 7.5 Stage-by-Stage Benchmark Evaluation & Error Diagnostics"))
    cells.append(
        code_cell(
            '''from clinical_nlp_lab.evaluation import evaluate_benchmark
from clinical_nlp_lab.pipeline import ClinicalNLPPipeline

eval_roots = []
if IS_KAGGLE:
    for root in KAGGLE_INPUT_ROOT.rglob("eval"):
        if (root / "gt").is_dir() and ((root / "input").is_dir() or (root / "input.zip").is_file()):
            eval_roots.append(root)
else:
    eval_local = PROJECT_ROOT.parent / "ai-race-clinical-data/eval"
    if (eval_local / "gt").is_dir():
        eval_roots.append(eval_local)

if eval_roots:
    eval_dir = eval_roots[0]
    print(f"[EVAL] Found evaluation benchmark dataset at: {eval_dir}")
    eval_input = eval_dir / "input" if (eval_dir / "input").is_dir() else eval_dir / "input.zip"
    eval_gt_dir = eval_dir / "gt"

    pipeline = ClinicalNLPPipeline(PROJECT_ROOT / "artifacts", ner_model_dir=ACTIVE_NER_MODEL)
    eval_report = evaluate_benchmark(eval_input, eval_gt_dir, pipeline, RUN_ROOT)
else:
    print("[EVAL] No benchmark `eval` dataset attached (eval/gt + eval/input). Skipping benchmark evaluation.")'''
        )
    )
    cells.append(markdown_cell("## 8. Validate submission and write run manifest"))
    cells.append(
        code_cell(
            '''import zipfile
from clinical_nlp_lab.schema import validate_submission_payload

schema_errors = {}
for document in INPUT_DOCUMENTS:
    prediction_path = OUTPUT_DIR / f"{document.document_id}.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    errors = validate_submission_payload(prediction, document.raw_text)
    if errors:
        schema_errors[document.document_id] = errors
if schema_errors:
    raise ValueError(f"Submission schema errors: {list(schema_errors.items())[:3]}")

with zipfile.ZipFile(OUTPUT_ZIP) as archive:
    zip_names = archive.namelist()
    assert len(zip_names) == len(INPUT_DOCUMENTS)
    assert all(name.startswith("output/") and not name.startswith("output/output/") for name in zip_names)
    assert archive.testzip() is None

RUN_MANIFEST = {
    "trained": bool(NER_TRAINING_RESULT.get("trained")),
    "active_ner": INFERENCE_SUMMARY.get("active_ner"),
    "train_documents": len(TRAIN_DOCUMENTS),
    "validation_documents": len(VALIDATION_DOCUMENTS),
    "input_documents": len(INPUT_DOCUMENTS),
    "submission_entities": INFERENCE_SUMMARY["submission_entity_count"],
    "schema_error_count": 0,
    "offset_error_count": INFERENCE_SUMMARY["offset_error_count"],
    "output_zip": str(OUTPUT_ZIP),
    "trained_artifacts_zip": str(TRAINED_ARTIFACTS_ZIP),
}
write_json(RUN_ROOT / "run_manifest.json", RUN_MANIFEST)
print(RUN_MANIFEST)'''
        )
    )
    cells.append(
        markdown_cell(
            """## 9. Download results

Kaggle outputs:

- `/kaggle/working/output.zip`
- `/kaggle/working/trained_ner_artifacts.zip`
- `/kaggle/working/run_manifest.json`

Chọn **Save Version → Save & Run All**, sau đó tải file từ tab Output."""
        )
    )

    # Keep project discovery valid when the runtime is packaged under Code_E_Platform.
    for cell in cells:
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        source = source.replace(
            'project_candidates.append(Path.cwd())\nif IS_KAGGLE:\n    project_candidates.extend(marker.parent for marker in KAGGLE_INPUT_ROOT.rglob("clinical_nlp_lab") if marker.is_dir())',
            '''project_candidates.append(Path.cwd() / "Code_E_Platform")
project_candidates.append(Path.cwd())
if IS_KAGGLE:
    for marker in KAGGLE_INPUT_ROOT.rglob("clinical_nlp_lab"):
        if marker.is_dir():
            project_candidates.extend([marker.parent, marker.parent.parent / "Code_E_Platform"])''',
        )
        source = source.replace(
            'clone_dir = KAGGLE_WORKING_ROOT / "AI-Race-Viettel"\nif PROJECT_ROOT is None and IS_KAGGLE:\n    if clone_dir.exists() and not _is_project(clone_dir):',
            'clone_dir = KAGGLE_WORKING_ROOT / "AI-Race-Viettel"\nif PROJECT_ROOT is None and IS_KAGGLE:\n    clone_candidates = [clone_dir / "Code_E_Platform", clone_dir]\n    if clone_dir.exists() and not any(_is_project(path) for path in clone_candidates):',
        )
        source = source.replace(
            '    PROJECT_ROOT = clone_dir.resolve()',
            '    PROJECT_ROOT = next((path.resolve() for path in clone_candidates if _is_project(path)), None)',
        )
        source = source.replace(
            'search_roots = [PROJECT_ROOT]\nif IS_KAGGLE:\n    search_roots = [path for path in KAGGLE_INPUT_ROOT.iterdir() if path.is_dir()] + search_roots',
            'search_roots = [PROJECT_ROOT]\nif IS_KAGGLE:\n    search_roots = [path for path in KAGGLE_INPUT_ROOT.iterdir() if path.is_dir()] + search_roots\nelse:\n    search_roots.extend(ancestor for ancestor in PROJECT_ROOT.parents if (ancestor / "input.zip").is_file())',
        )
        cell["source"] = source.splitlines(keepends=True)
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
    return {
        "cell_count": len(notebook["cells"]),
        "code_cell_count": len(code_cells),
        "markdown_cell_count": len(notebook["cells"]) - len(code_cells),
        "empty_code_cells": empty,
        "valid": notebook.get("nbformat") == 4 and not empty,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Kaggle Clinical NLP training notebook")
    parser.add_argument("--output", type=Path, default=Path("medical_information_extraction_kaggle.ipynb"))
    args = parser.parse_args()
    notebook = build_notebook()
    report = validate_notebook(notebook)
    if not report["valid"]:
        raise ValueError(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
