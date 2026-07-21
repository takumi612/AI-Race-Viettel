from __future__ import annotations

import argparse
import ast
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
    """Build the Kaggle notebook that performs inference from a supplied checkpoint."""
    cells: list[dict[str, Any]] = [
        markdown_cell(
            """# Clinical NLP inference on Kaggle

This notebook uses the packaged NER checkpoint in `results.zip` to create a
submission. It intentionally performs no model fitting. Attach the checkpoint
Dataset and an inference-input Dataset, then enable a Kaggle GPU before Run All.
"""
        ),
        markdown_cell("## 1. Runtime configuration"),
        code_cell(
            '''from pathlib import Path
import importlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
RESULTS_ZIP_OVERRIDE = ""
INPUT_SOURCE_OVERRIDE = ""
INSTALL_MISSING_DEPENDENCIES = True
INSTALL_VLLM = False
REQUIRE_GPU = True

if not KAGGLE_INPUT_ROOT.is_dir():
    raise RuntimeError("This notebook must run in a Kaggle environment.")
KAGGLE_WORKING_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")'''
        ),
        markdown_cell("## 2. Validate and safely unpack the checkpoint bundle"),
        code_cell(
            '''REQUIRED_RESULTS_MEMBERS = {
    "training_artifacts/ner_model/model.safetensors",
    "training_artifacts/ner_model/config.json",
    "training_artifacts/ner_model/tokenizer.json",
}
REQUIRED_RESULTS_PREFIXES = {
    "AI-Race-Viettel/v2/clinical_nlp_lab/",
    "AI-Race-Viettel/v2/artifacts/",
}
EXTRACT_PREFIXES = ("AI-Race-Viettel/v2/", "training_artifacts/ner_model/")

def _safe_archive_member(name: str) -> Path:
    member = Path(name)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError(f"Unsafe archive member: {name!r}")
    return member

results_candidates = []
if RESULTS_ZIP_OVERRIDE.strip():
    results_candidates.append(Path(RESULTS_ZIP_OVERRIDE).expanduser())
else:
    results_candidates.extend(KAGGLE_INPUT_ROOT.rglob("results.zip"))
RESULTS_ZIPS = [path.resolve() for path in results_candidates if path.is_file()]
if len(RESULTS_ZIPS) != 1:
    raise FileNotFoundError(f"Expected exactly one results.zip, found: {RESULTS_ZIPS}")
RESULTS_ZIP = RESULTS_ZIPS[0]

with zipfile.ZipFile(RESULTS_ZIP) as archive:
    member_names = archive.namelist()
    safe_members = {_safe_archive_member(name).as_posix() for name in member_names}
    missing_members = REQUIRED_RESULTS_MEMBERS - safe_members
    missing_prefixes = [
        prefix for prefix in REQUIRED_RESULTS_PREFIXES
        if not any(name.startswith(prefix) for name in safe_members)
    ]
    if missing_members or missing_prefixes:
        raise ValueError(
            f"results.zip is missing checkpoint/project members: "
            f"files={sorted(missing_members)}, prefixes={missing_prefixes}"
        )
    for member_name in member_names:
        member = _safe_archive_member(member_name)
        if not any(member_name.startswith(prefix) for prefix in EXTRACT_PREFIXES):
            continue
        destination = (KAGGLE_WORKING_ROOT / member).resolve()
        if KAGGLE_WORKING_ROOT.resolve() not in destination.parents and destination != KAGGLE_WORKING_ROOT.resolve():
            raise ValueError(f"Archive member escapes working directory: {member_name!r}")
        if member_name.endswith("/"):
            destination.mkdir(parents=True, exist_ok=True)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(member_name) as source, destination.open("wb") as target:
            shutil.copyfileobj(source, target)

PROJECT_ROOT = KAGGLE_WORKING_ROOT / "AI-Race-Viettel" / "v2"
NER_MODEL_DIR = KAGGLE_WORKING_ROOT / "training_artifacts" / "ner_model"
if not (PROJECT_ROOT / "clinical_nlp_lab").is_dir() or not (PROJECT_ROOT / "artifacts").is_dir():
    raise FileNotFoundError(f"Bundled project could not be resolved at {PROJECT_ROOT}")
if not all((NER_MODEL_DIR / name).is_file() for name in ("model.safetensors", "config.json", "tokenizer.json")):
    raise FileNotFoundError(f"NER checkpoint could not be resolved at {NER_MODEL_DIR}")
sys.path.insert(0, str(PROJECT_ROOT))'''
        ),
        markdown_cell("## 3. Discover inference documents"),
        code_cell(
            '''EXCLUDED_INPUT_PARTS = {
    "train", "training", "synthetic_train_v1", "archive", "diagnostics",
    "output", "training_artifacts", "ai-race-viettel",
}

def _is_inference_path(path: Path) -> bool:
    return not any(part.lower() in EXCLUDED_INPUT_PARTS for part in path.parts)

def _document_count(source: Path) -> int:
    if source.is_dir():
        return len(list(source.glob("*.txt")))
    if source.is_file() and source.name == "input.zip":
        with zipfile.ZipFile(source) as archive:
            return sum(
                not name.endswith("/") and name.lower().endswith(".txt")
                for name in archive.namelist()
            )
    return 0

input_candidates = []
if INPUT_SOURCE_OVERRIDE.strip():
    input_candidates.append(Path(INPUT_SOURCE_OVERRIDE).expanduser())
else:
    input_candidates.extend(path for path in KAGGLE_INPUT_ROOT.rglob("input.zip") if _is_inference_path(path))
    input_candidates.extend(
        path for path in KAGGLE_INPUT_ROOT.rglob("input")
        if path.is_dir() and _is_inference_path(path)
    )
valid_inputs = [(path.resolve(), _document_count(path)) for path in input_candidates]
valid_inputs = [(path, count) for path, count in valid_inputs if count > 0]
if not valid_inputs:
    raise FileNotFoundError("Attach input.zip or input/*.txt containing at least one inference document.")
if len(valid_inputs) != 1:
    raise RuntimeError(f"Expected exactly one inference input source, found: {[str(path) for path, _ in valid_inputs]}")
INPUT_SOURCE, discovered_document_count = valid_inputs[0]
print(f"Input source: {INPUT_SOURCE} ({discovered_document_count} text documents)")'''
        ),
        markdown_cell("## 4. Install inference dependencies and check the accelerator"),
        code_cell(
            '''required_imports = {
    "torch": "torch",
    "transformers": "transformers",
    "accelerate": "accelerate",
    "sentencepiece": "sentencepiece",
    "safetensors": "safetensors",
    "bm25s": "bm25s",
    "faiss-cpu": "faiss",
    "sentence-transformers": "sentence_transformers",
}
missing = [package for package, module in required_imports.items() if importlib.util.find_spec(module) is None]
if missing and INSTALL_MISSING_DEPENDENCIES:
    requirements = PROJECT_ROOT / "requirements-kaggle.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    importlib.invalidate_caches()
if INSTALL_VLLM and importlib.util.find_spec("vllm") is None:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm==0.25.1"], check=True)

missing_after = [package for package, module in required_imports.items() if importlib.util.find_spec(module) is None]
if missing_after:
    raise RuntimeError(f"Missing inference dependencies: {missing_after}")

import torch
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("GPU is required. Open Kaggle Settings and select a GPU accelerator.")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")'''
        ),
        markdown_cell("## 5. Run inference from the packaged checkpoint"),
        code_cell(
            '''from clinical_nlp_lab.data import load_input_documents
from clinical_nlp_lab.pipeline import run_inference

INPUT_DOCUMENTS = load_input_documents(INPUT_SOURCE)
if not INPUT_DOCUMENTS:
    raise ValueError("No inference documents were loaded.")

OUTPUT_DIR = KAGGLE_WORKING_ROOT / "output"
DIAGNOSTICS_DIR = KAGGLE_WORKING_ROOT / "diagnostics"
OUTPUT_ZIP = KAGGLE_WORKING_ROOT / "output.zip"
INFERENCE_SUMMARY = run_inference(
    INPUT_SOURCE,
    OUTPUT_DIR,
    PROJECT_ROOT / "artifacts",
    create_zip=True,
    diagnostics_dir=DIAGNOSTICS_DIR,
    zip_path=OUTPUT_ZIP,
    ner_model_dir=NER_MODEL_DIR,
)
print(INFERENCE_SUMMARY)'''
        ),
        markdown_cell("## 6. Validate the submission and write the inference manifest"),
        code_cell(
            '''from clinical_nlp_lab.schema import validate_submission_payload, write_json

for document in INPUT_DOCUMENTS:
    prediction_path = OUTPUT_DIR / f"{document.document_id}.json"
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Missing prediction: {prediction_path}")
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    errors = validate_submission_payload(prediction, document.raw_text)
    if errors:
        raise ValueError(f"Submission validation failed for {document.document_id}: {errors}")

with zipfile.ZipFile(OUTPUT_ZIP) as archive:
    zip_names = archive.namelist()
    expected_names = [f"output/{document.document_id}.json" for document in INPUT_DOCUMENTS]
    if len(zip_names) != len(INPUT_DOCUMENTS) or zip_names != expected_names:
        raise ValueError(f"Invalid output ZIP names: {zip_names}")
    bad_member = archive.testzip()
    if bad_member is not None:
        raise ValueError(f"Output ZIP CRC failure: {bad_member}")

RUN_MANIFEST = {
    "training_skipped": True,
    "checkpoint_source": str(NER_MODEL_DIR),
    "results_zip": str(RESULTS_ZIP),
    "input_documents": len(INPUT_DOCUMENTS),
    "submission_entities": INFERENCE_SUMMARY["submission_entity_count"],
    "output_zip": str(OUTPUT_ZIP),
}
write_json(KAGGLE_WORKING_ROOT / "run_manifest.json", RUN_MANIFEST)
print(RUN_MANIFEST)'''
        ),
        markdown_cell(
            """## 7. Download the result

After **Save Version → Save & Run All**, download these Kaggle outputs:

- `/kaggle/working/output.zip`
- `/kaggle/working/diagnostics/`
- `/kaggle/working/run_manifest.json`
"""
        ),
    ]
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
    empty_cells = [index for index, cell in enumerate(code_cells) if not "".join(cell["source"]).strip()]
    for cell in code_cells:
        if cell["execution_count"] is not None or cell["outputs"]:
            raise ValueError("Generated notebook must have no execution results")
        ast.parse("".join(cell["source"]))
    return {
        "cell_count": len(notebook["cells"]),
        "code_cell_count": len(code_cells),
        "empty_code_cells": empty_cells,
        "valid": notebook.get("nbformat") == 4 and not empty_cells,
    }


def write_notebook(output: Path) -> None:
    notebook = build_notebook()
    report = validate_notebook(notebook)
    if not report["valid"]:
        raise ValueError(report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), **report}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Kaggle Clinical NLP inference notebook")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "medical_information_extraction_inference_kaggle.ipynb",
    )
    args = parser.parse_args()
    write_notebook(args.output)


if __name__ == "__main__":
    main()
