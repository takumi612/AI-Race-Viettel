from __future__ import annotations

import argparse
import ast
import gzip
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
            """# Suy luận Clinical NLP trên Kaggle

Notebook này nạp checkpoint NER đã train từ `results.zip` hoặc dataset kết quả
đã được Kaggle tự giải nén để tạo bài nộp mới. Notebook không huấn luyện hoặc
fine-tune mô hình.

Trước khi chọn **Run All**, hãy attach dataset kết quả, dataset input mới, bật
GPU và bật Internet trong Kaggle.
"""
        ),
        markdown_cell("## 1. Cấu hình runtime và tìm dataset kết quả"),
        code_cell(
            '''from pathlib import Path
import importlib
import importlib.util
import gzip
import json
import os
import shutil
import subprocess
import sys
import zipfile

KAGGLE_INPUT_ROOT = Path("/kaggle/input")
KAGGLE_WORKING_ROOT = Path("/kaggle/working")
GITHUB_REPO_URL = "https://github.com/takumi612/AI-Race-Viettel.git"
GITHUB_BRANCH = "main"
GITHUB_COMMIT = "f2a699ee138f35311994da30b055739153e6dd2d"
PROJECT_ROOT_OVERRIDE = ""
RESULTS_ZIP_OVERRIDE = ""
INPUT_SOURCE_OVERRIDE = ""
# Kaggle ships a CUDA-compatible torch/transformers stack. Installing the
# unpinned requirements file can replace it with incompatible versions.
INSTALL_MISSING_DEPENDENCIES = True
ENABLE_QWEN_RERANKER = True
INSTALL_VLLM = ENABLE_QWEN_RERANKER
QWEN_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct-AWQ"
QWEN_GPU_MEMORY_UTILIZATION = 0.50
QWEN_MAX_MODEL_LEN = 4096
QWEN_BATCH_SIZE = 64
REQUIRE_GPU = True

if not KAGGLE_INPUT_ROOT.is_dir():
    raise RuntimeError("This notebook must run in a Kaggle environment.")
KAGGLE_WORKING_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_DISABLED", "true")
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

def log_gpu_state(stage: str) -> None:
    try:
        import torch
        if not torch.cuda.is_available():
            print(f"[GPU] {stage}: CUDA unavailable")
            return
        free, total = torch.cuda.mem_get_info()
        gib = 1024 ** 3
        print(
            f"[GPU] {stage}: device={torch.cuda.get_device_name(0)!r}, "
            f"free={free / gib:.2f}/{total / gib:.2f} GiB, "
            f"allocated={torch.cuda.memory_allocated() / gib:.2f} GiB, "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f} GiB, "
            f"peak={torch.cuda.max_memory_allocated() / gib:.2f} GiB"
        )
    except Exception as exc:
        print(f"[GPU] {stage}: status unavailable ({exc})")

def _is_project(path: Path) -> bool:
    return (path / "clinical_nlp_lab").is_dir() and (path / "requirements-kaggle.txt").is_file()

project_candidates = []
if PROJECT_ROOT_OVERRIDE.strip():
    project_candidates.append(Path(PROJECT_ROOT_OVERRIDE).expanduser())
clone_dir = KAGGLE_WORKING_ROOT / "AI-Race-Viettel"
project_candidates.append(clone_dir / "v2")
PROJECT_ROOT = next((path.resolve() for path in project_candidates if _is_project(path)), None)
if PROJECT_ROOT is None:
    if clone_dir.exists() and not _is_project(clone_dir / "v2"):
        raise RuntimeError(f"Clone destination exists but is not a valid project: {clone_dir}")
    if not clone_dir.exists():
        subprocess.run(
            ["git", "clone", GITHUB_REPO_URL, str(clone_dir)],
            check=True,
        )
        subprocess.run(
            ["git", "-C", str(clone_dir), "checkout", "--detach", GITHUB_COMMIT],
            check=True,
        )
    PROJECT_ROOT = (clone_dir / "v2").resolve()
if not _is_project(PROJECT_ROOT):
    raise FileNotFoundError(f"Could not resolve cloned project at {PROJECT_ROOT}")
sys.path.insert(0, str(PROJECT_ROOT))
log_gpu_state("notebook_bootstrap")'''
        ),
        markdown_cell("## 2. Khôi phục checkpoint và artifact suy luận"),
        code_cell(
            '''REQUIRED_RESULTS_MEMBERS = {
    "training_artifacts/ner_model/model.safetensors",
    "training_artifacts/ner_model/config.json",
    "training_artifacts/ner_model/tokenizer.json",
}
ARTIFACT_PREFIX_CANDIDATES = ("artifacts/", "AI-Race-Viettel/v2/artifacts/")
EXTRACT_PREFIXES = (
    "training_artifacts/ner_model/",
    "artifacts/",
    "AI-Race-Viettel/v2/artifacts/",
)

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
def _artifact_directory(source_root: Path) -> Path | None:
    candidates = (
        source_root / "artifacts",
        source_root / "AI-Race-Viettel" / "v2" / "artifacts",
    )
    return next((path for path in candidates if (path / "config.json").is_file()), None)

def _is_results_directory(path: Path) -> bool:
    return (
        path.is_dir()
        and (path / "training_artifacts" / "ner_model").is_dir()
        and _artifact_directory(path) is not None
    )

results_dirs = []
if RESULTS_ZIP_OVERRIDE.strip():
    override = Path(RESULTS_ZIP_OVERRIDE).expanduser()
    if override.is_dir():
        results_dirs.append(override)
else:
    results_dirs.extend(
        path.parent
        for path in KAGGLE_INPUT_ROOT.rglob("training_artifacts")
        if _is_results_directory(path.parent)
    )
RESULTS_DIRS = [path.resolve() for path in results_dirs if _is_results_directory(path)]
RESULTS_DIRS = sorted(set(RESULTS_DIRS))
if len(RESULTS_ZIPS) + len(RESULTS_DIRS) != 1:
    raise FileNotFoundError(
        "Expected exactly one results.zip or extracted results directory, "
        f"found zips={RESULTS_ZIPS}, directories={RESULTS_DIRS}"
    )
RESULTS_SOURCE = RESULTS_ZIPS[0] if RESULTS_ZIPS else RESULTS_DIRS[0]

def _copy_directory_bundle(source_root: Path) -> None:
    required_files = {
        "training_artifacts/ner_model/model.safetensors",
        "training_artifacts/ner_model/config.json",
        "training_artifacts/ner_model/tokenizer.json",
    }
    missing_files = [name for name in required_files if not (source_root / name).is_file()]
    artifact_source = _artifact_directory(source_root)
    if missing_files or artifact_source is None:
        raise ValueError(
            "Extracted results directory is missing checkpoint/artifact members: "
            f"files={sorted(missing_files)}, artifact_directory={artifact_source}"
        )
    shutil.copytree(artifact_source, KAGGLE_WORKING_ROOT / "artifacts", dirs_exist_ok=True)
    shutil.copytree(
        source_root / "training_artifacts" / "ner_model",
        KAGGLE_WORKING_ROOT / "training_artifacts" / "ner_model",
        dirs_exist_ok=True,
    )

if RESULTS_SOURCE.is_file():
    with zipfile.ZipFile(RESULTS_SOURCE) as archive:
        member_names = archive.namelist()
        safe_members = {_safe_archive_member(name).as_posix() for name in member_names}
        missing_members = REQUIRED_RESULTS_MEMBERS - safe_members
        artifact_prefixes = [
            prefix
            for prefix in ARTIFACT_PREFIX_CANDIDATES
            if f"{prefix}config.json" in safe_members
        ]
        if missing_members or len(artifact_prefixes) != 1:
            raise ValueError(
                f"results.zip is missing checkpoint/artifact members: "
                f"files={sorted(missing_members)}, artifact_prefixes={artifact_prefixes}"
            )
        RESULTS_ARTIFACT_PREFIX = artifact_prefixes[0]
        for member_name in member_names:
            member = _safe_archive_member(member_name)
            if member_name.startswith(RESULTS_ARTIFACT_PREFIX):
                artifact_relative = member.relative_to(Path(RESULTS_ARTIFACT_PREFIX))
                destination = (KAGGLE_WORKING_ROOT / "artifacts" / artifact_relative).resolve()
            elif member_name.startswith("training_artifacts/ner_model/"):
                destination = (KAGGLE_WORKING_ROOT / member).resolve()
            else:
                continue
            if KAGGLE_WORKING_ROOT.resolve() not in destination.parents and destination != KAGGLE_WORKING_ROOT.resolve():
                raise ValueError(f"Archive member escapes working directory: {member_name!r}")
            if member_name.endswith("/"):
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member_name) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
else:
    _copy_directory_bundle(RESULTS_SOURCE)

def _normalize_plain_knowledge_bases() -> None:
    """Kaggle may transparently decompress uploaded .jsonl.gz files."""
    for relative in (
        "artifacts/icd10/icd10_dictionary",
        "artifacts/rxnorm/rxnorm_dictionary",
        "artifacts/rxnorm/rxnorm_relations",
    ):
        plain = KAGGLE_WORKING_ROOT / f"{relative}.jsonl"
        compressed = KAGGLE_WORKING_ROOT / f"{relative}.jsonl.gz"
        if plain.is_file() and not compressed.exists():
            compressed.parent.mkdir(parents=True, exist_ok=True)
            with plain.open("rt", encoding="utf-8", newline="") as source, gzip.open(
                compressed, "wt", encoding="utf-8", newline=""
            ) as target:
                shutil.copyfileobj(source, target)

_normalize_plain_knowledge_bases()

NER_MODEL_DIR = KAGGLE_WORKING_ROOT / "training_artifacts" / "ner_model"
ARTIFACT_DIR = KAGGLE_WORKING_ROOT / "artifacts"
required_kb = (
    ARTIFACT_DIR / "icd10" / "icd10_dictionary.jsonl.gz",
    ARTIFACT_DIR / "rxnorm" / "rxnorm_dictionary.jsonl.gz",
)
missing_kb = [str(path) for path in required_kb if not path.is_file()]
if missing_kb:
    raise FileNotFoundError(
        "Knowledge-base artifacts are missing after Kaggle decompression normalization: "
        f"{missing_kb}"
    )
if not all((NER_MODEL_DIR / name).is_file() for name in ("model.safetensors", "config.json", "tokenizer.json")):
    raise FileNotFoundError(f"NER checkpoint could not be resolved at {NER_MODEL_DIR}")

from clinical_nlp_lab.config import load_config

REQUIRED_ARTIFACT_FILES = (
    "config.json",
    "entity_type_mapping.json",
    "assertion_mapping.json",
    "relation_mapping.json",
)
missing_artifact_files = [
    name for name in REQUIRED_ARTIFACT_FILES if not (ARTIFACT_DIR / name).is_file()
]
if missing_artifact_files:
    raise FileNotFoundError(f"Artifact metadata is incomplete: {missing_artifact_files}")

RUNTIME_CONFIG = load_config(ARTIFACT_DIR / "config.json")
REQUIRED_CONFIG_KEYS = {
    "max_length",
    "stride",
    "embedding_model_name",
    "candidate_top_k",
    "candidate_output_k",
    "thresholds",
}
missing_config_keys = sorted(REQUIRED_CONFIG_KEYS - set(RUNTIME_CONFIG))
if missing_config_keys:
    raise ValueError(f"Saved artifact config is incompatible: missing {missing_config_keys}")
if not isinstance(RUNTIME_CONFIG["thresholds"], dict):
    raise TypeError("Saved artifact config thresholds must be a dictionary")

MODEL_CONFIG = json.loads((NER_MODEL_DIR / "config.json").read_text(encoding="utf-8"))
if MODEL_CONFIG.get("model_type") != "xlm-roberta":
    raise ValueError(f"Unsupported checkpoint model type: {MODEL_CONFIG.get('model_type')!r}")
MODEL_LABELS = set(MODEL_CONFIG.get("label2id", {}))
required_model_labels = {
    "O",
    "B-DISEASE", "I-DISEASE",
    "B-DRUG", "I-DRUG",
    "B-LAB_NAME", "I-LAB_NAME",
    "B-LAB_RESULT", "I-LAB_RESULT",
    "B-SYMPTOM", "I-SYMPTOM",
}
if MODEL_LABELS != required_model_labels:
    raise ValueError(
        "Checkpoint label schema is incompatible: "
        f"expected={sorted(required_model_labels)}, actual={sorted(MODEL_LABELS)}"
    )
print(
    {
        "config_compatibility": "validated",
        "source_commit": GITHUB_COMMIT,
        "model_type": MODEL_CONFIG["model_type"],
        "model_transformers_version": MODEL_CONFIG.get("transformers_version"),
        "label_count": len(MODEL_LABELS),
    }
)
'''
        ),
        markdown_cell("## 3. Tìm dữ liệu input mới"),
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
        markdown_cell("## 4. Cài dependency suy luận và kiểm tra GPU"),
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
try:
    import torch
    import transformers
except Exception as exc:
    raise RuntimeError(
        "Kaggle's preinstalled torch/transformers stack is incompatible. "
        "Restart the Kaggle session and run this notebook from the first cell; "
        "do not enable INSTALL_MISSING_DEPENDENCIES."
    ) from exc
if missing and INSTALL_MISSING_DEPENDENCIES:
    requirements = PROJECT_ROOT / "requirements-kaggle.txt"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(requirements)], check=True)
    importlib.invalidate_caches()
if INSTALL_VLLM and importlib.util.find_spec("vllm") is None:
    # Do not let vLLM overwrite Kaggle's torch/transformers packages.
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "vllm==0.25.1"], check=True)

missing_after = [package for package, module in required_imports.items() if importlib.util.find_spec(module) is None]
if missing_after:
    raise RuntimeError(f"Missing inference dependencies: {missing_after}")

import torch
if REQUIRE_GPU and not torch.cuda.is_available():
    raise RuntimeError("GPU is required. Open Kaggle Settings and select a GPU accelerator.")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'not available'}")
log_gpu_state("dependencies_ready")'''
        ),
        markdown_cell("## 5. Chạy suy luận từ checkpoint đã đóng gói"),
        code_cell(
            '''from clinical_nlp_lab.data import load_input_documents
from clinical_nlp_lab.pipeline import run_inference

INPUT_DOCUMENTS = load_input_documents(INPUT_SOURCE)
if not INPUT_DOCUMENTS:
    raise ValueError("No inference documents were loaded.")

OUTPUT_DIR = KAGGLE_WORKING_ROOT / "output"
DIAGNOSTICS_DIR = KAGGLE_WORKING_ROOT / "diagnostics"
OUTPUT_ZIP = KAGGLE_WORKING_ROOT / "output.zip"
log_gpu_state("before_run_inference")
INFERENCE_SUMMARY = run_inference(
    INPUT_SOURCE,
    OUTPUT_DIR,
    ARTIFACT_DIR,
    create_zip=True,
    diagnostics_dir=DIAGNOSTICS_DIR,
    zip_path=OUTPUT_ZIP,
    ner_model_dir=NER_MODEL_DIR,
    enable_qwen_reranker=ENABLE_QWEN_RERANKER,
    qwen_model_name=QWEN_MODEL_NAME,
    qwen_gpu_memory_utilization=QWEN_GPU_MEMORY_UTILIZATION,
    qwen_max_model_len=QWEN_MAX_MODEL_LEN,
    qwen_batch_size=QWEN_BATCH_SIZE,
)
log_gpu_state("after_run_inference")
print(INFERENCE_SUMMARY)'''
        ),
        markdown_cell("## 6. Kiểm tra bài nộp và ghi manifest"),
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
    "config_compatibility": "validated",
    "source_commit": GITHUB_COMMIT,
    "enable_qwen_reranker": ENABLE_QWEN_RERANKER,
    "qwen_model_name": QWEN_MODEL_NAME if ENABLE_QWEN_RERANKER else None,
    "checkpoint_source": str(NER_MODEL_DIR),
    "project_root": str(PROJECT_ROOT),
    "results_zip": str(RESULTS_SOURCE),
    "input_documents": len(INPUT_DOCUMENTS),
    "submission_entities": INFERENCE_SUMMARY["submission_entity_count"],
    "output_zip": str(OUTPUT_ZIP),
}
write_json(KAGGLE_WORKING_ROOT / "run_manifest.json", RUN_MANIFEST)
print(RUN_MANIFEST)'''
        ),
        markdown_cell(
            """## 7. Tải kết quả

Sau khi chọn **Save Version → Save & Run All**, tải các output Kaggle sau:

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
