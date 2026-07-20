from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SOURCE_ROLES = {
    "Lab Clinical NLP End-to-End.docx": "project_contract_source",
    "input.zip": "unlabeled_inference_input_candidate",
    "ICD10.xlsx": "icd10_knowledge_source",
    "RxNorm_full_07062026.zip": "rxnorm_knowledge_source",
}

EXCLUDED_PARTS = {
    ".git",
    "__pycache__",
    ".test_tmp",
    ".pytest_cache",
    "test_artifacts",
    # Local transport clone used only to publish the selected deliverables.
    "git_push_workspace",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def artifact_type(relative: str) -> str:
    path = Path(relative)
    if relative == "medical_information_extraction_lab.ipynb":
        return "executable_notebook"
    if relative == "output.zip":
        return "submission_archive"
    if relative.endswith("README.md"):
        return "readme"
    if relative.endswith(".md"):
        return "project_documentation"
    if relative.endswith(".jsonl.gz"):
        return "compressed_knowledge_cache"
    if relative.endswith(".json"):
        return "machine_readable_artifact"
    if relative.endswith(".py"):
        return "python_source"
    if relative.endswith(".txt"):
        return "text_configuration"
    return "project_artifact"


def created_by(relative: str) -> str:
    if relative.startswith("artifacts/icd10") or relative.startswith("artifacts/rxnorm"):
        return "stage_2_knowledge_base_preprocessing"
    if relative.startswith("clinical_nlp_lab/data") or relative.startswith("reports/data"):
        return "stage_3_data_baseline"
    if relative.startswith("clinical_nlp_lab/ner") or relative.startswith("clinical_nlp_lab/training"):
        return "stage_4_entity_extraction"
    if relative.startswith("clinical_nlp_lab/assertions"):
        return "stage_5_clinical_context"
    if relative.startswith("clinical_nlp_lab/linking"):
        return "stage_6_entity_linking"
    if relative.startswith("clinical_nlp_lab/relations"):
        return "stage_7_relation_extraction"
    if relative.startswith("output") or relative.startswith("diagnostics") or relative.startswith("clinical_nlp_lab/pipeline"):
        return "stage_8_integration"
    if relative.endswith(".ipynb") or relative in {"README.md", "requirements.txt"}:
        return "stage_9_final_generation"
    return "project_maintenance"


def inference_required(relative: str) -> bool:
    return (
        relative.startswith("artifacts/")
        and not relative.endswith("build_report.json")
        and not relative.endswith("metadata.json")
    ) or relative.startswith("clinical_nlp_lab/")


def rebuild_instruction(relative: str) -> str:
    if relative.startswith("artifacts/icd10") or relative.startswith("artifacts/rxnorm"):
        return "Run tools/build_knowledge_bases.py from the project root."
    if relative.startswith("output") or relative.startswith("diagnostics"):
        return "Run tools/run_pipeline.py from the project root."
    if relative.endswith(".ipynb"):
        return "Run tools/build_notebook.py from the project root."
    if relative.startswith("stages/"):
        return "Regenerate from verified stage evidence and test reports."
    return "Restore from the project source tree or rerun the owning stage."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebuild ARTIFACT_MANIFEST.json")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--stage", type=int, required=True)
    parser.add_argument("--status", default="completed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    source_files: list[dict[str, Any]] = []
    for relative, role in SOURCE_ROLES.items():
        path = root / relative
        if path.exists():
            source_files.append(
                {
                    "path": relative,
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )

    artifacts: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = set(path.relative_to(root).parts)
        if parts & EXCLUDED_PARTS or relative in SOURCE_ROLES or relative == "ARTIFACT_MANIFEST.json":
            continue
        artifacts.append(
            {
                "path": relative,
                "artifact_type": artifact_type(relative),
                "created_by": created_by(relative),
                "used_by": ["project_pipeline"],
                "size_bytes": path.stat().st_size,
                "required_for_inference": inference_required(relative),
                "rebuild": rebuild_instruction(relative),
                "sha256": sha256_file(path),
            }
        )
    artifacts.append(
        {
            "path": "ARTIFACT_MANIFEST.json",
            "artifact_type": "artifact_manifest",
            "created_by": "project_maintenance",
            "used_by": ["all_stages"],
            "size_bytes": None,
            "required_for_inference": False,
            "rebuild": "Run tools/update_manifest.py with the latest completed stage.",
            "sha256": None,
            "checksum_note": "Self-checksum omitted to avoid recursive manifest dependency.",
        }
    )

    payload = {
        "manifest_version": "2.0.0",
        "project": "clinical-nlp-end-to-end-lab",
        "current_stage": args.stage,
        "stage_status": args.status,
        "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "source_files": source_files,
        "artifacts": artifacts,
        "artifact_count_excluding_self": len(artifacts) - 1,
        "inference_artifacts_created": sorted(
            item["path"] for item in artifacts if item.get("required_for_inference")
        ),
        "training_performed": False,
        "training_status_reason": "No annotated train/validation data was provided; supervised code is implemented but not executed.",
        "private_test_used_for_fitting": False,
    }
    destination = root / "ARTIFACT_MANIFEST.json"
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=False)
        stream.write("\n")
    print(json.dumps({"manifest": str(destination), "artifacts": len(artifacts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
