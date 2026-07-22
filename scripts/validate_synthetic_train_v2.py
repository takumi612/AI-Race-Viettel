from __future__ import annotations

import gzip
import hashlib
import json
import re
import sqlite3
from collections import Counter
from pathlib import Path
from typing import Any


DISEASE = "CHẨN_ĐOÁN"
DRUG = "THUỐC"
SYMPTOM = "TRIỆU_CHỨNG"
LAB_NAME = "TÊN_XÉT_NGHIỆM"
LAB_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
ALLOWED_KEYS = {
    DISEASE: {"text", "type", "position", "assertions", "candidates"},
    DRUG: {"text", "type", "position", "assertions", "candidates"},
    SYMPTOM: {"text", "type", "position", "assertions"},
    LAB_NAME: {"text", "type", "position"},
    LAB_RESULT: {"text", "type", "position"},
}
ALLOWED_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}


def _candidate_ids(path: Path) -> set[str]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return {
            str(record["candidate_id"])
            for line in stream
            if line.strip()
            for record in [json.loads(line)]
        }


def _raw_candidate_ids(kb_path: Path) -> tuple[set[str], set[str]]:
    connection = sqlite3.connect(kb_path)
    icd = {
        re.sub(r"[†*]", "", str(code)).strip()
        for (code,) in connection.execute("select code from icd10")
    }
    rx = {str(code) for (code,) in connection.execute("select distinct rxcui from rxnorm")}
    connection.close()
    return icd, rx


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def validate(
    root: Path,
    kb_path: Path,
    artifact_root: Path | None = None,
    expected_documents: int = 2200,
) -> dict[str, Any]:
    root = Path(root)
    artifact_root = artifact_root or Path(__file__).resolve().parents[1] / "v2" / "artifacts"
    icd_artifact = _candidate_ids(artifact_root / "icd10" / "icd10_dictionary.jsonl.gz")
    rx_artifact = _candidate_ids(artifact_root / "rxnorm" / "rxnorm_dictionary.jsonl.gz")
    icd_raw, rx_raw = _raw_candidate_ids(Path(kb_path))
    manifest = _load_manifest(root / "reports" / "dataset_manifest.jsonl")
    manifest_by_id = {str(item.get("document_id")): item for item in manifest}

    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    assertions: Counter[str] = Counter()
    lengths: list[int] = []
    expected_ids = {str(document_id) for document_id in range(1, expected_documents + 1)}
    input_ids = {path.stem for path in (root / "input").glob("*.txt")}
    gt_ids = {path.stem for path in (root / "gt").glob("*.json")}

    for missing_id in sorted(expected_ids - input_ids, key=int):
        errors.append({"case_id": int(missing_id), "kind": "missing_input"})
    for missing_id in sorted(expected_ids - gt_ids, key=int):
        errors.append({"case_id": int(missing_id), "kind": "missing_gt"})
    for unexpected_id in sorted((input_ids | gt_ids) - expected_ids, key=lambda value: int(value) if value.isdigit() else 10**12):
        errors.append({"case_id": unexpected_id, "kind": "unexpected_document"})
    if set(manifest_by_id) != expected_ids:
        errors.append(
            {
                "kind": "manifest_document_ids",
                "missing": sorted(expected_ids - set(manifest_by_id), key=int),
                "unexpected": sorted(set(manifest_by_id) - expected_ids),
            }
        )

    for document_id in sorted(expected_ids & input_ids & gt_ids, key=int):
        case_id = int(document_id)
        input_path = root / "input" / f"{document_id}.txt"
        gt_path = root / "gt" / f"{document_id}.json"
        try:
            text = input_path.read_text(encoding="utf-8", errors="strict")
            payload = json.loads(gt_path.read_text(encoding="utf-8", errors="strict"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            errors.append({"case_id": case_id, "kind": "decode_or_json", "message": str(exc)})
            continue
        item_manifest = manifest_by_id.get(document_id, {})
        train_eligible = bool(item_manifest.get("train_eligible", True))
        linking_train_eligible = bool(item_manifest.get("linking_train_eligible", train_eligible))
        expected_sha = item_manifest.get("sha256")
        actual_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if expected_sha and expected_sha != actual_sha:
            errors.append({"case_id": case_id, "kind": "manifest_input_hash"})
        if not isinstance(payload, list):
            errors.append({"case_id": case_id, "kind": "top_level_schema"})
            continue
        lengths.append(len(payload))
        for index, entity in enumerate(payload):
            if not isinstance(entity, dict):
                errors.append({"case_id": case_id, "index": index, "kind": "entity_schema"})
                continue
            entity_type = entity.get("type")
            counts[str(entity_type)] += 1
            if set(entity) != ALLOWED_KEYS.get(entity_type, set()):
                errors.append({"case_id": case_id, "index": index, "kind": "schema"})
                continue
            position = entity.get("position")
            if not isinstance(position, list) or len(position) != 2:
                errors.append({"case_id": case_id, "index": index, "kind": "position_schema"})
                continue
            start, end = position
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start <= end <= len(text):
                errors.append({"case_id": case_id, "index": index, "kind": "position_range"})
                continue
            if text[start:end] != entity.get("text"):
                errors.append({"case_id": case_id, "index": index, "kind": "span"})
            entity_assertions = entity.get("assertions", [])
            if not isinstance(entity_assertions, list) or any(value not in ALLOWED_ASSERTIONS for value in entity_assertions):
                errors.append({"case_id": case_id, "index": index, "kind": "assertion"})
            assertions.update(entity_assertions)
            candidates = entity.get("candidates", [])
            if entity_type in {DISEASE, DRUG} and not candidates:
                target = errors if linking_train_eligible else warnings
                target.append({"case_id": case_id, "index": index, "kind": "missing_candidate"})
            if entity_type == DISEASE:
                for candidate in candidates:
                    canonical_candidate = re.sub(r"[†*]", "", str(candidate)).strip()
                    if canonical_candidate not in icd_raw:
                        errors.append({"case_id": case_id, "index": index, "kind": "unknown_icd_candidate", "candidate": candidate})
                    elif candidate != canonical_candidate:
                        target = errors if linking_train_eligible else warnings
                        target.append(
                            {
                                "case_id": case_id,
                                "index": index,
                                "kind": "noncanonical_icd_display_candidate",
                                "candidate": candidate,
                                "canonical_candidate": canonical_candidate,
                            }
                        )
                    elif canonical_candidate not in icd_artifact:
                        target = errors if linking_train_eligible else warnings
                        target.append({"case_id": case_id, "index": index, "kind": "icd_candidate_missing_artifact", "candidate": candidate})
            elif entity_type == DRUG:
                for candidate in candidates:
                    if candidate not in rx_raw:
                        errors.append({"case_id": case_id, "index": index, "kind": "unknown_rxnorm_candidate", "candidate": candidate})
                    elif candidate not in rx_artifact:
                        target = errors if linking_train_eligible else warnings
                        target.append({"case_id": case_id, "index": index, "kind": "rxnorm_candidate_missing_artifact", "candidate": candidate})

    train_eligible_ids = [
        document_id
        for document_id, item in manifest_by_id.items()
        if bool(item.get("train_eligible", True))
    ]
    report = {
        "documents": len(lengths),
        "expected_documents": expected_documents,
        "train_eligible_documents": len(train_eligible_ids),
        "train_excluded_documents": expected_documents - len(train_eligible_ids),
        "entities": sum(counts.values()),
        "entity_counts": dict(counts),
        "assertion_counts": dict(assertions),
        "entities_per_document": {
            "min": min(lengths) if lengths else None,
            "max": max(lengths) if lengths else None,
            "mean": round(sum(lengths) / len(lengths), 3) if lengths else None,
        },
        "artifact_candidate_counts": {"icd10": len(icd_artifact), "rxnorm": len(rx_artifact)},
        "errors": errors,
        "warnings": warnings,
        "input_1_sha256": (
            hashlib.sha256((root / "input" / "1.txt").read_bytes()).hexdigest()
            if (root / "input" / "1.txt").is_file()
            else None
        ),
        "input_201_sha256": (
            hashlib.sha256((root / "input" / "201.txt").read_bytes()).hexdigest()
            if (root / "input" / "201.txt").is_file()
            else None
        ),
    }
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "data_v2" / "Training_data" / "synthetic_train_v2"
    report = validate(
        dataset,
        project_root / "data" / "kb" / "metadata.db",
        project_root / "v2" / "artifacts",
    )
    report_dir = dataset / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "documents": report["documents"],
                "entities": report["entities"],
                "errors": len(report["errors"]),
                "warnings": len(report["warnings"]),
            },
            ensure_ascii=False,
        )
    )
