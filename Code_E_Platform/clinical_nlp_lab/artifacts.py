from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from .config import save_config


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return destination


def initialize_runtime_artifacts(config: dict[str, Any], artifact_dir: str | Path) -> list[Path]:
    root = Path(artifact_dir)
    root.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    created.append(save_config(config, root / "config.json"))
    created.append(
        write_json(
            root / "entity_type_mapping.json",
            {
                "status": "CONFIRMED_FROM_REPOSITORY_VALIDATOR",
                "drop_unmapped": True,
                "internal_to_official": {
                    "DISEASE": "CHẨN_ĐOÁN",
                    "DRUG": "THUỐC",
                    "LAB_RESULT": "KẾT_QUẢ_XÉT_NGHIỆM",
                    "LAB_NAME": "TÊN_XÉT_NGHIỆM",
                    "PATIENT_INFO": None,
                    "SYMPTOM": "TRIỆU_CHỨNG",
                },
                "evidence": "Confirmed from src/validation/submission.py in takumi612/AI-Race-Viettel.",
            },
        )
    )
    assertion_axes = {
        "polarity": ["AFFIRMED", "NEGATED"],
        "temporality": ["CURRENT", "HISTORICAL", "PLANNED", "RESOLVED"],
        "certainty": ["CONFIRMED", "SUSPECTED", "POSSIBLE", "CONDITIONAL"],
        "experiencer": ["PATIENT", "FAMILY", "OTHER"],
    }
    created.append(
        write_json(
            root / "assertion_mapping.json",
            {
                "status": "CONFIRMED_FROM_REPOSITORY_VALIDATOR",
                "axes": assertion_axes,
                "internal_to_official": {
                    f"{axis}:{value}": (
                        "isNegated"
                        if (axis, value) == ("polarity", "NEGATED")
                        else "isHistorical"
                        if (axis, value) == ("temporality", "HISTORICAL")
                        else "isFamily"
                        if (axis, value) == ("experiencer", "FAMILY")
                        else None
                    )
                    for axis, values in assertion_axes.items()
                    for value in values
                },
            },
        )
    )
    created.append(
        write_json(
            root / "relation_mapping.json",
            {
                "status": "DIAGNOSTIC_ONLY",
                "submission_enabled": False,
                "internal_relations": [
                    "DRUG_TREATS_CONDITION",
                    "CONDITION_HAS_SYMPTOM",
                    "LAB_ASSOCIATED_WITH_CONDITION",
                    "LAB_ASSOCIATED_WITH_SYMPTOM",
                ],
            },
        )
    )
    created.append(write_json(root / "thresholds.json", config["thresholds"]))
    created.append(
        write_json(
            root / "model_status.json",
            {
                "ner_model": {"trained": False, "reason": "No annotated training data"},
                "assertion_model": {"trained": False, "reason": "No annotated assertion labels"},
                "relation_model": {"trained": False, "reason": "No annotated relation labels"},
                "active_ner": "ontology_dictionary_plus_generic_rules",
                "active_assertion": "hybrid_rule_baseline",
                "active_relation": "rule_baseline_diagnostics_only",
            },
        )
    )
    return created


def inventory_files(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        if not path.exists() or not path.is_file():
            continue
        records.append(
            {
                "path": path.as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records
