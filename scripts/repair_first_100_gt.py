from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


DISEASE_TYPE = "CHẨN_ĐOÁN"
DRUG_TYPE = "THUỐC"


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold().replace("đ", "d")
    normalized = "".join(
        character
        for character in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(character) != "Mn"
    )
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def _candidate_maps(kb_path: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    connection = sqlite3.connect(kb_path)
    disease: dict[str, set[str]] = defaultdict(set)
    for code, name_vi, name_en in connection.execute("select code, name_vi, name_en from icd10"):
        canonical_code = re.sub(r"[†*]", "", str(code)).strip()
        for name in (name_vi, name_en):
            normalized = _normalize(str(name or ""))
            if normalized:
                disease[normalized].add(canonical_code)
    drug: dict[str, set[str]] = defaultdict(set)
    for rxcui, name in connection.execute("select rxcui, name from rxnorm"):
        normalized = _normalize(str(name or ""))
        if normalized:
            drug[normalized].add(str(rxcui))
    connection.close()
    return (
        {name: sorted(codes) for name, codes in disease.items()},
        {name: sorted(codes, key=lambda value: (len(value), value)) for name, codes in drug.items()},
    )


def _is_current_home_medication(raw_text: str, start: int, end: int) -> bool:
    sentence_start = max(raw_text.rfind(".", 0, start), raw_text.rfind("\n", 0, start)) + 1
    period_end = raw_text.find(".", end)
    line_end = raw_text.find("\n", end)
    candidates = [value for value in (period_end, line_end) if value != -1]
    sentence_end = min(candidates) if candidates else len(raw_text)
    sentence = _normalize(raw_text[sentence_start:sentence_end])
    return "dang su dung thuoc" in sentence and "tai nha" in sentence


def repair_first_100_ground_truth(
    dataset_root: str | Path,
    kb_path: str | Path,
    document_ids: Iterable[int] = range(1, 101),
) -> dict[str, Any]:
    root = Path(dataset_root)
    ids = [int(document_id) for document_id in document_ids]
    if any(document_id < 1 or document_id > 100 for document_id in ids):
        raise ValueError("Only reconstructed ground truth IDs 1-100 may be repaired")
    disease_map, drug_map = _candidate_maps(Path(kb_path))
    changes: list[dict[str, Any]] = []
    candidate_entities_repaired = 0
    candidate_entities_canonicalized = 0
    historical_assertions_removed = 0

    for document_id in ids:
        input_path = root / "input" / f"{document_id}.txt"
        gt_path = root / "gt" / f"{document_id}.json"
        raw_text = input_path.read_text(encoding="utf-8")
        before_bytes = gt_path.read_bytes()
        annotations = json.loads(before_bytes.decode("utf-8"))
        for entity_index, entity in enumerate(annotations):
            start, end = map(int, entity["position"])
            if raw_text[start:end] != entity["text"]:
                raise ValueError(
                    f"Offset mismatch before repair: document={document_id}, entity={entity_index}"
                )
            if entity["type"] == DISEASE_TYPE and entity.get("candidates"):
                canonical_candidates = list(
                    dict.fromkeys(
                        re.sub(r"[†*]", "", str(candidate)).strip()
                        for candidate in entity["candidates"]
                    )
                )
                if canonical_candidates != entity["candidates"]:
                    entity["candidates"] = canonical_candidates
                    candidate_entities_canonicalized += 1
                    changes.append(
                        {
                            "document_id": str(document_id),
                            "entity_index": entity_index,
                            "action": "canonicalize_icd_display_markers",
                            "candidates": canonical_candidates,
                        }
                    )
            if entity["type"] in {DISEASE_TYPE, DRUG_TYPE} and not entity.get("candidates"):
                candidate_map = disease_map if entity["type"] == DISEASE_TYPE else drug_map
                candidates = candidate_map.get(_normalize(entity["text"]), [])
                if candidates:
                    entity["candidates"] = candidates
                    candidate_entities_repaired += 1
                    changes.append(
                        {
                            "document_id": str(document_id),
                            "entity_index": entity_index,
                            "action": "fill_exact_candidates",
                            "candidates": candidates,
                        }
                    )
            if (
                entity["type"] == DRUG_TYPE
                and "isHistorical" in entity.get("assertions", [])
                and _is_current_home_medication(raw_text, start, end)
            ):
                entity["assertions"] = [
                    assertion
                    for assertion in entity.get("assertions", [])
                    if assertion != "isHistorical"
                ]
                historical_assertions_removed += 1
                changes.append(
                    {
                        "document_id": str(document_id),
                        "entity_index": entity_index,
                        "action": "remove_false_historical",
                    }
                )
        gt_path.write_text(
            json.dumps(annotations, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        changes.append(
            {
                "document_id": str(document_id),
                "action": "document_hashes",
                "gt_sha256_before": hashlib.sha256(before_bytes).hexdigest(),
                "gt_sha256_after": hashlib.sha256(gt_path.read_bytes()).hexdigest(),
            }
        )

    report = {
        "documents_processed": len(ids),
        "candidate_entities_repaired": candidate_entities_repaired,
        "candidate_entities_canonicalized": candidate_entities_canonicalized,
        "historical_assertions_removed": historical_assertions_removed,
        "train_excluded_ids": [str(document_id) for document_id in ids],
        "train_exclusion_reason": (
            "Organizer inputs with reconstructed GT remain quarantined because the source text contains "
            "systemic clinical contradictions that cannot be corrected without rewriting organizer input."
        ),
        "changes": changes,
    }
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "first100_repair_log.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "data_v2" / "Training_data" / "synthetic_train_v2"
    print(
        json.dumps(
            repair_first_100_ground_truth(dataset, project_root / "data" / "kb" / "metadata.db"),
            ensure_ascii=False,
        )
    )
