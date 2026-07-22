from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


DISEASE = "CHẨN_ĐOÁN"
DRUG = "THUỐC"
SYMPTOM = "TRIỆU_CHỨNG"


def _summary(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "max": None, "mean": None, "median": None}
    return {
        "min": min(values),
        "max": max(values),
        "mean": round(sum(values) / len(values), 3),
        "median": round(float(statistics.median(values)), 3),
    }


def _top_share(counter: Counter[str], top_n: int = 20) -> float:
    total = sum(counter.values())
    if not total:
        return 0.0
    return round(sum(count for _item, count in counter.most_common(top_n)) / total, 6)


def _canonical_signature(text: str, annotations: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    cursor = 0
    for entity in sorted(annotations, key=lambda item: item["position"]):
        start, end = entity["position"]
        parts.append(text[cursor:start])
        parts.append(f"<{entity['type']}>")
        cursor = end
    parts.append(text[cursor:])
    normalized = re.sub(r"\d+", "<N>", "".join(parts).casefold())
    return re.sub(r"\s+", " ", normalized).strip()


def analyze_dataset(root: str | Path) -> dict[str, Any]:
    dataset = Path(root)
    manifest_path = dataset / "reports" / "dataset_manifest.jsonl"
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if manifest_path.is_file() else []
    manifest_by_id = {str(item["document_id"]): item for item in manifest}
    all_ids = sorted(
        {path.stem for path in (dataset / "input").glob("*.txt")},
        key=lambda value: int(value) if value.isdigit() else 10**12,
    )
    generated_ids = [
        document_id
        for document_id in all_ids
        if not manifest_by_id or manifest_by_id.get(document_id, {}).get("source_bucket") == "synthetic"
    ]

    word_counts: list[int] = []
    entity_counts: list[int] = []
    entity_type_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()
    icd_candidates: Counter[str] = Counter()
    rx_candidates: Counter[str] = Counter()
    documents: list[str] = []
    signatures: set[str] = set()
    duplicate_positive_reason_symptoms = 0
    adult_male_maternity = 0
    longtail_documents = 0
    longtail_neutral = 0

    for document_id in generated_ids:
        text = (dataset / "input" / f"{document_id}.txt").read_text(encoding="utf-8")
        annotations = json.loads((dataset / "gt" / f"{document_id}.json").read_text(encoding="utf-8"))
        documents.append(text)
        word_counts.append(len(text.split()))
        entity_counts.append(len(annotations))
        signatures.add(_canonical_signature(text, annotations))
        positive_symptoms = [
            entity["text"].casefold()
            for entity in annotations
            if entity["type"] == SYMPTOM and not entity.get("assertions")
        ]
        if len(positive_symptoms) >= 2 and positive_symptoms[0] == positive_symptoms[1]:
            duplicate_positive_reason_symptoms += 1
        for entity in annotations:
            entity_type_counts[entity["type"]] += 1
            assertion_counts.update(entity.get("assertions", []))
            target = icd_candidates if entity["type"] == DISEASE else rx_candidates if entity["type"] == DRUG else None
            if target is not None:
                target.update(map(str, entity.get("candidates", [])))
        if text.startswith("HỒ SƠ THEO DÕI THAI SẢN/NHI KHOA"):
            match = re.search(r"Bệnh nhân (nam|nữ) (\d+) tuổi", text)
            if match and match.group(1) == "nam" and int(match.group(2)) > 15:
                adult_male_maternity += 1
        item_manifest = manifest_by_id.get(document_id, {})
        if bool(item_manifest.get("long_tail", False)):
            longtail_documents += 1
            if "Đối chiếu mã hóa" in text and "không dùng để suy diễn chỉ định" in text:
                longtail_neutral += 1

    common_lines = set(documents[0].splitlines()) if documents else set()
    for document in documents[1:]:
        common_lines &= set(document.splitlines())
    fixed_characters = sum(len(line) for line in common_lines if line.strip())
    mean_characters = sum(map(len, documents)) / len(documents) if documents else 0.0
    fixed_share = round(fixed_characters / mean_characters, 6) if mean_characters else 0.0
    source_counts = Counter(
        str(item.get("source_bucket", "unknown"))
        for item in manifest
    )
    report = {
        "total_documents": len(all_ids),
        "documents": len(generated_ids),
        "source_counts": dict(source_counts),
        "train_eligible_documents": sum(bool(item.get("train_eligible", True)) for item in manifest),
        "words_per_document": _summary(word_counts),
        "entities_per_document": _summary(entity_counts),
        "entity_counts": dict(entity_type_counts),
        "assertion_counts": dict(assertion_counts),
        "fixed_line_character_share": fixed_share,
        "canonical_signature_count": len(signatures),
        "canonical_signature_ratio": round(len(signatures) / len(generated_ids), 6) if generated_ids else 0.0,
        "duplicate_positive_reason_symptoms": duplicate_positive_reason_symptoms,
        "adult_male_maternity_pediatric_documents": adult_male_maternity,
        "longtail_documents": longtail_documents,
        "longtail_neutral_context_documents": longtail_neutral,
        "candidate_coverage": {
            "unique_icd10": len(icd_candidates),
            "unique_rxnorm": len(rx_candidates),
            "icd10_top20_mention_share": _top_share(icd_candidates),
            "rxnorm_top20_mention_share": _top_share(rx_candidates),
        },
    }
    report["quality_gate_passed"] = bool(
        generated_ids
        and 320 <= float(report["words_per_document"]["mean"] or 0) <= 500
        and fixed_share < 0.25
        and duplicate_positive_reason_symptoms == 0
        and adult_male_maternity == 0
        and longtail_neutral == longtail_documents
        and float(report["canonical_signature_ratio"]) >= 0.5
    )
    return report


def console_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    dataset = project_root / "data_v2" / "Training_data" / "synthetic_train_v2"
    report = analyze_dataset(dataset)
    report_path = dataset / "reports" / "data_analysis_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(console_json(report))
