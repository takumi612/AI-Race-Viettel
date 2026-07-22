from __future__ import annotations

import json
import shutil
import hashlib
from pathlib import Path

try:
    from scripts.generate_synthetic_train_v2 import generate_dataset
    from scripts.repair_first_100_gt import repair_first_100_ground_truth
except ModuleNotFoundError:
    from generate_synthetic_train_v2 import generate_dataset
    from repair_first_100_gt import repair_first_100_ground_truth


LAB_TYPES = {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
SYMPTOM_TYPE = "TRIỆU_CHỨNG"


def _canonicalize_first_200(source: Path, destination: Path) -> dict[str, int]:
    destination.mkdir(parents=True, exist_ok=True)
    stats = {"copied": 0, "lab_assertions_removed": 0, "affirmative_negations_fixed": 0}
    for case_id in range(1, 201):
        text = (source / "input" / f"{case_id}.txt").read_text(encoding="utf-8")
        annotations = json.loads((source / "gt" / f"{case_id}.json").read_text(encoding="utf-8"))
        normalized = []
        for entity in annotations:
            item = dict(entity)
            if item["type"] in LAB_TYPES and "assertions" in item:
                item.pop("assertions", None)
                stats["lab_assertions_removed"] += 1
            if case_id <= 100 and item["type"] == SYMPTOM_TYPE and "isNegated" in item.get("assertions", []):
                start = item["position"][0]
                context = text[max(0, start - 140) : start].lower()
                affirmative = any(
                    marker in context
                    for marker in ("nhập viện khám vì triệu chứng", "cảm thấy ", "kèm theo cảm giác ")
                ) and not any(marker in context[-70:] for marker in ("không ", "không có ", "không hề có "))
                if affirmative:
                    item["assertions"] = [x for x in item["assertions"] if x != "isNegated"]
                    stats["affirmative_negations_fixed"] += 1
            normalized.append(item)
        (destination / "input" / f"{case_id}.txt").write_text(text, encoding="utf-8")
        (destination / "gt" / f"{case_id}.json").write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        stats["copied"] += 1
    return stats


def _source_manifest(destination: Path) -> list[dict[str, object]]:
    manifest: list[dict[str, object]] = []
    for case_id in range(1, 201):
        text = (destination / "input" / f"{case_id}.txt").read_text(encoding="utf-8")
        entities = json.loads((destination / "gt" / f"{case_id}.json").read_text(encoding="utf-8"))
        source_bucket = "reconstructed" if case_id <= 100 else "organizer_gt"
        manifest.append(
            {
                "document_id": str(case_id),
                "source_bucket": source_bucket,
                "genre": "organizer_source",
                "scenario": "organizer_source",
                "template_group": f"organizer_source:{source_bucket}",
                "long_tail": False,
                "train_eligible": case_id > 100,
                "linking_train_eligible": False,
                "train_exclusion_reason": (
                    None
                    if case_id > 100
                    else "Reconstructed labels are quarantined after audit found systemic clinical contradictions."
                ),
                "primary_candidates": sorted(
                    {
                        candidate
                        for entity in entities
                        if entity["type"] in {"CHẨN_ĐOÁN", "THUỐC"}
                        for candidate in entity.get("candidates", [])
                    }
                ),
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            }
        )
    return manifest


def build_v2(
    source: Path,
    destination: Path,
    kb_path: Path,
    seed: int = 20260722,
    synthetic_count: int = 2000,
) -> dict[str, int]:
    if destination.exists():
        shutil.rmtree(destination)
    (destination / "input").mkdir(parents=True)
    (destination / "gt").mkdir(parents=True)
    first = _canonicalize_first_200(source, destination)
    repaired = repair_first_100_ground_truth(destination, kb_path)
    source_manifest = _source_manifest(destination)
    generated = generate_dataset(
        kb_path,
        destination,
        count=synthetic_count,
        seed=seed,
        start_id=201,
    )
    manifest_path = destination / "reports" / "dataset_manifest.jsonl"
    generated_manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest_path.write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in source_manifest + generated_manifest
        ),
        encoding="utf-8",
    )
    return {
        **first,
        "synthetic_generated": generated["count"],
        "total": 200 + generated["count"],
        "longtail_generated": generated["longtail_count"],
        "first100_candidate_repaired": repaired["candidate_entities_repaired"],
        "first100_historical_removed": repaired["historical_assertions_removed"],
        "train_excluded_count": len(repaired["train_excluded_ids"]),
    }


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    print(
        build_v2(
            root / "data_v2" / "Training_data" / "synthetic_train_v1",
            root / "data_v2" / "Training_data" / "synthetic_train_v2",
            root / "data" / "kb" / "metadata.db",
        )
    )
