from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .dataset_quality import DatasetRecord
from .schema import ClinicalDocument, parse_entity


OFFICIAL_TO_INTERNAL_ENTITY_TYPES = {
    "CHẨN_ĐOÁN": "DISEASE",
    "THUỐC": "DRUG",
    "TRIỆU_CHỨNG": "SYMPTOM",
    "TÊN_XÉT_NGHIỆM": "LAB_NAME",
    "KẾT_QUẢ_XÉT_NGHIỆM": "LAB_RESULT",
}


def normalize_training_entity_type(entity_type: str) -> str:
    """Convert official submission labels to stable internal training labels."""
    return OFFICIAL_TO_INTERNAL_ENTITY_TYPES.get(entity_type, entity_type)


def _parse_training_entity(payload: dict[str, Any], raw_text: str):
    entity = parse_entity(payload, raw_text)
    entity.type = normalize_training_entity_type(entity.type)
    return entity


def natural_document_key(document_id: str) -> tuple[int, str]:
    match = re.fullmatch(r"(\d+)", document_id)
    return (int(match.group(1)), document_id) if match else (10**12, document_id)


def load_input_documents(source: str | Path) -> list[ClinicalDocument]:
    source_path = Path(source)
    documents: list[ClinicalDocument] = []
    if source_path.is_file() and source_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(source_path) as archive:
            for name in archive.namelist():
                if not name.lower().endswith(".txt") or name.endswith("/"):
                    continue
                raw = archive.read(name)
                try:
                    text = raw.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(f"{name} is not strict UTF-8: {exc}") from exc
                documents.append(ClinicalDocument(Path(name).stem, text))
    elif source_path.is_dir():
        for path in source_path.glob("*.txt"):
            documents.append(ClinicalDocument(path.stem, path.read_text(encoding="utf-8")))
    else:
        raise FileNotFoundError(f"Input source not found or unsupported: {source_path}")
    documents.sort(key=lambda document: natural_document_key(document.document_id))
    duplicate_ids = [key for key, count in Counter(document.document_id for document in documents).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"Duplicate document IDs: {duplicate_ids}")
    return documents


def _load_annotation_payload(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def load_annotated_documents(train_dir: str | Path) -> list[ClinicalDocument]:
    directory = Path(train_dir)
    if not directory.exists():
        return []

    documents: list[ClinicalDocument] = []

    # Canonical competition/training layout:
    # train/input/<document_id>.txt + train/gt/<document_id>.json
    paired_input_dir = directory / "input"
    paired_ground_truth_dir = directory / "gt"
    if paired_input_dir.is_dir() and paired_ground_truth_dir.is_dir():
        for text_path in sorted(paired_input_dir.glob("*.txt")):
            annotation_path = paired_ground_truth_dir / f"{text_path.stem}.json"
            if not annotation_path.exists():
                continue
            raw_text = text_path.read_text(encoding="utf-8")
            payload = _load_annotation_payload(annotation_path)
            if isinstance(payload, dict) and "entities" in payload:
                payload = payload["entities"]
            if not isinstance(payload, list):
                raise ValueError(f"Annotation must be a list: {annotation_path}")
            document = ClinicalDocument(text_path.stem, raw_text)
            document.entities = [_parse_training_entity(item, raw_text) for item in payload]
            documents.append(document)

    for text_path in sorted(directory.glob("*.txt")):
        annotation_path = text_path.with_suffix(".json")
        if not annotation_path.exists():
            continue
        raw_text = text_path.read_text(encoding="utf-8")
        payload = _load_annotation_payload(annotation_path)
        if isinstance(payload, dict) and "entities" in payload:
            payload = payload["entities"]
        if not isinstance(payload, list):
            raise ValueError(f"Annotation must be a list: {annotation_path}")
        document = ClinicalDocument(text_path.stem, raw_text)
        document.entities = [_parse_training_entity(item, raw_text) for item in payload]
        documents.append(document)

    for json_path in sorted(directory.glob("*.json")):
        if json_path.with_suffix(".txt").exists():
            continue
        payload = _load_annotation_payload(json_path)
        records = payload if isinstance(payload, list) else [payload]
        for record_index, record in enumerate(records):
            if not isinstance(record, dict) or "raw_text" not in record:
                continue
            document_id = str(record.get("document_id", f"{json_path.stem}_{record_index}"))
            raw_text = str(record["raw_text"])
            document = ClinicalDocument(document_id, raw_text)
            document.entities = [
                _parse_training_entity(item, raw_text) for item in record.get("entities", [])
            ]
            document.relations = list(record.get("relations", []))
            documents.append(document)

    documents.sort(key=lambda document: natural_document_key(document.document_id))
    return documents


def load_ner_training_documents(train_dir: str | Path) -> list[ClinicalDocument]:
    """Load only records explicitly eligible for NER training.

    The manifest is authoritative: organizer GT 1-100 is retained for audit
    but quarantined from training, while 101-200 and repaired synthetic data
    can be used when their hashes and eligibility flags are valid.
    """
    directory = Path(train_dir)
    documents = load_annotated_documents(directory)
    manifest_path = directory / "reports" / "dataset_manifest.jsonl"
    if not manifest_path.exists():
        return documents
    metadata = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            metadata[str(item["document_id"])] = item
    return [doc for doc in documents if bool(metadata.get(doc.document_id, {}).get("train_eligible", True))]


def dataset_fingerprint(documents: Iterable[ClinicalDocument]) -> str:
    digest = hashlib.sha256()
    for document in sorted(documents, key=lambda item: item.document_id):
        digest.update(document.document_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(document.raw_text.encode("utf-8"))
        digest.update(b"\0")
        for entity in sorted(document.entities, key=lambda item: (item.start, item.end, item.type)):
            digest.update(
                json.dumps(entity.to_diagnostic(), ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
    return digest.hexdigest()


def validate_documents(documents: Iterable[ClinicalDocument]) -> dict[str, Any]:
    document_list = list(documents)
    errors: list[dict[str, Any]] = []
    type_counts = Counter()
    assertion_counts = Counter()
    candidate_counts = Counter()
    overlap_count = 0
    duplicate_annotation_count = 0

    for document in document_list:
        seen: set[tuple[Any, ...]] = set()
        sorted_entities = sorted(document.entities, key=lambda item: (item.start, item.end))
        for index, entity in enumerate(sorted_entities):
            try:
                entity.validate_offset(document.raw_text)
            except ValueError as exc:
                errors.append(
                    {
                        "document_id": document.document_id,
                        "entity": entity.to_diagnostic(),
                        "error": str(exc),
                        "context": document.raw_text[max(0, entity.start - 30): min(len(document.raw_text), entity.end + 30)],
                    }
                )
            key = (entity.start, entity.end, entity.type, tuple(entity.candidates), tuple(entity.assertions))
            if key in seen:
                duplicate_annotation_count += 1
            seen.add(key)
            type_counts[entity.type] += 1
            assertion_counts.update(entity.assertions)
            candidate_counts.update(entity.candidates)
            if index and entity.start < sorted_entities[index - 1].end:
                overlap_count += 1

    return {
        "document_count": len(document_list),
        "entity_count": sum(len(document.entities) for document in document_list),
        "type_counts": dict(type_counts.most_common()),
        "assertion_counts": dict(assertion_counts.most_common()),
        "unique_candidate_count": len(candidate_counts),
        "overlap_count": overlap_count,
        "duplicate_annotation_count": duplicate_annotation_count,
        "errors": errors,
        "is_valid": not errors,
        "fingerprint": dataset_fingerprint(document_list),
    }


def document_train_validation_split(
    documents: Iterable[ClinicalDocument], validation_fraction: float = 0.2, seed: int = 42
) -> tuple[list[ClinicalDocument], list[ClinicalDocument]]:
    document_list = list(documents)
    if len(document_list) < 2:
        return document_list, []
    shuffled = document_list[:]
    random.Random(seed).shuffle(shuffled)
    validation_count = max(1, round(len(shuffled) * validation_fraction))
    validation = shuffled[:validation_count]
    train = shuffled[validation_count:]
    train_ids = {document.document_id for document in train}
    validation_ids = {document.document_id for document in validation}
    if train_ids & validation_ids:
        raise AssertionError("Document leakage detected between train and validation")
    return train, validation


def _connected_record_groups(records: Iterable[DatasetRecord]) -> list[list[str]]:
    record_list = list(records)
    parent = {record.document_id: record.document_id for record in record_list}

    def find(item: str) -> str:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    template_owner: dict[str, str] = {}
    surface_owner: dict[str, str] = {}
    for record in record_list:
        owner = template_owner.setdefault(record.template_group, record.document_id)
        union(record.document_id, owner)
        for surface in record.primary_surfaces:
            owner = surface_owner.setdefault(surface, record.document_id)
            union(record.document_id, owner)

    grouped: dict[str, list[str]] = {}
    for document_id in parent:
        grouped.setdefault(find(document_id), []).append(document_id)
    return [sorted(document_ids, key=natural_document_key) for document_ids in grouped.values()]


def grouped_train_validation_split(
    documents: Iterable[ClinicalDocument],
    records: Iterable[DatasetRecord],
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> tuple[list[ClinicalDocument], list[ClinicalDocument], dict[str, Any]]:
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must satisfy 0 <= value < 1")
    document_list = list(documents)
    record_list = list(records)
    document_ids = {document.document_id for document in document_list}
    record_ids = {record.document_id for record in record_list}
    if document_ids != record_ids:
        raise ValueError(
            "Document and manifest IDs differ: "
            f"missing_records={sorted(document_ids - record_ids, key=natural_document_key)}, "
            f"unknown_records={sorted(record_ids - document_ids, key=natural_document_key)}"
        )
    if len(document_list) < 2 or validation_fraction == 0:
        manifest = {
            "seed": seed,
            "validation_fraction": validation_fraction,
            "train_ids": sorted(document_ids, key=natural_document_key),
            "validation_ids": [],
        }
        return sorted(document_list, key=lambda item: natural_document_key(item.document_id)), [], manifest

    groups = _connected_record_groups(record_list)
    if len(groups) < 2:
        raise ValueError(
            "Cannot create a leakage-safe validation split because all documents belong to one connected group"
        )
    random.Random(seed).shuffle(groups)
    groups.sort(key=len)
    target = max(1, round(len(document_list) * validation_fraction))
    validation_ids: set[str] = set()
    for group in groups:
        if len(validation_ids) >= target:
            break
        remaining_after_selection = len(document_list) - len(validation_ids) - len(group)
        if remaining_after_selection <= 0:
            continue
        validation_ids.update(group)
    if not validation_ids:
        validation_ids.update(groups[0])
    train_ids = document_ids - validation_ids
    by_id = {document.document_id: document for document in document_list}
    train = [by_id[item] for item in sorted(train_ids, key=natural_document_key)]
    validation = [by_id[item] for item in sorted(validation_ids, key=natural_document_key)]
    manifest = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "train_ids": [document.document_id for document in train],
        "validation_ids": [document.document_id for document in validation],
        "connected_group_count": len(groups),
    }
    return train, validation, manifest


def audit_split_leakage(
    train_documents: Iterable[ClinicalDocument],
    validation_documents: Iterable[ClinicalDocument],
    records: Iterable[DatasetRecord],
) -> dict[str, list[str]]:
    train_ids = {document.document_id for document in train_documents}
    validation_ids = {document.document_id for document in validation_documents}
    record_by_id = {record.document_id: record for record in records}

    def values(document_ids: set[str], attribute: str) -> set[str]:
        result: set[str] = set()
        for document_id in document_ids:
            record = record_by_id[document_id]
            value = getattr(record, attribute)
            result.update(value if isinstance(value, tuple) else [value])
        return result

    return {
        "document_ids": sorted(train_ids & validation_ids, key=natural_document_key),
        "template_groups": sorted(
            values(train_ids, "template_group") & values(validation_ids, "template_group")
        ),
        "surface_groups": sorted(
            values(train_ids, "primary_surfaces") & values(validation_ids, "primary_surfaces")
        ),
    }


def describe_documents(documents: Iterable[ClinicalDocument]) -> dict[str, Any]:
    document_list = list(documents)
    lengths = [len(document.raw_text) for document in document_list]
    line_counts = [len(document.raw_text.splitlines()) for document in document_list]
    entity_counts = [len(document.entities) for document in document_list]

    def summary(values: list[int]) -> dict[str, float | int | None]:
        if not values:
            return {"min": None, "max": None, "mean": None}
        return {
            "min": min(values),
            "max": max(values),
            "mean": round(sum(values) / len(values), 2),
        }

    return {
        "document_count": len(document_list),
        "character_length": summary(lengths),
        "line_count": summary(line_counts),
        "entities_per_document": summary(entity_counts),
    }
