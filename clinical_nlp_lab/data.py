from __future__ import annotations

import hashlib
import json
import random
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from .schema import ClinicalDocument, parse_entity


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
        document.entities = [parse_entity(item, raw_text) for item in payload]
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
            document.entities = [parse_entity(item, raw_text) for item in record.get("entities", [])]
            document.relations = list(record.get("relations", []))
            documents.append(document)

    documents.sort(key=lambda document: natural_document_key(document.document_id))
    return documents


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
