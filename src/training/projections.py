"""Lossless and task-specific projections from canonical training records."""

from __future__ import annotations

from typing import Mapping, Sequence

from src.training.contracts import CanonicalRecord
from src.training.splits import SplitAssignment


def _ordered_records(
    records: Sequence[CanonicalRecord],
) -> tuple[CanonicalRecord, ...]:
    by_id: dict[str, CanonicalRecord] = {}
    for record in records:
        if record.record_id in by_id:
            raise ValueError(f"duplicate projection record ID: {record.record_id}")
        by_id[record.record_id] = record
    return tuple(by_id[record_id] for record_id in sorted(by_id))


def _assignment_fields(
    record: CanonicalRecord,
    assignments: Mapping[str, SplitAssignment],
) -> dict[str, object]:
    assignment = assignments.get(record.record_id)
    if assignment is None:
        raise ValueError(f"missing split assignment for {record.record_id}")
    if assignment.record_id != record.record_id:
        raise ValueError(
            f"split assignment identity mismatch for {record.record_id}"
        )
    if not isinstance(assignment.split, str) or not assignment.split.strip():
        raise ValueError(f"invalid split assignment for {record.record_id}")

    fields: dict[str, object] = {"split": assignment.split}
    if assignment.fold is not None:
        if isinstance(assignment.fold, bool) or not isinstance(assignment.fold, int):
            raise ValueError(f"invalid fold assignment for {record.record_id}")
        fields["fold"] = assignment.fold
    return fields


def project_ner_records(
    records: Sequence[CanonicalRecord],
    assignments: Mapping[str, SplitAssignment],
) -> tuple[dict[str, object], ...]:
    projected: list[dict[str, object]] = []
    for record in _ordered_records(records):
        item: dict[str, object] = {
            "record_id": record.record_id,
            "text": record.text,
            "entities": [entity.to_mapping() for entity in record.entities],
        }
        item.update(_assignment_fields(record, assignments))
        projected.append(item)
    return tuple(projected)


def project_embedding_seeds(
    records: Sequence[CanonicalRecord],
    assignments: Mapping[str, SplitAssignment],
) -> tuple[dict[str, object], ...]:
    projected: list[dict[str, object]] = []
    for record in _ordered_records(records):
        assignment_fields = _assignment_fields(record, assignments)
        for entity_index, entity in enumerate(record.entities):
            if entity.entity_type not in {"CHẨN_ĐOÁN", "THUỐC"} or not entity.codes:
                continue
            item: dict[str, object] = {
                "example_id": f"{record.record_id}:{entity_index}",
                "record_id": record.record_id,
                "query": entity.text,
                "context": record.text,
                "entity_type": entity.entity_type,
                "positive_codes": list(entity.codes),
            }
            item.update(assignment_fields)
            projected.append(item)
    return tuple(projected)


def project_reranker_seeds(
    records: Sequence[CanonicalRecord],
    assignments: Mapping[str, SplitAssignment],
) -> tuple[dict[str, object], ...]:
    projected: list[dict[str, object]] = []
    for record in _ordered_records(records):
        assignment_fields = _assignment_fields(record, assignments)
        for entity_index, entity in enumerate(record.entities):
            if entity.entity_type not in {"CHẨN_ĐOÁN", "THUỐC"} or not entity.codes:
                continue
            item: dict[str, object] = {
                "example_id": f"{record.record_id}:{entity_index}",
                "record_id": record.record_id,
                "context": record.text,
                "entity_text": entity.text,
                "entity_type": entity.entity_type,
                "assertions": list(entity.assertions),
                "ground_truth_codes": list(entity.codes),
            }
            item.update(assignment_fields)
            projected.append(item)
    return tuple(projected)
