"""Deterministic, group-safe split assignment for training data."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable, Sequence

from src.training.contracts import CanonicalRecord


@dataclass(frozen=True, order=True, slots=True)
class SplitAssignment:
    record_id: str
    split: str
    fold: int | None = None


@dataclass(frozen=True, slots=True)
class _RecordGroup:
    group_id: str
    record_ids: tuple[str, ...]
    signature: str

    @property
    def size(self) -> int:
        return len(self.record_ids)


def _stable_rank(seed: int, value: str) -> str:
    return sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _record_signature(record: CanonicalRecord) -> str:
    family = record.metadata.get("family", "unknown")
    families = (
        tuple(sorted(str(value) for value in family))
        if isinstance(family, (list, tuple, set, frozenset))
        else (str(family),)
    )
    entity_types = tuple(sorted({entity.entity_type for entity in record.entities}))
    assertion_sets = tuple(
        sorted(
            {
                "+".join(entity.assertions) if entity.assertions else "none"
                for entity in record.entities
            }
        )
    )
    return repr((families, entity_types, assertion_sets))


def _interleave_buckets(
    buckets: dict[str, list[_RecordGroup]],
    seed: int,
) -> list[_RecordGroup]:
    signature_order = sorted(
        buckets,
        key=lambda signature: _stable_rank(seed, f"signature:{signature}"),
    )
    for signature, groups in buckets.items():
        groups.sort(key=lambda group: _stable_rank(seed, group.group_id))

    interleaved: list[_RecordGroup] = []
    index = 0
    while True:
        added = False
        for signature in signature_order:
            groups = buckets[signature]
            if index < len(groups):
                interleaved.append(groups[index])
                added = True
        if not added:
            return interleaved
        index += 1


def _build_groups(
    records: Sequence[CanonicalRecord],
    seed: int,
) -> tuple[list[_RecordGroup], dict[str, CanonicalRecord]]:
    by_id: dict[str, CanonicalRecord] = {}
    grouped_records: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        if record.record_id in by_id:
            raise ValueError(f"duplicate record ID in split input: {record.record_id}")
        by_id[record.record_id] = record
        grouped_records[record.split_group].append(record)

    buckets: dict[str, list[_RecordGroup]] = defaultdict(list)
    for group_id, group_records in grouped_records.items():
        signatures = tuple(sorted({_record_signature(record) for record in group_records}))
        group = _RecordGroup(
            group_id=group_id,
            record_ids=tuple(sorted(record.record_id for record in group_records)),
            signature=repr(signatures),
        )
        buckets[group.signature].append(group)
    return _interleave_buckets(buckets, seed), by_id


def _select_exact_groups(
    groups: Sequence[_RecordGroup],
    target_size: int,
) -> set[str]:
    paths: dict[int, tuple[str, ...]] = {0: ()}
    for group in groups:
        additions: dict[int, tuple[str, ...]] = {}
        for current_size, selected in sorted(paths.items(), reverse=True):
            next_size = current_size + group.size
            if (
                next_size <= target_size
                and next_size not in paths
                and next_size not in additions
            ):
                additions[next_size] = selected + (group.group_id,)
        paths.update(additions)
        if target_size in paths:
            return set(paths[target_size])
    raise ValueError(
        "cannot produce exact validation_size without splitting a split_group"
    )


def build_synthetic_split(
    records: Sequence[CanonicalRecord],
    validation_size: int,
    seed: int,
) -> tuple[SplitAssignment, ...]:
    if (
        isinstance(validation_size, bool)
        or not isinstance(validation_size, int)
        or validation_size <= 0
        or validation_size >= len(records)
    ):
        raise ValueError("validation_size must be between 1 and record_count - 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    groups, by_id = _build_groups(records, seed)
    validation_groups = _select_exact_groups(groups, validation_size)
    assignments = tuple(
        SplitAssignment(
            record_id=record_id,
            split=(
                "synthetic_validation"
                if record.split_group in validation_groups
                else "synthetic_train"
            ),
        )
        for record_id, record in sorted(by_id.items())
    )
    assert_no_split_leakage(records, assignments)
    return assignments


def _interleave_records(
    records: Sequence[CanonicalRecord],
    seed: int,
) -> list[CanonicalRecord]:
    buckets: dict[str, list[CanonicalRecord]] = defaultdict(list)
    for record in records:
        buckets[_record_signature(record)].append(record)
    signatures = sorted(
        buckets,
        key=lambda signature: _stable_rank(seed, f"trusted-signature:{signature}"),
    )
    for signature, items in buckets.items():
        items.sort(key=lambda record: _stable_rank(seed, record.record_id))

    ordered: list[CanonicalRecord] = []
    index = 0
    while True:
        added = False
        for signature in signatures:
            items = buckets[signature]
            if index < len(items):
                ordered.append(items[index])
                added = True
        if not added:
            return ordered
        index += 1


def build_trusted_folds(
    records: Sequence[CanonicalRecord],
    folds: int,
    seed: int,
) -> tuple[SplitAssignment, ...]:
    if isinstance(folds, bool) or not isinstance(folds, int) or folds < 2:
        raise ValueError("folds must be an integer greater than one")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    by_numeric_id: dict[int, CanonicalRecord] = {}
    for record in records:
        if not record.record_id.isdigit():
            continue
        numeric_id = int(record.record_id)
        if 101 <= numeric_id <= 180:
            if numeric_id in by_numeric_id:
                raise ValueError(f"duplicate trusted numeric ID: {numeric_id}")
            by_numeric_id[numeric_id] = record

    expected_ids = set(range(101, 181))
    if set(by_numeric_id) != expected_ids:
        missing = sorted(expected_ids - set(by_numeric_id))
        raise ValueError(
            f"trusted fold input must contain every ID 101-180; missing: {missing}"
        )
    if len(by_numeric_id) % folds:
        raise ValueError("trusted record count must be divisible by folds")

    ordered = _interleave_records(tuple(by_numeric_id.values()), seed)
    assignments = tuple(
        sorted(
            (
                SplitAssignment(
                    record_id=record.record_id,
                    split="trusted_fold",
                    fold=index % folds,
                )
                for index, record in enumerate(ordered)
            )
        )
    )
    fold_counts = {
        fold: sum(assignment.fold == fold for assignment in assignments)
        for fold in range(folds)
    }
    expected_per_fold = len(ordered) // folds
    if set(fold_counts.values()) != {expected_per_fold}:
        raise ValueError(f"trusted folds are not balanced: {fold_counts}")
    assert_no_split_leakage(tuple(by_numeric_id.values()), assignments)
    return assignments


def assert_no_split_leakage(
    records: Sequence[CanonicalRecord],
    assignments: Sequence[SplitAssignment],
) -> None:
    records_by_id: dict[str, CanonicalRecord] = {}
    for record in records:
        if record.record_id in records_by_id:
            raise ValueError(f"duplicate record ID: {record.record_id}")
        records_by_id[record.record_id] = record

    assignments_by_id: dict[str, SplitAssignment] = {}
    for assignment in assignments:
        if assignment.record_id in assignments_by_id:
            raise ValueError(f"duplicate split assignment: {assignment.record_id}")
        assignments_by_id[assignment.record_id] = assignment

    if set(records_by_id) != set(assignments_by_id):
        missing = sorted(set(records_by_id) - set(assignments_by_id))
        unexpected = sorted(set(assignments_by_id) - set(records_by_id))
        raise ValueError(
            "assignment coverage mismatch; "
            f"missing={missing}, unexpected={unexpected}"
        )

    group_partitions: dict[str, set[tuple[str, int | None]]] = defaultdict(set)
    hash_partitions: dict[str, set[tuple[str, int | None]]] = defaultdict(set)
    for record_id, record in records_by_id.items():
        assignment = assignments_by_id[record_id]
        partition = (assignment.split, assignment.fold)
        group_partitions[record.split_group].add(partition)
        hash_partitions[record.sha256].add(partition)

    leaking_groups = sorted(
        group_id
        for group_id, partitions in group_partitions.items()
        if len(partitions) > 1
    )
    if leaking_groups:
        raise ValueError(f"split_group leakage detected: {leaking_groups}")

    leaking_hashes = sorted(
        content_hash
        for content_hash, partitions in hash_partitions.items()
        if len(partitions) > 1
    )
    if leaking_hashes:
        raise ValueError(f"content hash leakage detected: {leaking_hashes}")
