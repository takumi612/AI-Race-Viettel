from collections import Counter

import pytest

from src.training.contracts import CanonicalRecord
from src.training.splits import (
    SplitAssignment,
    assert_no_split_leakage,
    build_synthetic_split,
    build_trusted_folds,
)


def _record(
    record_id: str,
    *,
    split_group: str | None = None,
    family: str = "default",
    text: str | None = None,
) -> CanonicalRecord:
    return CanonicalRecord.create(
        record_id=record_id,
        source="fixture",
        trust_tier="fixture",
        text=text or f"Nội dung {record_id}",
        entity_mappings=[],
        split_group=split_group or record_id,
        metadata={"family": family},
    )


def test_synthetic_split_is_exact_deterministic_group_safe_and_stratified():
    records = tuple(
        _record(
            f"{index:04d}",
            split_group=f"group-{index // 2}",
            family="rare" if index < 6 else "rich",
        )
        for index in range(10)
    )

    first = build_synthetic_split(records, validation_size=4, seed=7)
    second = build_synthetic_split(tuple(reversed(records)), validation_size=4, seed=7)

    assert first == second
    assert sum(item.split == "synthetic_validation" for item in first) == 4
    assert_no_split_leakage(records, first)
    validation_ids = {
        item.record_id for item in first if item.split == "synthetic_validation"
    }
    validation_families = {
        record.metadata["family"]
        for record in records
        if record.record_id in validation_ids
    }
    assert validation_families == {"rare", "rich"}


def test_synthetic_split_rejects_impossible_exact_group_size():
    records = tuple(
        _record(str(index), split_group=f"group-{index // 2}")
        for index in range(4)
    )

    with pytest.raises(ValueError, match="exact validation_size"):
        build_synthetic_split(records, validation_size=3, seed=7)


def test_trusted_folds_exclude_pseudo_and_holdout_and_are_balanced():
    records = tuple(_record(str(index)) for index in range(1, 201))

    first = build_trusted_folds(records, folds=5, seed=11)
    second = build_trusted_folds(tuple(reversed(records)), folds=5, seed=11)

    assert first == second
    assert {item.record_id for item in first} == {
        str(index) for index in range(101, 181)
    }
    assert Counter(item.fold for item in first) == {
        0: 16,
        1: 16,
        2: 16,
        3: 16,
        4: 16,
    }
    assert all(item.split == "trusted_fold" for item in first)


def test_trusted_folds_require_all_80_records():
    records = tuple(_record(str(index)) for index in range(101, 180))

    with pytest.raises(ValueError, match="101-180"):
        build_trusted_folds(records, folds=5, seed=11)


def test_leakage_gate_rejects_same_content_across_splits():
    records = (
        _record("a", split_group="a", text="identical"),
        _record("b", split_group="b", text="identical"),
    )
    assignments = (
        SplitAssignment("a", "synthetic_train"),
        SplitAssignment("b", "synthetic_validation"),
    )

    with pytest.raises(ValueError, match="content hash"):
        assert_no_split_leakage(records, assignments)


def test_leakage_gate_requires_one_assignment_per_record():
    records = (_record("a"), _record("b"))

    with pytest.raises(ValueError, match="assignment coverage"):
        assert_no_split_leakage(
            records,
            (SplitAssignment("a", "synthetic_train"),),
        )
