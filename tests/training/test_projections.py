import pytest

from src.training.contracts import CanonicalRecord
from src.training.projections import (
    project_embedding_seeds,
    project_ner_records,
    project_reranker_seeds,
)
from src.training.splits import SplitAssignment


def _record() -> CanonicalRecord:
    return CanonicalRecord.create(
        record_id="synthetic-0001",
        source="synthetic",
        trust_tier="synthetic_validated",
        text="Sốt, dùng metformin và tăng huyết áp.",
        entity_mappings=[
            {
                "text": "Sốt",
                "type": "TRIỆU_CHỨNG",
                "position": [0, 3],
                "assertions": [],
            },
            {
                "text": "metformin",
                "type": "THUỐC",
                "position": [10, 19],
                "assertions": ["isHistorical"],
                "candidates": ["6809"],
            },
            {
                "text": "tăng huyết áp",
                "type": "CHẨN_ĐOÁN",
                "position": [23, 36],
                "assertions": [],
                "candidates": ["I10"],
            },
        ],
    )


def _assignments(
    record: CanonicalRecord,
    *,
    split: str = "synthetic_train",
    fold: int | None = None,
) -> dict[str, SplitAssignment]:
    return {
        record.record_id: SplitAssignment(
            record_id=record.record_id,
            split=split,
            fold=fold,
        )
    }


def test_ner_projection_keeps_all_entities_offsets_and_assertions():
    record = _record()

    projected = project_ner_records((record,), _assignments(record))

    assert len(projected[0]["entities"]) == 3
    assert projected[0]["entities"][1] == {
        "text": "metformin",
        "type": "THUỐC",
        "position": [10, 19],
        "assertions": ["isHistorical"],
        "candidates": ["6809"],
    }
    assert projected[0]["split"] == "synthetic_train"


def test_embedding_projection_keeps_only_coded_diagnosis_and_drug():
    record = _record()

    projected = project_embedding_seeds((record,), _assignments(record))

    assert {item["positive_codes"][0] for item in projected} == {"I10", "6809"}
    assert {item["entity_type"] for item in projected} == {"CHẨN_ĐOÁN", "THUỐC"}
    assert all(item["context"] == record.text for item in projected)


def test_reranker_seed_has_no_fabricated_candidate_pool():
    record = _record()

    projected = project_reranker_seeds((record,), _assignments(record))
    diagnosis = next(
        item for item in projected if item["entity_type"] == "CHẨN_ĐOÁN"
    )

    assert diagnosis["ground_truth_codes"] == ["I10"]
    assert "candidates" not in diagnosis


def test_trusted_projection_preserves_fold_without_changing_split():
    record = _record()

    projected = project_embedding_seeds(
        (record,),
        _assignments(record, split="trusted_fold", fold=3),
    )

    assert {item["fold"] for item in projected} == {3}
    assert {item["split"] for item in projected} == {"trusted_fold"}


def test_projection_is_deterministic_and_requires_assignment_coverage():
    first = _record()
    second = CanonicalRecord.create(
        record_id="synthetic-0002",
        source="synthetic",
        trust_tier="synthetic_validated",
        text="Không sốt.",
        entity_mappings=[
            {
                "text": "sốt",
                "type": "TRIỆU_CHỨNG",
                "position": [6, 9],
                "assertions": ["isNegated"],
            }
        ],
    )
    assignments = {
        first.record_id: SplitAssignment(first.record_id, "synthetic_train"),
        second.record_id: SplitAssignment(second.record_id, "synthetic_validation"),
    }

    assert project_ner_records(
        (first, second), assignments
    ) == project_ner_records((second, first), assignments)

    with pytest.raises(ValueError, match="missing split assignment"):
        project_ner_records((first,), {})
