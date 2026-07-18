from types import MappingProxyType

import pytest

from src.training.contracts import CanonicalEntity, CanonicalRecord


def test_entity_requires_exact_half_open_span():
    text = "Dùng metformin hằng ngày"

    entity = CanonicalEntity.from_mapping(
        {
            "text": "metformin",
            "type": "THUỐC",
            "position": [5, 14],
            "assertions": ["isHistorical", "isHistorical"],
            "candidates": [6809, "6809"],
        },
        text,
    )

    assert entity.start == 5
    assert entity.end == 14
    assert entity.assertions == ("isHistorical",)
    assert entity.codes == ("6809",)
    assert entity.to_mapping() == {
        "text": "metformin",
        "type": "THUỐC",
        "position": [5, 14],
        "assertions": ["isHistorical"],
        "candidates": ["6809"],
    }


def test_entity_rejects_mismatched_span():
    with pytest.raises(ValueError, match="span text mismatch"):
        CanonicalEntity.from_mapping(
            {
                "text": "metformin",
                "type": "THUỐC",
                "position": [0, 9],
            },
            "Dùng metformin",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("position", [False, 9], "integer offsets"),
        ("assertions", ["unknown"], "invalid assertions"),
        ("candidates", [True], "invalid candidates"),
        ("type", "UNKNOWN", "unsupported entity type"),
    ],
)
def test_entity_rejects_invalid_schema_values(field, value, message):
    mapping = {
        "text": "metformin",
        "type": "THUỐC",
        "position": [5, 14],
        "assertions": [],
        "candidates": [],
    }
    mapping[field] = value

    with pytest.raises(ValueError, match=message):
        CanonicalEntity.from_mapping(mapping, "Dùng metformin")


def test_record_create_is_immutable_and_round_trips():
    record = CanonicalRecord.create(
        record_id="synthetic-0001",
        source="synthetic",
        trust_tier="synthetic_validated",
        text="Tăng huyết áp.",
        entity_mappings=[
            {
                "text": "Tăng huyết áp",
                "type": "CHẨN_ĐOÁN",
                "position": [0, 13],
                "assertions": [],
                "candidates": ["I10"],
            }
        ],
        split_group="profile-I10",
        metadata={"family": "rare_diagnosis"},
    )

    assert isinstance(record.metadata, MappingProxyType)
    assert len(record.sha256) == 64
    assert record.to_mapping()["entities"][0]["candidates"] == ["I10"]
    with pytest.raises(TypeError):
        record.metadata["family"] = "changed"


def test_record_rejects_blank_identity_fields():
    with pytest.raises(ValueError, match="record_id"):
        CanonicalRecord.create(
            record_id=" ",
            source="synthetic",
            trust_tier="synthetic_validated",
            text="Nội dung",
            entity_mappings=[],
        )
