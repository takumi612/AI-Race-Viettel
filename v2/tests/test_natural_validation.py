from __future__ import annotations

import pytest

from clinical_nlp_lab.natural_validation import (
    NATURAL_VALIDATION_IDS,
    REQUIRED_DATASET_FINGERPRINT,
    apply_natural_validation_partition,
    build_natural_validation_manifest,
    validate_natural_validation_summary,
)


def test_natural_validation_partition_is_fixed_and_disjoint():
    train, validation = apply_natural_validation_partition(
        tuple(str(value) for value in range(101, 221)),
        ("501", "502"),
    )

    assert NATURAL_VALIDATION_IDS == tuple(str(value) for value in range(181, 201))
    assert validation == (*NATURAL_VALIDATION_IDS, "501", "502")
    assert not set(train) & set(validation)
    assert not set(train) & set(NATURAL_VALIDATION_IDS)


def test_manifest_rejects_wrong_dataset_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="fingerprint"):
        build_natural_validation_manifest(tmp_path, "wrong")


def test_natural_validation_summary_binds_expected_inventory():
    summary = {
        "document_ids": [str(value) for value in range(181, 201)],
        "entity_count": 1488,
        "type_counts": {
            "CHẨN_ĐOÁN": 496,
            "THUỐC": 248,
            "TRIỆU_CHỨNG": 496,
            "TÊN_XÉT_NGHIỆM": 124,
            "KẾT_QUẢ_XÉT_NGHIỆM": 124,
        },
        "over_50": 120,
        "over_100": 5,
        "max_length": 110,
    }

    manifest = validate_natural_validation_summary(
        summary, REQUIRED_DATASET_FINGERPRINT
    )

    assert manifest["schema_id"] == "clinical_nlp.natural_validation"
    assert manifest["summary"] == summary
