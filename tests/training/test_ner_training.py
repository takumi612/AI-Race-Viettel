import pytest

from src.training.ner.bio import (
    LABEL2ID,
    align_bio_labels,
    constrain_bio_labels,
    decode_bio_entities,
    merge_decoded_entities,
)
from src.training.ner.data import select_ner_records, tokenize_ner_records


def _entity(text, entity_type, start, end):
    return {
        "text": text,
        "type": entity_type,
        "position": [start, end],
        "assertions": [],
        "candidates": [],
    }


def test_bio_alignment_uses_offsets_and_masks_special_tokens():
    text = "Sốt cao"
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]

    labels = align_bio_labels(
        text,
        offsets,
        (_entity("Sốt cao", "TRIỆU_CHỨNG", 0, 7),),
    )

    assert labels == [
        -100,
        LABEL2ID["B-TRIỆU_CHỨNG"],
        LABEL2ID["I-TRIỆU_CHỨNG"],
        -100,
    ]


def test_invalid_inside_transitions_are_converted_to_begin():
    constrained = constrain_bio_labels(
        ["O", "I-THUỐC", "I-THUỐC", "I-CHẨN_ĐOÁN"]
    )

    assert constrained == [
        "O",
        "B-THUỐC",
        "I-THUỐC",
        "B-CHẨN_ĐOÁN",
    ]


def test_decode_returns_exact_document_slices_and_deduplicates_windows():
    text = "Sốt cao, dùng metformin."
    offsets = [(0, 0), (0, 3), (4, 7), (0, 0)]
    label_ids = [
        LABEL2ID["O"],
        LABEL2ID["B-TRIỆU_CHỨNG"],
        LABEL2ID["I-TRIỆU_CHỨNG"],
        LABEL2ID["O"],
    ]

    decoded = decode_bio_entities(
        text,
        offsets,
        label_ids,
        confidences=[1.0, 0.9, 0.8, 1.0],
        record_id="101",
    )
    duplicate = dict(decoded[0], confidence=0.7)

    assert decoded[0]["text"] == "Sốt cao"
    assert decoded[0]["position"] == [0, 7]
    assert merge_decoded_entities((*decoded, duplicate)) == decoded


class _FakeFastTokenizer:
    is_fast = True

    def __call__(self, text, **kwargs):
        assert kwargs["return_offsets_mapping"] is True
        assert kwargs["return_overflowing_tokens"] is True
        return {
            "input_ids": [
                [0, 10, 11, 12, 13, 2],
                [0, 20, 21, 22, 2],
            ],
            "attention_mask": [
                [1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1],
            ],
            "offset_mapping": [
                [(0, 0), (0, 2), (2, 4), (4, 6), (6, 8), (0, 0)],
                [(0, 0), (4, 6), (6, 8), (8, 10), (0, 0)],
            ],
        }


def test_overflow_chunks_mask_partial_entities_instead_of_false_negatives():
    record = {
        "record_id": "synthetic-0001",
        "text": "abcdefghij",
        "entities": [_entity("cdef", "THUỐC", 2, 6)],
        "split": "synthetic_train",
    }

    features = tokenize_ner_records(
        (record,),
        _FakeFastTokenizer(),
        max_length=6,
        stride=2,
    )

    assert features[0]["labels"][2:4] == [
        LABEL2ID["B-THUỐC"],
        LABEL2ID["I-THUỐC"],
    ]
    assert features[1]["labels"][1] == -100
    assert features[0]["absolute_offsets"][2] == [2, 4]


def test_stage_selection_never_exposes_holdout_or_wrong_fold_to_gradient():
    records = (
        {"record_id": "s1", "split": "synthetic_train"},
        {"record_id": "s2", "split": "synthetic_validation"},
        {"record_id": "101", "split": "trusted_fold", "fold": 0},
        {"record_id": "102", "split": "trusted_fold", "fold": 1},
        {"record_id": "181", "split": "holdout"},
    )

    train = select_ner_records(
        records,
        stage="trusted-fold",
        role="train",
        fold=1,
    )
    evaluation = select_ner_records(
        records,
        stage="trusted-fold",
        role="eval",
        fold=1,
    )

    assert {item["record_id"] for item in train} == {"101"}
    assert {item["record_id"] for item in evaluation} == {"102"}
    assert all(item["record_id"] != "181" for item in (*train, *evaluation))


def test_tokenizer_must_be_fast():
    class SlowTokenizer:
        is_fast = False

    with pytest.raises(ValueError, match="fast tokenizer"):
        tokenize_ner_records((), SlowTokenizer(), max_length=8, stride=2)
