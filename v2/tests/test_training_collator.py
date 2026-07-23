from __future__ import annotations

import pytest
import torch
from clinical_nlp_lab.examples import TokenWindow
from clinical_nlp_lab.collation import ClinicalTokenCollator


def test_clinical_token_collator_shape_and_padding():
    w1 = TokenWindow(
        document_id="doc1",
        record_id="rec1",
        window_id="w1",
        input_ids=(0, 10, 11, 2),
        attention_mask=(1, 1, 1, 1),
        raw_offsets=((-1, -1), (0, 4), (5, 9), (-1, -1)),
        label_ids=(-100, 1, 2, -100),
        loss_mask=(False, True, True, False),
        owned_entity_ids=("e1",),
    )

    w2 = TokenWindow(
        document_id="doc1",
        record_id="rec1",
        window_id="w2",
        input_ids=(0, 12, 2),
        attention_mask=(1, 1, 1),
        raw_offsets=((-1, -1), (10, 15), (-1, -1)),
        label_ids=(-100, 0, -100),
        loss_mask=(False, True, False),
        owned_entity_ids=(),
    )

    collator = ClinicalTokenCollator(pad_token_id=1)
    batch = collator([w1, w2])

    assert batch["input_ids"].shape == (2, 4)
    assert batch["attention_mask"].shape == (2, 4)
    assert batch["ner_labels"].shape == (2, 4)
    assert batch["token_offsets"].shape == (2, 4, 2)

    assert batch["input_ids"][1, 3].item() == 1
    assert batch["attention_mask"][1, 3].item() == 0
    assert batch["ner_labels"][1, 3].item() == -100
    assert tuple(batch["token_offsets"][1, 3].tolist()) == (-1, -1)


def test_collator_empty_list_raises():
    collator = ClinicalTokenCollator()
    with pytest.raises(ValueError, match="Cannot collate empty examples list"):
        collator([])
