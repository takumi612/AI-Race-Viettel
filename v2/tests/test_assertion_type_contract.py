from __future__ import annotations

from clinical_nlp_lab.collation import ClinicalTokenCollator
from clinical_nlp_lab.entity_types import ASSERTION_ENTITY_TYPES, ENTITY_TYPE_TO_ID
from clinical_nlp_lab.examples import OwnedEntity, TokenWindow


def _window(entity_type: str) -> TokenWindow:
    return TokenWindow(
        document_id="doc",
        record_id="record",
        window_id=f"window-{entity_type}",
        input_ids=(0, 10, 2),
        attention_mask=(1, 1, 1),
        raw_offsets=((-1, -1), (0, 4), (-1, -1)),
        label_ids=(-100, 1, -100),
        loss_mask=(False, True, False),
        owned_entity_ids=("entity",),
        owned_entities=(
            OwnedEntity("entity", entity_type, 1, 2, ("isNegated",)),
        ),
    )


def test_entity_type_ids_are_canonical_across_training_and_runtime():
    assert ENTITY_TYPE_TO_ID == {
        "DISEASE": 0,
        "DRUG": 1,
        "SYMPTOM": 2,
        "LAB_NAME": 3,
        "LAB_RESULT": 4,
    }
    assert ASSERTION_ENTITY_TYPES == frozenset({"DISEASE", "DRUG", "SYMPTOM"})


def test_collator_uses_explicit_entity_type_and_masks_only_lab_types():
    entity_types = ("DISEASE", "DRUG", "SYMPTOM", "LAB_NAME", "LAB_RESULT")
    batch = ClinicalTokenCollator(pad_token_id=1)([_window(item) for item in entity_types])

    assert batch["entity_types"].tolist() == [0, 1, 2, 3, 4]
    assert batch["assertion_mask"].tolist() == [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [False, False, False],
        [False, False, False],
    ]
