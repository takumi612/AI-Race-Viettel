from __future__ import annotations

import pytest
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation
from clinical_nlp_lab.records import ClinicalRecord
from clinical_nlp_lab.examples import build_owner_windows
from clinical_nlp_lab.training import build_bio_label_map


class DummyFastTokenizer:
    def __call__(
        self,
        text: str,
        truncation: bool = False,
        return_offsets_mapping: bool = True,
        add_special_tokens: bool = True,
    ):
        tokens = text.split(" ")
        offsets: list[tuple[int, int]] = []
        input_ids: list[int] = []

        if add_special_tokens:
            input_ids.append(0)
            offsets.append((0, 0))

        curr_pos = 0
        for idx, token_str in enumerate(tokens):
            if not token_str:
                curr_pos += 1
                continue
            start = text.find(token_str, curr_pos)
            end = start + len(token_str)
            curr_pos = end
            input_ids.append(10 + idx)
            offsets.append((start, end))

        if add_special_tokens:
            input_ids.append(2)
            offsets.append((0, 0))

        return {"input_ids": input_ids, "offset_mapping": offsets}


def test_owner_window_single_and_overlapping():
    raw_text = "Bệnh nhân bị Sốt xuất huyết. Đã dùng Paracetamol 500mg hôm qua."
    entities = [
        EntityAnnotation(text="Sốt xuất huyết", type="DISEASE", position=(13, 27)),
        EntityAnnotation(text="Paracetamol", type="DRUG", position=(37, 48)),
    ]
    doc = ClinicalDocument(document_id="doc1", raw_text=raw_text, entities=entities)
    records = [
        ClinicalRecord(
            document_id="doc1",
            record_id="rec1",
            raw_start=0,
            raw_end=len(raw_text),
            entity_indices=(0, 1),
        )
    ]
    label_to_id, _ = build_bio_label_map(["DISEASE", "DRUG", "SYMPTOM", "LAB_NAME", "LAB_RESULT"])
    tokenizer = DummyFastTokenizer()

    windows1 = build_owner_windows(doc, records, tokenizer, label_to_id, max_length=512, stride=128)
    windows2 = build_owner_windows(doc, records, tokenizer, label_to_id, max_length=10, stride=4)

    owned_counts1 = sum(len(w.owned_entity_ids) for w in windows1)
    owned_counts2 = sum(len(w.owned_entity_ids) for w in windows2)
    assert owned_counts1 == len(entities)
    assert owned_counts2 == len(entities)


def test_special_tokens_and_non_owner_tokens_masked():
    raw_text = "Bệnh nhân bị Bệnh tim."
    entities = [EntityAnnotation(text="Bệnh tim", type="DISEASE", position=(13, 21))]
    doc = ClinicalDocument(document_id="doc1", raw_text=raw_text, entities=entities)
    records = [
        ClinicalRecord(
            document_id="doc1",
            record_id="rec1",
            raw_start=0,
            raw_end=len(raw_text),
            entity_indices=(0,),
        )
    ]
    label_to_id, _ = build_bio_label_map(["DISEASE"])
    tokenizer = DummyFastTokenizer()

    windows = build_owner_windows(doc, records, tokenizer, label_to_id, max_length=512, stride=128)
    assert len(windows) == 1
    w = windows[0]

    assert w.label_ids[0] == -100
    assert w.loss_mask[0] is False
    assert w.label_ids[-1] == -100
    assert w.loss_mask[-1] is False


def test_raw_offsets_recovery():
    raw_text = "Khám lại sau 7 ngày."
    entities = [EntityAnnotation(text="7 ngày", type="LAB_RESULT", position=(13, 19))]
    doc = ClinicalDocument(document_id="doc2", raw_text=raw_text, entities=entities)
    records = [
        ClinicalRecord(
            document_id="doc2",
            record_id="rec1",
            raw_start=0,
            raw_end=len(raw_text),
            entity_indices=(0,),
        )
    ]
    label_to_id, _ = build_bio_label_map(["LAB_RESULT"])
    tokenizer = DummyFastTokenizer()

    windows = build_owner_windows(doc, records, tokenizer, label_to_id, max_length=512, stride=128)
    for w in windows:
        for (start, end) in w.raw_offsets:
            if (start, end) != (-1, -1):
                assert raw_text[start:end] != ""
