from __future__ import annotations

from clinical_nlp_lab.examples import build_owner_windows
from clinical_nlp_lab.records import ClinicalRecord
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation
from clinical_nlp_lab.training import build_bio_label_map


class NativeOverflowTokenizer:
    """Small deterministic tokenizer that mirrors HF overflow output shape."""

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        stride: int,
        return_offsets_mapping: bool,
        return_overflowing_tokens: bool,
        add_special_tokens: bool = True,
    ):
        assert truncation is True
        assert return_offsets_mapping is True
        assert return_overflowing_tokens is True
        words = text.split()
        offsets = []
        cursor = 0
        for word in words:
            start = text.index(word, cursor)
            end = start + len(word)
            offsets.append((start, end))
            cursor = end

        capacity = max_length - 2
        step = capacity - stride
        windows = []
        start = 0
        while start < len(words):
            end = min(len(words), start + capacity)
            ids = [0] + list(range(10 + start, 10 + end)) + [2]
            window_offsets = [(0, 0)] + offsets[start:end] + [(0, 0)]
            windows.append((ids, window_offsets))
            if end == len(words):
                break
            start += step
        return {
            "input_ids": [item[0] for item in windows],
            "attention_mask": [[1] * len(item[0]) for item in windows],
            "offset_mapping": [item[1] for item in windows],
            "overflow_to_sample_mapping": [0] * len(windows),
        }


def test_each_training_overflow_window_keeps_special_tokens_and_one_owner():
    raw_text = "zero one two three four five six seven"
    entity = EntityAnnotation(
        text="four five",
        type="SYMPTOM",
        position=(19, 28),
        assertions=["isHistorical"],
    )
    document = ClinicalDocument("doc", raw_text, entities=[entity])
    records = [ClinicalRecord("doc", "record", 0, len(raw_text), (0,))]
    label_to_id, _ = build_bio_label_map(["SYMPTOM"])

    windows = build_owner_windows(
        document,
        records,
        NativeOverflowTokenizer(),
        label_to_id,
        max_length=6,
        stride=2,
    )

    assert len(windows) == 3
    assert all(window.input_ids[0] == 0 and window.input_ids[-1] == 2 for window in windows)
    assert sum(len(window.owned_entities) for window in windows) == 1
    owner = next(window for window in windows if window.owned_entities)
    assert owner.owned_entities[0].entity_type == "SYMPTOM"
    assert owner.owned_entities[0].assertions == ("isHistorical",)
    assert raw_text[owner.raw_offsets[owner.owned_entities[0].token_start][0] : owner.raw_offsets[owner.owned_entities[0].token_end - 1][1]] == "four five"
