from __future__ import annotations

from typing import Sequence, TypedDict
import torch

from .examples import TokenWindow


class TrainingBatch(TypedDict):
    input_ids: torch.LongTensor
    attention_mask: torch.LongTensor
    ner_labels: torch.LongTensor
    token_offsets: torch.LongTensor
    entity_spans: torch.LongTensor
    entity_types: torch.LongTensor
    assertion_targets: torch.FloatTensor
    assertion_mask: torch.BoolTensor


class ClinicalTokenCollator:
    def __init__(self, pad_token_id: int = 1) -> None:
        self.pad_token_id = pad_token_id

    def __call__(self, examples: Sequence[TokenWindow]) -> TrainingBatch:
        if not examples:
            raise ValueError("Cannot collate empty examples list")

        max_len = max(len(ex.input_ids) for ex in examples)

        batch_input_ids: list[list[int]] = []
        batch_attention_mask: list[list[int]] = []
        batch_ner_labels: list[list[int]] = []
        batch_token_offsets: list[list[list[int]]] = []

        for ex in examples:
            seq_len = len(ex.input_ids)
            pad_len = max_len - seq_len

            input_ids = list(ex.input_ids) + [self.pad_token_id] * pad_len
            attention_mask = list(ex.attention_mask) + [0] * pad_len

            ner_labels: list[int] = []
            for lbl, mask in zip(ex.label_ids, ex.loss_mask):
                ner_labels.append(lbl if mask else -100)
            ner_labels.extend([-100] * pad_len)

            token_offsets = [list(off) for off in ex.raw_offsets] + [[-1, -1]] * pad_len

            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_ner_labels.append(ner_labels)
            batch_token_offsets.append(token_offsets)

        entity_spans_list: list[list[int]] = []
        entity_types_list: list[int] = []
        assertion_targets_list: list[list[float]] = []
        assertion_mask_list: list[list[bool]] = []

        for b_idx, ex in enumerate(examples):
            seq_len = len(ex.input_ids)
            idx = 0
            owned_entity_index = 0
            while idx < seq_len:
                lbl = ex.label_ids[idx]
                mask = ex.loss_mask[idx]
                if mask and lbl > 0:
                    start_tok = idx
                    end_tok = idx + 1
                    while end_tok < seq_len and ex.loss_mask[end_tok] and ex.label_ids[end_tok] > 0 and (ex.label_ids[end_tok] % 2 == 0):
                        end_tok += 1
                    entity_spans_list.append([b_idx, start_tok, end_tok])
                    entity_type_id = (lbl - 1) // 2
                    entity_types_list.append(entity_type_id)
                    labels = (
                        ex.assertion_labels[owned_entity_index]
                        if owned_entity_index < len(ex.assertion_labels)
                        else ()
                    )
                    label_set = set(labels)
                    assertion_targets_list.append(
                        [
                            1.0 if axis in label_set else 0.0
                            for axis in ("isNegated", "isHistorical", "isFamily")
                        ]
                    )
                    is_lab = entity_type_id in {3, 4}
                    assertion_mask_list.append([not is_lab, not is_lab, not is_lab])
                    owned_entity_index += 1
                    idx = end_tok
                else:
                    idx += 1

        return TrainingBatch(
            input_ids=torch.tensor(batch_input_ids, dtype=torch.long),
            attention_mask=torch.tensor(batch_attention_mask, dtype=torch.long),
            ner_labels=torch.tensor(batch_ner_labels, dtype=torch.long),
            token_offsets=torch.tensor(batch_token_offsets, dtype=torch.long),
            entity_spans=torch.tensor(entity_spans_list if entity_spans_list else torch.empty((0, 3)), dtype=torch.long),
            entity_types=torch.tensor(entity_types_list if entity_types_list else torch.empty((0,)), dtype=torch.long),
            assertion_targets=torch.tensor(assertion_targets_list if assertion_targets_list else torch.empty((0, 3)), dtype=torch.float),
            assertion_mask=torch.tensor(assertion_mask_list if assertion_mask_list else torch.empty((0, 3)), dtype=torch.bool),
        )
