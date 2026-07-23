from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .records import ClinicalRecord
from .schema import ClinicalDocument, EntityAnnotation


@dataclass(frozen=True)
class TokenWindow:
    document_id: str
    record_id: str
    window_id: str
    input_ids: tuple[int, ...]
    attention_mask: tuple[int, ...]
    raw_offsets: tuple[tuple[int, int], ...]
    label_ids: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    owned_entity_ids: tuple[str, ...]


def _find_owner_window(
    entity: EntityAnnotation,
    window_spans: list[tuple[int, int, int]],  # (window_idx, raw_start, raw_end)
) -> int | None:
    containing_windows = [
        (w_idx, w_start, w_end)
        for w_idx, w_start, w_end in window_spans
        if w_start <= entity.start and entity.end <= w_end
    ]
    if not containing_windows:
        return None

    def key_func(item: tuple[int, int, int]) -> tuple[int, int]:
        w_idx, w_start, w_end = item
        margin = min(entity.start - w_start, w_end - entity.end)
        return (margin, -w_idx)

    best_window = max(containing_windows, key=key_func)
    return best_window[0]


def build_owner_windows(
    document: ClinicalDocument,
    records: Sequence[ClinicalRecord],
    tokenizer: Any,
    label_to_id: Mapping[str, int],
    max_length: int = 512,
    stride: int = 128,
) -> tuple[TokenWindow, ...]:
    if max_length <= 0 or stride < 0 or stride >= max_length:
        raise ValueError("Require max_length > stride >= 0")

    entity_by_id: dict[str, EntityAnnotation] = {}
    for idx, entity in enumerate(document.entities):
        entity_id = f"{document.document_id}_e{idx}_{entity.start}_{entity.end}"
        entity_by_id[entity_id] = entity

    windows: list[TokenWindow] = []

    for record in records:
        record_text = document.raw_text[record.raw_start:record.raw_end]
        if not record_text and record.raw_start == record.raw_end:
            continue

        record_entities = [
            (f"{document.document_id}_e{idx}_{entity.start}_{entity.end}", entity)
            for idx, entity in enumerate(document.entities)
            if record.raw_start <= entity.start and entity.end <= record.raw_end
        ]

        encoded = tokenizer(
            record_text,
            truncation=False,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )

        input_ids_all = encoded["input_ids"]
        raw_offsets_rel = encoded["offset_mapping"]
        total_tokens = len(input_ids_all)

        raw_offsets_abs: list[tuple[int, int]] = []
        for start_rel, end_rel in raw_offsets_rel:
            if start_rel == end_rel == 0:
                raw_offsets_abs.append((-1, -1))
            else:
                raw_offsets_abs.append((record.raw_start + start_rel, record.raw_start + end_rel))

        sub_windows: list[tuple[int, int]] = []
        start_tok = 0
        while start_tok < total_tokens:
            end_tok = min(total_tokens, start_tok + max_length)
            sub_windows.append((start_tok, end_tok))
            if end_tok >= total_tokens:
                break
            start_tok = end_tok - stride

        window_spans: list[tuple[int, int, int]] = []
        for w_idx, (st, et) in enumerate(sub_windows):
            tok_offsets = [off for off in raw_offsets_abs[st:et] if off != (-1, -1)]
            if tok_offsets:
                w_start = min(s for s, _ in tok_offsets)
                w_end = max(e for _, e in tok_offsets)
            else:
                w_start = record.raw_start
                w_end = record.raw_start
            window_spans.append((w_idx, w_start, w_end))

        entity_owner_window: dict[str, int | None] = {}
        for ent_id, entity in record_entities:
            entity_owner_window[ent_id] = _find_owner_window(entity, window_spans)

        for w_idx, (st, et) in enumerate(sub_windows):
            w_input_ids = tuple(input_ids_all[st:et])
            w_offsets = tuple(raw_offsets_abs[st:et])
            w_attention = tuple([1] * len(w_input_ids))

            owned_ent_ids = tuple(
                ent_id
                for ent_id, entity in record_entities
                if entity_owner_window[ent_id] == w_idx
            )

            label_ids: list[int] = []
            loss_masks: list[bool] = []

            for tok_s, tok_e in w_offsets:
                if tok_s == -1 or tok_e == -1 or tok_s == tok_e:
                    label_ids.append(-100)
                    loss_masks.append(False)
                    continue

                overlapping_owned = [
                    (ent_id, ent)
                    for ent_id, ent in record_entities
                    if entity_owner_window[ent_id] == w_idx and max(tok_s, ent.start) < min(tok_e, ent.end)
                ]

                overlapping_unowned = [
                    (ent_id, ent)
                    for ent_id, ent in record_entities
                    if entity_owner_window[ent_id] != w_idx and max(tok_s, ent.start) < min(tok_e, ent.end)
                ]

                if overlapping_unowned:
                    label_ids.append(-100)
                    loss_masks.append(False)
                elif overlapping_owned:
                    ent_id, ent = overlapping_owned[0]
                    prev_tok_owned = False
                    for prev_s, prev_e in w_offsets:
                        if prev_s < tok_s and prev_s != -1 and max(prev_s, ent.start) < min(prev_e, ent.end):
                            prev_tok_owned = True
                            break
                    prefix = "I" if prev_tok_owned else "B"
                    lbl_str = f"{prefix}-{ent.type}"
                    label_ids.append(label_to_id[lbl_str])
                    loss_masks.append(True)
                else:
                    label_ids.append(label_to_id["O"])
                    loss_masks.append(True)

            win_id = f"{document.document_id}_{record.record_id}_w{w_idx}"
            windows.append(
                TokenWindow(
                    document_id=document.document_id,
                    record_id=record.record_id,
                    window_id=win_id,
                    input_ids=w_input_ids,
                    attention_mask=w_attention,
                    raw_offsets=w_offsets,
                    label_ids=tuple(label_ids),
                    loss_mask=tuple(loss_masks),
                    owned_entity_ids=owned_ent_ids,
                )
            )

    return tuple(windows)
