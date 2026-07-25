from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .records import ClinicalRecord
from .schema import ClinicalDocument, EntityAnnotation


@dataclass(frozen=True)
class OwnedEntity:
    entity_id: str
    entity_type: str
    token_start: int
    token_end: int
    assertions: tuple[str, ...] = ()


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
    assertion_labels: tuple[tuple[str, ...], ...] = ()
    owned_entities: tuple[OwnedEntity, ...] = ()


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
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            add_special_tokens=True,
        )
        input_windows = encoded["input_ids"]
        offset_windows = encoded["offset_mapping"]
        attention_windows = encoded.get("attention_mask")
        if input_windows and isinstance(input_windows[0], int):
            input_windows = [input_windows]
            offset_windows = [offset_windows]
            attention_windows = [attention_windows or [1] * len(input_windows[0])]
        elif attention_windows is None:
            attention_windows = [[1] * len(item) for item in input_windows]

        absolute_offset_windows: list[tuple[tuple[int, int], ...]] = []
        for offset_window in offset_windows:
            absolute_offset_windows.append(
                tuple(
                    (-1, -1)
                    if start_rel == end_rel == 0
                    else (record.raw_start + int(start_rel), record.raw_start + int(end_rel))
                    for start_rel, end_rel in offset_window
                )
            )

        window_spans: list[tuple[int, int, int]] = []
        for w_idx, window_offsets in enumerate(absolute_offset_windows):
            tok_offsets = [off for off in window_offsets if off != (-1, -1)]
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

        for w_idx, input_window in enumerate(input_windows):
            w_input_ids = tuple(int(item) for item in input_window)
            w_offsets = absolute_offset_windows[w_idx]
            w_attention = tuple(int(item) for item in attention_windows[w_idx])

            owned_metadata: list[OwnedEntity] = []
            for ent_id, entity in record_entities:
                if entity_owner_window[ent_id] != w_idx:
                    continue
                token_indices = [
                    index
                    for index, (tok_s, tok_e) in enumerate(w_offsets)
                    if tok_s >= 0 and max(tok_s, entity.start) < min(tok_e, entity.end)
                ]
                if token_indices:
                    owned_metadata.append(
                        OwnedEntity(
                            entity_id=ent_id,
                            entity_type=entity.type,
                            token_start=min(token_indices),
                            token_end=max(token_indices) + 1,
                            assertions=tuple(entity.assertions),
                        )
                    )
            owned_ent_ids = tuple(item.entity_id for item in owned_metadata)
            assertion_labels = tuple(item.assertions for item in owned_metadata)

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
                    assertion_labels=assertion_labels,
                    owned_entities=tuple(owned_metadata),
                )
            )

    return tuple(windows)
