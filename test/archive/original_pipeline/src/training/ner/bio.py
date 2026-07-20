"""Offset-safe BIO alignment and deterministic constrained decoding."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


ENTITY_TYPES = (
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
)
LABELS = ("O",) + tuple(
    label
    for entity_type in ENTITY_TYPES
    for label in (f"B-{entity_type}", f"I-{entity_type}")
)
LABEL2ID = {label: index for index, label in enumerate(LABELS)}
ID2LABEL = {index: label for label, index in LABEL2ID.items()}


def _offset(value: Any) -> tuple[int, int]:
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int)
            for item in value
        )
    ):
        raise ValueError("token offsets must contain two integers")
    start, end = value
    if start < 0 or end < start:
        raise ValueError("token offsets are invalid")
    return start, end


def _entity_span(
    text: str,
    entity: Mapping[str, Any],
) -> tuple[int, int, str]:
    if not isinstance(entity, Mapping):
        raise ValueError("entity must be a mapping")
    position = entity.get("position")
    start, end = _offset(position)
    entity_type = entity.get("type")
    entity_text = entity.get("text")
    if entity_type not in ENTITY_TYPES:
        raise ValueError(f"unsupported NER entity type: {entity_type}")
    if (
        not isinstance(entity_text, str)
        or start >= end
        or end > len(text)
        or text[start:end] != entity_text
    ):
        raise ValueError("NER entity span does not match document text")
    return start, end, entity_type


def align_bio_labels(
    text: str,
    offset_mapping: Sequence[Sequence[int]],
    entities: Sequence[Mapping[str, Any]],
) -> list[int]:
    offsets = tuple(_offset(value) for value in offset_mapping)
    visible_offsets = [(start, end) for start, end in offsets if end > start]
    if not visible_offsets:
        return [-100 for _ in offsets]
    window_start = min(start for start, _ in visible_offsets)
    window_end = max(end for _, end in visible_offsets)

    parsed = [
        (*_entity_span(text, entity), index)
        for index, entity in enumerate(entities)
    ]
    for left_index, left in enumerate(parsed):
        for right in parsed[left_index + 1 :]:
            if left[0] < right[1] and right[0] < left[1]:
                raise ValueError("NER entities overlap and cannot use flat BIO")

    full_entities = [
        value
        for value in parsed
        if value[0] >= window_start and value[1] <= window_end
    ]
    partial_entities = [
        value
        for value in parsed
        if value[0] < window_end
        and window_start < value[1]
        and value not in full_entities
    ]
    first_token_seen: set[int] = set()
    labels: list[int] = []
    for token_start, token_end in offsets:
        if token_end <= token_start:
            labels.append(-100)
            continue
        if any(
            entity_start < token_end and token_start < entity_end
            for entity_start, entity_end, _, _ in partial_entities
        ):
            labels.append(-100)
            continue
        matches = [
            value
            for value in full_entities
            if value[0] < token_end and token_start < value[1]
        ]
        if len(matches) > 1:
            raise ValueError("one token overlaps multiple NER entities")
        if not matches:
            labels.append(LABEL2ID["O"])
            continue
        _, _, entity_type, entity_index = matches[0]
        prefix = "B" if entity_index not in first_token_seen else "I"
        first_token_seen.add(entity_index)
        labels.append(LABEL2ID[f"{prefix}-{entity_type}"])
    return labels


def constrain_bio_labels(labels: Sequence[str]) -> list[str]:
    constrained: list[str] = []
    active_type: str | None = None
    for label in labels:
        if label == "O":
            constrained.append(label)
            active_type = None
            continue
        if not isinstance(label, str) or "-" not in label:
            raise ValueError(f"invalid BIO label: {label!r}")
        prefix, entity_type = label.split("-", 1)
        if prefix not in {"B", "I"} or entity_type not in ENTITY_TYPES:
            raise ValueError(f"invalid BIO label: {label!r}")
        if prefix == "I" and active_type != entity_type:
            prefix = "B"
        constrained.append(f"{prefix}-{entity_type}")
        active_type = entity_type
    return constrained


def decode_bio_entities(
    text: str,
    offset_mapping: Sequence[Sequence[int]],
    label_ids: Sequence[int],
    *,
    attention_mask: Sequence[int] | None = None,
    confidences: Sequence[float] | None = None,
    record_id: str | None = None,
    id2label: Mapping[int, str] = ID2LABEL,
) -> tuple[dict[str, Any], ...]:
    if len(offset_mapping) != len(label_ids):
        raise ValueError(f"offset and label lengths differ: {len(offset_mapping)} vs {len(label_ids)}")
    if confidences is not None and len(confidences) != len(label_ids):
        raise ValueError("confidence and label lengths differ")
    if attention_mask is not None and len(attention_mask) != len(label_ids):
        raise ValueError("attention_mask and label lengths differ")
    offsets = tuple(_offset(value) for value in offset_mapping)
    raw_labels = [
        "O" if label_id == -100 else id2label.get(int(label_id), "")
        for label_id in label_ids
    ]
    constrained = constrain_bio_labels(raw_labels)

    entities: list[dict[str, Any]] = []
    active_type: str | None = None
    active_start: int | None = None
    active_end: int | None = None
    active_confidences: list[float] = []

    def close_active() -> None:
        nonlocal active_type, active_start, active_end, active_confidences
        if active_type is None or active_start is None or active_end is None:
            return
        item: dict[str, Any] = {
            "text": text[active_start:active_end],
            "type": active_type,
            "position": [active_start, active_end],
        }
        if record_id is not None:
            item["record_id"] = record_id
        if active_confidences:
            item["confidence"] = sum(active_confidences) / len(active_confidences)
        entities.append(item)
        active_type = None
        active_start = None
        active_end = None
        active_confidences = []

    mask = attention_mask if attention_mask is not None else [1] * len(label_ids)
    
    for index, (label, (token_start, token_end), is_valid_token, raw_label_id) in enumerate(
        zip(constrained, offsets, mask, label_ids)
    ):
        valid = (
            is_valid_token == 1
            and raw_label_id != -100
            and token_end > token_start
        )
        if not valid or label == "O":
            close_active()
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or active_type != entity_type:
            close_active()
            active_type = entity_type
            active_start = token_start
            active_end = token_end
            active_confidences = []
        else:
            active_end = max(active_end or token_end, token_end)
        if confidences is not None:
            active_confidences.append(float(confidences[index]))
    close_active()
    return merge_decoded_entities(entities)


def merge_decoded_entities(
    entities: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    best: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entity in entities:
        position = entity.get("position")
        start, end = _offset(position)
        key = (
            entity.get("record_id"),
            entity.get("type"),
            start,
            end,
        )
        copied = dict(entity)
        current = best.get(key)
        if current is None or float(copied.get("confidence", 0.0)) > float(
            current.get("confidence", 0.0)
        ):
            best[key] = copied
    return tuple(
        best[key]
        for key in sorted(
            best,
            key=lambda value: (
                "" if value[0] is None else str(value[0]),
                value[2],
                value[3],
                str(value[1]),
            ),
        )
    )
