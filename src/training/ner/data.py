"""NER JSONL selection and tokenizer-overflow feature preparation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.ner.bio import align_bio_labels


def load_ner_jsonl(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"NER dataset is missing: {source}")
    records: list[dict[str, Any]] = []
    try:
        with source.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank NER JSONL line: {line_number}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"NER line {line_number} must be an object")
                records.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid NER JSONL: {source}") from exc
    return tuple(records)


def select_ner_records(
    records: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    role: str,
    fold: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if stage not in {"synthetic", "trusted-fold", "trusted-final"}:
        raise ValueError(f"unsupported NER stage: {stage}")
    if role not in {"train", "eval"}:
        raise ValueError("NER role must be train or eval")
    if stage == "trusted-fold":
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 5:
            raise ValueError("trusted-fold requires fold in [0, 4]")
    elif fold is not None:
        raise ValueError(f"{stage} does not accept a fold")

    selected: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping):
            raise ValueError("NER records must be mappings")
        split = record.get("split")
        if isinstance(split, str) and "pseudo" in split.casefold():
            raise ValueError("pseudo-label records are forbidden in NER training")
        include = False
        if stage == "synthetic":
            expected = "synthetic_train" if role == "train" else "synthetic_validation"
            include = split == expected
        elif stage == "trusted-fold" and split == "trusted_fold":
            record_fold = record.get("fold")
            include = record_fold != fold if role == "train" else record_fold == fold
        elif stage == "trusted-final":
            include = role == "train" and split == "trusted_fold"
        if include:
            selected.append(dict(record))
    return tuple(
        sorted(selected, key=lambda record: str(record.get("record_id", "")))
    )


def _windows(encoded: Mapping[str, Any]) -> list[dict[str, Any]]:
    required = {"input_ids", "attention_mask", "offset_mapping"}
    if not required <= set(encoded):
        raise ValueError("tokenizer output is missing required overflow fields")
    input_ids = encoded["input_ids"]
    attention_masks = encoded["attention_mask"]
    offsets = encoded["offset_mapping"]
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]
        attention_masks = [attention_masks]
        offsets = [offsets]
    if not (len(input_ids) == len(attention_masks) == len(offsets)):
        raise ValueError("tokenizer overflow fields have different lengths")
    return [
        {
            "input_ids": list(ids),
            "attention_mask": list(mask),
            "offset_mapping": list(mapping),
        }
        for ids, mask, mapping in zip(input_ids, attention_masks, offsets)
    ]


def tokenize_ner_records(
    records: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    stride: int,
) -> tuple[dict[str, Any], ...]:
    if getattr(tokenizer, "is_fast", False) is not True:
        raise ValueError("NER requires a fast tokenizer with offset mappings")
    if (
        isinstance(max_length, bool)
        or not isinstance(max_length, int)
        or max_length < 4
        or isinstance(stride, bool)
        or not isinstance(stride, int)
        or not 0 <= stride < max_length
    ):
        raise ValueError("invalid NER max_length/stride")

    features: list[dict[str, Any]] = []
    from src.training.ner.bio import LABEL2ID
    for record in records:
        record_id = record.get("record_id")
        text = record.get("text")
        entities = record.get("entities", [])
        tokens = record.get("tokens")
        ner_tags = record.get("ner_tags")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("NER record_id must be non-empty")
        if not text and not tokens:
            raise ValueError(f"NER text and tokens are empty for {record_id}")
        if not isinstance(entities, list):
            raise ValueError(f"NER entities must be a list for {record_id}")

        if tokens and ner_tags:
            encoded = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                stride=stride,
                padding=False,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
            )
        else:
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                stride=stride,
                padding=False,
                return_offsets_mapping=True,
                return_overflowing_tokens=True,
            )
            
        covered_entities: set[int] = set()
        for window_index, window in enumerate(_windows(encoded)):
            offsets = window.pop("offset_mapping")
            visible = [
                tuple(offset)
                for offset in offsets
                if len(offset) == 2 and offset[1] > offset[0]
            ]
            if not visible:
                continue
            window_start = min(offset[0] for offset in visible)
            window_end = max(offset[1] for offset in visible)
            
            if tokens and ner_tags:
                labels = []
                window_word_ids = encoded.word_ids(batch_index=window_index)
                previous_word_idx = None
                for word_idx in window_word_ids:
                    if word_idx is None:
                        labels.append(-100)
                    elif word_idx != previous_word_idx:
                        labels.append(LABEL2ID.get(ner_tags[word_idx], -100))
                    else:
                        labels.append(-100)
                    previous_word_idx = word_idx
            else:
                for entity_index, entity in enumerate(entities):
                    position = entity.get("position") if isinstance(entity, Mapping) else None
                    if (
                        isinstance(position, list)
                        and len(position) == 2
                        and position[0] >= window_start
                        and position[1] <= window_end
                    ):
                        covered_entities.add(entity_index)
                labels = align_bio_labels(text, offsets, entities)
                
            feature: dict[str, Any] = {
                **window,
                "labels": labels,
                "absolute_offsets": [list(offset) for offset in offsets],
                "record_id": record_id,
                "text": text or " ".join(tokens),
                "split": record.get("split"),
            }
            if "fold" in record:
                feature["fold"] = record["fold"]
            features.append(feature)

        missing_entities = set(range(len(entities))) - covered_entities
        if missing_entities:
            raise ValueError(
                f"NER entities are longer than every tokenizer window for "
                f"{record_id}: {sorted(missing_entities)}"
            )
    return tuple(features)
