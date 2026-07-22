from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

from .schema import EntityAnnotation
from .text import normalize_alias, tokenize_with_offsets


SYMPTOM_PATTERN = re.compile(
    r"(?iu)\b(?:đau\s+ngực|khó\s+thở|sốt|ho|mệt\s+mỏi|đánh\s+trống\s+ngực|buồn\s+nôn|nôn|đổ\s+mồ\s+hôi)\b"
)
LAB_PATTERN = re.compile(
    r"(?iu)\b(?P<name>glucose(?:\s+máu)?|đường\s+huyết|bạch\s+cầu|hemoglobin|creatinin(?:e)?|canxi|calci|natri|kali)"
    r"(?:\s*(?:là|:)?\s*(?P<value>\d+(?:[.,]\d+)?)(?:\s*(?P<unit>mmol/L|mg/dL|g/L|G/L|µmol/L|mEq/L))?)?"
)
PATIENT_INFO_PATTERN = re.compile(
    r"(?iu)\b(?:nam|nữ|thai\s+\d+\s+tuần|\d{1,3}\s+tuổi|đang\s+mang\s+thai)\b"
)
MEDICATION_ATTRIBUTE_PATTERN = re.compile(
    r"(?iu)^(?:\s+|\s*[,;:]\s*)"
    r"(?:\d+(?:[.,-]\d+)?\s*(?:mg|mcg|µg|g|ml|mL|unit|đơn\s+vị)"
    r"|po|iv|im|sc|oral|uống|tiêm"
    r"|daily|bid|tid|qid|q\d+h|qam|qhs|prn"
    r"|tablet|capsule|suspension|injection|viên|ống)"
)


class DictionaryRuleEntityDetector:
    """Ontology-driven exact phrase detector with offset-preserving rules.

    The ontology is external knowledge, not fitted from private input text.
    All emitted spans are sliced directly from the raw document.
    """

    def __init__(
        self,
        icd10_records: Iterable[dict[str, Any]],
        rxnorm_records: Iterable[dict[str, Any]],
        phrase_confidence: float = 0.94,
        regex_confidence: float = 0.78,
        max_alias_tokens: int = 10,
        enable_generic_regex: bool = False,
    ) -> None:
        self.trie: dict[str, Any] = {}
        self.max_alias_tokens = max_alias_tokens
        self.phrase_confidence = phrase_confidence
        self.regex_confidence = regex_confidence
        self.enable_generic_regex = enable_generic_regex
        self.alias_count = 0
        self._build_trie(icd10_records, "DISEASE")
        self._build_trie(rxnorm_records, "DRUG")

    def _build_trie(self, records: Iterable[dict[str, Any]], internal_type: str) -> None:
        seen: set[tuple[str, str, str]] = set()
        for record in records:
            candidate_id = str(record["candidate_id"])
            aliases = record.get("detection_aliases") or []
            canonical_name = record.get("canonical_name") or record.get("name_vi") or record.get("name_en") or ""
            for alias in aliases:
                normalized = normalize_alias(str(alias))
                if len(normalized) < 4 or normalized.isdigit():
                    continue
                tokens = normalized.split()
                if not tokens or len(tokens) > self.max_alias_tokens:
                    continue
                if len(tokens) == 1 and len(tokens[0]) < 5:
                    continue
                key = (normalized, internal_type, candidate_id)
                if key in seen:
                    continue
                seen.add(key)
                node = self.trie
                for token in tokens:
                    node = node.setdefault(token, {})
                node.setdefault("_hits", []).append(
                    {
                        "type": internal_type,
                        "candidate_id": candidate_id,
                        "mention_head": str(canonical_name or alias),
                    }
                )
                self.alias_count += 1

    @staticmethod
    def _extend_drug_span(raw_text: str, end: int) -> int:
        cursor = end
        line_end = raw_text.find("\n", cursor)
        if line_end == -1:
            line_end = len(raw_text)
        limit = min(line_end, end + 96)
        while cursor < limit:
            match = MEDICATION_ATTRIBUTE_PATTERN.match(raw_text[cursor:limit])
            if not match:
                break
            cursor += match.end()
        return cursor

    def _dictionary_entities(self, raw_text: str) -> list[EntityAnnotation]:
        raw_tokens = tokenize_with_offsets(raw_text)
        normalized_tokens = [normalize_alias(token) for token, _start, _end in raw_tokens]
        entities: list[EntityAnnotation] = []

        for token_index in range(len(raw_tokens)):
            node = self.trie
            best_hits: list[dict[str, str]] | None = None
            best_end_index: int | None = None
            for lookahead in range(token_index, min(len(raw_tokens), token_index + self.max_alias_tokens)):
                token = normalized_tokens[lookahead]
                if token not in node:
                    break
                node = node[token]
                if "_hits" in node:
                    best_hits = node["_hits"]
                    best_end_index = lookahead
            if best_hits is None or best_end_index is None:
                continue
            start = raw_tokens[token_index][1]
            base_end = raw_tokens[best_end_index][2]
            for hit in best_hits:
                end = self._extend_drug_span(raw_text, base_end) if hit["type"] == "DRUG" else base_end
                entity = EntityAnnotation(
                    text=raw_text[start:end],
                    type=hit["type"],
                    position=(start, end),
                    candidates=[hit["candidate_id"]],
                    confidence=self.phrase_confidence,
                    mention_head=raw_text[start:base_end],
                    evidence=["ontology_exact_phrase"],
                )
                entity.validate_offset(raw_text)
                entities.append(entity)
        return entities

    def _regex_entities(self, raw_text: str) -> list[EntityAnnotation]:
        entities: list[EntityAnnotation] = []
        for internal_type, pattern, evidence in (
            ("SYMPTOM", SYMPTOM_PATTERN, "generic_clinical_symptom_rule"),
            ("LAB_RESULT", LAB_PATTERN, "generic_lab_value_rule"),
            ("PATIENT_INFO", PATIENT_INFO_PATTERN, "generic_patient_info_rule"),
        ):
            for match in pattern.finditer(raw_text):
                start, end = match.span()
                entity = EntityAnnotation(
                    text=raw_text[start:end],
                    type=internal_type,
                    position=(start, end),
                    confidence=self.regex_confidence,
                    mention_head=match.groupdict().get("name") if match.groupdict() else match.group(0),
                    evidence=[evidence],
                )
                entity.validate_offset(raw_text)
                entities.append(entity)
        return entities

    def detect(self, raw_text: str) -> list[EntityAnnotation]:
        dictionary_entities = self._dictionary_entities(raw_text)
        regex_entities = self._regex_entities(raw_text) if self.enable_generic_regex else []
        symptom_spans = [entity for entity in regex_entities if entity.type == "SYMPTOM"]
        # Generic symptom rules take precedence over ICD-10 phrase matches so a
        # symptom is not silently routed into the disease ontology.
        dictionary_entities = [
            entity
            for entity in dictionary_entities
            if not (
                entity.type == "DISEASE"
                and any(
                    spans_overlap(entity, symptom)
                    and normalize_alias(entity.text) == normalize_alias(symptom.text)
                    for symptom in symptom_spans
                )
            )
        ]
        detected = dictionary_entities + regex_entities
        return resolve_overlaps(detected, raw_text)


class TransformerNERDetector:
    """Load a saved Hugging Face token-classification checkpoint for inference."""

    def __init__(
        self,
        model_dir: str | Path,
        max_length: int = 512,
        stride: int = 128,
        device: str | None = None,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Transformer NER inference requires torch and transformers"
            ) from exc

        self.torch = torch
        self.model_dir = Path(model_dir)
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"NER checkpoint directory not found: {self.model_dir}")
        self.max_length = max_length
        self.stride = stride
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir), use_fast=True)
        self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_dir))
        self.model.to(self.device)
        self.model.eval()
        self.id_to_label = {
            int(index): str(label)
            for index, label in self.model.config.id2label.items()
        }

    def release(self) -> None:
        """Release the GPU-resident checkpoint before an LLM takes the device."""
        model = getattr(self, "model", None)
        self.model = None
        self.tokenizer = None
        if model is not None:
            model.to("cpu")
            del model
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def detect(self, raw_text: str) -> list[EntityAnnotation]:
        from .training import bio_predictions_to_spans

        encoded = self.tokenizer(
            raw_text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
            padding=True,
        )
        offsets = encoded.pop("offset_mapping")
        encoded.pop("overflow_to_sample_mapping", None)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            probabilities = self.model(**model_inputs).logits.softmax(dim=-1).cpu()
        label_ids = probabilities.argmax(dim=-1)
        confidences = probabilities.max(dim=-1).values

        chunk_entities: list[EntityAnnotation] = []
        for chunk_index in range(label_ids.shape[0]):
            chunk_offsets = [tuple(map(int, item)) for item in offsets[chunk_index].tolist()]
            chunk_entities.extend(
                bio_predictions_to_spans(
                    label_ids[chunk_index].tolist(),
                    chunk_offsets,
                    self.id_to_label,
                    raw_text,
                    confidences[chunk_index].tolist(),
                )
            )
        return merge_chunk_predictions(chunk_entities, raw_text)


def spans_overlap(left: EntityAnnotation, right: EntityAnnotation) -> bool:
    return left.start < right.end and right.start < left.end


def resolve_overlaps(entities: Iterable[EntityAnnotation], raw_text: str) -> list[EntityAnnotation]:
    ranked = sorted(
        entities,
        key=lambda entity: (-entity.confidence, -(entity.end - entity.start), entity.start, entity.type),
    )
    selected: list[EntityAnnotation] = []
    seen: set[tuple[int, int, str]] = set()
    for entity in ranked:
        entity.validate_offset(raw_text)
        key = (entity.start, entity.end, entity.type)
        if key in seen:
            continue
        if any(spans_overlap(entity, existing) for existing in selected):
            continue
        selected.append(entity)
        seen.add(key)
    return sorted(selected, key=lambda entity: (entity.start, entity.end, entity.type))


def merge_chunk_predictions(
    chunk_predictions: Iterable[EntityAnnotation], raw_text: str
) -> list[EntityAnnotation]:
    return resolve_overlaps(chunk_predictions, raw_text)


def refine_boundaries(entities: Iterable[EntityAnnotation], raw_text: str) -> list[EntityAnnotation]:
    refined: list[EntityAnnotation] = []
    for entity in entities:
        if '\n' in entity.text:
            start = entity.start
            for part in entity.text.split('\n'):
                part_len = len(part)
                if part_len > 0:
                    new_end = start + part_len
                    refined.append(replace(entity, text=part, position=(start, new_end)))
                start += part_len + 1
        else:
            refined.append(entity)

    final_refined: list[EntityAnnotation] = []
    for entity in refined:
        start, end = entity.position
        while start < end and (raw_text[start].isspace() or raw_text[start] in "-*•+"):
            start += 1
        while end > start and (raw_text[end - 1].isspace() or raw_text[end - 1] in "-*•+,;"):
            end -= 1
        if start < end:
            updated = replace(entity, text=raw_text[start:end], position=(start, end))
            updated.validate_offset(raw_text)
            final_refined.append(updated)
    return resolve_overlaps(final_refined, raw_text)
