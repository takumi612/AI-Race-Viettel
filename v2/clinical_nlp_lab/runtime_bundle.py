"""Runtime adapters used after the final Kaggle checkpoint is published."""

from __future__ import annotations

from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .assertion_model import AssertionHead, build_frozen_assertion_adapter
from .candidate_policy import CandidatePolicy
from .inference import FinalModelBundle, SpanProposal
from .ner import TransformerNERDetector
from .text import normalize_alias, normalize_with_mapping
from .entity_types import ENTITY_TYPE_TO_ID


_UNSAFE_SINGLE_TOKEN_ALIASES = {
    "benh",
    "bệnh",
    "co",
    "da",
    "dau",
    "ho",
    "khong",
    "không",
    "khoa",
    "luc",
    "lúc",
    "mang",
    "phau",
    "phẫu",
    "the",
    "thể",
    "thuat",
    "thuật",
    "tin",
    "va",
    "và",
    "xet",
    "xét",
}

_NON_DRUG_CLINICAL_TERMS = {
    "creatinine",
    "glucose",
    "protein",
    "prothrombin",
}


def is_safe_recovery_alias(alias: str, entity_type: str) -> bool:
    tokens = alias.split()
    if not tokens or not all(any(character.isalnum() for character in token) for token in tokens):
        return False
    if len(tokens) == 1:
        token = tokens[0]
        if entity_type == "DRUG" and token in _NON_DRUG_CLINICAL_TERMS:
            return False
        return len(token) >= 5 and token not in _UNSAFE_SINGLE_TOKEN_ALIASES
    return len(alias) >= 6 and any(
        token not in _UNSAFE_SINGLE_TOKEN_ALIASES for token in tokens
    )


def _candidate_display_name(record: dict[str, Any], alias: str) -> str:
    for key in ("name", "canonical_name", "name_vi", "name_en"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return alias


class KBFirstRecovery:
    """Exact raw-text alias recovery with ranked candidates."""

    def __init__(self, icd10_records: Iterable[dict[str, Any]], rxnorm_records: Iterable[dict[str, Any]]) -> None:
        grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = {}
        for entity_type, system, records in (
            ("DISEASE", "ICD10", icd10_records),
            ("DRUG", "RXNORM", rxnorm_records),
        ):
            for record in records:
                candidate_id = str(record.get("candidate_id", ""))
                aliases = list(record.get("aliases") or [])
                for alias in aliases:
                    normalized = normalize_alias(str(alias))
                    if candidate_id and is_safe_recovery_alias(normalized, entity_type):
                        grouped.setdefault((entity_type, system, normalized), {})[
                            candidate_id
                        ] = record
        self._entries: list[
            tuple[str, str, str, tuple[dict[str, Any], ...]]
        ] = [
            (
                entity_type,
                system,
                alias,
                tuple(records_by_id[key] for key in sorted(records_by_id)),
            )
            for (entity_type, system, alias), records_by_id in grouped.items()
        ]
        self._entries.sort(key=lambda item: (-len(item[2]), item[0], item[2]))
        self._exact_entries: dict[tuple[str, str], set[int]] = defaultdict(set)
        self._token_entries: dict[tuple[str, str], set[int]] = defaultdict(set)
        for index, (entity_type, _system, alias, _records) in enumerate(self._entries):
            self._exact_entries[(entity_type, alias)].add(index)
            for token in alias.split():
                if len(token) >= 4 and token not in _UNSAFE_SINGLE_TOKEN_ALIASES:
                    self._token_entries[(entity_type, token)].add(index)

    def rank_candidates(
        self, entity_type: str, mention: str, limit: int = 20
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("candidate ranking limit must be positive")
        normalized = normalize_alias(mention)
        if not normalized or entity_type not in {"DISEASE", "DRUG"}:
            return []
        entry_indices = set(self._exact_entries.get((entity_type, normalized), ()))
        if not entry_indices:
            for token in normalized.split():
                if len(token) >= 4:
                    entry_indices.update(
                        self._token_entries.get((entity_type, token), ())
                    )

        mention_tokens = set(normalized.split())
        ranked_by_id: dict[str, dict[str, Any]] = {}
        for index in entry_indices:
            _entry_type, system, alias, records = self._entries[index]
            alias_tokens = set(alias.split())
            union = mention_tokens | alias_tokens
            token_score = len(mention_tokens & alias_tokens) / len(union) if union else 0.0
            sequence_score = SequenceMatcher(None, normalized, alias).ratio() * 0.95
            score = 1.0 if normalized == alias else max(token_score, sequence_score)
            for record in records:
                candidate_id = str(record["candidate_id"])
                previous = ranked_by_id.get(candidate_id)
                if previous is not None and float(previous["score"]) >= score:
                    continue
                ranked_by_id[candidate_id] = {
                    "candidate_id": candidate_id,
                    "official_display_id": candidate_id,
                    "canonical_id": candidate_id,
                    "name": _candidate_display_name(record, alias),
                    "canonical_name": str(record.get("canonical_name", "")),
                    "score": round(float(score), 6),
                    "system": system,
                }
        return sorted(
            ranked_by_id.values(),
            key=lambda item: (-float(item["score"]), str(item["candidate_id"])),
        )[:limit]

    def scan_raw_text(self, raw_text: str) -> list[SpanProposal]:
        normalized_view = normalize_with_mapping(raw_text)
        normalized_text = normalized_view.model_text
        proposals: list[SpanProposal] = []
        seen: set[tuple[int, int, str]] = set()
        for entity_type, system, alias, records in self._entries:
            start = normalized_text.find(alias)
            while start >= 0:
                end = start + len(alias)
                left_is_word = start > 0 and normalized_text[start - 1].isalnum()
                right_is_word = end < len(normalized_text) and normalized_text[end].isalnum()
                if left_is_word or right_is_word:
                    start = normalized_text.find(alias, start + 1)
                    continue
                raw_start = normalized_view.model_to_raw[start]
                raw_end = normalized_view.model_to_raw[end - 1] + 1
                raw_candidate = raw_text[raw_start:raw_end]
                if normalize_alias(raw_candidate) == alias and (raw_start, raw_end, entity_type) not in seen:
                    ranked = tuple(
                        {
                            "candidate_id": str(record["candidate_id"]),
                            "official_display_id": str(record["candidate_id"]),
                            "canonical_id": str(record["candidate_id"]),
                            "name": _candidate_display_name(record, alias),
                            "score": 1.0,
                            "system": system,
                        }
                        for record in records
                    )
                    proposals.append(SpanProposal(raw_candidate, entity_type, raw_start, raw_end, 0.99, "kb_first", ranked_candidates=ranked))
                    seen.add((raw_start, raw_end, entity_type))
                start = normalized_text.find(alias, start + 1)
        return proposals


class AssertionRuntimePredictor:
    """Load the frozen encoder/head artifact and predict three assertion axes."""

    def __init__(self, checkpoint: str | Path, head_dir: str | Path, device: str | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
        self.encoder = AutoModel.from_pretrained(str(checkpoint)).to(self.device)
        self.encoder.eval()
        self.head_dir = Path(head_dir)
        binding_payload = __import__("json").loads((self.head_dir / "assertion_binding.json").read_text(encoding="utf-8"))
        self.thresholds = tuple(__import__("json").loads((self.head_dir / "assertion_thresholds.json").read_text(encoding="utf-8"))["thresholds"])
        self.type_ids = ENTITY_TYPE_TO_ID
        self.adapter = build_frozen_assertion_adapter(
            self.encoder,
            hidden_dim=int(self.encoder.config.hidden_size),
            encoder_hash=str(binding_payload["encoder_hash"]),
            tokenizer_hash=str(binding_payload["tokenizer_hash"]),
        ).to(self.device)
        self.adapter.head.load_state_dict(torch.load(self.head_dir / "assertion_head.pt", map_location=self.device))
        self.adapter.eval()

    def predict(self, raw_text: str, entities: list[Any]) -> dict[tuple[int, int, str], list[str]]:
        results: dict[tuple[int, int, str], list[str]] = {}
        for entity in entities:
            if entity.type in {"LAB_NAME", "LAB_RESULT"}:
                results[(entity.start, entity.end, entity.type)] = []
                continue
            left = max(0, entity.start - 160)
            right = min(len(raw_text), entity.end + 160)
            context = raw_text[left:right]
            encoded = self.tokenizer(context, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=512)
            offsets = encoded.pop("offset_mapping")[0].tolist()
            overlapping = [index for index, (start, end) in enumerate(offsets) if end > start and start < entity.end - left and end > entity.start - left]
            if not overlapping:
                results[(entity.start, entity.end, entity.type)] = []
                continue
            inputs = {key: value.to(self.device) for key, value in encoded.items()}
            spans = self.torch.tensor([[0, min(overlapping), max(overlapping) + 1]], dtype=self.torch.long, device=self.device)
            types = self.torch.tensor([self.type_ids.get(entity.type, 0)], dtype=self.torch.long, device=self.device)
            with self.torch.inference_mode():
                logits = self.adapter(inputs["input_ids"], inputs["attention_mask"], spans, types)[0]
            probabilities = self.torch.sigmoid(logits).detach().cpu().tolist()
            labels = [axis for axis, probability, threshold in zip(("isNegated", "isHistorical", "isFamily"), probabilities, self.thresholds) if probability >= threshold]
            results[(entity.start, entity.end, entity.type)] = labels
        return results


def load_final_model_bundle(checkpoint: str | Path, head_dir: str | Path, icd10_records: Iterable[dict[str, Any]], rxnorm_records: Iterable[dict[str, Any]], candidate_policy: CandidatePolicy, qwen_reranker: Any | None = None) -> FinalModelBundle:
    detector = TransformerNERDetector(checkpoint, max_length=512, stride=128)
    assertion = AssertionRuntimePredictor(checkpoint, head_dir)
    return FinalModelBundle(
        ner_model=detector,
        tokenizer=assertion.tokenizer,
        assertion_model=assertion,
        candidate_policy=candidate_policy,
        kb_linker=KBFirstRecovery(icd10_records, rxnorm_records),
        qwen_reranker=qwen_reranker,
    )


__all__ = ["AssertionRuntimePredictor", "KBFirstRecovery", "load_final_model_bundle"]
