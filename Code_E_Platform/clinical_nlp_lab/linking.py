from __future__ import annotations

import re
from collections import defaultdict
from difflib import SequenceMatcher
from typing import Any, Iterable

from .schema import EntityAnnotation
from .text import normalize_alias


STRENGTH_PATTERN = re.compile(r"(?iu)\b\d+(?:[.,-]\d+)?\s*(?:mg|mcg|µg|g|ml|mL|unit|đơn\s+vị)\b")
ROUTE_PATTERN = re.compile(r"(?iu)\b(?:po|iv|im|sc|oral|uống|tiêm)\b")
FREQUENCY_PATTERN = re.compile(r"(?iu)\b(?:daily|bid|tid|qid|q\d+h|qam|qhs|prn)\b")
DOSE_FORM_PATTERN = re.compile(r"(?iu)\b(?:tablet|capsule|suspension|injection|viên|ống)\b")


def token_jaccard(left: str, right: str) -> float:
    left_tokens = set(normalize_alias(left).split())
    right_tokens = set(normalize_alias(right).split())
    if not left_tokens and not right_tokens:
        return 1.0
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 0.0


def character_ngram_similarity(left: str, right: str, ngram_range: tuple[int, int] = (2, 5)) -> float:
    left_normalized = normalize_alias(left).replace(" ", "_")
    right_normalized = normalize_alias(right).replace(" ", "_")
    left_ngrams = {
        left_normalized[index:index + size]
        for size in range(ngram_range[0], ngram_range[1] + 1)
        for index in range(max(0, len(left_normalized) - size + 1))
    }
    right_ngrams = {
        right_normalized[index:index + size]
        for size in range(ngram_range[0], ngram_range[1] + 1)
        for index in range(max(0, len(right_normalized) - size + 1))
    }
    if not left_ngrams and not right_ngrams:
        return 1.0
    union = left_ngrams | right_ngrams
    return len(left_ngrams & right_ngrams) / len(union) if union else 0.0


def parse_medication_attributes(text: str) -> dict[str, str | None]:
    strength = STRENGTH_PATTERN.search(text)
    route = ROUTE_PATTERN.search(text)
    frequency = FREQUENCY_PATTERN.search(text)
    dose_form = DOSE_FORM_PATTERN.search(text)
    first_attribute_start = min(
        [match.start() for match in (strength, route, frequency, dose_form) if match] or [len(text)]
    )
    drug_name = text[:first_attribute_start].strip(" ,;:-")
    return {
        "drug_name": drug_name or text.strip(),
        "strength": strength.group(0) if strength else None,
        "route": route.group(0) if route else None,
        "frequency": frequency.group(0) if frequency else None,
        "dose_form": dose_form.group(0) if dose_form else None,
    }


class LexicalCandidateIndex:
    def __init__(self, records: Iterable[dict[str, Any]], name: str) -> None:
        self.name = name
        self.records: dict[str, dict[str, Any]] = {}
        self.alias_to_ids: dict[str, list[str]] = defaultdict(list)
        self.token_to_ids: dict[str, set[str]] = defaultdict(set)
        self.normalized_names: dict[str, list[str]] = {}
        self.candidate_token_sets: dict[str, set[str]] = {}
        for record in records:
            candidate_id = str(record["candidate_id"])
            self.records[candidate_id] = record
            aliases = list(record.get("aliases") or [])
            if not aliases:
                fallback = record.get("canonical_name") or record.get("name_vi") or record.get("name_en")
                aliases = [fallback] if fallback else []
            for alias in aliases:
                normalized = normalize_alias(str(alias))
                if not normalized:
                    continue
                self.normalized_names.setdefault(candidate_id, []).append(normalized)
                self.alias_to_ids[normalized].append(candidate_id)
                for token in set(normalized.split()):
                    if len(token) >= 3:
                        self.token_to_ids[token].add(candidate_id)
            self.candidate_token_sets[candidate_id] = set(
                token for name in self.normalized_names.get(candidate_id, []) for token in name.split()
            )

    def _record_names(self, candidate_id: str) -> list[str]:
        record = self.records[candidate_id]
        names = list(record.get("aliases") or [])
        for key in ("canonical_name", "name_vi", "name_en"):
            if record.get(key):
                names.append(str(record[key]))
        return list(dict.fromkeys(names))

    def retrieve(self, query: str, top_k: int = 20) -> list[dict[str, Any]]:
        normalized_query = normalize_alias(query)
        if not normalized_query:
            return []
        exact_ids = self.alias_to_ids.get(normalized_query, [])
        scored: dict[str, tuple[float, str]] = {}
        for candidate_id in exact_ids:
            scored[candidate_id] = (1.0, "exact")

        pool: set[str] = set()
        for token in normalized_query.split():
            pool.update(self.token_to_ids.get(token, set()))
        if not pool:
            prefix = normalized_query[:4]
            pool.update(
                candidate_id
                for alias, candidate_ids in self.alias_to_ids.items()
                if alias.startswith(prefix)
                for candidate_id in candidate_ids
            )
        if len(pool) > 1000:
            query_tokens = set(normalized_query.split())
            pool = set(
                candidate_id
                for candidate_id, _overlap in sorted(
                    ((candidate_id, len(query_tokens & self.candidate_token_sets.get(candidate_id, set()))) for candidate_id in pool),
                    key=lambda item: (-item[1], item[0]),
                )[:1000]
            )

        for candidate_id in pool:
            if candidate_id in scored:
                continue
            best_score = 0.0
            for normalized_alias in self.normalized_names.get(candidate_id, [])[:6]:
                sequence = SequenceMatcher(None, normalized_query, normalized_alias).ratio()
                overlap = token_jaccard(normalized_query, normalized_alias)
                char_score = character_ngram_similarity(normalized_query, normalized_alias)
                score = 0.45 * sequence + 0.25 * overlap + 0.30 * char_score
                best_score = max(best_score, score)
            scored[candidate_id] = (best_score, "lexical")

        ranked = sorted(scored.items(), key=lambda item: (-item[1][0], item[0]))[:top_k]
        return [
            {
                "candidate_id": candidate_id,
                "score": round(score, 6),
                "method": method,
                "name": self._record_names(candidate_id)[0] if self._record_names(candidate_id) else "",
            }
            for candidate_id, (score, method) in ranked
        ]


class EntityLinker:
    def __init__(
        self,
        icd10_index: LexicalCandidateIndex,
        rxnorm_index: LexicalCandidateIndex,
        top_k: int = 20,
        output_k: int = 1,
        minimum_score: float = 0.50,
    ) -> None:
        self.icd10_index = icd10_index
        self.rxnorm_index = rxnorm_index
        self.top_k = top_k
        self.output_k = output_k
        self.minimum_score = minimum_score

    def retrieve(self, entity: EntityAnnotation) -> tuple[list[str], list[dict[str, Any]]]:
        if entity.type == "DISEASE":
            index = self.icd10_index
            query = entity.mention_head or entity.text
        elif entity.type == "DRUG":
            index = self.rxnorm_index
            parsed = parse_medication_attributes(entity.text)
            query = entity.mention_head or str(parsed["drug_name"])
        else:
            return [], []

        existing = [candidate_id for candidate_id in entity.candidates if candidate_id in index.records]
        if existing:
            exact_ranked = [
                {
                    "candidate_id": candidate_id,
                    "score": 1.0,
                    "method": "detector_exact_phrase",
                    "name": index._record_names(candidate_id)[0] if index._record_names(candidate_id) else "",
                }
                for candidate_id in existing[: self.output_k]
            ]
            return existing[: self.output_k], exact_ranked
        ranked = index.retrieve(query, top_k=self.top_k)
        combined: list[str] = []
        for candidate_id in existing:
            if candidate_id not in combined:
                combined.append(candidate_id)
        for item in ranked:
            if item["score"] >= self.minimum_score and item["candidate_id"] not in combined:
                combined.append(item["candidate_id"])
        return combined[: self.output_k], ranked
