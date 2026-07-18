"""Contextual, precision-first resolution of detected clinical mentions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from types import MappingProxyType
import unicodedata

from src.chunking.clinical_chunker import ClinicalChunk, ClinicalChunker
from src.config import NERConfig
from src.ner.lexicon_loader import ENTITY_TYPES
from src.ner.types import MentionCandidate, TypeDecision


_RULE_KEYS = frozenset(
    {
        "schema_version",
        "section_priors",
        "medication_signals",
        "laboratory_units",
        "dosage_units",
        "route_terms",
        "frequency_patterns",
        "generic_terms",
        "source_confidence",
        "weights",
    }
)
_WEIGHT_KEYS = frozenset(
    {"exact", "medication_signal", "laboratory_signal", "route_signal", "generic_penalty"}
)


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


def _finite_number(value: object, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"type rules {location} must be a finite number")
    return float(value)


def _string_terms(value: object, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"type rules {location} must be a non-empty list")
    normalized: set[str] = set()
    for index, term in enumerate(value):
        if not isinstance(term, str) or not term.strip():
            raise ValueError(f"type rules {location}[{index}] must be a non-empty string")
        normalized.add(_normalize(term))
    return tuple(sorted(normalized))


def _frozen_mapping(values: Mapping[str, float]) -> Mapping[str, float]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class FrequencyPattern:
    """A provenance-bearing token grammar with one numeric placeholder."""

    tokens: tuple[str, ...]
    source: str

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, (list, tuple)) or len(self.tokens) < 2:
            raise ValueError("frequency pattern tokens must be a sequence of at least two tokens")
        normalized: list[str] = []
        for index, token in enumerate(self.tokens):
            if not isinstance(token, str) or not token.strip():
                raise ValueError(f"frequency pattern tokens[{index}] must be a non-empty string")
            value = _normalize(token)
            if value != "{number}" and ("{" in value or "}" in value):
                raise ValueError("frequency pattern only supports the {number} placeholder")
            normalized.append(value)
        if normalized.count("{number}") != 1:
            raise ValueError("frequency pattern must contain exactly one {number} placeholder")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("frequency pattern source must be a non-empty string")
        object.__setattr__(self, "tokens", tuple(normalized))
        object.__setattr__(self, "source", self.source.strip())


def _load_frequency_patterns(value: object) -> tuple[FrequencyPattern, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("type rules frequency_patterns must be a non-empty list")
    selected: dict[tuple[str, ...], FrequencyPattern] = {}
    for index, raw in enumerate(value):
        location = f"frequency_patterns[{index}]"
        if not isinstance(raw, dict):
            raise ValueError(f"type rules {location} must be an object")
        unknown = set(raw) - {"pattern", "source"}
        if unknown:
            raise ValueError(f"type rules {location} has unknown keys: {', '.join(sorted(unknown))}")
        pattern = raw.get("pattern")
        source = raw.get("source")
        if not isinstance(pattern, str) or not pattern.strip():
            raise ValueError(f"type rules {location}.pattern must be a non-empty string")
        if not isinstance(source, str) or not source.strip():
            raise ValueError(f"type rules {location}.source must be a non-empty string")
        candidate = FrequencyPattern(tuple(pattern.split()), source)
        current = selected.get(candidate.tokens)
        if current is None or candidate.source < current.source:
            selected[candidate.tokens] = candidate
    return tuple(selected[key] for key in sorted(selected))


@dataclass(frozen=True)
class TypeRules:
    section_priors: Mapping[str, Mapping[str, float]]
    medication_signals: tuple[str, ...]
    laboratory_units: tuple[str, ...]
    dosage_units: tuple[str, ...]
    route_terms: tuple[str, ...]
    frequency_patterns: tuple[FrequencyPattern, ...]
    generic_terms: tuple[str, ...]
    source_confidence: Mapping[str, float]
    weights: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.section_priors, Mapping) or not self.section_priors:
            raise ValueError("type rules section_priors must be a non-empty mapping")
        frozen_sections: dict[str, Mapping[str, float]] = {}
        for section, raw_priors in self.section_priors.items():
            if not isinstance(section, str) or not section.strip():
                raise ValueError("type rules section_priors names must be non-empty strings")
            if not isinstance(raw_priors, Mapping) or not raw_priors:
                raise ValueError(f"type rules section_priors.{section} must be a non-empty mapping")
            priors: dict[str, float] = {}
            for entity_type, value in raw_priors.items():
                if entity_type not in ENTITY_TYPES:
                    raise ValueError(f"type rules section_priors.{section} has invalid type")
                priors[entity_type] = _finite_number(
                    value, f"section_priors.{section}.{entity_type}"
                )
            frozen_sections[section] = _frozen_mapping(priors)

        terms: dict[str, tuple[str, ...]] = {}
        for name in (
            "medication_signals",
            "laboratory_units",
            "dosage_units",
            "route_terms",
            "generic_terms",
        ):
            raw_terms = getattr(self, name)
            if not isinstance(raw_terms, (list, tuple)) or not raw_terms:
                raise ValueError(f"type rules {name} must be a non-empty sequence")
            normalized: set[str] = set()
            for index, term in enumerate(raw_terms):
                if not isinstance(term, str) or not term.strip():
                    raise ValueError(f"type rules {name}[{index}] must be a non-empty string")
                normalized.add(_normalize(term))
            terms[name] = tuple(sorted(normalized))

        if not isinstance(self.frequency_patterns, (list, tuple)) or not self.frequency_patterns:
            raise ValueError("type rules frequency_patterns must be a non-empty sequence")
        if any(not isinstance(pattern, FrequencyPattern) for pattern in self.frequency_patterns):
            raise ValueError("type rules frequency_patterns must contain FrequencyPattern values")
        frequency_patterns = tuple(
            sorted(self.frequency_patterns, key=lambda pattern: (pattern.tokens, pattern.source))
        )

        if not isinstance(self.source_confidence, Mapping) or not self.source_confidence:
            raise ValueError("type rules source_confidence must be a non-empty mapping")
        source_confidence: dict[str, float] = {}
        for source, value in self.source_confidence.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError("type rules source_confidence names must be non-empty strings")
            source_confidence[source] = _finite_number(value, f"source_confidence.{source}")

        if not isinstance(self.weights, Mapping):
            raise ValueError("type rules weights must be a mapping")
        if set(self.weights) != _WEIGHT_KEYS:
            raise ValueError(
                "type rules weights must contain exactly: " + ", ".join(sorted(_WEIGHT_KEYS))
            )
        weights = {
            name: _finite_number(self.weights[name], f"weights.{name}")
            for name in sorted(_WEIGHT_KEYS)
        }
        if weights["generic_penalty"] > 0:
            raise ValueError("type rules weights.generic_penalty must not be positive")

        object.__setattr__(self, "section_priors", MappingProxyType(dict(sorted(frozen_sections.items()))))
        for name, values in terms.items():
            object.__setattr__(self, name, values)
        object.__setattr__(self, "frequency_patterns", frequency_patterns)
        object.__setattr__(self, "source_confidence", _frozen_mapping(dict(sorted(source_confidence.items()))))
        object.__setattr__(self, "weights", _frozen_mapping(weights))

    @classmethod
    def load(cls, path: str | Path) -> "TypeRules":
        resource_path = Path(path)
        try:
            payload = json.loads(resource_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"type rules resource does not exist: {resource_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"type rules resource is invalid JSON: {error.msg}") from error
        if not isinstance(payload, dict):
            raise ValueError("type rules resource must be a JSON object")
        unknown = set(payload) - _RULE_KEYS
        if unknown:
            raise ValueError(f"type rules has unknown keys: {', '.join(sorted(unknown))}")
        missing = _RULE_KEYS - set(payload)
        if missing:
            raise ValueError(f"type rules missing keys: {', '.join(sorted(missing))}")
        version = payload["schema_version"]
        if isinstance(version, bool) or version != 1:
            raise ValueError("type rules schema_version must be integer 1")

        raw_sections = payload["section_priors"]
        if not isinstance(raw_sections, dict) or not raw_sections:
            raise ValueError("type rules section_priors must be a non-empty object")
        sections: dict[str, Mapping[str, float]] = {}
        for section, raw_priors in raw_sections.items():
            if not isinstance(section, str) or not section.strip():
                raise ValueError("type rules section_priors names must be non-empty strings")
            if not isinstance(raw_priors, dict) or not raw_priors:
                raise ValueError(f"type rules section_priors.{section} must be a non-empty object")
            priors: dict[str, float] = {}
            for entity_type, value in raw_priors.items():
                if entity_type not in ENTITY_TYPES:
                    raise ValueError(f"type rules section_priors.{section} has invalid type")
                priors[entity_type] = _finite_number(
                    value, f"section_priors.{section}.{entity_type}"
                )
            sections[section] = _frozen_mapping(priors)

        raw_source_confidence = payload["source_confidence"]
        if not isinstance(raw_source_confidence, dict) or not raw_source_confidence:
            raise ValueError("type rules source_confidence must be a non-empty object")
        source_confidence: dict[str, float] = {}
        for source, value in raw_source_confidence.items():
            if not isinstance(source, str) or not source.strip():
                raise ValueError("type rules source_confidence names must be non-empty strings")
            source_confidence[source] = _finite_number(value, f"source_confidence.{source}")

        raw_weights = payload["weights"]
        if not isinstance(raw_weights, dict):
            raise ValueError("type rules weights must be an object")
        unknown_weights = set(raw_weights) - _WEIGHT_KEYS
        missing_weights = _WEIGHT_KEYS - set(raw_weights)
        if unknown_weights or missing_weights:
            raise ValueError(
                "type rules weights must contain exactly: " + ", ".join(sorted(_WEIGHT_KEYS))
            )
        weights = {
            name: _finite_number(raw_weights[name], f"weights.{name}")
            for name in sorted(_WEIGHT_KEYS)
        }
        if weights["generic_penalty"] > 0:
            raise ValueError("type rules weights.generic_penalty must not be positive")

        return cls(
            MappingProxyType(dict(sorted(sections.items()))),
            _string_terms(payload["medication_signals"], "medication_signals"),
            _string_terms(payload["laboratory_units"], "laboratory_units"),
            _string_terms(payload["dosage_units"], "dosage_units"),
            _string_terms(payload["route_terms"], "route_terms"),
            _load_frequency_patterns(payload["frequency_patterns"]),
            _string_terms(payload["generic_terms"], "generic_terms"),
            _frozen_mapping(dict(sorted(source_confidence.items()))),
            _frozen_mapping(weights),
        )


class ContextualTypeResolver:
    """Score candidate types from resource-backed source, section, and local evidence."""

    def __init__(
        self,
        config: NERConfig | None = None,
        *,
        rules_path: str | Path | None = None,
        rules: TypeRules | None = None,
        source_statuses: Mapping[str, str] | None = None,
        source_lookup: Callable[[MentionCandidate, str], frozenset[str]] | None = None,
        chunker: ClinicalChunker | None = None,
    ) -> None:
        if rules is not None and rules_path is not None:
            raise ValueError("provide rules or rules_path, not both")
        self.config = config or NERConfig()
        default_path = Path(__file__).resolve().parents[1] / "resources" / "type_rules.json"
        self.rules = rules or TypeRules.load(rules_path or default_path)
        self._source_statuses = MappingProxyType(dict(source_statuses or {}))
        if any(status not in {"verified", "unverified"} for status in self._source_statuses.values()):
            raise ValueError("source statuses must be verified or unverified")
        if source_lookup is not None and not callable(source_lookup):
            raise ValueError("source_lookup must be callable")
        self._source_lookup = source_lookup
        self._chunker = chunker or ClinicalChunker()

    @staticmethod
    def _sigmoid(raw: float) -> float:
        if raw >= 0:
            return 1.0 / (1.0 + math.exp(-raw))
        exponent = math.exp(raw)
        return exponent / (1.0 + exponent)

    def _source_prior(self, sources: frozenset[str]) -> float:
        priors: list[float] = []
        fallback = self.rules.source_confidence.get("unverified", 0.0)
        for source in sources:
            if source in self.rules.source_confidence:
                priors.append(self.rules.source_confidence[source])
                continue
            status = self._source_statuses.get(source, "unverified")
            priors.append(self.rules.source_confidence.get(status, fallback))
        return max(priors, default=fallback)

    @staticmethod
    def _contains_term(context: str, terms: tuple[str, ...]) -> bool:
        return any(
            re.search(rf"(?<!\w){re.escape(term)}(?!\w)", context, re.UNICODE) is not None
            for term in terms
        )

    def resolve(
        self,
        mention: MentionCandidate,
        document: str,
        chunk: ClinicalChunk,
    ) -> TypeDecision:
        if not isinstance(document, str):
            raise TypeError("document must be a string")
        if not (0 <= mention.start < mention.end <= len(document)):
            raise ValueError("mention bounds are invalid")
        if document[mention.start : mention.end] != mention.text:
            raise ValueError("mention text does not match the document slice")
        if not (chunk.start <= mention.start and mention.end <= chunk.end):
            raise ValueError("mention is outside the chunk")

        normalized_mention = _normalize(mention.text)
        generic = normalized_mention in self.rules.generic_terms or (
            " " not in normalized_mention
            and any(normalized_mention == term for term in self.rules.generic_terms)
        )
        raw_scores: dict[str, float] = {}
        section_priors = self.rules.section_priors.get(chunk.section_type, {})
        for entity_type in sorted(mention.candidate_types):
            supporting_sources = (
                self._source_lookup(mention, entity_type)
                if self._source_lookup is not None
                else mention.sources
            )
            if not supporting_sources:
                supporting_sources = mention.sources
            context = _normalize(
                self._chunker.context_for_span(
                    document, chunk, mention.start, mention.end, entity_type
                )
            )
            raw = self._source_prior(supporting_sources) + section_priors.get(entity_type, 0.0)
            if mention.exact:
                raw += self.rules.weights["exact"]
            if generic:
                raw += self.rules.weights["generic_penalty"]
            if entity_type == "THUỐC":
                if self._contains_term(context, self.rules.medication_signals):
                    raw += self.rules.weights["medication_signal"]
                if self._contains_term(context, self.rules.route_terms):
                    raw += self.rules.weights["route_signal"]
            if entity_type in {"TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"} and self._contains_term(
                context, self.rules.laboratory_units
            ):
                raw += self.rules.weights["laboratory_signal"]
            raw_scores[entity_type] = raw

        scores = {name: self._sigmoid(value) for name, value in raw_scores.items()}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        best_type, best_score = ranked[0]
        threshold = self.config.per_type_thresholds.get(best_type, self.config.default_threshold)
        if best_score < threshold:
            return TypeDecision(None, best_score, scores, f"below threshold for {best_type}")
        if len(ranked) > 1 and best_score - ranked[1][1] < self.config.ambiguity_margin:
            return TypeDecision(None, best_score, scores, "ambiguous candidate type margin")
        return TypeDecision(best_type, best_score, scores, "accepted by contextual score")
