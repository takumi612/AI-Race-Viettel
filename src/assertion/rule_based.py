"""Deterministic, section-aware assertion analysis for clinical entities."""

from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    _MODULE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.normcase(sys.path[0]) == os.path.normcase(_MODULE_DIRECTORY):
        sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(os.path.dirname(_MODULE_DIRECTORY)))

from collections.abc import Mapping
from functools import lru_cache
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any
import unicodedata

from src.config import AssertionConfig, PipelineConfig


_ASSERTION_ORDER = ("isNegated", "isHistorical", "isFamily")
_RULE_GROUPS = {
    "negation_cues",
    "compound_negation_cues",
    "negation_exclusions",
    "historical_cues",
    "family_cues",
    "assertion_terminators",
    "scope_boundaries",
    "section_priors",
    "patient_return_cues",
    "confidence_weights",
}
_TOP_LEVEL_KEYS = {"version", "provenance", *_RULE_GROUPS}
_CONFIDENCE_KEYS = {
    "negation_cue",
    "historical_cue",
    "family_cue",
    "historical_section_prior",
    "family_section_prior",
}
_DEFAULT_RULES_PATH = Path(__file__).resolve().parents[1] / "resources" / "assertion_rules.json"


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _require_object(value: object, location: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"assertion rule resource {location} must be an object")
    return value


def _require_exact_keys(
    value: Mapping[str, object], expected: set[str], location: str
) -> None:
    unknown = set(value) - expected
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(
            f"assertion rule resource {location} has unknown keys: {names}"
        )
    missing = expected - set(value)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"assertion rule resource {location} is missing keys: {names}")


def _require_string_list(
    value: object, location: str, *, single_character: bool = False
) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"assertion rule resource {location} must be a non-empty list"
        )
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or (
            not single_character and not item.strip()
        ):
            raise ValueError(
                f"assertion rule resource {location}[{index}] must be a non-empty string"
            )
        if single_character and len(item) != 1:
            raise ValueError(
                f"assertion rule resource {location}[{index}] must be one character"
            )
        normalized.append(_normalize(item if single_character else item.strip()))
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"assertion rule resource {location} contains duplicate cues")
    return normalized


def _validate_payload(payload: object) -> dict[str, object]:
    rules = _require_object(payload, "root")
    _require_exact_keys(rules, _TOP_LEVEL_KEYS, "root")
    if isinstance(rules["version"], bool) or rules["version"] != 1:
        raise ValueError("assertion rule resource version must be 1")

    provenance = _require_object(rules["provenance"], "provenance")
    _require_exact_keys(provenance, _RULE_GROUPS, "provenance")
    for group, raw_entry in provenance.items():
        entry = _require_object(raw_entry, f"provenance.{group}")
        _require_exact_keys(entry, {"source"}, f"provenance.{group}")
        source = entry["source"]
        if not isinstance(source, str) or not source.strip():
            raise ValueError(
                f"assertion rule resource provenance.{group}.source must be a non-empty string"
            )

    for group in (
        "negation_cues",
        "compound_negation_cues",
        "negation_exclusions",
        "historical_cues",
        "family_cues",
        "patient_return_cues",
    ):
        _require_string_list(rules[group], group)

    terminators = _require_object(
        rules["assertion_terminators"], "assertion_terminators"
    )
    _require_exact_keys(
        terminators, set(_ASSERTION_ORDER), "assertion_terminators"
    )
    for assertion, cues in terminators.items():
        _require_string_list(cues, f"assertion_terminators.{assertion}")

    boundaries = _require_object(rules["scope_boundaries"], "scope_boundaries")
    _require_exact_keys(
        boundaries,
        {"max_prefix_characters", "sentence_characters", "bullet_markers"},
        "scope_boundaries",
    )
    maximum = boundaries["max_prefix_characters"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum < 1:
        raise ValueError(
            "assertion rule resource scope_boundaries.max_prefix_characters must be a positive integer"
        )
    _require_string_list(
        boundaries["sentence_characters"],
        "scope_boundaries.sentence_characters",
        single_character=True,
    )
    _require_string_list(boundaries["bullet_markers"], "scope_boundaries.bullet_markers")

    weights = _require_object(rules["confidence_weights"], "confidence_weights")
    _require_exact_keys(weights, _CONFIDENCE_KEYS, "confidence_weights")
    for name, value in weights.items():
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(
                f"assertion rule resource confidence_weights.{name} must be in [0, 1]"
            )

    section_priors = _require_object(rules["section_priors"], "section_priors")
    if not section_priors:
        raise ValueError("assertion rule resource section_priors must not be empty")
    required_sections = {
        "pre_admission_medications",
        "past_medical_history",
        "family_history",
    }
    missing_sections = required_sections - set(section_priors)
    if missing_sections:
        names = ", ".join(sorted(missing_sections))
        raise ValueError(
            f"assertion rule resource section_priors is missing sections: {names}"
        )
    for section_type, raw_prior in section_priors.items():
        if not isinstance(section_type, str) or not section_type.strip():
            raise ValueError(
                "assertion rule resource section_priors keys must be non-empty strings"
            )
        prior = _require_object(raw_prior, f"section_priors.{section_type}")
        _require_exact_keys(
            prior, {"assertion", "weight"}, f"section_priors.{section_type}"
        )
        assertion = prior["assertion"]
        weight = prior["weight"]
        if assertion not in _ASSERTION_ORDER:
            raise ValueError(
                f"assertion rule resource section_priors.{section_type}.assertion is invalid"
            )
        if not isinstance(weight, str) or weight not in weights:
            raise ValueError(
                f"assertion rule resource section_priors.{section_type}.weight is invalid"
            )

    return rules


@lru_cache(maxsize=None)
def _load_rules(path: str) -> Mapping[str, Any]:
    rule_path = Path(path)
    try:
        payload = json.loads(rule_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"assertion rule resource does not exist: {rule_path}") from error
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"assertion rule resource is invalid UTF-8 JSON: {rule_path}") from error
    return _freeze(_validate_payload(payload))


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = r"\s+".join(re.escape(token) for token in phrase.split())
    return re.compile(rf"(?<!\w){escaped}(?!\w)", re.UNICODE)


def _matches(text: str, phrases: tuple[str, ...]) -> list[re.Match[str]]:
    found = [match for phrase in phrases for match in _phrase_pattern(phrase).finditer(text)]
    return sorted(found, key=lambda match: (match.start(), match.end()))


class AssertionAnalyzer:
    """Assign calibrated assertion scores within sentence, bullet, and section scope."""

    def __init__(
        self,
        config: PipelineConfig | AssertionConfig | None = None,
        *,
        rules_path: str | Path | None = None,
    ) -> None:
        if config is None:
            self._config = AssertionConfig()
        elif isinstance(config, PipelineConfig):
            self._config = config.assertion
        elif isinstance(config, AssertionConfig):
            self._config = config
        else:
            raise TypeError("config must be PipelineConfig or AssertionConfig")

        path = Path(rules_path) if rules_path is not None else _DEFAULT_RULES_PATH
        self._rules = _load_rules(str(path.resolve()))
        self._normalized_lists = {
            name: tuple(_normalize(value) for value in self._rules[name])
            for name in (
                "negation_cues",
                "compound_negation_cues",
                "negation_exclusions",
                "historical_cues",
                "family_cues",
                "patient_return_cues",
            )
        }
        self._terminators = {
            assertion: tuple(_normalize(value) for value in cues)
            for assertion, cues in self._rules["assertion_terminators"].items()
        }

    @property
    def rules(self) -> Mapping[str, Any]:
        """Return the cached, recursively immutable rule resource."""

        return self._rules

    @staticmethod
    def _validate_contract(
        full_text: object,
        start_idx: object,
        end_idx: object,
        section_type: object,
        header_text: object,
    ) -> None:
        if not isinstance(full_text, str):
            raise TypeError("full_text must be a string")
        if (
            isinstance(start_idx, bool)
            or isinstance(end_idx, bool)
            or not isinstance(start_idx, int)
            or not isinstance(end_idx, int)
            or not 0 <= start_idx < end_idx <= len(full_text)
        ):
            raise ValueError("entity span must satisfy 0 <= start_idx < end_idx <= len(full_text)")
        if not isinstance(section_type, str):
            raise TypeError("section_type must be a string")
        if not isinstance(header_text, str):
            raise TypeError("header_text must be a string")

    def _scope_prefix(self, full_text: str, start_idx: int) -> str:
        boundaries = self._rules["scope_boundaries"]
        maximum = boundaries["max_prefix_characters"]
        prefix = _normalize(full_text[max(0, start_idx - maximum) : start_idx])
        boundary = max(
            (prefix.rfind(character) for character in boundaries["sentence_characters"]),
            default=-1,
        )
        scope = prefix[boundary + 1 :]
        stripped = scope.lstrip()
        for marker in sorted(
            boundaries["bullet_markers"], key=lambda item: (-len(item), item)
        ):
            if stripped.startswith(marker) and (
                len(stripped) == len(marker) or stripped[len(marker)].isspace()
            ):
                return stripped[len(marker) :].lstrip()
        return scope

    @staticmethod
    def _redact(text: str, phrases: tuple[str, ...]) -> str:
        redacted = text
        for match in reversed(_matches(text, phrases)):
            redacted = redacted[: match.start()] + " " * (match.end() - match.start()) + redacted[match.end() :]
        return redacted

    def _after_last_terminator(
        self,
        text: str,
        assertion: str,
        protected: tuple[tuple[int, int], ...] = (),
    ) -> str:
        end = 0
        for match in _matches(text, self._terminators[assertion]):
            if any(start <= match.start() and match.end() <= stop for start, stop in protected):
                continue
            end = max(end, match.end())
        return text[end:]

    def _section_prefix(
        self, full_text: str, start_idx: int, header_text: str
    ) -> str:
        normalized = _normalize(full_text[:start_idx])
        header = _normalize(header_text.strip())
        if header:
            header_start = normalized.rfind(header)
            if header_start != -1:
                return normalized[header_start + len(header) :]
        maximum = self._rules["scope_boundaries"]["max_prefix_characters"]
        return normalized[max(0, len(normalized) - maximum) :]

    def score(
        self,
        full_text: str,
        start_idx: int,
        end_idx: int,
        section_type: str = "unknown",
        header_text: str = "",
    ) -> dict[str, float]:
        """Return assertion confidence scores in the public deterministic order."""

        self._validate_contract(
            full_text, start_idx, end_idx, section_type, header_text
        )
        scores = {assertion: 0.0 for assertion in _ASSERTION_ORDER}
        weights = self._rules["confidence_weights"]
        scope = self._scope_prefix(full_text, start_idx)

        negation_scope = self._redact(
            scope, self._normalized_lists["negation_exclusions"]
        )
        compound_matches = _matches(
            negation_scope, self._normalized_lists["compound_negation_cues"]
        )
        protected = tuple((match.start(), match.end()) for match in compound_matches)
        negation_tail = self._after_last_terminator(
            negation_scope, "isNegated", protected
        )
        negation_phrases = (
            self._normalized_lists["compound_negation_cues"]
            + self._normalized_lists["negation_cues"]
        )
        if _matches(negation_tail, negation_phrases):
            scores["isNegated"] = float(weights["negation_cue"])

        historical_tail = self._after_last_terminator(scope, "isHistorical")
        if _matches(historical_tail, self._normalized_lists["historical_cues"]):
            scores["isHistorical"] = float(weights["historical_cue"])

        family_tail = self._after_last_terminator(scope, "isFamily")
        if _matches(family_tail, self._normalized_lists["family_cues"]):
            scores["isFamily"] = float(weights["family_cue"])

        prior = self._rules["section_priors"].get(section_type)
        if prior is not None:
            assertion = prior["assertion"]
            section_prefix = self._section_prefix(full_text, start_idx, header_text)
            patient_returned = assertion == "isFamily" and bool(
                _matches(
                    section_prefix, self._normalized_lists["patient_return_cues"]
                )
            )
            if not patient_returned:
                scores[assertion] = max(
                    scores[assertion], float(weights[prior["weight"]])
                )

        return scores

    def analyze(
        self,
        full_text: str,
        start_idx: int,
        end_idx: int,
        section_type: str = "unknown",
        header_text: str = "",
    ) -> list[str]:
        """Return assertions whose calibrated scores meet configured thresholds."""

        scores = self.score(
            full_text, start_idx, end_idx, section_type, header_text
        )
        thresholds = {
            "isNegated": self._config.negated_threshold,
            "isHistorical": self._config.historical_threshold,
            "isFamily": self._config.family_threshold,
        }
        return [
            assertion
            for assertion in _ASSERTION_ORDER
            if scores[assertion] >= thresholds[assertion]
        ]


if __name__ == "__main__":
    analyzer = AssertionAnalyzer()
    examples = (
        ("Bệnh nhân không có ho", "ho", {}, "isNegated"),
        (
            "Thuốc trước khi nhập viện\n- metoprolol 25mg po bid",
            "metoprolol 25mg po bid",
            {
                "section_type": "pre_admission_medications",
                "header_text": "Thuốc trước khi nhập viện",
            },
            "isHistorical",
        ),
    )
    for text, entity, context, expected in examples:
        start = text.index(entity)
        result = analyzer.analyze(text, start, start + len(entity), **context)
        assert expected in result, (text, result)
    print("Assertion Rule-based Test: Passed")
