"""Immutable contracts shared by NER mention detection and type resolution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType


@dataclass(frozen=True)
class MentionCandidate:
    text: str
    start: int
    end: int
    candidate_types: frozenset[str]
    sources: frozenset[str]
    exact: bool

    def __post_init__(self) -> None:
        if not isinstance(self.text, str) or not self.text:
            raise ValueError("mention text must be a non-empty string")
        if isinstance(self.start, bool) or not isinstance(self.start, int):
            raise ValueError("mention start must be an integer")
        if isinstance(self.end, bool) or not isinstance(self.end, int) or self.end <= self.start:
            raise ValueError("mention end must be greater than start")
        if not isinstance(self.exact, bool):
            raise ValueError("mention exact must be a boolean")
        candidate_types = frozenset(self.candidate_types)
        sources = frozenset(self.sources)
        if not candidate_types or any(not isinstance(value, str) or not value for value in candidate_types):
            raise ValueError("mention candidate_types must contain non-empty strings")
        if not sources or any(not isinstance(value, str) or not value for value in sources):
            raise ValueError("mention sources must contain non-empty strings")
        object.__setattr__(self, "candidate_types", candidate_types)
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True)
class TypeDecision:
    entity_type: str | None
    confidence: float
    scores: Mapping[str, float]
    reason: str

    def __post_init__(self) -> None:
        if self.entity_type is not None and not isinstance(self.entity_type, str):
            raise ValueError("entity_type must be a string or None")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not isinstance(self.scores, Mapping):
            raise ValueError("scores must be a mapping")
        copied = dict(self.scores)
        if any(not isinstance(key, str) or not key for key in copied):
            raise ValueError("score names must be non-empty strings")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in copied.values()
        ):
            raise ValueError("scores must be finite numbers in [0, 1]")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(self, "scores", MappingProxyType(copied))
