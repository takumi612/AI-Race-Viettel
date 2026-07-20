"""Deterministic precision-first selection from scored retrieval candidates."""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Mapping, Sequence

from src.config import CandidateSelectionConfig, PipelineConfig


def _get(value: object, name: str, default: object) -> object:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _selection_config(config: object | None) -> object:
    if config is None:
        return PipelineConfig().selection
    nested = _get(config, "selection", None)
    return nested if nested is not None else config


def _entity_kind(entity_type: object) -> str | None:
    """Map both proper Vietnamese labels and legacy mojibake labels."""
    text = unicodedata.normalize("NFC", str(entity_type or "")).strip().upper()
    compact = text.replace(" ", "").replace("-", "").replace("_", "")
    if compact in {"THUỐC", "THUOC", "DRUG", "MEDICATION"}:
        return "rxnorm"
    if compact in {"CHẨNĐOÁN", "CHANDOAN", "DIAGNOSIS", "DIAGNOSE"}:
        return "icd"
    return None


def _candidate_code(candidate: object) -> str:
    if isinstance(candidate, Mapping):
        value = candidate.get("code", "")
    else:
        value = getattr(candidate, "code", candidate)
    return str(value).strip()


def _candidate_score(candidate: object) -> float:
    if isinstance(candidate, Mapping):
        value = candidate.get("fusion_score", candidate.get("score", 0.0))
    else:
        value = getattr(candidate, "fusion_score", getattr(candidate, "score", 0.0))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


class CandidateSelector:
    """Apply clinical validity before calibrated score and margin gates."""

    def __init__(self, config: object | None = None):
        self.config = _selection_config(config)

    def select(
        self,
        entity_type: str,
        ranked: Sequence[object] | None,
        is_valid,
    ) -> list[str]:
        kind = _entity_kind(entity_type)
        if kind is None or not ranked:
            return []

        minimum_name = "icd_min_score" if kind == "icd" else "rxnorm_min_score"
        minimum = _get(self.config, minimum_name, None)
        top1_margin = _get(self.config, "top1_margin", 0.12)
        top2_margin = _get(self.config, "top2_margin", 0.04)
        try:
            minimum = float(minimum)
            top1_margin = float(top1_margin)
            top2_margin = float(top2_margin)
        except (TypeError, ValueError):
            return []

        # Validation deliberately happens before score gates: a low-scoring
        # candidate can still be the only clinically valid answer.
        valid = []
        seen: set[str] = set()
        for candidate in ranked:
            code = _candidate_code(candidate)
            score = _candidate_score(candidate)
            if not code or code in seen or not math.isfinite(score):
                continue
            seen.add(code)
            try:
                accepted = bool(is_valid(code))
            except Exception:
                accepted = False
            if accepted:
                valid.append((code, score))

        above_minimum = [(code, score) for code, score in valid if score >= minimum]
        if not above_minimum:
            return []
        if len(above_minimum) == 1:
            return [above_minimum[0][0]]

        first, second = above_minimum[0], above_minimum[1]
        margin = first[1] - second[1]
        if margin >= top1_margin:
            return [first[0]]
        if margin <= top2_margin:
            return [first[0], second[0]]
        return [first[0]]


__all__ = ["CandidateSelector"]
