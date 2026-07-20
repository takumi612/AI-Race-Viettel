"""Precision-first metrics and deterministic model-selection policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class PrecisionMetrics:
    precision: float
    recall: float
    f_beta: float
    beta: float
    true_positives: int
    false_positives: int
    false_negatives: int

    def __post_init__(self) -> None:
        for name, value in (
            ("precision", self.precision),
            ("recall", self.recall),
            ("f_beta", self.f_beta),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be finite and in [0, 1]")
        if (
            isinstance(self.beta, bool)
            or not isinstance(self.beta, (int, float))
            or not math.isfinite(self.beta)
            or self.beta <= 0
        ):
            raise ValueError("beta must be a positive finite number")
        for name, value in (
            ("true_positives", self.true_positives),
            ("false_positives", self.false_positives),
            ("false_negatives", self.false_negatives),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_mapping(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f_beta": self.f_beta,
            "beta": self.beta,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


def exact_fbeta(
    gold: Iterable[Hashable],
    predicted: Iterable[Hashable],
    *,
    beta: float = 0.5,
) -> PrecisionMetrics:
    if (
        isinstance(beta, bool)
        or not isinstance(beta, (int, float))
        or not math.isfinite(beta)
        or beta <= 0
    ):
        raise ValueError("beta must be a positive finite number")
    gold_set = set(gold)
    predicted_set = set(predicted)
    true_positives = len(gold_set & predicted_set)
    false_positives = len(predicted_set - gold_set)
    false_negatives = len(gold_set - predicted_set)
    precision = (
        true_positives / (true_positives + false_positives)
        if predicted_set
        else (1.0 if not gold_set else 0.0)
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if gold_set
        else 1.0
    )
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    f_beta = (
        (1.0 + beta_squared) * precision * recall / denominator
        if denominator
        else 0.0
    )
    return PrecisionMetrics(
        precision=float(precision),
        recall=float(recall),
        f_beta=float(f_beta),
        beta=float(beta),
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def choose_precision_first(
    candidates: Mapping[str, PrecisionMetrics],
    *,
    recall_floor: float,
    score_tolerance: float = 0.005,
) -> str:
    if not candidates:
        raise ValueError("candidates must not be empty")
    for name, value in (
        ("recall_floor", recall_floor),
        ("score_tolerance", score_tolerance),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            raise ValueError(f"{name} must be finite and in [0, 1]")
    eligible = {
        name: metrics
        for name, metrics in candidates.items()
        if metrics.recall >= recall_floor
    }
    if not eligible:
        raise ValueError("no candidate satisfies the recall floor")

    highest_score = max(metrics.f_beta for metrics in eligible.values())
    near_best = {
        name: metrics
        for name, metrics in eligible.items()
        if highest_score - metrics.f_beta <= score_tolerance
    }
    return min(
        near_best,
        key=lambda name: (
            -near_best[name].precision,
            -near_best[name].f_beta,
            -near_best[name].recall,
            name,
        ),
    )
