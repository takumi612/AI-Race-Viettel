from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    min_score: float = 0.5
    min_margin: float = 0.05
    output_k: int = 1

    def __post_init__(self) -> None:
        if self.output_k < 1:
            raise ValueError("output_k must be at least one")
        if self.min_score < 0 or self.min_margin < 0:
            raise ValueError("candidate thresholds cannot be negative")


def apply_candidate_policy(
    ranked: Iterable[dict[str, Any]],
    policy: CandidatePolicy,
) -> list[str]:
    candidates = [item for item in ranked if item.get("candidate_id")]
    if not candidates:
        return []
    top_score = float(candidates[0].get("score", 0.0))
    if top_score < policy.min_score:
        return []
    if len(candidates) > 1:
        second_score = float(candidates[1].get("score", 0.0))
        if top_score - second_score < policy.min_margin:
            return []
    selected: list[str] = []
    for item in candidates[: policy.output_k]:
        output_id = str(item["candidate_id"])
        official_display_id = item.get("official_display_id")
        canonical_id = item.get("canonical_id")
        if official_display_id is not None and output_id != str(official_display_id):
            raise ValueError("candidate_id must retain the selected official_display_id")
        if canonical_id is not None and not str(canonical_id):
            raise ValueError("canonical_id must be non-empty when supplied")
        selected.append(output_id)
    return selected
