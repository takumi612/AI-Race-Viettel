from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    min_score: float = 0.5
    min_margin: float = 0.05
    output_k: int = 1
    calibration_kb_hash: str | None = None

    def __post_init__(self) -> None:
        if self.output_k < 1:
            raise ValueError("output_k must be at least one")
        if self.min_score < 0 or self.min_margin < 0:
            raise ValueError("candidate thresholds cannot be negative")

    @classmethod
    def from_calibration(cls, calibration: Any, *, min_margin: float = 0.05) -> "CandidatePolicy":
        """Bind a persisted calibration artifact to the runtime policy.

        The artifact is intentionally accepted by protocol (rather than importing its
        concrete type) so policy loading remains independent from training modules.
        """
        if isinstance(calibration, Mapping):
            get_value = calibration.get
        else:
            get_value = lambda key, default=None: getattr(calibration, key, default)
        threshold = float(get_value("confidence_threshold", -1.0))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("calibration confidence_threshold must be in [0, 1]")
        objective = str(get_value("objective", ""))
        if objective != "precision_first":
            raise ValueError(f"unsupported candidate calibration objective: {objective!r}")
        kb_hash = get_value("kb_hash", None)
        if kb_hash is not None and (not isinstance(kb_hash, str) or len(kb_hash) != 64):
            raise ValueError("calibration kb_hash must be a 64-character digest")
        return cls(
            min_score=threshold,
            min_margin=float(min_margin),
            output_k=1,
            calibration_kb_hash=kb_hash,
        )

    def validate_kb_hash(self, kb_hash: str) -> None:
        """Fail closed when a policy is paired with a different knowledge base."""
        if self.calibration_kb_hash is not None and self.calibration_kb_hash != kb_hash:
            raise ValueError("candidate calibration KB fingerprint mismatch")


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
