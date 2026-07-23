from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence
import numpy as np

from .schema import EntityAnnotation


@dataclass(frozen=True)
class CandidateFeatures:
    code: str
    system: Literal["ICD10", "RXNORM"]
    exact_alias: bool
    lexical_score: float
    semantic_score: float | None
    mention_candidate_similarity: float | None
    type_match: bool
    strength_match: bool | None
    dose_form_match: bool | None
    route_match: bool | None
    ambiguity_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "system": self.system,
            "exact_alias": self.exact_alias,
            "lexical_score": self.lexical_score,
            "semantic_score": self.semantic_score,
            "mention_candidate_similarity": self.mention_candidate_similarity,
            "type_match": self.type_match,
            "strength_match": self.strength_match,
            "dose_form_match": self.dose_form_match,
            "route_match": self.route_match,
            "ambiguity_count": self.ambiguity_count,
        }


@dataclass(frozen=True)
class HardNegative:
    code: str
    system: Literal["ICD10", "RXNORM"]
    negative_reason: str
    kb_hash: str
    generator_version: str


@dataclass(frozen=True)
class ScoredCandidate:
    mention_text: str
    entity_type: str
    code: str
    system: str
    score: float
    is_positive: bool


@dataclass(frozen=True)
class CandidateCalibrationArtifact:
    schema_id: str
    schema_version: int
    objective: str
    confidence_threshold: float
    policy: str
    kb_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "objective": self.objective,
            "confidence_threshold": self.confidence_threshold,
            "policy": self.policy,
            "kb_hash": self.kb_hash,
        }


def build_candidate_features(
    mention: EntityAnnotation,
    candidate: Any,  # KnowledgeBaseEntry or Dict
    context: str = "",
) -> CandidateFeatures:
    cand_code = getattr(candidate, "code", candidate.get("code", "")) if isinstance(candidate, (object, dict)) else str(candidate)
    cand_system = getattr(candidate, "system", candidate.get("system", "ICD10")) if isinstance(candidate, (object, dict)) else "ICD10"
    aliases = getattr(candidate, "aliases", candidate.get("aliases", [])) if isinstance(candidate, (object, dict)) else []

    exact_alias = mention.text.lower().strip() in [a.lower().strip() for a in aliases]
    type_match = True
    if mention.type == "DISEASE" and cand_system != "ICD10":
        type_match = False
    elif mention.type == "DRUG" and cand_system != "RXNORM":
        type_match = False

    return CandidateFeatures(
        code=cand_code,
        system=cand_system,
        exact_alias=exact_alias,
        lexical_score=1.0 if exact_alias else 0.7,
        semantic_score=None,
        mention_candidate_similarity=None,
        type_match=type_match,
        strength_match=None,
        dose_form_match=None,
        route_match=None,
        ambiguity_count=len(aliases),
    )


def generate_hard_negatives(
    positive: Any,
    retrieved: Sequence[Any],
    limit: int = 5,
    kb_hash: str = "",
) -> tuple[HardNegative, ...]:
    pos_code = getattr(positive, "code", positive.get("code", "")) if isinstance(positive, (object, dict)) else str(positive)
    pos_system = getattr(positive, "system", positive.get("system", "ICD10")) if isinstance(positive, (object, dict)) else "ICD10"

    negatives: list[HardNegative] = []
    for item in retrieved:
        code = getattr(item, "code", item.get("code", "")) if isinstance(item, (object, dict)) else str(item)
        system = getattr(item, "system", item.get("system", "ICD10")) if isinstance(item, (object, dict)) else "ICD10"

        if code != pos_code and system == pos_system:
            negatives.append(
                HardNegative(
                    code=code,
                    system=system,
                    negative_reason="retrieval_hard_negative",
                    kb_hash=kb_hash,
                    generator_version="1.0.0",
                )
            )
            if len(negatives) >= limit:
                break

    return tuple(negatives)


def fit_candidate_calibration(
    scored_examples: Sequence[ScoredCandidate],
    objective: Literal["precision_first"] = "precision_first",
    kb_hash: str = "",
) -> CandidateCalibrationArtifact:
    if not scored_examples or len(scored_examples) < 5:
        # Falling back to deterministic policy when insufficient ranking labels
        return CandidateCalibrationArtifact(
            schema_id="clinical_nlp.candidate_calibration",
            schema_version=1,
            objective=objective,
            confidence_threshold=0.65,
            policy="deterministic_fallback",
            kb_hash=kb_hash,
        )

    positives = [ex for ex in scored_examples if ex.is_positive]
    if not positives:
        return CandidateCalibrationArtifact(
            schema_id="clinical_nlp.candidate_calibration",
            schema_version=1,
            objective=objective,
            confidence_threshold=0.70,
            policy="deterministic_fallback",
            kb_hash=kb_hash,
        )

    # Precision-first threshold search
    scores = np.array([ex.score for ex in scored_examples])
    labels = np.array([ex.is_positive for ex in scored_examples])

    best_th = 0.65
    best_prec = -1.0

    for th in np.linspace(0.5, 0.9, 10):
        preds = scores >= th
        if preds.sum() == 0:
            continue
        prec = (preds & labels).sum() / preds.sum()
        if prec >= 0.90 and prec >= best_prec:
            best_prec = prec
            best_th = float(th)

    return CandidateCalibrationArtifact(
        schema_id="clinical_nlp.candidate_calibration",
        schema_version=1,
        objective=objective,
        confidence_threshold=best_th,
        policy="calibrated_ranker",
        kb_hash=kb_hash,
    )
