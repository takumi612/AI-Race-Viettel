import math
from collections.abc import Iterable
from numbers import Real

from src.retrieval.types import ComponentCandidate, RetrievedCandidate


def _normalize_code(code: object) -> str:
    return str(code).strip().upper()


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
        raise ValueError("alpha must be a finite number in [0, 1]")
    alpha_value = float(alpha)
    if not math.isfinite(alpha_value) or not 0.0 <= alpha_value <= 1.0:
        raise ValueError("alpha must be a finite number in [0, 1]")
    return alpha_value


def _prepare_candidates(
    candidates: Iterable[ComponentCandidate], valid_codes: set[str] | None = None
) -> dict[str, ComponentCandidate]:
    prepared: dict[str, ComponentCandidate] = {}
    for candidate in candidates:
        code = _normalize_code(candidate.code)
        if not code or (valid_codes is not None and code not in valid_codes):
            continue

        score = float(candidate.score)
        if not math.isfinite(score):
            continue
        normalized = ComponentCandidate(code=code, score=score, rank=int(candidate.rank))
        current = prepared.get(code)
        if current is None or (normalized.score, -normalized.rank) > (
            current.score,
            -current.rank,
        ):
            prepared[code] = normalized
    return prepared


def minmax_scores(candidates: Iterable[ComponentCandidate]) -> dict[str, float]:
    """Return query-local min-max scores, with equal values mapped to one."""

    prepared = _prepare_candidates(candidates)
    if not prepared:
        return {}

    values = [candidate.score for candidate in prepared.values()]
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        return {code: 1.0 for code in prepared}

    scale = maximum - minimum
    return {
        code: (candidate.score - minimum) / scale
        for code, candidate in prepared.items()
    }


def fuse_candidates(
    bm25: Iterable[ComponentCandidate],
    semantic: Iterable[ComponentCandidate],
    alpha: float,
    valid_codes: Iterable[str] | None = None,
) -> list[RetrievedCandidate]:
    """Fuse query-local BM25 and semantic scores with weights summing to one."""

    bm25_weight = _validate_alpha(alpha)
    semantic_weight = 1.0 - bm25_weight
    normalized_valid_codes = (
        {_normalize_code(code) for code in valid_codes if _normalize_code(code)}
        if valid_codes is not None
        else None
    )
    bm25_candidates = _prepare_candidates(bm25, normalized_valid_codes)
    semantic_candidates = _prepare_candidates(semantic, normalized_valid_codes)
    bm25_scores = minmax_scores(bm25_candidates.values())
    semantic_scores = minmax_scores(semantic_candidates.values())

    fused = []
    for code in bm25_candidates.keys() | semantic_candidates.keys():
        bm25_candidate = bm25_candidates.get(code)
        semantic_candidate = semantic_candidates.get(code)
        bm25_score = bm25_scores.get(code, 0.0)
        semantic_score = semantic_scores.get(code, 0.0)
        fused.append(
            RetrievedCandidate(
                code=code,
                fusion_score=bm25_weight * bm25_score + semantic_weight * semantic_score,
                bm25_score=bm25_score,
                semantic_score=semantic_score,
                bm25_rank=bm25_candidate.rank if bm25_candidate else None,
                semantic_rank=semantic_candidate.rank if semantic_candidate else None,
            )
        )

    missing_rank = math.inf
    return sorted(
        fused,
        key=lambda candidate: (
            -candidate.fusion_score,
            candidate.bm25_rank
            if candidate.bm25_rank is not None
            else missing_rank,
            candidate.semantic_rank
            if candidate.semantic_rank is not None
            else missing_rank,
            candidate.code,
        ),
    )
