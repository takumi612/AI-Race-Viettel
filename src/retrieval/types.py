from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentCandidate:
    """One ranked candidate returned by a retrieval component."""

    code: str
    score: float
    rank: int


@dataclass(frozen=True)
class RetrievedCandidate:
    """A candidate scored by normalized BM25/semantic fusion."""

    code: str
    fusion_score: float
    bm25_score: float
    semantic_score: float
    bm25_rank: int | None
    semantic_rank: int | None
