from collections.abc import Mapping
from dataclasses import dataclass, field
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"


def _require_number(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


def _require_int(name: str, value: object) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")


def _require_bool(name: str, value: object) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")


def _validate_unit_interval(name: str, value: float) -> None:
    _require_number(name, value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")


@dataclass(frozen=True)
class RetrievalConfig:
    alpha: float = 0.75
    internal_top_k: int = 20
    hierarchical_expansion: bool = False
    embedding_model_type: str = "BGE-M3"
    embedding_model_artifact: str | None = None
    icd_index_artifact: str | None = None
    rxnorm_index_artifact: str | None = None

    def __post_init__(self) -> None:
        _validate_unit_interval("alpha", self.alpha)
        _require_int("internal_top_k", self.internal_top_k)
        _require_bool("hierarchical_expansion", self.hierarchical_expansion)
        if self.internal_top_k < 1:
            raise ValueError("internal_top_k must be positive")
        if self.embedding_model_type not in {"BGE-M3", "SAPBERT"}:
            raise ValueError("embedding_model_type must be BGE-M3 or SAPBERT")
        for field, value in (
            ("embedding_model_artifact", self.embedding_model_artifact),
            ("icd_index_artifact", self.icd_index_artifact),
            ("rxnorm_index_artifact", self.rxnorm_index_artifact),
        ):
            if value is None:
                continue
            if (
                not isinstance(value, str)
                or not value.strip()
            ):
                raise ValueError(
                    f"{field} must be project-relative"
                )
            raw = value.strip()
            if (
                PurePosixPath(raw).is_absolute()
                or PureWindowsPath(raw).is_absolute()
                or PureWindowsPath(raw).drive
                or ".." in PurePosixPath(raw).parts
            ):
                raise ValueError(
                    f"{field} must be project-relative"
                )

    @property
    def bm25_weight(self) -> float:
        return self.alpha

    @property
    def semantic_weight(self) -> float:
        return 1.0 - self.alpha


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 384
    overlap_tokens: int = 64

    def __post_init__(self) -> None:
        _require_int("max_tokens", self.max_tokens)
        _require_int("overlap_tokens", self.overlap_tokens)
        if self.max_tokens < 1 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("chunk token bounds are invalid")


@dataclass(frozen=True)
class NERConfig:
    mode: str = "rule"
    model_artifact: str | None = None
    model_threshold: float = 0.70
    beta: float = 0.5
    ambiguity_margin: float = 0.15
    default_threshold: float = 0.70
    per_type_thresholds: Mapping[str, float] = field(
        default_factory=lambda: {
            "CHẨN_ĐOÁN": 0.72,
            "TRIỆU_CHỨNG": 0.76,
            "THUỐC": 0.72,
            "TÊN_XÉT_NGHIỆM": 0.72,
            "KẾT_QUẢ_XÉT_NGHIỆM": 0.75,
        }
    )

    def __post_init__(self) -> None:
        if self.mode not in {"rule", "model", "hybrid"}:
            raise ValueError("NER mode must be rule, model, or hybrid")
        if self.mode != "rule":
            if not isinstance(self.model_artifact, str) or not self.model_artifact.strip():
                raise ValueError("model_artifact is required for model/hybrid NER")
            artifact_path = Path(self.model_artifact)
            if artifact_path.is_absolute() or ".." in artifact_path.parts:
                raise ValueError("model_artifact must be project-relative")
        elif self.model_artifact is not None and not isinstance(self.model_artifact, str):
            raise ValueError("model_artifact must be a string or null")
        _validate_unit_interval("model_threshold", self.model_threshold)
        _require_number("beta", self.beta)
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        _validate_unit_interval("ambiguity_margin", self.ambiguity_margin)
        _validate_unit_interval("default_threshold", self.default_threshold)
        if not isinstance(self.per_type_thresholds, Mapping):
            raise ValueError("per_type_thresholds must be a mapping")
        frozen_thresholds = MappingProxyType(dict(self.per_type_thresholds))
        if any(not isinstance(key, str) for key in frozen_thresholds):
            raise ValueError("per-type threshold names must be strings")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
            for value in frozen_thresholds.values()
        ):
            raise ValueError("per-type thresholds must be in [0, 1]")
        object.__setattr__(self, "per_type_thresholds", frozen_thresholds)


@dataclass(frozen=True)
class AssertionConfig:
    negated_threshold: float = 0.70
    historical_threshold: float = 0.70
    family_threshold: float = 0.70

    def __post_init__(self) -> None:
        values = (
            self.negated_threshold,
            self.historical_threshold,
            self.family_threshold,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError("assertion thresholds must be in [0, 1]")


@dataclass(frozen=True)
class CandidateSelectionConfig:
    icd_min_score: float = 0.55
    rxnorm_min_score: float = 0.60
    top1_margin: float = 0.12
    top2_margin: float = 0.04
    load_historical_rxnorm: bool = False

    def __post_init__(self) -> None:
        values = (
            self.icd_min_score,
            self.rxnorm_min_score,
            self.top1_margin,
            self.top2_margin,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError("candidate thresholds must be in [0, 1]")
        _require_bool("load_historical_rxnorm", self.load_historical_rxnorm)


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    backend: str = "http"
    model_artifact: str | None = None
    max_new_tokens: int = 64
    timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        _require_bool("enabled", self.enabled)
        if self.backend not in {"http", "local_transformers"}:
            raise ValueError("reranker backend must be http or local_transformers")
        _require_int("max_new_tokens", self.max_new_tokens)
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.model_artifact is not None:
            if (
                not isinstance(self.model_artifact, str)
                or not self.model_artifact.strip()
            ):
                raise ValueError("model_artifact must be project-relative")
            raw = self.model_artifact.strip()
            if (
                PurePosixPath(raw).is_absolute()
                or PureWindowsPath(raw).is_absolute()
                or PureWindowsPath(raw).drive
                or ".." in PurePosixPath(raw).parts
            ):
                raise ValueError("model_artifact must be project-relative")
        if (
            self.enabled
            and self.backend == "local_transformers"
            and self.model_artifact is None
        ):
            raise ValueError(
                "model_artifact is required for local_transformers reranker"
            )
        _require_number("timeout_seconds", self.timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


def _strict_section(
    values: Mapping[str, object], section: str, allowed: set[str]
) -> dict[str, object]:
    raw_section = values.get(section, {})
    if not isinstance(raw_section, Mapping):
        raise ValueError(f"{section} must be a mapping")
    unknown = set(raw_section) - allowed
    if unknown:
        names = ", ".join(sorted(str(name) for name in unknown))
        raise ValueError(f"unknown {section} configuration keys: {names}")
    return dict(raw_section)


@dataclass(frozen=True)
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    assertion: AssertionConfig = field(default_factory=AssertionConfig)
    selection: CandidateSelectionConfig = field(default_factory=CandidateSelectionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "PipelineConfig":
        if not isinstance(values, Mapping):
            raise ValueError("pipeline configuration must be a mapping")
        sections = {"retrieval", "chunking", "ner", "assertion", "selection", "reranker"}
        unknown = set(values) - sections
        if unknown:
            names = ", ".join(sorted(str(name) for name in unknown))
            raise ValueError(f"unknown pipeline configuration keys: {names}")

        retrieval = _strict_section(
            values,
            "retrieval",
            {
                "alpha",
                "internal_top_k",
                "hierarchical_expansion",
                "embedding_model_type",
                "embedding_model_artifact",
                "icd_index_artifact",
                "rxnorm_index_artifact",
            },
        )
        chunking = _strict_section(
            values, "chunking", {"max_tokens", "overlap_tokens"}
        )
        ner = _strict_section(
            values,
            "ner",
            {
                "mode",
                "model_artifact",
                "model_threshold",
                "beta",
                "ambiguity_margin",
                "default_threshold",
                "per_type_thresholds",
            },
        )
        assertion = _strict_section(
            values,
            "assertion",
            {"negated_threshold", "historical_threshold", "family_threshold"},
        )
        selection = _strict_section(
            values,
            "selection",
            {
                "icd_min_score",
                "rxnorm_min_score",
                "top1_margin",
                "top2_margin",
                "load_historical_rxnorm",
            },
        )
        reranker = _strict_section(
            values,
            "reranker",
            {
                "enabled",
                "backend",
                "model_artifact",
                "max_new_tokens",
                "timeout_seconds",
            },
        )

        return cls(
            retrieval=RetrievalConfig(**retrieval),
            chunking=ChunkingConfig(**chunking),
            ner=NERConfig(**ner),
            assertion=AssertionConfig(**assertion),
            selection=CandidateSelectionConfig(**selection),
            reranker=RerankerConfig(**reranker),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval": {
                "alpha": self.retrieval.alpha,
                "internal_top_k": self.retrieval.internal_top_k,
                "hierarchical_expansion": self.retrieval.hierarchical_expansion,
                "embedding_model_type": self.retrieval.embedding_model_type,
                "embedding_model_artifact": self.retrieval.embedding_model_artifact,
                "icd_index_artifact": self.retrieval.icd_index_artifact,
                "rxnorm_index_artifact": self.retrieval.rxnorm_index_artifact,
            },
            "chunking": {
                "max_tokens": self.chunking.max_tokens,
                "overlap_tokens": self.chunking.overlap_tokens,
            },
            "ner": {
                "mode": self.ner.mode,
                "model_artifact": self.ner.model_artifact,
                "model_threshold": self.ner.model_threshold,
                "beta": self.ner.beta,
                "ambiguity_margin": self.ner.ambiguity_margin,
                "default_threshold": self.ner.default_threshold,
                "per_type_thresholds": dict(self.ner.per_type_thresholds),
            },
            "assertion": {
                "negated_threshold": self.assertion.negated_threshold,
                "historical_threshold": self.assertion.historical_threshold,
                "family_threshold": self.assertion.family_threshold,
            },
            "selection": {
                "icd_min_score": self.selection.icd_min_score,
                "rxnorm_min_score": self.selection.rxnorm_min_score,
                "top1_margin": self.selection.top1_margin,
                "top2_margin": self.selection.top2_margin,
                "load_historical_rxnorm": self.selection.load_historical_rxnorm,
            },
            "reranker": {
                "enabled": self.reranker.enabled,
                "backend": self.reranker.backend,
                "model_artifact": self.reranker.model_artifact,
                "max_new_tokens": self.reranker.max_new_tokens,
                "timeout_seconds": self.reranker.timeout_seconds,
            },
        }
