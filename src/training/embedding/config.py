"""Strict BGE-M3 LoRA configuration with BM25-first invariants."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be project-relative")
    raw = value.strip()
    if (
        PurePosixPath(raw).is_absolute()
        or PureWindowsPath(raw).is_absolute()
        or PureWindowsPath(raw).drive
        or ".." in PurePosixPath(raw).parts
    ):
        raise ValueError(f"{field} must be project-relative")
    return Path(*PurePosixPath(raw).parts)


@dataclass(frozen=True, slots=True)
class EmbeddingTrainingConfig:
    schema_version: int
    base_model: str
    dataset_dir: Path
    database: Path
    output_dir: Path
    seed: int
    num_train_epochs: float
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    fp16: bool
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    hard_negatives: int
    retrieval_top_k: int
    bm25_alpha: float
    recall_floor: float
    local_files_only: bool

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any]
    ) -> "EmbeddingTrainingConfig":
        if not isinstance(mapping, Mapping):
            raise ValueError("embedding config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - expected)
        missing = sorted(expected - set(mapping))
        if unknown:
            raise ValueError(f"unknown embedding config keys: {unknown}")
        if missing:
            raise ValueError(f"missing embedding config keys: {missing}")
        values = dict(mapping)
        for field in ("dataset_dir", "database", "output_dir"):
            values[field] = _relative_path(values[field], field)
        config = cls(**values)
        config._validate()
        return config

    def _validate(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("embedding schema_version must be 1")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("embedding base_model must be non-empty")
        integer_values = (
            self.seed,
            self.train_batch_size,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
            self.lora_rank,
            self.lora_alpha,
            self.hard_negatives,
            self.retrieval_top_k,
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in integer_values[1:]
            )
        ):
            raise ValueError("invalid embedding integer hyperparameter")
        numeric = (
            self.num_train_epochs,
            self.learning_rate,
            self.warmup_ratio,
            self.lora_dropout,
            self.bm25_alpha,
            self.recall_floor,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("embedding numeric hyperparameters must be finite")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        if not 0 <= self.warmup_ratio <= 1 or not 0 <= self.lora_dropout <= 1:
            raise ValueError("invalid warmup/dropout")
        if not 0.5 <= self.bm25_alpha <= 1:
            raise ValueError("BM25-first requires bm25_alpha in [0.5, 1]")
        if not 0 <= self.recall_floor <= 1:
            raise ValueError("recall_floor must be in [0, 1]")
        if self.retrieval_top_k < self.hard_negatives:
            raise ValueError("retrieval_top_k must cover hard_negatives")
        if not isinstance(self.fp16, bool) or not isinstance(
            self.gradient_checkpointing, bool
        ) or not isinstance(self.local_files_only, bool):
            raise ValueError("embedding boolean hyperparameters are invalid")

    @property
    def bm25_weight(self) -> float:
        return self.bm25_alpha

    @property
    def semantic_weight(self) -> float:
        return 1.0 - self.bm25_alpha

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        for field in ("dataset_dir", "database", "output_dir"):
            value[field] = getattr(self, field).as_posix()
        return value
