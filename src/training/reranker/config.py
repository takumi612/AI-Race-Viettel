"""Strict free-Colab Qwen2.5-7B QLoRA configuration."""

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
class RerankerTrainingConfig:
    schema_version: int
    base_model: str
    dataset_dir: Path
    candidate_dataset: Path
    database: Path
    output_dir: Path
    seed: int
    num_train_epochs: float
    learning_rate: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    max_seq_length: int
    max_new_tokens: int
    warmup_ratio: float
    fp16: bool
    gradient_checkpointing: bool
    lora_rank: int
    lora_alpha: int
    lora_dropout: float
    load_in_4bit: bool
    bnb_quant_type: str
    bnb_use_double_quant: bool
    local_files_only: bool

    @classmethod
    def from_mapping(
        cls, mapping: Mapping[str, Any]
    ) -> "RerankerTrainingConfig":
        if not isinstance(mapping, Mapping):
            raise ValueError("reranker config must be a mapping")
        expected = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - expected)
        missing = sorted(expected - set(mapping))
        if unknown:
            raise ValueError(f"unknown reranker config keys: {unknown}")
        if missing:
            raise ValueError(f"missing reranker config keys: {missing}")
        values = dict(mapping)
        for field in (
            "dataset_dir",
            "candidate_dataset",
            "database",
            "output_dir",
        ):
            values[field] = _relative_path(values[field], field)
        config = cls(**values)
        config._validate()
        return config

    def _validate(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("reranker schema_version must be 1")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("reranker base_model must be non-empty")
        positive_integers = (
            self.train_batch_size,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
            self.max_seq_length,
            self.max_new_tokens,
            self.lora_rank,
            self.lora_alpha,
        )
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in positive_integers
            )
        ):
            raise ValueError("invalid reranker integer hyperparameter")
        if self.train_batch_size != 1 or self.eval_batch_size != 1:
            raise ValueError("free-Colab QLoRA requires micro-batch one")
        numeric = (
            self.num_train_epochs,
            self.learning_rate,
            self.warmup_ratio,
            self.lora_dropout,
        )
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric
        ):
            raise ValueError("reranker numeric hyperparameters must be finite")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        if not 0 <= self.warmup_ratio <= 1 or not 0 <= self.lora_dropout <= 1:
            raise ValueError("invalid reranker warmup/dropout")
        boolean_values = (
            self.fp16,
            self.gradient_checkpointing,
            self.load_in_4bit,
            self.bnb_use_double_quant,
            self.local_files_only,
        )
        if any(not isinstance(value, bool) for value in boolean_values):
            raise ValueError("invalid reranker boolean hyperparameter")
        if not self.load_in_4bit:
            raise ValueError("Qwen reranker must use 4-bit QLoRA")
        if self.bnb_quant_type.casefold() != "nf4":
            raise ValueError("Qwen reranker bnb_quant_type must be nf4")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        for field in (
            "dataset_dir",
            "candidate_dataset",
            "database",
            "output_dir",
        ):
            value[field] = getattr(self, field).as_posix()
        return value
