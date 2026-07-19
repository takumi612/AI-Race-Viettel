"""Strict configuration for Colab-sized XLM-R token classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Mapping


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a project-relative path")
    raw = value.strip()
    if (
        PureWindowsPath(raw).is_absolute()
        or PureWindowsPath(raw).drive
        or PurePosixPath(raw).is_absolute()
        or ".." in PurePosixPath(raw).parts
    ):
        raise ValueError(f"{field} must be a project-relative path")
    return Path(*PurePosixPath(raw).parts)


@dataclass(frozen=True, slots=True)
class NERTrainingConfig:
    schema_version: int
    base_model: str
    dataset_dir: Path
    output_dir: Path
    seed: int = 20260719
    max_length: int = 384
    stride: int = 64
    num_train_epochs: float = 3.0
    learning_rate: float = 2e-5
    train_batch_size: int = 4
    eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    fp16: bool = True
    gradient_checkpointing: bool = True
    early_stopping_patience: int = 2
    save_steps: int = 100
    logging_steps: int = 20
    recall_floor: float = 0.60
    local_files_only: bool = False

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "NERTrainingConfig":
        if not isinstance(mapping, Mapping):
            raise ValueError("NER training config must be a mapping")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(mapping) - allowed)
        if unknown:
            raise ValueError(f"unknown NER training config keys: {unknown}")
        required = {"schema_version", "base_model", "dataset_dir", "output_dir"}
        missing = sorted(required - set(mapping))
        if missing:
            raise ValueError(f"missing NER training config keys: {missing}")
        values = dict(mapping)
        values["dataset_dir"] = _relative_path(values["dataset_dir"], "dataset_dir")
        values["output_dir"] = _relative_path(values["output_dir"], "output_dir")
        config = cls(**values)
        config._validate()
        return config

    def _validate(self) -> None:
        if self.schema_version != 1 or isinstance(self.schema_version, bool):
            raise ValueError("NER schema_version must be 1")
        if not isinstance(self.base_model, str) or not self.base_model.strip():
            raise ValueError("base_model must be non-empty")
        integer_fields = {
            "seed": self.seed,
            "max_length": self.max_length,
            "stride": self.stride,
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "early_stopping_patience": self.early_stopping_patience,
            "save_steps": self.save_steps,
            "logging_steps": self.logging_steps,
        }
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in integer_fields.values()
        ):
            raise ValueError("NER integer hyperparameters must be non-negative")
        if (
            self.max_length < 8
            or not 0 <= self.stride < self.max_length
            or min(
                self.train_batch_size,
                self.eval_batch_size,
                self.gradient_accumulation_steps,
                self.save_steps,
                self.logging_steps,
            )
            < 1
        ):
            raise ValueError("invalid NER batch/chunk/step configuration")
        numeric_fields = {
            "num_train_epochs": self.num_train_epochs,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "recall_floor": self.recall_floor,
        }
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            for value in numeric_fields.values()
        ):
            raise ValueError("NER numeric hyperparameters must be finite")
        if self.num_train_epochs <= 0 or self.learning_rate <= 0:
            raise ValueError("epochs and learning rate must be positive")
        if not 0 <= self.weight_decay or not 0 <= self.warmup_ratio <= 1:
            raise ValueError("invalid weight decay or warmup ratio")
        if not 0 <= self.recall_floor <= 1:
            raise ValueError("recall_floor must be in [0, 1]")
        for field, value in (
            ("fp16", self.fp16),
            ("gradient_checkpointing", self.gradient_checkpointing),
            ("local_files_only", self.local_files_only),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"{field} must be a boolean")

    def to_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        value["dataset_dir"] = self.dataset_dir.as_posix()
        value["output_dir"] = self.output_dir.as_posix()
        return value
