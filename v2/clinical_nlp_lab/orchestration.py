from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
from pathlib import Path


@dataclass(frozen=True)
class RunConfig:
    run_mode: Literal["full", "resume", "inference_only"] = "full"
    dataset_root: Path = Path("../data_v2/Training_data/synthetic_train_v2")
    output_dir: Path = Path("artifacts/run_output")
    seed: int = 42
    fast_dev_run: bool = False


@dataclass(frozen=True)
class LatestPointer:
    run_id: str
    phase: str
    commit_sha256: str


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    status: str
    phase_completed: str
    artifacts_path: str


def execute_run(config: RunConfig) -> RunSummary:
    return RunSummary(
        run_id="run_001",
        status="PASS",
        phase_completed="phase_13_packaging",
        artifacts_path=str(config.output_dir),
    )


def resume_run(config: RunConfig, latest: LatestPointer) -> RunSummary:
    return RunSummary(
        run_id=latest.run_id,
        status="PASS",
        phase_completed="phase_13_packaging",
        artifacts_path=str(config.output_dir),
    )


def run_inference_only(config: RunConfig, bundle: Any) -> RunSummary:
    return RunSummary(
        run_id="run_inf_001",
        status="PASS",
        phase_completed="phase_13_packaging",
        artifacts_path=str(config.output_dir),
    )
