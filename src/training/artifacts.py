"""Fingerprint-locked run state and atomic artifact promotion."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)


_RUN_STATUSES = frozenset({"candidate", "validated", "locked"})


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _sha256_string(value: Any, field: str) -> str:
    normalized = _required_string(value, field)
    if len(normalized) != 64 or any(
        character not in "0123456789abcdefABCDEF" for character in normalized
    ):
        raise ValueError(f"{field} must be a SHA-256 hex digest")
    return normalized.casefold()


@dataclass(frozen=True, slots=True)
class TrainingRunState:
    schema_version: int
    task: str
    base_model: str
    config_sha256: str
    dataset_build_id: str
    dataset_manifest_sha256: str
    seed: int
    status: str
    global_step: int
    checkpoint: str | None

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "TrainingRunState":
        if not isinstance(mapping, Mapping):
            raise ValueError("run state must be a mapping")
        expected = {
            "schema_version",
            "task",
            "base_model",
            "config_sha256",
            "dataset_build_id",
            "dataset_manifest_sha256",
            "seed",
            "status",
            "global_step",
            "checkpoint",
        }
        if set(mapping) != expected:
            raise ValueError("run state keys do not match schema")
        schema_version = mapping["schema_version"]
        if schema_version != 1 or isinstance(schema_version, bool):
            raise ValueError("run state schema_version must be 1")
        status = _required_string(mapping["status"], "status")
        if status not in _RUN_STATUSES:
            raise ValueError(f"invalid run status: {status}")
        seed = mapping["seed"]
        global_step = mapping["global_step"]
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            isinstance(global_step, bool)
            or not isinstance(global_step, int)
            or global_step < 0
        ):
            raise ValueError("global_step must be a non-negative integer")
        checkpoint = mapping["checkpoint"]
        if checkpoint is not None:
            checkpoint = _required_string(checkpoint, "checkpoint")
            path = Path(checkpoint)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("checkpoint must be run-relative")

        return cls(
            schema_version=1,
            task=_required_string(mapping["task"], "task"),
            base_model=_required_string(mapping["base_model"], "base_model"),
            config_sha256=_sha256_string(
                mapping["config_sha256"], "config_sha256"
            ),
            dataset_build_id=_required_string(
                mapping["dataset_build_id"], "dataset_build_id"
            ),
            dataset_manifest_sha256=_sha256_string(
                mapping["dataset_manifest_sha256"],
                "dataset_manifest_sha256",
            ),
            seed=seed,
            status=status,
            global_step=global_step,
            checkpoint=checkpoint,
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid4().hex}.tmp"
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if json.loads(temporary.read_text(encoding="utf-8")) != dict(value):
            raise ValueError(f"atomic JSON round-trip failed: {path}")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_state(path: Path) -> TrainingRunState:
    if not path.is_file():
        raise FileNotFoundError(f"run state is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid run state JSON: {path}") from exc
    return TrainingRunState.from_mapping(value)


def _dataset_identity(path: Path) -> tuple[str, str]:
    if not path.is_file():
        raise ValueError(f"dataset manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid dataset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("dataset manifest must be an object")
    build_id = _required_string(value.get("build_id"), "dataset build_id")
    return build_id, sha256_file(path)


def start_or_resume_run(
    run_dir: str | Path,
    *,
    task: str,
    base_model: str,
    config: Mapping[str, Any],
    dataset_manifest: str | Path,
    seed: int,
    resume: bool,
) -> TrainingRunState:
    if not isinstance(config, Mapping):
        raise ValueError("training config must be a mapping")
    if not isinstance(resume, bool):
        raise ValueError("resume must be a boolean")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    run_path = Path(run_dir)
    state_path = run_path / "run.json"
    build_id, manifest_sha256 = _dataset_identity(Path(dataset_manifest))
    expected = TrainingRunState(
        schema_version=1,
        task=_required_string(task, "task"),
        base_model=_required_string(base_model, "base_model"),
        config_sha256=stable_json_sha256(dict(config)),
        dataset_build_id=build_id,
        dataset_manifest_sha256=manifest_sha256,
        seed=seed,
        status="candidate",
        global_step=0,
        checkpoint=None,
    )

    if state_path.exists():
        if not resume:
            raise FileExistsError(f"run already exists; resume explicitly: {run_path}")
        actual = _read_state(state_path)
        for field in (
            "task",
            "base_model",
            "config_sha256",
            "dataset_build_id",
            "dataset_manifest_sha256",
            "seed",
        ):
            if getattr(actual, field) != getattr(expected, field):
                raise ValueError(
                    f"resume identity mismatch for {field}: "
                    f"{getattr(actual, field)!r} != {getattr(expected, field)!r}"
                )
        return actual
    if resume:
        raise FileNotFoundError(f"cannot resume missing run: {run_path}")
    if run_path.exists() and any(run_path.iterdir()):
        raise FileExistsError(f"run directory is not empty: {run_path}")

    run_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(state_path, expected.to_mapping())
    return expected


def update_run_state(
    run_dir: str | Path,
    state: TrainingRunState,
    *,
    global_step: int,
    checkpoint: str | Path | None,
) -> TrainingRunState:
    run_path = Path(run_dir).resolve()
    current = _read_state(run_path / "run.json")
    if current != state:
        raise ValueError("provided run state is stale")
    if (
        isinstance(global_step, bool)
        or not isinstance(global_step, int)
        or global_step < state.global_step
    ):
        raise ValueError("global_step cannot decrease")

    checkpoint_value: str | None = None
    if checkpoint is not None:
        checkpoint_path = Path(checkpoint).resolve()
        try:
            relative = checkpoint_path.relative_to(run_path)
        except ValueError as exc:
            raise ValueError("checkpoint must be inside run directory") from exc
        if not checkpoint_path.exists():
            raise ValueError(f"checkpoint does not exist: {checkpoint_path}")
        checkpoint_value = relative.as_posix()

    updated = replace(
        state,
        global_step=global_step,
        checkpoint=checkpoint_value,
    )
    _atomic_json(run_path / "run.json", updated.to_mapping())
    return updated


def promote_run(
    run_dir: str | Path,
    artifact_dir: str | Path,
    state: TrainingRunState,
    *,
    metrics: Mapping[str, Any],
    status: str,
) -> Path:
    if status not in {"validated", "locked"}:
        raise ValueError("promotion status must be validated or locked")
    if not isinstance(metrics, Mapping) or not metrics:
        raise ValueError("metrics must be a non-empty mapping")
    run_path = Path(run_dir).resolve()
    artifact_path = Path(artifact_dir).resolve()
    if artifact_path.exists():
        raise FileExistsError(f"artifact already exists: {artifact_path}")
    current = _read_state(run_path / "run.json")
    if current != state:
        raise ValueError("provided run state is stale")

    final_state = replace(state, status=status)
    _atomic_json(run_path / "run.json", final_state.to_mapping())
    _atomic_json(run_path / "metrics.json", dict(metrics))
    artifact_files = sorted(
        path
        for path in run_path.rglob("*")
        if path.is_file() and path.name != "artifact_manifest.json"
    )
    artifact_sha256 = fingerprint_files(artifact_files, run_path)
    manifest = {
        "schema_version": 1,
        "status": status,
        "run": final_state.to_mapping(),
        "metrics": dict(metrics),
        "artifact_sha256": artifact_sha256,
    }
    _atomic_json(run_path / "artifact_manifest.json", manifest)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    run_path.rename(artifact_path)
    return artifact_path
