from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Callable, Literal, Mapping
import uuid

from .curriculum import plan_curriculum


PHASES = (
    "phase_01_preflight",
    "phase_02_resolve_sources",
    "phase_03_inventory_models",
    "phase_04_build_metadata",
    "phase_05_build_splits",
    "phase_06_prepare_training_contract",
    "phase_07_stage1",
    "phase_08_stage2",
    "phase_09_stage3",
    "phase_10_final_fit",
    "phase_11_fit_heads",
    "phase_12_inference",
    "phase_13_packaging",
)

PhaseRunner = Callable[["RunConfig", str, Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class RunConfig:
    run_mode: Literal["full", "resume", "inference_only"] = "full"
    dataset_root: Path = Path("../data_v2/Training_data/synthetic_train_v2")
    output_dir: Path = Path("artifacts/run_output")
    artifact_dir: Path = Path("artifacts")
    artifact_source_dir: Path | None = None
    input_source: Path = Path("input.zip")
    model_source: str = "xlm-roberta-base"
    config_path: Path | None = None
    expected_gpu_count: int = 2
    use_distributed: bool = True
    seed: int = 42
    fast_dev_run: bool = False
    run_id: str | None = None
    dataset_fingerprint: str = "unbound"
    config_fingerprint: str = "default"
    phase_runners: Mapping[str, PhaseRunner] = field(default_factory=dict, repr=False, compare=False)


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
    mode: str = "full"
    phases: tuple[str, ...] = ()


@dataclass
class RunSession:
    config: RunConfig
    run_id: str
    run_dir: Path
    phases: tuple[str, ...]
    context: dict[str, Any]
    completed: list[str] = field(default_factory=list)


class OrchestrationError(RuntimeError):
    """Raised when a run cannot safely advance or resume."""


def validate_phase_runners(
    phase_runners: Mapping[str, PhaseRunner],
    phases: tuple[str, ...] = PHASES,
) -> None:
    """Validate the dispatcher contract before creating any run artifacts."""
    missing = [phase for phase in phases if phase not in phase_runners]
    if missing:
        raise OrchestrationError(
            "phase runner missing for " + ", ".join(missing)
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Windows may not permit fsync on directory handles; the atomic rename
        # is still the publication boundary there.
        pass


def _append_event(path: Path, event: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(dict(event), ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _missing_phase_runner(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    """Fail closed when the notebook has not bound a real phase implementation."""
    raise OrchestrationError(
        f"phase runner missing for {phase}; bind a Kaggle phase runner before executing"
    )


def _run_id(config: RunConfig) -> str:
    return config.run_id or datetime.now(timezone.utc).strftime("run-%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:8]


def _phase_context(config: RunConfig, run_id: str, run_dir: Path, bundle: Any = None) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "dataset_root": str(config.dataset_root),
        "dataset_fingerprint": config.dataset_fingerprint,
        "seed": config.seed,
        "fast_dev_run": config.fast_dev_run,
        "bundle": bundle,
        "curriculum_stages": [stage.to_dict() for stage in plan_curriculum(config.run_mode)],
    }


def _publish_latest(config: RunConfig, run_id: str, phase: str) -> None:
    _atomic_write_json(
        Path(config.output_dir) / "LATEST.json",
        {
            "run_id": run_id,
            "phase": phase,
            "commit_sha256": config.config_fingerprint,
        },
    )


def start_run(
    config: RunConfig,
    *,
    phases: tuple[str, ...] = PHASES,
    run_id: str | None = None,
    bundle: Any = None,
) -> RunSession:
    """Open a visible phase session for notebook cell-by-cell execution."""
    validate_phase_runners(config.phase_runners, phases)
    resolved_run_id = run_id or _run_id(config)
    run_dir = Path(config.output_dir) / resolved_run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return RunSession(
        config=config,
        run_id=resolved_run_id,
        run_dir=run_dir,
        phases=phases,
        context=_phase_context(config, resolved_run_id, run_dir, bundle),
    )


def run_phase(session: RunSession, phase: str) -> Mapping[str, Any]:
    """Execute exactly one phase and publish its checkpoint/artifact boundary."""
    if phase not in session.phases:
        raise OrchestrationError(f"phase is not active for this run mode: {phase}")
    if phase in session.completed:
        raise OrchestrationError(f"phase already completed in this session: {phase}")
    canonical_index = PHASES.index(phase) + 1
    events_path = session.run_dir / "run.jsonl"
    _append_event(
        events_path,
        {"event": "PHASE_START", "phase": phase, "phase_index": canonical_index, "timestamp": _utc_now()},
    )
    runner = session.config.phase_runners.get(phase, _missing_phase_runner)
    try:
        result = dict(runner(session.config, phase, session.context))
        _atomic_write_json(
            session.run_dir / "artifacts" / f"{phase}.json",
            {"phase": phase, "phase_index": canonical_index, "status": "PASS", "result": result},
        )
        _append_event(
            events_path,
            {"event": "PHASE_END", "phase": phase, "phase_index": canonical_index, "status": "PASS", "timestamp": _utc_now()},
        )
        session.completed.append(phase)
        _publish_latest(session.config, session.run_id, phase)
        return result
    except Exception as exc:
        _atomic_write_json(
            session.run_dir / "artifacts" / f"{phase}.error.json",
            {"phase": phase, "phase_index": canonical_index, "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc)},
        )
        _append_event(
            events_path,
            {"event": "PHASE_ERROR", "phase": phase, "phase_index": canonical_index, "status": "ERROR", "error_type": type(exc).__name__, "error": str(exc), "timestamp": _utc_now()},
        )
        raise


def finish_run(session: RunSession) -> RunSummary:
    if not session.completed:
        raise OrchestrationError("cannot finish a run before any phase completes")
    phase_completed = session.completed[-1]
    _atomic_write_json(
        session.run_dir / "run_manifest.json",
        {
            "schema_id": "clinical_nlp.run_manifest",
            "schema_version": 1,
            "run_id": session.run_id,
            "mode": session.config.run_mode,
            "status": "PASS",
            "phase_completed": phase_completed,
            "phases": list(session.phases),
            "dataset_fingerprint": session.config.dataset_fingerprint,
            "config_fingerprint": session.config.config_fingerprint,
            "curriculum_stages": session.context["curriculum_stages"],
        },
    )
    return RunSummary(
        run_id=session.run_id,
        status="PASS",
        phase_completed=phase_completed,
        artifacts_path=str(session.run_dir),
        mode=session.config.run_mode,
        phases=tuple(session.completed),
    )


def _execute_phases(
    config: RunConfig,
    phases: tuple[str, ...],
    start_index: int,
    run_id: str,
    bundle: Any = None,
) -> RunSummary:
    session = start_run(config, phases=phases[start_index:], run_id=run_id, bundle=bundle)
    for phase in session.phases:
        run_phase(session, phase)
    return finish_run(session)


def execute_run(config: RunConfig) -> RunSummary:
    if config.run_mode != "full":
        raise ValueError("execute_run requires run_mode='full'")
    return _execute_phases(config, PHASES, 0, _run_id(config))


def resume_run(config: RunConfig, latest: LatestPointer) -> RunSummary:
    if latest.commit_sha256 != config.config_fingerprint:
        raise ValueError("resume config fingerprint mismatch")
    if config.run_id is not None and latest.run_id != config.run_id:
        raise ValueError("resume run_id mismatch")
    try:
        phase_index = PHASES.index(latest.phase)
    except ValueError as exc:
        raise ValueError(f"resume phase is not in canonical phase list: {latest.phase}") from exc
    run_config = RunConfig(
        run_mode="resume",
        dataset_root=config.dataset_root,
        output_dir=config.output_dir,
        artifact_dir=config.artifact_dir,
        artifact_source_dir=config.artifact_source_dir,
        input_source=config.input_source,
        model_source=config.model_source,
        config_path=config.config_path,
        expected_gpu_count=config.expected_gpu_count,
        use_distributed=config.use_distributed,
        seed=config.seed,
        fast_dev_run=config.fast_dev_run,
        run_id=latest.run_id,
        dataset_fingerprint=config.dataset_fingerprint,
        config_fingerprint=config.config_fingerprint,
        phase_runners=config.phase_runners,
    )
    if phase_index == len(PHASES) - 1:
        return RunSummary(latest.run_id, "PASS", latest.phase, str(Path(config.output_dir) / latest.run_id), "resume", ())
    return _execute_phases(run_config, PHASES, phase_index + 1, latest.run_id)


def run_inference_only(config: RunConfig, bundle: Any) -> RunSummary:
    if bundle is None:
        raise OrchestrationError("inference_only requires a bound final model bundle")
    inference_phases = (PHASES[0], PHASES[1], PHASES[2], PHASES[11], PHASES[12])
    inference_config = RunConfig(
        run_mode="inference_only",
        dataset_root=config.dataset_root,
        output_dir=config.output_dir,
        artifact_dir=config.artifact_dir,
        artifact_source_dir=config.artifact_source_dir,
        input_source=config.input_source,
        model_source=config.model_source,
        config_path=config.config_path,
        expected_gpu_count=config.expected_gpu_count,
        use_distributed=config.use_distributed,
        seed=config.seed,
        fast_dev_run=config.fast_dev_run,
        run_id=config.run_id,
        dataset_fingerprint=config.dataset_fingerprint,
        config_fingerprint=config.config_fingerprint,
        phase_runners=config.phase_runners,
    )
    return _execute_phases(inference_config, inference_phases, 0, _run_id(inference_config), bundle=bundle)
