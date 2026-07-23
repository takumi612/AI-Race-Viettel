from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_nlp_lab.orchestration import (
    PHASES,
    LatestPointer,
    RunConfig,
    execute_run,
    resume_run,
    run_inference_only,
    OrchestrationError,
)


def test_execute_run_dispatches_thirteen_phases_and_publishes_latest(tmp_path: Path):
    seen: list[str] = []
    contexts: list[dict] = []

    def runner(config, phase, context):
        seen.append(phase)
        contexts.append(dict(context))
        return {"phase": phase, "training_skipped": True}

    config = RunConfig(
        output_dir=tmp_path,
        run_id="run-001",
        phase_runners={phase: runner for phase in PHASES},
    )
    summary = execute_run(config)

    assert tuple(seen) == PHASES
    assert [stage["name"] for stage in contexts[0]["curriculum_stages"]] == [
        "stage1", "stage2", "stage3", "final_fit"
    ]
    assert summary.status == "PASS"
    assert summary.phase_completed == PHASES[-1]
    latest = json.loads((tmp_path / "LATEST.json").read_text(encoding="utf-8"))
    assert latest["run_id"] == "run-001"
    assert (tmp_path / "run-001" / "run_manifest.json").is_file()
    events = [json.loads(line) for line in (tmp_path / "run-001" / "run.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [event["event"] for event in events].count("PHASE_START") == 13
    assert [event["event"] for event in events].count("PHASE_END") == 13


def test_resume_continues_after_latest_completed_phase_and_rejects_stale_config(tmp_path: Path):
    failed_once = {"value": True}

    def failing_runner(config, phase, context):
        if phase == PHASES[3] and failed_once["value"]:
            failed_once["value"] = False
            raise RuntimeError("synthetic phase failure")
        return {"phase": phase}

    config = RunConfig(
        output_dir=tmp_path,
        run_id="run-002",
        phase_runners={phase: failing_runner for phase in PHASES},
    )
    with pytest.raises(RuntimeError, match="synthetic phase failure"):
        execute_run(config)

    assert (tmp_path / "run-002" / "artifacts" / f"{PHASES[3]}.error.json").is_file()

    latest_payload = json.loads((tmp_path / "LATEST.json").read_text(encoding="utf-8"))
    latest = LatestPointer(**latest_payload)
    seen: list[str] = []

    def resume_runner(config, phase, context):
        seen.append(phase)
        return {"phase": phase}

    resumed = resume_run(
        RunConfig(output_dir=tmp_path, run_id="run-002", phase_runners={phase: resume_runner for phase in PHASES}),
        latest,
    )
    assert resumed.status == "PASS"
    assert seen == list(PHASES[3:])

    with pytest.raises(ValueError, match="fingerprint"):
        resume_run(
            RunConfig(
                output_dir=tmp_path,
                run_id="run-002",
                config_fingerprint="changed",
                phase_runners={phase: resume_runner for phase in PHASES},
            ),
            latest,
        )


def test_inference_only_skips_training_phases(tmp_path: Path):
    seen: list[str] = []

    def runner(config, phase, context):
        seen.append(phase)
        return {"phase": phase}

    summary = run_inference_only(
        RunConfig(output_dir=tmp_path, run_id="run-inf", phase_runners={phase: runner for phase in PHASES}),
        bundle=object(),
    )

    assert summary.status == "PASS"
    assert "phase_07_stage1" not in seen
    assert "phase_12_inference" in seen
    assert seen[-1] == "phase_13_packaging"


def test_missing_phase_runner_fails_closed_instead_of_claiming_pass(tmp_path: Path):
    with pytest.raises(OrchestrationError, match="phase runner missing"):
        execute_run(RunConfig(output_dir=tmp_path, run_id="run-missing"))
    assert not (tmp_path / "LATEST.json").exists()


def test_inference_only_requires_a_bound_final_bundle(tmp_path: Path):
    with pytest.raises(OrchestrationError, match="final model bundle"):
        run_inference_only(RunConfig(output_dir=tmp_path, run_id="run-no-bundle"), bundle=None)
