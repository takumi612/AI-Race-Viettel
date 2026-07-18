import json

import pytest

from src.training.artifacts import (
    TrainingRunState,
    promote_run,
    start_or_resume_run,
    update_run_state,
)
from src.training.metrics import (
    PrecisionMetrics,
    choose_precision_first,
    exact_fbeta,
)
from src.training.promote import main as promote_main


def _dataset_manifest(tmp_path):
    path = tmp_path / "build.json"
    path.write_text(
        json.dumps({"schema_version": 1, "build_id": "dataset-build-1"}),
        encoding="utf-8",
    )
    return path


def test_exact_f05_weights_precision_more_than_recall():
    high_precision = exact_fbeta({"a", "b", "c"}, {"a", "b"}, beta=0.5)
    high_recall = exact_fbeta({"a", "b"}, {"a", "b", "c"}, beta=0.5)

    assert high_precision.precision == 1.0
    assert high_recall.recall == 1.0
    assert high_precision.f_beta > high_recall.f_beta
    assert high_precision.to_mapping()["beta"] == 0.5


def test_precision_first_selection_applies_recall_floor_and_tiebreaker():
    candidates = {
        "unsafe": PrecisionMetrics(0.99, 0.20, 0.70, 0.5, 20, 1, 80),
        "recall-heavy": PrecisionMetrics(0.80, 0.75, 0.79, 0.5, 75, 19, 25),
        "precision-heavy": PrecisionMetrics(0.90, 0.70, 0.792, 0.5, 70, 8, 30),
    }

    selected = choose_precision_first(
        candidates,
        recall_floor=0.60,
        score_tolerance=0.005,
    )

    assert selected == "precision-heavy"


def test_start_and_resume_require_identical_fingerprints(tmp_path):
    run_dir = tmp_path / "runs" / "ner"
    manifest = _dataset_manifest(tmp_path)
    state = start_or_resume_run(
        run_dir,
        task="ner",
        base_model="xlm-roberta-base",
        config={"learning_rate": 2e-5},
        dataset_manifest=manifest,
        seed=7,
        resume=False,
    )

    resumed = start_or_resume_run(
        run_dir,
        task="ner",
        base_model="xlm-roberta-base",
        config={"learning_rate": 2e-5},
        dataset_manifest=manifest,
        seed=7,
        resume=True,
    )
    assert resumed == state

    with pytest.raises(ValueError, match="config_sha256"):
        start_or_resume_run(
            run_dir,
            task="ner",
            base_model="xlm-roberta-base",
            config={"learning_rate": 3e-5},
            dataset_manifest=manifest,
            seed=7,
            resume=True,
        )


def test_run_state_updates_are_monotonic_and_checkpoint_is_relative(tmp_path):
    run_dir = tmp_path / "run"
    state = start_or_resume_run(
        run_dir,
        task="embedding",
        base_model="bge-m3",
        config={"epochs": 1},
        dataset_manifest=_dataset_manifest(tmp_path),
        seed=11,
        resume=False,
    )
    checkpoint = run_dir / "checkpoint-10"
    checkpoint.mkdir()

    updated = update_run_state(
        run_dir,
        state,
        global_step=10,
        checkpoint=checkpoint,
    )

    assert updated.global_step == 10
    assert updated.checkpoint == "checkpoint-10"
    with pytest.raises(ValueError, match="cannot decrease"):
        update_run_state(
            run_dir,
            updated,
            global_step=9,
            checkpoint=checkpoint,
        )


def test_promote_run_writes_validated_manifest_and_moves_atomically(tmp_path):
    run_dir = tmp_path / "candidate"
    state = start_or_resume_run(
        run_dir,
        task="reranker",
        base_model="Qwen2.5-7B-Instruct",
        config={"lora_rank": 16},
        dataset_manifest=_dataset_manifest(tmp_path),
        seed=13,
        resume=False,
    )
    (run_dir / "adapter").mkdir()
    (run_dir / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    artifact_dir = tmp_path / "artifacts" / "reranker-v1"

    promoted = promote_run(
        run_dir,
        artifact_dir,
        state,
        metrics={"precision": 0.9, "recall": 0.7, "f0_5": 0.85},
        status="validated",
    )

    assert promoted == artifact_dir
    assert not run_dir.exists()
    manifest = json.loads(
        (artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "validated"
    assert manifest["run"]["task"] == "reranker"
    assert len(manifest["artifact_sha256"]) == 64


def test_run_state_rejects_unknown_status():
    with pytest.raises(ValueError, match="status"):
        TrainingRunState.from_mapping(
            {
                "schema_version": 1,
                "task": "ner",
                "base_model": "base",
                "config_sha256": "a" * 64,
                "dataset_build_id": "build",
                "dataset_manifest_sha256": "b" * 64,
                "seed": 1,
                "status": "published",
                "global_step": 0,
                "checkpoint": None,
            }
        )


def test_promotion_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        promote_main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--run-dir" in output
    assert "--artifact-dir" in output
    assert "--metrics-json" in output
