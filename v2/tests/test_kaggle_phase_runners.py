from __future__ import annotations

from pathlib import Path

from clinical_nlp_lab.orchestration import PHASES, RunConfig
from clinical_nlp_lab.kaggle_phases import build_kaggle_phase_runners, _build_training_command
from scripts.train_ner_subprocess import load_stage_selection


def test_builtin_kaggle_dispatcher_binds_all_thirteen_phases(tmp_path: Path):
    config = RunConfig(
        output_dir=tmp_path / "run_output",
        artifact_dir=tmp_path / "artifacts",
        model_source="local-model",
        expected_gpu_count=2,
    )
    runners = build_kaggle_phase_runners(config)
    assert tuple(runners) == PHASES
    assert all(callable(runners[phase]) for phase in PHASES)


def test_phase_runner_context_is_explicitly_bound_to_run_paths(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    input_zip = tmp_path / "input.zip"
    input_zip.write_bytes(b"placeholder")
    config = RunConfig(
        output_dir=tmp_path / "run_output",
        artifact_dir=tmp_path / "artifacts",
        dataset_root=dataset,
        input_source=input_zip,
        model_source="local-model",
    )
    runners = build_kaggle_phase_runners(config)
    context = {
        "run_id": "run-test",
        "run_dir": str(tmp_path / "run_output" / "run-test"),
        "dataset_root": str(dataset),
        "seed": 42,
        "fast_dev_run": True,
    }
    result = runners["phase_02_resolve_sources"](config, "phase_02_resolve_sources", context)
    assert result["phase"] == "phase_02_resolve_sources"
    assert result["dataset_root"] == str(dataset.resolve())
    assert result["model_source"] == "local-model"


def test_stage_selection_manifest_is_deterministic(tmp_path: Path):
    manifest = tmp_path / "stage.json"
    manifest.write_text(
        '{"stage_name":"stage2","train_ids":["3","1"],"validation_ids":["4"],"dataset_fingerprint":"d","split_fingerprint":"s"}',
        encoding="utf-8",
    )
    selection = load_stage_selection(manifest)
    assert selection == {
        "stage_name": "stage2",
        "train_ids": ("1", "3"),
        "validation_ids": ("4",),
        "dataset_fingerprint": "d",
        "split_fingerprint": "s",
    }


def test_training_command_uses_two_gpu_distributed_launcher():
    config = RunConfig(expected_gpu_count=2, use_distributed=True)
    command = _build_training_command(config, Path("train.py"), ["--stage-name", "stage1"], gpu_count=2)
    assert Path(command[0]).name.startswith("python")
    assert command[1:5] == ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node"]
    assert command[5] == "2"
    assert command[-2:] == ["--stage-name", "stage1"]
    assert command.count("train.py") == 1


def test_training_command_uses_direct_script_without_duplicate_path():
    config = RunConfig(expected_gpu_count=1, use_distributed=False)
    command = _build_training_command(config, Path("train.py"), ["--stage-name", "stage1"], gpu_count=1)
    assert command[:2] == [command[0], "train.py"]
    assert command.count("train.py") == 1
