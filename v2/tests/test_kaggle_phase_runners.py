from __future__ import annotations

from pathlib import Path

from clinical_nlp_lab.orchestration import PHASES, RunConfig
from clinical_nlp_lab.kaggle_phases import (
    _build_training_command,
    _write_stage_input,
    build_kaggle_phase_runners,
)
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


def test_final_fit_keeps_validation_partition_for_entity_calibration(tmp_path: Path):
    run_dir = tmp_path / "run-output" / "run-1"
    split_dir = run_dir / "artifacts" / "splits"
    split_dir.mkdir(parents=True)
    (split_dir / "split_descriptor.json").write_text(
        __import__("json").dumps(
            {
                "fixed_partitions": {
                    "synthetic_train_ids": ["201", "202"],
                    "synthetic_validation_ids": ["203"],
                    "organizer_train_ids": ["101"],
                    "organizer_validation_ids": ["102"],
                },
                "dataset_fingerprint": "dataset",
                "fixed_split_sha256": "split",
            }
        ),
        encoding="utf-8",
    )
    config = RunConfig(output_dir=tmp_path / "run-output")

    path = _write_stage_input(
        config,
        {"run_dir": str(run_dir)},
        "final_fit",
    )

    payload = __import__("json").loads(path.read_text(encoding="utf-8"))
    assert payload["validation_ids"] == [
        *[str(value) for value in range(181, 201)],
        "203",
        "102",
    ]
    assert set(payload["train_ids"]).isdisjoint(payload["validation_ids"])
    assert not set(str(value) for value in range(181, 201)) & set(payload["train_ids"])
