import json
from pathlib import Path
import sqlite3

import pytest

from src.training.build_datasets import (
    DatasetBuildConfig,
    build_training_datasets,
    main,
)


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _write_pair(root: Path, record_id: str, text: str) -> None:
    (root / "input").mkdir(parents=True, exist_ok=True)
    (root / "gt").mkdir(parents=True, exist_ok=True)
    (root / "input" / f"{record_id}.txt").write_text(text, encoding="utf-8")
    _write_json(root / "gt" / f"{record_id}.json", [])


def _config_mapping(
    *,
    synthetic_root: str = "data/synthetic_train_v1",
    output: str = "data/training",
) -> dict:
    return {
        "schema_version": 1,
        "seed": 20260719,
        "synthetic": {
            "root": synthetic_root,
            "expected_count": 2,
            "validation_size": 1,
        },
        "trusted": {
            "root": "data/dev",
            "first_id": 101,
            "last_id": 180,
            "folds": 5,
        },
        "holdout": {
            "root": "data/dev",
            "first_id": 181,
            "last_id": 200,
        },
        "database": "data/kb/metadata.db",
        "output": output,
    }


def _build_fixture_project(project_root: Path) -> None:
    synthetic = project_root / "data" / "synthetic_train_v1"
    _write_pair(synthetic, "0001", "Synthetic one")
    _write_pair(synthetic, "0002", "Synthetic two")
    (synthetic / "qa").mkdir()
    _write_json(
        synthetic / "qa" / "validation_report.json",
        {"passed": True, "coverage": {"record_count": 2}},
    )

    dev = project_root / "data" / "dev"
    for record_id in range(101, 201):
        _write_pair(dev, str(record_id), f"Trusted record {record_id}")

    database = project_root / "data" / "kb" / "metadata.db"
    database.parent.mkdir(parents=True)
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE icd10 (code TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE rxnorm (rxcui TEXT)")


@pytest.fixture
def valid_build_config(tmp_path) -> DatasetBuildConfig:
    project_root = tmp_path / "repo"
    project_root.mkdir()
    _build_fixture_project(project_root)
    return DatasetBuildConfig.from_mapping(
        _config_mapping(),
        project_root=project_root,
        allow_non_production_count=True,
    )


def test_build_refuses_failed_synthetic_and_preserves_existing_output(tmp_path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    failed_root = project_root / "data" / "synthetic.failed-validation"
    (failed_root / "input").mkdir(parents=True)
    (failed_root / "gt").mkdir()
    output = project_root / "data" / "training"
    output.mkdir(parents=True)
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    config = DatasetBuildConfig.from_mapping(
        _config_mapping(synthetic_root="data/synthetic.failed-validation"),
        project_root=project_root,
        allow_non_production_count=True,
    )

    with pytest.raises(ValueError, match="failed-validation"):
        build_training_datasets(config)

    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_writes_verified_manifest_and_task_jsonl(valid_build_config):
    output = build_training_datasets(valid_build_config)

    expected_files = (
        "manifests/build.json",
        "canonical/records.jsonl",
        "splits/assignments.jsonl",
        "ner/records.jsonl",
        "embedding/seeds.jsonl",
        "reranker/seeds.jsonl",
    )
    assert all((output / relative_path).is_file() for relative_path in expected_files)

    manifest = json.loads(
        (output / "manifests" / "build.json").read_text(encoding="utf-8")
    )
    assert manifest["counts"]["canonical_records"] == 102
    assert manifest["flags"]["allow_non_production_count"] is True
    assert len(manifest["fingerprints"]["database_sha256"]) == 64
    assert len(manifest["fingerprints"]["split_sha256"]) == 64
    assert set(manifest["fingerprints"]["artifacts"]) == {
        "canonical/records.jsonl",
        "splits/assignments.jsonl",
        "ner/records.jsonl",
        "embedding/seeds.jsonl",
        "reranker/seeds.jsonl",
    }

    assignments = [
        json.loads(line)
        for line in (output / "splits" / "assignments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert not ({str(index) for index in range(1, 101)} & {
        item["record_id"] for item in assignments
    })
    assert {item["record_id"] for item in assignments if item["split"] == "holdout"} == {
        str(index) for index in range(181, 201)
    }


def test_build_refuses_to_replace_output_without_explicit_flag(valid_build_config):
    output = build_training_datasets(valid_build_config)
    manifest_before = (output / "manifests" / "build.json").read_bytes()

    with pytest.raises(FileExistsError, match="--replace"):
        build_training_datasets(valid_build_config)

    assert (output / "manifests" / "build.json").read_bytes() == manifest_before


def test_build_replaces_existing_output_only_when_explicit(valid_build_config):
    output = build_training_datasets(valid_build_config)
    marker = output / "old-output.txt"
    marker.write_text("old", encoding="utf-8")
    replacement = DatasetBuildConfig.from_mapping(
        valid_build_config.to_mapping(),
        project_root=valid_build_config.project_root,
        replace=True,
        allow_non_production_count=True,
    )

    replaced_output = build_training_datasets(replacement)

    assert replaced_output == output
    assert not marker.exists()
    assert (output / "manifests" / "build.json").is_file()


def test_config_rejects_machine_specific_and_competition_input_paths(tmp_path):
    absolute = _config_mapping()
    absolute["synthetic"]["root"] = r"D:\private\synthetic"
    with pytest.raises(ValueError, match="project-relative"):
        DatasetBuildConfig.from_mapping(
            absolute,
            project_root=tmp_path,
            allow_non_production_count=True,
        )

    competition = _config_mapping(synthetic_root="data/input")
    with pytest.raises(ValueError, match="data/input"):
        DatasetBuildConfig.from_mapping(
            competition,
            project_root=tmp_path,
            allow_non_production_count=True,
        )


def test_config_rejects_output_nested_inside_a_protected_source(tmp_path):
    mapping = _config_mapping(output="data/dev/training-output")

    with pytest.raises(ValueError, match="protected inputs"):
        DatasetBuildConfig.from_mapping(
            mapping,
            project_root=tmp_path,
            allow_non_production_count=True,
        )


def test_config_locks_trusted_holdout_boundaries_and_production_count(tmp_path):
    wrong_boundary = _config_mapping()
    wrong_boundary["trusted"]["first_id"] = 100
    with pytest.raises(ValueError, match="101-180"):
        DatasetBuildConfig.from_mapping(
            wrong_boundary,
            project_root=tmp_path,
            allow_non_production_count=True,
        )

    with pytest.raises(ValueError, match="2,000"):
        DatasetBuildConfig.from_mapping(
            _config_mapping(),
            project_root=tmp_path,
        )


def test_cli_help_documents_safety_flags(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "--replace" in help_text
    assert "--allow-non-production-count" in help_text
