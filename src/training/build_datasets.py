"""Atomic CLI builder for canonical and task-specific training datasets."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import subprocess
from typing import Any, Mapping, Sequence
from uuid import uuid4

import yaml

from src.training.contracts import CanonicalRecord
from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)
from src.training.projections import (
    project_embedding_seeds,
    project_ner_records,
    project_reranker_seeds,
)
from src.training.sources import SourceSpec, load_source_records
from src.training.splits import (
    SplitAssignment,
    assert_no_split_leakage,
    build_synthetic_split,
    build_trusted_folds,
)
from src.training.validation import require_valid_source


_PRODUCTION_SYNTHETIC_COUNT = 2000
_TRUSTED_BOUNDARY = (101, 180)
_HOLDOUT_BOUNDARY = (181, 200)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _integer(value: Any, field: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _relative_path(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty project-relative path")
    raw = value.strip()
    windows_path = PureWindowsPath(raw)
    posix_path = PurePosixPath(raw)
    if windows_path.is_absolute() or windows_path.drive or posix_path.is_absolute():
        raise ValueError(f"{field} must be a project-relative path")
    if "\\" in raw:
        raise ValueError(f"{field} must use portable project-relative '/' separators")
    if any(part in {"", ".", "..", "~"} for part in posix_path.parts):
        raise ValueError(f"{field} contains an unsafe path segment")
    return Path(*posix_path.parts)


def _require_keys(
    mapping: Mapping[str, Any],
    expected: set[str],
    field: str,
) -> None:
    missing = sorted(expected - set(mapping))
    unexpected = sorted(set(mapping) - expected)
    if missing or unexpected:
        raise ValueError(
            f"{field} keys mismatch; missing={missing}, unexpected={unexpected}"
        )


@dataclass(frozen=True, slots=True)
class DatasetBuildConfig:
    schema_version: int
    seed: int
    synthetic_root: Path
    synthetic_expected_count: int
    synthetic_validation_size: int
    trusted_root: Path
    trusted_first_id: int
    trusted_last_id: int
    trusted_folds: int
    holdout_root: Path
    holdout_first_id: int
    holdout_last_id: int
    database: Path
    output: Path
    project_root: Path
    replace: bool = False
    allow_non_production_count: bool = False

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        *,
        project_root: str | Path = ".",
        replace: bool = False,
        allow_non_production_count: bool = False,
    ) -> "DatasetBuildConfig":
        mapping = _mapping(mapping, "config")
        _require_keys(
            mapping,
            {
                "schema_version",
                "seed",
                "synthetic",
                "trusted",
                "holdout",
                "database",
                "output",
            },
            "config",
        )
        schema_version = _integer(mapping["schema_version"], "schema_version")
        if schema_version != 1:
            raise ValueError(f"unsupported schema_version: {schema_version}")
        seed = _integer(mapping["seed"], "seed", minimum=0)

        synthetic = _mapping(mapping["synthetic"], "synthetic")
        _require_keys(
            synthetic,
            {"root", "expected_count", "validation_size"},
            "synthetic",
        )
        synthetic_root = _relative_path(synthetic["root"], "synthetic.root")
        synthetic_count = _integer(
            synthetic["expected_count"],
            "synthetic.expected_count",
        )
        validation_size = _integer(
            synthetic["validation_size"],
            "synthetic.validation_size",
        )
        if validation_size >= synthetic_count:
            raise ValueError(
                "synthetic.validation_size must be smaller than expected_count"
            )
        if (
            synthetic_count != _PRODUCTION_SYNTHETIC_COUNT
            and not allow_non_production_count
        ):
            raise ValueError(
                "production synthetic data must contain exactly 2,000 records; "
                "use --allow-non-production-count only for fixtures"
            )

        trusted = _mapping(mapping["trusted"], "trusted")
        _require_keys(
            trusted,
            {"root", "first_id", "last_id", "folds"},
            "trusted",
        )
        trusted_root = _relative_path(trusted["root"], "trusted.root")
        trusted_first = _integer(trusted["first_id"], "trusted.first_id")
        trusted_last = _integer(trusted["last_id"], "trusted.last_id")
        trusted_folds = _integer(trusted["folds"], "trusted.folds", minimum=2)
        if (trusted_first, trusted_last) != _TRUSTED_BOUNDARY:
            raise ValueError("trusted boundaries are locked to IDs 101-180")
        if (trusted_last - trusted_first + 1) % trusted_folds:
            raise ValueError("trusted record count must be divisible by folds")

        holdout = _mapping(mapping["holdout"], "holdout")
        _require_keys(holdout, {"root", "first_id", "last_id"}, "holdout")
        holdout_root = _relative_path(holdout["root"], "holdout.root")
        holdout_first = _integer(holdout["first_id"], "holdout.first_id")
        holdout_last = _integer(holdout["last_id"], "holdout.last_id")
        if (holdout_first, holdout_last) != _HOLDOUT_BOUNDARY:
            raise ValueError("holdout boundaries are locked to IDs 181-200")
        if trusted_root != holdout_root:
            raise ValueError("trusted and holdout must use the same data/dev source")

        database = _relative_path(mapping["database"], "database")
        output = _relative_path(mapping["output"], "output")
        competition_input = Path("data") / "input"
        for field, path in (
            ("synthetic.root", synthetic_root),
            ("trusted.root", trusted_root),
            ("holdout.root", holdout_root),
        ):
            if path == competition_input or competition_input in path.parents:
                raise ValueError(f"{field} must never point to data/input")
        protected_roots = (synthetic_root, trusted_root, holdout_root)
        if (
            output == Path(".")
            or output == database
            or any(
                output == protected
                or protected in output.parents
                or output in protected.parents
                for protected in protected_roots
            )
        ):
            raise ValueError("output must be separate from all protected inputs")
        if not isinstance(replace, bool):
            raise ValueError("replace must be a boolean")
        if not isinstance(allow_non_production_count, bool):
            raise ValueError("allow_non_production_count must be a boolean")

        return cls(
            schema_version=schema_version,
            seed=seed,
            synthetic_root=synthetic_root,
            synthetic_expected_count=synthetic_count,
            synthetic_validation_size=validation_size,
            trusted_root=trusted_root,
            trusted_first_id=trusted_first,
            trusted_last_id=trusted_last,
            trusted_folds=trusted_folds,
            holdout_root=holdout_root,
            holdout_first_id=holdout_first,
            holdout_last_id=holdout_last,
            database=database,
            output=output,
            project_root=Path(project_root).resolve(),
            replace=replace,
            allow_non_production_count=allow_non_production_count,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "seed": self.seed,
            "synthetic": {
                "root": self.synthetic_root.as_posix(),
                "expected_count": self.synthetic_expected_count,
                "validation_size": self.synthetic_validation_size,
            },
            "trusted": {
                "root": self.trusted_root.as_posix(),
                "first_id": self.trusted_first_id,
                "last_id": self.trusted_last_id,
                "folds": self.trusted_folds,
            },
            "holdout": {
                "root": self.holdout_root.as_posix(),
                "first_id": self.holdout_first_id,
                "last_id": self.holdout_last_id,
            },
            "database": self.database.as_posix(),
            "output": self.output.as_posix(),
        }


def _resolve_inside_project(project_root: Path, path: Path) -> Path:
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError as exc:
        raise ValueError(f"path escapes project root: {path}") from exc
    return resolved


def _source_files(spec: SourceSpec) -> list[Path]:
    root = spec.root.resolve()
    paths: list[Path] = []
    for record_id in spec.expected_ids:
        paths.append(root / "input" / f"{record_id}.txt")
        paths.append(root / "gt" / f"{record_id}.json")
    manifest = spec.manifest_path
    if manifest is None:
        default_manifest = root / "manifest.jsonl"
        if default_manifest.is_file():
            paths.append(default_manifest)
    else:
        paths.append(manifest if manifest.is_absolute() else root / manifest)
    if spec.name == "synthetic":
        paths.append(root / "qa" / "validation_report.json")
    return paths


def _git_fingerprint(project_root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "-C", str(project_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(project_root), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unavailable", "dirty": None, "status_sha256": None}
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": sha256(status.encode("utf-8")).hexdigest(),
    }


def _assignment_mapping(assignment: SplitAssignment) -> dict[str, Any]:
    value: dict[str, Any] = {
        "record_id": assignment.record_id,
        "split": assignment.split,
    }
    if assignment.fold is not None:
        value["fold"] = assignment.fold
    return value


def _write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
            )
            stream.write("\n")


def _verify_jsonl(path: Path, expected_count: int) -> str:
    count = 0
    try:
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    raise ValueError(f"blank JSONL line {line_number}: {path}")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL line is not an object: {path}")
                count += 1
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"generated JSONL verification failed: {path}") from exc
    if count != expected_count:
        raise ValueError(
            f"generated JSONL count mismatch for {path}: "
            f"expected {expected_count}, got {count}"
        )
    return sha256_file(path)


def _install_output(
    temporary: Path,
    output: Path,
    *,
    replace: bool,
) -> None:
    if not output.exists():
        temporary.rename(output)
        return
    if not replace:
        raise FileExistsError(
            f"output already exists; pass --replace to replace it: {output}"
        )
    if not output.is_dir():
        raise ValueError(f"existing output is not a directory: {output}")

    backup = output.parent / f".training-backup-{uuid4().hex}"
    output.rename(backup)
    try:
        temporary.rename(output)
    except BaseException:
        if not output.exists() and backup.exists():
            backup.rename(output)
        raise
    try:
        shutil.rmtree(backup)
    except BaseException:
        output.rename(temporary)
        backup.rename(output)
        shutil.rmtree(temporary)
        raise


def build_training_datasets(config: DatasetBuildConfig) -> Path:
    project_root = config.project_root
    if not project_root.is_dir():
        raise ValueError(f"project root is not a directory: {project_root}")

    synthetic_root = _resolve_inside_project(project_root, config.synthetic_root)
    trusted_root = _resolve_inside_project(project_root, config.trusted_root)
    holdout_root = _resolve_inside_project(project_root, config.holdout_root)
    database = _resolve_inside_project(project_root, config.database)
    output = _resolve_inside_project(project_root, config.output)

    synthetic_spec = SourceSpec(
        name="synthetic",
        root=synthetic_root,
        trust_tier="synthetic_validated",
        expected_ids=tuple(
            f"{index:04d}"
            for index in range(1, config.synthetic_expected_count + 1)
        ),
    )
    trusted_spec = SourceSpec(
        name="trusted",
        root=trusted_root,
        trust_tier="trusted",
        expected_ids=tuple(
            str(index)
            for index in range(config.trusted_first_id, config.trusted_last_id + 1)
        ),
    )
    holdout_spec = SourceSpec(
        name="holdout",
        root=holdout_root,
        trust_tier="holdout",
        expected_ids=tuple(
            str(index)
            for index in range(config.holdout_first_id, config.holdout_last_id + 1)
        ),
    )

    # No output path is created until every protected source passes its gates.
    synthetic_records = load_source_records(synthetic_spec)
    trusted_records = load_source_records(trusted_spec)
    holdout_records = load_source_records(holdout_spec)
    require_valid_source(synthetic_spec, synthetic_records, database)
    require_valid_source(trusted_spec, trusted_records, database)
    require_valid_source(holdout_spec, holdout_records, database)

    synthetic_assignments = build_synthetic_split(
        synthetic_records,
        validation_size=config.synthetic_validation_size,
        seed=config.seed,
    )
    trusted_assignments = build_trusted_folds(
        trusted_records,
        folds=config.trusted_folds,
        seed=config.seed,
    )
    holdout_assignments = tuple(
        SplitAssignment(record.record_id, "holdout")
        for record in sorted(holdout_records, key=lambda item: item.record_id)
    )

    all_records = tuple(
        sorted(
            synthetic_records + trusted_records + holdout_records,
            key=lambda record: record.record_id,
        )
    )
    all_assignments = tuple(
        sorted(
            synthetic_assignments + trusted_assignments + holdout_assignments
        )
    )
    assert_no_split_leakage(all_records, all_assignments)
    assignment_index = {
        assignment.record_id: assignment for assignment in all_assignments
    }

    canonical = tuple(record.to_mapping() for record in all_records)
    assignment_rows = tuple(
        _assignment_mapping(assignment) for assignment in all_assignments
    )
    ner_records = project_ner_records(all_records, assignment_index)
    embedding_seeds = project_embedding_seeds(all_records, assignment_index)
    reranker_seeds = project_reranker_seeds(all_records, assignment_index)

    git_fingerprint = _git_fingerprint(project_root)
    if (
        not config.allow_non_production_count
        and git_fingerprint["commit"] == "unavailable"
    ):
        raise ValueError("production build requires a Git commit fingerprint")
    if (
        not config.allow_non_production_count
        and git_fingerprint["dirty"] is True
    ):
        raise ValueError("production build requires a clean Git working tree")

    source_fingerprints = {
        spec.name: fingerprint_files(_source_files(spec), spec.root)
        for spec in (synthetic_spec, trusted_spec, holdout_spec)
    }
    config_sha256 = stable_json_sha256(config.to_mapping())
    database_sha256 = sha256_file(database)
    split_sha256 = stable_json_sha256(assignment_rows)

    if output.exists() and not config.replace:
        raise FileExistsError(
            f"output already exists; pass --replace to replace it: {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".training-build-{uuid4().hex}"
    if temporary.exists():
        raise RuntimeError(f"temporary build path collision: {temporary}")
    temporary.mkdir()

    artifact_records: tuple[
        tuple[str, Sequence[Mapping[str, Any]]], ...
    ] = (
        ("canonical/records.jsonl", canonical),
        ("splits/assignments.jsonl", assignment_rows),
        ("ner/records.jsonl", ner_records),
        ("embedding/seeds.jsonl", embedding_seeds),
        ("reranker/seeds.jsonl", reranker_seeds),
    )

    try:
        for relative_path, records in artifact_records:
            _write_jsonl(temporary / relative_path, records)

        artifact_fingerprints = {
            relative_path: _verify_jsonl(temporary / relative_path, len(records))
            for relative_path, records in artifact_records
        }
        counts = {
            "canonical_records": len(canonical),
            "split_assignments": len(assignment_rows),
            "ner_records": len(ner_records),
            "embedding_seeds": len(embedding_seeds),
            "reranker_seeds": len(reranker_seeds),
            "synthetic_records": len(synthetic_records),
            "trusted_records": len(trusted_records),
            "holdout_records": len(holdout_records),
        }
        base_fingerprints = {
            "config_sha256": config_sha256,
            "database_sha256": database_sha256,
            "source_sha256": source_fingerprints,
            "split_sha256": split_sha256,
            "git": git_fingerprint,
        }
        manifest = {
            "schema_version": config.schema_version,
            "build_id": stable_json_sha256(base_fingerprints),
            "config": config.to_mapping(),
            "counts": counts,
            "flags": {
                "allow_non_production_count": config.allow_non_production_count,
                "replace": config.replace,
            },
            "fingerprints": {
                **base_fingerprints,
                "artifacts": artifact_fingerprints,
            },
        }
        manifest_path = temporary / "manifests" / "build.json"
        manifest_path.parent.mkdir(parents=True)
        manifest_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        if json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
            raise ValueError("generated build manifest failed round-trip verification")

        _install_output(temporary, output, replace=config.replace)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise

    return output


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ValueError(f"config file is missing: {path}")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot load YAML config: {path}") from exc
    return _mapping(value, "config")


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build validated precision-first training datasets atomically."
    )
    parser.add_argument(
        "--config",
        default="configs/training/data.yaml",
        help="YAML config path, relative to --project-root by default.",
    )
    parser.add_argument(
        "--project-root",
        default=".",
        help="Repository root containing project-relative data paths.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Atomically replace an existing validated output.",
    )
    parser.add_argument(
        "--allow-non-production-count",
        action="store_true",
        help="Allow a non-2,000 synthetic count for fixtures/smoke builds only.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    project_root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = project_root / config_path
    try:
        config = DatasetBuildConfig.from_mapping(
            _load_yaml_mapping(config_path),
            project_root=project_root,
            replace=args.replace,
            allow_non_production_count=args.allow_non_production_count,
        )
        output = build_training_datasets(config)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
