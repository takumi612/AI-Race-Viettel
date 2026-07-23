from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from clinical_nlp_lab.provenance import (
    MANIFEST_ROW_SCHEMA_ID,
    DatasetSnapshot,
    ProvenanceError,
    build_legacy_report_status_index,
    build_provenance_descriptor,
    build_v2_manifest_rows,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_json_strict,
    load_jsonl_strict,
    is_path_link_or_reparse,
    scan_dataset_layout,
    sha256_bytes,
    validate_v2_manifest,
    validate_provenance_descriptor,
    validate_artifact_path,
    validate_report_status_index,
    verify_dataset_provenance,
)


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    dataset_root: Path
    snapshot: DatasetSnapshot
    manifest_path: Path
    original_manifest_bytes: bytes
    original_manifest_sha256: str
    manifest_bytes: bytes
    descriptor_path: Path
    descriptor: dict[str, Any]
    descriptor_bytes: bytes
    archive_path: Path
    archive_relative_path: str
    report_index_path: Path
    report_index_bytes: bytes
    migration_required: bool


@dataclass(frozen=True, slots=True)
class MigrationResult:
    mode: str
    migration_required: bool
    dataset_root: Path
    document_count: int
    dataset_fingerprint: str
    manifest_sha256: str
    manifest_bytes: bytes
    descriptor: dict[str, Any]
    report_index_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "migration_required": self.migration_required,
            "dataset_root": str(self.dataset_root),
            "document_count": self.document_count,
            "dataset_fingerprint": self.dataset_fingerprint,
            "manifest_sha256": self.manifest_sha256,
            "legacy_manifest_sha256": self.descriptor["legacy_manifest"]["sha256"],
            "legacy_archive_path": self.descriptor["legacy_manifest"]["archive_path"],
            "report_index_sha256": self.report_index_sha256,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _detect_git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=V2_ROOT.parent,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def _is_v2_manifest(rows: tuple[dict[str, Any], ...]) -> bool:
    schema_values = {row.get("schema_id") for row in rows}
    if schema_values == {MANIFEST_ROW_SCHEMA_ID}:
        return True
    if MANIFEST_ROW_SCHEMA_ID in schema_values or any(
        "schema_version" in row
        or "input_sha256" in row
        or "gt_sha256" in row
        or "pair_sha256" in row
        for row in rows
    ):
        raise ProvenanceError("Manifest mixes legacy and v2 provenance fields")
    return False


def _report_payloads(dataset_root: Path) -> tuple[tuple[str, bytes], ...]:
    reports_root = validate_artifact_path(dataset_root, dataset_root / "reports")
    excluded = {
        "dataset_manifest.jsonl",
        "dataset_provenance.json",
        "report_index.jsonl",
    }
    payloads: list[tuple[str, bytes]] = []
    for path in reports_root.iterdir():
        if path.name in excluded or path.name == "archive":
            continue
        validate_artifact_path(dataset_root, path)
        if is_path_link_or_reparse(path):
            raise ProvenanceError("Historical report must not be a symlink/reparse point")
        if path.is_dir():
            raise ProvenanceError(f"Unexpected nested report directory: {path}")
        if path.is_file() and path.suffix.lower() in {".json", ".md"}:
            payloads.append((path.relative_to(dataset_root).as_posix(), path.read_bytes()))
    return tuple(payloads)


def build_migration_plan(
    dataset_root: str | Path,
    *,
    created_at: str | None = None,
    git_commit: str | None = None,
) -> MigrationPlan:
    snapshot = scan_dataset_layout(dataset_root)
    root = snapshot.dataset_root
    reports_path = validate_artifact_path(root, root / "reports")
    manifest_path = validate_artifact_path(root, reports_path / "dataset_manifest.jsonl")
    if not manifest_path.is_file():
        raise ProvenanceError("Missing non-symlink dataset manifest")
    original_manifest_bytes = manifest_path.read_bytes()
    original_manifest_sha256 = sha256_bytes(original_manifest_bytes)
    rows = load_jsonl_strict(original_manifest_bytes, source=str(manifest_path))
    descriptor_path = validate_artifact_path(
        root, reports_path / "dataset_provenance.json", allow_missing_leaf=True
    )
    report_index_path = validate_artifact_path(
        root, reports_path / "report_index.jsonl", allow_missing_leaf=True
    )

    if _is_v2_manifest(rows):
        try:
            verification = verify_dataset_provenance(
                root, manifest_path=manifest_path, descriptor_path=descriptor_path
            )
        except ProvenanceError:
            raise ProvenanceError(
                "Detected invalid or partial v2 publication; audited recovery is required"
            ) from None
        archive_relative = verification.descriptor["legacy_manifest"]["archive_path"]
        archive_path = root / Path(archive_relative)
        report_index_bytes = verification.report_index_bytes or b""
        return MigrationPlan(
            dataset_root=root,
            snapshot=verification.snapshot,
            manifest_path=manifest_path,
            original_manifest_bytes=original_manifest_bytes,
            original_manifest_sha256=original_manifest_sha256,
            manifest_bytes=original_manifest_bytes,
            descriptor_path=descriptor_path,
            descriptor=verification.descriptor,
            descriptor_bytes=verification.descriptor_bytes,
            archive_path=archive_path,
            archive_relative_path=archive_relative,
            report_index_path=report_index_path,
            report_index_bytes=report_index_bytes,
            migration_required=False,
        )

    if os.path.lexists(descriptor_path):
        raise ProvenanceError("Legacy manifest has an orphan descriptor; audited recovery required")
    if os.path.lexists(report_index_path):
        raise ProvenanceError("Legacy manifest has an orphan report index; audited recovery required")

    upgraded_rows = build_v2_manifest_rows(rows, snapshot)
    manifest_bytes = canonical_jsonl_bytes(upgraded_rows)
    # Self-check the exact bytes that would be published.
    parsed_candidate = load_jsonl_strict(manifest_bytes, source="candidate v2 manifest")
    validate_v2_manifest(parsed_candidate, snapshot)
    if canonical_jsonl_bytes(parsed_candidate) != manifest_bytes:
        raise ProvenanceError("Candidate manifest failed canonical byte self-check")

    archive_relative = (
        "reports/archive/dataset_manifest.legacy."
        f"{original_manifest_sha256}.jsonl"
    )
    report_index_rows = build_legacy_report_status_index(_report_payloads(root))
    validate_report_status_index(report_index_rows, root)
    report_index_bytes = canonical_jsonl_bytes(report_index_rows) if report_index_rows else b""
    descriptor = build_provenance_descriptor(
        snapshot,
        manifest_bytes,
        legacy_manifest_sha256=original_manifest_sha256,
        legacy_archive_path=archive_relative,
        created_at=created_at or _utc_now(),
        git_commit=git_commit if git_commit is not None else _detect_git_commit(),
        report_index_bytes=report_index_bytes or None,
    )
    validate_provenance_descriptor(
        descriptor,
        snapshot,
        manifest_bytes,
        report_index_bytes=report_index_bytes or None,
    )
    descriptor_bytes = canonical_json_bytes(descriptor)
    archive_path = root / Path(archive_relative)
    validate_artifact_path(root, archive_path, allow_missing_leaf=True)
    return MigrationPlan(
        dataset_root=root,
        snapshot=snapshot,
        manifest_path=manifest_path,
        original_manifest_bytes=original_manifest_bytes,
        original_manifest_sha256=original_manifest_sha256,
        manifest_bytes=manifest_bytes,
        descriptor_path=descriptor_path,
        descriptor=descriptor,
        descriptor_bytes=descriptor_bytes,
        archive_path=archive_path,
        archive_relative_path=archive_relative,
        report_index_path=report_index_path,
        report_index_bytes=report_index_bytes,
        migration_required=True,
    )


def _write_content_addressed_archive(path: Path, payload: bytes, expected_sha256: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file():
            raise ProvenanceError(f"Legacy archive path is not a regular file: {path}")
        actual = sha256_bytes(path.read_bytes())
        if actual != expected_sha256:
            raise ProvenanceError(
                f"Existing legacy archive hash mismatch: expected={expected_sha256}, actual={actual}"
            )
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    actual = sha256_bytes(path.read_bytes())
    if actual != expected_sha256:
        raise ProvenanceError(
            f"New legacy archive hash mismatch: expected={expected_sha256}, actual={actual}"
        )


def _atomic_replace_bytes(
    destination: Path,
    payload: bytes,
    validator: Callable[[bytes], None] | None = None,
) -> None:
    if is_path_link_or_reparse(destination) or is_path_link_or_reparse(destination.parent):
        raise ProvenanceError("Publication destination contains a symlink/reparse point")
    if not destination.parent.is_dir():
        raise ProvenanceError("Publication destination parent is missing")
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        written = temporary_path.read_bytes()
        if written != payload:
            raise ProvenanceError(f"Temporary publication bytes changed for {destination}")
        if validator is not None:
            validator(written)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _assert_unchanged_since_planning(plan: MigrationPlan) -> DatasetSnapshot:
    validate_artifact_path(plan.dataset_root, plan.manifest_path)
    current_snapshot = scan_dataset_layout(plan.dataset_root)
    current_manifest_bytes = plan.manifest_path.read_bytes()
    if (
        current_snapshot.dataset_fingerprint != plan.snapshot.dataset_fingerprint
        or current_manifest_bytes != plan.original_manifest_bytes
    ):
        raise ProvenanceError("Dataset or manifest changed since migration planning")
    return current_snapshot


def _assert_published_manifest_state(plan: MigrationPlan) -> DatasetSnapshot:
    validate_artifact_path(plan.dataset_root, plan.manifest_path)
    current_snapshot = scan_dataset_layout(plan.dataset_root)
    if (
        current_snapshot.dataset_fingerprint != plan.snapshot.dataset_fingerprint
        or plan.manifest_path.read_bytes() != plan.manifest_bytes
    ):
        raise ProvenanceError("Dataset or manifest changed during provenance publication")
    return current_snapshot


@contextmanager
def _dataset_publication_lock(dataset_root: Path):
    lock_path = validate_artifact_path(
        dataset_root, dataset_root / ".dataset-provenance.lock", allow_missing_leaf=True
    )
    token = f"pid={os.getpid()}\n".encode("ascii")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileExistsError:
        raise ProvenanceError("Dataset provenance publication lock is already held") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(token)
            stream.flush()
            os.fsync(stream.fileno())
        yield
    finally:
        try:
            if lock_path.is_file() and lock_path.read_bytes() == token:
                lock_path.unlink()
        except OSError:
            pass


def apply_migration_plan(plan: MigrationPlan) -> MigrationResult:
    if not plan.migration_required:
        verification = verify_dataset_provenance(plan.dataset_root)
        return MigrationResult(
            mode="write",
            migration_required=False,
            dataset_root=plan.dataset_root,
            document_count=verification.document_count,
            dataset_fingerprint=verification.dataset_fingerprint,
            manifest_sha256=verification.manifest_sha256,
            manifest_bytes=verification.manifest_bytes,
            descriptor=verification.descriptor,
            report_index_sha256=(
                sha256_bytes(plan.report_index_bytes) if plan.report_index_bytes else None
            ),
        )

    with _dataset_publication_lock(plan.dataset_root):
        before_publish = _assert_unchanged_since_planning(plan)
        validate_artifact_path(plan.dataset_root, plan.archive_path, allow_missing_leaf=True)
        _write_content_addressed_archive(
            plan.archive_path, plan.original_manifest_bytes, plan.original_manifest_sha256
        )
        # CAS after archive creation closes the original post-precheck overwrite window.
        _assert_unchanged_since_planning(plan)
        manifest_published = False
        report_index_published = False
        try:
            validate_artifact_path(plan.dataset_root, plan.manifest_path)
            _atomic_replace_bytes(
                plan.manifest_path,
                plan.manifest_bytes,
                lambda raw: validate_v2_manifest(
                    load_jsonl_strict(raw, source="temporary v2 manifest"), before_publish
                ),
            )
            manifest_published = True
            _assert_published_manifest_state(plan)
            if plan.report_index_bytes:
                validate_artifact_path(
                    plan.dataset_root, plan.report_index_path, allow_missing_leaf=True
                )
                _atomic_replace_bytes(
                    plan.report_index_path,
                    plan.report_index_bytes,
                    lambda raw: validate_report_status_index(
                        load_jsonl_strict(raw, source="temporary report index"),
                        plan.dataset_root,
                    ),
                )
                report_index_published = True
                _assert_published_manifest_state(plan)

            def validate_descriptor_bytes(raw: bytes) -> None:
                payload = load_json_strict(raw, source="temporary descriptor")
                if not isinstance(payload, dict):
                    raise ProvenanceError("Temporary descriptor is not an object")
                if canonical_json_bytes(payload) != raw:
                    raise ProvenanceError("Temporary descriptor is not canonical JSON")
                validate_provenance_descriptor(
                    payload,
                    before_publish,
                    plan.manifest_bytes,
                    report_index_bytes=plan.report_index_bytes or None,
                )

            validate_artifact_path(
                plan.dataset_root, plan.descriptor_path, allow_missing_leaf=True
            )
            _atomic_replace_bytes(
                plan.descriptor_path,
                plan.descriptor_bytes,
                validate_descriptor_bytes,
            )
            _assert_published_manifest_state(plan)
        except Exception as exc:
            if manifest_published:
                state = "manifest+index" if report_index_published else "manifest-only"
                raise ProvenanceError(
                    f"Incomplete provenance publication ({state}); preflight must remain closed"
                ) from exc
            raise

    verification = verify_dataset_provenance(plan.dataset_root)
    if verification.dataset_fingerprint != plan.snapshot.dataset_fingerprint:
        raise ProvenanceError(
            "Post-publication dataset fingerprint changed; input/GT provenance is invalid"
        )
    if verification.manifest_bytes != plan.manifest_bytes:
        raise ProvenanceError("Post-publication manifest bytes differ from the migration plan")
    return MigrationResult(
        mode="write",
        migration_required=True,
        dataset_root=plan.dataset_root,
        document_count=verification.document_count,
        dataset_fingerprint=verification.dataset_fingerprint,
        manifest_sha256=verification.manifest_sha256,
        manifest_bytes=verification.manifest_bytes,
        descriptor=verification.descriptor,
        report_index_sha256=(
            sha256_bytes(plan.report_index_bytes) if plan.report_index_bytes else None
        ),
    )


def migrate_dataset_provenance(
    dataset_root: str | Path,
    *,
    write: bool = False,
    created_at: str | None = None,
    git_commit: str | None = None,
) -> MigrationResult:
    plan = build_migration_plan(
        dataset_root, created_at=created_at, git_commit=git_commit
    )
    if write:
        return apply_migration_plan(plan)
    return MigrationResult(
        mode="check",
        migration_required=plan.migration_required,
        dataset_root=plan.dataset_root,
        document_count=plan.snapshot.document_count,
        dataset_fingerprint=plan.snapshot.dataset_fingerprint,
        manifest_sha256=sha256_bytes(plan.manifest_bytes),
        manifest_bytes=plan.manifest_bytes,
        descriptor=plan.descriptor,
        report_index_sha256=(
            sha256_bytes(plan.report_index_bytes) if plan.report_index_bytes else None
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check or non-destructively publish strict dataset provenance v2 metadata."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Explicit dataset root containing input/, gt/, and reports/.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Read-only validation (default).")
    mode.add_argument("--write", action="store_true", help="Explicitly publish migration files.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = migrate_dataset_provenance(args.dataset_root, write=bool(args.write))
    except ProvenanceError:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_code": "provenance_validation_failed",
                    "message": "Dataset provenance validation failed; no clinical text is logged.",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
