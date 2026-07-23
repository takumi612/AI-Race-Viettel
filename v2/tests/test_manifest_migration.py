from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from clinical_nlp_lab.provenance import (
    ProvenanceError,
    canonical_jsonl_bytes,
    compute_legacy_input_text_sha256,
    load_json_strict,
    load_jsonl_strict,
    sha256_bytes,
    verify_dataset_provenance,
)
from clinical_nlp_lab.data import load_ner_training_documents
from clinical_nlp_lab.schema import OFFICIAL_SCHEMA_KEYS
from tools.upgrade_dataset_provenance import (
    apply_migration_plan,
    build_migration_plan,
    main,
    migrate_dataset_provenance,
)
import tools.upgrade_dataset_provenance as upgrade_tool


def _create_legacy_dataset(root: Path) -> tuple[bytes, dict[str, bytes]]:
    (root / "input").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "reports").mkdir()
    payload_bytes: dict[str, bytes] = {}
    rows = []
    for document_id in ("1", "2"):
        text_bytes = f"Document {document_id}\r\n".encode("utf-8")
        gt_bytes = b"[]\r\n"
        (root / "input" / f"{document_id}.txt").write_bytes(text_bytes)
        (root / "gt" / f"{document_id}.json").write_bytes(gt_bytes)
        payload_bytes[f"input/{document_id}.txt"] = text_bytes
        payload_bytes[f"gt/{document_id}.json"] = gt_bytes
        rows.append(
            {
                "document_id": document_id,
                "source_bucket": "reconstructed",
                "genre": "fixture",
                "scenario": "fixture",
                "template_group": f"fixture:{document_id}",
                "long_tail": False,
                "train_eligible": False,
                "linking_train_eligible": False,
                "train_exclusion_reason": "fixture quarantine",
                "primary_candidates": [f"C{document_id}"],
                "sha256": compute_legacy_input_text_sha256(text_bytes),
                "domain_payload": {"keep": [document_id, 1]},
            }
        )
    manifest_bytes = canonical_jsonl_bytes(rows)
    (root / "reports" / "dataset_manifest.jsonl").write_bytes(manifest_bytes)
    (root / "reports" / "quality_report.json").write_text(
        json.dumps({"document_count": 2}), encoding="utf-8"
    )
    return manifest_bytes, payload_bytes


def _payload_hashes(root: Path, relative_paths: dict[str, bytes]) -> dict[str, str]:
    return {
        relative: hashlib.sha256((root / relative).read_bytes()).hexdigest()
        for relative in relative_paths
    }


def _create_single_legacy_dataset(
    root: Path, document_id: str, **row_overrides: object
) -> None:
    numeric_id = int(document_id)
    if numeric_id <= 100:
        source, train_eligible, linking_eligible, reason = (
            "reconstructed",
            False,
            False,
            "fixture quarantine",
        )
    elif numeric_id <= 200:
        source, train_eligible, linking_eligible, reason = (
            "organizer_gt",
            True,
            False,
            None,
        )
    else:
        source, train_eligible, linking_eligible, reason = (
            "synthetic",
            True,
            True,
            None,
        )
    text_bytes = f"Document {document_id}\r\n".encode()
    (root / "input").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "reports").mkdir()
    (root / "input" / f"{document_id}.txt").write_bytes(text_bytes)
    (root / "gt" / f"{document_id}.json").write_bytes(b"[]\r\n")
    row: dict[str, object] = {
        "document_id": document_id,
        "source_bucket": source,
        "genre": "fixture",
        "scenario": "fixture",
        "template_group": f"fixture:{document_id}",
        "long_tail": False,
        "train_eligible": train_eligible,
        "linking_train_eligible": linking_eligible,
        "train_exclusion_reason": reason,
        "primary_candidates": [],
        "sha256": compute_legacy_input_text_sha256(text_bytes),
    }
    row.update(row_overrides)
    (root / "reports" / "dataset_manifest.jsonl").write_bytes(
        canonical_jsonl_bytes([row])
    )


def test_default_check_is_read_only_and_preserves_domain_fields(tmp_path: Path):
    legacy_bytes, payload_bytes = _create_legacy_dataset(tmp_path)
    before = _payload_hashes(tmp_path, payload_bytes)

    result = migrate_dataset_provenance(tmp_path)

    assert result.mode == "check"
    assert result.migration_required is True
    assert (tmp_path / "reports" / "dataset_manifest.jsonl").read_bytes() == legacy_bytes
    assert not (tmp_path / "reports" / "dataset_provenance.json").exists()
    assert _payload_hashes(tmp_path, payload_bytes) == before
    candidate_rows = load_jsonl_strict(result.manifest_bytes, source="candidate")
    assert candidate_rows[0]["domain_payload"] == {"keep": ["1", 1]}
    assert candidate_rows[0]["sha256"] == compute_legacy_input_text_sha256(b"Document 1\r\n")


def test_write_archives_exact_bytes_publishes_atomically_and_is_idempotent(tmp_path: Path):
    legacy_bytes, payload_bytes = _create_legacy_dataset(tmp_path)
    before_payload_hashes = _payload_hashes(tmp_path, payload_bytes)

    first = migrate_dataset_provenance(
        tmp_path, write=True, created_at="2026-07-23T00:00:00Z", git_commit="abc123"
    )
    manifest_path = tmp_path / "reports" / "dataset_manifest.jsonl"
    descriptor_path = tmp_path / "reports" / "dataset_provenance.json"
    archive_path = tmp_path / first.descriptor["legacy_manifest"]["archive_path"]
    first_hashes = {
        "manifest": sha256_bytes(manifest_path.read_bytes()),
        "descriptor": sha256_bytes(descriptor_path.read_bytes()),
        "index": sha256_bytes((tmp_path / "reports" / "report_index.jsonl").read_bytes()),
    }

    assert first.mode == "write"
    assert first.migration_required is True
    assert archive_path.read_bytes() == legacy_bytes
    assert _payload_hashes(tmp_path, payload_bytes) == before_payload_hashes
    verification = verify_dataset_provenance(tmp_path)
    assert verification.document_count == 2
    assert verification.manifest_sha256 == first_hashes["manifest"]

    second = migrate_dataset_provenance(tmp_path, write=True)

    assert second.migration_required is False
    assert sha256_bytes(manifest_path.read_bytes()) == first_hashes["manifest"]
    assert sha256_bytes(descriptor_path.read_bytes()) == first_hashes["descriptor"]
    assert sha256_bytes((tmp_path / "reports" / "report_index.jsonl").read_bytes()) == first_hashes[
        "index"
    ]
    assert _payload_hashes(tmp_path, payload_bytes) == before_payload_hashes


def test_apply_rejects_concurrent_dataset_or_manifest_change_before_publication(tmp_path: Path):
    legacy_bytes, _ = _create_legacy_dataset(tmp_path)
    plan = build_migration_plan(tmp_path, created_at="2026-07-23T00:00:00Z")
    (tmp_path / "input" / "1.txt").write_bytes(b"changed")

    with pytest.raises(ProvenanceError, match="changed since migration planning"):
        apply_migration_plan(plan)

    assert (tmp_path / "reports" / "dataset_manifest.jsonl").read_bytes() == legacy_bytes
    assert not (tmp_path / "reports" / "dataset_provenance.json").exists()


def test_publication_lock_and_post_archive_cas_reject_late_manifest_writer(
    tmp_path: Path, monkeypatch
):
    legacy_bytes, _ = _create_legacy_dataset(tmp_path)
    plan = build_migration_plan(tmp_path, created_at="2026-07-23T00:00:00Z")
    real_archive = upgrade_tool._write_content_addressed_archive

    def archive_then_mutate(path: Path, payload: bytes, expected_sha256: str) -> None:
        real_archive(path, payload, expected_sha256)
        plan.manifest_path.write_bytes(legacy_bytes + b" ")

    monkeypatch.setattr(upgrade_tool, "_write_content_addressed_archive", archive_then_mutate)
    with pytest.raises(ProvenanceError, match="changed since migration planning"):
        apply_migration_plan(plan)

    assert plan.manifest_path.read_bytes() == legacy_bytes + b" "
    assert not plan.descriptor_path.exists()
    assert not (tmp_path / ".dataset-provenance.lock").exists()


def test_existing_dataset_publication_lock_fails_closed(tmp_path: Path):
    legacy_bytes, _ = _create_legacy_dataset(tmp_path)
    plan = build_migration_plan(tmp_path, created_at="2026-07-23T00:00:00Z")
    (tmp_path / ".dataset-provenance.lock").write_text("other writer", encoding="utf-8")

    with pytest.raises(ProvenanceError, match="publication lock"):
        apply_migration_plan(plan)

    assert plan.manifest_path.read_bytes() == legacy_bytes


def test_atomic_replace_failure_does_not_replace_active_manifest(tmp_path: Path, monkeypatch):
    legacy_bytes, _ = _create_legacy_dataset(tmp_path)
    plan = build_migration_plan(tmp_path, created_at="2026-07-23T00:00:00Z")

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("injected atomic replace failure")

    monkeypatch.setattr("tools.upgrade_dataset_provenance.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected atomic replace failure"):
        apply_migration_plan(plan)

    assert (tmp_path / "reports" / "dataset_manifest.jsonl").read_bytes() == legacy_bytes
    assert not (tmp_path / "reports" / "dataset_provenance.json").exists()


def test_corrupt_descriptor_temp_is_rejected_and_preflight_stays_closed(
    tmp_path: Path, monkeypatch
):
    _create_legacy_dataset(tmp_path)
    plan = build_migration_plan(tmp_path, created_at="2026-07-23T00:00:00Z")
    real_atomic_replace = upgrade_tool._atomic_replace_bytes

    def corrupt_descriptor(destination: Path, payload: bytes, validator=None) -> None:
        if destination.name == "dataset_provenance.json":
            payload = b'{"schema_id":"corrupt"}\n'
        real_atomic_replace(destination, payload, validator)

    monkeypatch.setattr(upgrade_tool, "_atomic_replace_bytes", corrupt_descriptor)
    with pytest.raises(ProvenanceError, match="Incomplete provenance publication"):
        apply_migration_plan(plan)

    assert not (tmp_path / "reports" / "dataset_provenance.json").exists()
    with pytest.raises(ProvenanceError, match="descriptor"):
        verify_dataset_provenance(tmp_path)


def test_report_index_is_descriptor_bound_and_tamper_detected(tmp_path: Path):
    _create_legacy_dataset(tmp_path)
    result = migrate_dataset_provenance(
        tmp_path, write=True, created_at="2026-07-23T00:00:00Z", git_commit="abc123"
    )
    index_path = tmp_path / "reports" / "report_index.jsonl"

    assert result.descriptor["report_index"]["sha256"] == sha256_bytes(
        index_path.read_bytes()
    )
    index_path.write_bytes(index_path.read_bytes() + b" ")
    with pytest.raises(ProvenanceError, match="report index"):
        verify_dataset_provenance(tmp_path)


def test_report_index_failure_leaves_unaccepted_partial_state(tmp_path: Path, monkeypatch):
    _create_legacy_dataset(tmp_path)
    plan = build_migration_plan(tmp_path, created_at="2026-07-23T00:00:00Z")
    real_atomic_replace = upgrade_tool._atomic_replace_bytes

    def fail_index(destination: Path, payload: bytes, validator=None) -> None:
        if destination.name == "report_index.jsonl":
            raise OSError("injected report index failure")
        real_atomic_replace(destination, payload, validator)

    monkeypatch.setattr(upgrade_tool, "_atomic_replace_bytes", fail_index)
    with pytest.raises(ProvenanceError, match="Incomplete provenance publication"):
        apply_migration_plan(plan)

    assert not plan.descriptor_path.exists()
    with pytest.raises(ProvenanceError, match="descriptor|report index"):
        verify_dataset_provenance(tmp_path)
    with pytest.raises(ProvenanceError, match="descriptor|partial"):
        build_migration_plan(tmp_path)


def test_migration_rejects_legacy_hash_mismatch(tmp_path: Path):
    _create_legacy_dataset(tmp_path)
    manifest_path = tmp_path / "reports" / "dataset_manifest.jsonl"
    rows = load_jsonl_strict(manifest_path.read_bytes(), source=str(manifest_path))
    rows[0]["sha256"] = "0" * 64
    manifest_path.write_bytes(canonical_jsonl_bytes(rows))

    with pytest.raises(ProvenanceError, match="legacy sha256"):
        build_migration_plan(tmp_path)


def test_legacy_manifest_rejects_orphan_descriptor(tmp_path: Path):
    legacy_bytes, _ = _create_legacy_dataset(tmp_path)
    descriptor_path = tmp_path / "reports" / "dataset_provenance.json"
    descriptor_path.write_bytes(b'{"orphan":true}\n')

    with pytest.raises(ProvenanceError, match="orphan descriptor"):
        build_migration_plan(tmp_path)

    assert (tmp_path / "reports" / "dataset_manifest.jsonl").read_bytes() == legacy_bytes
    assert descriptor_path.read_bytes() == b'{"orphan":true}\n'


def test_reports_parent_symlink_is_rejected_before_external_write(tmp_path: Path):
    _create_single_legacy_dataset(tmp_path, "201")
    reports = tmp_path / "reports"
    manifest_bytes = (reports / "dataset_manifest.jsonl").read_bytes()
    (reports / "dataset_manifest.jsonl").unlink()
    reports.rmdir()
    redirected = tmp_path / "redirected_reports"
    redirected.mkdir()
    (redirected / "dataset_manifest.jsonl").write_bytes(manifest_bytes)
    try:
        os.symlink(redirected, reports, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlink creation not permitted")

    with pytest.raises(ProvenanceError, match="symlink|reparse"):
        build_migration_plan(tmp_path)

    assert not (redirected / "dataset_provenance.json").exists()
    assert not (redirected / "archive").exists()


def test_descriptor_is_strict_json_and_binds_archive(tmp_path: Path):
    legacy_bytes, _ = _create_legacy_dataset(tmp_path)
    result = migrate_dataset_provenance(
        tmp_path, write=True, created_at="2026-07-23T00:00:00Z", git_commit=None
    )
    descriptor_bytes = (tmp_path / "reports" / "dataset_provenance.json").read_bytes()
    descriptor = load_json_strict(descriptor_bytes, source="descriptor")

    assert descriptor == result.descriptor
    assert descriptor["legacy_manifest"]["sha256"] == sha256_bytes(legacy_bytes)
    assert descriptor["created_at"] == "2026-07-23T00:00:00Z"


def test_dataset_without_historical_reports_omits_optional_status_index(tmp_path: Path):
    _create_legacy_dataset(tmp_path)
    (tmp_path / "reports" / "quality_report.json").unlink()

    result = migrate_dataset_provenance(
        tmp_path, write=True, created_at="2026-07-23T00:00:00Z", git_commit="abc123"
    )

    assert result.report_index_sha256 is None
    assert not (tmp_path / "reports" / "report_index.jsonl").exists()
    assert verify_dataset_provenance(tmp_path).document_count == 2


def test_training_loader_requires_verified_v2_manifest_and_explicit_eligibility(tmp_path: Path):
    _create_legacy_dataset(tmp_path)

    with pytest.raises(ProvenanceError, match="v2|descriptor|schema"):
        load_ner_training_documents(tmp_path)

    manifest_path = tmp_path / "reports" / "dataset_manifest.jsonl"
    rows = load_jsonl_strict(manifest_path.read_bytes(), source=str(manifest_path))
    migrate_dataset_provenance(
        tmp_path, write=True, created_at="2026-07-23T00:00:00Z", git_commit="abc123"
    )

    documents = load_ner_training_documents(tmp_path)

    assert documents == []


@pytest.mark.parametrize(
    ("document_id", "wrong_source"),
    [("1", "synthetic"), ("101", "reconstructed"), ("201", "organizer_gt")],
)
def test_migration_and_loader_reject_wrong_policy_in_every_id_range(
    tmp_path: Path, document_id: str, wrong_source: str
):
    migration_root = tmp_path / "migration"
    _create_single_legacy_dataset(
        migration_root, document_id, source_bucket=wrong_source
    )
    with pytest.raises(ProvenanceError, match="policy"):
        build_migration_plan(migration_root)

    loader_root = tmp_path / "loader"
    _create_single_legacy_dataset(loader_root, document_id)
    migrate_dataset_provenance(
        loader_root,
        write=True,
        created_at="2026-07-23T00:00:00Z",
        git_commit="abc123",
    )
    manifest_path = loader_root / "reports" / "dataset_manifest.jsonl"
    rows = load_jsonl_strict(manifest_path.read_bytes(), source=str(manifest_path))
    rows[0]["source_bucket"] = wrong_source
    manifest_path.write_bytes(canonical_jsonl_bytes(rows))
    with pytest.raises(ProvenanceError, match="policy"):
        load_ner_training_documents(loader_root)


def test_cli_returns_sanitized_error_without_clinical_text(tmp_path: Path, capsys):
    secret = "SUPER_SECRET_PATIENT_TEXT"
    entity_type = next(
        key
        for key, required in OFFICIAL_SCHEMA_KEYS.items()
        if required == {"text", "type", "position"}
    )
    _create_single_legacy_dataset(tmp_path, "201")
    payload = [{"text": secret, "type": entity_type, "position": [0, len(secret)]}]
    (tmp_path / "input" / "201.txt").write_bytes(b"x" * len(secret))
    (tmp_path / "gt" / "201.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    assert main(["--dataset-root", str(tmp_path), "--check"]) == 2
    output = capsys.readouterr()
    assert secret not in output.out
    assert secret not in output.err
