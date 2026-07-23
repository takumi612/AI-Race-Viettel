from __future__ import annotations

import hashlib
import json
import os
import stat
import sys
from types import SimpleNamespace
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import clinical_nlp_lab.provenance as provenance_module
from clinical_nlp_lab.provenance import (
    DATASET_ALGORITHM,
    LEGACY_SHA256_SEMANTICS,
    MANIFEST_ROW_SCHEMA_ID,
    PAIR_ALGORITHM,
    REPORT_ENVELOPE_SCHEMA_ID,
    ProvenanceError,
    build_provenance_descriptor,
    build_v2_manifest_rows,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compute_dataset_fingerprint,
    compute_legacy_input_text_sha256,
    compute_pair_sha256,
    detect_report_conflicts,
    detect_report_fact_conflicts,
    evaluate_report_status,
    load_json_strict,
    is_path_link_or_reparse,
    scan_dataset_layout,
    sha256_bytes,
    validate_v2_manifest,
    validate_report_envelope,
)
from clinical_nlp_lab.schema import OFFICIAL_SCHEMA_KEYS


def _policy(document_id: str) -> tuple[str, bool, bool, str | None]:
    numeric_id = int(document_id)
    if numeric_id <= 100:
        return "reconstructed", False, False, "fixture quarantine"
    if numeric_id <= 200:
        return "organizer_gt", True, False, None
    return "synthetic", True, True, None


def _legacy_row(document_id: str, input_bytes: bytes, **overrides: object) -> dict[str, object]:
    source_bucket, train_eligible, linking_train_eligible, exclusion_reason = _policy(
        document_id
    )
    row: dict[str, object] = {
        "document_id": document_id,
        "source_bucket": source_bucket,
        "genre": "fixture",
        "scenario": "fixture",
        "template_group": f"fixture:{document_id}",
        "long_tail": False,
        "train_eligible": train_eligible,
        "linking_train_eligible": linking_train_eligible,
        "train_exclusion_reason": exclusion_reason,
        "primary_candidates": [],
        "sha256": compute_legacy_input_text_sha256(input_bytes),
    }
    row.update(overrides)
    return row


def _write_dataset(root: Path, documents: dict[str, tuple[bytes, object]]) -> None:
    (root / "input").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "reports").mkdir()
    rows = []
    for document_id, (input_bytes, gt_payload) in documents.items():
        (root / "input" / f"{document_id}.txt").write_bytes(input_bytes)
        gt_bytes = json.dumps(gt_payload, ensure_ascii=False, indent=2).encode("utf-8") + b"\r\n"
        (root / "gt" / f"{document_id}.json").write_bytes(gt_bytes)
        rows.append(_legacy_row(document_id, input_bytes))
    (root / "reports" / "dataset_manifest.jsonl").write_bytes(canonical_jsonl_bytes(rows))


def _report_envelope(
    report_type: str,
    payload: object,
    fingerprints: dict[str, str],
    *,
    facts: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema_id": REPORT_ENVELOPE_SCHEMA_ID,
        "schema_version": 1,
        "report_type": report_type,
        "scope": {"document_ids": ["2", "1"]},
        "status_at_creation": "current",
        "created_at": "2026-07-23T00:00:00Z",
        "producer_version": "fixture-1",
        "validator_version": "fixture-1",
        "fingerprints": fingerprints,
        "facts": facts or {},
        "payload": payload,
        "payload_sha256": sha256_bytes(canonical_json_bytes(payload)),
    }


def test_raw_hash_contract_distinguishes_crlf_from_legacy_normalized_text():
    crlf = b"alpha\r\nbeta\r\n"
    lf = b"alpha\nbeta\n"

    assert sha256_bytes(crlf) != sha256_bytes(lf)
    assert compute_legacy_input_text_sha256(crlf) == sha256_bytes(lf)
    assert compute_pair_sha256("1", crlf, b"[]\r\n") != compute_pair_sha256(
        "1", lf, b"[]\r\n"
    )


def test_pair_and_dataset_hashes_frame_bytes_and_include_raw_gt(tmp_path: Path):
    # These byte sequences collide under delimiter-free concatenation.
    assert compute_pair_sha256("1", b"ab", b"c") != compute_pair_sha256("1", b"a", b"bc")

    _write_dataset(tmp_path, {"1": (b"text\r\n", [])})
    before = scan_dataset_layout(tmp_path)
    gt_path = tmp_path / "gt" / "1.json"
    gt_path.write_bytes(gt_path.read_bytes() + b" ")
    after = scan_dataset_layout(tmp_path)

    assert before.pairs[0].gt_sha256 != after.pairs[0].gt_sha256
    assert before.dataset_fingerprint != after.dataset_fingerprint
    assert PAIR_ALGORITHM == "clinical-nlp-dataset-pair/v1"
    assert DATASET_ALGORITHM == "clinical-nlp-dataset/v1"


def test_layout_uses_numeric_order_not_lexicographic(tmp_path: Path):
    _write_dataset(tmp_path, {"10": (b"ten", []), "2": (b"two", [])})

    snapshot = scan_dataset_layout(tmp_path)

    assert [pair.document_id for pair in snapshot.pairs] == ["2", "10"]
    assert snapshot.dataset_fingerprint == compute_dataset_fingerprint(snapshot.pairs)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda root: (root / "input" / "nested").mkdir(), "nested"),
        (lambda root: (root / "input" / "note.md").write_text("x"), "unexpected"),
        (
            lambda root: (root / "input" / "01.txt").write_text("x", encoding="utf-8"),
            "canonical",
        ),
    ],
)
def test_layout_rejects_nested_unexpected_and_leading_zero_payloads(
    tmp_path: Path, mutate, message: str
):
    _write_dataset(tmp_path, {"1": (b"one", [])})
    mutate(tmp_path)

    with pytest.raises(ProvenanceError, match=message):
        scan_dataset_layout(tmp_path)


def test_layout_rejects_symlink_payload(tmp_path: Path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    _write_dataset(tmp_path, {"1": (b"one", [])})
    target = tmp_path / "elsewhere.txt"
    target.write_text("elsewhere", encoding="utf-8")
    try:
        os.symlink(target, tmp_path / "input" / "2.txt")
    except OSError:
        pytest.skip("symlink creation not permitted")

    with pytest.raises(ProvenanceError, match="symlink"):
        scan_dataset_layout(tmp_path)


def test_windows_reparse_attribute_is_rejected_by_stdlib_probe(
    tmp_path: Path, monkeypatch
):
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    fake_stat = SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=reparse_flag)
    monkeypatch.setattr(provenance_module.os, "lstat", lambda _path: fake_stat)

    assert is_path_link_or_reparse(tmp_path) is True


def test_layout_rejects_pairing_mismatch_and_malformed_payload(tmp_path: Path):
    _write_dataset(tmp_path, {"1": (b"one", [])})
    (tmp_path / "gt" / "1.json").unlink()
    with pytest.raises(ProvenanceError, match="pairing mismatch"):
        scan_dataset_layout(tmp_path)

    (tmp_path / "gt" / "1.json").write_bytes(b"not-json")
    with pytest.raises(ProvenanceError, match="invalid_json"):
        scan_dataset_layout(tmp_path)

    (tmp_path / "gt" / "1.json").write_bytes(b"[]")
    (tmp_path / "input" / "1.txt").write_bytes(b"\xff")
    with pytest.raises(ProvenanceError, match="strict UTF-8"):
        scan_dataset_layout(tmp_path)


@pytest.mark.parametrize(
    "invalid_position",
    ([False, 1], [0.0, 1], [0, 1.0], ["0", 1], [0, "1"]),
)
def test_gt_rejects_non_integer_offset_scalars(tmp_path: Path, invalid_position: list[object]):
    lab_type = next(
        entity_type
        for entity_type, keys in OFFICIAL_SCHEMA_KEYS.items()
        if keys == {"text", "type", "position"}
    )
    payload = [{"text": "x", "type": lab_type, "position": invalid_position}]
    _write_dataset(tmp_path, {"201": (b"x", payload)})

    with pytest.raises(ProvenanceError, match="invalid_position_type"):
        scan_dataset_layout(tmp_path)


def test_gt_rejects_non_string_text_and_candidate_members(tmp_path: Path):
    disease_type = next(
        entity_type
        for entity_type, keys in OFFICIAL_SCHEMA_KEYS.items()
        if "candidates" in keys
    )
    numeric_text = [
        {
            "text": 1,
            "type": disease_type,
            "position": [0, 1],
            "assertions": [],
            "candidates": ["C1"],
        }
    ]
    _write_dataset(tmp_path / "text", {"201": (b"1", numeric_text)})
    with pytest.raises(ProvenanceError, match="invalid_text_type"):
        scan_dataset_layout(tmp_path / "text")

    numeric_candidate = [
        {
            "text": "x",
            "type": disease_type,
            "position": [0, 1],
            "assertions": [],
            "candidates": [123],
        }
    ]
    _write_dataset(tmp_path / "candidate", {"201": (b"x", numeric_candidate)})
    with pytest.raises(ProvenanceError, match="invalid_candidate_type"):
        scan_dataset_layout(tmp_path / "candidate")


def test_gt_offset_errors_do_not_disclose_clinical_text(tmp_path: Path):
    lab_type = next(
        entity_type
        for entity_type, keys in OFFICIAL_SCHEMA_KEYS.items()
        if keys == {"text", "type", "position"}
    )
    secret = "SUPER_SECRET_PATIENT_TEXT"
    payload = [{"text": secret, "type": lab_type, "position": [0, len(secret)]}]
    _write_dataset(tmp_path, {"201": (b"x" * len(secret), payload)})

    with pytest.raises(ProvenanceError, match="offset_text_mismatch") as caught:
        scan_dataset_layout(tmp_path)

    assert secret not in str(caught.value)
    assert "x" * len(secret) not in str(caught.value)

def test_strict_json_rejects_duplicate_keys_and_nonfinite_values():
    with pytest.raises(ProvenanceError, match="duplicate JSON key"):
        load_json_strict(b'{"a":1,"a":2}', source="fixture")
    with pytest.raises(ProvenanceError, match="non-finite"):
        load_json_strict(b'{"a":NaN}', source="fixture")


def test_canonical_json_and_manifest_bytes_are_stable(tmp_path: Path):
    assert canonical_json_bytes({"b": 1, "a": "é"}) == '{"a":"é","b":1}\n'.encode()

    _write_dataset(tmp_path, {"1": (b"one\r\n", [])})
    snapshot = scan_dataset_layout(tmp_path)
    rows = build_v2_manifest_rows(
        [_legacy_row("1", b"one\r\n", custom={"z": 2, "a": 1})], snapshot
    )
    manifest_bytes = canonical_jsonl_bytes(rows)

    assert manifest_bytes.endswith(b"\n") and not manifest_bytes.endswith(b"\n\n")
    assert b'"schema_id":"clinical_nlp.dataset_manifest_row"' in manifest_bytes
    assert sha256_bytes(manifest_bytes) == hashlib.sha256(manifest_bytes).hexdigest()
    assert validate_v2_manifest(rows, snapshot) == rows


def test_manifest_validation_fails_closed_on_eligibility_source_and_hash(tmp_path: Path):
    _write_dataset(tmp_path, {"1": (b"one", [])})
    snapshot = scan_dataset_layout(tmp_path)
    base = build_v2_manifest_rows([_legacy_row("1", b"one")], snapshot)[0]

    for key in ("train_eligible", "source_bucket"):
        invalid = dict(base)
        del invalid[key]
        with pytest.raises(ProvenanceError, match=key):
            validate_v2_manifest([invalid], snapshot)

    invalid = dict(base, input_sha256="0" * 64)
    with pytest.raises(ProvenanceError, match="input_sha256"):
        validate_v2_manifest([invalid], snapshot)

    with pytest.raises(ProvenanceError, match="Duplicate manifest document_id"):
        validate_v2_manifest([base, base], snapshot)


@pytest.mark.parametrize(
    ("document_id", "field", "invalid_value"),
    [
        ("1", "source_bucket", "synthetic"),
        ("1", "train_eligible", True),
        ("101", "source_bucket", "reconstructed"),
        ("101", "train_eligible", False),
        ("201", "source_bucket", "organizer_gt"),
        ("201", "train_eligible", False),
    ],
)
def test_legacy_and_v2_manifest_enforce_numeric_id_source_and_eligibility_policy(
    tmp_path: Path, document_id: str, field: str, invalid_value: object
):
    input_bytes = f"document {document_id}".encode()
    _write_dataset(tmp_path, {document_id: (input_bytes, [])})
    snapshot = scan_dataset_layout(tmp_path)
    invalid_legacy = _legacy_row(document_id, input_bytes)
    invalid_legacy[field] = invalid_value

    with pytest.raises(ProvenanceError, match="policy"):
        build_v2_manifest_rows([invalid_legacy], snapshot)

    valid_v2 = build_v2_manifest_rows(
        [_legacy_row(document_id, input_bytes)], snapshot
    )[0]
    invalid_v2 = dict(valid_v2)
    invalid_v2[field] = invalid_value
    with pytest.raises(ProvenanceError, match="policy"):
        validate_v2_manifest([invalid_v2], snapshot)


def test_descriptor_publishes_detached_manifest_hash_without_self_hash(tmp_path: Path):
    _write_dataset(tmp_path, {"1": (b"one", [])})
    snapshot = scan_dataset_layout(tmp_path)
    rows = build_v2_manifest_rows([_legacy_row("1", b"one")], snapshot)
    manifest_bytes = canonical_jsonl_bytes(rows)
    descriptor = build_provenance_descriptor(
        snapshot,
        manifest_bytes,
        legacy_manifest_sha256="1" * 64,
        legacy_archive_path="reports/archive/dataset_manifest.legacy." + "1" * 64 + ".jsonl",
        created_at="2026-07-23T00:00:00Z",
        git_commit="abc123",
    )

    assert descriptor["manifest"]["sha256"] == sha256_bytes(manifest_bytes)
    assert descriptor["manifest"]["schema_id"] == MANIFEST_ROW_SCHEMA_ID
    assert descriptor["dataset"]["fingerprint"] == snapshot.dataset_fingerprint
    assert "descriptor_sha256" not in canonical_json_bytes(descriptor).decode("utf-8")


def test_report_status_is_recomputed_and_conflicts_are_typed():
    expected = {"dataset_fingerprint": "d1", "manifest_sha256": "m1"}
    missing = {
        "report_type": "quality_report",
        "scope": {"document_ids": ["2", "1"]},
        "status_at_creation": "current",
        "fingerprints": {"dataset_fingerprint": "d1"},
        "payload_sha256": "a" * 64,
    }
    mismatched = {
        **missing,
        "fingerprints": {"dataset_fingerprint": "d1", "manifest_sha256": "old"},
    }

    assert evaluate_report_status(missing, expected).reason_codes == ("missing_fingerprint",)
    mismatch_status = evaluate_report_status(mismatched, expected)
    assert mismatch_status.effective_status == "stale"
    assert mismatch_status.details[0]["expected"] == "m1"
    assert mismatch_status.details[0]["actual"] == "old"

    quality_a = _report_envelope("quality_report", {"score": 1}, expected)
    quality_b = _report_envelope("quality_report", {"score": 2}, expected)
    diversity = _report_envelope("diversity_report", {"score": 2}, expected)
    conflicts = detect_report_conflicts([quality_a, quality_b, diversity], expected.keys())

    assert len(conflicts) == 1
    assert conflicts[0]["report_type"] == "quality_report"
    assert set(conflicts[0]["payload_sha256"]) == {
        quality_a["payload_sha256"],
        quality_b["payload_sha256"],
    }


def test_report_envelope_recomputes_or_requires_verified_payload_binding():
    expected = {"dataset_fingerprint": "d1"}
    first = _report_envelope("quality_report", {"score": 1}, expected)
    second = _report_envelope("quality_report", {"score": 2}, expected)
    second["payload_sha256"] = first["payload_sha256"]

    with pytest.raises(ProvenanceError, match="payload hash mismatch"):
        detect_report_conflicts([first, second], expected.keys())

    external = dict(first)
    del external["payload"]
    external["payload_path"] = "reports/quality_payload.json"
    with pytest.raises(ProvenanceError, match="verified payload binding"):
        validate_report_envelope(external)
    validated = validate_report_envelope(
        external,
        verified_payload_hashes={
            "reports/quality_payload.json": str(external["payload_sha256"])
        },
    )
    assert validated["payload_sha256"] == external["payload_sha256"]


def test_cross_report_consistency_compares_only_shared_canonical_facts():
    fingerprints = {"dataset_fingerprint": "d1"}
    quality = _report_envelope(
        "quality_report",
        {"quality": True},
        fingerprints,
        facts={"documents": 2, "entities": 10},
    )
    diversity = _report_envelope(
        "diversity_report",
        {"diversity": True},
        fingerprints,
        facts={"documents": 2, "genres": 3},
    )
    assert detect_report_fact_conflicts([quality, diversity], ["dataset_fingerprint"]) == ()

    diversity["facts"] = {"documents": 3, "genres": 3}
    conflicts = detect_report_fact_conflicts(
        [quality, diversity], ["dataset_fingerprint"]
    )
    assert len(conflicts) == 1
    assert conflicts[0]["fact"] == "documents"
