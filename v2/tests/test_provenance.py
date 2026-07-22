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
    DATASET_ALGORITHM,
    LEGACY_SHA256_SEMANTICS,
    MANIFEST_ROW_SCHEMA_ID,
    PAIR_ALGORITHM,
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
    scan_dataset_layout,
    sha256_bytes,
    validate_v2_manifest,
)


def _legacy_row(document_id: str, input_bytes: bytes, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "document_id": document_id,
        "source_bucket": "synthetic",
        "genre": "fixture",
        "scenario": "fixture",
        "template_group": f"fixture:{document_id}",
        "long_tail": False,
        "train_eligible": True,
        "linking_train_eligible": True,
        "train_exclusion_reason": None,
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


def test_layout_rejects_pairing_mismatch_and_malformed_payload(tmp_path: Path):
    _write_dataset(tmp_path, {"1": (b"one", [])})
    (tmp_path / "gt" / "1.json").unlink()
    with pytest.raises(ProvenanceError, match="pairing mismatch"):
        scan_dataset_layout(tmp_path)

    (tmp_path / "gt" / "1.json").write_bytes(b"not-json")
    with pytest.raises(ProvenanceError, match="Malformed JSON"):
        scan_dataset_layout(tmp_path)

    (tmp_path / "gt" / "1.json").write_bytes(b"[]")
    (tmp_path / "input" / "1.txt").write_bytes(b"\xff")
    with pytest.raises(ProvenanceError, match="strict UTF-8"):
        scan_dataset_layout(tmp_path)


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

    quality_a = {
        **mismatched,
        "fingerprints": expected,
        "payload_sha256": "a" * 64,
    }
    quality_b = {**quality_a, "payload_sha256": "b" * 64}
    diversity = {**quality_b, "report_type": "diversity_report"}
    conflicts = detect_report_conflicts([quality_a, quality_b, diversity], expected.keys())

    assert len(conflicts) == 1
    assert conflicts[0]["report_type"] == "quality_report"
    assert set(conflicts[0]["payload_sha256"]) == {"a" * 64, "b" * 64}


def test_cross_report_consistency_compares_only_shared_canonical_facts():
    base = {
        "scope": {"document_ids": ["1", "2"]},
        "status_at_creation": "current",
        "fingerprints": {"dataset_fingerprint": "d1"},
    }
    quality = {
        **base,
        "report_type": "quality_report",
        "facts": {"documents": 2, "entities": 10},
    }
    diversity = {
        **base,
        "report_type": "diversity_report",
        "facts": {"documents": 2, "genres": 3},
    }
    assert detect_report_fact_conflicts([quality, diversity], ["dataset_fingerprint"]) == ()

    diversity["facts"] = {"documents": 3, "genres": 3}
    conflicts = detect_report_fact_conflicts(
        [quality, diversity], ["dataset_fingerprint"]
    )
    assert len(conflicts) == 1
    assert conflicts[0]["fact"] == "documents"
