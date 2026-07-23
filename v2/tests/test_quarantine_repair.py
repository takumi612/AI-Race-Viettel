from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

import clinical_nlp_lab.quarantine_repair as repair_module
from clinical_nlp_lab.provenance import (
    canonical_jsonl_bytes,
    compute_legacy_input_text_sha256,
    load_json_strict,
    scan_dataset_layout,
)
from clinical_nlp_lab.quarantine_repair import (
    CandidateRepairRule,
    QuarantineRepairError,
    apply_quarantine_repair_plan,
    build_quarantine_repair_plan,
    repair_quarantine_gt,
)


ROOT = Path(__file__).parents[1]


def _gt_bytes(candidates: list[str]) -> bytes:
    payload = [
        {
            "text": "alpha",
            "type": "CHẨN_ĐOÁN",
            "position": [0, 5],
            "assertions": [],
            "candidates": candidates,
        }
    ]
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    return (text.replace("\n", "\r\n") + "\r\n").encode("utf-8")


def _write_fixture(root: Path, *, document_id: str = "1", candidates=None) -> Path:
    candidates = ["A00", "A01"] if candidates is None else candidates
    (root / "input").mkdir(parents=True)
    (root / "gt").mkdir()
    (root / "reports").mkdir()
    input_bytes = b"alpha"
    (root / "input" / f"{document_id}.txt").write_bytes(input_bytes)
    (root / "gt" / f"{document_id}.json").write_bytes(_gt_bytes(candidates))
    numeric_id = int(document_id)
    if numeric_id <= 100:
        source, eligible, linking, reason = "reconstructed", False, False, "fixture quarantine"
    elif numeric_id <= 200:
        source, eligible, linking, reason = "organizer_gt", True, False, None
    else:
        source, eligible, linking, reason = "synthetic", True, True, None
    row = {
        "document_id": document_id,
        "source_bucket": source,
        "genre": "fixture",
        "scenario": "fixture",
        "template_group": f"fixture:{document_id}",
        "long_tail": False,
        "train_eligible": eligible,
        "linking_train_eligible": linking,
        "train_exclusion_reason": reason,
        "primary_candidates": [],
        "sha256": compute_legacy_input_text_sha256(input_bytes),
    }
    (root / "reports" / "dataset_manifest.jsonl").write_bytes(canonical_jsonl_bytes([row]))
    icd_path = root / "icd.jsonl.gz"
    with gzip.GzipFile(filename=str(icd_path), mode="wb", mtime=0) as stream:
        for code in ("A00", "A01"):
            stream.write(
                (json.dumps({"candidate_id": code, "canonical_id": code}) + "\n").encode(
                    "utf-8"
                )
            )
    return icd_path


def _rule(document_id: str = "1") -> CandidateRepairRule:
    return CandidateRepairRule(
        document_id=document_id,
        entity_index=0,
        expected_candidates=("A00", "A01"),
        replacement_candidates=("A00",),
        evidence_code="fixture_exact_disambiguation",
    )


def test_check_builds_fingerprint_bound_plan_without_writing(tmp_path: Path):
    icd = _write_fixture(tmp_path)
    before = scan_dataset_layout(tmp_path).dataset_fingerprint
    original = (tmp_path / "gt" / "1.json").read_bytes()

    result = repair_quarantine_gt(tmp_path, icd, rules=[_rule()])

    assert result.mode == "check"
    assert result.migration_required is True
    assert result.repaired_entity_count == 1
    assert result.dataset_fingerprint_before == before
    assert result.dataset_fingerprint_after != before
    assert (tmp_path / "gt" / "1.json").read_bytes() == original
    assert not (tmp_path / "reports" / "quarantine_gt_repair.json").exists()


def test_write_is_audited_recoverable_and_idempotent(tmp_path: Path):
    icd = _write_fixture(tmp_path)

    first = repair_quarantine_gt(tmp_path, icd, write=True, rules=[_rule()])

    payload = load_json_strict((tmp_path / "gt" / "1.json").read_bytes(), source="fixture")
    assert payload[0]["candidates"] == ["A00"]
    assert scan_dataset_layout(tmp_path).dataset_fingerprint == first.dataset_fingerprint_after
    evidence_path = tmp_path / str(first.evidence_path)
    evidence = load_json_strict(evidence_path.read_bytes(), source="evidence")
    assert evidence["repair_count"] == 1
    assert "alpha" not in evidence_path.read_text(encoding="utf-8")
    archive_files = list((tmp_path / "reports" / "archive").rglob("*.json"))
    assert len(archive_files) == 1

    second = repair_quarantine_gt(tmp_path, icd, write=True, rules=[_rule()])
    assert second.migration_required is False
    assert second.repaired_entity_count == 0


def test_non_quarantine_target_is_rejected(tmp_path: Path):
    icd = _write_fixture(tmp_path, document_id="101")
    with pytest.raises(QuarantineRepairError, match="non-quarantine"):
        build_quarantine_repair_plan(tmp_path, icd, rules=[_rule("101")])


def test_changed_gt_after_planning_is_rejected_without_overwrite(tmp_path: Path):
    icd = _write_fixture(tmp_path)
    plan = build_quarantine_repair_plan(tmp_path, icd, rules=[_rule()])
    gt_path = tmp_path / "gt" / "1.json"
    changed = gt_path.read_bytes() + b" "
    gt_path.write_bytes(changed)

    with pytest.raises(QuarantineRepairError, match="changed"):
        apply_quarantine_repair_plan(plan)
    assert gt_path.read_bytes() == changed


def test_publication_failure_rolls_back_gt(tmp_path: Path, monkeypatch):
    icd = _write_fixture(tmp_path)
    plan = build_quarantine_repair_plan(tmp_path, icd, rules=[_rule()])
    original = (tmp_path / "gt" / "1.json").read_bytes()
    real_replace = repair_module._atomic_replace

    def fail_evidence(path: Path, payload: bytes) -> None:
        if path.name == "quarantine_gt_repair.json":
            raise OSError("injected publication failure")
        real_replace(path, payload)

    monkeypatch.setattr(repair_module, "_atomic_replace", fail_evidence)
    with pytest.raises(OSError, match="injected"):
        apply_quarantine_repair_plan(plan)

    assert (tmp_path / "gt" / "1.json").read_bytes() == original
    assert not (tmp_path / "reports" / "quarantine_gt_repair.json").exists()


def test_real_dataset_check_finds_only_pinned_conflicts_without_mutation():
    dataset = ROOT.parent / "data_v2" / "Training_data" / "synthetic_train_v2"
    icd = ROOT / "artifacts" / "icd10" / "icd10_dictionary.jsonl.gz"
    if not dataset.is_dir() or not icd.is_file():
        pytest.skip("real workspace dataset is not attached")
    before = scan_dataset_layout(dataset).dataset_fingerprint

    result = repair_quarantine_gt(dataset, icd)

    assert result.mode == "check"
    if before == "65a01d7d658e1e79c5fb50494e38634469bb808313f3f5e0aea72ec11cab6c5d":
        assert result.repaired_entity_count == 13
        assert result.repaired_file_count == 12
    else:
        assert before == "18a391e51786630b482bb500d5129eb102ae144450d7fc18b149f2799054f028"
        assert result.repaired_entity_count == 0
        assert result.repaired_file_count == 0
        assert (dataset / "reports" / "quarantine_gt_repair.json").is_file()
    assert scan_dataset_layout(dataset).dataset_fingerprint == before
