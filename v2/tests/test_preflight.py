from __future__ import annotations

import json
import importlib.util
import subprocess
import sys
import types
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
package = sys.modules.get("clinical_nlp_lab")
if package is None:
    package = types.ModuleType("clinical_nlp_lab")
    package.__path__ = [str(ROOT / "clinical_nlp_lab")]
    sys.modules["clinical_nlp_lab"] = package

from clinical_nlp_lab.kb import write_jsonl_gz
from clinical_nlp_lab.preflight import (
    REPORT_SCOPE,
    REPORT_TYPE,
    audit_organizer_kb_coverage,
    build_preflight_report,
    canonicalize_icd10_id,
    inspect_dataset_layout,
)
from clinical_nlp_lab.provenance import (
    build_provenance_descriptor,
    build_v2_manifest_rows,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    compute_legacy_input_text_sha256,
    scan_dataset_layout,
    sha256_bytes,
)
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation


def _entity(
    text: str,
    entity_type: str = "CHẨN_ĐOÁN",
    candidate: str = "I10",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "text": text,
        "type": entity_type,
        "position": [0, len(text)],
    }
    if entity_type in {"CHẨN_ĐOÁN", "THUỐC"}:
        payload["assertions"] = []
        payload["candidates"] = [candidate] if candidate else []
    elif entity_type == "TRIỆU_CHỨNG":
        payload["assertions"] = []
    return payload


def _policy(document_id: str) -> tuple[str, bool]:
    numeric = int(document_id)
    if numeric <= 100:
        return "reconstructed", False
    if numeric <= 200:
        return "organizer_gt", True
    return "synthetic", True


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(payload))


def _make_fixture(
    tmp_path: Path,
    *,
    ids: tuple[str, ...] = ("101", "201"),
    entities: dict[str, dict[str, object]] | None = None,
) -> dict[str, Path]:
    dataset = tmp_path / "dataset"
    reports = dataset / "reports"
    (dataset / "input").mkdir(parents=True)
    (dataset / "gt").mkdir()
    reports.mkdir()
    legacy_rows: list[dict[str, object]] = []
    for document_id in ids:
        item = (entities or {}).get(document_id)
        if item is None:
            item = _entity("drug", "THUỐC", "123") if document_id == "101" else _entity("disease")
        text = str(item["text"])
        input_bytes = text.encode("utf-8")
        (dataset / "input" / f"{document_id}.txt").write_bytes(input_bytes)
        _write_json(dataset / "gt" / f"{document_id}.json", [item])
        source_bucket, eligible = _policy(document_id)
        legacy_rows.append(
            {
                "document_id": document_id,
                "source_bucket": source_bucket,
                "train_eligible": eligible,
                "sha256": compute_legacy_input_text_sha256(input_bytes),
            }
        )

    snapshot = scan_dataset_layout(dataset)
    legacy_bytes = canonical_jsonl_bytes(legacy_rows)
    archive = reports / "archive" / f"dataset_manifest.legacy.{sha256_bytes(legacy_bytes)}.jsonl"
    archive.parent.mkdir()
    archive.write_bytes(legacy_bytes)
    rows = build_v2_manifest_rows(legacy_rows, snapshot)
    manifest = reports / "dataset_manifest.jsonl"
    manifest_bytes = canonical_jsonl_bytes(rows)
    manifest.write_bytes(manifest_bytes)
    descriptor = build_provenance_descriptor(
        snapshot,
        manifest_bytes,
        legacy_manifest_sha256=sha256_bytes(legacy_bytes),
        legacy_archive_path=f"reports/archive/{archive.name}",
        created_at="2026-07-23T00:00:00+00:00",
        git_commit=None,
    )
    _write_json(reports / "dataset_provenance.json", descriptor)

    artifacts = tmp_path / "artifacts"
    write_jsonl_gz(
        artifacts / "icd10" / "icd10_dictionary.jsonl.gz",
        [{"candidate_id": "I10"}],
    )
    write_jsonl_gz(
        artifacts / "rxnorm" / "rxnorm_dictionary.jsonl.gz",
        [{"candidate_id": "123"}],
    )
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "candidate_output_k": 1,
            "candidate_top_k": 20,
            "enable_regex_fallback": False,
        },
    )
    return {
        "dataset": dataset,
        "manifest": manifest,
        "descriptor": reports / "dataset_provenance.json",
        "artifacts": artifacts,
        "config": config,
    }


def _manifest_rows(paths: dict[str, Path]) -> list[dict[str, object]]:
    return [json.loads(line) for line in paths["manifest"].read_text(encoding="utf-8").splitlines()]


def _refresh_descriptor_manifest(paths: dict[str, Path], descriptor: dict[str, object] | None = None) -> None:
    payload = descriptor or json.loads(paths["descriptor"].read_text(encoding="utf-8"))
    manifest_bytes = paths["manifest"].read_bytes()
    manifest = payload["manifest"]
    assert isinstance(manifest, dict)
    manifest["sha256"] = sha256_bytes(manifest_bytes)
    manifest["size_bytes"] = len(manifest_bytes)
    _write_json(paths["descriptor"], payload)


def _codes(report: dict[str, object]) -> set[str]:
    return {str(error["code"]) for error in report["errors"]}  # type: ignore[index]


def test_exact_contract_passes_and_writes_atomic_report(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    output = tmp_path / "preflight.json"

    report = build_preflight_report(
        paths["dataset"], paths["artifacts"], paths["config"], output
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []
    assert report["source_bucket_counts"] == {"organizer": 1, "synthetic": 1}
    assert report["organizer_kb_coverage"]["coverage_rate"] == 1.0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "PASS"


@pytest.mark.parametrize("mutation", ["missing", "old_schema"])
def test_missing_or_old_detached_provenance_fails(tmp_path: Path, mutation: str):
    paths = _make_fixture(tmp_path)
    if mutation == "missing":
        paths["descriptor"].unlink()
    else:
        payload = json.loads(paths["descriptor"].read_text(encoding="utf-8"))
        payload["schema_version"] = 0
        _write_json(paths["descriptor"], payload)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert report["status"] == "FAIL"
    assert "E_DATASET_PROVENANCE" in _codes(report)


def test_changed_gt_raw_byte_is_detected(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    gt_path = paths["dataset"] / "gt" / "101.json"
    gt_path.write_bytes(gt_path.read_bytes() + b"\n")

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_DATASET_FILE_HASH" in _codes(report)


def test_pair_hash_mismatch_is_detected(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    rows = _manifest_rows(paths)
    rows[0]["pair_sha256"] = "0" * 64
    paths["manifest"].write_bytes(canonical_jsonl_bytes(rows))
    _refresh_descriptor_manifest(paths)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_DATASET_PAIR_HASH" in _codes(report)


def test_manifest_file_hash_mismatch_is_detected(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    descriptor = json.loads(paths["descriptor"].read_text(encoding="utf-8"))
    descriptor["manifest"]["sha256"] = "0" * 64
    _write_json(paths["descriptor"], descriptor)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_DATASET_MANIFEST_HASH" in _codes(report)


def test_missing_manifest_entry_fails_closed(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    paths["manifest"].write_bytes(canonical_jsonl_bytes(_manifest_rows(paths)[:-1]))
    _refresh_descriptor_manifest(paths)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_MANIFEST_ENTRY" in _codes(report)


def test_quarantine_cannot_fail_open(tmp_path: Path):
    paths = _make_fixture(tmp_path, ids=("1", "101"))
    rows = _manifest_rows(paths)
    rows[0]["train_eligible"] = True
    paths["manifest"].write_bytes(canonical_jsonl_bytes(rows))
    _refresh_descriptor_manifest(paths)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_SOURCE_POLICY" in _codes(report)


def test_bad_raw_file_hash_fails(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    rows = _manifest_rows(paths)
    rows[0]["input_sha256"] = "f" * 64
    paths["manifest"].write_bytes(canonical_jsonl_bytes(rows))
    _refresh_descriptor_manifest(paths)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_DATASET_FILE_HASH" in _codes(report)


def test_bad_offset_is_reported_without_clinical_text(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    gt_path = paths["dataset"] / "gt" / "101.json"
    payload = json.loads(gt_path.read_text(encoding="utf-8"))
    payload[0]["position"] = [1, 4]
    _write_json(gt_path, payload)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_GT_OFFSET" in _codes(report)
    assert "drug" not in json.dumps(report, ensure_ascii=False)


def test_assertion_on_lab_is_rejected(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    gt_path = paths["dataset"] / "gt" / "101.json"
    payload = [_entity("lab", "TÊN_XÉT_NGHIỆM", "")]
    payload[0]["assertions"] = ["isNegated"]
    _write_json(gt_path, payload)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_ASSERTION_SCOPE" in _codes(report)


def test_candidate_wrong_ontology_is_rejected(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    gt_path = paths["dataset"] / "gt" / "101.json"
    _write_json(gt_path, [_entity("disease", "CHẨN_ĐOÁN", "123")])

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_CANDIDATE_ONTOLOGY" in _codes(report)


def test_quarantine_semantic_error_is_labeled_by_source_bucket(tmp_path: Path):
    paths = _make_fixture(tmp_path, ids=("1", "101"))
    gt_path = paths["dataset"] / "gt" / "1.json"
    _write_json(gt_path, [_entity("disease", "CHẨN_ĐOÁN", "123")])

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    errors = [error for error in report["errors"] if error["code"] == "E_CANDIDATE_ONTOLOGY"]
    assert errors[0]["source_bucket"] == "quarantine"
    assert report["schema_offset_validation"]["error_counts_by_source_bucket"] == {
        "quarantine": 1
    }


def test_icd_marker_is_canonicalized_for_lookup_and_display_is_preserved():
    document = ClinicalDocument(
        "101",
        "",
        entities=[EntityAnnotation("", "DISEASE", (0, 0), candidates=["I10†"])],
    )

    coverage = audit_organizer_kb_coverage([document], {"I10"}, set())

    assert canonicalize_icd10_id("I10*") == "I10"
    assert coverage["status"] == "PASS"
    assert coverage["icd10"]["missing_occurrences"] == []


def test_missing_rxnorm_candidate_is_a_hard_error(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    write_jsonl_gz(
        paths["artifacts"] / "rxnorm" / "rxnorm_dictionary.jsonl.gz",
        [{"candidate_id": "999"}],
    )

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_ORGANIZER_KB_COVERAGE" in _codes(report)
    assert report["organizer_kb_coverage"]["rxnorm"]["missing_unique_ids"] == ["123"]


def test_malformed_runtime_kb_is_a_contract_failure_not_an_operational_crash(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    write_jsonl_gz(paths["artifacts"] / "rxnorm" / "rxnorm_dictionary.jsonl.gz", [42])

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_RUNTIME_KB" in _codes(report)


def test_stale_report_is_inventoried_but_not_pass_evidence(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    _write_json(
        paths["dataset"] / "reports" / "old_preflight_report.json",
        {
            "schema_id": "clinical_nlp.preflight_report",
            "schema_version": 1,
            "report_type": REPORT_TYPE,
            "scope": REPORT_SCOPE,
            "status": "PASS",
            "dataset_pair_fingerprint": "0" * 64,
        },
    )

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert report["status"] == "PASS"
    assert report["prior_report_inventory"] == [
        {
            "relative_path": "old_preflight_report.json",
            "effective_status": "stale",
            "reason": "dataset_fingerprint_mismatch",
        }
    ]
    assert report["warnings"][0]["code"] == "W_PRIOR_REPORTS_NOT_CURRENT"


def test_same_type_scope_and_fingerprint_reports_with_divergent_payloads_conflict(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    baseline = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])
    left = json.loads(json.dumps(baseline))
    right = json.loads(json.dumps(baseline))
    right["status"] = "FAIL"
    right["errors"] = [{"code": "E_DIFFERENT", "message": "Different evidence."}]
    left.pop("payload_sha256", None)
    right.pop("payload_sha256", None)
    _write_json(paths["dataset"] / "reports" / "preflight_a.json", left)
    _write_json(paths["dataset"] / "reports" / "preflight_b.json", right)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert "E_REPORT_CONFLICT" in _codes(report)
    assert len(report["report_conflicts"]) == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [("candidate_top_k", 10), ("candidate_output_k", 2), ("enable_regex_fallback", True)],
)
def test_linking_config_contract_is_exact(tmp_path: Path, field: str, value: object):
    paths = _make_fixture(tmp_path)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config[field] = value
    _write_json(paths["config"], config)

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    matching = [error for error in report["errors"] if error["code"] == "E_CONFIG_CONTRACT"]
    assert any(error.get("field") == field for error in matching)


def test_inspection_exposes_provenance_identity_without_internal_documents(tmp_path: Path):
    paths = _make_fixture(tmp_path)

    inspection = inspect_dataset_layout(paths["dataset"])

    assert inspection["status"] == "PASS"
    assert len(inspection["dataset_pair_fingerprint"]) == 64
    assert inspection["provenance_schema"]["schema_version"] == 1
    assert "_documents" not in inspection


def test_independent_hard_errors_are_aggregated(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    paths["descriptor"].unlink()
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["candidate_top_k"] = 10
    _write_json(paths["config"], config)
    write_jsonl_gz(
        paths["artifacts"] / "rxnorm" / "rxnorm_dictionary.jsonl.gz",
        [{"candidate_id": "999"}],
    )

    report = build_preflight_report(paths["dataset"], paths["artifacts"], paths["config"])

    assert {
        "E_DATASET_PROVENANCE",
        "E_CONFIG_CONTRACT",
        "E_ORGANIZER_KB_COVERAGE",
    } <= _codes(report)


def test_cli_exit_codes_and_summary_json(tmp_path: Path):
    paths = _make_fixture(tmp_path)
    output = tmp_path / "cli-report.json"
    command = [
        sys.executable,
        str(ROOT / "tools" / "preflight_pipeline.py"),
        "--dataset",
        str(paths["dataset"]),
        "--artifacts",
        str(paths["artifacts"]),
        "--config",
        str(paths["config"]),
        "--output",
        str(output),
    ]

    passed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)
    config = json.loads(paths["config"].read_text(encoding="utf-8"))
    config["candidate_output_k"] = 2
    _write_json(paths["config"], config)
    failed = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True)

    assert passed.returncode == 0
    assert json.loads(passed.stdout)["status"] == "PASS"
    assert failed.returncode == 2
    assert json.loads(failed.stdout)["status"] == "FAIL"


def test_cli_unexpected_operation_returns_one_with_safe_json(monkeypatch, capsys, tmp_path: Path):
    spec = importlib.util.spec_from_file_location(
        "preflight_pipeline_for_test", ROOT / "tools" / "preflight_pipeline.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fail(*_args, **_kwargs):
        raise RuntimeError("sensitive operational detail")

    monkeypatch.setattr(module, "build_preflight_report", fail)
    exit_code = module.main(
        [
            "--dataset",
            str(tmp_path / "dataset"),
            "--artifacts",
            str(tmp_path / "artifacts"),
            "--config",
            str(tmp_path / "config.json"),
            "--output",
            str(tmp_path / "report.json"),
        ]
    )
    summary = json.loads(capsys.readouterr().out)

    assert exit_code == 1
    assert summary["status"] == "ERROR"
    assert "sensitive" not in json.dumps(summary)


def test_cpu_preflight_sources_do_not_import_model_frameworks():
    source = (ROOT / "clinical_nlp_lab" / "preflight.py").read_text(encoding="utf-8")
    cli_source = (ROOT / "tools" / "preflight_pipeline.py").read_text(encoding="utf-8")

    for prohibited in ("import torch", "import transformers", "import sentence_transformers"):
        assert prohibited not in source
        assert prohibited not in cli_source
