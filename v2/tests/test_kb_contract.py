from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest
from openpyxl import Workbook

from clinical_nlp_lab.kb import (
    RXNCONSO_FIELDS,
    build_icd10_dictionary,
    build_rxnorm_dictionary,
    iter_jsonl_gz,
    sha256_file,
)
from clinical_nlp_lab.kb_contract import (
    CandidateIdentity,
    KBContractError,
    OrganizerCandidateOccurrence,
    audit_gold_candidate_coverage,
    canonical_icd_id,
    official_output_id,
    validate_icd_record_identity,
)


ROOT = Path(__file__).parents[1]


def _occurrence(candidate_id: str, mention: str = "Example Drug", *, entity_index: int = 0):
    return OrganizerCandidateOccurrence(
        document_id="101",
        entity_index=entity_index,
        ontology="rxnorm",
        identity=CandidateIdentity(candidate_id, candidate_id),
        mention_text=mention,
        mention_sha256=hashlib.sha256(mention.encode()).hexdigest(),
    )


def _rrf_row(**overrides: str) -> bytes:
    record = {field: "" for field in RXNCONSO_FIELDS}
    record.update(
        {
            "RXCUI": "100",
            "LAT": "ENG",
            "RXAUI": "A100",
            "SAB": "RXNORM",
            "TTY": "SCD",
            "CODE": "C100",
            "STR": "Normal Drug",
            "SUPPRESS": "N",
        }
    )
    record.update(overrides)
    return ("|".join(record[field] for field in RXNCONSO_FIELDS) + "|\n").encode()


def _rx_zip(path: Path, rows: list[bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("rrf/RXNCONSO.RRF", b"".join(rows))
    return path


def _build_rx(tmp_path: Path, rows: list[bytes], requirements, *, expected=None):
    zip_path = _rx_zip(tmp_path / "rx.zip", rows)
    output = tmp_path / "rx.jsonl.gz"
    evidence = tmp_path / "evidence.jsonl.gz"
    metadata = build_rxnorm_dictionary(
        zip_path,
        "rrf/RXNCONSO.RRF",
        output,
        tmp_path / "metadata.json",
        ["ENG"],
        ["RXNORM"],
        ["SCD", "BN"],
        ["N"],
        organizer_requirements=requirements,
        dataset_pair_fingerprint="a" * 64,
        evidence_path=evidence,
        evidence_metadata_path=tmp_path / "evidence.metadata.json",
        expected_supplement_count=expected,
    )
    return metadata, output, evidence


def test_icd_suffix_marker_contract_and_round_trip():
    assert canonical_icd_id("K93.1*") == "K93.1"
    assert canonical_icd_id("B57.3†") == "B57.3"
    assert canonical_icd_id("A*B") == "A*B"
    record = {
        "candidate_id": "G99",
        "canonical_id": "G99",
        "official_display_ids": ["G99", "G99*"],
    }
    assert [item.official_display_id for item in validate_icd_record_identity(record)] == ["G99", "G99*"]
    assert official_output_id(record, "G99*").to_dict() == {
        "canonical_id": "G99",
        "official_display_id": "G99*",
    }
    with pytest.raises(KBContractError, match="Unknown official display"):
        official_output_id(record, "G99†")


def test_icd_builder_preserves_multi_display_identities(tmp_path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "ICD10"
    sheet.append([])
    sheet.append([])
    sheet.append(["MÃ BỆNH", "MÃ BỆNH KHÔNG DẤU", "DISEASE NAME", "TÊN BỆNH"])
    sheet.append(["G99", "G99", "Disease", "Bệnh"])
    sheet.append(["G99*", "G99", "Disease in elsewhere", "Bệnh trong bệnh khác"])
    sheet.append(["B57.3†", "B573", "Chagas disease", "Bệnh Chagas"])
    source = tmp_path / "icd.xlsx"
    workbook.save(source)
    output = tmp_path / "icd.jsonl.gz"
    metadata = build_icd10_dictionary(source, output, tmp_path / "metadata.json")
    records = {row["candidate_id"]: row for row in iter_jsonl_gz(output)}
    assert records["G99"]["official_display_ids"] == ["G99", "G99*"]
    assert records["B57.3"]["official_display_ids"] == ["B57.3†"]
    assert metadata["identity_pair_count"] == 3
    assert metadata["multi_display_canonical_count"] == 1


def test_bounded_rxnorm_supplement_leaves_normal_filter_unchanged(tmp_path: Path):
    rows = [
        _rrf_row(),
        _rrf_row(
            RXCUI="200",
            RXAUI="A200",
            CODE="C200",
            STR="Example Drug",
            SUPPRESS="O",
        ),
        _rrf_row(
            RXCUI="999",
            RXAUI="A999",
            CODE="C999",
            STR="Unrequested Drug",
            SUPPRESS="O",
        ),
    ]
    metadata, output, evidence = _build_rx(
        tmp_path, rows, {"200": (_occurrence("200"),)}, expected=1
    )
    records = {row["candidate_id"]: row for row in iter_jsonl_gz(output)}
    assert set(records) == {"100", "200"}
    assert not records["100"].get("linking_only", False)
    assert records["200"]["detection_aliases"] == []
    assert metadata["filters"]["SUPPRESS"] == ["N"]
    assert metadata["supplement_candidate_count"] == 1
    evidence_row = list(iter_jsonl_gz(evidence))[0]
    assert "mention_text" not in evidence_row
    assert evidence_row["SUPPRESS"] == "O"


@pytest.mark.parametrize(
    ("rows", "requirements", "message"),
    [
        ([_rrf_row()], {"200": (_occurrence("200"),)}, "found 0"),
        (
            [
                _rrf_row(RXCUI="200", STR="Example Drug", SUPPRESS="O"),
                _rrf_row(RXCUI="200", RXAUI="A2", STR="Example Drug", SUPPRESS="E"),
            ],
            {"200": (_occurrence("200"),)},
            "found 2",
        ),
        (
            [_rrf_row(RXCUI="200", STR="Other Drug", SUPPRESS="O")],
            {"200": (_occurrence("200"),)},
            "surface does not match",
        ),
        (
            [_rrf_row(RXCUI="200", STR="Example Drug", SAB="OTHER", SUPPRESS="O")],
            {"200": (_occurrence("200"),)},
            "found 0",
        ),
        (
            [_rrf_row(RXCUI="200", STR="Example Drug", TTY="IN", SUPPRESS="O")],
            {"200": (_occurrence("200"),)},
            "found 0",
        ),
    ],
)
def test_rxnorm_supplement_fail_closed(tmp_path: Path, rows, requirements, message):
    with pytest.raises(KBContractError, match=message):
        _build_rx(tmp_path, rows, requirements)


def test_duplicate_missing_organizer_occurrence_fails_closed(tmp_path: Path):
    rows = [_rrf_row(RXCUI="200", STR="Example Drug", SUPPRESS="O")]
    requirements = {"200": (_occurrence("200"), _occurrence("200", entity_index=1))}
    with pytest.raises(KBContractError, match="exactly one organizer occurrence"):
        _build_rx(tmp_path, rows, requirements)


def test_rxnorm_dictionary_and_evidence_are_byte_reproducible(tmp_path: Path):
    rows = [
        _rrf_row(),
        _rrf_row(RXCUI="200", RXAUI="A200", STR="Example Drug", SUPPRESS="O"),
    ]
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _, output_1, evidence_1 = _build_rx(first, rows, {"200": (_occurrence("200"),)})
    _, output_2, evidence_2 = _build_rx(second, rows, {"200": (_occurrence("200"),)})
    assert sha256_file(output_1) == sha256_file(output_2)
    assert sha256_file(evidence_1) == sha256_file(evidence_2)


def test_coverage_accepts_empty_abstention_and_checks_exact_icd_display():
    icd = [{"candidate_id": "G99", "official_display_ids": ["G99", "G99*"]}]
    rx = [{"candidate_id": "200"}]
    occurrences = [
        {
            "document_id": "101",
            "ontology": "icd10",
            "official_display_id": "G99*",
            "entity_index": 0,
        },
        {
            "document_id": "101",
            "ontology": "rxnorm",
            "official_display_id": "200",
            "entity_index": 1,
        },
    ]
    report = audit_gold_candidate_coverage(occurrences, icd, rx, abstentions=1)
    assert report["status"] == "PASS"
    assert report["covered_occurrences"] == 2
    assert report["abstentions"] == 1
    occurrences[0]["official_display_id"] = "G99†"
    assert audit_gold_candidate_coverage(occurrences, icd, rx)["status"] == "FAIL"


def test_current_canonical_artifacts_meet_pinned_contract_when_present():
    coverage_path = ROOT / "artifacts" / "kb_coverage_report.json"
    evidence_path = ROOT / "artifacts" / "rxnorm" / "organizer_supplement_evidence.jsonl.gz"
    if not coverage_path.is_file() or not evidence_path.is_file():
        pytest.skip("Canonical evidence-bounded artifacts have not been published yet")
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    assert coverage["status"] == "PASS"
    assert coverage["coverage"]["total_non_empty_occurrences"] == 2_239
    assert coverage["coverage"]["covered_occurrences"] == 2_239
    assert coverage["coverage"]["abstentions"] == 1_373
    assert len(list(iter_jsonl_gz(evidence_path))) == 258
