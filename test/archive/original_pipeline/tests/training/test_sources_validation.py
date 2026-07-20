from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from src.training.contracts import CanonicalEntity, CanonicalRecord
from src.training.sources import SourceSpec, load_source_records
from src.training.validation import require_valid_source, validate_records


def _write_json(path: Path, value) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False),
        encoding="utf-8",
    )


def _make_source(
    tmp_path: Path,
    records: dict[str, tuple[str, list[dict]]],
    *,
    name: str = "source",
) -> Path:
    root = tmp_path / name
    input_dir = root / "input"
    gt_dir = root / "gt"
    input_dir.mkdir(parents=True)
    gt_dir.mkdir()
    for record_id, (text, entities) in records.items():
        (input_dir / f"{record_id}.txt").write_text(text, encoding="utf-8")
        _write_json(gt_dir / f"{record_id}.json", entities)
    return root


def _source_spec(
    root: Path,
    expected_ids: tuple[str, ...],
    *,
    name: str = "trusted",
    trust_tier: str = "trusted",
    manifest_path: Path | None = None,
) -> SourceSpec:
    return SourceSpec(
        name=name,
        root=root,
        trust_tier=trust_tier,
        expected_ids=expected_ids,
        manifest_path=manifest_path,
    )


def _build_metadata_db(
    tmp_path: Path,
    *,
    icd: tuple[str, ...] = ("I10",),
    rxnorm: tuple[str, ...] = ("6809",),
) -> Path:
    path = tmp_path / "metadata.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE icd10 (code TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE rxnorm (rxcui TEXT)")
        connection.executemany(
            "INSERT INTO icd10 VALUES (?)",
            ((code,) for code in icd),
        )
        connection.executemany(
            "INSERT INTO rxnorm VALUES (?)",
            ((code,) for code in rxnorm),
        )
    return path


def _record(
    *,
    record_id: str = "101",
    text: str = "tăng huyết áp và metformin",
    entities: tuple[CanonicalEntity, ...] | None = None,
) -> CanonicalRecord:
    entity_mappings = [
        {
            "text": "tăng huyết áp",
            "type": "CHẨN_ĐOÁN",
            "position": [0, 13],
            "assertions": [],
            "candidates": ["I10"],
        },
        {
            "text": "metformin",
            "type": "THUỐC",
            "position": [17, 26],
            "assertions": [],
            "candidates": ["6809"],
        },
    ]
    record = CanonicalRecord.create(
        record_id=record_id,
        source="trusted",
        trust_tier="trusted",
        text=text,
        entity_mappings=entity_mappings,
    )
    if entities is not None:
        return replace(record, entities=entities)
    return record


def test_failed_validation_directory_is_never_read(tmp_path):
    root = _make_source(
        tmp_path,
        {"0001": ("text", [])},
        name="synthetic.failed-validation",
    )

    with pytest.raises(ValueError, match="failed-validation"):
        load_source_records(
            _source_spec(
                root,
                ("0001",),
                name="synthetic",
                trust_tier="synthetic_validated",
            )
        )


def test_source_rejects_missing_pair(tmp_path):
    root = _make_source(tmp_path, {"0001": ("text", [])})
    (root / "gt" / "0001.json").unlink()

    with pytest.raises(ValueError, match="missing GT"):
        load_source_records(_source_spec(root, ("0001",)))


def test_synthetic_source_rejects_unexpected_ids(tmp_path):
    root = _make_source(
        tmp_path,
        {
            "0001": ("first", []),
            "0002": ("second", []),
        },
    )

    with pytest.raises(ValueError, match="unexpected source IDs"):
        load_source_records(
            _source_spec(
                root,
                ("0001",),
                name="synthetic",
                trust_tier="synthetic_validated",
            )
        )


def test_source_loads_manifest_group_and_missing_optional_fields(tmp_path):
    text = "Sốt"
    root = _make_source(
        tmp_path,
        {
            "0001": (
                text,
                [{"text": "Sốt", "type": "TRIỆU_CHỨNG", "position": [0, 3]}],
            )
        },
    )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "record_id": "0001",
                "profile_id": "symptom-profile",
                "family": "rare",
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_source_records(
        _source_spec(
            root,
            ("0001",),
            name="synthetic",
            trust_tier="synthetic_validated",
        )
    )

    assert len(records) == 1
    assert records[0].split_group == "symptom-profile"
    assert records[0].metadata["family"] == "rare"
    assert records[0].entities[0].assertions == ()
    assert records[0].entities[0].codes == ()


def test_source_normalizes_windows_newlines_before_checking_spans(tmp_path):
    root = _make_source(tmp_path, {"101": ("placeholder", [])})
    (root / "input" / "101.txt").write_bytes("Sốt\r\nmetformin".encode("utf-8"))
    _write_json(
        root / "gt" / "101.json",
        [
            {
                "text": "metformin",
                "type": "THUỐC",
                "position": [4, 13],
                "candidates": [],
            }
        ],
    )

    records = load_source_records(_source_spec(root, ("101",)))

    assert records[0].text == "Sốt\nmetformin"
    assert records[0].entities[0].text == "metformin"


def test_trusted_source_audits_and_removes_blank_legacy_candidates(tmp_path):
    root = _make_source(
        tmp_path,
        {
            "129": (
                "endoxaban",
                [
                    {
                        "text": "endoxaban",
                        "type": "THUỐC",
                        "position": [0, 9],
                        "candidates": [" "],
                    }
                ],
            )
        },
    )

    records = load_source_records(_source_spec(root, ("129",)))

    assert records[0].entities[0].codes == ()
    assert records[0].metadata["source_normalizations"] == {
        "blank_candidates_removed": 1
    }


def test_trusted_source_audits_and_removes_exact_duplicate_entities(tmp_path):
    duplicate = {
        "text": "đau",
        "type": "TRIỆU_CHỨNG",
        "position": [0, 3],
        "assertions": [],
    }
    root = _make_source(
        tmp_path,
        {"193": ("đau", [duplicate, dict(duplicate), dict(duplicate)])},
    )

    records = load_source_records(_source_spec(root, ("193",)))

    assert len(records[0].entities) == 1
    assert records[0].metadata["source_normalizations"] == {
        "duplicate_entities_removed": 2
    }


def test_source_verifies_manifest_content_fingerprints(tmp_path):
    root = _make_source(tmp_path, {"0001": ("content", [])})
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "record_id": "0001",
                "input_sha256": "0" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input_sha256 mismatch"):
        load_source_records(
            _source_spec(
                root,
                ("0001",),
                name="synthetic",
                trust_tier="synthetic_validated",
            )
        )


def test_validator_checks_code_namespace(tmp_path):
    db = _build_metadata_db(tmp_path)
    record = CanonicalRecord.create(
        record_id="101",
        source="trusted",
        trust_tier="trusted",
        text="tăng huyết áp metformin",
        entity_mappings=[
            {
                "text": "tăng huyết áp",
                "type": "CHẨN_ĐOÁN",
                "position": [0, 13],
                "candidates": ["6809"],
            },
            {
                "text": "metformin",
                "type": "THUỐC",
                "position": [14, 23],
                "candidates": ["I10"],
            },
        ],
    )

    findings = validate_records((record,), db)

    assert {finding.code for finding in findings} == {
        "unknown_icd10_code",
        "unknown_rxnorm_code",
    }


def test_validator_reports_duplicates_hash_and_overlaps(tmp_path):
    db = _build_metadata_db(tmp_path)
    base = _record()
    overlapping = CanonicalEntity(
        text=base.text[5:20],
        entity_type="TRIỆU_CHỨNG",
        start=5,
        end=20,
    )
    corrupt = replace(
        base,
        entities=base.entities + (overlapping,),
        sha256="0" * 64,
    )

    findings = validate_records((corrupt, base), db)
    codes = {finding.code for finding in findings}

    assert "duplicate_record_id" in codes
    assert "text_hash_mismatch" in codes
    assert "overlapping_spans" in codes


def test_synthetic_source_requires_a_passing_validation_report(tmp_path):
    root = _make_source(tmp_path, {"0001": ("content", [])})
    (root / "qa").mkdir()
    _write_json(root / "qa" / "validation_report.json", {"passed": False})
    spec = _source_spec(
        root,
        ("0001",),
        name="synthetic",
        trust_tier="synthetic_validated",
    )
    records = load_source_records(spec)
    db = _build_metadata_db(tmp_path)

    with pytest.raises(ValueError, match="validation report did not pass"):
        require_valid_source(spec, records, db)


def test_require_valid_source_accepts_clean_trusted_records(tmp_path):
    root = _make_source(
        tmp_path,
        {
            "101": (
                "tăng huyết áp và metformin",
                [
                    {
                        "text": "tăng huyết áp",
                        "type": "CHẨN_ĐOÁN",
                        "position": [0, 13],
                        "candidates": ["I10"],
                    },
                    {
                        "text": "metformin",
                        "type": "THUỐC",
                        "position": [17, 26],
                        "candidates": ["6809"],
                    },
                ],
            )
        },
    )
    spec = _source_spec(root, ("101",))
    records = load_source_records(spec)

    assert require_valid_source(spec, records, _build_metadata_db(tmp_path)) is None
