import json
import sqlite3
import zipfile

import pytest

from src.validation.submission import (
    package_submission,
    validate_output_directory,
    validate_prediction,
    write_failure_output,
)


SYMPTOM = "TRIỆU_CHỨNG"
DIAGNOSIS = "CHẨN_ĐOÁN"
DRUG = "THUỐC"


def test_validator_rejects_text_offset_mismatch():
    errors = validate_prediction(
        "bệnh nhân ho",
        [{"text": "sốt", "type": SYMPTOM, "position": [10, 12], "assertions": []}],
        db_path=None,
    )

    assert any("text slice" in error for error in errors)


def test_validator_aggregates_schema_offset_and_candidate_errors():
    errors = validate_prediction(
        "ho",
        [
            {"text": "ho", "type": SYMPTOM, "position": [True, 1], "assertions": ["bad"]},
            {"text": "ho", "type": DIAGNOSIS, "position": [0, 2], "assertions": [], "candidates": [7], "extra": 1},
        ],
        db_path=None,
    )

    assert any("entity 0 position" in error and "integer" in error for error in errors)
    assert any("entity 0 assertions" in error for error in errors)
    assert any("entity 1 has unexpected keys" in error for error in errors)
    assert any("entity 1 candidates" in error for error in errors)


def test_validator_rejects_duplicate_assertions():
    errors = validate_prediction(
        "ho",
        [{"text": "ho", "type": SYMPTOM, "position": [0, 2], "assertions": ["isNegated", "isNegated"]}],
        db_path=None,
    )

    assert any("duplicate assertion" in error for error in errors)


def test_validator_checks_candidate_existence_in_type_specific_tables(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE icd10 (code TEXT PRIMARY KEY)")
        connection.execute("CREATE TABLE rxnorm (rxcui TEXT)")
        connection.execute("INSERT INTO icd10 VALUES ('I10')")
        connection.execute("INSERT INTO rxnorm VALUES ('123')")

    errors = validate_prediction(
        "ho",
        [
            {"text": "ho", "type": DIAGNOSIS, "position": [0, 2], "assertions": [], "candidates": ["MISSING"]},
            {"text": "ho", "type": DRUG, "position": [0, 2], "assertions": [], "candidates": ["999"]},
        ],
        db_path=str(db_path),
    )

    assert any("ICD-10 candidate 'MISSING'" in error for error in errors)
    assert any("RxNorm candidate '999'" in error for error in errors)


def test_validate_output_directory_aggregates_missing_invalid_and_prediction_errors(tmp_path):
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    (input_dir / "1.txt").write_text("ho", encoding="utf-8")
    (input_dir / "2.txt").write_text("sốt", encoding="utf-8")
    (output_dir / "1.json").write_text("{not json", encoding="utf-8")

    errors = validate_output_directory(input_dir, output_dir, range(1, 4))

    assert any("1.json is not valid JSON" in error for error in errors)
    assert any("missing output file 2.json" in error for error in errors)
    assert any("missing input file 3.txt" in error for error in errors)
    assert any("missing output file 3.json" in error for error in errors)


def test_packager_puts_json_under_output_folder(tmp_path):
    output = tmp_path / "predictions"
    output.mkdir()
    for file_id in range(1, 101):
        (output / f"{file_id}.json").write_text("[]", encoding="utf-8")
    zip_path = tmp_path / "output.zip"

    package_submission(output, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist()[0].startswith("output/")
        assert set(archive.namelist()) == {f"output/{i}.json" for i in range(1, 101)}


def test_packager_preflight_keeps_existing_destination_on_failure(tmp_path):
    output = tmp_path / "predictions"
    output.mkdir()
    for file_id in range(1, 101):
        (output / f"{file_id}.json").write_text("[]", encoding="utf-8")
    (output / "100.json").write_text("{broken", encoding="utf-8")
    zip_path = tmp_path / "output.zip"
    zip_path.write_bytes(b"known-good-archive")

    with pytest.raises(ValueError, match="100.json"):
        package_submission(output, zip_path)

    assert zip_path.read_bytes() == b"known-good-archive"
    assert not (tmp_path / "output.zip.tmp").exists()


def test_failure_writer_creates_empty_output_and_structured_log(tmp_path):
    output_path = tmp_path / "nested" / "35.json"
    log_path = tmp_path / "logs" / "errors.jsonl"

    write_failure_output(output_path, log_path, "35", ValueError("bad offset"))

    assert json.loads(output_path.read_text(encoding="utf-8")) == []
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record == {
        "file_id": "35",
        "error_type": "ValueError",
        "message": "bad offset",
    }


def test_pipeline_writes_failure_artifacts_for_each_failed_file(tmp_path, monkeypatch):
    from src.pipeline.main import BaselinePipeline

    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "35.txt").write_text("text", encoding="utf-8")
    pipeline = BaselinePipeline.__new__(BaselinePipeline)
    monkeypatch.setattr(pipeline, "process_file", lambda _: (_ for _ in ()).throw(ValueError("bad offset")))

    pipeline.run(input_dir=str(input_dir), output_dir=str(output_dir))

    assert json.loads((output_dir / "35.json").read_text(encoding="utf-8")) == []
    assert json.loads((output_dir / "errors.jsonl").read_text(encoding="utf-8")) == {
        "file_id": "35",
        "error_type": "ValueError",
        "message": "bad offset",
    }
