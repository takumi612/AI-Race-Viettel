"""Validation and packaging safeguards for competition submissions."""

from __future__ import annotations

import json
import os
import sqlite3
import zipfile
from pathlib import Path
from typing import Any, Iterable


SCHEMA_KEYS = {
    "CHẨN_ĐOÁN": {"text", "type", "position", "assertions", "candidates"},
    "THUỐC": {"text", "type", "position", "assertions", "candidates"},
    "TRIỆU_CHỨNG": {"text", "type", "position", "assertions"},
    "TÊN_XÉT_NGHIỆM": {"text", "type", "position"},
    "KẾT_QUẢ_XÉT_NGHIỆM": {"text", "type", "position"},
}
ALLOWED_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}
_CANDIDATE_TABLES = {
    "CHẨN_ĐOÁN": ("icd10", "code", "ICD-10"),
    "THUỐC": ("rxnorm", "rxcui", "RxNorm"),
}


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _read_only_connection(db_path: str) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db_path).resolve().as_uri()}?mode=ro", uri=True)


def validate_prediction(
    text: str, entities: list[dict], db_path: str | None
) -> list[str]:
    """Return every structural and semantic submission error for one prediction."""
    errors: list[str] = []
    if not isinstance(text, str):
        return ["prediction text must be a string"]
    if not isinstance(entities, list):
        return ["prediction entities must be a list"]

    candidates_to_check: list[tuple[int, str, str]] = []
    for index, entity in enumerate(entities):
        prefix = f"entity {index}"
        if not isinstance(entity, dict):
            errors.append(f"{prefix} must be an object")
            continue

        entity_type = entity.get("type")
        if not isinstance(entity_type, str) or entity_type not in SCHEMA_KEYS:
            errors.append(f"{prefix} has unsupported type {entity_type!r}")
            continue

        expected_keys = SCHEMA_KEYS[entity_type]
        actual_keys = set(entity)
        missing = expected_keys - actual_keys
        unexpected = actual_keys - expected_keys
        if missing:
            errors.append(f"{prefix} is missing keys: {', '.join(sorted(missing))}")
        if unexpected:
            errors.append(
                f"{prefix} has unexpected keys: {', '.join(sorted(map(str, unexpected)))}"
            )

        entity_text = entity.get("text")
        if not isinstance(entity_text, str):
            errors.append(f"{prefix} text must be a string")

        position = entity.get("position")
        position_valid = isinstance(position, list) and len(position) == 2
        if not position_valid:
            errors.append(f"{prefix} position must be a two-item list")
        else:
            start, end = position
            offsets_are_integers = _is_integer(start) and _is_integer(end)
            if not _is_integer(start):
                errors.append(f"{prefix} position[0] must be an integer")
            if not _is_integer(end):
                errors.append(f"{prefix} position[1] must be an integer")
            if offsets_are_integers:
                if not 0 <= start < end <= len(text):
                    errors.append(f"{prefix} position must satisfy 0 <= start < end <= text length")
                elif isinstance(entity_text, str) and text[start:end] != entity_text:
                    errors.append(f"{prefix} text slice does not match text at position")

        if "assertions" in expected_keys:
            assertions = entity.get("assertions")
            if not isinstance(assertions, list):
                errors.append(f"{prefix} assertions must be a list")
            else:
                seen_assertions: set[str] = set()
                for assertion_index, assertion in enumerate(assertions):
                    if not isinstance(assertion, str) or assertion not in ALLOWED_ASSERTIONS:
                        errors.append(
                            f"{prefix} assertions[{assertion_index}] is not allowed: {assertion!r}"
                        )
                    elif assertion in seen_assertions:
                        errors.append(f"{prefix} has duplicate assertion: {assertion}")
                    else:
                        seen_assertions.add(assertion)

        if "candidates" in expected_keys:
            candidates = entity.get("candidates")
            if not isinstance(candidates, list):
                errors.append(f"{prefix} candidates must be a list")
            else:
                for candidate_index, candidate in enumerate(candidates):
                    if not isinstance(candidate, str):
                        errors.append(
                            f"{prefix} candidates[{candidate_index}] must be a string"
                        )
                    else:
                        candidates_to_check.append((index, entity_type, candidate))

    if db_path is None or not candidates_to_check:
        return errors

    try:
        with _read_only_connection(db_path) as connection:
            cursor = connection.cursor()
            for index, entity_type, candidate in candidates_to_check:
                table, column, label = _CANDIDATE_TABLES[entity_type]
                found = cursor.execute(
                    f"SELECT 1 FROM {table} WHERE {column} = ? LIMIT 1", (candidate,)
                ).fetchone()
                if found is None:
                    errors.append(
                        f"entity {index} {label} candidate {candidate!r} does not exist"
                    )
    except sqlite3.Error as error:
        errors.append(f"knowledge-base validation failed: {error}")

    return errors


def validate_output_directory(
    input_dir: str, output_dir: str, expected_ids: Iterable[int], db_path: str | None = None
) -> list[str]:
    """Validate all expected output files and collect errors across the directory."""
    errors: list[str] = []
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    for file_id in expected_ids:
        input_file = input_path / f"{file_id}.txt"
        output_file = output_path / f"{file_id}.json"
        if not input_file.is_file():
            errors.append(f"missing input file {file_id}.txt")
            text: str | None = None
        else:
            try:
                text = input_file.read_text(encoding="utf-8")
            except OSError as error:
                errors.append(f"could not read input file {file_id}.txt: {error}")
                text = None

        if not output_file.is_file():
            errors.append(f"missing output file {file_id}.json")
            continue
        try:
            entities: Any = json.loads(output_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"{file_id}.json is not valid JSON: {error}")
            continue
        if text is not None:
            errors.extend(
                f"{file_id}.json: {error}"
                for error in validate_prediction(text, entities, db_path=db_path)
            )
    return errors


def package_submission(output_dir: str, zip_path: str) -> None:
    """Atomically create a ZIP only after every expected JSON file is valid."""
    output_path = Path(output_dir)
    files: list[tuple[int, Path]] = []
    preflight_errors: list[str] = []
    for file_id in range(1, 101):
        prediction_file = output_path / f"{file_id}.json"
        if not prediction_file.is_file():
            preflight_errors.append(f"missing output file {prediction_file.name}")
            continue
        try:
            json.loads(prediction_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            preflight_errors.append(f"{prediction_file.name} is not valid JSON: {error}")
            continue
        files.append((file_id, prediction_file))

    if preflight_errors:
        raise ValueError("submission preflight failed: " + "; ".join(preflight_errors))

    destination = Path(zip_path)
    temporary = Path(f"{destination}.tmp")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file_id, prediction_file in files:
                archive.write(prediction_file, arcname=f"output/{file_id}.json")
        os.replace(temporary, destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def write_failure_output(
    output_path: str | Path,
    error_log_path: str | Path,
    file_id: str,
    error: Exception,
) -> None:
    """Write the required empty prediction and a single structured failure record."""
    output_file = Path(output_path)
    error_log = Path(error_log_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    error_log.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("[]", encoding="utf-8")
    record = {
        "file_id": str(file_id),
        "error_type": type(error).__name__,
        "message": str(error),
    }
    with error_log.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
