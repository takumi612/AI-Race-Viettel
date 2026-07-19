"""Validation gates for canonical training data and code namespaces."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from typing import Iterable, Sequence

from src.training.contracts import (
    ALLOWED_ASSERTIONS,
    ALLOWED_ENTITY_TYPES,
    CanonicalEntity,
    CanonicalRecord,
)
from src.training.sources import SourceSpec, is_synthetic_source


@dataclass(frozen=True, order=True, slots=True)
class ValidationFinding:
    severity: str
    code: str
    message: str
    record_id: str | None = None
    entity_index: int | None = None


def _finding(
    code: str,
    message: str,
    record_id: str | None = None,
    entity_index: int | None = None,
) -> ValidationFinding:
    return ValidationFinding(
        severity="error",
        code=code,
        message=message,
        record_id=record_id,
        entity_index=entity_index,
    )


def _open_read_only_database(path: Path) -> sqlite3.Connection:
    resolved = path.resolve()
    if not resolved.is_file() or resolved.stat().st_size == 0:
        raise ValueError(f"metadata database is missing or empty: {resolved}")
    try:
        connection = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
        connection.execute("SELECT 1 FROM icd10 LIMIT 1").fetchone()
        connection.execute("SELECT 1 FROM rxnorm LIMIT 1").fetchone()
    except sqlite3.Error as exc:
        raise ValueError(f"invalid metadata database: {resolved}: {exc}") from exc
    return connection


def _known_codes(
    connection: sqlite3.Connection,
    *,
    table: str,
    column: str,
    codes: Iterable[str],
) -> set[str]:
    pending = sorted(set(codes))
    known: set[str] = set()
    for start in range(0, len(pending), 500):
        batch = pending[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        rows = connection.execute(
            f"SELECT DISTINCT CAST({column} AS TEXT) FROM {table} "
            f"WHERE CAST({column} AS TEXT) IN ({placeholders})",
            batch,
        )
        known.update(str(row[0]) for row in rows)
    return known


def _validate_entity(
    record: CanonicalRecord,
    entity: CanonicalEntity,
    entity_index: int,
) -> list[ValidationFinding]:
    findings: list[ValidationFinding] = []
    record_id = record.record_id

    if entity.entity_type not in ALLOWED_ENTITY_TYPES:
        findings.append(
            _finding(
                "unsupported_entity_type",
                f"unsupported entity type: {entity.entity_type}",
                record_id,
                entity_index,
            )
        )
    invalid_assertions = sorted(set(entity.assertions) - ALLOWED_ASSERTIONS)
    if invalid_assertions:
        findings.append(
            _finding(
                "invalid_assertion",
                f"invalid assertions: {invalid_assertions}",
                record_id,
                entity_index,
            )
        )
    if (
        isinstance(entity.start, bool)
        or isinstance(entity.end, bool)
        or not isinstance(entity.start, int)
        or not isinstance(entity.end, int)
        or entity.start < 0
        or entity.end <= entity.start
        or entity.end > len(record.text)
    ):
        findings.append(
            _finding(
                "invalid_span",
                f"invalid half-open span [{entity.start}, {entity.end})",
                record_id,
                entity_index,
            )
        )
    elif record.text[entity.start : entity.end] != entity.text:
        findings.append(
            _finding(
                "span_text_mismatch",
                "entity text does not match its half-open source span",
                record_id,
                entity_index,
            )
        )

    if entity.codes and entity.entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}:
        findings.append(
            _finding(
                "invalid_code_namespace",
                f"{entity.entity_type} entities cannot carry ontology candidates",
                record_id,
                entity_index,
            )
        )
    return findings


def validate_records(
    records: Sequence[CanonicalRecord],
    db_path: str | Path,
) -> tuple[ValidationFinding, ...]:
    """Return deterministic errors without mutating the source records."""

    findings: list[ValidationFinding] = []
    seen_ids: set[str] = set()
    diagnosis_codes: set[str] = set()
    drug_codes: set[str] = set()
    code_locations: list[tuple[str, int, str, str]] = []

    for record in records:
        if record.record_id in seen_ids:
            findings.append(
                _finding(
                    "duplicate_record_id",
                    f"duplicate record ID: {record.record_id}",
                    record.record_id,
                )
            )
        seen_ids.add(record.record_id)

        if not isinstance(record.text, str) or not record.text.strip():
            findings.append(
                _finding(
                    "empty_text",
                    "record text must be non-empty",
                    record.record_id,
                )
            )
        elif sha256(record.text.encode("utf-8")).hexdigest() != record.sha256:
            findings.append(
                _finding(
                    "text_hash_mismatch",
                    "record text does not match its canonical SHA-256",
                    record.record_id,
                )
            )

        indexed_entities = sorted(
            enumerate(record.entities),
            key=lambda item: (item[1].start, item[1].end, item[0]),
        )
        furthest_end = -1
        for entity_index, entity in indexed_entities:
            findings.extend(_validate_entity(record, entity, entity_index))
            if isinstance(entity.start, int) and not isinstance(entity.start, bool):
                if entity.start < furthest_end:
                    findings.append(
                        _finding(
                            "overlapping_spans",
                            "entity spans overlap within the same record",
                            record.record_id,
                            entity_index,
                        )
                    )
                if isinstance(entity.end, int) and not isinstance(entity.end, bool):
                    furthest_end = max(furthest_end, entity.end)

            if entity.entity_type == "CHẨN_ĐOÁN":
                diagnosis_codes.update(entity.codes)
                code_locations.extend(
                    (record.record_id, entity_index, "icd10", code)
                    for code in entity.codes
                )
            elif entity.entity_type == "THUỐC":
                drug_codes.update(entity.codes)
                code_locations.extend(
                    (record.record_id, entity_index, "rxnorm", code)
                    for code in entity.codes
                )

    with _open_read_only_database(Path(db_path)) as connection:
        try:
            known_diagnosis = _known_codes(
                connection,
                table="icd10",
                column="code",
                codes=diagnosis_codes,
            )
            known_drugs = _known_codes(
                connection,
                table="rxnorm",
                column="rxcui",
                codes=drug_codes,
            )
        except sqlite3.Error as exc:
            raise ValueError(f"cannot validate ontology codes: {exc}") from exc

    for record_id, entity_index, namespace, code in code_locations:
        if namespace == "icd10" and code not in known_diagnosis:
            findings.append(
                _finding(
                    "unknown_icd10_code",
                    f"ICD-10 code is not present in metadata.db: {code}",
                    record_id,
                    entity_index,
                )
            )
        elif namespace == "rxnorm" and code not in known_drugs:
            findings.append(
                _finding(
                    "unknown_rxnorm_code",
                    f"RxNorm code is not present in metadata.db: {code}",
                    record_id,
                    entity_index,
                )
            )

    return tuple(sorted(findings))


def _require_passing_synthetic_report(
    spec: SourceSpec,
    record_count: int,
) -> None:
    report_path = spec.root.resolve() / "qa" / "validation_report.json"
    if not report_path.is_file():
        raise ValueError(
            f"synthetic validation report is missing: {report_path}"
        )
    try:
        report = json.loads(report_path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid synthetic validation report: {report_path}") from exc
    if not isinstance(report, dict) or report.get("passed") is not True:
        raise ValueError(f"synthetic validation report did not pass: {report_path}")

    coverage = report.get("coverage")
    if isinstance(coverage, dict) and "record_count" in coverage:
        if coverage["record_count"] != record_count:
            raise ValueError(
                "synthetic validation report record_count does not match loaded data"
            )


def require_valid_source(
    spec: SourceSpec,
    records: Sequence[CanonicalRecord],
    db_path: str | Path,
) -> None:
    if is_synthetic_source(spec):
        _require_passing_synthetic_report(spec, len(records))

    findings = validate_records(records, db_path)
    if findings:
        preview = ", ".join(
            f"{finding.record_id or '-'}:{finding.code}" for finding in findings[:5]
        )
        raise ValueError(
            f"source validation failed with {len(findings)} error(s): {preview}"
        )
