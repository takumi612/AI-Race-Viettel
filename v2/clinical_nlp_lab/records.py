from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .provenance import (
    canonical_jsonl_bytes,
    load_json_strict,
    sha256_bytes,
    verify_dataset_provenance,
)


RECORD_SCHEMA_ID = "clinical_nlp.record_metadata"
RECORD_SCHEMA_VERSION = 1
RECORD_DETECTOR_VERSION = "1.0.0"
OFFSET_SPACE = "utf8-universal-newline-codepoint/v1"
_ORGANIZER_HEADER_RE = re.compile(
    r"(?m)^-\s+Chi tiết hồ sơ bệnh nhân thứ ([0-9]+):[ \t]*$"
)
_SYNTHETIC_RECORD_RE = re.compile(r"\bHS-([0-9]{4})\b")


class RecordContractError(ValueError):
    """Raised when patient-record boundaries are ambiguous or unsafe."""


@dataclass(frozen=True, slots=True)
class RecordSpan:
    patient_block_id: str
    start: int
    end: int
    confidence: str
    evidence: str
    ordinal: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_block_id": self.patient_block_id,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "ordinal": self.ordinal,
        }


@dataclass(frozen=True, slots=True)
class ClinicalRecord:
    document_id: str
    record_id: str
    raw_start: int
    raw_end: int
    entity_indices: tuple[int, ...]


def parse_document_records(
    document_id: str,
    raw_text: str,
    entities: Sequence[Any],
) -> tuple[ClinicalRecord, ...]:
    dict_entities: list[dict[str, Any]] = []
    for e in entities:
        if hasattr(e, "start") and hasattr(e, "end"):
            dict_entities.append({"position": [e.start, e.end]})
        elif isinstance(e, Mapping) and "position" in e:
            dict_entities.append(dict(e))
        else:
            dict_entities.append({"position": [0, 0]})
    spans = detect_record_spans(document_id, raw_text, dict_entities)
    records: list[ClinicalRecord] = []
    for span in spans:
        indices: list[int] = []
        for idx, e in enumerate(entities):
            s = e.start if hasattr(e, "start") else e.get("position", [0, 0])[0]
            end_pos = e.end if hasattr(e, "end") else e.get("position", [0, 0])[1]
            if span.start <= s and end_pos <= span.end:
                indices.append(idx)
        records.append(
            ClinicalRecord(
                document_id=document_id,
                record_id=span.patient_block_id,
                raw_start=span.start,
                raw_end=span.end,
                entity_indices=tuple(indices),
            )
        )
    return tuple(records)


@dataclass(frozen=True, slots=True)
class RecordMetadataRow:
    document_id: str
    pair_sha256: str
    input_sha256: str
    source_role: str
    train_eligible: bool
    record_spans: tuple[RecordSpan, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": RECORD_SCHEMA_ID,
            "schema_version": RECORD_SCHEMA_VERSION,
            "detector_version": RECORD_DETECTOR_VERSION,
            "offset_space": OFFSET_SPACE,
            "document_id": self.document_id,
            "pair_sha256": self.pair_sha256,
            "input_sha256": self.input_sha256,
            "source_role": self.source_role,
            "train_eligible": self.train_eligible,
            "record_spans": [span.to_dict() for span in self.record_spans],
        }


@dataclass(frozen=True, slots=True)
class RecordMetadataSnapshot:
    dataset_root: Path
    dataset_fingerprint: str
    manifest_sha256: str
    rows: tuple[RecordMetadataRow, ...]
    metadata_bytes: bytes
    metadata_sha256: str

    @property
    def record_count(self) -> int:
        return sum(len(row.record_spans) for row in self.rows)


def normalized_text_from_raw_bytes(raw: bytes) -> str:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RecordContractError("Input is not strict UTF-8") from exc
    return decoded.replace("\r\n", "\n").replace("\r", "\n")


def _entity_bounds(entities: Iterable[Mapping[str, Any]]) -> tuple[tuple[int, int], ...]:
    bounds: list[tuple[int, int]] = []
    for index, entity in enumerate(entities):
        if not isinstance(entity, Mapping):
            raise RecordContractError(f"Entity {index} is not an object")
        position = entity.get("position")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
        ):
            raise RecordContractError(f"Entity {index} has invalid position type")
        bounds.append((position[0], position[1]))
    return tuple(bounds)


def _validate_partition(
    document_id: str,
    text: str,
    spans: Sequence[RecordSpan],
    entities: Iterable[Mapping[str, Any]],
) -> tuple[RecordSpan, ...]:
    if not spans:
        raise RecordContractError(f"Document {document_id} has no record span")
    if spans[0].start != 0 or spans[-1].end != len(text):
        raise RecordContractError(f"Document {document_id} record spans do not cover the document")
    previous_end = 0
    block_ids: set[str] = set()
    for expected_ordinal, span in enumerate(spans, 1):
        if (
            span.ordinal != expected_ordinal
            or span.start != previous_end
            or not span.start < span.end
            or span.patient_block_id in block_ids
        ):
            raise RecordContractError(f"Document {document_id} record partition is not contiguous")
        previous_end = span.end
        block_ids.add(span.patient_block_id)
    for entity_index, (start, end) in enumerate(_entity_bounds(entities)):
        owners = [span for span in spans if span.start <= start and end <= span.end]
        if len(owners) != 1:
            raise RecordContractError(
                f"Document {document_id} entity {entity_index} crosses a patient-record boundary"
            )
    return tuple(spans)


def detect_record_spans(
    document_id: str,
    text: str,
    entities: Iterable[Mapping[str, Any]],
) -> tuple[RecordSpan, ...]:
    try:
        numeric_id = int(document_id)
    except (TypeError, ValueError) as exc:
        raise RecordContractError("Document ID must be numeric") from exc
    if str(numeric_id) != document_id or numeric_id <= 0:
        raise RecordContractError("Document ID must be canonical positive decimal")

    if numeric_id <= 200:
        matches = list(_ORGANIZER_HEADER_RE.finditer(text))
        if not matches:
            raise RecordContractError(
                f"Document {document_id} has credible multi-patient scope but no recognized headers"
            )
        ordinals = [int(match.group(1)) for match in matches]
        if ordinals != list(range(1, len(matches) + 1)):
            raise RecordContractError(f"Document {document_id} patient header ordinals are ambiguous")
        starts = [0, *(match.start() for match in matches[1:])]
        ends = [*starts[1:], len(text)]
        spans = tuple(
            RecordSpan(
                patient_block_id=f"{document_id}:record-{ordinal:04d}",
                start=start,
                end=end,
                confidence="high",
                evidence=(
                    "document_preamble_plus_sequential_patient_header"
                    if ordinal == 1
                    else "sequential_patient_header"
                ),
                ordinal=ordinal,
            )
            for ordinal, (start, end) in enumerate(zip(starts, ends), 1)
        )
        return _validate_partition(document_id, text, spans, entities)

    record_codes = _SYNTHETIC_RECORD_RE.findall(text)
    expected_code = f"{numeric_id:04d}"
    if record_codes != [expected_code]:
        raise RecordContractError(
            f"Document {document_id} must contain exactly one matching synthetic record ID"
        )
    spans = (
        RecordSpan(
            patient_block_id=f"{document_id}:record-0001",
            start=0,
            end=len(text),
            confidence="high",
            evidence="single_matching_hospital_record_id",
            ordinal=1,
        ),
    )
    return _validate_partition(document_id, text, spans, entities)


def _source_role(raw_bucket: Any) -> str:
    mapping = {
        "reconstructed": "quarantine",
        "quarantine": "quarantine",
        "organizer_gt": "organizer",
        "organizer": "organizer",
        "synthetic": "synthetic",
    }
    if not isinstance(raw_bucket, str) or raw_bucket not in mapping:
        raise RecordContractError("Manifest source bucket is unsupported")
    return mapping[raw_bucket]


def build_record_metadata(dataset_root: str | Path) -> RecordMetadataSnapshot:
    verification = verify_dataset_provenance(dataset_root)
    manifest_by_id = {str(row["document_id"]): row for row in verification.rows}
    rows: list[RecordMetadataRow] = []
    for pair in verification.snapshot.pairs:
        manifest_row = manifest_by_id[pair.document_id]
        payload = load_json_strict(pair.gt_bytes, source=f"GT {pair.document_id}")
        if not isinstance(payload, list):
            raise RecordContractError(f"GT document {pair.document_id} is not an array")
        text = normalized_text_from_raw_bytes(pair.input_bytes)
        spans = detect_record_spans(pair.document_id, text, payload)
        eligible = manifest_row.get("train_eligible")
        if type(eligible) is not bool:
            raise RecordContractError("Manifest train_eligible must be an exact boolean")
        rows.append(
            RecordMetadataRow(
                document_id=pair.document_id,
                pair_sha256=pair.pair_sha256,
                input_sha256=pair.input_sha256,
                source_role=_source_role(manifest_row.get("source_bucket")),
                train_eligible=eligible,
                record_spans=spans,
            )
        )
    metadata_bytes = canonical_jsonl_bytes(row.to_dict() for row in rows)
    return RecordMetadataSnapshot(
        dataset_root=verification.dataset_root,
        dataset_fingerprint=verification.dataset_fingerprint,
        manifest_sha256=verification.manifest_sha256,
        rows=tuple(rows),
        metadata_bytes=metadata_bytes,
        metadata_sha256=sha256_bytes(metadata_bytes),
    )
