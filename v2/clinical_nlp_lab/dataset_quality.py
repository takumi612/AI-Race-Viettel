from __future__ import annotations

import hashlib
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .provenance import (
    LEGACY_SHA256_SEMANTICS,
    MANIFEST_ROW_SCHEMA_ID,
    MANIFEST_ROW_SCHEMA_VERSION,
    ProvenanceError,
)
from .schema import ALLOWED_ASSERTIONS, ClinicalDocument
from .text import normalize_alias


TRAINING_ENTITY_TYPES = {"DISEASE", "DRUG", "SYMPTOM", "LAB_NAME", "LAB_RESULT"}
ASSERTION_ENTITY_TYPES = {"DISEASE", "DRUG", "SYMPTOM"}


def normalize_surface(text: str) -> str:
    normalized = normalize_alias(text).replace("đ", "d")
    return "".join(
        character
        for character in unicodedata.normalize("NFD", normalized)
        if unicodedata.category(character) != "Mn"
    )


@dataclass(frozen=True, slots=True)
class DatasetRecord:
    document_id: str
    source_bucket: str
    template_group: str
    genre: str
    long_tail: bool
    primary_surfaces: tuple[str, ...]
    sha256: str
    train_eligible: bool = True
    train_exclusion_reason: str | None = None
    schema_id: str | None = None
    schema_version: int | None = None
    input_sha256: str | None = None
    input_size_bytes: int | None = None
    gt_sha256: str | None = None
    gt_size_bytes: int | None = None
    pair_sha256: str | None = None
    legacy_sha256_semantics: str = LEGACY_SHA256_SEMANTICS

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["primary_surfaces"] = list(self.primary_surfaces)
        return payload

    @classmethod
    def from_v2_manifest_row(cls, row: Mapping[str, Any]) -> "DatasetRecord":
        """Build a training record only from a previously verified v2 row."""
        document_id = row.get("document_id")
        if row.get("schema_id") != MANIFEST_ROW_SCHEMA_ID or row.get(
            "schema_version"
        ) != MANIFEST_ROW_SCHEMA_VERSION:
            raise ProvenanceError(f"Manifest row {document_id!r} is not schema v2")
        if type(row.get("train_eligible")) is not bool:
            raise ProvenanceError(
                f"Manifest row {document_id!r} requires explicit boolean train_eligible"
            )
        required_hash_fields = ("input_sha256", "gt_sha256", "pair_sha256")
        if any(not isinstance(row.get(field), str) for field in required_hash_fields):
            raise ProvenanceError(f"Manifest row {document_id!r} lacks raw provenance hashes")
        return cls(
            document_id=str(document_id),
            source_bucket=str(row["source_bucket"]),
            template_group=str(row.get("template_group", document_id)),
            genre=str(row.get("genre", "unknown")),
            long_tail=bool(row.get("long_tail", False)),
            primary_surfaces=tuple(str(item) for item in row.get("primary_surfaces", [])),
            sha256=str(row["sha256"]),
            train_eligible=row["train_eligible"],
            train_exclusion_reason=(
                str(row["train_exclusion_reason"])
                if row.get("train_exclusion_reason")
                else None
            ),
            schema_id=MANIFEST_ROW_SCHEMA_ID,
            schema_version=MANIFEST_ROW_SCHEMA_VERSION,
            input_sha256=str(row["input_sha256"]),
            input_size_bytes=int(row["input_size_bytes"]),
            gt_sha256=str(row["gt_sha256"]),
            gt_size_bytes=int(row["gt_size_bytes"]),
            pair_sha256=str(row["pair_sha256"]),
            legacy_sha256_semantics=str(row["legacy_sha256_semantics"]),
        )


def source_bucket_for_document(document_id: str) -> str:
    try:
        numeric_id = int(document_id)
    except ValueError:
        return "unknown"
    if 1 <= numeric_id <= 100:
        return "reconstructed"
    if 101 <= numeric_id <= 200:
        return "organizer_gt"
    if numeric_id >= 201:
        return "synthetic"
    return "unknown"


def build_dataset_manifest(
    documents: Iterable[ClinicalDocument],
    metadata_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[DatasetRecord]:
    """Build legacy semantic records; this is not a raw-byte provenance writer."""
    metadata = metadata_by_id or {}
    records: list[DatasetRecord] = []
    for document in documents:
        item_metadata = metadata.get(document.document_id, {})
        source_bucket = source_bucket_for_document(document.document_id)
        genre = str(item_metadata.get("genre", "unknown"))
        template_group = str(
            item_metadata.get("template_group", f"{source_bucket}:{genre}")
        )
        surfaces = tuple(
            sorted(
                {
                    normalized
                    for entity in document.entities
                    if entity.type in {"DISEASE", "DRUG"}
                    for normalized in [normalize_surface(entity.mention_head or entity.text)]
                    if normalized
                }
            )
        )
        records.append(
            DatasetRecord(
                document_id=document.document_id,
                source_bucket=source_bucket,
                template_group=template_group,
                genre=genre,
                long_tail=bool(item_metadata.get("long_tail", False)),
                primary_surfaces=surfaces,
                sha256=hashlib.sha256(document.raw_text.encode("utf-8")).hexdigest(),
                train_eligible=bool(item_metadata.get("train_eligible", True)),
                train_exclusion_reason=(
                    str(item_metadata["train_exclusion_reason"])
                    if item_metadata.get("train_exclusion_reason")
                    else None
                ),
            )
        )
    return records


def validate_dataset_contract(
    documents: Iterable[ClinicalDocument],
    icd_ids: set[str],
    rxnorm_ids: set[str],
) -> dict[str, Any]:
    document_list = list(documents)
    errors: list[dict[str, Any]] = []
    type_counts: Counter[str] = Counter()
    assertion_counts: Counter[str] = Counter()

    def add_error(document_id: str, entity_index: int, code: str, **details: Any) -> None:
        errors.append(
            {
                "document_id": document_id,
                "entity_index": entity_index,
                "code": code,
                **details,
            }
        )

    for document in document_list:
        for entity_index, entity in enumerate(document.entities):
            type_counts[entity.type] += 1
            assertion_counts.update(entity.assertions)
            try:
                entity.validate_offset(document.raw_text)
            except ValueError as exc:
                add_error(document.document_id, entity_index, "invalid_offset", message=str(exc))

            if entity.type not in TRAINING_ENTITY_TYPES:
                add_error(
                    document.document_id,
                    entity_index,
                    "unsupported_entity_type",
                    entity_type=entity.type,
                )

            invalid_assertions = sorted(set(entity.assertions) - ALLOWED_ASSERTIONS)
            if invalid_assertions:
                add_error(
                    document.document_id,
                    entity_index,
                    "unsupported_assertion",
                    assertions=invalid_assertions,
                )
            if entity.assertions and entity.type not in ASSERTION_ENTITY_TYPES:
                add_error(
                    document.document_id,
                    entity_index,
                    "assertion_not_allowed",
                    entity_type=entity.type,
                )

            if entity.type == "DISEASE":
                if not entity.candidates:
                    add_error(document.document_id, entity_index, "missing_icd_candidate")
                for candidate in entity.candidates:
                    if candidate not in icd_ids:
                        add_error(
                            document.document_id,
                            entity_index,
                            "unknown_icd_candidate",
                            candidate=candidate,
                        )
            elif entity.type == "DRUG":
                if not entity.candidates:
                    add_error(document.document_id, entity_index, "missing_rxnorm_candidate")
                for candidate in entity.candidates:
                    if candidate not in rxnorm_ids:
                        add_error(
                            document.document_id,
                            entity_index,
                            "unknown_rxnorm_candidate",
                            candidate=candidate,
                        )
            elif entity.candidates:
                add_error(
                    document.document_id,
                    entity_index,
                    "candidate_not_allowed",
                    entity_type=entity.type,
                )

    return {
        "document_count": len(document_list),
        "entity_count": sum(type_counts.values()),
        "type_counts": dict(type_counts.most_common()),
        "assertion_counts": dict(assertion_counts.most_common()),
        "errors": errors,
        "is_valid": not errors,
    }
