from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .provenance import DatasetSnapshot, scan_dataset_layout
from .text import normalize_alias


KB_CONTRACT_VERSION = "1.0.0"
_RXCUI_RE = re.compile(r"[1-9][0-9]*\Z")
_ICD_TRAILING_MARKERS_RE = re.compile(r"[*†]+\Z")


class KBContractError(ValueError):
    """Raised when a KB build cannot prove the pinned organizer contract."""


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    canonical_id: str
    official_display_id: str

    def __post_init__(self) -> None:
        if not self.canonical_id or not self.official_display_id:
            raise KBContractError("Candidate identity fields must be non-empty strings")

    def to_dict(self) -> dict[str, str]:
        return {
            "canonical_id": self.canonical_id,
            "official_display_id": self.official_display_id,
        }


@dataclass(frozen=True, slots=True)
class OrganizerCandidateOccurrence:
    document_id: str
    entity_index: int
    ontology: str
    identity: CandidateIdentity
    mention_text: str
    mention_sha256: str

    def evidence_dict(self) -> dict[str, Any]:
        """Return evidence safe for reports; clinical mention text is never emitted."""

        return {
            "document_id": self.document_id,
            "entity_index": self.entity_index,
            "ontology": self.ontology,
            **self.identity.to_dict(),
            "mention_sha256": self.mention_sha256,
        }


def canonical_icd_id(display_id: str) -> str:
    """Strip only official ICD suffix markers, never marker-like internal text."""

    cleaned = str(display_id).strip()
    canonical = _ICD_TRAILING_MARKERS_RE.sub("", cleaned).strip()
    if not canonical:
        raise KBContractError(f"Invalid ICD display ID: {display_id!r}")
    return canonical


def candidate_identity(ontology: str, official_display_id: str) -> CandidateIdentity:
    display = str(official_display_id).strip()
    if not display:
        raise KBContractError("Candidate ID must be non-empty")
    if ontology == "icd10":
        canonical = canonical_icd_id(display)
    elif ontology == "rxnorm":
        if _RXCUI_RE.fullmatch(display) is None:
            raise KBContractError(f"Invalid RxNorm candidate ID: {display!r}")
        canonical = display
    else:
        raise KBContractError(f"Unsupported ontology: {ontology!r}")
    return CandidateIdentity(canonical, display)


def official_output_id(
    record: Mapping[str, Any], requested_display_id: str | None = None
) -> CandidateIdentity:
    canonical = str(record.get("canonical_id") or record.get("candidate_id") or "").strip()
    if not canonical:
        raise KBContractError("Runtime candidate record has no canonical ID")
    displays = tuple(
        dict.fromkeys(
            str(value).strip()
            for value in (record.get("official_display_ids") or record.get("display_codes") or [canonical])
            if str(value).strip()
        )
    )
    if requested_display_id is not None:
        requested = str(requested_display_id).strip()
        if requested not in displays:
            raise KBContractError(
                f"Unknown official display ID {requested!r} for canonical candidate {canonical!r}"
            )
        return CandidateIdentity(canonical, requested)
    preferred = canonical if canonical in displays else displays[0]
    return CandidateIdentity(canonical, preferred)


def validate_icd_record_identity(record: Mapping[str, Any]) -> tuple[CandidateIdentity, ...]:
    canonical = str(record.get("canonical_id") or record.get("candidate_id") or "").strip()
    displays = record.get("official_display_ids") or record.get("display_codes")
    if not canonical or not isinstance(displays, list) or not displays:
        raise KBContractError("ICD record must contain canonical_id and official_display_ids")
    identities: list[CandidateIdentity] = []
    for display in displays:
        identity = CandidateIdentity(canonical, str(display).strip())
        if canonical_icd_id(identity.official_display_id) != canonical:
            raise KBContractError(
                f"ICD display/canonical mismatch: {identity.official_display_id!r} != {canonical!r}"
            )
        identities.append(identity)
    if len({item.official_display_id for item in identities}) != len(identities):
        raise KBContractError(f"Duplicate official display ID for canonical candidate {canonical!r}")
    return tuple(identities)


def _load_gt_entities(raw: bytes, document_id: str) -> list[dict[str, Any]]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KBContractError(f"Invalid GT JSON for document {document_id}") from exc
    if isinstance(payload, dict) and "entities" in payload:
        payload = payload["entities"]
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise KBContractError(f"GT for document {document_id} must be an entity array")
    return payload


def collect_organizer_candidate_occurrences(
    dataset_root: str | Path,
    *,
    start_id: int = 101,
    end_id: int = 200,
) -> tuple[DatasetSnapshot, tuple[OrganizerCandidateOccurrence, ...], int]:
    snapshot = scan_dataset_layout(dataset_root)
    occurrences: list[OrganizerCandidateOccurrence] = []
    abstentions = 0
    expected_ids = {str(value) for value in range(start_id, end_id + 1)}
    actual_ids = {pair.document_id for pair in snapshot.pairs if start_id <= int(pair.document_id) <= end_id}
    if actual_ids != expected_ids:
        raise KBContractError(
            f"Organizer document range is incomplete: missing={sorted(expected_ids - actual_ids)}"
        )
    for pair in snapshot.pairs:
        numeric_id = int(pair.document_id)
        if not start_id <= numeric_id <= end_id:
            continue
        # Competition offsets are defined after Python's universal-newline text read.
        raw_text = pair.input_bytes.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        for entity_index, entity in enumerate(_load_gt_entities(pair.gt_bytes, pair.document_id)):
            entity_type = str(entity.get("type", ""))
            if entity_type in {"CHẨN_ĐOÁN", "DISEASE"}:
                ontology = "icd10"
            elif entity_type in {"THUỐC", "DRUG"}:
                ontology = "rxnorm"
            else:
                continue
            candidates = entity.get("candidates", [])
            if not isinstance(candidates, list):
                raise KBContractError(
                    f"Candidates must be a list at document {pair.document_id}, entity {entity_index}"
                )
            non_empty = [str(value).strip() for value in candidates if str(value).strip()]
            if not non_empty:
                abstentions += 1
                continue
            mention = str(entity.get("text", ""))
            position = entity.get("position")
            if (
                not isinstance(position, list)
                or len(position) != 2
                or type(position[0]) is not int
                or type(position[1]) is not int
                or not (0 <= position[0] <= position[1] <= len(raw_text))
                or raw_text[position[0] : position[1]] != mention
            ):
                raise KBContractError(
                    f"Invalid entity offset at document {pair.document_id}, entity {entity_index}"
                )
            mention_sha = hashlib.sha256(mention.encode("utf-8")).hexdigest()
            for display_id in non_empty:
                occurrences.append(
                    OrganizerCandidateOccurrence(
                        document_id=pair.document_id,
                        entity_index=entity_index,
                        ontology=ontology,
                        identity=candidate_identity(ontology, display_id),
                        mention_text=mention,
                        mention_sha256=mention_sha,
                    )
                )
    occurrences.sort(
        key=lambda item: (int(item.document_id), item.entity_index, item.identity.official_display_id)
    )
    return snapshot, tuple(occurrences), abstentions


def organizer_rxnorm_requirements(
    occurrences: Iterable[OrganizerCandidateOccurrence],
) -> dict[str, tuple[OrganizerCandidateOccurrence, ...]]:
    grouped: dict[str, list[OrganizerCandidateOccurrence]] = {}
    for occurrence in occurrences:
        if occurrence.ontology == "rxnorm":
            grouped.setdefault(occurrence.identity.canonical_id, []).append(occurrence)
    return {candidate_id: tuple(rows) for candidate_id, rows in grouped.items()}


def audit_gold_candidate_coverage(
    occurrences: Iterable[OrganizerCandidateOccurrence | Mapping[str, Any]],
    icd_records: Iterable[Mapping[str, Any]] | set[str],
    rxnorm_records: Iterable[Mapping[str, Any]] | set[str] | None = None,
    *,
    abstentions: int = 0,
) -> dict[str, Any]:
    """Audit canonical lookup and exact ICD display identity without leaking mentions."""

    if isinstance(icd_records, set):
        icd_map = {value: {"candidate_id": value, "canonical_id": value, "official_display_ids": [value]} for value in icd_records}
    else:
        icd_map = {str(row["candidate_id"]): row for row in icd_records}
    if rxnorm_records is None:
        rx_map: dict[str, Mapping[str, Any]] = {}
    elif isinstance(rxnorm_records, set):
        rx_map = {value: {"candidate_id": value, "canonical_id": value} for value in rxnorm_records}
    else:
        rx_map = {str(row["candidate_id"]): row for row in rxnorm_records}

    totals = {"icd10": 0, "rxnorm": 0}
    covered = {"icd10": 0, "rxnorm": 0}
    unresolved: list[dict[str, Any]] = []
    for raw_occurrence in occurrences:
        if isinstance(raw_occurrence, OrganizerCandidateOccurrence):
            occurrence = raw_occurrence
        else:
            ontology = str(raw_occurrence.get("ontology", "icd10"))
            display = str(raw_occurrence.get("official_display_id") or raw_occurrence.get("candidate") or "")
            identity = candidate_identity(ontology, display)
            occurrence = OrganizerCandidateOccurrence(
                document_id=str(raw_occurrence.get("document_id", "")),
                entity_index=int(raw_occurrence.get("entity_index", 0)),
                ontology=ontology,
                identity=identity,
                mention_text="",
                mention_sha256=str(raw_occurrence.get("mention_sha256", "")),
            )
        totals[occurrence.ontology] += 1
        admitted = icd_map if occurrence.ontology == "icd10" else rx_map
        record = admitted.get(occurrence.identity.canonical_id)
        reason: str | None = None
        if record is None:
            reason = "canonical_id_missing"
        elif occurrence.ontology == "icd10":
            displays = set(record.get("official_display_ids") or record.get("display_codes") or [])
            if occurrence.identity.official_display_id not in displays:
                reason = "official_display_id_missing"
        if reason is None:
            covered[occurrence.ontology] += 1
        else:
            unresolved.append({**occurrence.evidence_dict(), "reason": reason})

    total = sum(totals.values())
    covered_total = sum(covered.values())
    return {
        "status": "PASS" if covered_total == total else "FAIL",
        "total_non_empty_occurrences": total,
        "covered_occurrences": covered_total,
        "missing_occurrence_count": len(unresolved),
        "abstentions": abstentions,
        "coverage_rate": covered_total / total if total else 1.0,
        "counts_by_ontology": totals,
        "covered_by_ontology": covered,
        "unresolved": unresolved,
    }


def normalized_surface_matches(left: str, right: str) -> bool:
    return normalize_alias(left) == normalize_alias(right)


__all__ = [
    "CandidateIdentity",
    "KBContractError",
    "KB_CONTRACT_VERSION",
    "OrganizerCandidateOccurrence",
    "audit_gold_candidate_coverage",
    "candidate_identity",
    "canonical_icd_id",
    "collect_organizer_candidate_occurrences",
    "normalized_surface_matches",
    "official_output_id",
    "organizer_rxnorm_requirements",
    "validate_icd_record_identity",
]
