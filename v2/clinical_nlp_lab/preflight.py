"""CPU-only, fail-closed validation for the training dataset and runtime KBs."""

from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dataset_quality import ASSERTION_ENTITY_TYPES, TRAINING_ENTITY_TYPES
from .kb import load_candidate_dictionary, sha256_file
from .provenance import (
    PROVENANCE_SCHEMA_ID,
    PROVENANCE_SCHEMA_VERSION,
    ProvenanceError,
    DatasetSnapshot,
    canonical_json_bytes,
    detect_report_conflicts,
    load_json_strict,
    load_jsonl_strict,
    scan_dataset_layout,
    sha256_bytes,
    verify_dataset_provenance,
)
from .runtime_control import PipelineContractError, atomic_write_json
from .schema import ALLOWED_ASSERTIONS, ClinicalDocument, EntityAnnotation, OFFICIAL_SCHEMA_KEYS, validate_submission_payload


VALIDATOR_VERSION = "1.0.0"
REPORT_SCHEMA_ID = "clinical_nlp.preflight_report"
REPORT_SCHEMA_VERSION = 1
REPORT_TYPE = "data_kb_preflight"
REPORT_SCOPE = {"dataset_role": "training_corpus", "kb_scope": "organizer_101_200"}

_OFFICIAL_TO_INTERNAL = {
    "CHẨN_ĐOÁN": "DISEASE",
    "THUỐC": "DRUG",
    "TRIỆU_CHỨNG": "SYMPTOM",
    "TÊN_XÉT_NGHIỆM": "LAB_NAME",
    "KẾT_QUẢ_XÉT_NGHIỆM": "LAB_RESULT",
}
_SOURCE_BUCKETS = {
    "reconstructed": "quarantine",
    "organizer_gt": "organizer",
    "synthetic": "synthetic",
}
_ICD_ID_RE = re.compile(r"[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?(?:[*†])?\Z")
_RXNORM_ID_RE = re.compile(r"[0-9]+\Z")


def normalize_source_bucket(source_bucket: str) -> str:
    """Normalize only the three source labels admitted by the dataset contract."""

    try:
        return _SOURCE_BUCKETS[source_bucket]
    except (KeyError, TypeError) as exc:
        raise PipelineContractError(
            "E_SOURCE_POLICY",
            "Manifest source_bucket is outside the explicit source policy.",
            context={"source_bucket": str(source_bucket)},
            next_action="Use reconstructed, organizer_gt, or synthetic explicitly.",
        ) from exc


def canonicalize_icd10_id(display_id: str) -> str:
    """Return the runtime lookup ID while preserving the caller's display ID."""

    return re.sub(r"[*†]+\Z", "", display_id).strip()


def _error(code: str, message: str, **details: Any) -> dict[str, Any]:
    error = PipelineContractError(code, message, context=details)
    return {"code": error.code, "message": error.message, **details}


def _append_unique(errors: list[dict[str, Any]], item: dict[str, Any]) -> None:
    # Collection stays O(1); duplicate suppression happens once at report boundaries.
    errors.append(item)


def _deduplicated_errors(errors: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    identities: set[str] = set()
    for item in errors:
        identity = json.dumps(item, ensure_ascii=False, sort_keys=True, default=str)
        if identity in identities:
            continue
        identities.add(identity)
        result.append(item)
    return result


def _provenance_error(error: ProvenanceError) -> dict[str, Any]:
    text = str(error)
    lowered = text.casefold()
    if "input/gt pairing mismatch" in lowered or "payload directory" in lowered or "no input/gt pairs" in lowered:
        code, reason = "E_DATASET_LAYOUT", "layout_or_pairing"
    elif "gt_sha256 mismatch" in lowered or "input_sha256 mismatch" in lowered or "_size_bytes mismatch" in lowered:
        code, reason = "E_DATASET_FILE_HASH", "raw_file_hash_mismatch"
    elif "pair_sha256 mismatch" in lowered:
        code, reason = "E_DATASET_PAIR_HASH", "pair_hash_mismatch"
    elif (
        "manifest.sha256 mismatch" in lowered
        or "manifest.size_bytes mismatch" in lowered
        or "descriptor manifest binding mismatch" in lowered
    ):
        code, reason = "E_DATASET_MANIFEST_HASH", "manifest_hash_mismatch"
    elif "dataset.fingerprint mismatch" in lowered:
        code, reason = "E_DATASET_FINGERPRINT", "dataset_fingerprint_mismatch"
    elif "manifest ids/order differ" in lowered or "duplicate manifest document_id" in lowered:
        code, reason = "E_MANIFEST_ENTRY", "manifest_identity_mismatch"
    elif "strict utf-8" in lowered or "utf-8 bom" in lowered:
        code, reason = "E_DATASET_ENCODING", "noncanonical_encoding"
    elif "schema/offset" in lowered:
        code, reason = "E_GT_SCHEMA", "invalid_gt_schema_or_offset"
    elif "missing active manifest" in lowered:
        code, reason = "E_DATASET_MANIFEST", "missing_manifest"
    else:
        code, reason = "E_DATASET_PROVENANCE", "invalid_or_missing_pair_provenance_contract"
    return _error(code, "Dataset provenance verification failed.", reason=reason)


def _numeric_id(document_id: str) -> int | None:
    if not re.fullmatch(r"[1-9][0-9]*", document_id):
        return None
    return int(document_id)


def _expected_policy(document_id: str) -> tuple[str, bool] | None:
    numeric = _numeric_id(document_id)
    if numeric is None:
        return None
    if 1 <= numeric <= 100:
        return "quarantine", False
    if 101 <= numeric <= 200:
        return "organizer", True
    if 201 <= numeric <= 2200:
        return "synthetic", True
    return None


def _safe_payload_paths(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    def paths(directory: Path, suffix: str) -> dict[str, Path]:
        if not directory.is_dir():
            return {}
        return {
            item.stem: item
            for item in directory.iterdir()
            if item.is_file() and not item.is_symlink() and item.suffix == suffix
        }

    return paths(root / "input", ".txt"), paths(root / "gt", ".json")


def _entity_error(
    errors: list[dict[str, Any]],
    code: str,
    document_id: str,
    entity_index: int,
    **details: Any,
) -> None:
    _append_unique(
        errors,
        _error(
            code,
            "Ground-truth entity violates the preflight contract.",
            document_id=document_id,
            entity_index=entity_index,
            **details,
        ),
    )


def _semantic_audit(root: Path) -> tuple[list[ClinicalDocument], dict[str, Any], list[dict[str, Any]]]:
    """Best-effort aggregate audit; error payloads never contain clinical text."""

    input_paths, gt_paths = _safe_payload_paths(root)
    errors: list[dict[str, Any]] = []
    documents: list[ClinicalDocument] = []
    entity_count = 0
    type_counts: Counter[str] = Counter()
    checked = 0
    for document_id in sorted(input_paths.keys() & gt_paths.keys(), key=lambda value: (_numeric_id(value) is None, _numeric_id(value) or 0, value)):
        try:
            raw_text = input_paths[document_id].read_text(encoding="utf-8")
            payload = load_json_strict(gt_paths[document_id].read_bytes(), source=str(gt_paths[document_id]))
        except (OSError, UnicodeError, ProvenanceError):
            _append_unique(errors, _error("E_GT_JSON", "Ground-truth JSON could not be read strictly.", document_id=document_id))
            continue
        checked += 1
        if not isinstance(payload, list):
            _append_unique(errors, _error("E_GT_SCHEMA", "Ground-truth top level must be an array.", document_id=document_id))
            continue
        document = ClinicalDocument(document_id=document_id, raw_text="")
        helper_errors = validate_submission_payload(payload, raw_text)
        document_had_specific_error = False
        for entity_index, item in enumerate(payload):
            entity_count += 1
            if not isinstance(item, dict):
                _entity_error(errors, "E_GT_SCHEMA", document_id, entity_index)
                document_had_specific_error = True
                continue
            entity_type = item.get("type")
            if not isinstance(entity_type, str) or entity_type not in OFFICIAL_SCHEMA_KEYS:
                _entity_error(errors, "E_GT_SCHEMA", document_id, entity_index)
                document_had_specific_error = True
                continue
            internal_type = _OFFICIAL_TO_INTERNAL[entity_type]
            type_counts[internal_type] += 1
            if internal_type not in TRAINING_ENTITY_TYPES:
                _entity_error(errors, "E_GT_TYPE", document_id, entity_index, entity_type=internal_type)
                document_had_specific_error = True
            expected_keys = OFFICIAL_SCHEMA_KEYS[entity_type]
            if set(item) != expected_keys:
                _entity_error(errors, "E_GT_SCHEMA", document_id, entity_index)
                document_had_specific_error = True
            text = item.get("text")
            position = item.get("position")
            valid_position = (
                isinstance(text, str)
                and isinstance(position, list | tuple)
                and len(position) == 2
                and all(type(value) is int for value in position)
                and 0 <= position[0] <= position[1] <= len(raw_text)
            )
            if not valid_position or raw_text[position[0] : position[1]] != text:
                _entity_error(errors, "E_GT_OFFSET", document_id, entity_index)
                document_had_specific_error = True
            assertions = item.get("assertions", [])
            if internal_type not in ASSERTION_ENTITY_TYPES and "assertions" in item:
                _entity_error(errors, "E_ASSERTION_SCOPE", document_id, entity_index, entity_type=internal_type)
                document_had_specific_error = True
            if not isinstance(assertions, list) or any(
                not isinstance(assertion, str) or assertion not in ALLOWED_ASSERTIONS
                for assertion in assertions
            ):
                _entity_error(errors, "E_ASSERTION_SCHEMA", document_id, entity_index)
                document_had_specific_error = True
                assertions = []
            candidates = item.get("candidates", [])
            if not isinstance(candidates, list) or any(not isinstance(candidate, str) for candidate in candidates):
                _entity_error(errors, "E_CANDIDATE_SCHEMA", document_id, entity_index)
                document_had_specific_error = True
                candidates = []
            elif len(candidates) > 1:
                _entity_error(errors, "E_CANDIDATE_CARDINALITY", document_id, entity_index, candidate_count=len(candidates))
                document_had_specific_error = True
            if candidates and internal_type not in {"DISEASE", "DRUG"}:
                _entity_error(errors, "E_CANDIDATE_SCOPE", document_id, entity_index, entity_type=internal_type)
                document_had_specific_error = True
            for candidate in candidates:
                pattern = _ICD_ID_RE if internal_type == "DISEASE" else _RXNORM_ID_RE if internal_type == "DRUG" else None
                if pattern is not None and pattern.fullmatch(candidate) is None:
                    _entity_error(
                        errors,
                        "E_CANDIDATE_ONTOLOGY",
                        document_id,
                        entity_index,
                        candidate_id=candidate,
                        entity_type=internal_type,
                    )
                    document_had_specific_error = True
            if isinstance(text, str) and valid_position:
                document.entities.append(
                    EntityAnnotation(
                        text="",
                        type=internal_type,
                        position=(int(position[0]), int(position[1])),
                        candidates=list(candidates),
                        assertions=list(assertions),
                    )
                )
        if helper_errors and not document_had_specific_error:
            _append_unique(errors, _error("E_GT_SCHEMA", "Ground-truth schema validation failed.", document_id=document_id))
        documents.append(document)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "documents_checked": checked,
        "entity_count": entity_count,
        "type_counts": dict(sorted(type_counts.items())),
        "error_count": len(errors),
    }
    return documents, result, errors


def _load_manifest_rows(path: Path, errors: list[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    if not path.is_file() or path.is_symlink():
        _append_unique(errors, _error("E_DATASET_MANIFEST", "Canonical dataset manifest is missing."))
        return ()
    try:
        return load_jsonl_strict(path.read_bytes(), source=str(path))
    except (OSError, ProvenanceError):
        _append_unique(errors, _error("E_DATASET_MANIFEST", "Canonical dataset manifest is invalid."))
        return ()


def _audit_manifest_policy(
    rows: Sequence[Mapping[str, Any]],
    document_ids: Sequence[str],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    by_id: dict[str, Mapping[str, Any]] = {}
    duplicates: list[str] = []
    for row in rows:
        document_id = row.get("document_id")
        if not isinstance(document_id, str):
            _append_unique(errors, _error("E_MANIFEST_SCHEMA", "Manifest record has no valid document_id."))
            continue
        if document_id in by_id:
            duplicates.append(document_id)
        by_id[document_id] = row
    expected = set(document_ids)
    actual = set(by_id)
    if duplicates or expected != actual:
        _append_unique(
            errors,
            _error(
                "E_MANIFEST_ENTRY",
                "Manifest must contain exactly one record for every document.",
                duplicate_ids=sorted(set(duplicates), key=lambda value: (_numeric_id(value) is None, _numeric_id(value) or 0, value)),
                missing_ids=sorted(expected - actual, key=lambda value: (_numeric_id(value) is None, _numeric_id(value) or 0, value)),
                unknown_ids=sorted(actual - expected, key=lambda value: (_numeric_id(value) is None, _numeric_id(value) or 0, value)),
            ),
        )
    counts: Counter[str] = Counter()
    for document_id in sorted(expected & actual, key=lambda value: (_numeric_id(value) is None, _numeric_id(value) or 0, value)):
        row = by_id[document_id]
        policy = _expected_policy(document_id)
        if policy is None:
            _append_unique(errors, _error("E_SOURCE_POLICY", "Document ID is outside the admitted numeric policy.", document_id=document_id))
            continue
        expected_bucket, expected_eligible = policy
        source_bucket = row.get("source_bucket")
        train_eligible = row.get("train_eligible")
        try:
            normalized_bucket = normalize_source_bucket(source_bucket)  # type: ignore[arg-type]
        except PipelineContractError:
            _append_unique(errors, _error("E_SOURCE_POLICY", "Manifest source_bucket is invalid.", document_id=document_id))
            continue
        counts[normalized_bucket] += 1
        if normalized_bucket != expected_bucket or type(train_eligible) is not bool or train_eligible is not expected_eligible:
            _append_unique(
                errors,
                _error(
                    "E_SOURCE_POLICY",
                    "Manifest source or eligibility violates numeric-ID policy.",
                    document_id=document_id,
                    source_bucket=normalized_bucket,
                    train_eligible=train_eligible if type(train_eligible) is bool else None,
                ),
            )
    return dict(sorted(counts.items())), errors


def _inspect_dataset_layout(dataset_root: str | Path) -> dict[str, Any]:
    """Validate dataset identity, detached provenance, policy, schema, and offsets."""

    root = Path(dataset_root)
    errors: list[dict[str, Any]] = []
    snapshot: DatasetSnapshot | None = None
    try:
        snapshot = scan_dataset_layout(root)
    except (OSError, ProvenanceError) as exc:
        _append_unique(errors, _provenance_error(ProvenanceError(str(exc))))

    documents, schema_result, semantic_errors = _semantic_audit(root)
    for item in semantic_errors:
        _append_unique(errors, item)

    manifest_path = root / "reports" / "dataset_manifest.jsonl"
    rows = _load_manifest_rows(manifest_path, errors)
    input_paths, gt_paths = _safe_payload_paths(root)
    document_ids = snapshot.document_ids if snapshot is not None else tuple(sorted(input_paths.keys() & gt_paths.keys(), key=lambda value: (_numeric_id(value) is None, _numeric_id(value) or 0, value)))
    source_counts, policy_errors = _audit_manifest_policy(rows, document_ids)
    for item in policy_errors:
        _append_unique(errors, item)
    source_bucket_by_id: dict[str, str] = {}
    for row in rows:
        document_id = row.get("document_id")
        source_bucket = row.get("source_bucket")
        if not isinstance(document_id, str) or not isinstance(source_bucket, str):
            continue
        try:
            source_bucket_by_id[document_id] = normalize_source_bucket(source_bucket)
        except PipelineContractError:
            continue
    semantic_bucket_counts: Counter[str] = Counter()
    for item in semantic_errors:
        document_id = item.get("document_id")
        normalized_bucket = source_bucket_by_id.get(document_id) if isinstance(document_id, str) else None
        if normalized_bucket is not None:
            item["source_bucket"] = normalized_bucket
            semantic_bucket_counts[normalized_bucket] += 1
    schema_result["error_counts_by_source_bucket"] = dict(sorted(semantic_bucket_counts.items()))

    manifest_sha256: str | None = None
    descriptor: dict[str, Any] | None = None
    if snapshot is not None:
        try:
            verification = verify_dataset_provenance(root)
            manifest_sha256 = verification.manifest_sha256
            descriptor = verification.descriptor
        except (OSError, ProvenanceError) as exc:
            _append_unique(errors, _provenance_error(ProvenanceError(str(exc))))
            if manifest_path.is_file():
                manifest_sha256 = sha256_file(manifest_path)

    errors = _deduplicated_errors(errors)
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "counts": {
            "input_files": len(input_paths),
            "gt_files": len(gt_paths),
            "paired_documents": len(document_ids),
            "manifest_records": len(rows),
        },
        "document_ids": list(document_ids),
        "dataset_pair_fingerprint": snapshot.dataset_fingerprint if snapshot is not None else None,
        "manifest_sha256": manifest_sha256,
        "source_bucket_counts": source_counts,
        "schema_offset_validation": schema_result,
        "provenance_schema": {
            "schema_id": descriptor.get("schema_id") if descriptor else None,
            "schema_version": descriptor.get("schema_version") if descriptor else None,
            "expected_schema_id": PROVENANCE_SCHEMA_ID,
            "expected_schema_version": PROVENANCE_SCHEMA_VERSION,
        },
        "_documents": documents,
    }


def inspect_dataset_layout(dataset_root: str | Path) -> dict[str, Any]:
    """Return the serializable public dataset inspection without parsed entities."""

    result = _inspect_dataset_layout(dataset_root)
    result.pop("_documents", None)
    return result


def _coverage_bucket(total: int, covered: int, missing: list[dict[str, Any]]) -> dict[str, Any]:
    display_ids = sorted({str(item["display_id"]) for item in missing})
    lookup_ids = sorted({str(item["lookup_id"]) for item in missing})
    return {
        "total_non_empty_occurrences": total,
        "covered_occurrences": covered,
        "missing_occurrence_count": len(missing),
        "missing_occurrences": missing,
        "missing_unique_ids": display_ids,
        "missing_unique_lookup_ids": lookup_ids,
        "coverage_rate": covered / total if total else 1.0,
    }


def audit_organizer_kb_coverage(
    documents: Iterable[ClinicalDocument],
    icd_ids: set[str],
    rxnorm_ids: set[str],
) -> dict[str, Any]:
    """Audit non-empty organizer gold candidates against runtime dictionaries."""

    totals = {"icd10": 0, "rxnorm": 0}
    covered = {"icd10": 0, "rxnorm": 0}
    missing: dict[str, list[dict[str, Any]]] = {"icd10": [], "rxnorm": []}
    for document in documents:
        numeric = _numeric_id(document.document_id)
        if numeric is None or not 101 <= numeric <= 200:
            continue
        for entity_index, entity in enumerate(document.entities):
            if entity.type not in {"DISEASE", "DRUG"}:
                continue
            ontology = "icd10" if entity.type == "DISEASE" else "rxnorm"
            admitted = icd_ids if ontology == "icd10" else rxnorm_ids
            for display_id in entity.candidates:
                if not display_id:
                    continue
                totals[ontology] += 1
                lookup_id = canonicalize_icd10_id(display_id) if ontology == "icd10" else display_id
                if lookup_id in admitted:
                    covered[ontology] += 1
                else:
                    missing[ontology].append(
                        {
                            "document_id": document.document_id,
                            "entity_index": entity_index,
                            "display_id": display_id,
                            "lookup_id": lookup_id,
                        }
                    )
    buckets = {
        ontology: _coverage_bucket(totals[ontology], covered[ontology], missing[ontology])
        for ontology in ("icd10", "rxnorm")
    }
    total = sum(totals.values())
    total_covered = sum(covered.values())
    return {
        "target_coverage_rate": 1.0,
        "status": "PASS" if total == total_covered else "FAIL",
        "total_non_empty_occurrences": total,
        "covered_occurrences": total_covered,
        "missing_occurrence_count": total - total_covered,
        "coverage_rate": total_covered / total if total else 1.0,
        **buckets,
    }


def _load_kb_ids(path: Path, ontology: str, errors: list[dict[str, Any]]) -> tuple[set[str], str | None]:
    if not path.is_file() or path.is_symlink():
        _append_unique(errors, _error("E_RUNTIME_KB", "Required runtime KB artifact is missing.", ontology=ontology))
        return set(), None
    artifact_hash = sha256_file(path)
    try:
        records = load_candidate_dictionary(path)
    except (OSError, EOFError, ValueError, json.JSONDecodeError):
        _append_unique(errors, _error("E_RUNTIME_KB", "Required runtime KB artifact is invalid.", ontology=ontology))
        return set(), artifact_hash
    if not records or any(not isinstance(record, Mapping) for record in records):
        _append_unique(errors, _error("E_RUNTIME_KB", "Runtime KB records are invalid.", ontology=ontology))
        return set(), artifact_hash
    candidate_ids = [record.get("candidate_id") for record in records]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        _append_unique(errors, _error("E_RUNTIME_KB", "Runtime KB candidate IDs are invalid.", ontology=ontology))
        return set(), artifact_hash
    ids = set(candidate_ids)
    if len(ids) != len(candidate_ids):
        _append_unique(errors, _error("E_RUNTIME_KB", "Runtime KB contains duplicate candidate IDs.", ontology=ontology))
    return ids, artifact_hash


def _audit_config(path: Path) -> tuple[dict[str, Any], str | None, list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    if not path.is_file() or path.is_symlink():
        return {}, None, [_error("E_CONFIG_CONTRACT", "Runtime config is missing.")]
    config_hash = sha256_file(path)
    try:
        payload = load_json_strict(path.read_bytes(), source=str(path))
    except (OSError, ProvenanceError):
        return {}, config_hash, [_error("E_CONFIG_CONTRACT", "Runtime config is invalid.")]
    if not isinstance(payload, dict):
        return {}, config_hash, [_error("E_CONFIG_CONTRACT", "Runtime config must be an object.")]
    expected = {
        "candidate_output_k": 1,
        "candidate_top_k": 20,
        "enable_regex_fallback": False,
    }
    for field, required in expected.items():
        actual = payload.get(field)
        same_type = type(actual) is type(required)
        if not same_type or actual != required:
            errors.append(
                _error(
                    "E_CONFIG_CONTRACT",
                    "Runtime linking configuration violates the preflight contract.",
                    field=field,
                    expected=required,
                    actual=actual if isinstance(actual, bool | int | float | str) or actual is None else type(actual).__name__,
                )
            )
    return payload, config_hash, errors


def _prior_report_inventory(
    reports_dir: Path,
    dataset_fingerprint: str | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    envelopes: list[Mapping[str, Any]] = []
    if not reports_dir.is_dir():
        return inventory, warnings, []
    for path in sorted(reports_dir.glob("*.json"), key=lambda item: item.name):
        if path.name == "dataset_provenance.json":
            continue
        try:
            payload = load_json_strict(path.read_bytes(), source=str(path))
        except (OSError, ProvenanceError):
            inventory.append({"relative_path": path.name, "effective_status": "unverifiable", "reason": "invalid_json"})
            continue
        if not isinstance(payload, dict):
            inventory.append({"relative_path": path.name, "effective_status": "unverifiable", "reason": "not_an_object"})
            continue
        if payload.get("report_type") != REPORT_TYPE or payload.get("scope") != REPORT_SCOPE:
            looks_like_legacy_report = any(
                marker in path.stem.casefold() for marker in ("report", "audit")
            )
            if "report_type" not in payload or "scope" not in payload:
                if not looks_like_legacy_report:
                    continue
                inventory.append({"relative_path": path.name, "effective_status": "unverifiable", "reason": "missing_report_identity"})
            else:
                inventory.append({"relative_path": path.name, "effective_status": "unrelated", "reason": "different_report_type_or_scope"})
            continue
        stored_fingerprint = payload.get("dataset_pair_fingerprint")
        if stored_fingerprint is None and isinstance(payload.get("fingerprints"), Mapping):
            stored_fingerprint = payload["fingerprints"].get("dataset")
        required_fields = ("schema_id", "schema_version", "status")
        if any(field not in payload for field in required_fields) or not isinstance(stored_fingerprint, str):
            effective, reason = "unverifiable", "missing_schema_status_or_fingerprint"
        elif dataset_fingerprint is None or stored_fingerprint != dataset_fingerprint:
            effective, reason = "stale", "dataset_fingerprint_mismatch"
        else:
            effective, reason = "current", "matching_fingerprint"
        inventory.append({"relative_path": path.name, "effective_status": effective, "reason": reason})
        envelope = dict(payload)
        if not isinstance(envelope.get("payload_sha256"), str):
            envelope["payload_sha256"] = _stable_report_payload_sha256(envelope)
        envelopes.append(envelope)
    stale_or_unverifiable = sum(item["effective_status"] in {"stale", "unverifiable"} for item in inventory)
    if stale_or_unverifiable:
        warnings.append(
            {
                "code": "W_PRIOR_REPORTS_NOT_CURRENT",
                "message": "Prior reports were inventoried as stale or unverifiable and were not used as PASS evidence.",
                "count": stale_or_unverifiable,
            }
        )
    conflicts = list(detect_report_conflicts(envelopes, ("dataset",)))
    return inventory, warnings, conflicts


def _stable_report_payload_sha256(report: Mapping[str, Any]) -> str:
    """Hash semantic evidence, excluding timestamps and recursive inventory metadata."""

    fields = (
        "status",
        "errors",
        "counts",
        "dataset_pair_fingerprint",
        "manifest_sha256",
        "config_sha256",
        "kb_hashes",
        "source_bucket_counts",
        "schema_offset_validation",
        "dataset_provenance",
        "organizer_kb_coverage",
    )
    payload = {field: report.get(field) for field in fields}
    return sha256_bytes(canonical_json_bytes(payload))


def build_preflight_report(
    dataset_root: str | Path,
    artifact_dir: str | Path,
    config_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Aggregate every independent hard gate and optionally publish one report."""

    dataset = _inspect_dataset_layout(dataset_root)
    documents = dataset.pop("_documents")
    errors = list(dataset["errors"])
    warnings: list[dict[str, Any]] = []

    artifacts = Path(artifact_dir)
    icd_ids, icd_hash = _load_kb_ids(
        artifacts / "icd10" / "icd10_dictionary.jsonl.gz", "icd10", errors
    )
    rxnorm_ids, rxnorm_hash = _load_kb_ids(
        artifacts / "rxnorm" / "rxnorm_dictionary.jsonl.gz", "rxnorm", errors
    )
    coverage = audit_organizer_kb_coverage(documents, icd_ids, rxnorm_ids)
    if coverage["status"] != "PASS":
        errors.append(
            _error(
                "E_ORGANIZER_KB_COVERAGE",
                "Organizer non-empty gold candidates are missing from runtime KB artifacts.",
                missing_occurrence_count=coverage["missing_occurrence_count"],
                icd10_missing_unique_count=len(coverage["icd10"]["missing_unique_ids"]),
                rxnorm_missing_unique_count=len(coverage["rxnorm"]["missing_unique_ids"]),
            )
        )

    _config, config_hash, config_errors = _audit_config(Path(config_path))
    errors.extend(config_errors)
    inventory, inventory_warnings, conflicts = _prior_report_inventory(
        Path(dataset_root) / "reports", dataset["dataset_pair_fingerprint"]
    )
    warnings.extend(inventory_warnings)
    if conflicts:
        errors.append(
            _error(
                "E_REPORT_CONFLICT",
                "Current reports with the same report type and scope conflict.",
                conflict_count=len(conflicts),
            )
        )

    errors = _deduplicated_errors(errors)

    fingerprints = {
        "dataset": dataset["dataset_pair_fingerprint"],
        "manifest": dataset["manifest_sha256"],
        "config": config_hash,
        "icd10": icd_hash,
        "rxnorm": rxnorm_hash,
    }
    report = {
        "schema_id": REPORT_SCHEMA_ID,
        "schema_version": REPORT_SCHEMA_VERSION,
        "report_type": REPORT_TYPE,
        "scope": REPORT_SCOPE,
        "validator_version": VALIDATOR_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status_at_creation": "current",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "counts": dataset["counts"],
        "dataset_pair_fingerprint": dataset["dataset_pair_fingerprint"],
        "manifest_sha256": dataset["manifest_sha256"],
        "config_sha256": config_hash,
        "kb_hashes": {"icd10": icd_hash, "rxnorm": rxnorm_hash},
        "fingerprints": fingerprints,
        "source_bucket_counts": dataset["source_bucket_counts"],
        "schema_offset_validation": dataset["schema_offset_validation"],
        "dataset_provenance": dataset["provenance_schema"],
        "organizer_kb_coverage": coverage,
        "prior_report_inventory": inventory,
        "report_conflicts": conflicts,
    }
    report["payload_sha256"] = _stable_report_payload_sha256(report)
    if output_path is not None:
        atomic_write_json(output_path, report)
    return report


__all__ = [
    "REPORT_SCHEMA_ID",
    "REPORT_SCHEMA_VERSION",
    "VALIDATOR_VERSION",
    "audit_organizer_kb_coverage",
    "build_preflight_report",
    "canonicalize_icd10_id",
    "inspect_dataset_layout",
    "normalize_source_bucket",
]
