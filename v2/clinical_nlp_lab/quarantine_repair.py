from __future__ import annotations

import gzip
import json
import os
import re
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .provenance import (
    MANIFEST_ROW_SCHEMA_ID,
    DatasetPair,
    ProvenanceError,
    compute_dataset_fingerprint,
    compute_pair_sha256,
    load_json_strict,
    load_jsonl_strict,
    scan_dataset_layout,
    sha256_bytes,
    validate_artifact_path,
    validate_legacy_manifest,
    validate_v2_manifest,
)
from .schema import OFFICIAL_SCHEMA_KEYS


REPAIR_SCHEMA_ID = "clinical_nlp.quarantine_gt_repair"
REPAIR_SCHEMA_VERSION = 1
REPAIR_TOOL_VERSION = "1.0.0"
_ICD_ID_RE = re.compile(r"[A-Z][0-9][0-9A-Z](?:\.[0-9A-Z]+)?(?:[*†])?\Z")


class QuarantineRepairError(ValueError):
    """Raised when a quarantine-only repair cannot be proven safe."""


@dataclass(frozen=True, slots=True)
class CandidateRepairRule:
    document_id: str
    entity_index: int
    expected_candidates: tuple[str, ...]
    replacement_candidates: tuple[str, ...]
    evidence_code: str


PINNED_REPAIR_RULES: tuple[CandidateRepairRule, ...] = (
    CandidateRepairRule("11", 50, ("T05.5", "T05.6"), ("T05.5",), "english_specific_both_legs"),
    CandidateRepairRule("28", 38, ("J99", "J99.8"), ("J99",), "generic_parent_description"),
    CandidateRepairRule("62", 14, ("S77.1", "S771"), ("S77.1",), "dotted_alias_canonicalization"),
    CandidateRepairRule("62", 38, ("D80.2", "D80.3"), ("D80.2",), "iga_specific_description"),
    CandidateRepairRule("66", 13, ("S27.80", "S27.81"), (), "ambiguous_kb_surface_abstention"),
    CandidateRepairRule("68", 10, ("D63", "D63.8"), ("D63",), "generic_parent_description"),
    CandidateRepairRule("70", 74, ("Q77.0", "Q77.4"), (), "ambiguous_kb_surface_abstention"),
    CandidateRepairRule("73", 36, ("E07.1", "E07.8"), ("E07.8",), "other_specified_description"),
    CandidateRepairRule("84", 82, ("D56.2", "D562"), ("D56.2",), "dotted_alias_canonicalization"),
    CandidateRepairRule("85", 36, ("Q77.0", "Q77.4"), (), "ambiguous_kb_surface_abstention"),
    CandidateRepairRule("86", 2, ("C53", "C54.2"), ("C54.2",), "myometrium_specific_description"),
    CandidateRepairRule("90", 13, ("D42", "D43"), ("D42",), "meninges_specific_description"),
    CandidateRepairRule("99", 48, ("Q90.1", "Q91.1"), (), "ambiguous_kb_surface_abstention"),
)


@dataclass(frozen=True, slots=True)
class PlannedGTFile:
    document_id: str
    path: Path
    before_bytes: bytes
    after_bytes: bytes
    before_sha256: str
    after_sha256: str
    changes: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class QuarantineRepairPlan:
    dataset_root: Path
    manifest_sha256: str
    dataset_fingerprint_before: str
    dataset_fingerprint_after: str
    icd_artifact_sha256: str
    files: tuple[PlannedGTFile, ...]
    evidence: dict[str, Any]

    @property
    def repair_count(self) -> int:
        return sum(len(item.changes) for item in self.files)


@dataclass(frozen=True, slots=True)
class QuarantineRepairResult:
    mode: str
    migration_required: bool
    dataset_fingerprint_before: str
    dataset_fingerprint_after: str
    repaired_entity_count: int
    repaired_file_count: int
    evidence_path: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "migration_required": self.migration_required,
            "dataset_fingerprint_before": self.dataset_fingerprint_before,
            "dataset_fingerprint_after": self.dataset_fingerprint_after,
            "repaired_entity_count": self.repaired_entity_count,
            "repaired_file_count": self.repaired_file_count,
            "evidence_path": self.evidence_path,
        }


def _serialize_gt(payload: Any) -> bytes:
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)
    return (text.replace("\n", "\r\n") + "\r\n").encode("utf-8")


def _validated_manifest_rows(root: Path, snapshot: Any) -> tuple[tuple[dict[str, Any], ...], bytes]:
    manifest_path = validate_artifact_path(root, root / "reports/dataset_manifest.jsonl")
    raw = manifest_path.read_bytes()
    rows = load_jsonl_strict(raw, source="active dataset manifest")
    if rows and rows[0].get("schema_id") == MANIFEST_ROW_SCHEMA_ID:
        return validate_v2_manifest(rows, snapshot), raw
    return validate_legacy_manifest(rows, snapshot), raw


def _load_icd_ids(path: Path) -> tuple[set[str], str]:
    if path.is_symlink() or not path.is_file():
        raise QuarantineRepairError("Runtime ICD artifact is missing or unsafe")
    raw = path.read_bytes()
    admitted: set[str] = set()
    try:
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise QuarantineRepairError("Runtime ICD artifact row is not an object")
                canonical = row.get("canonical_id") or row.get("candidate_id")
                if not isinstance(canonical, str) or not canonical:
                    raise QuarantineRepairError("Runtime ICD artifact row has no canonical ID")
                admitted.add(canonical)
                for display in row.get("official_display_ids", []):
                    if isinstance(display, str):
                        admitted.add(display)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise QuarantineRepairError("Runtime ICD artifact cannot be parsed") from exc
    return admitted, sha256_bytes(raw)


def _validate_replacement(candidates: Sequence[str], admitted_icd_ids: set[str]) -> None:
    if len(candidates) > 1:
        raise QuarantineRepairError("Repair replacement violates candidate cardinality")
    for candidate in candidates:
        if not isinstance(candidate, str) or not _ICD_ID_RE.fullmatch(candidate):
            raise QuarantineRepairError("Repair replacement has invalid ICD shape")
        canonical = candidate.rstrip("*†")
        if candidate not in admitted_icd_ids and canonical not in admitted_icd_ids:
            raise QuarantineRepairError("Repair replacement is absent from runtime ICD")


def _strict_gt_payload(payload: Any, document_id: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise QuarantineRepairError(f"GT document {document_id} is not an array")
    result: list[dict[str, Any]] = []
    for entity_index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise QuarantineRepairError(
                f"GT document {document_id} entity {entity_index} is not an object"
            )
        entity_type = item.get("type")
        if entity_type not in OFFICIAL_SCHEMA_KEYS or set(item) != OFFICIAL_SCHEMA_KEYS[entity_type]:
            raise QuarantineRepairError(
                f"GT document {document_id} entity {entity_index} violates schema"
            )
        result.append(dict(item))
    return result


def _planned_fingerprint(snapshot: Any, replacement_bytes: Mapping[str, bytes]) -> str:
    pairs: list[DatasetPair] = []
    for pair in snapshot.pairs:
        gt_bytes = replacement_bytes.get(pair.document_id, pair.gt_bytes)
        pairs.append(
            replace(
                pair,
                gt_bytes=gt_bytes,
                gt_sha256=sha256_bytes(gt_bytes),
                gt_size_bytes=len(gt_bytes),
                pair_sha256=compute_pair_sha256(pair.document_id, pair.input_bytes, gt_bytes),
            )
        )
    return compute_dataset_fingerprint(pairs)


def build_quarantine_repair_plan(
    dataset_root: str | Path,
    icd_artifact: str | Path,
    *,
    rules: Iterable[CandidateRepairRule] = PINNED_REPAIR_RULES,
) -> QuarantineRepairPlan:
    snapshot = scan_dataset_layout(dataset_root)
    root = snapshot.dataset_root
    descriptor_path = root / "reports" / "dataset_provenance.json"
    if descriptor_path.exists():
        raise QuarantineRepairError(
            "Quarantine repair must run before final dataset provenance publication"
        )
    manifest_rows, manifest_bytes = _validated_manifest_rows(root, snapshot)
    manifest_by_id = {str(row["document_id"]): row for row in manifest_rows}
    pair_by_id = {pair.document_id: pair for pair in snapshot.pairs}
    admitted_icd_ids, icd_sha = _load_icd_ids(Path(icd_artifact).resolve(strict=True))
    require_complete_pinned_plan = rules is PINNED_REPAIR_RULES
    rule_list = tuple(rules)
    if len({(rule.document_id, rule.entity_index) for rule in rule_list}) != len(rule_list):
        raise QuarantineRepairError("Duplicate quarantine repair rule")

    changes_by_document: dict[str, list[tuple[CandidateRepairRule, dict[str, Any]]]] = {}
    payloads: dict[str, list[dict[str, Any]]] = {}
    for rule in rule_list:
        row = manifest_by_id.get(rule.document_id)
        pair = pair_by_id.get(rule.document_id)
        if row is None or pair is None:
            if require_complete_pinned_plan:
                raise QuarantineRepairError("Pinned quarantine repair target is missing")
            continue
        if row.get("source_bucket") not in {"reconstructed", "quarantine"} or row.get(
            "train_eligible"
        ) is not False:
            raise QuarantineRepairError("A repair rule targets a non-quarantine document")
        payload = payloads.setdefault(
            rule.document_id,
            _strict_gt_payload(
                load_json_strict(pair.gt_bytes, source=f"GT {rule.document_id}"),
                rule.document_id,
            ),
        )
        if not 0 <= rule.entity_index < len(payload):
            raise QuarantineRepairError("Repair entity index is outside GT bounds")
        entity = payload[rule.entity_index]
        if entity.get("type") != "CHẨN_ĐOÁN":
            raise QuarantineRepairError("Repair rule does not target a diagnosis")
        actual = entity.get("candidates")
        if not isinstance(actual, list) or any(not isinstance(value, str) for value in actual):
            raise QuarantineRepairError("Repair target candidates are not a string array")
        _validate_replacement(rule.replacement_candidates, admitted_icd_ids)
        if tuple(actual) == rule.replacement_candidates:
            continue
        if tuple(actual) != rule.expected_candidates:
            raise QuarantineRepairError("Repair target no longer matches its pinned before-state")
        changes_by_document.setdefault(rule.document_id, []).append((rule, entity))

    known_keys = {(rule.document_id, rule.entity_index) for rule in rule_list}
    for pair in snapshot.pairs:
        numeric_id = int(pair.document_id)
        if numeric_id > 100:
            continue
        payload = payloads.setdefault(
            pair.document_id,
            _strict_gt_payload(
                load_json_strict(pair.gt_bytes, source=f"GT {pair.document_id}"),
                pair.document_id,
            ),
        )
        for entity_index, entity in enumerate(payload):
            if entity.get("type") != "CHẨN_ĐOÁN":
                continue
            candidates = entity.get("candidates")
            if candidates is None:
                continue
            malformed = not isinstance(candidates, list) or any(
                not isinstance(value, str) or not _ICD_ID_RE.fullmatch(value)
                for value in candidates
            )
            if (malformed or len(candidates) > 1) and (pair.document_id, entity_index) not in known_keys:
                raise QuarantineRepairError("Unplanned quarantine candidate conflict detected")

    planned_files: list[PlannedGTFile] = []
    replacement_bytes: dict[str, bytes] = {}
    evidence_changes: list[dict[str, Any]] = []
    for document_id in sorted(changes_by_document, key=int):
        pair = pair_by_id[document_id]
        payload = payloads[document_id]
        file_changes: list[dict[str, Any]] = []
        for rule, entity in sorted(changes_by_document[document_id], key=lambda item: item[0].entity_index):
            before = list(entity["candidates"])
            after = list(rule.replacement_candidates)
            entity["candidates"] = after
            change = {
                "document_id": document_id,
                "entity_index": rule.entity_index,
                "before_candidates": before,
                "after_candidates": after,
                "evidence_code": rule.evidence_code,
            }
            file_changes.append(change)
            evidence_changes.append(change)
        after_bytes = _serialize_gt(payload)
        replacement_bytes[document_id] = after_bytes
        planned_files.append(
            PlannedGTFile(
                document_id=document_id,
                path=pair.gt_path,
                before_bytes=pair.gt_bytes,
                after_bytes=after_bytes,
                before_sha256=pair.gt_sha256,
                after_sha256=sha256_bytes(after_bytes),
                changes=tuple(file_changes),
            )
        )

    fingerprint_after = _planned_fingerprint(snapshot, replacement_bytes)
    evidence = {
        "schema_id": REPAIR_SCHEMA_ID,
        "schema_version": REPAIR_SCHEMA_VERSION,
        "tool_version": REPAIR_TOOL_VERSION,
        "status_at_creation": "current",
        "dataset_fingerprint_before": snapshot.dataset_fingerprint,
        "dataset_fingerprint_after": fingerprint_after,
        "manifest_sha256": sha256_bytes(manifest_bytes),
        "icd_artifact_sha256": icd_sha,
        "repair_count": len(evidence_changes),
        "file_count": len(planned_files),
        "quarantine_remains_train_eligible": False,
        "changes": evidence_changes,
    }
    return QuarantineRepairPlan(
        dataset_root=root,
        manifest_sha256=sha256_bytes(manifest_bytes),
        dataset_fingerprint_before=snapshot.dataset_fingerprint,
        dataset_fingerprint_after=fingerprint_after,
        icd_artifact_sha256=icd_sha,
        files=tuple(planned_files),
        evidence=evidence,
    )


def _atomic_replace(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise QuarantineRepairError("Temporary repair payload changed before publication")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _archive_original(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = sha256_bytes(payload)
    if path.exists():
        if path.is_symlink() or not path.is_file() or sha256_bytes(path.read_bytes()) != expected:
            raise QuarantineRepairError("Existing quarantine repair archive is inconsistent")
        return
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def apply_quarantine_repair_plan(plan: QuarantineRepairPlan) -> QuarantineRepairResult:
    if not plan.files:
        return QuarantineRepairResult(
            mode="write",
            migration_required=False,
            dataset_fingerprint_before=plan.dataset_fingerprint_before,
            dataset_fingerprint_after=plan.dataset_fingerprint_after,
            repaired_entity_count=0,
            repaired_file_count=0,
            evidence_path=None,
        )
    root = plan.dataset_root
    lock_path = validate_artifact_path(
        root, root / "reports/.quarantine_gt_repair.lock", allow_missing_leaf=True
    )
    evidence_path = validate_artifact_path(
        root, root / "reports/quarantine_gt_repair.json", allow_missing_leaf=True
    )
    if evidence_path.exists():
        raise QuarantineRepairError("Quarantine repair evidence already exists")
    lock_fd: int | None = None
    replaced_files: list[PlannedGTFile] = []
    try:
        try:
            lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise QuarantineRepairError("Another quarantine repair is already active") from exc
        os.write(lock_fd, b"locked\n")
        os.fsync(lock_fd)

        current = scan_dataset_layout(root)
        manifest_path = root / "reports" / "dataset_manifest.jsonl"
        if (
            current.dataset_fingerprint != plan.dataset_fingerprint_before
            or sha256_bytes(manifest_path.read_bytes()) != plan.manifest_sha256
        ):
            raise QuarantineRepairError("Dataset changed since quarantine repair planning")
        for item in plan.files:
            if item.path.read_bytes() != item.before_bytes:
                raise QuarantineRepairError("GT file changed since quarantine repair planning")
            archive_relative = (
                f"reports/archive/quarantine_gt_repair/{plan.dataset_fingerprint_before}/"
                f"{item.document_id}.{item.before_sha256}.json"
            )
            archive_path = validate_artifact_path(
                root, root / archive_relative, allow_missing_leaf=True
            )
            archive_path.parent.mkdir(parents=True, exist_ok=True)
            archive_path = validate_artifact_path(
                root, archive_path, allow_missing_leaf=True
            )
            _archive_original(archive_path, item.before_bytes)

        # Recheck immediately before the first replacement. The dataset-scoped
        # lock serializes cooperating writers; the byte comparison rejects stale
        # plans from non-cooperating writers before publication.
        if any(item.path.read_bytes() != item.before_bytes for item in plan.files):
            raise QuarantineRepairError("GT file changed before quarantine repair publication")
        try:
            for item in plan.files:
                _atomic_replace(item.path, item.after_bytes)
                replaced_files.append(item)
            evidence_bytes = _serialize_gt(plan.evidence)
            _atomic_replace(evidence_path, evidence_bytes)
        except BaseException:
            for item in reversed(replaced_files):
                _atomic_replace(item.path, item.before_bytes)
            if evidence_path.exists():
                evidence_path.unlink()
            raise

        verified = scan_dataset_layout(root)
        if verified.dataset_fingerprint != plan.dataset_fingerprint_after:
            for item in reversed(replaced_files):
                _atomic_replace(item.path, item.before_bytes)
            if evidence_path.exists():
                evidence_path.unlink()
            raise QuarantineRepairError("Post-repair dataset fingerprint mismatch")
        return QuarantineRepairResult(
            mode="write",
            migration_required=True,
            dataset_fingerprint_before=plan.dataset_fingerprint_before,
            dataset_fingerprint_after=plan.dataset_fingerprint_after,
            repaired_entity_count=plan.repair_count,
            repaired_file_count=len(plan.files),
            evidence_path=evidence_path.relative_to(root).as_posix(),
        )
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if lock_path.exists():
            lock_path.unlink()


def repair_quarantine_gt(
    dataset_root: str | Path,
    icd_artifact: str | Path,
    *,
    write: bool = False,
    rules: Iterable[CandidateRepairRule] = PINNED_REPAIR_RULES,
) -> QuarantineRepairResult:
    plan = build_quarantine_repair_plan(dataset_root, icd_artifact, rules=rules)
    if write:
        return apply_quarantine_repair_plan(plan)
    return QuarantineRepairResult(
        mode="check",
        migration_required=bool(plan.files),
        dataset_fingerprint_before=plan.dataset_fingerprint_before,
        dataset_fingerprint_after=plan.dataset_fingerprint_after,
        repaired_entity_count=plan.repair_count,
        repaired_file_count=len(plan.files),
        evidence_path=None,
    )
