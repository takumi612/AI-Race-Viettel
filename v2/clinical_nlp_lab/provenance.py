from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import validate_submission_payload


PAIR_ALGORITHM = "clinical-nlp-dataset-pair/v1"
DATASET_ALGORITHM = "clinical-nlp-dataset/v1"
PAIR_DOMAIN = b"clinical-nlp-dataset-pair/v1\0"
DATASET_DOMAIN = b"clinical-nlp-dataset/v1\0"
MANIFEST_ROW_SCHEMA_ID = "clinical_nlp.dataset_manifest_row"
MANIFEST_ROW_SCHEMA_VERSION = 2
PROVENANCE_SCHEMA_ID = "clinical_nlp.dataset_provenance"
PROVENANCE_SCHEMA_VERSION = 1
REPORT_STATUS_SCHEMA_ID = "clinical_nlp.report_status"
REPORT_STATUS_SCHEMA_VERSION = 1
LEGACY_SHA256_SEMANTICS = "utf8-decoded-universal-newline-text-sha256"
DOCUMENT_ID_ORDER = "canonical-positive-decimal-numeric-ascending"
TOOL_VERSION = "1.0.0"

_DOCUMENT_ID_RE = re.compile(r"[1-9][0-9]*\Z")
_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_SOURCE_BUCKETS = frozenset({"reconstructed", "organizer_gt", "synthetic"})


class ProvenanceError(ValueError):
    """Raised when dataset provenance is ambiguous or invalid."""


@dataclass(frozen=True, slots=True)
class DatasetPair:
    document_id: str
    input_path: Path
    gt_path: Path
    input_bytes: bytes
    gt_bytes: bytes
    input_sha256: str
    input_size_bytes: int
    gt_sha256: str
    gt_size_bytes: int
    pair_sha256: str


@dataclass(frozen=True, slots=True)
class DatasetSnapshot:
    dataset_root: Path
    pairs: tuple[DatasetPair, ...]
    dataset_fingerprint: str

    @property
    def document_count(self) -> int:
        return len(self.pairs)

    @property
    def document_ids(self) -> tuple[str, ...]:
        return tuple(pair.document_id for pair in self.pairs)


@dataclass(frozen=True, slots=True)
class ProvenanceVerification:
    dataset_root: Path
    snapshot: DatasetSnapshot
    manifest_path: Path
    manifest_bytes: bytes
    manifest_sha256: str
    rows: tuple[dict[str, Any], ...]
    descriptor_path: Path
    descriptor_bytes: bytes
    descriptor: dict[str, Any]

    @property
    def dataset_fingerprint(self) -> str:
        return self.snapshot.dataset_fingerprint

    @property
    def document_count(self) -> int:
        return self.snapshot.document_count


@dataclass(frozen=True, slots=True)
class ReportStatus:
    effective_status: str
    reason_codes: tuple[str, ...]
    details: tuple[dict[str, Any], ...]


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _u32be(value: int) -> bytes:
    if not 0 <= value < 2**32:
        raise ProvenanceError(f"Value cannot be framed as u32: {value}")
    return value.to_bytes(4, "big")


def _u64be(value: int) -> bytes:
    if not 0 <= value < 2**64:
        raise ProvenanceError(f"Value cannot be framed as u64: {value}")
    return value.to_bytes(8, "big")


def _validate_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not _DOCUMENT_ID_RE.fullmatch(document_id):
        raise ProvenanceError(
            f"Document ID must be canonical positive decimal without leading zero: {document_id!r}"
        )
    return document_id


def _framed_raw_triple(document_id: str, input_bytes: bytes, gt_bytes: bytes) -> bytes:
    encoded_id = _validate_document_id(document_id).encode("utf-8")
    return b"".join(
        (
            _u32be(len(encoded_id)),
            encoded_id,
            _u64be(len(input_bytes)),
            input_bytes,
            _u64be(len(gt_bytes)),
            gt_bytes,
        )
    )


def compute_pair_sha256(document_id: str, input_bytes: bytes, gt_bytes: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(PAIR_DOMAIN)
    digest.update(_framed_raw_triple(document_id, input_bytes, gt_bytes))
    return digest.hexdigest()


def compute_dataset_fingerprint(pairs: Iterable[DatasetPair]) -> str:
    pair_list = sorted(pairs, key=lambda pair: int(_validate_document_id(pair.document_id)))
    document_ids = [pair.document_id for pair in pair_list]
    if len(document_ids) != len(set(document_ids)):
        raise ProvenanceError("Duplicate document IDs cannot be hashed as a dataset")
    digest = hashlib.sha256()
    digest.update(DATASET_DOMAIN)
    for pair in pair_list:
        digest.update(_framed_raw_triple(pair.document_id, pair.input_bytes, pair.gt_bytes))
    return digest.hexdigest()


def _decode_utf8(raw: bytes, source: str) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProvenanceError(f"{source} is not strict UTF-8: {exc}") from exc


def _universal_newline_text(raw: bytes, source: str) -> str:
    return _decode_utf8(raw, source).replace("\r\n", "\n").replace("\r", "\n")


def compute_legacy_input_text_sha256(input_bytes: bytes) -> str:
    normalized = _universal_newline_text(input_bytes, "input bytes")
    return sha256_bytes(normalized.encode("utf-8"))


def _reject_constant(value: str) -> Any:
    raise ProvenanceError(f"JSON contains non-finite numeric constant: {value}")


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProvenanceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_json_strict(raw: bytes | str, *, source: str = "JSON") -> Any:
    text = _decode_utf8(raw, source) if isinstance(raw, bytes) else raw
    if text.startswith("\ufeff"):
        raise ProvenanceError(f"{source} must not contain a UTF-8 BOM")
    try:
        return json.loads(
            text,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except ProvenanceError:
        raise
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProvenanceError(f"Malformed JSON in {source}: {exc}") from exc


def load_jsonl_strict(raw: bytes | str, *, source: str = "JSONL") -> tuple[dict[str, Any], ...]:
    text = _decode_utf8(raw, source) if isinstance(raw, bytes) else raw
    if text.startswith("\ufeff"):
        raise ProvenanceError(f"{source} must not contain a UTF-8 BOM")
    lines = text.splitlines()
    if not lines:
        raise ProvenanceError(f"{source} is empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            raise ProvenanceError(f"Blank JSONL row in {source} at line {line_number}")
        item = load_json_strict(line, source=f"{source}:{line_number}")
        if not isinstance(item, dict):
            raise ProvenanceError(f"JSONL row in {source} at line {line_number} is not an object")
        rows.append(item)
    return tuple(rows)


def canonical_json_bytes(payload: Any) -> bytes:
    try:
        text = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ProvenanceError(f"Payload is not finite canonical JSON: {exc}") from exc
    return text.encode("utf-8") + b"\n"


def canonical_jsonl_bytes(rows: Iterable[Mapping[str, Any]]) -> bytes:
    row_list = list(rows)
    if not row_list:
        raise ProvenanceError("Canonical JSONL must contain at least one row")
    return b"".join(canonical_json_bytes(dict(row)) for row in row_list)


def _scan_payload_directory(directory: Path, expected_suffix: str) -> dict[str, Path]:
    if directory.is_symlink():
        raise ProvenanceError(f"Dataset payload directory must not be a symlink: {directory}")
    if not directory.is_dir():
        raise ProvenanceError(f"Missing dataset payload directory: {directory}")
    paths: dict[str, Path] = {}
    casefold_names: dict[str, str] = {}
    for path in directory.iterdir():
        if path.is_symlink():
            raise ProvenanceError(f"Dataset payload must not be a symlink: {path}")
        if path.is_dir():
            raise ProvenanceError(f"Unexpected nested payload path: {path}")
        if not path.is_file():
            raise ProvenanceError(f"Unexpected payload entry: {path}")
        folded = path.name.casefold()
        prior_name = casefold_names.setdefault(folded, path.name)
        if prior_name != path.name:
            raise ProvenanceError(f"Case-colliding payload names: {prior_name!r}, {path.name!r}")
        if path.suffix != expected_suffix:
            raise ProvenanceError(f"unexpected payload file: {path.name}")
        document_id = path.stem
        _validate_document_id(document_id)
        folded_id = document_id.casefold()
        if folded_id in paths:
            raise ProvenanceError(f"Duplicate or case-colliding document stem: {document_id!r}")
        paths[folded_id] = path
    return paths


def _validate_gt(document_id: str, input_bytes: bytes, gt_bytes: bytes, gt_path: Path) -> None:
    raw_text = _universal_newline_text(input_bytes, str(gt_path.parent.parent / "input" / f"{document_id}.txt"))
    payload = load_json_strict(gt_bytes, source=str(gt_path))
    if not isinstance(payload, list):
        raise ProvenanceError(f"GT JSON must be a list: {gt_path}")
    errors = validate_submission_payload(payload, raw_text)
    if errors:
        raise ProvenanceError(f"Invalid GT schema/offset for document {document_id}: {errors[0]}")


def scan_dataset_layout(dataset_root: str | Path) -> DatasetSnapshot:
    requested_root = Path(dataset_root)
    if requested_root.is_symlink():
        raise ProvenanceError(f"Dataset root must not be a symlink: {requested_root}")
    if not requested_root.is_dir():
        raise ProvenanceError(f"Dataset root does not exist: {requested_root}")
    root = requested_root.resolve()
    input_paths = _scan_payload_directory(root / "input", ".txt")
    gt_paths = _scan_payload_directory(root / "gt", ".json")
    input_ids = set(input_paths)
    gt_ids = set(gt_paths)
    if input_ids != gt_ids:
        missing_gt = sorted(input_ids - gt_ids, key=int)
        missing_input = sorted(gt_ids - input_ids, key=int)
        raise ProvenanceError(
            "Input/GT pairing mismatch: "
            f"missing_gt={missing_gt}, missing_input={missing_input}"
        )
    if not input_ids:
        raise ProvenanceError("Dataset contains no input/GT pairs")

    pairs: list[DatasetPair] = []
    for folded_id in sorted(input_ids, key=int):
        input_path = input_paths[folded_id]
        gt_path = gt_paths[folded_id]
        document_id = input_path.stem
        if gt_path.stem != document_id:
            raise ProvenanceError(
                f"Case-colliding input/GT stems: {input_path.stem!r}, {gt_path.stem!r}"
            )
        input_bytes = input_path.read_bytes()
        gt_bytes = gt_path.read_bytes()
        _validate_gt(document_id, input_bytes, gt_bytes, gt_path)
        pairs.append(
            DatasetPair(
                document_id=document_id,
                input_path=input_path,
                gt_path=gt_path,
                input_bytes=input_bytes,
                gt_bytes=gt_bytes,
                input_sha256=sha256_bytes(input_bytes),
                input_size_bytes=len(input_bytes),
                gt_sha256=sha256_bytes(gt_bytes),
                gt_size_bytes=len(gt_bytes),
                pair_sha256=compute_pair_sha256(document_id, input_bytes, gt_bytes),
            )
        )
    pair_tuple = tuple(pairs)
    return DatasetSnapshot(root, pair_tuple, compute_dataset_fingerprint(pair_tuple))


def _require_manifest_identity(
    rows: Sequence[Mapping[str, Any]], snapshot: DatasetSnapshot
) -> dict[str, Mapping[str, Any]]:
    by_id: dict[str, Mapping[str, Any]] = {}
    ordered_ids: list[str] = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, Mapping):
            raise ProvenanceError(f"Manifest row {index} is not an object")
        document_id = row.get("document_id")
        if not isinstance(document_id, str):
            raise ProvenanceError(f"Manifest row {index} has invalid document_id")
        _validate_document_id(document_id)
        if document_id in by_id:
            raise ProvenanceError(f"Duplicate manifest document_id: {document_id}")
        by_id[document_id] = row
        ordered_ids.append(document_id)
    expected_ids = list(snapshot.document_ids)
    if ordered_ids != expected_ids:
        raise ProvenanceError(
            f"Manifest IDs/order differ from dataset: expected={expected_ids}, actual={ordered_ids}"
        )
    return by_id


def _require_eligibility_and_source(row: Mapping[str, Any], document_id: str) -> None:
    if type(row.get("train_eligible")) is not bool:
        raise ProvenanceError(f"Manifest row {document_id} requires explicit boolean train_eligible")
    source_bucket = row.get("source_bucket")
    if not isinstance(source_bucket, str) or source_bucket not in _SOURCE_BUCKETS:
        raise ProvenanceError(
            f"Manifest row {document_id} has invalid source_bucket: {source_bucket!r}"
        )


def _require_sha256(value: Any, field: str, document_id: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ProvenanceError(f"Manifest row {document_id} has invalid {field}")
    return value


def validate_legacy_manifest(
    rows: Sequence[Mapping[str, Any]], snapshot: DatasetSnapshot
) -> tuple[dict[str, Any], ...]:
    by_id = _require_manifest_identity(rows, snapshot)
    validated: list[dict[str, Any]] = []
    for pair in snapshot.pairs:
        row = by_id[pair.document_id]
        _require_eligibility_and_source(row, pair.document_id)
        legacy_sha = _require_sha256(row.get("sha256"), "legacy sha256", pair.document_id)
        expected = compute_legacy_input_text_sha256(pair.input_bytes)
        if legacy_sha != expected:
            raise ProvenanceError(
                f"Manifest row {pair.document_id} legacy sha256 mismatch: "
                f"expected={expected}, actual={legacy_sha}"
            )
        validated.append(dict(row))
    return tuple(validated)


def build_v2_manifest_rows(
    legacy_rows: Sequence[Mapping[str, Any]], snapshot: DatasetSnapshot
) -> tuple[dict[str, Any], ...]:
    validated = validate_legacy_manifest(legacy_rows, snapshot)
    provenance_fields = {
        "schema_id",
        "schema_version",
        "input_sha256",
        "input_size_bytes",
        "gt_sha256",
        "gt_size_bytes",
        "pair_sha256",
        "legacy_sha256_semantics",
    }
    result: list[dict[str, Any]] = []
    for row, pair in zip(validated, snapshot.pairs):
        unexpected = provenance_fields.intersection(row)
        if unexpected:
            raise ProvenanceError(
                f"Legacy manifest row {pair.document_id} already contains provenance fields: "
                f"{sorted(unexpected)}"
            )
        upgraded = dict(row)
        upgraded.update(
            {
                "schema_id": MANIFEST_ROW_SCHEMA_ID,
                "schema_version": MANIFEST_ROW_SCHEMA_VERSION,
                "input_sha256": pair.input_sha256,
                "input_size_bytes": pair.input_size_bytes,
                "gt_sha256": pair.gt_sha256,
                "gt_size_bytes": pair.gt_size_bytes,
                "pair_sha256": pair.pair_sha256,
                "legacy_sha256_semantics": LEGACY_SHA256_SEMANTICS,
            }
        )
        result.append(upgraded)
    return tuple(result)


def validate_v2_manifest(
    rows: Sequence[Mapping[str, Any]], snapshot: DatasetSnapshot
) -> tuple[dict[str, Any], ...]:
    by_id = _require_manifest_identity(rows, snapshot)
    result: list[dict[str, Any]] = []
    for pair in snapshot.pairs:
        row = by_id[pair.document_id]
        _require_eligibility_and_source(row, pair.document_id)
        expected_values: dict[str, Any] = {
            "schema_id": MANIFEST_ROW_SCHEMA_ID,
            "schema_version": MANIFEST_ROW_SCHEMA_VERSION,
            "input_sha256": pair.input_sha256,
            "input_size_bytes": pair.input_size_bytes,
            "gt_sha256": pair.gt_sha256,
            "gt_size_bytes": pair.gt_size_bytes,
            "pair_sha256": pair.pair_sha256,
            "legacy_sha256_semantics": LEGACY_SHA256_SEMANTICS,
        }
        for field, expected in expected_values.items():
            actual = row.get(field)
            if actual != expected or (
                field.endswith("_size_bytes") and type(actual) is not int
            ):
                raise ProvenanceError(
                    f"Manifest row {pair.document_id} {field} mismatch: "
                    f"expected={expected!r}, actual={actual!r}"
                )
        _require_sha256(row.get("sha256"), "legacy sha256", pair.document_id)
        expected_legacy = compute_legacy_input_text_sha256(pair.input_bytes)
        if row["sha256"] != expected_legacy:
            raise ProvenanceError(
                f"Manifest row {pair.document_id} legacy sha256 mismatch: "
                f"expected={expected_legacy}, actual={row['sha256']}"
            )
        for field in ("input_sha256", "gt_sha256", "pair_sha256"):
            _require_sha256(row.get(field), field, pair.document_id)
        result.append(dict(row))
    return tuple(result)


def build_provenance_descriptor(
    snapshot: DatasetSnapshot,
    manifest_bytes: bytes,
    *,
    legacy_manifest_sha256: str,
    legacy_archive_path: str,
    created_at: str,
    git_commit: str | None,
    manifest_path: str = "reports/dataset_manifest.jsonl",
) -> dict[str, Any]:
    _require_sha256(legacy_manifest_sha256, "legacy manifest sha256", "descriptor")
    if not created_at:
        raise ProvenanceError("Descriptor created_at must be non-empty")
    return {
        "schema_id": PROVENANCE_SCHEMA_ID,
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "manifest": {
            "path": manifest_path,
            "schema_id": MANIFEST_ROW_SCHEMA_ID,
            "schema_version": MANIFEST_ROW_SCHEMA_VERSION,
            "sha256": sha256_bytes(manifest_bytes),
            "size_bytes": len(manifest_bytes),
        },
        "dataset": {
            "pair_algorithm": PAIR_ALGORITHM,
            "fingerprint_algorithm": DATASET_ALGORITHM,
            "fingerprint": snapshot.dataset_fingerprint,
            "document_count": snapshot.document_count,
            "document_id_min": snapshot.document_ids[0],
            "document_id_max": snapshot.document_ids[-1],
            "document_id_order": DOCUMENT_ID_ORDER,
        },
        "legacy_manifest": {
            "sha256": legacy_manifest_sha256,
            "archive_path": legacy_archive_path,
        },
        "producer": {
            "tool": "upgrade_dataset_provenance.py",
            "version": TOOL_VERSION,
            "git_commit": git_commit,
        },
        "created_at": created_at,
        "identity_excludes": ["created_at"],
    }


def validate_provenance_descriptor(
    descriptor: Mapping[str, Any],
    snapshot: DatasetSnapshot,
    manifest_bytes: bytes,
    *,
    manifest_path: str = "reports/dataset_manifest.jsonl",
) -> dict[str, Any]:
    """Validate the detached descriptor against already-validated manifest bytes."""
    expected_top_level = {
        "schema_id",
        "schema_version",
        "manifest",
        "dataset",
        "legacy_manifest",
        "producer",
        "created_at",
        "identity_excludes",
    }
    if set(descriptor) != expected_top_level:
        raise ProvenanceError(
            "Dataset provenance descriptor fields mismatch: "
            f"missing={sorted(expected_top_level - set(descriptor))}, "
            f"extra={sorted(set(descriptor) - expected_top_level)}"
        )
    if descriptor.get("schema_id") != PROVENANCE_SCHEMA_ID or descriptor.get(
        "schema_version"
    ) != PROVENANCE_SCHEMA_VERSION:
        raise ProvenanceError("Dataset provenance descriptor schema mismatch")
    if not isinstance(descriptor.get("created_at"), str) or not descriptor["created_at"]:
        raise ProvenanceError("Dataset provenance descriptor created_at is invalid")
    if descriptor.get("identity_excludes") != ["created_at"]:
        raise ProvenanceError("Dataset provenance descriptor identity_excludes is invalid")

    manifest_info = descriptor.get("manifest")
    dataset_info = descriptor.get("dataset")
    legacy_info = descriptor.get("legacy_manifest")
    producer = descriptor.get("producer")
    if not all(isinstance(item, Mapping) for item in (manifest_info, dataset_info, legacy_info, producer)):
        raise ProvenanceError("Dataset provenance descriptor sections are invalid")
    assert isinstance(manifest_info, Mapping)
    assert isinstance(dataset_info, Mapping)
    assert isinstance(legacy_info, Mapping)
    assert isinstance(producer, Mapping)
    expected_manifest = {
        "path": manifest_path,
        "schema_id": MANIFEST_ROW_SCHEMA_ID,
        "schema_version": MANIFEST_ROW_SCHEMA_VERSION,
        "sha256": sha256_bytes(manifest_bytes),
        "size_bytes": len(manifest_bytes),
    }
    if dict(manifest_info) != expected_manifest:
        raise ProvenanceError(
            f"Descriptor manifest binding mismatch: expected={expected_manifest!r}, "
            f"actual={dict(manifest_info)!r}"
        )
    expected_dataset = {
        "pair_algorithm": PAIR_ALGORITHM,
        "fingerprint_algorithm": DATASET_ALGORITHM,
        "fingerprint": snapshot.dataset_fingerprint,
        "document_count": snapshot.document_count,
        "document_id_min": snapshot.document_ids[0],
        "document_id_max": snapshot.document_ids[-1],
        "document_id_order": DOCUMENT_ID_ORDER,
    }
    if dict(dataset_info) != expected_dataset:
        raise ProvenanceError(
            f"Descriptor dataset binding mismatch: expected={expected_dataset!r}, "
            f"actual={dict(dataset_info)!r}"
        )
    if set(legacy_info) != {"sha256", "archive_path"}:
        raise ProvenanceError("Descriptor legacy_manifest fields are invalid")
    _require_sha256(legacy_info.get("sha256"), "legacy manifest sha256", "descriptor")
    archive_relative = legacy_info.get("archive_path")
    expected_archive_relative = (
        "reports/archive/dataset_manifest.legacy."
        f"{legacy_info.get('sha256')}.jsonl"
    )
    if (
        not isinstance(archive_relative, str)
        or archive_relative != expected_archive_relative
        or "\\" in archive_relative
        or Path(archive_relative).is_absolute()
        or ".." in Path(archive_relative).parts
    ):
        raise ProvenanceError("Descriptor legacy_manifest.archive_path is invalid")
    if set(producer) != {"tool", "version", "git_commit"}:
        raise ProvenanceError("Descriptor producer fields are invalid")
    if producer.get("tool") != "upgrade_dataset_provenance.py" or producer.get(
        "version"
    ) != TOOL_VERSION:
        raise ProvenanceError("Descriptor producer identity is invalid")
    if producer.get("git_commit") is not None and not isinstance(producer.get("git_commit"), str):
        raise ProvenanceError("Descriptor producer.git_commit is invalid")
    return dict(descriptor)


def _resolve_relative_artifact(root: Path, relative: Any, field: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise ProvenanceError(f"Descriptor {field} must be a non-empty POSIX relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ProvenanceError(f"Descriptor {field} escapes the dataset root: {relative!r}")
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ProvenanceError(f"Descriptor {field} escapes the dataset root: {relative!r}") from exc
    return resolved


def verify_dataset_provenance(
    dataset_root: str | Path,
    *,
    manifest_path: str | Path | None = None,
    descriptor_path: str | Path | None = None,
) -> ProvenanceVerification:
    snapshot = scan_dataset_layout(dataset_root)
    root = snapshot.dataset_root
    manifest = Path(manifest_path) if manifest_path is not None else root / "reports" / "dataset_manifest.jsonl"
    descriptor_file = (
        Path(descriptor_path)
        if descriptor_path is not None
        else root / "reports" / "dataset_provenance.json"
    )
    for path, label in ((manifest, "manifest"), (descriptor_file, "descriptor")):
        if path.is_symlink():
            raise ProvenanceError(f"Active {label} must not be a symlink: {path}")
        if not path.is_file():
            raise ProvenanceError(f"Missing active {label}: {path}")

    manifest_bytes = manifest.read_bytes()
    rows = validate_v2_manifest(
        load_jsonl_strict(manifest_bytes, source=str(manifest)), snapshot
    )
    if canonical_jsonl_bytes(rows) != manifest_bytes:
        raise ProvenanceError("Active v2 manifest is not canonical UTF-8/LF JSONL")
    descriptor_bytes = descriptor_file.read_bytes()
    descriptor_payload = load_json_strict(descriptor_bytes, source=str(descriptor_file))
    if not isinstance(descriptor_payload, dict):
        raise ProvenanceError("Dataset provenance descriptor must be an object")
    descriptor: dict[str, Any] = descriptor_payload
    if canonical_json_bytes(descriptor) != descriptor_bytes:
        raise ProvenanceError("Dataset provenance descriptor is not canonical JSON")
    try:
        manifest_relative = manifest.resolve().relative_to(root).as_posix()
    except ValueError as exc:
        raise ProvenanceError("Active manifest must be inside the dataset root") from exc
    descriptor = validate_provenance_descriptor(
        descriptor, snapshot, manifest_bytes, manifest_path=manifest_relative
    )
    legacy_info = descriptor["legacy_manifest"]
    archive_sha = _require_sha256(
        legacy_info.get("sha256"), "legacy manifest sha256", "descriptor"
    )
    archive_path = _resolve_relative_artifact(
        root, legacy_info.get("archive_path"), "legacy_manifest.archive_path"
    )
    if archive_path.is_symlink() or not archive_path.is_file():
        raise ProvenanceError(f"Missing legacy manifest archive: {archive_path}")
    actual_archive_sha = sha256_bytes(archive_path.read_bytes())
    if actual_archive_sha != archive_sha:
        raise ProvenanceError(
            f"Legacy manifest archive hash mismatch: expected={archive_sha}, actual={actual_archive_sha}"
        )
    return ProvenanceVerification(
        dataset_root=root,
        snapshot=snapshot,
        manifest_path=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=sha256_bytes(manifest_bytes),
        rows=rows,
        descriptor_path=descriptor_file,
        descriptor_bytes=descriptor_bytes,
        descriptor=descriptor,
    )


def evaluate_report_status(
    envelope: Mapping[str, Any], expected_fingerprints: Mapping[str, str]
) -> ReportStatus:
    stored_status = envelope.get("status_at_creation")
    if stored_status == "archived":
        return ReportStatus("archived", ("archived",), ())
    details: list[dict[str, Any]] = []
    fingerprints = envelope.get("fingerprints")
    if not isinstance(fingerprints, Mapping):
        fingerprints = {}
    for name, expected in expected_fingerprints.items():
        if name not in fingerprints:
            details.append({"reason": "missing_fingerprint", "fingerprint": name})
        elif fingerprints[name] != expected:
            details.append(
                {
                    "reason": "fingerprint_mismatch",
                    "fingerprint": name,
                    "expected": expected,
                    "actual": fingerprints[name],
                }
            )
    if stored_status not in {"current", "stale", "archived"}:
        details.append({"reason": "invalid_status_at_creation", "actual": stored_status})
    elif stored_status == "stale":
        details.append({"reason": "stored_stale"})
    if details:
        reason_codes = tuple(dict.fromkeys(detail["reason"] for detail in details))
        return ReportStatus("stale", reason_codes, tuple(details))
    return ReportStatus("current", (), ())


def _normalized_scope(scope: Any) -> Any:
    if isinstance(scope, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(scope):
            value = scope[key]
            if key == "document_ids" and isinstance(value, list):
                try:
                    value = sorted((_validate_document_id(str(item)) for item in value), key=int)
                except ProvenanceError:
                    value = sorted(str(item) for item in value)
            result[str(key)] = _normalized_scope(value)
        return result
    if isinstance(scope, list):
        return [_normalized_scope(item) for item in scope]
    return scope


def detect_report_conflicts(
    envelopes: Iterable[Mapping[str, Any]], required_fingerprints: Iterable[str]
) -> tuple[dict[str, Any], ...]:
    required = tuple(sorted(set(required_fingerprints)))
    groups: dict[tuple[Any, ...], set[str]] = {}
    scopes: dict[tuple[Any, ...], Any] = {}
    for envelope in envelopes:
        if envelope.get("status_at_creation") != "current":
            continue
        report_type = envelope.get("report_type")
        if not isinstance(report_type, str) or not report_type:
            continue
        fingerprints = envelope.get("fingerprints")
        if not isinstance(fingerprints, Mapping) or any(
            name not in fingerprints for name in required
        ):
            continue
        normalized_scope = _normalized_scope(envelope.get("scope", {}))
        scope_bytes = canonical_json_bytes(normalized_scope)
        fingerprint_key = tuple((name, str(fingerprints[name])) for name in required)
        group_key = (report_type, scope_bytes, fingerprint_key)
        payload_sha = envelope.get("payload_sha256")
        if not isinstance(payload_sha, str):
            if "payload" not in envelope:
                continue
            payload_sha = sha256_bytes(canonical_json_bytes(envelope["payload"]))
        groups.setdefault(group_key, set()).add(payload_sha)
        scopes[group_key] = normalized_scope
    conflicts: list[dict[str, Any]] = []
    for (report_type, _scope_bytes, fingerprint_key), payload_hashes in groups.items():
        if len(payload_hashes) > 1:
            conflicts.append(
                {
                    "report_type": report_type,
                    "scope": scopes[(report_type, _scope_bytes, fingerprint_key)],
                    "fingerprints": dict(fingerprint_key),
                    "payload_sha256": sorted(payload_hashes),
                }
            )
    conflicts.sort(key=lambda item: canonical_json_bytes(item))
    return tuple(conflicts)


def detect_report_fact_conflicts(
    envelopes: Iterable[Mapping[str, Any]], required_fingerprints: Iterable[str]
) -> tuple[dict[str, Any], ...]:
    """Compare only shared canonical facts across current report types."""
    required = tuple(sorted(set(required_fingerprints)))
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for envelope in envelopes:
        if envelope.get("status_at_creation") != "current":
            continue
        fingerprints = envelope.get("fingerprints")
        facts = envelope.get("facts")
        if (
            not isinstance(fingerprints, Mapping)
            or not isinstance(facts, Mapping)
            or any(name not in fingerprints for name in required)
        ):
            continue
        scope_bytes = canonical_json_bytes(_normalized_scope(envelope.get("scope", {})))
        fingerprint_key = tuple((name, str(fingerprints[name])) for name in required)
        groups.setdefault((scope_bytes, fingerprint_key), []).append(envelope)

    conflicts: list[dict[str, Any]] = []
    for (_scope_bytes, fingerprint_key), reports in groups.items():
        values_by_fact: dict[str, dict[bytes, list[str]]] = {}
        for report in reports:
            report_type = str(report.get("report_type", ""))
            for fact_name, value in report["facts"].items():
                encoded = canonical_json_bytes(value)
                values_by_fact.setdefault(str(fact_name), {}).setdefault(encoded, []).append(
                    report_type
                )
        for fact_name, values in values_by_fact.items():
            if len(values) <= 1:
                continue
            conflicts.append(
                {
                    "fact": fact_name,
                    "fingerprints": dict(fingerprint_key),
                    "values": [
                        {
                            "value": load_json_strict(encoded, source="canonical fact"),
                            "report_types": sorted(report_types),
                        }
                        for encoded, report_types in sorted(values.items())
                    ],
                }
            )
    conflicts.sort(key=lambda item: canonical_json_bytes(item))
    return tuple(conflicts)


def build_legacy_report_status_index(
    report_payloads: Iterable[tuple[str, bytes]],
) -> tuple[dict[str, Any], ...]:
    explicitly_archived = {
        "agent_audit_first_100_final.json",
        "agent_audit_first_100_final.md",
        "agent_audit_first_100_repair_plan.json",
        "first100_repair_log.json",
        "agent_audit_generated_2000_final.json",
        "agent_audit_generated_2000_final.md",
    }
    rows: list[dict[str, Any]] = []
    for relative_path, payload in sorted(report_payloads):
        name = Path(relative_path).name
        status = "archived" if name in explicitly_archived else "stale"
        reason = "historical_artifact" if status == "archived" else "missing_fingerprint"
        rows.append(
            {
                "schema_id": REPORT_STATUS_SCHEMA_ID,
                "schema_version": REPORT_STATUS_SCHEMA_VERSION,
                "relative_path": relative_path,
                "payload_sha256": sha256_bytes(payload),
                "effective_status": status,
                "reason": reason,
            }
        )
    return tuple(rows)
