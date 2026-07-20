"""Strict readers for paired Viettel input and ground-truth sources."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from src.training.contracts import CanonicalRecord
from src.training.fingerprints import sha256_file


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    root: Path
    trust_tier: str
    expected_ids: tuple[str, ...]
    manifest_path: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("source name must be a non-empty string")
        if not isinstance(self.trust_tier, str) or not self.trust_tier.strip():
            raise ValueError("trust_tier must be a non-empty string")
        if not self.expected_ids:
            raise ValueError("expected_ids must not be empty")
        normalized_ids = tuple(str(record_id).strip() for record_id in self.expected_ids)
        if any(not record_id for record_id in normalized_ids):
            raise ValueError("expected_ids must contain non-empty IDs")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("expected_ids contains duplicates")

        object.__setattr__(self, "root", Path(self.root))
        object.__setattr__(self, "expected_ids", normalized_ids)
        if self.manifest_path is not None:
            object.__setattr__(self, "manifest_path", Path(self.manifest_path))


def is_synthetic_source(spec: SourceSpec) -> bool:
    return (
        spec.name.casefold().startswith("synthetic")
        or spec.trust_tier.casefold().startswith("synthetic")
    )


def _load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    entries: dict[str, dict[str, Any]] = {}
    try:
        lines = path.read_bytes().decode("utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read source manifest: {path}") from exc

    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid manifest JSON at line {line_number}: {path}"
            ) from exc
        if not isinstance(entry, dict):
            raise ValueError(f"manifest line {line_number} must be an object")
        record_id = entry.get("record_id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"manifest line {line_number} has invalid record_id")
        record_id = record_id.strip()
        if record_id in entries:
            raise ValueError(f"duplicate manifest record_id: {record_id}")
        entries[record_id] = entry
    return entries


def _read_text(path: Path) -> str:
    try:
        # Ground-truth offsets are produced from Python text-mode reads. Normalize
        # CRLF/CR to LF here as well so a Windows checkout is identical to Colab.
        with path.open("r", encoding="utf-8", newline=None) as stream:
            return stream.read()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read UTF-8 input: {path}") from exc


def _read_ground_truth(path: Path) -> list[Mapping[str, Any]]:
    try:
        value = json.loads(path.read_bytes().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid ground-truth JSON: {path}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"ground truth must be a list of entity objects: {path}")
    return value


def _normalize_trusted_entities(
    entity_mappings: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], dict[str, int]]:
    normalized: list[Mapping[str, Any]] = []
    blank_candidates_removed = 0
    duplicate_entities_removed = 0
    seen_entities: set[str] = set()
    for entity in entity_mappings:
        candidates = entity.get("candidates")
        normalized_entity = entity
        if isinstance(candidates, list):
            kept_candidates = [
                candidate
                for candidate in candidates
                if not (isinstance(candidate, str) and not candidate.strip())
            ]
            removed = len(candidates) - len(kept_candidates)
        else:
            kept_candidates = []
            removed = 0
        if removed:
            copied_entity = dict(entity)
            copied_entity["candidates"] = kept_candidates
            normalized_entity = copied_entity
            blank_candidates_removed += removed

        signature = json.dumps(
            normalized_entity,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if signature in seen_entities:
            duplicate_entities_removed += 1
        else:
            seen_entities.add(signature)
            normalized.append(normalized_entity)

    statistics: dict[str, int] = {}
    if blank_candidates_removed:
        statistics["blank_candidates_removed"] = blank_candidates_removed
    if duplicate_entities_removed:
        statistics["duplicate_entities_removed"] = duplicate_entities_removed
    return normalized, statistics


def _verify_manifest_hash(
    metadata: Mapping[str, Any],
    field: str,
    path: Path,
) -> None:
    expected = metadata.get(field)
    if expected is None:
        return
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"invalid {field} in manifest for {path.stem}")
    actual = sha256_file(path)
    if actual.casefold() != expected.casefold():
        raise ValueError(
            f"{field} mismatch for {path.stem}: expected {expected}, got {actual}"
        )


def load_source_records(spec: SourceSpec) -> tuple[CanonicalRecord, ...]:
    """Load only the explicitly expected paired files into canonical records."""

    root = spec.root.resolve()
    if root.name.casefold().endswith(".failed-validation"):
        raise ValueError(f"refusing failed-validation source directory: {root}")
    if not root.is_dir():
        raise ValueError(f"source root is not a directory: {root}")

    input_dir = root / "input"
    gt_dir = root / "gt"
    if not input_dir.is_dir():
        raise ValueError(f"missing input directory: {input_dir}")
    if not gt_dir.is_dir():
        raise ValueError(f"missing GT directory: {gt_dir}")

    expected = set(spec.expected_ids)
    input_ids = {path.stem for path in input_dir.glob("*.txt") if path.is_file()}
    gt_ids = {path.stem for path in gt_dir.glob("*.json") if path.is_file()}

    missing_input = sorted(expected - input_ids)
    if missing_input:
        raise ValueError(f"missing input for IDs: {', '.join(missing_input)}")
    missing_gt = sorted(expected - gt_ids)
    if missing_gt:
        raise ValueError(f"missing GT for IDs: {', '.join(missing_gt)}")

    if is_synthetic_source(spec):
        unexpected = sorted((input_ids | gt_ids) - expected)
        if unexpected:
            raise ValueError(f"unexpected source IDs: {', '.join(unexpected)}")
        unpaired = sorted(input_ids ^ gt_ids)
        if unpaired:
            raise ValueError(f"unpaired synthetic source IDs: {', '.join(unpaired)}")

    manifest_path = spec.manifest_path
    if manifest_path is None:
        default_manifest = root / "manifest.jsonl"
        manifest_path = default_manifest if default_manifest.is_file() else None
    elif not manifest_path.is_absolute():
        manifest_path = root / manifest_path

    manifest: dict[str, dict[str, Any]] = {}
    if manifest_path is not None:
        manifest_path = manifest_path.resolve()
        try:
            manifest_path.relative_to(root)
        except ValueError as exc:
            raise ValueError("source manifest must be inside source root") from exc
        if not manifest_path.is_file():
            raise ValueError(f"source manifest is not a file: {manifest_path}")
        manifest = _load_manifest(manifest_path)
        missing_manifest = sorted(expected - set(manifest))
        if missing_manifest:
            raise ValueError(
                f"missing manifest records: {', '.join(missing_manifest)}"
            )
        if is_synthetic_source(spec):
            unexpected_manifest = sorted(set(manifest) - expected)
            if unexpected_manifest:
                raise ValueError(
                    f"unexpected manifest IDs: {', '.join(unexpected_manifest)}"
                )

    records: list[CanonicalRecord] = []
    for record_id in sorted(spec.expected_ids):
        input_path = input_dir / f"{record_id}.txt"
        gt_path = gt_dir / f"{record_id}.json"
        metadata = dict(manifest.get(record_id, {}))
        _verify_manifest_hash(metadata, "input_sha256", input_path)
        _verify_manifest_hash(metadata, "ground_truth_sha256", gt_path)

        text = _read_text(input_path)
        entity_mappings = _read_ground_truth(gt_path)
        if not is_synthetic_source(spec):
            entity_mappings, normalization_statistics = _normalize_trusted_entities(
                entity_mappings
            )
            if normalization_statistics:
                metadata["source_normalizations"] = normalization_statistics
        profile_id = metadata.get("profile_id")
        split_group = (
            profile_id.strip()
            if isinstance(profile_id, str) and profile_id.strip()
            else record_id
        )
        metadata.setdefault("source_id", record_id)

        try:
            record = CanonicalRecord.create(
                record_id=record_id,
                source=spec.name,
                trust_tier=spec.trust_tier,
                text=text,
                entity_mappings=entity_mappings,
                split_group=split_group,
                metadata=metadata,
            )
        except ValueError as exc:
            raise ValueError(f"invalid source record {record_id}: {exc}") from exc
        records.append(record)

    return tuple(records)
