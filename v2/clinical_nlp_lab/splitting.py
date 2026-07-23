from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .provenance import (
    canonical_json_bytes,
    canonical_jsonl_bytes,
    load_json_strict,
    load_jsonl_strict,
    sha256_bytes,
    verify_dataset_provenance,
)
from .records import RecordMetadataSnapshot, build_record_metadata, normalized_text_from_raw_bytes


NEAR_DUPLICATE_SCHEMA_ID = "clinical_nlp.near_duplicate_groups"
NEAR_DUPLICATE_SCHEMA_VERSION = 1
NEAR_DUPLICATE_ALGORITHM_VERSION = "five_token_jaccard/v1"
NEAR_DUPLICATE_THRESHOLD = 0.92
SHINGLE_SIZE = 5
SPLIT_SCHEMA_ID = "clinical_nlp.development_split"
SPLIT_SCHEMA_VERSION = 1


class SplitContractError(ValueError):
    """Raised when an exact leakage-safe split cannot be constructed."""


@dataclass(frozen=True, slots=True)
class SimilarityDocument:
    document_id: str
    text: str
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class NearDuplicateEdge:
    left_id: str
    right_id: str
    similarity: float
    evidence: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "similarity": self.similarity,
            "evidence": self.evidence,
        }


@dataclass(frozen=True, slots=True)
class NearDuplicateGroups:
    dataset_fingerprint: str
    algorithm_hash: str
    threshold: float
    edges: tuple[NearDuplicateEdge, ...]
    group_by_document: Mapping[str, str]
    members_by_group: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class SplitPlan:
    manifest: dict[str, Any]
    manifest_bytes: bytes
    manifest_sha256: str

    @property
    def partitions(self) -> Mapping[str, Sequence[str]]:
        return self.manifest["partitions"]


def _algorithm_config() -> dict[str, Any]:
    return {
        "algorithm_version": NEAR_DUPLICATE_ALGORITHM_VERSION,
        "unicode_normalization": "NFKC",
        "casefold": True,
        "punctuation": "unicode-category-P-to-space",
        "whitespace": "collapse",
        "tokenization": "whitespace",
        "shingle_size": SHINGLE_SIZE,
        "short_document": "exact-normalized-equality-only",
        "similarity": "set-jaccard",
        "threshold": NEAR_DUPLICATE_THRESHOLD,
        "candidate_search": "exhaustive-with-safe-length-ratio-filter",
        "components": "undirected-transitive-closure",
    }


def near_duplicate_algorithm_hash() -> str:
    return sha256_bytes(canonical_json_bytes(_algorithm_config()))


def normalize_for_near_duplicate(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    characters = [" " if unicodedata.category(char).startswith("P") else char for char in normalized]
    return " ".join("".join(characters).split())


def token_shingles(text: str, size: int = SHINGLE_SIZE) -> frozenset[tuple[str, ...]]:
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise SplitContractError("Shingle size must be a positive exact integer")
    tokens = normalize_for_near_duplicate(text).split()
    if len(tokens) < size:
        return frozenset()
    return frozenset(tuple(tokens[index : index + size]) for index in range(len(tokens) - size + 1))


def near_duplicate_similarity(left: str, right: str) -> float:
    normalized_left = normalize_for_near_duplicate(left)
    normalized_right = normalize_for_near_duplicate(right)
    left_shingles = token_shingles(normalized_left)
    right_shingles = token_shingles(normalized_right)
    if not left_shingles or not right_shingles:
        return 1.0 if normalized_left == normalized_right else 0.0
    return len(left_shingles & right_shingles) / len(left_shingles | right_shingles)


def _group_id(algorithm_hash: str, members: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"clinical-nlp-near-duplicate-group/v1\0")
    digest.update(algorithm_hash.encode("ascii"))
    for document_id in members:
        encoded = document_id.encode("ascii")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return "ndg-" + digest.hexdigest()


def compute_near_duplicate_groups(
    documents: Iterable[SimilarityDocument],
    *,
    dataset_fingerprint: str,
) -> NearDuplicateGroups:
    document_list = sorted(documents, key=lambda item: int(item.document_id))
    if not document_list or len({item.document_id for item in document_list}) != len(document_list):
        raise SplitContractError("Near-duplicate input must contain unique documents")
    algorithm_hash = near_duplicate_algorithm_hash()
    prepared: list[tuple[SimilarityDocument, str, frozenset[tuple[str, ...]]]] = []
    for item in document_list:
        if str(int(item.document_id)) != item.document_id:
            raise SplitContractError("Near-duplicate document ID is not canonical numeric")
        if not isinstance(item.text, str) or not isinstance(item.raw_sha256, str):
            raise SplitContractError("Near-duplicate document fields have invalid types")
        normalized = normalize_for_near_duplicate(item.text)
        prepared.append((item, normalized, token_shingles(normalized)))

    parent = {item.document_id: item.document_id for item in document_list}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return
        if int(left_root) <= int(right_root):
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    by_length = sorted(prepared, key=lambda value: (len(value[2]), int(value[0].document_id)))
    edges: list[NearDuplicateEdge] = []
    for left_index, (left, left_normalized, left_shingles) in enumerate(by_length):
        for right, right_normalized, right_shingles in by_length[left_index + 1 :]:
            exact_raw = left.raw_sha256 == right.raw_sha256
            if exact_raw:
                similarity = 1.0
                evidence = "exact_raw_txt_sha256"
            elif not left_shingles or not right_shingles:
                if left_normalized != right_normalized:
                    continue
                similarity = 1.0
                evidence = "exact_normalized_short_document"
            else:
                smaller, larger = len(left_shingles), len(right_shingles)
                if smaller / larger < NEAR_DUPLICATE_THRESHOLD:
                    # Since items are length-sorted, later right documents can
                    # only make the safe upper bound smaller.
                    break
                intersection = len(left_shingles & right_shingles)
                union_size = smaller + larger - intersection
                similarity = intersection / union_size
                if similarity < NEAR_DUPLICATE_THRESHOLD:
                    continue
                evidence = "five_token_shingle_jaccard"
            left_id, right_id = sorted((left.document_id, right.document_id), key=int)
            edges.append(
                NearDuplicateEdge(
                    left_id=left_id,
                    right_id=right_id,
                    similarity=round(similarity, 12),
                    evidence=evidence,
                )
            )
            union(left_id, right_id)

    components: dict[str, list[str]] = {}
    for item in document_list:
        components.setdefault(find(item.document_id), []).append(item.document_id)
    members_by_group: dict[str, tuple[str, ...]] = {}
    group_by_document: dict[str, str] = {}
    for members in sorted(components.values(), key=lambda values: tuple(map(int, values))):
        ordered = tuple(sorted(members, key=int))
        group_id = _group_id(algorithm_hash, ordered)
        members_by_group[group_id] = ordered
        for document_id in ordered:
            group_by_document[document_id] = group_id
    edges.sort(key=lambda item: (int(item.left_id), int(item.right_id)))
    return NearDuplicateGroups(
        dataset_fingerprint=dataset_fingerprint,
        algorithm_hash=algorithm_hash,
        threshold=NEAR_DUPLICATE_THRESHOLD,
        edges=tuple(edges),
        group_by_document=group_by_document,
        members_by_group=members_by_group,
    )


def build_near_duplicate_groups(dataset_root: str | Path) -> NearDuplicateGroups:
    verification = verify_dataset_provenance(dataset_root)
    documents = (
        SimilarityDocument(
            document_id=pair.document_id,
            text=normalized_text_from_raw_bytes(pair.input_bytes),
            raw_sha256=pair.input_sha256,
        )
        for pair in verification.snapshot.pairs
    )
    return compute_near_duplicate_groups(
        documents, dataset_fingerprint=verification.dataset_fingerprint
    )


def _seeded_group_order(group_ids: Iterable[str], seed: int, label: str) -> list[str]:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise SplitContractError("Split seed must be an exact integer")
    return sorted(
        group_ids,
        key=lambda group_id: hashlib.sha256(
            f"clinical-nlp-split/v1\0{seed}\0{label}\0{group_id}".encode("utf-8")
        ).digest(),
    )


def _select_exact_groups(
    weighted_groups: Mapping[str, int], target: int, *, seed: int, label: str
) -> set[str]:
    if target < 0:
        raise SplitContractError("Split target cannot be negative")
    ordered = _seeded_group_order(weighted_groups, seed, label)
    reachable: dict[int, tuple[str, ...]] = {0: ()}
    for group_id in ordered:
        weight = weighted_groups[group_id]
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise SplitContractError("Hard-group split weight must be a positive integer")
        updates: dict[int, tuple[str, ...]] = {}
        for subtotal, selected in sorted(reachable.items(), reverse=True):
            candidate = subtotal + weight
            if candidate <= target and candidate not in reachable and candidate not in updates:
                updates[candidate] = (*selected, group_id)
        reachable.update(updates)
    if target not in reachable:
        raise SplitContractError(
            f"Hard groups cannot satisfy exact target {target}; blocking sizes={sorted(weighted_groups.values())}"
        )
    return set(reachable[target])


def _ids_in_groups(groups: Iterable[str], members: Mapping[str, Sequence[str]], allowed: set[str]) -> set[str]:
    result: set[str] = set()
    for group_id in groups:
        result.update(document_id for document_id in members[group_id] if document_id in allowed)
    return result


def build_split_plan(
    dataset_root: str | Path,
    *,
    seed: int = 42,
    eval_profile: str = "fixed_fold",
    fold_index: int = 0,
    record_metadata: RecordMetadataSnapshot | None = None,
    near_duplicates: NearDuplicateGroups | None = None,
) -> SplitPlan:
    if eval_profile not in {"fixed_fold", "oof_extended"}:
        raise SplitContractError("Unsupported evaluation profile")
    if isinstance(fold_index, bool) or not isinstance(fold_index, int) or not 0 <= fold_index < 5:
        raise SplitContractError("fold_index must be an exact integer in [0,4]")
    records = record_metadata or build_record_metadata(dataset_root)
    near = near_duplicates or build_near_duplicate_groups(dataset_root)
    if records.dataset_fingerprint != near.dataset_fingerprint:
        raise SplitContractError("Record and near-duplicate metadata bind different datasets")
    rows_by_id = {row.document_id: row for row in records.rows}
    quarantine = {document_id for document_id, row in rows_by_id.items() if row.source_role == "quarantine"}
    organizer = {document_id for document_id, row in rows_by_id.items() if row.source_role == "organizer"}
    synthetic = {document_id for document_id, row in rows_by_id.items() if row.source_role == "synthetic"}
    if len(quarantine) != 100 or len(organizer) != 100 or len(synthetic) != 2_000:
        raise SplitContractError("Dataset source-role counts do not match the frozen contract")
    if any(rows_by_id[document_id].train_eligible for document_id in quarantine):
        raise SplitContractError("Quarantine document is train eligible")
    if any(not rows_by_id[document_id].train_eligible for document_id in organizer | synthetic):
        raise SplitContractError("Eligible organizer/synthetic document is disabled")

    organizer_weights = {
        group_id: len(set(members) & organizer)
        for group_id, members in near.members_by_group.items()
        if set(members) & organizer
    }
    blind_groups = _select_exact_groups(organizer_weights, 10, seed=seed, label="organizer-blind")
    remaining_weights = {
        group_id: weight
        for group_id, weight in organizer_weights.items()
        if group_id not in blind_groups
    }
    fold_groups: list[set[str]] = []
    for current_fold in range(4):
        selected = _select_exact_groups(
            remaining_weights,
            18,
            seed=seed,
            label=f"organizer-fold-{current_fold}",
        )
        fold_groups.append(selected)
        remaining_weights = {
            group_id: weight
            for group_id, weight in remaining_weights.items()
            if group_id not in selected
        }
    if sum(remaining_weights.values()) != 18:
        raise SplitContractError("Final organizer fold is not exactly 18 documents")
    fold_groups.append(set(remaining_weights))

    blind_ids = _ids_in_groups(blind_groups, near.members_by_group, organizer)
    organizer_folds = [
        _ids_in_groups(groups, near.members_by_group, organizer) for groups in fold_groups
    ]
    organizer_validation = organizer_folds[fold_index]
    organizer_train = organizer - blind_ids - organizer_validation
    if (len(blind_ids), len(organizer_validation), len(organizer_train)) != (10, 18, 72):
        raise SplitContractError("Organizer partition counts violate 10/18/72")

    validation_groups = fold_groups[fold_index]
    excluded_groups = blind_groups | validation_groups
    synthetic_cross_exclusions = _ids_in_groups(
        excluded_groups, near.members_by_group, synthetic
    )
    pure_synthetic_weights: dict[str, int] = {}
    for group_id, members in near.members_by_group.items():
        member_set = set(members)
        synthetic_members = member_set & synthetic
        if not synthetic_members or member_set & organizer:
            continue
        pure_synthetic_weights[group_id] = len(synthetic_members)
    synthetic_validation_groups = _select_exact_groups(
        pure_synthetic_weights, 400, seed=seed, label="synthetic-validation"
    )
    synthetic_validation = _ids_in_groups(
        synthetic_validation_groups, near.members_by_group, synthetic
    )
    synthetic_train = synthetic - synthetic_validation - synthetic_cross_exclusions
    if len(synthetic_validation) != 400:
        raise SplitContractError("Synthetic validation is not exactly 400 documents")
    if len(synthetic_train) != 1_600 - len(synthetic_cross_exclusions):
        raise SplitContractError("Synthetic train/exclusion accounting is inconsistent")

    train_ids = organizer_train | synthetic_train
    validation_ids = organizer_validation | synthetic_validation
    challenge_ids = blind_ids
    if train_ids & validation_ids or train_ids & challenge_ids or validation_ids & challenge_ids:
        raise SplitContractError("Document leakage exists across split partitions")
    for group_id, members in near.members_by_group.items():
        partitions = sum(
            bool(set(members) & ids)
            for ids in (train_ids, validation_ids, challenge_ids)
        )
        if partitions > 1:
            raise SplitContractError(f"Hard near-duplicate group crosses partitions: {group_id}")

    manifest: dict[str, Any] = {
        "schema_id": SPLIT_SCHEMA_ID,
        "schema_version": SPLIT_SCHEMA_VERSION,
        "dataset_pair_fingerprint": records.dataset_fingerprint,
        "dataset_manifest_sha256": records.manifest_sha256,
        "record_metadata_sha256": records.metadata_sha256,
        "near_duplicate_algorithm_hash": near.algorithm_hash,
        "near_duplicate_threshold": near.threshold,
        "seed": seed,
        "eval_profile": eval_profile,
        "fold_index": fold_index,
        "partitions": {
            "quarantine_ids": sorted(quarantine, key=int),
            "organizer_blind_ids": sorted(challenge_ids, key=int),
            "organizer_train_ids": sorted(organizer_train, key=int),
            "organizer_validation_ids": sorted(organizer_validation, key=int),
            "synthetic_train_ids": sorted(synthetic_train, key=int),
            "synthetic_validation_ids": sorted(synthetic_validation, key=int),
            "synthetic_cross_source_exclusions": sorted(synthetic_cross_exclusions, key=int),
        },
        "organizer_folds": [sorted(ids, key=int) for ids in organizer_folds],
        "hard_group_ids": {
            "blind": sorted(blind_groups),
            "organizer_validation": sorted(validation_groups),
            "synthetic_validation": sorted(synthetic_validation_groups),
        },
        "counts": {
            "quarantine": len(quarantine),
            "organizer_blind": len(challenge_ids),
            "organizer_train": len(organizer_train),
            "organizer_validation": len(organizer_validation),
            "synthetic_train": len(synthetic_train),
            "synthetic_validation": len(synthetic_validation),
            "synthetic_cross_source_exclusions": len(synthetic_cross_exclusions),
        },
    }
    manifest_bytes = canonical_json_bytes(manifest)
    return SplitPlan(
        manifest=manifest,
        manifest_bytes=manifest_bytes,
        manifest_sha256=sha256_bytes(manifest_bytes),
    )


def authorize_blind_access(
    split_plan: SplitPlan,
    *,
    run_mode: str,
    eval_profile: str,
    fast_dev_run: bool,
    frozen_contract: Mapping[str, Any],
    already_used: bool,
) -> tuple[str, ...]:
    if type(fast_dev_run) is not bool or type(already_used) is not bool:
        raise SplitContractError("Blind access flags must be exact booleans")
    if (
        run_mode != "full"
        or eval_profile != "fixed_fold"
        or fast_dev_run
        or already_used
    ):
        raise SplitContractError("Blind labels are sealed for this run/profile/state")
    required = {
        "status": "frozen",
        "split_manifest_sha256": split_plan.manifest_sha256,
        "dataset_pair_fingerprint": split_plan.manifest["dataset_pair_fingerprint"],
        "blind_ids": split_plan.manifest["partitions"]["organizer_blind_ids"],
    }
    if any(frozen_contract.get(key) != value for key, value in required.items()):
        raise SplitContractError("Blind acceptance contract is missing or mismatched")
    return tuple(required["blind_ids"])


def metadata_artifact_payloads(
    records: RecordMetadataSnapshot, near: NearDuplicateGroups
) -> tuple[bytes, bytes]:
    if records.dataset_fingerprint != near.dataset_fingerprint:
        raise SplitContractError("Metadata inputs bind different datasets")
    rows = []
    for row in records.rows:
        payload = row.to_dict()
        payload["near_duplicate_group"] = near.group_by_document[row.document_id]
        payload["near_duplicate_algorithm_hash"] = near.algorithm_hash
        rows.append(payload)
    manifest_bytes = canonical_jsonl_bytes(rows)
    edges_bytes = near_duplicate_edges_payload(near)
    descriptor = {
        "schema_id": "clinical_nlp.dataset_metadata_provenance",
        "schema_version": 1,
        "dataset_pair_fingerprint": records.dataset_fingerprint,
        "dataset_manifest_sha256": records.manifest_sha256,
        "record_metadata_sha256": records.metadata_sha256,
        "near_duplicate_algorithm_hash": near.algorithm_hash,
        "metadata_manifest_sha256": sha256_bytes(manifest_bytes),
        "metadata_manifest_size_bytes": len(manifest_bytes),
        "near_duplicate_edges_sha256": sha256_bytes(edges_bytes),
        "near_duplicate_edges_size_bytes": len(edges_bytes),
        "record_count": records.record_count,
        "document_count": len(records.rows),
        "near_duplicate_edge_count": len(near.edges),
    }
    return manifest_bytes, canonical_json_bytes(descriptor)


def near_duplicate_edges_payload(near: NearDuplicateGroups) -> bytes:
    return canonical_json_bytes(
        {
            "schema_id": NEAR_DUPLICATE_SCHEMA_ID,
            "schema_version": NEAR_DUPLICATE_SCHEMA_VERSION,
            "dataset_pair_fingerprint": near.dataset_fingerprint,
            "algorithm": _algorithm_config(),
            "algorithm_hash": near.algorithm_hash,
            "edges": [edge.to_dict() for edge in near.edges],
            "groups": [
                {"group_id": group_id, "document_ids": list(members)}
                for group_id, members in sorted(near.members_by_group.items())
            ],
        }
    )


def verify_metadata_artifacts(
    output_dir: str | Path,
    records: RecordMetadataSnapshot,
    near: NearDuplicateGroups,
) -> dict[str, Any]:
    output = Path(output_dir)
    manifest_path = output / "metadata_manifest.jsonl"
    edges_path = output / "near_duplicate_edges.json"
    descriptor_path = output / "metadata_provenance.json"
    for path in (manifest_path, edges_path, descriptor_path):
        if path.is_symlink() or not path.is_file():
            raise SplitContractError(f"Required metadata artifact is missing: {path.name}")
    expected_manifest, expected_descriptor = metadata_artifact_payloads(records, near)
    expected_edges = near_duplicate_edges_payload(near)
    actual_manifest = manifest_path.read_bytes()
    actual_edges = edges_path.read_bytes()
    actual_descriptor = descriptor_path.read_bytes()
    if (
        actual_manifest != expected_manifest
        or actual_edges != expected_edges
        or actual_descriptor != expected_descriptor
    ):
        raise SplitContractError("Metadata artifact bytes do not match their bound inputs")
    manifest_rows = load_jsonl_strict(actual_manifest, source="metadata manifest")
    edge_payload = load_json_strict(actual_edges, source="near-duplicate edges")
    descriptor = load_json_strict(actual_descriptor, source="metadata descriptor")
    if not isinstance(edge_payload, dict) or not isinstance(descriptor, dict):
        raise SplitContractError("Metadata descriptor/edge payload is not an object")
    if len(manifest_rows) != len(records.rows):
        raise SplitContractError("Metadata manifest document count mismatch")
    return descriptor


def atomic_write_bytes(path: str | Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != payload:
            raise SplitContractError("Temporary metadata bytes changed before publication")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination
