"""Shared contracts and utilities for reproducible training pipelines."""

from src.training.contracts import CanonicalEntity, CanonicalRecord
from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)
from src.training.sources import SourceSpec, load_source_records
from src.training.projections import (
    project_embedding_seeds,
    project_ner_records,
    project_reranker_seeds,
)
from src.training.splits import (
    SplitAssignment,
    assert_no_split_leakage,
    build_synthetic_split,
    build_trusted_folds,
)
from src.training.validation import (
    ValidationFinding,
    require_valid_source,
    validate_records,
)

__all__ = [
    "CanonicalEntity",
    "CanonicalRecord",
    "SourceSpec",
    "SplitAssignment",
    "ValidationFinding",
    "assert_no_split_leakage",
    "build_synthetic_split",
    "build_trusted_folds",
    "fingerprint_files",
    "load_source_records",
    "project_embedding_seeds",
    "project_ner_records",
    "project_reranker_seeds",
    "require_valid_source",
    "sha256_file",
    "stable_json_sha256",
    "validate_records",
]
