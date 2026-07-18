"""Shared contracts and utilities for reproducible training pipelines."""

from src.training.contracts import CanonicalEntity, CanonicalRecord
from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)
from src.training.sources import SourceSpec, load_source_records
from src.training.validation import (
    ValidationFinding,
    require_valid_source,
    validate_records,
)

__all__ = [
    "CanonicalEntity",
    "CanonicalRecord",
    "SourceSpec",
    "ValidationFinding",
    "fingerprint_files",
    "load_source_records",
    "require_valid_source",
    "sha256_file",
    "stable_json_sha256",
    "validate_records",
]
