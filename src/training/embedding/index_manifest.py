"""Fingerprint contract binding a FAISS index to model, adapter, DB, and vectors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

from src.training.fingerprints import fingerprint_files, sha256_file


@dataclass(frozen=True, slots=True)
class EmbeddingIndexManifest:
    schema_version: int
    base_model: str
    adapter_sha256: str
    database_sha256: str
    embeddings_sha256: str
    index_sha256: str
    codes_sha256: str
    embeddings_file: str
    index_file: str
    codes_file: str
    count: int
    dimension: int

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "EmbeddingIndexManifest":
        if set(value) != set(cls.__dataclass_fields__):
            raise ValueError("index manifest keys do not match schema")
        manifest = cls(**value)
        if manifest.schema_version != 1:
            raise ValueError("index manifest schema_version must be 1")
        if manifest.count < 1 or manifest.dimension < 1:
            raise ValueError("index count and dimension must be positive")
        return manifest


def _inside(directory: Path, path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(directory.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("index artifacts must be inside index_dir") from exc


def write_index_manifest(
    index_dir: str | Path,
    *,
    base_model: str,
    adapter_dir: str | Path,
    database: str | Path,
    embeddings: str | Path,
    index: str | Path,
    codes: str | Path,
    count: int,
    dimension: int,
) -> Path:
    directory = Path(index_dir).resolve()
    adapter = Path(adapter_dir).resolve()
    adapter_files = sorted(path for path in adapter.rglob("*") if path.is_file())
    if not adapter_files:
        raise ValueError("adapter directory has no files")
    manifest = EmbeddingIndexManifest(
        schema_version=1,
        base_model=str(base_model),
        adapter_sha256=fingerprint_files(adapter_files, adapter),
        database_sha256=sha256_file(database),
        embeddings_sha256=sha256_file(embeddings),
        index_sha256=sha256_file(index),
        codes_sha256=sha256_file(codes),
        embeddings_file=_inside(directory, Path(embeddings)),
        index_file=_inside(directory, Path(index)),
        codes_file=_inside(directory, Path(codes)),
        count=count,
        dimension=dimension,
    )
    path = directory / "manifest.json"
    path.write_text(
        json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def validate_index_manifest(
    index_dir: str | Path,
    *,
    expected_base_model: str | None = None,
    expected_adapter_sha256: str | None = None,
    expected_database: str | Path | None = None,
    expected_adapter_dir: str | Path | None = None,
) -> EmbeddingIndexManifest:
    directory = Path(index_dir).resolve()
    path = directory / "manifest.json"
    if not path.is_file():
        raise ValueError(f"index manifest is missing: {path}")
    manifest = EmbeddingIndexManifest.from_mapping(
        json.loads(path.read_text(encoding="utf-8"))
    )
    checks = {
        "embeddings_sha256": sha256_file(directory / manifest.embeddings_file),
        "index_sha256": sha256_file(directory / manifest.index_file),
        "codes_sha256": sha256_file(directory / manifest.codes_file),
    }
    for field, actual in checks.items():
        if getattr(manifest, field) != actual:
            raise ValueError(f"{field} mismatch")
    if expected_database is not None:
        database_sha256 = sha256_file(expected_database)
        if manifest.database_sha256 != database_sha256:
            raise ValueError("database_sha256 mismatch")
    if expected_adapter_dir is not None:
        adapter = Path(expected_adapter_dir).resolve()
        adapter_files = sorted(path for path in adapter.rglob("*") if path.is_file())
        if not adapter_files:
            raise ValueError("expected adapter directory has no files")
        adapter_sha256 = fingerprint_files(adapter_files, adapter)
        if manifest.adapter_sha256 != adapter_sha256:
            raise ValueError("adapter_sha256 mismatch")
    if expected_base_model is not None and manifest.base_model != expected_base_model:
        raise ValueError("base_model mismatch")
    if (
        expected_adapter_sha256 is not None
        and manifest.adapter_sha256 != expected_adapter_sha256
    ):
        raise ValueError("adapter_sha256 mismatch")
    return manifest
