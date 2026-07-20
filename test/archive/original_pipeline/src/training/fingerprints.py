"""Deterministic SHA-256 helpers for data lineage and run manifests."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    resolved = Path(path)
    if not resolved.is_file():
        raise ValueError(f"not a file: {resolved}")

    digest = sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_files(paths: Iterable[str | Path], root: str | Path) -> str:
    resolved_root = Path(root).resolve()
    entries: list[dict[str, str]] = []

    for path in paths:
        resolved_path = Path(path).resolve()
        try:
            relative_path = resolved_path.relative_to(resolved_root)
        except ValueError as exc:
            raise ValueError(
                f"path is outside fingerprint root: {resolved_path}"
            ) from exc
        entries.append(
            {
                "path": relative_path.as_posix(),
                "sha256": sha256_file(resolved_path),
            }
        )

    entries.sort(key=lambda entry: entry["path"])
    return stable_json_sha256(entries)
