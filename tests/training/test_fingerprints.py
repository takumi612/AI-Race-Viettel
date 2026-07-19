from pathlib import Path

import pytest

from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)


def test_stable_json_hash_ignores_mapping_order():
    assert stable_json_sha256({"a": 1, "b": 2}) == stable_json_sha256(
        {"b": 2, "a": 1}
    )


def test_file_hash_tracks_exact_bytes(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_bytes("xin chào".encode("utf-8"))

    first = sha256_file(path)
    path.write_bytes("xin chào!".encode("utf-8"))

    assert len(first) == 64
    assert sha256_file(path) != first


def test_file_set_hash_is_root_relative(tmp_path):
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_root.mkdir()
    second_root.mkdir()
    first_path = first_root / "a.txt"
    second_path = second_root / "a.txt"
    first_path.write_text("hello", encoding="utf-8")
    second_path.write_text("hello", encoding="utf-8")

    assert fingerprint_files([first_path], first_root) == fingerprint_files(
        [second_path], second_root
    )


def test_file_set_hash_rejects_path_outside_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("no", encoding="utf-8")

    with pytest.raises(ValueError, match="outside fingerprint root"):
        fingerprint_files([outside], root)


def test_file_hash_requires_a_file(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        sha256_file(Path(tmp_path) / "missing")
