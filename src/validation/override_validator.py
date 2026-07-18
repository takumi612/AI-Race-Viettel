import ast
import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType


_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_POSIX_MACHINE_PATH = re.compile(r"^/(?:Users|home|mnt|opt|private|var)/")


@dataclass(frozen=True)
class PathFinding:
    path: Path
    line_number: int
    value: str


_OVERRIDE_FIELDS = {"term", "type", "codes", "source", "note"}


def normalize_override_term(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _entry_shape_errors(entry: object, index: int) -> list[str]:
    if not isinstance(entry, dict):
        return [f"entry[{index}] must be an object"]

    errors: list[str] = []
    for field in _OVERRIDE_FIELDS:
        if field not in entry:
            errors.append(f"entry[{index}] missing {field}")
    if errors:
        return errors

    for field in ("term", "type", "source", "note"):
        if not isinstance(entry[field], str) or not entry[field].strip():
            errors.append(f"entry[{index}] {field} must be a non-empty string")
    codes = entry["codes"]
    if not isinstance(codes, list) or not codes:
        errors.append(f"entry[{index}] codes must be a non-empty list")
    elif any(not isinstance(code, (str, int)) or not str(code).strip() for code in codes):
        errors.append(f"entry[{index}] codes must contain non-empty code values")
    return errors


def _validate_code(conn: sqlite3.Connection, entry: dict, code: str) -> str | None:
    entry_type = entry["type"]
    if entry_type == "THUỐC":
        rows = conn.execute(
            "SELECT DISTINCT name FROM rxnorm WHERE rxcui = ? ORDER BY name",
            (code,),
        ).fetchall()
        names = [str(row[0]) for row in rows if row[0]]
        if not names:
            return f"RxNorm code {code} does not exist"
        term = normalize_override_term(entry["term"])
        verified_alias = "verified brand alias" in normalize_override_term(entry["note"])
        if not verified_alias and not any(
            term in normalize_override_term(name) for name in names
        ):
            display_names = ", ".join(
                sorted({normalize_override_term(name) for name in names})
            )
            return (
                f"RxNorm code {code} names [{display_names}] are incompatible "
                f"with term {entry['term']!r}"
            )
        return None

    if entry_type == "CHẨN_ĐOÁN":
        row = conn.execute("SELECT 1 FROM icd10 WHERE code = ?", (code,)).fetchone()
        if row is None:
            return f"ICD-10 code {code} does not exist"
        return None

    return f"unsupported override type {entry_type!r} for code {code}"


def validate_override_entries(entries: list[dict], db_path: str) -> list[str]:
    errors: list[str] = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for index, entry in enumerate(entries):
            entry_errors = _entry_shape_errors(entry, index)
            errors.extend(entry_errors)
            if entry_errors:
                continue
            for raw_code in entry["codes"]:
                code = str(raw_code).strip()
                error = _validate_code(conn, entry, code)
                if error:
                    errors.append(f"entry[{index}] {error}")
    return errors


def load_verified_overrides(path: Path) -> tuple[dict, ...]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid verified override schema: {exc}") from exc

    if (
        not isinstance(payload, dict)
        or set(payload) != {"schema_version", "entries"}
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("entries"), list)
    ):
        raise ValueError("invalid verified override schema")

    frozen_entries = []
    for index, entry in enumerate(payload["entries"]):
        shape_errors = _entry_shape_errors(entry, index)
        unknown = set(entry) - _OVERRIDE_FIELDS if isinstance(entry, dict) else set()
        if shape_errors or unknown:
            details = shape_errors or [
                f"entry[{index}] unknown fields: {', '.join(sorted(unknown))}"
            ]
            raise ValueError(f"invalid verified override schema: {'; '.join(details)}")
        frozen = dict(entry)
        frozen["term"] = normalize_override_term(entry["term"])
        frozen["codes"] = tuple(str(code).strip() for code in entry["codes"])
        frozen_entries.append(MappingProxyType(frozen))
    return tuple(frozen_entries)


def _python_files(paths: list[Path]) -> list[Path]:
    files: set[Path] = set()
    for candidate in paths:
        if not candidate.exists():
            raise ValueError(f"scan target does not exist: {candidate}")
        if candidate.is_dir():
            files.update(candidate.rglob("*.py"))
        elif candidate.suffix == ".py":
            files.add(candidate)
        else:
            raise ValueError(f"unsupported scan target: {candidate}")
    return sorted(files, key=lambda item: str(item))


def _is_machine_specific_path(value: str) -> bool:
    return bool(
        _WINDOWS_ABSOLUTE_PATH.match(value)
        or value.startswith(chr(92) * 2)
        or _POSIX_MACHINE_PATH.match(value)
    )


def find_machine_specific_paths(paths: list[Path]) -> list[PathFinding]:
    findings: list[PathFinding] = []
    for path in _python_files(paths):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError, UnicodeDecodeError) as exc:
            raise ValueError(f"cannot audit {path}: {exc}") from exc
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and _is_machine_specific_path(node.value)
            ):
                findings.append(PathFinding(path, node.lineno, node.value))
    return sorted(
        findings, key=lambda item: (str(item.path), item.line_number, item.value)
    )
