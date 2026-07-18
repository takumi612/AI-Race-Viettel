import argparse
import json
import sqlite3
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.validation.override_validator import (
    find_machine_specific_paths,
    load_verified_overrides,
    validate_override_entries,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit runtime resources and paths")
    parser.add_argument(
        "--scan-paths",
        nargs="+",
        type=Path,
        metavar="PATH",
        help="scan Python files or directories for machine-specific absolute paths",
    )
    parser.add_argument("--db", type=Path, help="read-only metadata SQLite database")
    parser.add_argument("--overrides", type=Path, help="override JSON file to audit")
    parser.add_argument(
        "--legacy-format",
        action="store_true",
        help="forensically audit the quarantined legacy nested mapping schema",
    )
    return parser.parse_args()


def _load_legacy_entries(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("legacy override root must be an object")

    entries = []
    for entry_type, mappings in payload.items():
        if not isinstance(mappings, dict):
            raise ValueError(f"legacy override section {entry_type!r} must be an object")
        for term, codes in mappings.items():
            entries.append(
                {
                    "term": term,
                    "type": entry_type,
                    "codes": codes,
                    "source": "legacy",
                    "note": "quarantined legacy mapping",
                }
            )
    return entries


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = _parse_args()
    if args.scan_paths:
        if args.db or args.overrides or args.legacy_format:
            raise SystemExit("--scan-paths cannot be combined with override audit options")
        try:
            findings = find_machine_specific_paths(args.scan_paths)
        except ValueError as exc:
            print(exc)
            return 1
        for finding in findings:
            print(f"{finding.path}:{finding.line_number}: {finding.value}")
        return 1 if findings else 0

    if not args.db or not args.overrides:
        raise SystemExit("--db and --overrides are required for an override audit")
    if not args.db.is_file():
        print(f"metadata DB does not exist: {args.db}")
        return 1
    if not args.overrides.is_file():
        print(f"override file does not exist: {args.overrides}")
        return 1

    try:
        if args.legacy_format:
            entries = _load_legacy_entries(args.overrides)
        else:
            frozen_entries = load_verified_overrides(args.overrides)
            entries = [
                {**dict(entry), "codes": list(entry["codes"])}
                for entry in frozen_entries
            ]
        errors = validate_override_entries(entries, str(args.db))
    except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
        print(exc)
        return 1

    for error in errors:
        print(error)
    if not errors:
        print(f"override audit passed: {args.overrides}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
