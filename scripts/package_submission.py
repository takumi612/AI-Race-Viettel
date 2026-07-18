"""Validate predictions and create a submission ZIP safely."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.validation.submission import package_submission, validate_output_directory


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and package submission outputs")
    parser.add_argument("--input", default="data/input", help="directory containing 1.txt through 100.txt")
    parser.add_argument("--output", default="data/output", help="directory containing 1.json through 100.json")
    parser.add_argument("--zip", default="output.zip", help="submission archive path")
    parser.add_argument("--validate-only", action="store_true", help="validate without writing an archive")
    args = parser.parse_args()

    errors = validate_output_directory(args.input, args.output, range(1, 101))
    if errors:
        print("Submission validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    if args.validate_only:
        print("Submission validation passed.")
        return 0

    try:
        package_submission(args.output, args.zip)
    except ValueError as error:
        print(f"Submission packaging failed: {error}", file=sys.stderr)
        return 1
    print(f"Created submission archive: {args.zip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
