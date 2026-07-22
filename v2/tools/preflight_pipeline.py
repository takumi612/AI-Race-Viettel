from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.preflight import build_preflight_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate dataset and runtime KB contracts before model loading.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = build_preflight_report(
            args.dataset,
            args.artifacts,
            args.config,
            output_path=args.output,
        )
    except Exception as exc:  # Unexpected operational boundary; never print clinical content.
        print(
            json.dumps(
                {"status": "ERROR", "error_type": type(exc).__name__, "output": str(args.output)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    summary = {
        "status": report["status"],
        "error_count": len(report["errors"]),
        "warning_count": len(report["warnings"]),
        "dataset_pair_fingerprint": report["dataset_pair_fingerprint"],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
