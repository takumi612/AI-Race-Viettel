"""Explicitly promote a reviewed candidate run to a validated/locked artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.training.artifacts import load_run_state, promote_run


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically promote a reviewed training run."
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--metrics-json")
    parser.add_argument(
        "--status",
        choices=["validated", "locked"],
        default="validated",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    artifact_dir = Path(args.artifact_dir).resolve()
    metrics_path = (
        Path(args.metrics_json).resolve()
        if args.metrics_json
        else run_dir / "training_metrics.json"
    )
    try:
        if not metrics_path.is_file():
            raise ValueError(f"metrics JSON is missing: {metrics_path}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("metrics JSON must be a non-empty object")
        state = load_run_state(run_dir)
        promoted = promote_run(
            run_dir,
            artifact_dir,
            state,
            metrics=metrics,
            status=args.status,
        )
    except (
        OSError,
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        FileNotFoundError,
        FileExistsError,
    ) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "artifact_dir": str(promoted),
                "status": args.status,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
