from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.artifacts import write_json
from clinical_nlp_lab.pipeline import reload_equivalence_check, run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline Clinical NLP inference and create output.zip.")
    parser.add_argument("--input", type=Path, default=Path("input.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--diagnostics-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--zip-path", type=Path, default=Path("output.zip"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_inference(
        args.input,
        args.output_dir,
        args.artifact_dir,
        create_zip=True,
        diagnostics_dir=args.diagnostics_dir,
        zip_path=args.zip_path,
    )
    reload_check = reload_equivalence_check(args.input, args.artifact_dir)
    report = {"inference": summary, "reload_check": reload_check}
    write_json(args.diagnostics_dir / "integration_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
