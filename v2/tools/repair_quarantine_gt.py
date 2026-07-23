from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.quarantine_repair import QuarantineRepairError, repair_quarantine_gt
from clinical_nlp_lab.provenance import ProvenanceError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit or apply evidence-bounded candidate repairs to quarantined GT IDs 1-100."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--icd-artifact", type=Path, required=True)
    parser.add_argument("--write", action="store_true", help="Apply the reviewed plan; default is read-only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = repair_quarantine_gt(
            args.dataset_root,
            args.icd_artifact,
            write=bool(args.write),
        )
    except (QuarantineRepairError, ProvenanceError) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "code": "E_QUARANTINE_REPAIR"},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    except Exception as exc:
        print(
            json.dumps(
                {"status": "ERROR", "error_type": type(exc).__name__},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    payload = result.to_dict()
    payload["status"] = "PASS"
    print(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
