from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.records import build_record_metadata
from clinical_nlp_lab.provenance import ProvenanceError, canonical_json_bytes, sha256_bytes
from clinical_nlp_lab.splitting import (
    SplitContractError,
    atomic_write_bytes,
    build_near_duplicate_groups,
    build_split_plan,
    metadata_artifact_payloads,
    near_duplicate_edges_payload,
    verify_metadata_artifacts,
)
from clinical_nlp_lab.records import RecordContractError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fingerprint-bound patient-record, near-duplicate and split metadata."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--write", action="store_true", help="Publish artifacts; default is read-only.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        records = build_record_metadata(args.dataset_root)
        near = build_near_duplicate_groups(args.dataset_root)
        fixed = build_split_plan(
            args.dataset_root,
            seed=args.seed,
            eval_profile="fixed_fold",
            fold_index=0,
            record_metadata=records,
            near_duplicates=near,
        )
        oof = [
            build_split_plan(
                args.dataset_root,
                seed=args.seed,
                eval_profile="oof_extended",
                fold_index=fold,
                record_metadata=records,
                near_duplicates=near,
            )
            for fold in range(5)
        ]
        manifest_bytes, descriptor_bytes = metadata_artifact_payloads(records, near)
        edges_bytes = near_duplicate_edges_payload(near)
        split_index_bytes = canonical_json_bytes(
            {
                "schema_id": "clinical_nlp.split_index",
                "schema_version": 1,
                "dataset_pair_fingerprint": records.dataset_fingerprint,
                "seed": args.seed,
                "fixed_split_sha256": fixed.manifest_sha256,
                "oof_split_sha256": [plan.manifest_sha256 for plan in oof],
            }
        )
        if args.write:
            output = args.output_dir
            atomic_write_bytes(output / "metadata_manifest.jsonl", manifest_bytes)
            atomic_write_bytes(output / "near_duplicate_edges.json", edges_bytes)
            atomic_write_bytes(output / "split_fixed_fold.json", fixed.manifest_bytes)
            for fold, plan in enumerate(oof):
                atomic_write_bytes(output / f"split_oof_fold_{fold}.json", plan.manifest_bytes)
            atomic_write_bytes(output / "split_index.json", split_index_bytes)
            # Descriptor is the metadata commit marker and is intentionally published last.
            atomic_write_bytes(output / "metadata_provenance.json", descriptor_bytes)
            verify_metadata_artifacts(output, records, near)
    except (RecordContractError, SplitContractError, ProvenanceError) as exc:
        print(
            json.dumps(
                {"status": "FAIL", "error_type": type(exc).__name__, "code": "E_DATASET_METADATA"},
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
    summary = {
        "status": "PASS",
        "mode": "write" if args.write else "check",
        "dataset_pair_fingerprint": records.dataset_fingerprint,
        "document_count": len(records.rows),
        "record_count": records.record_count,
        "near_duplicate_edge_count": len(near.edges),
        "near_duplicate_group_count": len(near.members_by_group),
        "near_duplicate_algorithm_hash": near.algorithm_hash,
        "metadata_manifest_sha256": sha256_bytes(manifest_bytes),
        "fixed_split_sha256": fixed.manifest_sha256,
        "fixed_counts": fixed.manifest["counts"],
        "oof_split_sha256": [plan.manifest_sha256 for plan in oof],
        "output_dir": str(args.output_dir),
    }
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
