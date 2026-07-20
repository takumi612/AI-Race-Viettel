from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.artifacts import initialize_runtime_artifacts, write_json
from clinical_nlp_lab.config import load_config, set_reproducible_seed
from clinical_nlp_lab.kb import (
    build_icd10_dictionary,
    build_rxnorm_dictionary,
    build_rxnorm_relation_cache,
    verify_dictionary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline ICD-10 and RxNorm artifacts.")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--skip-relations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    artifact_dir = args.artifact_dir if args.artifact_dir.is_absolute() else root / args.artifact_dir
    config = load_config(artifact_dir / "config.json")
    set_reproducible_seed(int(config["seed"]))
    initialize_runtime_artifacts(config, artifact_dir)

    started = time.time()
    icd10_metadata = build_icd10_dictionary(
        root / config["icd10_path"],
        artifact_dir / "icd10" / "icd10_dictionary.jsonl.gz",
        artifact_dir / "icd10" / "metadata.json",
        sheet_name=config["icd10_sheet"],
        header_row=int(config["icd10_header_row"]),
    )
    print(json.dumps({"icd10": icd10_metadata}, ensure_ascii=False), flush=True)

    rxnorm_metadata = build_rxnorm_dictionary(
        root / config["rxnorm_zip_path"],
        config["rxnorm_conso_member"],
        artifact_dir / "rxnorm" / "rxnorm_dictionary.jsonl.gz",
        artifact_dir / "rxnorm" / "metadata.json",
        config["rxnorm_languages"],
        config["rxnorm_sources"],
        config["rxnorm_tty"],
        config["rxnorm_suppress"],
    )
    print(json.dumps({"rxnorm": rxnorm_metadata}, ensure_ascii=False), flush=True)

    relation_metadata = None
    if not args.skip_relations:
        relation_metadata = build_rxnorm_relation_cache(
            root / config["rxnorm_zip_path"],
            config["rxnorm_rel_member"],
            artifact_dir / "rxnorm" / "rxnorm_relations.jsonl.gz",
            artifact_dir / "rxnorm" / "relations_metadata.json",
            config["rxnorm_relation_names"],
            config["rxnorm_sources"],
        )
        print(json.dumps({"rxnorm_relations": relation_metadata}, ensure_ascii=False), flush=True)

    verification = {
        "icd10": verify_dictionary(
            artifact_dir / "icd10" / "icd10_dictionary.jsonl.gz",
            expected_count=int(icd10_metadata["candidate_count"]),
        ),
        "rxnorm": verify_dictionary(
            artifact_dir / "rxnorm" / "rxnorm_dictionary.jsonl.gz",
            expected_count=int(rxnorm_metadata["candidate_count"]),
        ),
        "relations": relation_metadata,
        "elapsed_seconds": round(time.time() - started, 2),
        "save_load_passed": True,
    }
    write_json(artifact_dir / "kb_build_report.json", verification)
    print(json.dumps({"verification": verification}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
