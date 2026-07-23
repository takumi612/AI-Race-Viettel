from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.config import load_config, save_config, set_reproducible_seed
from clinical_nlp_lab.kb import (
    build_icd10_dictionary,
    build_rxnorm_dictionary,
    build_rxnorm_relation_cache,
    load_candidate_dictionary,
    sha256_file,
    verify_dictionary,
    write_metadata,
)
from clinical_nlp_lab.kb_contract import (
    KB_CONTRACT_VERSION,
    KBContractError,
    audit_gold_candidate_coverage,
    collect_organizer_candidate_occurrences,
    organizer_rxnorm_requirements,
)
from clinical_nlp_lab.provenance import scan_dataset_layout


BUILDER_VERSION = "2.0.0"
PINNED_ORGANIZER_NON_EMPTY = 2_239
PINNED_ORGANIZER_ABSTENTIONS = 1_373
PINNED_SUPPLEMENT_COUNT = 258
PINNED_RXNORM_RECORD_COUNT = 56_311


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build deterministic, evidence-bounded ICD-10 and RxNorm artifacts."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPOSITORY_ROOT / "data" / "kb",
        help="Pinned raw KB directory (backward-compatible name).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=REPOSITORY_ROOT / "data_v2" / "Training_data" / "synthetic_train_v2",
    )
    parser.add_argument("--artifact-dir", type=Path, default=PROJECT_ROOT / "artifacts")
    parser.add_argument(
        "--skip-relations",
        action="store_true",
        help="Keep the existing relation cache byte-for-byte and verify its hash.",
    )
    parser.add_argument(
        "--allow-unpinned-test-data",
        action="store_true",
        help="Disable only the current-real-data count assertions; all semantic gates remain active.",
    )
    return parser.parse_args()


def _repo_logical_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError as exc:
        raise KBContractError(f"Build input/output must remain inside the repository: {resolved}") from exc


def _verified_existing_relation(artifact_dir: Path) -> dict[str, object]:
    relation_path = artifact_dir / "rxnorm" / "rxnorm_relations.jsonl.gz"
    metadata_path = artifact_dir / "rxnorm" / "relations_metadata.json"
    if not relation_path.is_file() or not metadata_path.is_file():
        raise KBContractError("--skip-relations requires an existing relation artifact and metadata")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise KBContractError("Existing relation metadata is invalid") from exc
    actual_hash = sha256_file(relation_path)
    if metadata.get("sha256") != actual_hash:
        raise KBContractError("Existing relation artifact hash does not match its metadata")
    return {
        "artifact": "artifacts/rxnorm/rxnorm_relations.jsonl.gz",
        "metadata": "artifacts/rxnorm/relations_metadata.json",
        "sha256": actual_hash,
        "metadata_sha256": sha256_file(metadata_path),
        "preserved_byte_for_byte": True,
    }


def _publish_files(stage: Path, artifact_dir: Path, relative_paths: list[Path]) -> None:
    for relative in relative_paths:
        source = stage / relative
        destination = artifact_dir / relative
        if not source.is_file():
            raise KBContractError(f"Staged artifact is missing: {relative.as_posix()}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(source, destination)


def build(args: argparse.Namespace) -> dict[str, object]:
    kb_root = args.root.resolve()
    dataset_dir = args.dataset_dir.resolve()
    artifact_dir = args.artifact_dir.resolve()
    for path, label in ((kb_root, "KB root"), (dataset_dir, "dataset")):
        if not path.is_dir():
            raise KBContractError(f"{label} directory does not exist: {path}")
    _repo_logical_path(kb_root)
    _repo_logical_path(dataset_dir)
    _repo_logical_path(artifact_dir)

    config = load_config(artifact_dir / "config.json")
    config["candidate_top_k"] = 20
    config["candidate_output_k"] = 1
    config["enable_regex_fallback"] = False
    set_reproducible_seed(int(config["seed"]))

    snapshot, occurrences, abstentions = collect_organizer_candidate_occurrences(dataset_dir)
    requirements = organizer_rxnorm_requirements(occurrences)
    manifest_path = dataset_dir / "reports" / "dataset_manifest.jsonl"
    if not manifest_path.is_file():
        raise KBContractError("Dataset manifest is required for a fingerprint-pinned KB build")
    manifest_sha256 = sha256_file(manifest_path)

    relation_before: dict[str, object] | None = None
    if args.skip_relations:
        relation_before = _verified_existing_relation(artifact_dir)

    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kb-build-", dir=artifact_dir.parent) as temp_name:
        stage = Path(temp_name)
        save_config(config, stage / "config.json")
        config_sha256 = sha256_file(stage / "config.json")

        icd_path = stage / "icd10" / "icd10_dictionary.jsonl.gz"
        icd_meta_path = stage / "icd10" / "metadata.json"
        icd_metadata = build_icd10_dictionary(
            kb_root / config["icd10_path"],
            icd_path,
            icd_meta_path,
            sheet_name=config["icd10_sheet"],
            header_row=int(config["icd10_header_row"]),
            source_label=f"data/kb/{config['icd10_path']}",
            artifact_label="artifacts/icd10/icd10_dictionary.jsonl.gz",
        )

        evidence_path = stage / "rxnorm" / "organizer_supplement_evidence.jsonl.gz"
        evidence_meta_path = stage / "rxnorm" / "organizer_supplement_evidence_metadata.json"
        rx_path = stage / "rxnorm" / "rxnorm_dictionary.jsonl.gz"
        rx_meta_path = stage / "rxnorm" / "metadata.json"
        rxnorm_metadata = build_rxnorm_dictionary(
            kb_root / config["rxnorm_zip_path"],
            config["rxnorm_conso_member"],
            rx_path,
            rx_meta_path,
            config["rxnorm_languages"],
            config["rxnorm_sources"],
            config["rxnorm_tty"],
            config["rxnorm_suppress"],
            organizer_requirements=requirements,
            dataset_pair_fingerprint=snapshot.dataset_fingerprint,
            evidence_path=evidence_path,
            evidence_metadata_path=evidence_meta_path,
            expected_supplement_count=None if args.allow_unpinned_test_data else PINNED_SUPPLEMENT_COUNT,
            source_label=f"data/kb/{config['rxnorm_zip_path']}",
            artifact_label="artifacts/rxnorm/rxnorm_dictionary.jsonl.gz",
            evidence_label="artifacts/rxnorm/organizer_supplement_evidence.jsonl.gz",
        )

        if args.skip_relations:
            relation_metadata = relation_before
        else:
            relation_path = stage / "rxnorm" / "rxnorm_relations.jsonl.gz"
            relation_meta_path = stage / "rxnorm" / "relations_metadata.json"
            relation_metadata = build_rxnorm_relation_cache(
                kb_root / config["rxnorm_zip_path"],
                config["rxnorm_rel_member"],
                relation_path,
                relation_meta_path,
                config["rxnorm_relation_names"],
                config["rxnorm_sources"],
            )

        icd_records = load_candidate_dictionary(icd_path)
        rx_records = load_candidate_dictionary(rx_path)
        coverage = audit_gold_candidate_coverage(
            occurrences, icd_records, rx_records, abstentions=abstentions
        )
        if coverage["status"] != "PASS":
            raise KBContractError(
                f"Organizer KB coverage is incomplete: {coverage['missing_occurrence_count']} unresolved"
            )
        if not args.allow_unpinned_test_data:
            pinned_actual = (
                coverage["total_non_empty_occurrences"],
                coverage["abstentions"],
                rxnorm_metadata["supplement_candidate_count"],
                rxnorm_metadata["candidate_count"],
            )
            pinned_expected = (
                PINNED_ORGANIZER_NON_EMPTY,
                PINNED_ORGANIZER_ABSTENTIONS,
                PINNED_SUPPLEMENT_COUNT,
                PINNED_RXNORM_RECORD_COUNT,
            )
            if pinned_actual != pinned_expected:
                raise KBContractError(
                    f"Pinned KB invariants drifted: expected={pinned_expected}, actual={pinned_actual}"
                )

        verification = {
            "builder_version": BUILDER_VERSION,
            "contract_version": KB_CONTRACT_VERSION,
            "dataset_pair_fingerprint": snapshot.dataset_fingerprint,
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "icd10": verify_dictionary(icd_path, expected_count=int(icd_metadata["candidate_count"])),
            "rxnorm": verify_dictionary(rx_path, expected_count=int(rxnorm_metadata["candidate_count"])),
            "supplement_evidence": {
                "artifact": "artifacts/rxnorm/organizer_supplement_evidence.jsonl.gz",
                "record_count": int(rxnorm_metadata["supplement_candidate_count"]),
                "sha256": sha256_file(evidence_path),
                "metadata_sha256": sha256_file(evidence_meta_path),
            },
            "relations": relation_metadata,
            "save_load_passed": True,
        }
        # Never write host-specific paths into build reports.
        verification["icd10"]["path"] = "artifacts/icd10/icd10_dictionary.jsonl.gz"
        verification["rxnorm"]["path"] = "artifacts/rxnorm/rxnorm_dictionary.jsonl.gz"
        build_report_path = stage / "kb_build_report.json"
        write_metadata(build_report_path, verification)

        coverage_report = {
            "schema_id": "clinical_nlp.kb_coverage",
            "schema_version": 1,
            "builder_version": BUILDER_VERSION,
            "contract_version": KB_CONTRACT_VERSION,
            "status": "PASS",
            "dataset_pair_fingerprint": snapshot.dataset_fingerprint,
            "manifest_sha256": manifest_sha256,
            "config_sha256": config_sha256,
            "source_hashes": {
                "icd10_workbook": icd_metadata["source_sha256"],
                "rxnorm_zip": rxnorm_metadata["source_sha256"],
                "rxnorm_conso_member": rxnorm_metadata["member_sha256"],
            },
            "evidence_sha256": rxnorm_metadata["supplement_evidence_sha256"],
            "artifact_hashes": {
                "icd10": icd_metadata["sha256"],
                "rxnorm": rxnorm_metadata["sha256"],
                "relations": relation_metadata["sha256"] if relation_metadata else None,
            },
            "kb_build_report_sha256": sha256_file(build_report_path),
            "coverage": coverage,
            "unresolved": coverage["unresolved"],
            "linking_train_eligible_bypass_allowed": False,
        }
        coverage_report_path = stage / "kb_coverage_report.json"
        write_metadata(coverage_report_path, coverage_report)

        post_snapshot = scan_dataset_layout(dataset_dir)
        if post_snapshot.dataset_fingerprint != snapshot.dataset_fingerprint:
            raise KBContractError("Dataset bytes changed during the KB build")
        if args.skip_relations and _verified_existing_relation(artifact_dir) != relation_before:
            raise KBContractError("Skipped relation artifact changed during the KB build")

        publish = [
            Path("config.json"),
            Path("icd10/icd10_dictionary.jsonl.gz"),
            Path("icd10/metadata.json"),
            Path("rxnorm/rxnorm_dictionary.jsonl.gz"),
            Path("rxnorm/metadata.json"),
            Path("rxnorm/organizer_supplement_evidence.jsonl.gz"),
            Path("rxnorm/organizer_supplement_evidence_metadata.json"),
            Path("kb_build_report.json"),
            Path("kb_coverage_report.json"),
        ]
        if not args.skip_relations:
            publish.extend(
                [Path("rxnorm/rxnorm_relations.jsonl.gz"), Path("rxnorm/relations_metadata.json")]
            )
        _publish_files(stage, artifact_dir, publish)

    if args.skip_relations and _verified_existing_relation(artifact_dir) != relation_before:
        raise KBContractError("Relation artifact was not preserved byte-for-byte")
    return {
        "status": "PASS",
        "dataset_pair_fingerprint": snapshot.dataset_fingerprint,
        "coverage": coverage,
        "icd10_sha256": icd_metadata["sha256"],
        "rxnorm_sha256": rxnorm_metadata["sha256"],
        "supplement_evidence_sha256": rxnorm_metadata["supplement_evidence_sha256"],
        "relation_sha256": relation_metadata["sha256"] if relation_metadata else None,
    }


def main() -> int:
    try:
        result = build(parse_args())
    except (KBContractError, OSError, ValueError) as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "message": str(exc)}))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
