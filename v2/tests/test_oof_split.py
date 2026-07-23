from __future__ import annotations

from pathlib import Path

import pytest

from clinical_nlp_lab.records import build_record_metadata
from clinical_nlp_lab.splitting import (
    SimilarityDocument,
    SplitContractError,
    authorize_blind_access,
    build_near_duplicate_groups,
    build_split_plan,
    compute_near_duplicate_groups,
    metadata_artifact_payloads,
    near_duplicate_algorithm_hash,
    near_duplicate_similarity,
    _select_exact_groups,
    atomic_write_bytes,
    near_duplicate_edges_payload,
    verify_metadata_artifacts,
)


ROOT = Path(__file__).parents[1]


@pytest.fixture(scope="module")
def real_metadata():
    dataset = ROOT.parent / "data_v2" / "Training_data" / "synthetic_train_v2"
    if not dataset.is_dir():
        pytest.skip("real workspace dataset is not attached")
    return dataset, build_record_metadata(dataset), build_near_duplicate_groups(dataset)


def test_near_duplicate_contract_groups_exact_copy_not_shared_clinical_surface():
    documents = [
        SimilarityDocument("1", "alpha beta gamma delta epsilon zeta", "a" * 64),
        SimilarityDocument("2", "alpha beta gamma delta epsilon zeta", "a" * 64),
        SimilarityDocument("3", "alpha bệnh tim one two three four", "b" * 64),
        SimilarityDocument("4", "alpha bệnh tim red blue green black", "c" * 64),
    ]
    result = compute_near_duplicate_groups(documents, dataset_fingerprint="d" * 64)

    assert result.group_by_document["1"] == result.group_by_document["2"]
    assert result.group_by_document["3"] != result.group_by_document["4"]
    assert len(result.edges) == 1
    assert result.edges[0].evidence == "exact_raw_txt_sha256"
    assert len(near_duplicate_algorithm_hash()) == 64


def test_similarity_is_deterministic_and_short_documents_require_exact_match():
    assert near_duplicate_similarity("A, B C D E F", "a b c d e f") == 1.0
    assert near_duplicate_similarity("short text", "short text") == 1.0
    assert near_duplicate_similarity("short text", "short texts") == 0.0


def test_blind_access_is_fail_closed_without_exact_frozen_contract():
    class FakePlan:
        manifest_sha256 = "s" * 64
        manifest = {
            "dataset_pair_fingerprint": "d" * 64,
            "partitions": {"organizer_blind_ids": ["101"]},
        }

    plan = FakePlan()
    contract = {
        "status": "frozen",
        "split_manifest_sha256": plan.manifest_sha256,
        "dataset_pair_fingerprint": "d" * 64,
        "blind_ids": ["101"],
    }
    with pytest.raises(SplitContractError, match="sealed"):
        authorize_blind_access(
            plan, run_mode="resume", eval_profile="fixed_fold", fast_dev_run=False,
            frozen_contract=contract, already_used=False,
        )
    with pytest.raises(SplitContractError, match="mismatched"):
        authorize_blind_access(
            plan, run_mode="full", eval_profile="fixed_fold", fast_dev_run=False,
            frozen_contract={**contract, "blind_ids": ["102"]}, already_used=False,
        )
    assert authorize_blind_access(
        plan, run_mode="full", eval_profile="fixed_fold", fast_dev_run=False,
        frozen_contract=contract, already_used=False,
    ) == ("101",)


def test_impossible_hard_group_target_fails_closed():
    with pytest.raises(SplitContractError, match="cannot satisfy"):
        _select_exact_groups({"large": 11, "small": 1}, 10, seed=42, label="fixture")


def test_real_split_is_deterministic_exact_and_leakage_safe(real_metadata):
    dataset, records, near = real_metadata

    first = build_split_plan(dataset, seed=42, record_metadata=records, near_duplicates=near)
    second = build_split_plan(dataset, seed=42, record_metadata=records, near_duplicates=near)

    assert first.manifest_bytes == second.manifest_bytes
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest["counts"] == {
        "quarantine": 100,
        "organizer_blind": 10,
        "organizer_train": 72,
        "organizer_validation": 18,
        "synthetic_train": 1600 - first.manifest["counts"]["synthetic_cross_source_exclusions"],
        "synthetic_validation": 400,
        "synthetic_cross_source_exclusions": first.manifest["counts"]["synthetic_cross_source_exclusions"],
    }
    folds = first.manifest["organizer_folds"]
    assert len(folds) == 5 and all(len(fold) == 18 for fold in folds)
    assert len(set().union(*(set(fold) for fold in folds))) == 90
    assert not set(first.partitions["quarantine_ids"]) & (
        set(first.partitions["organizer_train_ids"]) | set(first.partitions["organizer_validation_ids"])
    )
    metadata_bytes, descriptor_bytes = metadata_artifact_payloads(records, near)
    assert len(metadata_bytes) > 0 and len(descriptor_bytes) > 0


def test_each_oof_fold_has_18_validation_documents_and_never_changes_blind(real_metadata):
    dataset, records, near = real_metadata
    plans = [
        build_split_plan(
            dataset,
            seed=42,
            eval_profile="oof_extended",
            fold_index=fold,
            record_metadata=records,
            near_duplicates=near,
        )
        for fold in range(5)
    ]
    blind_sets = {tuple(plan.partitions["organizer_blind_ids"]) for plan in plans}
    synthetic_validation_sets = {
        tuple(plan.partitions["synthetic_validation_ids"]) for plan in plans
    }
    assert len(blind_sets) == 1
    assert len(synthetic_validation_sets) == 1
    assert all(len(plan.partitions["organizer_validation_ids"]) == 18 for plan in plans)
    assert set().union(*(set(plan.partitions["organizer_validation_ids"]) for plan in plans)) == set(
        plans[0].manifest["organizer_folds"][0]
        + plans[0].manifest["organizer_folds"][1]
        + plans[0].manifest["organizer_folds"][2]
        + plans[0].manifest["organizer_folds"][3]
        + plans[0].manifest["organizer_folds"][4]
    )


def test_metadata_artifacts_are_byte_bound_and_tamper_evident(real_metadata, tmp_path: Path):
    _dataset, records, near = real_metadata
    manifest_bytes, descriptor_bytes = metadata_artifact_payloads(records, near)
    edges_bytes = near_duplicate_edges_payload(near)
    atomic_write_bytes(tmp_path / "metadata_manifest.jsonl", manifest_bytes)
    atomic_write_bytes(tmp_path / "near_duplicate_edges.json", edges_bytes)
    atomic_write_bytes(tmp_path / "metadata_provenance.json", descriptor_bytes)

    descriptor = verify_metadata_artifacts(tmp_path, records, near)
    assert descriptor["document_count"] == 2_200
    (tmp_path / "near_duplicate_edges.json").write_bytes(edges_bytes + b" ")
    with pytest.raises(SplitContractError, match="do not match"):
        verify_metadata_artifacts(tmp_path, records, near)
