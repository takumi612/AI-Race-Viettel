import json

import pytest

from src.evaluation.benchmark import required_metric_paths, run_dev, run_holdout, trusted_ids


def test_benchmark_metric_contract_is_stable():
    assert required_metric_paths() == frozenset(
        {
            "entity.micro.precision",
            "entity.micro.recall",
            "entity.micro.f0_5",
            "entity.by_type",
            "entity.errors_by_section",
            "assertion.by_label",
            "assertion.macro_f0_5",
            "candidates.jaccard",
            "candidates.precision",
            "candidates.top1_hit_rate",
            "retrieval.recall_at_20",
            "diagnostic.relaxed_overlap",
            "final_score",
        }
    )


def test_trusted_split_excludes_pseudo_gt():
    assert trusted_ids("dev")[0] == 101
    assert trusted_ids("dev")[-1] == 180
    assert trusted_ids("holdout") == tuple(range(181, 201))
    assert trusted_ids("untrusted") == tuple(range(1, 101))


def test_dev_report_and_locked_config_are_written(tmp_path):
    report_path = tmp_path / "reports" / "dev.json"
    lock_path = tmp_path / "reports" / "locked.json"
    report = run_dev(report_path, lock_path, [0.6, 0.75])
    assert report["ids"] == list(range(101, 181))
    assert report["trusted"] is True
    assert report_path.is_file() and lock_path.is_file()
    assert json.loads(lock_path.read_text(encoding="utf-8"))["config_sha256"]


def test_holdout_requires_valid_lock_and_runs_once(tmp_path):
    with pytest.raises(ValueError, match="locked config"):
        run_holdout(tmp_path / "holdout.json", tmp_path / "missing.json")

    lock_path = tmp_path / "locked.json"
    run_dev(tmp_path / "dev.json", lock_path)
    output = tmp_path / "holdout.json"
    report = run_holdout(output, lock_path)
    assert report["ids"] == list(range(181, 201))
    with pytest.raises(ValueError, match="already run"):
        run_holdout(tmp_path / "second.json", lock_path)
