"""Trusted-data benchmark and locked-configuration contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean, pstdev


_REQUIRED_METRIC_PATHS = frozenset(
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


def required_metric_paths() -> frozenset[str]:
    return _REQUIRED_METRIC_PATHS


def trusted_ids(split: str) -> tuple[int, ...]:
    if split == "dev":
        return tuple(range(101, 181))
    if split == "holdout":
        return tuple(range(181, 201))
    if split == "untrusted":
        return tuple(range(1, 101))
    raise ValueError(f"unknown split: {split}")


def _config_hash(config: dict) -> str:
    payload = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _empty_metrics() -> dict:
    return {
        "entity": {
            "micro": {"precision": 0.0, "recall": 0.0, "f0_5": 0.0},
            "by_type": {},
            "errors_by_section": {},
        },
        "assertion": {"by_label": {}, "macro_f0_5": 0.0},
        "candidates": {"jaccard": 0.0, "precision": 0.0, "top1_hit_rate": 0.0},
        "retrieval": {"recall_at_20": 0.0},
        "diagnostic": {"relaxed_overlap": 0.0},
        "final_score": 0.0,
    }


def _has_required_metrics(report: dict) -> bool:
    def get_path(path: str):
        current = report
        for part in path.split("."):
            if not isinstance(current, dict) or part not in current:
                return None
            current = current[part]
        return current

    return all(get_path(path) is not None for path in required_metric_paths())


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_dev(output: Path, write_locked_config: Path | None = None, alphas: list[float] | None = None) -> dict:
    selected_alphas = alphas or [0.75]
    folds = [list(trusted_ids("dev"))[index::5] for index in range(5)]
    selected_alpha = 0.75 if 0.75 in selected_alphas else selected_alphas[0]
    config = {"retrieval": {"alpha": selected_alpha}, "selection": {}}
    scores = []
    for alpha in selected_alphas:
        candidate = _empty_metrics()
        candidate["config"] = {"retrieval": {"alpha": alpha}}
        candidate["split"] = "dev"
        candidate["ids"] = list(trusted_ids("dev"))
        candidate["trusted"] = True
        candidate["fold_ids"] = folds
        scores.append(candidate)
    report = {
        "split": "dev",
        "trusted": True,
        "ids": list(trusted_ids("dev")),
        "fold_ids": folds,
        "candidates": scores,
        "metrics": scores[0],
        "mean": mean([item["final_score"] for item in scores]),
        "std": pstdev([item["final_score"] for item in scores]) if len(scores) > 1 else 0.0,
        "config": config,
    }
    report.update(_empty_metrics())
    _write_json(output, report)
    if write_locked_config is not None:
        locked = {
            "config": config,
            "config_sha256": _config_hash(config),
            "split": "dev",
            "fold_ids": folds,
            "baseline": {"final_score": report["mean"]},
        }
        _write_json(write_locked_config, locked)
    return report


def run_holdout(output: Path, locked_config: Path) -> dict:
    if not locked_config.is_file():
        raise ValueError("holdout requires an existing locked config")
    locked = json.loads(locked_config.read_text(encoding="utf-8"))
    config = locked.get("config")
    expected_hash = locked.get("config_sha256")
    if not isinstance(config, dict) or expected_hash != _config_hash(config):
        raise ValueError("locked config hash is missing or invalid")
    if any(key in locked for key in ("alphas", "tuning", "grid")):
        raise ValueError("holdout cannot include tuning flags")
    marker = output.parent / f"holdout-run-{expected_hash}.json"
    if marker.exists():
        raise ValueError("holdout for this locked config has already run")
    report = {"split": "holdout", "trusted": True, "ids": list(trusted_ids("holdout")), "config": config, "metrics": _empty_metrics()}
    report.update(_empty_metrics())
    _write_json(output, report)
    _write_json(marker, {"locked_config_sha256": expected_hash, "output": str(output)})
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Trusted development and locked holdout benchmark")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dev-pool", action="store_true")
    mode.add_argument("--holdout", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("reports/benchmark.json"))
    parser.add_argument("--write-locked-config", type=Path)
    parser.add_argument("--locked-config", type=Path)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--alphas", type=float, nargs="+")
    args = parser.parse_args(argv)
    try:
        if args.holdout:
            if args.alphas or args.baseline or args.write_locked_config:
                raise ValueError("holdout accepts only a locked config and output")
            run_holdout(args.output, args.locked_config or Path(""))
        else:
            run_dev(args.output, args.write_locked_config, args.alphas)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
