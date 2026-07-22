from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError

import pytest

import clinical_nlp_lab.runtime_control as runtime_control
from clinical_nlp_lab.runtime_control import (
    ModelInventoryItem,
    PipelineContractError,
    ResourcePlan,
    RuntimeEventLogger,
    atomic_write_json,
    choose_resource_plan,
    oom_retry_plan,
    resolve_unique_source,
    validate_model_budget,
)


def _gpu_snapshot(*, name: str = "Tesla T4", capability: str = "7.5", total: float = 16, free: float = 13):
    return {
        "gpu": {
            "available": True,
            "name": name,
            "capability": capability,
            "total_gib": total,
            "free_gib": free,
        },
        "host": {"ram_available_gib": 20, "disk_free_gib": 50},
    }


def _item(model_id: str, count: int | None, *, artifact_hash: str = "", active: bool = True):
    return ModelInventoryItem(model_id, "main", "test", count, artifact_hash, "none", active)


def _plan(**overrides):
    values = {
        "train_batch_size": 2,
        "eval_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "eval_accumulation_steps": 16,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": True,
        "qwen_enabled": False,
        "qwen_profile": None,
    }
    values.update(overrides)
    return ResourcePlan(**values)


def test_atomic_write_json_is_utf8_strict_and_returns_path(tmp_path):
    path = tmp_path / "nested" / "state.json"
    assert atomic_write_json(path, {"message": "Tiếng Việt"}) == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "Tiếng Việt"}
    assert not list(path.parent.glob("*.tmp"))

    with pytest.raises(ValueError):
        atomic_write_json(path, {"bad": math.nan})
    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "Tiếng Việt"}


def test_phase_emits_exactly_one_success_terminal_and_redacts_raw_text(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        runtime_control,
        "hardware_snapshot",
        lambda: {"gpu": {"available": False}, "host": {"ram_available_gib": 1, "disk_free_gib": 2}},
    )
    events = tmp_path / "events.jsonl"
    logger = RuntimeEventLogger("run-1", events)
    with logger.phase("preflight", context={"raw_text": "do not log", "documents": 3}):
        pass

    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["PHASE_START", "PHASE_END"]
    assert records[1]["duration_ms"] >= 0
    assert records[0]["context"] == {"raw_text": "[REDACTED]", "documents": 3}
    stdout_lines = capsys.readouterr().out.splitlines()
    assert len(stdout_lines) == 2
    assert all(line.startswith("[CLINICAL_PIPELINE] {") for line in stdout_lines)


def test_phase_emits_error_terminal_and_reraises_same_exception(tmp_path, monkeypatch):
    monkeypatch.setattr(
        runtime_control,
        "hardware_snapshot",
        lambda: {"gpu": {}, "host": {}},
    )
    logger = RuntimeEventLogger("run-2", tmp_path / "events.jsonl")
    failure = PipelineContractError("E_TEST", "failure", next_action="fix it")
    with pytest.raises(PipelineContractError) as caught:
        with logger.phase("train", attempt=2):
            raise failure
    assert caught.value is failure
    records = [json.loads(line) for line in logger.jsonl_path.read_text(encoding="utf-8").splitlines()]
    assert [record["event"] for record in records] == ["PHASE_START", "PHASE_ERROR"]
    assert records[-1]["error"]["code"] == "E_TEST"
    assert records[-1]["duration_ms"] >= 0


def test_resource_plan_for_16gb_gpu_uses_exact_safe_defaults():
    plan = choose_resource_plan(_gpu_snapshot(), require_gpu=True, qwen_requested=True)
    assert (plan.train_batch_size, plan.eval_batch_size) == (2, 2)
    assert plan.gradient_accumulation_steps == 8
    assert plan.eval_accumulation_steps == 16
    assert (plan.fp16, plan.bf16, plan.gradient_checkpointing) == (True, False, True)
    assert plan.qwen_enabled
    assert plan.qwen_profile == {"gpu_memory_utilization": 0.40, "max_model_len": 2048, "batch_size": 16}
    with pytest.raises(FrozenInstanceError):
        plan.train_batch_size = 4


def test_resource_plan_rejects_insufficient_vram():
    with pytest.raises(PipelineContractError, match="14 GiB") as caught:
        choose_resource_plan(_gpu_snapshot(free=11.99), require_gpu=True, qwen_requested=False)
    assert caught.value.code == "E_GPU_BUDGET"


def test_cpu_is_only_allowed_when_gpu_is_not_required():
    cpu = {"gpu": {"available": False}, "host": {}}
    plan = choose_resource_plan(cpu, require_gpu=False, qwen_requested=True)
    assert not plan.fp16 and not plan.bf16 and not plan.qwen_enabled
    with pytest.raises(PipelineContractError) as caught:
        choose_resource_plan(cpu, require_gpu=True, qwen_requested=False)
    assert caught.value.code == "E_GPU_BUDGET"


def test_p100_disables_qwen_with_reason():
    plan = choose_resource_plan(
        _gpu_snapshot(name="Tesla P100-PCIE-16GB", capability="6.0"),
        require_gpu=True,
        qwen_requested=True,
    )
    assert not plan.qwen_enabled
    assert plan.qwen_profile is None
    assert "P100" in plan.qwen_disabled_reason


def test_fast_dev_gpu_uses_small_batches():
    plan = choose_resource_plan(
        _gpu_snapshot(), require_gpu=True, qwen_requested=False, fast_dev_run=True
    )
    assert (plan.train_batch_size, plan.eval_batch_size, plan.gradient_accumulation_steps) == (1, 1, 2)


def test_oom_retry_has_exactly_one_rung():
    retry = oom_retry_plan(_plan())
    assert (retry.train_batch_size, retry.eval_batch_size) == (1, 1)
    assert retry.gradient_accumulation_steps == 16
    with pytest.raises(PipelineContractError) as caught:
        oom_retry_plan(retry)
    assert caught.value.code == "E_TRAIN_CUDA_OOM"


def test_model_budget_under_cap_and_hash_deduplication():
    report = validate_model_budget(
        [_item("encoder", 500_000_000, artifact_hash="same"), _item("shared", 500_000_000, artifact_hash="same")]
    )
    assert report["total_parameters"] == 500_000_000
    assert len(report["unique_items"]) == 1
    assert report["remaining"] == 8_500_000_000
    assert not report["warning"]


def test_model_budget_deduplicates_empty_hash_by_model_and_revision():
    report = validate_model_budget([_item("encoder", 10), _item("encoder", 10), _item("other", 20, active=False)])
    assert report["total_parameters"] == 10


@pytest.mark.parametrize("count", [None, 0, -1])
def test_model_budget_rejects_unknown_or_invalid_count(count):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("unknown", count)])
    assert caught.value.code == "E_MODEL_SIZE_UNKNOWN"


def test_model_budget_rejects_more_than_9b():
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("too-large", 9_000_000_001)])
    assert caught.value.code == "E_MODEL_OVER_9B"


def test_strict_override_never_falls_back(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    missing = tmp_path / "missing"
    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("input", missing, [valid], lambda path: path.is_dir())
    assert caught.value.code == "E_INPUT_OVERRIDE_INVALID"


def test_unique_source_zero_one_and_duplicate_candidates(tmp_path):
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    valid.mkdir()
    invalid.write_text("not a directory", encoding="utf-8")
    validator = lambda path: path.is_dir()

    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("model", None, [invalid], validator)
    assert caught.value.code == "E_MODEL_MISSING"
    assert resolve_unique_source("model", "", [valid, valid], validator) == valid.resolve()


def test_unique_source_ambiguity_is_sorted_case_insensitively(tmp_path):
    second = tmp_path / "z-source"
    first = tmp_path / "A-source"
    first.mkdir()
    second.mkdir()
    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("kb", None, [second, first], lambda path: path.is_dir())
    assert caught.value.code == "E_KB_AMBIGUOUS"
    assert caught.value.context["candidates"] == [str(first.resolve()), str(second.resolve())]
