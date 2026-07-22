from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

import clinical_nlp_lab.runtime_control as runtime_control
from clinical_nlp_lab.runtime_control import (
    DEFAULT_QWEN_MODEL_ID,
    QWEN_7B_MODEL_ID,
    ModelInventoryItem,
    PipelineContractError,
    QwenProfile,
    ResourcePlan,
    RuntimeEventLogger,
    SourceResolution,
    atomic_write_json,
    choose_resource_plan,
    hardware_snapshot,
    oom_retry_plan,
    resolve_unique_source,
    validate_model_budget,
)


def _gpu_snapshot(
    *,
    name: str = "Tesla T4",
    capability: str | None = "7.5",
    total: float = 16,
    free: float = 13,
    ram: float | None = 20,
    disk: float | None = 50,
):
    return {
        "gpu": {
            "available": True,
            "name": name,
            "capability": capability,
            "total_gib": total,
            "free_gib": free,
        },
        "host": {"ram_available_gib": ram, "disk_free_gib": disk},
    }


def _item(
    model_id: str,
    count: int | None,
    *,
    artifact_hash: str = "a" * 64,
    active: bool = True,
    role: str = "test",
    revision: str = "main",
    quantization: str = "none",
    source: str = "attached-model-dataset",
):
    return ModelInventoryItem(
        model_id=model_id,
        revision=revision,
        role=role,
        parameter_count=count,
        artifact_hash=artifact_hash,
        quantization=quantization,
        active=active,
        source=source,
    )


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


def _stub_hardware(monkeypatch):
    monkeypatch.setattr(
        runtime_control,
        "hardware_snapshot",
        lambda: {"gpu": {"available": False}, "host": {"ram_available_gib": 1, "disk_free_gib": 2}},
    )


def _records(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_atomic_write_json_is_utf8_strict_and_returns_path(tmp_path):
    path = tmp_path / "nested" / "state.json"
    assert atomic_write_json(path, {"message": "Tiếng Việt"}) == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "Tiếng Việt"}
    assert not list(path.parent.glob("*.tmp"))

    with pytest.raises(ValueError):
        atomic_write_json(path, {"bad": math.nan})
    assert json.loads(path.read_text(encoding="utf-8")) == {"message": "Tiếng Việt"}


def test_phase_emits_success_terminal_and_uses_positive_schema_redaction(tmp_path, capsys, monkeypatch):
    _stub_hardware(monkeypatch)
    events = tmp_path / "events.jsonl"
    logger = RuntimeEventLogger("run-1", events)
    with logger.phase(
        "preflight",
        context={
            "raw_text": "do not log",
            "patientNote": "sensitive note",
            "access_token": "credential",
            "unapproved_field": "also hidden",
            "documents": 3,
        },
    ):
        pass

    records = _records(events)
    assert [record["event"] for record in records] == ["PHASE_START", "PHASE_END"]
    assert [record["scope"] for record in records] == ["attempt", "attempt"]
    assert records[1]["duration_ms"] >= 0
    assert records[0]["context"] == {
        "raw_text": "[REDACTED]",
        "patientNote": "[REDACTED]",
        "access_token": "[REDACTED]",
        "unapproved_field": "[REDACTED]",
        "documents": 3,
    }
    serialized = events.read_text(encoding="utf-8") + capsys.readouterr().out
    assert "do not log" not in serialized
    assert "sensitive note" not in serialized
    assert "credential" not in serialized


def test_phase_error_never_logs_arbitrary_exception_message(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-2", tmp_path / "events.jsonl")
    failure = RuntimeError("clinical text and bearer-token-value must never persist")

    with pytest.raises(RuntimeError) as caught:
        with logger.phase("train", attempt=1):
            raise failure

    assert caught.value is failure
    record = _records(logger.jsonl_path)[-1]
    assert record["error"] == {
        "code": "E_UNEXPECTED",
        "type": "RuntimeError",
        "message": "RuntimeError reported pipeline error E_UNEXPECTED.",
        "retriable": False,
        "next_action": "Follow the documented recovery action for E_UNEXPECTED.",
        "context": {},
    }
    assert "bearer-token-value" not in logger.jsonl_path.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "error",
    [
        {"code": "E_TEST"},
        {
            "code": "E_TEST",
            "type": "TestError",
            "message": "free text",
            "retriable": "yes",
            "next_action": "free text",
        },
        {
            "code": "E_TEST",
            "type": "TestError",
            "message": "free text",
            "retriable": False,
            "next_action": "free text",
            "context": "not-a-mapping",
        },
    ],
)
def test_emit_rejects_invalid_error_schema(error, tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-error", tmp_path / "events.jsonl")
    with pytest.raises(PipelineContractError) as caught:
        logger.emit("preflight", "PREFLIGHT_RESULT", "ERROR", error=error)
    assert caught.value.code == "E_EVENT_ERROR_SCHEMA"
    assert not logger.jsonl_path.exists()


def test_phase_lifecycle_rejects_duplicate_and_post_terminal_events(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-state", tmp_path / "events.jsonl")
    with logger.phase("train", attempt=1):
        with pytest.raises(PipelineContractError) as duplicate:
            logger.emit("train", "PHASE_START", "RUNNING", attempt=1)
        assert duplicate.value.code == "E_EVENT_SEQUENCE"

    with pytest.raises(PipelineContractError) as post_terminal:
        logger.emit("train", "MEMORY_SNAPSHOT", "SUCCESS", attempt=1)
    assert post_terminal.value.code == "E_EVENT_SEQUENCE"


def test_retry_lifecycle_has_attempt_terminals_then_one_aggregate(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-retry", tmp_path / "events.jsonl")
    retryable = PipelineContractError(
        "E_TRAIN_CUDA_OOM",
        "message must be replaced",
        retriable=True,
    )
    with pytest.raises(PipelineContractError):
        with logger.phase("train.stage2", attempt=1):
            raise retryable
    logger.emit_oom_retry("train.stage2", from_attempt=1, to_attempt=2)
    with logger.phase("train.stage2", attempt=2):
        pass
    with pytest.raises(PipelineContractError) as mismatch:
        logger.aggregate_terminal(
            "train.stage2",
            succeeded=False,
            error=PipelineContractError("E_TEST", "must not be logged"),
        )
    assert mismatch.value.code == "E_EVENT_SEQUENCE"
    aggregate = logger.aggregate_terminal("train.stage2", succeeded=True)

    records = _records(logger.jsonl_path)
    assert [(record["scope"], record["event"], record["attempt"]) for record in records] == [
        ("attempt", "PHASE_START", 1),
        ("attempt", "PHASE_ERROR", 1),
        ("transition", "OOM_RETRY", None),
        ("attempt", "PHASE_START", 2),
        ("attempt", "PHASE_END", 2),
        ("aggregate", "PHASE_END", None),
    ]
    assert records[2]["context"] == {"from_attempt": 1, "to_attempt": 2}
    assert records[1]["error"]["retriable"] is True
    assert aggregate["scope"] == "aggregate" and aggregate["attempt"] is None
    with pytest.raises(PipelineContractError):
        logger.aggregate_terminal("train.stage2", succeeded=True)
    with pytest.raises(PipelineContractError):
        logger.emit("train.stage2", "PHASE_START", "RUNNING", attempt=3)


def test_transition_and_aggregate_scopes_require_null_attempt(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-scope", tmp_path / "events.jsonl")
    with pytest.raises(PipelineContractError) as transition:
        logger.emit(
            "train",
            "OOM_RETRY",
            "RUNNING",
            scope="transition",
            attempt=1,
            context={"from_attempt": 1, "to_attempt": 2},
        )
    assert transition.value.code == "E_EVENT_SEQUENCE"

    with pytest.raises(PipelineContractError) as aggregate:
        logger.emit("train", "PHASE_END", "SUCCESS", scope="aggregate", attempt=0)
    assert aggregate.value.code == "E_EVENT_SEQUENCE"


def test_only_one_a1_to_a2_oom_transition_is_permitted(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-transition", tmp_path / "events.jsonl")
    with pytest.raises(PipelineContractError) as unsupported:
        logger.emit(
            "train",
            "OPTIONAL_FALLBACK",
            "RUNNING",
            scope="transition",
            attempt=None,
        )
    assert unsupported.value.code == "E_EVENT_SEQUENCE"

    with pytest.raises(PipelineContractError) as third_attempt:
        logger.emit_oom_retry("train", from_attempt=2, to_attempt=3)
    assert third_attempt.value.code == "E_EVENT_SEQUENCE"


def test_logger_rejects_cross_process_writes(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-parent", tmp_path / "events.jsonl")
    owner_pid = logger._owner_pid
    monkeypatch.setattr(runtime_control.os, "getpid", lambda: owner_pid + 1)
    with pytest.raises(PipelineContractError) as caught:
        logger.emit("train", "PHASE_START", "RUNNING")
    assert caught.value.code == "E_EVENT_WRITER_PROCESS"


def test_resource_plan_for_t4_uses_immutable_3b_safe_profile():
    plan = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        kernel_probe=True,
    )
    assert (plan.train_batch_size, plan.eval_batch_size) == (2, 2)
    assert plan.gradient_accumulation_steps == 8
    assert plan.eval_accumulation_steps == 16
    assert (plan.fp16, plan.bf16, plan.gradient_checkpointing) == (True, False, True)
    assert plan.qwen_enabled
    assert plan.qwen_profile == QwenProfile(
        model_id=DEFAULT_QWEN_MODEL_ID,
        gpu_memory_utilization=0.40,
        max_model_len=1024,
        batch_ladder=(8, 4, 1),
    )
    with pytest.raises(FrozenInstanceError):
        plan.qwen_profile.max_model_len = 2048
    with pytest.raises(TypeError):
        plan.qwen_profile.batch_ladder[0] = 16


def test_resource_plan_rejects_insufficient_vram_for_training_but_allows_cpu_validation():
    with pytest.raises(PipelineContractError, match="14 GiB") as caught:
        choose_resource_plan(_gpu_snapshot(free=11.99), require_gpu=True, qwen_requested=False)
    assert caught.value.code == "E_GPU_BUDGET"

    plan = choose_resource_plan(
        _gpu_snapshot(free=2),
        require_gpu=False,
        qwen_requested=True,
    )
    assert not plan.fp16 and not plan.qwen_enabled


def test_cpu_is_only_allowed_when_gpu_is_not_required():
    cpu = {"gpu": {"available": False}, "host": {}}
    plan = choose_resource_plan(cpu, require_gpu=False, qwen_requested=True)
    assert not plan.fp16 and not plan.bf16 and not plan.qwen_enabled
    with pytest.raises(PipelineContractError) as caught:
        choose_resource_plan(cpu, require_gpu=True, qwen_requested=False)
    assert caught.value.code == "E_GPU_BUDGET"


@pytest.mark.parametrize(
    ("snapshot", "kernel_probe", "reason_fragment"),
    [
        (_gpu_snapshot(name="Tesla P100-PCIE-16GB", capability="6.0"), True, "P100"),
        (_gpu_snapshot(capability=None), True, "unknown"),
        (_gpu_snapshot(name="Tesla V100", capability="7.0"), True, "unsupported"),
        (_gpu_snapshot(), False, "kernel probe"),
        (_gpu_snapshot(), None, "kernel probe"),
    ],
)
def test_qwen_is_skipped_before_load_for_p100_unknown_unsupported_or_unprobed(
    snapshot, kernel_probe, reason_fragment
):
    plan = choose_resource_plan(
        snapshot,
        require_gpu=True,
        qwen_requested=True,
        kernel_probe=kernel_probe,
    )
    assert not plan.qwen_enabled
    assert plan.qwen_profile is None
    assert reason_fragment.casefold() in plan.qwen_disabled_reason.casefold()


def test_qwen_7b_requires_explicit_override_and_probed_t4():
    without_override = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        kernel_probe=True,
        qwen_model_id=QWEN_7B_MODEL_ID,
    )
    assert not without_override.qwen_enabled

    allowed = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        kernel_probe=True,
        qwen_model_id=QWEN_7B_MODEL_ID,
        allow_qwen_7b_override=True,
    )
    assert allowed.qwen_profile.model_id == QWEN_7B_MODEL_ID

    wrong_gpu = choose_resource_plan(
        _gpu_snapshot(name="NVIDIA A100", capability="8.0"),
        require_gpu=True,
        qwen_requested=True,
        kernel_probe=True,
        qwen_model_id=QWEN_7B_MODEL_ID,
        allow_qwen_7b_override=True,
    )
    assert not wrong_gpu.qwen_enabled
    assert "only" in wrong_gpu.qwen_disabled_reason


@pytest.mark.parametrize(
    ("ram", "disk", "expected_code"),
    [(None, 50, "E_RUNTIME_BUDGET"), (9.99, 50, "E_RUNTIME_BUDGET"), (20, None, "E_DISK_BUDGET"), (20, 14.99, "E_DISK_BUDGET")],
)
def test_training_admission_fails_closed_for_unknown_or_low_host_resources(ram, disk, expected_code):
    with pytest.raises(PipelineContractError) as caught:
        choose_resource_plan(
            _gpu_snapshot(ram=ram, disk=disk),
            require_gpu=True,
            qwen_requested=False,
        )
    assert caught.value.code == expected_code


def test_training_admission_thresholds_are_configurable():
    plan = choose_resource_plan(
        _gpu_snapshot(ram=9, disk=14),
        require_gpu=True,
        qwen_requested=False,
        min_host_ram_gib=8,
        min_disk_free_gib=12,
    )
    assert plan.train_batch_size == 2


def test_hardware_snapshot_keeps_core_gpu_data_when_optional_peak_probe_fails(monkeypatch):
    gib = 1024**3

    class FakeCuda:
        def is_available(self):
            return True

        def current_device(self):
            return 0

        def get_device_properties(self, device):
            return SimpleNamespace(name="Tesla T4")

        def get_device_capability(self, device):
            return (7, 5)

        def mem_get_info(self, device):
            return (13 * gib, 16 * gib)

        def memory_allocated(self, device):
            return 1 * gib

        def memory_reserved(self, device):
            return 2 * gib

        def max_memory_allocated(self, device):
            raise RuntimeError("optional counter unavailable")

    fake_torch = SimpleNamespace(cuda=FakeCuda())

    def fake_import(name):
        if name == "torch":
            return fake_torch
        raise ImportError(name)

    monkeypatch.setattr(runtime_control.importlib, "import_module", fake_import)
    snapshot = hardware_snapshot()
    assert snapshot["gpu"] == {
        "available": True,
        "name": "Tesla T4",
        "capability": "7.5",
        "free_gib": 13.0,
        "total_gib": 16.0,
        "allocated_gib": 1.0,
        "reserved_gib": 2.0,
        "peak_gib": None,
    }


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
        [
            _item("encoder", 500_000_000, artifact_hash="A" * 64, role="reranking"),
            _item("encoder", 500_000_000, artifact_hash="a" * 64, role="assertion"),
        ]
    )
    assert report["total_parameters"] == 500_000_000
    assert len(report["unique_items"]) == 1
    assert report["unique_items"][0]["artifact_hash"] == "a" * 64
    assert report["remaining"] == 8_500_000_000
    assert not report["warning"]


@pytest.mark.parametrize(("source", "artifact_hash"), [("", "a" * 64), ("dataset", "")])
def test_model_budget_requires_source_and_hash(source, artifact_hash):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("encoder", 10, source=source, artifact_hash=artifact_hash)])
    assert caught.value.code == "E_MODEL_INVENTORY_INVALID"


@pytest.mark.parametrize("artifact_hash", ["short", "g" * 64, "a" * 63])
def test_model_budget_rejects_non_sha256_artifact_hash(artifact_hash):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("encoder", 10, artifact_hash=artifact_hash)])
    assert caught.value.code == "E_MODEL_INVENTORY_INVALID"


@pytest.mark.parametrize("reverse", [False, True])
def test_model_budget_rejects_contradictory_duplicate_sizes_regardless_of_order(reverse):
    items = [
        _item("encoder", 1, artifact_hash="b" * 64),
        _item("encoder", 9_000_000_001, artifact_hash="b" * 64),
    ]
    if reverse:
        items.reverse()
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget(items)
    assert caught.value.code == "E_MODEL_INVENTORY_CONFLICT"


def test_model_budget_rejects_same_identity_with_conflicting_hashes():
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget(
            [_item("encoder", 10, artifact_hash="c" * 64), _item("encoder", 10, artifact_hash="d" * 64)]
        )
    assert caught.value.code == "E_MODEL_INVENTORY_CONFLICT"


@pytest.mark.parametrize("count", [None, 0, -1])
def test_model_budget_rejects_unknown_or_invalid_count(count):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("unknown", count)])
    assert caught.value.code == "E_MODEL_SIZE_UNKNOWN"


def test_model_budget_rejects_more_than_9b():
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("too-large", 9_000_000_001)])
    assert caught.value.code == "E_MODEL_OVER_9B"


def test_strict_override_never_falls_back_and_reports_decision(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    missing = tmp_path / "missing"
    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("input", missing, [valid], lambda path: path.is_dir())
    assert caught.value.code == "E_INPUT_OVERRIDE_INVALID"
    assert caught.value.context["decisions"] == [
        {"path": str(missing.resolve()), "accepted": False, "reason": "missing"}
    ]


def test_unique_source_returns_deterministic_accept_reject_decisions(tmp_path):
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    valid.mkdir()
    invalid.write_text("not a directory", encoding="utf-8")
    resolution = resolve_unique_source(
        "model",
        None,
        [valid, invalid, valid],
        lambda path: path.is_dir(),
        return_decisions=True,
    )
    assert isinstance(resolution, SourceResolution)
    assert resolution.selected == valid.resolve()
    assert [(item.path.name, item.accepted, item.reason) for item in resolution.decisions] == [
        ("invalid", False, "validator_rejected"),
        ("valid", True, "accepted"),
        ("valid", False, "duplicate_path"),
    ]
    assert resolve_unique_source("model", "", [valid, valid], lambda path: path.is_dir()) == valid.resolve()


def test_unique_source_ambiguity_is_sorted_case_insensitively_with_tiebreaker(tmp_path):
    second = tmp_path / "z-source"
    first = tmp_path / "A-source"
    first.mkdir()
    second.mkdir()
    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("kb", None, [second, first], lambda path: path.is_dir())
    assert caught.value.code == "E_KB_AMBIGUOUS"
    assert caught.value.context["candidates"] == [str(first.resolve()), str(second.resolve())]


def test_source_identity_does_not_casefold_distinct_linux_paths(tmp_path, monkeypatch):
    upper = (tmp_path / "Source").resolve()
    lower = (tmp_path / "source").resolve()
    monkeypatch.setattr(runtime_control.os.path, "normcase", lambda value: value)
    monkeypatch.setattr(runtime_control.Path, "exists", lambda self: True)

    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("input", None, [lower, upper], lambda path: True)
    assert caught.value.code == "E_INPUT_AMBIGUOUS"
    assert caught.value.context["candidates"] == sorted(
        [str(upper), str(lower)], key=lambda value: (value.casefold(), value)
    )


def test_source_resolver_emits_auditable_decisions(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    valid = tmp_path / "valid"
    invalid = tmp_path / "invalid"
    valid.mkdir()
    invalid.write_text("not a directory", encoding="utf-8")
    logger = RuntimeEventLogger("run-source", tmp_path / "events.jsonl")

    selected = resolve_unique_source(
        "model",
        None,
        [invalid, valid],
        lambda path: path.is_dir(),
        logger=logger,
    )

    assert selected == valid.resolve()
    record = _records(logger.jsonl_path)[0]
    assert record["event"] == "SOURCE_RESOLVED"
    assert record["context"]["selected"] == str(valid.resolve())
    assert record["context"]["decisions"] == [
        {"path": str(invalid.resolve()), "accepted": False, "reason": "validator_rejected"},
        {"path": str(valid.resolve()), "accepted": True, "reason": "accepted"},
    ]
