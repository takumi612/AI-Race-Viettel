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


def _valid_qwen_probe(
    *,
    capability: tuple[int, int] = (7, 5),
    kernel_supported: bool = True,
):
    return runtime_control.QwenRuntimeProbe.from_evidence(
        runtime_engine=runtime_control.SafeIdentifier("vllm"),
        runtime_version=runtime_control.SafeIdentifier("v0.6.0"),
        awq_package=runtime_control.SafeIdentifier("awq"),
        awq_version=runtime_control.SafeIdentifier("v0.2.6"),
        cuda_version=runtime_control.SafeIdentifier("v12.1"),
        cuda_available=True,
        capability=capability,
        kernel_supported=kernel_supported,
        kernel_probe_id=runtime_control.SafeIdentifier("awq-kernel-v1"),
    )


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
    assert records[0]["context"]["documents"] == 3
    assert list(records[0]["context"].values()).count("[REDACTED]") == 4
    serialized = events.read_text(encoding="utf-8") + capsys.readouterr().out
    for unsafe_text in (
        "raw_text",
        "patientNote",
        "access_token",
        "unapproved_field",
        "do not log",
        "sensitive note",
        "credential",
    ):
        assert unsafe_text not in serialized


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
    assert unsupported.value.code == "E_EVENT_SCHEMA"

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
        qwen_runtime_probe=_valid_qwen_probe(),
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
    ("snapshot", "runtime_probe", "reason_fragment"),
    [
        (_gpu_snapshot(name="Tesla P100-PCIE-16GB", capability="6.0"), _valid_qwen_probe(), "P100"),
        (_gpu_snapshot(capability=None), _valid_qwen_probe(), "unknown"),
        (_gpu_snapshot(name="Tesla V100", capability="7.0"), _valid_qwen_probe(), "unsupported"),
        (_gpu_snapshot(), _valid_qwen_probe(kernel_supported=False), "kernel probe"),
        (_gpu_snapshot(), None, "probe"),
    ],
)
def test_qwen_is_skipped_before_load_for_p100_unknown_unsupported_or_unprobed(
    snapshot, runtime_probe, reason_fragment
):
    plan = choose_resource_plan(
        snapshot,
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=runtime_probe,
    )
    assert not plan.qwen_enabled
    assert plan.qwen_profile is None
    assert reason_fragment.casefold() in plan.qwen_disabled_reason.casefold()


def test_qwen_7b_requires_explicit_override_and_probed_t4():
    without_override = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=_valid_qwen_probe(),
        qwen_model_id=QWEN_7B_MODEL_ID,
    )
    assert not without_override.qwen_enabled

    allowed = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=_valid_qwen_probe(),
        qwen_model_id=QWEN_7B_MODEL_ID,
        allow_qwen_7b_override=True,
    )
    assert allowed.qwen_profile.model_id == QWEN_7B_MODEL_ID

    wrong_gpu = choose_resource_plan(
        _gpu_snapshot(name="NVIDIA A100", capability="8.0"),
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=_valid_qwen_probe(capability=(8, 0)),
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


@pytest.mark.parametrize("active", [0, 1, "yes", None])
def test_model_inventory_active_flag_must_be_exact_boolean(active):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("invalid-active", 5, active=active)])
    assert caught.value.code == "E_MODEL_INVENTORY_ACTIVE"


def test_strict_override_never_falls_back_and_reports_decision(tmp_path):
    valid = tmp_path / "valid"
    valid.mkdir()
    missing = tmp_path / "missing"
    with pytest.raises(PipelineContractError) as caught:
        resolve_unique_source("input", missing, [valid], lambda path: path.is_dir())
    assert caught.value.code == "E_INPUT_OVERRIDE_INVALID"
    decision = caught.value.context["decisions"][0]
    assert decision["accepted"] is False
    assert decision["reason"] is runtime_control.EventReason.SOURCE_MISSING
    assert isinstance(decision["path"], runtime_control.SafePathRef)
    assert str(missing.resolve()) not in json.dumps(
        runtime_control._safe_context_value(caught.value.context),
        allow_nan=False,
    )


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
        ("invalid", False, runtime_control.EventReason.SOURCE_VALIDATOR_REJECTED),
        ("valid", True, runtime_control.EventReason.SOURCE_ACCEPTED),
        ("valid", False, runtime_control.EventReason.SOURCE_DUPLICATE_PATH),
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

    with pytest.raises(PipelineContractError) as orphan:
        resolve_unique_source(
            "model",
            None,
            [invalid, valid],
            lambda path: path.is_dir(),
            logger=logger,
            trusted_roots={"dataset": tmp_path},
        )
    assert orphan.value.code == "E_EVENT_SEQUENCE"

    with logger.phase("source_resolution", attempt=1):
        selected = resolve_unique_source(
            "model",
            None,
            [invalid, valid],
            lambda path: path.is_dir(),
            logger=logger,
            trusted_roots={"dataset": tmp_path},
        )
    logger.aggregate_terminal("source_resolution", succeeded=True)
    logger.finalize()

    assert selected == valid.resolve()
    record = next(item for item in _records(logger.jsonl_path) if item["event"] == "SOURCE_RESOLVED")
    assert record["event"] == "SOURCE_RESOLVED"
    assert record["context"]["selected"]["root_alias"] == "dataset"
    assert [item["reason"] for item in record["context"]["decisions"]] == [
        runtime_control.EventReason.SOURCE_VALIDATOR_REJECTED.value,
        runtime_control.EventReason.SOURCE_ACCEPTED.value,
    ]
    assert str(tmp_path) not in logger.jsonl_path.read_text(encoding="utf-8")


def test_event_context_plain_strings_never_cross_logging_boundary(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-safe-values", tmp_path / "events.jsonl")
    secrets = {
        "reason": "Bearer secret-123 raw clinical note",
        "path": "C:/patients/Alice/diagnosis.txt",
        "source": "patient HIV status",
        "selected": "token-value",
        "candidates": ["clinical sentence", {"model_id": "secret-model-value"}],
        "request_id": "patient-name-as-id",
        "Alice_diagnosis": {"bearer-token-key": 7},
    }
    with logger.phase("privacy", attempt=1):
        logger.emit("privacy", "MEMORY_SNAPSHOT", "RUNNING", attempt=1, context=secrets)
    logger.aggregate_terminal("privacy", succeeded=True)

    serialized = logger.jsonl_path.read_text(encoding="utf-8")
    for secret in (
        "secret-123",
        "Alice",
        "HIV",
        "token-value",
        "clinical sentence",
        "secret-model-value",
        "patient-name-as-id",
        "Alice_diagnosis",
        "bearer-token-key",
    ):
        assert secret not in serialized


def test_event_context_accepts_only_typed_safe_string_values(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-typed-values", tmp_path / "events.jsonl")
    safe_identifier = runtime_control.SafeIdentifier("source-01")
    safe_hash = runtime_control.SafeHash("a" * 64)
    safe_path = runtime_control.SafePathRef("input", "src-0123456789abcdef", "b" * 64)
    with logger.phase("privacy", attempt=1):
        logger.emit(
            "privacy",
            "MEMORY_SNAPSHOT",
            "RUNNING",
            attempt=1,
            context={
                "source_id": safe_identifier,
                "artifact_hash": safe_hash,
                "reason": runtime_control.EventReason.SOURCE_VALIDATOR_REJECTED,
                "path": safe_path,
            },
        )
    logger.aggregate_terminal("privacy", succeeded=True)

    context = _records(logger.jsonl_path)[1]["context"]
    assert context == {
        "source_id": "source-01",
        "artifact_hash": "a" * 64,
        "reason": "SOURCE_VALIDATOR_REJECTED",
        "path": {
            "root_alias": "input",
            "source_id": "src-0123456789abcdef",
            "path_hash": "b" * 64,
        },
    }


@pytest.mark.parametrize("value", ["secret-123", "bearer-token", "patient-alice", "clinical-note"])
def test_safe_identifier_rejects_sensitive_looking_values(value):
    with pytest.raises(ValueError):
        runtime_control.SafeIdentifier(value)


def test_runtime_control_api_version_has_serializable_migration_report():
    assert runtime_control.RUNTIME_CONTROL_API_VERSION == "3.0"
    report = runtime_control.runtime_control_migration_report()
    assert report["api_version"] == "3.0"
    assert "typed_event_values" in report["breaking_changes"]
    assert "qwen_runtime_probe" in report["breaking_changes"]
    json.dumps(report, allow_nan=False)


def test_lifecycle_rejects_orphan_ordinary_attempt_event(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-orphan", tmp_path / "events.jsonl")
    with pytest.raises(PipelineContractError) as caught:
        logger.emit("train", "MEMORY_SNAPSHOT", "RUNNING", attempt=1)
    assert caught.value.code == "E_EVENT_SEQUENCE"


@pytest.mark.parametrize(
    ("event", "status", "error"),
    [
        ("PHASE_ERROR", "SUCCESS", PipelineContractError("E_TEST", "hidden")),
        ("PHASE_ERROR", "ERROR", None),
        ("PHASE_END", "ERROR", None),
        ("PHASE_END", "SUCCESS", PipelineContractError("E_TEST", "hidden")),
    ],
)
def test_lifecycle_rejects_contradictory_terminal_matrix(event, status, error, tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-matrix", tmp_path / "events.jsonl")
    logger.emit("train", "PHASE_START", "RUNNING", attempt=1)
    with pytest.raises(PipelineContractError) as caught:
        logger.emit("train", event, status, attempt=1, error=error)
    assert caught.value.code == "E_EVENT_SCHEMA"


def test_lifecycle_rejects_aggregate_while_retry_target_is_pending(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-pending", tmp_path / "events.jsonl")
    retryable = PipelineContractError("E_TRAIN_CUDA_OOM", "hidden", retriable=True)
    with pytest.raises(PipelineContractError):
        with logger.phase("train", attempt=1):
            raise retryable
    logger.emit_oom_retry("train", from_attempt=1, to_attempt=2)
    with pytest.raises(PipelineContractError) as caught:
        logger.aggregate_terminal("train", succeeded=False, error=retryable)
    assert caught.value.code == "E_EVENT_SEQUENCE"


def test_finalize_requires_aggregate_and_prevents_reuse(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-finalize", tmp_path / "events.jsonl")
    with logger.phase("train", attempt=1):
        pass
    with pytest.raises(PipelineContractError) as missing:
        logger.finalize()
    assert missing.value.code == "E_EVENT_SEQUENCE"
    logger.aggregate_terminal("train", succeeded=True)
    summary = logger.finalize()
    assert summary == {"logical_phases": 1, "aggregate_terminals": 1}
    with pytest.raises(PipelineContractError):
        logger.emit("other", "PHASE_START", "RUNNING", attempt=1)


def test_logger_rejects_restart_on_nonempty_jsonl(tmp_path, monkeypatch):
    _stub_hardware(monkeypatch)
    path = tmp_path / "events.jsonl"
    logger = RuntimeEventLogger("run-first", path)
    with logger.phase("train", attempt=1):
        pass
    logger.aggregate_terminal("train", succeeded=True)
    logger.finalize()
    with pytest.raises(PipelineContractError) as caught:
        RuntimeEventLogger("run-second", path)
    assert caught.value.code == "E_EVENT_LOG_EXISTS"


def test_source_role_requirement_matrix_covers_every_role_and_mode():
    required = runtime_control.SourceRequirement.REQUIRED
    optional = runtime_control.SourceRequirement.OPTIONAL
    forbidden = runtime_control.SourceRequirement.FORBIDDEN
    expected = {
        runtime_control.RunMode.FULL: {
            runtime_control.SourceRole.INFERENCE_INPUT: required,
            runtime_control.SourceRole.TRAIN_CORPUS: required,
            runtime_control.SourceRole.RUNTIME_KB: required,
            runtime_control.SourceRole.NER_BASE: required,
            runtime_control.SourceRole.FINAL_MODEL_ARTIFACT: forbidden,
            runtime_control.SourceRole.EMBEDDING_MODEL: optional,
            runtime_control.SourceRole.QWEN_MODEL: optional,
            runtime_control.SourceRole.WHEELHOUSE: forbidden,
            runtime_control.SourceRole.RESUME_BUNDLE: forbidden,
        },
        runtime_control.RunMode.RESUME: {
            runtime_control.SourceRole.INFERENCE_INPUT: required,
            runtime_control.SourceRole.TRAIN_CORPUS: required,
            runtime_control.SourceRole.RUNTIME_KB: required,
            runtime_control.SourceRole.NER_BASE: optional,
            runtime_control.SourceRole.FINAL_MODEL_ARTIFACT: optional,
            runtime_control.SourceRole.EMBEDDING_MODEL: optional,
            runtime_control.SourceRole.QWEN_MODEL: optional,
            runtime_control.SourceRole.WHEELHOUSE: forbidden,
            runtime_control.SourceRole.RESUME_BUNDLE: required,
        },
        runtime_control.RunMode.INFERENCE_ONLY: {
            runtime_control.SourceRole.INFERENCE_INPUT: required,
            runtime_control.SourceRole.TRAIN_CORPUS: forbidden,
            runtime_control.SourceRole.RUNTIME_KB: required,
            runtime_control.SourceRole.NER_BASE: forbidden,
            runtime_control.SourceRole.FINAL_MODEL_ARTIFACT: required,
            runtime_control.SourceRole.EMBEDDING_MODEL: optional,
            runtime_control.SourceRole.QWEN_MODEL: optional,
            runtime_control.SourceRole.WHEELHOUSE: forbidden,
            runtime_control.SourceRole.RESUME_BUNDLE: forbidden,
        },
    }
    for mode, role_expectations in expected.items():
        for role, requirement in role_expectations.items():
            assert runtime_control.source_role_requirement(
                mode,
                role,
                install_mode=runtime_control.InstallMode.PREINSTALLED,
                resume_bundle_has_ner_base=True,
                final_artifact_self_contained=True,
            ) is requirement


def test_source_role_requirement_conditionals():
    required = runtime_control.SourceRequirement.REQUIRED
    assert runtime_control.source_role_requirement(
        runtime_control.RunMode.RESUME,
        runtime_control.SourceRole.NER_BASE,
        install_mode=runtime_control.InstallMode.PREINSTALLED,
        resume_bundle_has_ner_base=False,
        final_artifact_self_contained=True,
    ) is required
    assert runtime_control.source_role_requirement(
        runtime_control.RunMode.INFERENCE_ONLY,
        runtime_control.SourceRole.NER_BASE,
        install_mode=runtime_control.InstallMode.PREINSTALLED,
        resume_bundle_has_ner_base=True,
        final_artifact_self_contained=False,
    ) is required
    for mode in runtime_control.RunMode:
        assert runtime_control.source_role_requirement(
            mode,
            runtime_control.SourceRole.WHEELHOUSE,
            install_mode=runtime_control.InstallMode.OFFLINE_WHEELHOUSE,
            resume_bundle_has_ner_base=True,
            final_artifact_self_contained=True,
        ) is required


def _full_required_role_inputs(tmp_path):
    candidates = {}
    validators = {}
    for role in (
        runtime_control.SourceRole.INFERENCE_INPUT,
        runtime_control.SourceRole.TRAIN_CORPUS,
        runtime_control.SourceRole.RUNTIME_KB,
        runtime_control.SourceRole.NER_BASE,
    ):
        path = tmp_path / role.value.casefold()
        path.mkdir(exist_ok=True)
        candidates[role] = [path]
        validators[role] = lambda candidate: candidate.is_dir()
    return candidates, validators


def test_mode_aware_source_resolver_enforces_matrix_and_returns_typed_sorted_decisions(tmp_path):
    required_roles = (
        runtime_control.SourceRole.INFERENCE_INPUT,
        runtime_control.SourceRole.TRAIN_CORPUS,
        runtime_control.SourceRole.RUNTIME_KB,
        runtime_control.SourceRole.NER_BASE,
    )
    candidates = {}
    validators = {}
    for index, role in enumerate(reversed(required_roles)):
        path = tmp_path / f"source-{index}"
        path.mkdir()
        candidates[role] = [path]
        validators[role] = lambda candidate: candidate.is_dir()

    result = runtime_control.resolve_source_roles(
        runtime_control.RunMode.FULL,
        overrides={},
        candidates=candidates,
        validators=validators,
        install_mode=runtime_control.InstallMode.PREINSTALLED,
        resume_bundle_has_ner_base=False,
        final_artifact_self_contained=False,
        trusted_roots={"dataset": tmp_path},
    )

    assert isinstance(result, runtime_control.SourceRoleResolution)
    assert tuple(item.role for item in result.roles) == tuple(runtime_control.SourceRole)
    assert all(result.for_role(role).selected is not None for role in required_roles)
    event_payload = result.as_dict()
    serialized = json.dumps(event_payload, allow_nan=False)
    assert str(tmp_path) not in serialized
    assert event_payload["roles"][0]["decisions"][0]["path"]["root_alias"] == "dataset"


def test_mode_aware_source_resolver_rejects_forbidden_optional_many_and_strict_override(tmp_path):
    forbidden = tmp_path / "forbidden"
    forbidden.mkdir()
    with pytest.raises(PipelineContractError) as forbidden_error:
        runtime_control.resolve_source_roles(
            runtime_control.RunMode.INFERENCE_ONLY,
            overrides={},
            candidates={runtime_control.SourceRole.TRAIN_CORPUS: [forbidden]},
            validators={},
            install_mode=runtime_control.InstallMode.PREINSTALLED,
            resume_bundle_has_ner_base=True,
            final_artifact_self_contained=True,
            trusted_roots={"dataset": tmp_path},
        )
    assert forbidden_error.value.code == "E_SOURCE_ROLE_FORBIDDEN"

    first = tmp_path / "embed-a"
    second = tmp_path / "embed-b"
    first.mkdir()
    second.mkdir()
    required_candidates, required_validators = _full_required_role_inputs(tmp_path)
    required_candidates[runtime_control.SourceRole.EMBEDDING_MODEL] = [second, first]
    required_validators[runtime_control.SourceRole.EMBEDDING_MODEL] = lambda path: path.is_dir()
    with pytest.raises(PipelineContractError) as ambiguous:
        runtime_control.resolve_source_roles(
            runtime_control.RunMode.FULL,
            overrides={},
            candidates=required_candidates,
            validators=required_validators,
            install_mode=runtime_control.InstallMode.PREINSTALLED,
            resume_bundle_has_ner_base=True,
            final_artifact_self_contained=True,
            trusted_roots={"dataset": tmp_path},
        )
    assert ambiguous.value.code == "E_EMBEDDING_MODEL_AMBIGUOUS"

    missing = tmp_path / "missing"
    with pytest.raises(PipelineContractError) as strict:
        runtime_control.resolve_source_roles(
            runtime_control.RunMode.FULL,
            overrides={runtime_control.SourceRole.INFERENCE_INPUT: missing},
            candidates={runtime_control.SourceRole.INFERENCE_INPUT: [forbidden]},
            validators={runtime_control.SourceRole.INFERENCE_INPUT: lambda path: path.is_dir()},
            install_mode=runtime_control.InstallMode.PREINSTALLED,
            resume_bundle_has_ner_base=True,
            final_artifact_self_contained=True,
            trusted_roots={"dataset": tmp_path},
        )
    assert strict.value.code == "E_INFERENCE_INPUT_OVERRIDE_INVALID"


def test_mode_aware_source_resolver_rejects_cross_role_canonical_path_collision(tmp_path):
    shared = tmp_path / "shared"
    shared.mkdir()
    candidates, validators = _full_required_role_inputs(tmp_path)
    candidates[runtime_control.SourceRole.INFERENCE_INPUT] = [shared]
    candidates[runtime_control.SourceRole.RUNTIME_KB] = [shared]
    with pytest.raises(PipelineContractError) as caught:
        runtime_control.resolve_source_roles(
            runtime_control.RunMode.FULL,
            overrides={},
            candidates=candidates,
            validators=validators,
            install_mode=runtime_control.InstallMode.PREINSTALLED,
            resume_bundle_has_ner_base=True,
            final_artifact_self_contained=True,
            trusted_roots={"dataset": tmp_path},
        )
    assert caught.value.code == "E_SOURCE_ROLE_COLLISION"


def _stage_resource_input(**overrides):
    values = {
        "stage_id": runtime_control.SafeIdentifier("stage-1"),
        "train_chunks": 10,
        "eval_chunks": 5,
        "inference_chunks": 9,
        "epochs": 2,
        "primary_train_batch_size": 2,
        "primary_eval_batch_size": 2,
        "primary_inference_batch_size": 3,
        "primary_gradient_accumulation_steps": 4,
        "retry_train_batch_size": 1,
        "retry_eval_batch_size": 1,
        "retry_inference_batch_size": 1,
        "retry_gradient_accumulation_steps": 8,
        "configured_train_seconds_per_step": 2.0,
        "configured_eval_seconds_per_step": 1.0,
        "configured_inference_seconds_per_step": 0.5,
        "measured_train_seconds_per_step": 1.0,
        "measured_eval_seconds_per_step": None,
        "measured_inference_seconds_per_step": 0.25,
        "checkpoint_bytes": 100,
        "artifact_bytes": 40,
        "workspace_bytes": 60,
    }
    values.update(overrides)
    return runtime_control.StageResourceInput(**values)


def test_resource_estimator_reports_steps_retry_margin_and_disk_deterministically():
    estimate = runtime_control.estimate_resource_budget(
        [_stage_resource_input()],
        declared_runtime_limit_seconds=63.7,
        usable_disk_quota_bytes=250,
    )
    stage = estimate.stages[0]
    assert stage.primary_train_micro_steps == 10
    assert stage.primary_optimizer_steps == 3
    assert stage.primary_eval_steps == 6
    assert stage.primary_inference_steps == 3
    assert stage.retry_train_micro_steps == 20
    assert stage.retry_optimizer_steps == 3
    assert stage.retry_eval_steps == 10
    assert stage.retry_inference_steps == 9
    assert stage.primary_seconds == pytest.approx(16.75)
    assert stage.retry_seconds == pytest.approx(32.25)
    assert stage.worst_case_seconds == pytest.approx(49.0)
    assert estimate.safety_adjusted_total_seconds == pytest.approx(63.7)
    assert estimate.peak_disk_bytes == 200
    assert estimate.remaining_disk_bytes == 50
    assert json.loads(json.dumps(estimate.as_dict(), allow_nan=False))["stages"][0]["stage_id"] == "stage-1"


def test_resource_estimator_sorts_stages_and_fails_runtime_or_disk_before_work():
    first = _stage_resource_input(stage_id=runtime_control.SafeIdentifier("stage-a"))
    second = _stage_resource_input(stage_id=runtime_control.SafeIdentifier("stage-b"))
    ordered = runtime_control.estimate_resource_budget(
        [second, first],
        declared_runtime_limit_seconds=200,
        usable_disk_quota_bytes=400,
    )
    assert [stage.stage_id for stage in ordered.stages] == ["stage-a", "stage-b"]

    with pytest.raises(PipelineContractError) as runtime_error:
        runtime_control.estimate_resource_budget(
            [first],
            declared_runtime_limit_seconds=63.699,
            usable_disk_quota_bytes=250,
        )
    assert runtime_error.value.code == "E_RUNTIME_BUDGET"

    with pytest.raises(PipelineContractError) as disk_error:
        runtime_control.estimate_resource_budget(
            [first],
            declared_runtime_limit_seconds=100,
            usable_disk_quota_bytes=199,
        )
    assert disk_error.value.code == "E_DISK_BUDGET"


@pytest.mark.parametrize(
    "overrides",
    [
        {"train_chunks": True},
        {"eval_chunks": 1.5},
        {"epochs": 0},
        {"primary_train_batch_size": 0},
        {"retry_gradient_accumulation_steps": "8"},
        {"configured_train_seconds_per_step": math.nan},
        {"measured_eval_seconds_per_step": math.inf},
        {"checkpoint_bytes": -1},
    ],
)
def test_resource_estimator_rejects_invalid_stage_numeric_inputs(overrides):
    with pytest.raises(PipelineContractError) as caught:
        runtime_control.estimate_resource_budget(
            [_stage_resource_input(**overrides)],
            declared_runtime_limit_seconds=100,
            usable_disk_quota_bytes=250,
        )
    assert caught.value.code == "E_RESOURCE_INPUT_INVALID"


@pytest.mark.parametrize(
    ("runtime_limit", "disk_quota"),
    [(True, 250), (math.nan, 250), ("100", 250), (100, False), (100, 1.5), (100, 0)],
)
def test_resource_estimator_rejects_invalid_declared_budgets(runtime_limit, disk_quota):
    with pytest.raises(PipelineContractError) as caught:
        runtime_control.estimate_resource_budget(
            [_stage_resource_input()],
            declared_runtime_limit_seconds=runtime_limit,
            usable_disk_quota_bytes=disk_quota,
        )
    assert caught.value.code == "E_RESOURCE_INPUT_INVALID"


@pytest.mark.parametrize(
    "count",
    [True, False, 1.5, 9_000_000_000.9, math.nan, math.inf, "100", [100]],
)
def test_model_budget_requires_positive_exact_integer_parameter_count(count):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("invalid-count", count)])
    assert caught.value.code == "E_MODEL_SIZE_UNKNOWN"


@pytest.mark.parametrize(
    ("limit", "warning"),
    [
        (True, 5),
        (10.0, 5),
        (math.nan, 5),
        (math.inf, 5),
        ("10", 5),
        (10, False),
        (10, 5.0),
        (10, 0),
        (10, 11),
    ],
)
def test_model_budget_requires_exact_integer_limit_and_warning(limit, warning):
    with pytest.raises(PipelineContractError) as caught:
        validate_model_budget([_item("valid", 5)], limit=limit, warning=warning)
    assert caught.value.code == "E_MODEL_BUDGET_CONFIG"


def test_lazy_qwen_runtime_probe_collects_concrete_evidence_without_model_load():
    fake_runtime = SimpleNamespace(__version__="0.6.0")
    fake_awq = SimpleNamespace(__version__="0.2.6")

    class FakeCuda:
        def is_available(self):
            return True

        def get_device_capability(self):
            return (7, 5)

    fake_torch = SimpleNamespace(__version__="2.4.0", version=SimpleNamespace(cuda="12.1"), cuda=FakeCuda())

    def importer(name):
        return {"vllm": fake_runtime, "awq": fake_awq, "torch": fake_torch}[name]

    probe = runtime_control.probe_qwen_runtime(
        importer=importer,
        kernel_probe=lambda runtime, awq, torch, capability: capability == (7, 5),
        kernel_probe_id=runtime_control.SafeIdentifier("awq-kernel-v1"),
    )
    assert probe.runtime_engine == runtime_control.SafeIdentifier("vllm")
    assert probe.awq_package == runtime_control.SafeIdentifier("awq")
    assert probe.cuda_available and probe.capability == (7, 5) and probe.kernel_supported
    assert len(probe.evidence_hash.value) == 64
    with pytest.raises(FrozenInstanceError):
        probe.kernel_supported = False


def test_qwen_runtime_probe_rejects_tampered_hash_and_admission_rejects_tampered_field():
    valid = _valid_qwen_probe()
    with pytest.raises(ValueError, match="evidence hash"):
        runtime_control.QwenRuntimeProbe(
            runtime_engine=valid.runtime_engine,
            runtime_version=valid.runtime_version,
            awq_package=valid.awq_package,
            awq_version=valid.awq_version,
            cuda_version=valid.cuda_version,
            cuda_available=valid.cuda_available,
            capability=valid.capability,
            kernel_supported=valid.kernel_supported,
            kernel_probe_id=valid.kernel_probe_id,
            evidence_hash=runtime_control.SafeHash("f" * 64),
        )

    object.__setattr__(valid, "kernel_supported", False)
    rejected = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=valid,
    )
    assert not rejected.qwen_enabled
    assert "integrity" in rejected.qwen_disabled_reason.casefold()


def test_qwen_admission_requires_typed_concrete_probe_and_timeout_profile():
    disabled = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=None,
    )
    assert not disabled.qwen_enabled
    assert "probe" in disabled.qwen_disabled_reason.casefold()

    enabled = choose_resource_plan(
        _gpu_snapshot(),
        require_gpu=True,
        qwen_requested=True,
        qwen_runtime_probe=_valid_qwen_probe(),
    )
    assert enabled.qwen_enabled
    assert enabled.qwen_profile.engine_init_timeout_seconds == 120.0
    assert enabled.qwen_profile.generation_timeout_seconds == 300.0

    with pytest.raises(TypeError):
        choose_resource_plan(
            _gpu_snapshot(),
            require_gpu=True,
            qwen_requested=True,
            kernel_probe=True,
        )


def _execute_qwen_case(tmp_path, monkeypatch, *, initialize, generate, parse, cleanup, runner):
    _stub_hardware(monkeypatch)
    logger = RuntimeEventLogger("run-qwen", tmp_path / "events.jsonl")
    baseline = {"deterministic": [1, 2, 3]}
    with logger.phase("qwen.optional", attempt=1):
        result = runtime_control.execute_optional_qwen(
            deterministic_result=baseline,
            profile=runtime_control.QwenProfile(),
            logger=logger,
            phase="qwen.optional",
            attempt=1,
            process_timeout_runner=runner,
            initialize=initialize,
            generate=generate,
            parse=parse,
            cleanup=cleanup,
        )
    logger.aggregate_terminal("qwen.optional", succeeded=True)
    logger.finalize()
    return baseline, result, _records(logger.jsonl_path)


def test_optional_qwen_success_uses_process_timeout_runner_and_returns_refinement(tmp_path, monkeypatch):
    calls = []

    def runner(operation, timeout_seconds):
        calls.append(timeout_seconds)
        return operation()

    baseline, result, records = _execute_qwen_case(
        tmp_path,
        monkeypatch,
        initialize=lambda profile: {"engine": profile.model_id},
        generate=lambda engine: {"raw": engine["engine"]},
        parse=lambda raw: {"refined": raw["raw"]},
        cleanup=lambda engine: None,
        runner=runner,
    )
    assert result == {"refined": DEFAULT_QWEN_MODEL_ID}
    assert result is not baseline
    assert calls == [120.0, 300.0]
    assert "OPTIONAL_FALLBACK" not in [record["event"] for record in records]


@pytest.mark.parametrize(
    "failure_case",
    ["init", "oom", "init_timeout", "generation_timeout", "parse", "cleanup"],
)
def test_optional_qwen_failures_preserve_same_deterministic_result_and_emit_closed_fallback(
    failure_case, tmp_path, monkeypatch
):
    calls = 0

    def initialize(profile):
        if failure_case == "init":
            raise RuntimeError("secret init detail")
        return {"engine": True}

    def generate(engine):
        if failure_case == "oom":
            raise runtime_control.QwenCudaOomError("secret cuda detail")
        return {"raw": True}

    def parse(raw):
        if failure_case == "parse":
            raise ValueError("clinical text parse detail")
        return {"refined": True}

    def cleanup(engine):
        if failure_case == "cleanup":
            raise RuntimeError("secret cleanup detail")

    def runner(operation, timeout_seconds):
        nonlocal calls
        calls += 1
        if failure_case == "init_timeout" and calls == 1:
            raise TimeoutError("thread timeout is not used")
        if failure_case == "generation_timeout" and calls == 2:
            raise TimeoutError("process timeout")
        return operation()

    baseline, result, records = _execute_qwen_case(
        tmp_path,
        monkeypatch,
        initialize=initialize,
        generate=generate,
        parse=parse,
        cleanup=cleanup,
        runner=runner,
    )
    assert result is baseline
    fallback = next(record for record in records if record["event"] == "OPTIONAL_FALLBACK")
    expected = {
        "init": "QWEN_INIT_FAILED",
        "oom": "QWEN_CUDA_OOM",
        "init_timeout": "QWEN_INIT_TIMEOUT",
        "generation_timeout": "QWEN_GENERATION_TIMEOUT",
        "parse": "QWEN_PARSE_FAILED",
        "cleanup": "QWEN_CLEANUP_FAILED",
    }[failure_case]
    assert fallback["context"]["reason"] == expected
    assert fallback["error"]["code"] == f"E_{expected}"
    serialized = json.dumps(records)
    assert "secret" not in serialized and "clinical text" not in serialized
