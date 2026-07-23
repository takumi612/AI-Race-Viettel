"""Resource-safe runtime controls for the Kaggle pipeline.

This module deliberately has no import-time dependency on torch, transformers, or
psutil.  It is safe to import in CPU-only validation and notebook preflight code.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


_EVENT_PREFIX = "[CLINICAL_PIPELINE] "
_GIB = 1024**3
_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PATTERN = re.compile(
    r"clinical|patient|document.*text|raw|text|token|secret|password|prompt|"
    r"credential|authorization|bearer|api.?key|content",
    re.IGNORECASE,
)
_SAFE_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")
_SAFE_CONTEXT_KEYS = {
    "accepted",
    "aggregate",
    "batch_ladder",
    "candidate",
    "candidates",
    "capability",
    "decisions",
    "documents",
    "eval_batch_size",
    "free_gib",
    "from_attempt",
    "gradient_accumulation_steps",
    "kind",
    "limit",
    "max_model_len",
    "min_disk_free_gib",
    "min_host_ram_gib",
    "model_id",
    "override",
    "path",
    "quantization",
    "reason",
    "remaining",
    "revision",
    "role",
    "roles",
    "root_alias",
    "run_mode",
    "selected",
    "source",
    "source_id",
    "total_gib",
    "total_parameters",
    "to_attempt",
    "train_batch_size",
    "warning_threshold",
    "path_hash",
    "requirement",
}
_SAFE_CONTEXT_SUFFIXES = (
    "_bytes",
    "_count",
    "_gib",
    "_hash",
    "_id",
    "_ms",
    "_revision",
    "_status",
)
_PROBE_EXCEPTIONS = (Exception,)

RUNTIME_CONTROL_API_VERSION = "3.0"
DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct-AWQ"
QWEN_7B_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"


class EventReason(str, Enum):
    SOURCE_ACCEPTED = "SOURCE_ACCEPTED"
    SOURCE_MISSING = "SOURCE_MISSING"
    SOURCE_VALIDATOR_REJECTED = "SOURCE_VALIDATOR_REJECTED"
    SOURCE_VALIDATOR_ERROR = "SOURCE_VALIDATOR_ERROR"
    SOURCE_DUPLICATE_PATH = "SOURCE_DUPLICATE_PATH"
    SOURCE_RESOLUTION_ERROR = "SOURCE_RESOLUTION_ERROR"
    SOURCE_ROLE_FORBIDDEN = "SOURCE_ROLE_FORBIDDEN"
    SOURCE_ROLE_COLLISION = "SOURCE_ROLE_COLLISION"
    QWEN_INIT_FAILED = "QWEN_INIT_FAILED"
    QWEN_CUDA_OOM = "QWEN_CUDA_OOM"
    QWEN_INIT_TIMEOUT = "QWEN_INIT_TIMEOUT"
    QWEN_GENERATION_TIMEOUT = "QWEN_GENERATION_TIMEOUT"
    QWEN_GENERATION_FAILED = "QWEN_GENERATION_FAILED"
    QWEN_PARSE_FAILED = "QWEN_PARSE_FAILED"
    QWEN_CLEANUP_FAILED = "QWEN_CLEANUP_FAILED"


@dataclass(frozen=True)
class SafeIdentifier:
    value: str

    def __post_init__(self) -> None:
        if not _SAFE_IDENTIFIER_PATTERN.fullmatch(self.value) or _SENSITIVE_KEY_PATTERN.search(self.value):
            raise ValueError("SafeIdentifier must be a short operational identifier")


@dataclass(frozen=True)
class SafeHash:
    value: str

    def __post_init__(self) -> None:
        normalized = self.value.casefold()
        if re.fullmatch(r"[0-9a-f]{64}", normalized) is None:
            raise ValueError("SafeHash must contain exactly 64 hexadecimal characters")
        object.__setattr__(self, "value", normalized)


@dataclass(frozen=True)
class SafePathRef:
    root_alias: str
    source_id: str
    path_hash: str

    def __post_init__(self) -> None:
        SafeIdentifier(self.root_alias)
        SafeIdentifier(self.source_id)
        normalized_hash = SafeHash(self.path_hash).value
        object.__setattr__(self, "path_hash", normalized_hash)


def runtime_control_migration_report() -> dict[str, Any]:
    """Return the machine-readable compatibility contract for this module revision."""

    return {
        "api_version": RUNTIME_CONTROL_API_VERSION,
        "breaking_changes": [
            "typed_event_values",
            "explicit_logger_finalize",
            "mode_aware_source_roles",
            "resource_budget_estimator",
            "exact_integer_model_counts",
            "qwen_runtime_probe",
        ],
        "compatibility": {
            "resolve_unique_source_default": "path",
            "qwen_kernel_probe_bool": "removed_use_QwenRuntimeProbe",
        },
    }


class PipelineContractError(RuntimeError):
    """An actionable, machine-readable pipeline contract violation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        context: dict[str, Any] | None = None,
        next_action: str = "",
        retriable: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.next_action = next_action
        self.retriable = bool(retriable)


def atomic_write_json(path: str | os.PathLike[str], payload: Any) -> Path:
    """Atomically publish a strict UTF-8 JSON file beside its destination."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return destination


def _fsync_directory(directory: Path) -> None:
    """Best-effort persistence for the directory entry published by os.replace."""

    flags = getattr(os, "O_RDONLY", 0)
    directory_fd: int | None = None
    try:
        directory_fd = os.open(directory, flags)
        os.fsync(directory_fd)
    except (AttributeError, OSError):
        # Windows and some filesystems do not allow opening/fsyncing a directory.
        pass
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _gib(value: int | float | None) -> float | None:
    if value is None:
        return None
    return float(value) / _GIB


def _empty_snapshot() -> dict[str, dict[str, Any]]:
    return {
        "gpu": {
            "available": False,
            "name": None,
            "capability": None,
            "free_gib": None,
            "total_gib": None,
            "allocated_gib": None,
            "reserved_gib": None,
            "peak_gib": None,
        },
        "host": {"ram_available_gib": None, "disk_free_gib": None},
    }


def hardware_snapshot() -> dict[str, dict[str, Any]]:
    """Return best-effort GPU, RAM, and disk measurements.

    Every field is present even when a platform API or optional dependency is not
    available.  Failures are represented by ``None`` rather than breaking import or
    preflight diagnostics.
    """

    snapshot = _empty_snapshot()
    gpu = snapshot["gpu"]
    host = snapshot["host"]

    torch: Any | None = None
    try:
        torch = importlib.import_module("torch")
    except _PROBE_EXCEPTIONS:
        pass

    if torch is not None:
        try:
            cuda = torch.cuda
            gpu["available"] = bool(cuda.is_available())
        except _PROBE_EXCEPTIONS:
            cuda = None

        if gpu["available"] and cuda is not None:
            try:
                device = int(cuda.current_device())
            except _PROBE_EXCEPTIONS:
                device = 0

            try:
                properties = cuda.get_device_properties(device)
                name = getattr(properties, "name", None)
                gpu["name"] = str(name) if name is not None else None
            except _PROBE_EXCEPTIONS:
                pass
            if gpu["name"] is None:
                try:
                    gpu["name"] = str(cuda.get_device_name(device))
                except _PROBE_EXCEPTIONS:
                    pass
            try:
                capability = cuda.get_device_capability(device)
                gpu["capability"] = ".".join(str(part) for part in capability)
            except _PROBE_EXCEPTIONS:
                pass
            try:
                free_bytes, total_bytes = cuda.mem_get_info(device)
                gpu["free_gib"] = _gib(free_bytes)
                gpu["total_gib"] = _gib(total_bytes)
            except _PROBE_EXCEPTIONS:
                pass
            for field_name, probe in (
                ("allocated_gib", cuda.memory_allocated),
                ("reserved_gib", cuda.memory_reserved),
                ("peak_gib", cuda.max_memory_allocated),
            ):
                try:
                    gpu[field_name] = _gib(probe(device))
                except _PROBE_EXCEPTIONS:
                    pass

    try:
        psutil = importlib.import_module("psutil")
        host["ram_available_gib"] = _gib(psutil.virtual_memory().available)
    except _PROBE_EXCEPTIONS:
        pass

    try:
        host["disk_free_gib"] = _gib(shutil.disk_usage(Path.cwd()).free)
    except _PROBE_EXCEPTIONS:
        pass
    return snapshot


def _safe_context_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else _REDACTED
    if isinstance(value, SafeIdentifier):
        return value.value
    if isinstance(value, SafeHash):
        return value.value
    if isinstance(value, EventReason):
        return value.value
    if isinstance(value, SafePathRef):
        return {
            "root_alias": value.root_alias,
            "source_id": value.source_id,
            "path_hash": value.path_hash,
        }
    if isinstance(value, str) or isinstance(value, Path):
        return _REDACTED
    if isinstance(value, Mapping):
        return {
            _safe_context_key(str(item_key)): _safe_context_value(item)
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_context_value(item) for item in value]
    return f"<{type(value).__name__}>"


def _safe_context_key(key: str) -> str:
    normalized = key.casefold()
    if (
        not _SENSITIVE_KEY_PATTERN.search(normalized)
        and (normalized in _SAFE_CONTEXT_KEYS or normalized.endswith(_SAFE_CONTEXT_SUFFIXES))
    ):
        return normalized
    digest = hashlib.sha256(key.encode("utf-8", errors="replace")).hexdigest()[:12]
    return f"field_{digest}"


def _safe_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        _safe_context_key(str(key)): _safe_context_value(value)
        for key, value in dict(context or {}).items()
    }


def _validated_identifier(value: Any, field_name: str) -> str:
    text = str(value)
    if not _SAFE_IDENTIFIER_PATTERN.fullmatch(text):
        raise PipelineContractError(
            "E_EVENT_FIELD_INVALID",
            f"Structured event field {field_name!r} is not a safe identifier.",
            context={"reason": f"invalid_{field_name}"},
            next_action="Use a short operational identifier without whitespace or free text.",
        )
    return text


def _safe_error_message(code: str, error_type: str) -> str:
    return f"{error_type} reported pipeline error {code}."


def _safe_next_action(code: str) -> str:
    return f"Follow the documented recovery action for {code}."


def _error_payload(error: BaseException | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, Mapping):
        required = {"code", "type", "message", "retriable", "next_action"}
        missing = sorted(required.difference(error))
        if missing:
            raise PipelineContractError(
                "E_EVENT_ERROR_SCHEMA",
                "Structured error mapping is missing mandatory fields.",
                context={"reason": "missing_error_fields"},
                next_action="Provide code, type, message, retriable, and next_action.",
            )
        if not isinstance(error["retriable"], bool):
            raise PipelineContractError(
                "E_EVENT_ERROR_SCHEMA",
                "Structured error retriable field must be boolean.",
                context={"reason": "invalid_retriable"},
                next_action="Set retriable to true or false.",
            )
        if any(not isinstance(error[field], str) for field in ("code", "type", "message", "next_action")):
            raise PipelineContractError(
                "E_EVENT_ERROR_SCHEMA",
                "Structured error text fields must be strings.",
                context={"reason": "invalid_error_field_type"},
                next_action="Provide string code, type, message, and next_action fields.",
            )
        if "context" in error and not isinstance(error["context"], Mapping):
            raise PipelineContractError(
                "E_EVENT_ERROR_SCHEMA",
                "Structured error context must be a mapping.",
                context={"reason": "invalid_error_context"},
                next_action="Provide a mapping for structured error context.",
            )
        code = _validated_identifier(error["code"], "error.code")
        error_type = _validated_identifier(error["type"], "error.type")
        return {
            "code": code,
            "type": error_type,
            # Never persist caller-provided free text from an error boundary.
            "message": _safe_error_message(code, error_type),
            "retriable": error["retriable"],
            "next_action": _safe_next_action(code),
            "context": _safe_context(error.get("context")),
        }
    if isinstance(error, PipelineContractError):
        code = _validated_identifier(error.code, "error.code")
        error_type = type(error).__name__
        return {
            "code": code,
            "type": error_type,
            "message": _safe_error_message(code, error_type),
            "retriable": error.retriable,
            "next_action": _safe_next_action(code),
            "context": _safe_context(error.context),
        }
    code = "E_UNEXPECTED"
    error_type = _validated_identifier(type(error).__name__, "error.type")
    return {
        "code": code,
        "type": error_type,
        "message": _safe_error_message(code, error_type),
        "retriable": False,
        "next_action": _safe_next_action(code),
        "context": {},
    }


class RuntimeEventLogger:
    """Emit one-line structured events to stdout and an append-only JSONL file."""

    def __init__(self, run_id: str, jsonl_path: str | os.PathLike[str]) -> None:
        self.run_id = _validated_identifier(run_id, "run_id")
        self.jsonl_path = Path(jsonl_path)
        if self.jsonl_path.exists() and self.jsonl_path.stat().st_size > 0:
            raise PipelineContractError(
                "E_EVENT_LOG_EXISTS",
                "Refusing to reset lifecycle state for a nonempty event log.",
                context={"reason": "nonempty_event_log"},
                next_action="Use a new run_id and JSONL path or replay the existing log externally.",
            )
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()
        self._attempt_states: dict[tuple[str, int], str] = {}
        self._attempt_error_codes: dict[tuple[str, int], str] = {}
        self._retry_transitions: set[tuple[str, int, int]] = set()
        self._pending_retry_targets: set[tuple[str, int]] = set()
        self._aggregate_terminals: set[str] = set()
        self._finalized = False

    def _assert_single_writer(self) -> None:
        if os.getpid() != self._owner_pid:
            raise PipelineContractError(
                "E_EVENT_WRITER_PROCESS",
                "RuntimeEventLogger may only write from its creating process.",
                context={"reason": "cross_process_writer"},
                next_action="Relay subprocess events to the parent single-writer logger.",
            )

    def _lifecycle_transition(
        self,
        phase: str,
        scope: str,
        event: str,
        attempt: int | None,
        *,
        context: Mapping[str, Any] | None,
        error_data: Mapping[str, Any] | None,
    ) -> tuple[str, Any] | None:
        if scope not in {"attempt", "transition", "aggregate"}:
            raise PipelineContractError(
                "E_EVENT_SEQUENCE",
                "Event scope must be attempt, transition, or aggregate.",
                context={"reason": "invalid_scope"},
                next_action="Use one of the three documented event scopes.",
            )
        if scope == "aggregate":
            if event not in {"PHASE_END", "PHASE_ERROR"} or attempt is not None:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Aggregate terminal must use attempt null and a terminal event.",
                    context={"reason": "invalid_aggregate_terminal"},
                    next_action="Use aggregate_terminal() after the final attempt.",
                )
            if phase in self._aggregate_terminals:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Logical phase already has an aggregate terminal.",
                    context={"reason": "duplicate_aggregate_terminal"},
                    next_action="Do not emit more events for the completed logical phase.",
                )
            if any(logical_phase == phase for logical_phase, _ in self._pending_retry_targets):
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Aggregate terminal is forbidden while a retry target is pending.",
                    context={"reason": "pending_retry_target"},
                    next_action="Start and terminate attempt 2 before aggregate completion.",
                )
            phase_states = [
                (attempt_number, state)
                for (logical_phase, attempt_number), state in self._attempt_states.items()
                if logical_phase == phase
            ]
            if not phase_states or any(state == "STARTED" for _, state in phase_states):
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Aggregate terminal requires every started attempt to be terminal.",
                    context={"reason": "attempt_not_terminal"},
                    next_action="End the final attempt before the aggregate terminal.",
                )
            _, final_state = max(phase_states, key=lambda item: item[0])
            aggregate_matches_final = (event == "PHASE_END" and final_state == "END") or (
                event == "PHASE_ERROR" and final_state in {"ERROR", "ERROR_RETRIABLE"}
            )
            if not aggregate_matches_final:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Aggregate terminal must match the final attempt outcome.",
                    context={"reason": "aggregate_outcome_mismatch"},
                    next_action="Emit aggregate END after success or aggregate ERROR after failure.",
                )
            return ("AGGREGATE_TERMINAL", phase)

        if scope == "transition":
            if attempt is not None:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Transition events require attempt null.",
                    context={"reason": "transition_with_attempt"},
                    next_action="Move retry attempt numbers into from_attempt and to_attempt.",
                )
            if phase in self._aggregate_terminals:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "No transition may follow a logical phase aggregate terminal.",
                    context={"reason": "post_aggregate_event"},
                    next_action="Start a new logical phase name.",
                )
            if event != "OOM_RETRY":
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "OOM_RETRY is the only permitted transition event.",
                    context={"reason": "unsupported_transition_event"},
                    next_action="Use attempt scope for ordinary events.",
                )
            transition_context = dict(context or {})
            from_attempt = transition_context.get("from_attempt")
            to_attempt = transition_context.get("to_attempt")
            if (
                isinstance(from_attempt, bool)
                or isinstance(to_attempt, bool)
                or not isinstance(from_attempt, int)
                or not isinstance(to_attempt, int)
                or from_attempt != 1
                or to_attempt != 2
            ):
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "The sole OOM_RETRY transition must be from attempt 1 to attempt 2.",
                    context={"reason": "invalid_retry_transition"},
                    next_action="Set from_attempt=1 and to_attempt=2.",
                )
            if self._attempt_states.get((phase, from_attempt)) != "ERROR_RETRIABLE":
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "OOM_RETRY requires a retriable PHASE_ERROR for the source attempt.",
                    context={"reason": "retry_without_retriable_error"},
                    next_action="End the source attempt with a retriable error first.",
                )
            if self._attempt_error_codes.get((phase, from_attempt)) != "E_TRAIN_CUDA_OOM":
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "OOM_RETRY requires E_TRAIN_CUDA_OOM on the source attempt.",
                    context={"reason": "retry_without_cuda_oom"},
                    next_action="Emit OOM_RETRY only after a retriable E_TRAIN_CUDA_OOM.",
                )
            retry_key = (phase, from_attempt, to_attempt)
            if retry_key in self._retry_transitions or (phase, to_attempt) in self._attempt_states:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "OOM retry transition is duplicate or targets an existing attempt.",
                    context={"reason": "duplicate_retry_transition"},
                    next_action="Emit exactly one transition before starting the target attempt.",
                )
            return ("RETRY_TRANSITION", retry_key)

        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise PipelineContractError(
                "E_EVENT_SEQUENCE",
                "Attempt events require a positive attempt number.",
                context={"reason": "invalid_attempt"},
                next_action="Use attempt 1 or greater.",
            )
        if phase in self._aggregate_terminals:
            raise PipelineContractError(
                "E_EVENT_SEQUENCE",
                "No event may follow a logical phase aggregate terminal.",
                context={"reason": "post_aggregate_event"},
                next_action="Start a new logical phase name.",
            )

        key = (phase, attempt)
        state = self._attempt_states.get(key)
        if state in {"END", "ERROR", "ERROR_RETRIABLE"}:
            raise PipelineContractError(
                "E_EVENT_SEQUENCE",
                "No event may follow the terminal event for an attempt.",
                context={"reason": "post_terminal_event"},
                next_action="Start the next retry attempt or emit the aggregate terminal.",
            )
        if event == "PHASE_START":
            if state is not None:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "An attempt may have exactly one PHASE_START.",
                    context={"reason": "duplicate_phase_start"},
                    next_action="Do not enter the same phase attempt twice.",
                )
            if attempt > 1 and (phase, attempt - 1, attempt) not in self._retry_transitions:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Retry attempt requires one preceding OOM_RETRY transition.",
                    context={"reason": "attempt_without_retry_transition"},
                    next_action="Emit the transition from the retriable prior attempt first.",
                )
            return ("ATTEMPT_STARTED", key)
        if event in {"PHASE_END", "PHASE_ERROR"}:
            if state != "STARTED":
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "A terminal event requires one preceding PHASE_START.",
                    context={"reason": "terminal_without_start"},
                    next_action="Emit PHASE_START before its terminal event.",
                )
            if event == "PHASE_END":
                return ("ATTEMPT_END", key)
            retriable = bool(error_data and error_data.get("retriable") is True)
            error_code = str(error_data["code"]) if error_data is not None else "E_UNEXPECTED"
            return (
                "ATTEMPT_ERROR_RETRIABLE" if retriable else "ATTEMPT_ERROR",
                (key, error_code),
            )
        if state != "STARTED":
            raise PipelineContractError(
                "E_EVENT_SEQUENCE",
                "Ordinary attempt events require an active started attempt.",
                context={"reason": "orphan_attempt_event"},
                next_action="Emit PHASE_START before attempt-scoped diagnostics.",
            )
        return None

    @staticmethod
    def _validate_event_schema(
        scope: str,
        event: str,
        status: str,
        error_data: Mapping[str, Any] | None,
    ) -> None:
        valid = False
        if event == "PHASE_START":
            valid = scope == "attempt" and status == "RUNNING" and error_data is None
        elif event == "PHASE_END":
            valid = scope in {"attempt", "aggregate"} and status == "SUCCESS" and error_data is None
        elif event == "PHASE_ERROR":
            valid = scope in {"attempt", "aggregate"} and status == "ERROR" and error_data is not None
        elif event == "OOM_RETRY":
            valid = scope == "transition" and status == "RUNNING" and error_data is None
        elif event == "OPTIONAL_FALLBACK":
            valid = scope == "attempt" and status == "FALLBACK" and error_data is not None
        elif event == "SOURCE_RESOLVED":
            valid = scope == "attempt" and (
                (status == "SUCCESS" and error_data is None)
                or (status == "ERROR" and error_data is not None)
            )
        else:
            valid = scope == "attempt" and (
                (status in {"RUNNING", "SUCCESS"} and error_data is None)
                or (status == "ERROR" and error_data is not None)
            )
        if not valid:
            raise PipelineContractError(
                "E_EVENT_SCHEMA",
                "Event scope/status/error combination violates the structured event contract.",
                context={"reason": "event_matrix_violation"},
                next_action="Use the documented event/status/error matrix.",
            )

    def _commit_lifecycle(self, transition: tuple[str, Any] | None) -> None:
        if transition is None:
            return
        action, key = transition
        if action == "ATTEMPT_STARTED":
            self._attempt_states[key] = "STARTED"
            self._pending_retry_targets.discard(key)
        elif action == "ATTEMPT_END":
            self._attempt_states[key] = "END"
        elif action == "ATTEMPT_ERROR":
            attempt_key, error_code = key
            self._attempt_states[attempt_key] = "ERROR"
            self._attempt_error_codes[attempt_key] = error_code
        elif action == "ATTEMPT_ERROR_RETRIABLE":
            attempt_key, error_code = key
            self._attempt_states[attempt_key] = "ERROR_RETRIABLE"
            self._attempt_error_codes[attempt_key] = error_code
            if attempt_key[1] == 1 and error_code == "E_TRAIN_CUDA_OOM":
                self._pending_retry_targets.add((attempt_key[0], 2))
        elif action == "RETRY_TRANSITION":
            self._retry_transitions.add(key)
            self._pending_retry_targets.add((key[0], key[2]))
        elif action == "AGGREGATE_TERMINAL":
            self._aggregate_terminals.add(str(key))

    def _emit(
        self,
        phase: str,
        event: str,
        status: str,
        *,
        scope: str,
        attempt: int | None,
        context: Mapping[str, Any] | None,
        error: BaseException | Mapping[str, Any] | None,
        duration_ms: float | None,
    ) -> dict[str, Any]:
        self._assert_single_writer()
        if self._finalized:
            raise PipelineContractError(
                "E_EVENT_SEQUENCE",
                "No event may be emitted after logger finalization.",
                context={"reason": "post_finalize_event"},
                next_action="Create a new logger for a new run.",
            )
        phase_value = _validated_identifier(phase, "phase")
        event_value = _validated_identifier(event, "event")
        status_value = _validated_identifier(status, "status")
        scope_value = _validated_identifier(scope, "scope")
        resources = hardware_snapshot()
        error_data = _error_payload(error)
        self._validate_event_schema(scope_value, event_value, status_value, error_data)
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": self.run_id,
            "phase": phase_value,
            "scope": scope_value,
            "event": event_value,
            "status": status_value,
            "attempt": attempt,
            "duration_ms": duration_ms,
            "gpu": resources["gpu"],
            "host": resources["host"],
            "context": _safe_context(context),
        }
        if error_data is not None:
            payload["error"] = error_data
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

        with self._lock:
            # The resource snapshot is intentionally collected outside the writer
            # lock.  Finalization may therefore win the race while an emitter is
            # sampling hardware; re-check the sealed state at the actual commit
            # boundary so no event can be appended after ``finalize()``.
            if self._finalized:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "No event may be emitted after logger finalization.",
                    context={"reason": "post_finalize_event"},
                    next_action="Create a new logger for a new run.",
                )
            transition = self._lifecycle_transition(
                phase_value,
                scope_value,
                event_value,
                attempt,
                context=context,
                error_data=error_data,
            )
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
            self._commit_lifecycle(transition)
            print(_EVENT_PREFIX + line, flush=True)
        return payload

    def finalize(self) -> dict[str, int]:
        """Seal the logger only when every logical phase has one aggregate terminal."""

        self._assert_single_writer()
        with self._lock:
            logical_phases = {phase for phase, _ in self._attempt_states}
            if not logical_phases:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Cannot finalize an empty run without lifecycle evidence.",
                    context={"reason": "empty_run_history"},
                    next_action="Record and aggregate at least one logical phase before finalization.",
                )
            if self._pending_retry_targets:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Cannot finalize with a pending retry target.",
                    context={"reason": "pending_retry_target"},
                    next_action="Complete attempt 2 and its aggregate terminal.",
                )
            if any(state == "STARTED" for state in self._attempt_states.values()):
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Cannot finalize while an attempt is still running.",
                    context={"reason": "attempt_still_started"},
                    next_action="Emit the attempt terminal first.",
                )
            missing = logical_phases.difference(self._aggregate_terminals)
            if missing:
                raise PipelineContractError(
                    "E_EVENT_SEQUENCE",
                    "Every logical phase requires one aggregate terminal before finalization.",
                    context={"reason": "missing_aggregate_terminal"},
                    next_action="Call aggregate_terminal() for each completed logical phase.",
                )
            self._finalized = True
            return {
                "logical_phases": len(logical_phases),
                "aggregate_terminals": len(self._aggregate_terminals),
            }

    def emit(
        self,
        phase: str,
        event: str,
        status: str,
        *,
        scope: str = "attempt",
        attempt: int | None = 1,
        context: Mapping[str, Any] | None = None,
        error: BaseException | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            phase,
            event,
            status,
            scope=scope,
            attempt=attempt,
            context=context,
            error=error,
            duration_ms=None,
        )

    def emit_oom_retry(
        self,
        phase: str,
        *,
        from_attempt: int,
        to_attempt: int,
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        transition_context = dict(context or {})
        transition_context.update({"from_attempt": from_attempt, "to_attempt": to_attempt})
        return self.emit(
            phase,
            "OOM_RETRY",
            "RUNNING",
            scope="transition",
            attempt=None,
            context=transition_context,
        )

    def aggregate_terminal(
        self,
        phase: str,
        *,
        succeeded: bool,
        context: Mapping[str, Any] | None = None,
        error: BaseException | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Emit the sole logical-phase terminal after all attempts have ended."""

        if succeeded and error is not None:
            raise PipelineContractError(
                "E_EVENT_ERROR_SCHEMA",
                "A successful aggregate terminal cannot contain an error.",
                context={"reason": "success_with_error"},
                next_action="Remove the error or mark the aggregate terminal unsuccessful.",
            )
        if not succeeded and error is None:
            raise PipelineContractError(
                "E_EVENT_ERROR_SCHEMA",
                "A failed aggregate terminal requires a structured error.",
                context={"reason": "failure_without_error"},
                next_action="Provide the final phase error.",
            )
        return self._emit(
            phase,
            "PHASE_END" if succeeded else "PHASE_ERROR",
            "SUCCESS" if succeeded else "ERROR",
            scope="aggregate",
            attempt=None,
            context=context,
            error=error,
            duration_ms=None,
        )

    @contextmanager
    def phase(
        self,
        phase: str,
        attempt: int = 1,
        context: Mapping[str, Any] | None = None,
    ) -> Iterator[None]:
        self.emit(phase, "PHASE_START", "RUNNING", attempt=attempt, context=context)
        started = time.perf_counter()
        try:
            yield
        except BaseException as error:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self._emit(
                phase,
                "PHASE_ERROR",
                "ERROR",
                scope="attempt",
                attempt=attempt,
                context=context,
                error=error,
                duration_ms=duration_ms,
            )
            raise
        else:
            duration_ms = round((time.perf_counter() - started) * 1000, 3)
            self._emit(
                phase,
                "PHASE_END",
                "SUCCESS",
                scope="attempt",
                attempt=attempt,
                context=context,
                error=None,
                duration_ms=duration_ms,
            )


@dataclass(frozen=True)
class QwenRuntimeProbe:
    runtime_engine: SafeIdentifier | None
    runtime_version: SafeIdentifier | None
    awq_package: SafeIdentifier | None
    awq_version: SafeIdentifier | None
    cuda_version: SafeIdentifier | None
    cuda_available: bool
    capability: tuple[int, int] | None
    kernel_supported: bool
    kernel_probe_id: SafeIdentifier
    evidence_hash: SafeHash

    def __post_init__(self) -> None:
        identifier_values = (
            self.runtime_engine,
            self.runtime_version,
            self.awq_package,
            self.awq_version,
            self.cuda_version,
        )
        if any(value is not None and not isinstance(value, SafeIdentifier) for value in identifier_values):
            raise ValueError("Qwen runtime probe identifiers must be typed SafeIdentifier values")
        if not isinstance(self.kernel_probe_id, SafeIdentifier):
            raise ValueError("Qwen kernel probe id must be a typed SafeIdentifier")
        if not isinstance(self.evidence_hash, SafeHash):
            raise ValueError("Qwen evidence hash must be a typed SafeHash")
        if not isinstance(self.cuda_available, bool) or not isinstance(self.kernel_supported, bool):
            raise ValueError("Qwen runtime probe flags must be boolean")
        if self.capability is not None and (
            not isinstance(self.capability, tuple)
            or len(self.capability) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in self.capability)
        ):
            raise ValueError("Qwen capability must be a nonnegative (major, minor) tuple")
        if not self.has_valid_evidence_hash():
            raise ValueError("Qwen runtime probe evidence hash does not match its canonical fields")

    @classmethod
    def from_evidence(
        cls,
        *,
        runtime_engine: SafeIdentifier | None,
        runtime_version: SafeIdentifier | None,
        awq_package: SafeIdentifier | None,
        awq_version: SafeIdentifier | None,
        cuda_version: SafeIdentifier | None,
        cuda_available: bool,
        capability: tuple[int, int] | None,
        kernel_supported: bool,
        kernel_probe_id: SafeIdentifier,
    ) -> "QwenRuntimeProbe":
        """Build an immutable probe whose hash is derived from its canonical evidence fields."""

        evidence_hash = cls._evidence_hash_for(
            runtime_engine=runtime_engine,
            runtime_version=runtime_version,
            awq_package=awq_package,
            awq_version=awq_version,
            cuda_version=cuda_version,
            cuda_available=cuda_available,
            capability=capability,
            kernel_supported=kernel_supported,
            kernel_probe_id=kernel_probe_id,
        )
        return cls(
            runtime_engine=runtime_engine,
            runtime_version=runtime_version,
            awq_package=awq_package,
            awq_version=awq_version,
            cuda_version=cuda_version,
            cuda_available=cuda_available,
            capability=capability,
            kernel_supported=kernel_supported,
            kernel_probe_id=kernel_probe_id,
            evidence_hash=SafeHash(evidence_hash),
        )

    @staticmethod
    def _evidence_hash_for(
        *,
        runtime_engine: SafeIdentifier | None,
        runtime_version: SafeIdentifier | None,
        awq_package: SafeIdentifier | None,
        awq_version: SafeIdentifier | None,
        cuda_version: SafeIdentifier | None,
        cuda_available: bool,
        capability: tuple[int, int] | None,
        kernel_supported: bool,
        kernel_probe_id: SafeIdentifier,
    ) -> str:
        evidence = {
            "awq_package": awq_package.value if isinstance(awq_package, SafeIdentifier) else None,
            "awq_version": awq_version.value if isinstance(awq_version, SafeIdentifier) else None,
            "capability": list(capability) if isinstance(capability, tuple) else None,
            "cuda_available": cuda_available,
            "cuda_version": cuda_version.value if isinstance(cuda_version, SafeIdentifier) else None,
            "kernel_probe_id": (
                kernel_probe_id.value if isinstance(kernel_probe_id, SafeIdentifier) else None
            ),
            "kernel_supported": kernel_supported,
            "runtime_engine": (
                runtime_engine.value if isinstance(runtime_engine, SafeIdentifier) else None
            ),
            "runtime_version": (
                runtime_version.value if isinstance(runtime_version, SafeIdentifier) else None
            ),
        }
        return hashlib.sha256(
            json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

    def has_valid_evidence_hash(self) -> bool:
        try:
            expected = self._evidence_hash_for(
                runtime_engine=self.runtime_engine,
                runtime_version=self.runtime_version,
                awq_package=self.awq_package,
                awq_version=self.awq_version,
                cuda_version=self.cuda_version,
                cuda_available=self.cuda_available,
                capability=self.capability,
                kernel_supported=self.kernel_supported,
                kernel_probe_id=self.kernel_probe_id,
            )
        except Exception:
            return False
        return isinstance(self.evidence_hash, SafeHash) and self.evidence_hash.value == expected


def _version_identifier(value: Any) -> SafeIdentifier:
    normalized = re.sub(r"[^A-Za-z0-9_.:-]+", "-", str(value or "unknown"))[:120]
    candidate = f"v{normalized}" if normalized and not normalized.startswith("v") else normalized
    try:
        return SafeIdentifier(candidate or "unknown")
    except ValueError:
        return SafeIdentifier("unknown")


def probe_qwen_runtime(
    *,
    importer: Callable[[str], Any] = importlib.import_module,
    kernel_probe: Callable[[Any, Any, Any, tuple[int, int]], bool] | None = None,
    kernel_probe_id: SafeIdentifier = SafeIdentifier("awq-kernel-unconfigured"),
) -> QwenRuntimeProbe:
    """Lazily collect runtime/AWQ/CUDA evidence without importing or loading a model at module import."""

    runtime_module: Any | None = None
    awq_module: Any | None = None
    torch_module: Any | None = None
    try:
        runtime_module = importer("vllm")
    except Exception:
        pass
    try:
        awq_module = importer("awq")
    except Exception:
        pass
    try:
        torch_module = importer("torch")
    except Exception:
        pass

    cuda_available = False
    cuda_version: SafeIdentifier | None = None
    capability: tuple[int, int] | None = None
    if torch_module is not None:
        try:
            cuda_available = bool(torch_module.cuda.is_available())
        except Exception:
            cuda_available = False
        try:
            raw_cuda_version = getattr(getattr(torch_module, "version", None), "cuda", None)
            cuda_version = _version_identifier(raw_cuda_version) if raw_cuda_version else None
        except Exception:
            cuda_version = None
        if cuda_available:
            try:
                capability = _capability_tuple(torch_module.cuda.get_device_capability())
            except Exception:
                capability = None

    kernel_supported = False
    if (
        runtime_module is not None
        and awq_module is not None
        and torch_module is not None
        and cuda_available
        and capability is not None
        and kernel_probe is not None
    ):
        try:
            probe_result = kernel_probe(runtime_module, awq_module, torch_module, capability)
            kernel_supported = probe_result if isinstance(probe_result, bool) else False
        except Exception:
            kernel_supported = False

    return QwenRuntimeProbe.from_evidence(
        runtime_engine=SafeIdentifier("vllm") if runtime_module is not None else None,
        runtime_version=(
            _version_identifier(getattr(runtime_module, "__version__", "unknown"))
            if runtime_module is not None
            else None
        ),
        awq_package=SafeIdentifier("awq") if awq_module is not None else None,
        awq_version=(
            _version_identifier(getattr(awq_module, "__version__", "unknown"))
            if awq_module is not None
            else None
        ),
        cuda_version=cuda_version,
        cuda_available=cuda_available,
        capability=capability,
        kernel_supported=kernel_supported,
        kernel_probe_id=kernel_probe_id,
    )


@dataclass(frozen=True)
class QwenProfile:
    model_id: str = DEFAULT_QWEN_MODEL_ID
    gpu_memory_utilization: float = 0.40
    max_model_len: int = 1024
    batch_ladder: tuple[int, ...] = (8, 4, 1)
    engine_init_timeout_seconds: float = 120.0
    generation_timeout_seconds: float = 300.0

    def __post_init__(self) -> None:
        if self.model_id not in {DEFAULT_QWEN_MODEL_ID, QWEN_7B_MODEL_ID}:
            raise ValueError("Qwen profile model is not approved")
        if self.gpu_memory_utilization != 0.40 or self.max_model_len != 1024:
            raise ValueError("Qwen profile must use the approved T4 memory/context values")
        if self.batch_ladder != (8, 4, 1):
            raise ValueError("Qwen profile must use the approved batch ladder")
        for value in (self.engine_init_timeout_seconds, self.generation_timeout_seconds):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError("Qwen timeouts must be positive finite seconds")


def _canonical_sha256(payload: Mapping[str, Any]) -> SafeHash:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return SafeHash(hashlib.sha256(encoded).hexdigest())


def _qwen_profile_hash(profile: QwenProfile) -> SafeHash:
    return _canonical_sha256(asdict(profile))


def _qwen_hardware_hash(gpu: Mapping[str, Any]) -> SafeHash:
    return _canonical_sha256(
        {
            "capability": list(_capability_tuple(gpu.get("capability")) or ()),
            "free_gib": _finite_measurement(gpu.get("free_gib")),
            "name": str(gpu.get("name") or ""),
            "total_gib": _finite_measurement(gpu.get("total_gib")),
        }
    )


@dataclass(frozen=True, init=False)
class QwenAdmission:
    """Planner-issued binding between a profile and admitted hardware/runtime evidence."""

    profile_hash: SafeHash
    probe_evidence_hash: SafeHash
    hardware_hash: SafeHash
    gpu_capability: tuple[int, int]
    allow_qwen_7b_override: bool
    admission_hash: SafeHash

    @classmethod
    def _issue(
        cls,
        *,
        profile: QwenProfile,
        gpu: Mapping[str, Any],
        runtime_probe: QwenRuntimeProbe,
        allow_qwen_7b_override: bool,
    ) -> "QwenAdmission":
        capability = _capability_tuple(gpu.get("capability"))
        if capability is None or not runtime_probe.has_valid_evidence_hash():
            raise ValueError("Qwen admission requires canonical hardware and runtime evidence")
        profile_hash = _qwen_profile_hash(profile)
        hardware_hash = _qwen_hardware_hash(gpu)
        admission_payload = {
            "allow_qwen_7b_override": allow_qwen_7b_override,
            "gpu_capability": list(capability),
            "hardware_hash": hardware_hash.value,
            "probe_evidence_hash": runtime_probe.evidence_hash.value,
            "profile_hash": profile_hash.value,
        }
        admission = object.__new__(cls)
        object.__setattr__(admission, "profile_hash", profile_hash)
        object.__setattr__(admission, "probe_evidence_hash", runtime_probe.evidence_hash)
        object.__setattr__(admission, "hardware_hash", hardware_hash)
        object.__setattr__(admission, "gpu_capability", capability)
        object.__setattr__(admission, "allow_qwen_7b_override", allow_qwen_7b_override)
        object.__setattr__(admission, "admission_hash", _canonical_sha256(admission_payload))
        return admission

    def has_valid_binding(self, profile: QwenProfile) -> bool:
        try:
            if (
                not isinstance(profile, QwenProfile)
                or not isinstance(self.profile_hash, SafeHash)
                or not isinstance(self.probe_evidence_hash, SafeHash)
                or not isinstance(self.hardware_hash, SafeHash)
                or not isinstance(self.admission_hash, SafeHash)
                or not isinstance(self.allow_qwen_7b_override, bool)
                or not isinstance(self.gpu_capability, tuple)
                or self.gpu_capability < (7, 5)
                or self.profile_hash != _qwen_profile_hash(profile)
            ):
                return False
            expected = _canonical_sha256(
                {
                    "allow_qwen_7b_override": self.allow_qwen_7b_override,
                    "gpu_capability": list(self.gpu_capability),
                    "hardware_hash": self.hardware_hash.value,
                    "probe_evidence_hash": self.probe_evidence_hash.value,
                    "profile_hash": self.profile_hash.value,
                }
            )
        except Exception:
            return False
        return self.admission_hash == expected


@dataclass(frozen=True)
class ResourcePlan:
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    eval_accumulation_steps: int
    fp16: bool
    bf16: bool
    gradient_checkpointing: bool
    qwen_enabled: bool
    qwen_profile: QwenProfile | None
    qwen_admission: QwenAdmission | None = None
    qwen_disabled_reason: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.qwen_enabled, bool):
            raise ValueError("ResourcePlan qwen_enabled must be an exact boolean")
        if self.qwen_enabled:
            if (
                not isinstance(self.qwen_profile, QwenProfile)
                or not isinstance(self.qwen_admission, QwenAdmission)
                or not self.qwen_admission.has_valid_binding(self.qwen_profile)
            ):
                raise ValueError("Enabled Qwen requires one valid planner-issued admission")
        elif self.qwen_profile is not None or self.qwen_admission is not None:
            raise ValueError("Disabled Qwen cannot retain a profile or admission")


def _gpu_details(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = snapshot.get("gpu")
    return nested if isinstance(nested, Mapping) else snapshot


def _is_p100(gpu: Mapping[str, Any]) -> bool:
    name = str(gpu.get("name") or "").upper()
    capability = gpu.get("capability")
    if isinstance(capability, Sequence) and not isinstance(capability, str):
        capability = ".".join(str(part) for part in capability)
    return "P100" in name or str(capability or "") in {"6", "6.0"}


def _capability_tuple(capability: Any) -> tuple[int, int] | None:
    if isinstance(capability, Sequence) and not isinstance(capability, str):
        parts = list(capability)
    else:
        parts = str(capability or "").split(".")
    if not parts or not str(parts[0]).isdigit():
        return None
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except (TypeError, ValueError):
        return None
    return major, minor


def _qwen_disabled_reason(
    gpu: Mapping[str, Any],
    *,
    runtime_probe: QwenRuntimeProbe | None,
    model_id: str,
    allow_qwen_7b_override: bool,
) -> str:
    if _is_p100(gpu):
        return "Qwen is disabled on NVIDIA P100 / compute capability 6.0."
    capability = _capability_tuple(gpu.get("capability"))
    if capability is None:
        return "Qwen is disabled because compute capability is unknown."
    if capability < (7, 5):
        return f"Qwen is disabled on unsupported compute capability {capability[0]}.{capability[1]}."
    if runtime_probe is None:
        return "Qwen is disabled because concrete runtime/AWQ/CUDA/kernel probe evidence is missing."
    if not isinstance(runtime_probe, QwenRuntimeProbe):
        return "Qwen is disabled because runtime probe evidence has an invalid type."
    if not runtime_probe.has_valid_evidence_hash():
        return "Qwen is disabled because runtime probe evidence integrity validation failed."
    if runtime_probe.runtime_engine is None or runtime_probe.runtime_version is None:
        return "Qwen is disabled because the runtime engine probe did not pass."
    if runtime_probe.awq_package is None or runtime_probe.awq_version is None:
        return "Qwen is disabled because the AWQ package probe did not pass."
    if not runtime_probe.cuda_available or runtime_probe.cuda_version is None:
        return "Qwen is disabled because CUDA runtime evidence is unavailable."
    if runtime_probe.capability != capability:
        return "Qwen is disabled because probed CUDA capability does not match hardware evidence."
    if not runtime_probe.kernel_supported:
        return "Qwen is disabled because the required AWQ kernel probe did not pass."

    if model_id == QWEN_7B_MODEL_ID:
        name = str(gpu.get("name") or "").upper()
        if not allow_qwen_7b_override:
            return "Qwen 7B requires an explicit advanced override."
        if "T4" not in name or capability != (7, 5):
            return "Qwen 7B override is allowed only on a probed NVIDIA T4."
    elif model_id != DEFAULT_QWEN_MODEL_ID:
        return "Qwen model override is not an approved resource profile."
    return ""


def _finite_measurement(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _resource_control_error(field_name: str) -> PipelineContractError:
    return PipelineContractError(
        "E_RESOURCE_CONTROL_INVALID",
        f"Resource control or measurement {field_name!r} is invalid.",
        context={"reason": "invalid_resource_control"},
        next_action="Provide exact booleans and finite numeric resource values.",
    )


def _exact_resource_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise _resource_control_error(field_name)
    return value


def _positive_resource_control(value: Any, field_name: str) -> float:
    measurement = _finite_measurement(value)
    if measurement is None or measurement <= 0:
        raise _resource_control_error(field_name)
    return measurement


def _optional_resource_measurement(value: Any, field_name: str) -> float | None:
    if value is None:
        return None
    measurement = _finite_measurement(value)
    if measurement is None:
        raise _resource_control_error(field_name)
    return measurement


def choose_resource_plan(
    snapshot: Mapping[str, Any],
    *,
    require_gpu: bool,
    qwen_requested: bool,
    fast_dev_run: bool = False,
    qwen_runtime_probe: QwenRuntimeProbe | None = None,
    qwen_model_id: str | None = None,
    allow_qwen_7b_override: bool = False,
    min_host_ram_gib: float = 10.0,
    min_disk_free_gib: float = 15.0,
) -> ResourcePlan:
    """Validate the hardware budget and choose the single-GPU safe profile."""

    if not isinstance(snapshot, Mapping):
        raise _resource_control_error("snapshot")
    require_gpu = _exact_resource_bool(require_gpu, "require_gpu")
    qwen_requested = _exact_resource_bool(qwen_requested, "qwen_requested")
    fast_dev_run = _exact_resource_bool(fast_dev_run, "fast_dev_run")
    allow_qwen_7b_override = _exact_resource_bool(
        allow_qwen_7b_override, "allow_qwen_7b_override"
    )
    min_host_ram = _positive_resource_control(min_host_ram_gib, "min_host_ram_gib")
    min_disk_free = _positive_resource_control(min_disk_free_gib, "min_disk_free_gib")
    if qwen_model_id is not None and (
        not isinstance(qwen_model_id, str) or not qwen_model_id.strip()
    ):
        raise _resource_control_error("qwen_model_id")
    if "gpu" in snapshot and not isinstance(snapshot["gpu"], Mapping):
        raise _resource_control_error("gpu")
    gpu = _gpu_details(snapshot)
    total = gpu.get("total_gib")
    free = gpu.get("free_gib")
    available_value = gpu.get("available", total is not None)
    gpu_available = _exact_resource_bool(available_value, "gpu.available")
    total_measurement = _optional_resource_measurement(total, "gpu.total_gib")
    free_measurement = _optional_resource_measurement(free, "gpu.free_gib")
    selected_qwen_model = qwen_model_id or DEFAULT_QWEN_MODEL_ID

    if not gpu_available:
        if require_gpu:
            raise PipelineContractError(
                "E_GPU_BUDGET",
                "A CUDA GPU is required for this run.",
                context={"total_gib": total, "free_gib": free},
                next_action="Attach a GPU and rerun preflight.",
            )
        return ResourcePlan(
            train_batch_size=1 if fast_dev_run else 2,
            eval_batch_size=1 if fast_dev_run else 2,
            gradient_accumulation_steps=2 if fast_dev_run else 8,
            eval_accumulation_steps=16,
            fp16=False,
            bf16=False,
            gradient_checkpointing=True,
            qwen_enabled=False,
            qwen_profile=None,
            qwen_disabled_reason="Qwen requires a supported CUDA GPU." if qwen_requested else "",
        )

    gpu_budget_satisfied = (
        total_measurement is not None
        and free_measurement is not None
        and total_measurement >= 14.0
        and free_measurement >= 12.0
    )
    if not gpu_budget_satisfied:
        if not require_gpu:
            return ResourcePlan(
                train_batch_size=1 if fast_dev_run else 2,
                eval_batch_size=1 if fast_dev_run else 2,
                gradient_accumulation_steps=2 if fast_dev_run else 8,
                eval_accumulation_steps=16,
                fp16=False,
                bf16=False,
                gradient_checkpointing=True,
                qwen_enabled=False,
                qwen_profile=None,
                qwen_disabled_reason="Qwen requires an admitted CUDA GPU." if qwen_requested else "",
            )
        raise PipelineContractError(
            "E_GPU_BUDGET",
            "GPU budget requires at least 14 GiB total and 12 GiB free VRAM.",
            context={"total_gib": total, "free_gib": free},
            next_action="Release GPU memory or attach a GPU that satisfies the preflight budget.",
        )

    host = snapshot.get("host")
    if host is not None and not isinstance(host, Mapping):
        raise _resource_control_error("host")
    host = host if isinstance(host, Mapping) else {}
    host_available = _optional_resource_measurement(
        host.get("ram_available_gib"), "host.ram_available_gib"
    )
    disk_free = _optional_resource_measurement(
        host.get("disk_free_gib"), "host.disk_free_gib"
    )
    if require_gpu and (host_available is None or host_available < min_host_ram):
        raise PipelineContractError(
            "E_RUNTIME_BUDGET",
            "Training host RAM admission failed.",
            context={
                "ram_available_gib": host_available,
                "min_host_ram_gib": min_host_ram,
            },
            next_action="Free host RAM or attach a runtime with more memory.",
        )
    if require_gpu and (disk_free is None or disk_free < min_disk_free):
        raise PipelineContractError(
            "E_DISK_BUDGET",
            "Training disk admission failed.",
            context={
                "disk_free_gib": disk_free,
                "min_disk_free_gib": min_disk_free,
            },
            next_action="Free working disk space or increase the declared disk quota.",
        )

    reason = ""
    if qwen_requested:
        reason = _qwen_disabled_reason(
            gpu,
            runtime_probe=qwen_runtime_probe,
            model_id=selected_qwen_model,
            allow_qwen_7b_override=allow_qwen_7b_override,
        )
    qwen_enabled = bool(qwen_requested and not reason)
    qwen_profile = QwenProfile(model_id=selected_qwen_model) if qwen_enabled else None
    qwen_admission = (
        QwenAdmission._issue(
            profile=qwen_profile,
            gpu=gpu,
            runtime_probe=qwen_runtime_probe,
            allow_qwen_7b_override=allow_qwen_7b_override,
        )
        if qwen_enabled
        and qwen_profile is not None
        and isinstance(qwen_runtime_probe, QwenRuntimeProbe)
        else None
    )
    return ResourcePlan(
        train_batch_size=1 if fast_dev_run else 2,
        eval_batch_size=1 if fast_dev_run else 2,
        gradient_accumulation_steps=2 if fast_dev_run else 8,
        eval_accumulation_steps=16,
        fp16=True,
        bf16=False,
        gradient_checkpointing=True,
        qwen_enabled=qwen_enabled,
        qwen_profile=qwen_profile,
        qwen_admission=qwen_admission,
        qwen_disabled_reason=reason,
    )


class QwenCudaOomError(RuntimeError):
    """Typed optional-Qwen CUDA OOM boundary; raw backend text is never logged."""


def _emit_qwen_fallback(
    logger: RuntimeEventLogger,
    phase: str,
    attempt: int,
    reason: EventReason,
) -> None:
    error = PipelineContractError(
        f"E_{reason.value}",
        "Optional Qwen execution fell back to the deterministic result.",
        context={"reason": reason},
        next_action="Inspect typed diagnostics; deterministic output was preserved.",
    )
    logger.emit(
        phase,
        "OPTIONAL_FALLBACK",
        "FALLBACK",
        attempt=attempt,
        context={"reason": reason},
        error=error,
    )


def execute_optional_qwen(
    *,
    deterministic_result: Any,
    resource_plan: ResourcePlan,
    logger: RuntimeEventLogger,
    phase: str,
    attempt: int,
    process_timeout_runner: Callable[[Callable[[], Any], float], Any],
    initialize: Callable[[QwenProfile], Any],
    generate: Callable[[Any], Any],
    parse: Callable[[Any], Any],
    cleanup: Callable[[Any], None],
) -> Any:
    """Run optional Qwen behind a caller-supplied process timeout boundary.

    This function intentionally does not create or claim cancellation of a thread.
    ``process_timeout_runner`` must enforce termination outside the model process.
    """

    if (
        not isinstance(resource_plan, ResourcePlan)
        or resource_plan.qwen_enabled is not True
        or not isinstance(resource_plan.qwen_profile, QwenProfile)
        or not isinstance(resource_plan.qwen_admission, QwenAdmission)
        or not resource_plan.qwen_admission.has_valid_binding(resource_plan.qwen_profile)
    ):
        raise PipelineContractError(
            "E_QWEN_ADMISSION",
            "Optional Qwen execution requires a valid planner-issued admission.",
            context={"reason": "qwen_not_admitted"},
            next_action="Use the enabled ResourcePlan returned by choose_resource_plan().",
        )
    if not callable(process_timeout_runner):
        raise PipelineContractError(
            "E_QWEN_EXECUTION_CONFIG",
            "Optional Qwen execution requires a process-boundary timeout runner.",
            context={"reason": "invalid_qwen_execution_config"},
            next_action="Provide a process-boundary timeout runner.",
        )
    profile = resource_plan.qwen_profile
    engine: Any | None = None
    result = deterministic_result
    failure_reason: EventReason | None = None
    try:
        try:
            engine = process_timeout_runner(
                lambda: initialize(profile), profile.engine_init_timeout_seconds
            )
        except TimeoutError:
            failure_reason = EventReason.QWEN_INIT_TIMEOUT
        except QwenCudaOomError:
            failure_reason = EventReason.QWEN_CUDA_OOM
        except Exception:
            failure_reason = EventReason.QWEN_INIT_FAILED

        generated: Any | None = None
        if failure_reason is None:
            try:
                generated = process_timeout_runner(
                    lambda: generate(engine), profile.generation_timeout_seconds
                )
            except TimeoutError:
                failure_reason = EventReason.QWEN_GENERATION_TIMEOUT
            except QwenCudaOomError:
                failure_reason = EventReason.QWEN_CUDA_OOM
            except Exception:
                failure_reason = EventReason.QWEN_GENERATION_FAILED

        if failure_reason is None:
            try:
                result = parse(generated)
            except Exception:
                failure_reason = EventReason.QWEN_PARSE_FAILED
    finally:
        if engine is not None:
            try:
                cleanup(engine)
            except Exception:
                failure_reason = EventReason.QWEN_CLEANUP_FAILED

    if failure_reason is not None:
        _emit_qwen_fallback(logger, phase, attempt, failure_reason)
        return deterministic_result
    return result


def oom_retry_plan(plan: ResourcePlan) -> ResourcePlan:
    """Return the only permitted OOM degradation rung."""

    if plan.train_batch_size <= 1 or plan.eval_batch_size <= 1:
        raise PipelineContractError(
            "E_TRAIN_CUDA_OOM",
            "Training exhausted the single permitted CUDA OOM retry.",
            context={
                "train_batch_size": plan.train_batch_size,
                "eval_batch_size": plan.eval_batch_size,
                "gradient_accumulation_steps": plan.gradient_accumulation_steps,
            },
            next_action="Keep diagnostics and resume on a GPU with more free VRAM.",
        )
    return replace(
        plan,
        train_batch_size=1,
        eval_batch_size=1,
        gradient_accumulation_steps=max(16, plan.gradient_accumulation_steps),
    )


@dataclass(frozen=True)
class StageResourceInput:
    stage_id: SafeIdentifier
    train_chunks: int
    eval_chunks: int
    inference_chunks: int
    epochs: int
    primary_train_batch_size: int
    primary_eval_batch_size: int
    primary_inference_batch_size: int
    primary_gradient_accumulation_steps: int
    retry_train_batch_size: int
    retry_eval_batch_size: int
    retry_inference_batch_size: int
    retry_gradient_accumulation_steps: int
    configured_train_seconds_per_step: float
    configured_eval_seconds_per_step: float
    configured_inference_seconds_per_step: float
    measured_train_seconds_per_step: float | None
    measured_eval_seconds_per_step: float | None
    measured_inference_seconds_per_step: float | None
    checkpoint_bytes: int
    artifact_bytes: int
    workspace_bytes: int


@dataclass(frozen=True)
class StageResourceEstimate:
    stage_id: str
    primary_train_micro_steps: int
    primary_optimizer_steps: int
    primary_eval_steps: int
    primary_inference_steps: int
    retry_train_micro_steps: int
    retry_optimizer_steps: int
    retry_eval_steps: int
    retry_inference_steps: int
    primary_seconds: float
    retry_seconds: float
    worst_case_seconds: float
    checkpoint_bytes: int
    artifact_bytes: int
    workspace_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResourceBudgetEstimate:
    stages: tuple[StageResourceEstimate, ...]
    worst_case_total_seconds: float
    safety_margin: float
    safety_adjusted_total_seconds: float
    declared_runtime_limit_seconds: float
    peak_disk_bytes: int
    usable_disk_quota_bytes: int
    remaining_disk_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stages": [stage.as_dict() for stage in self.stages],
            "worst_case_total_seconds": self.worst_case_total_seconds,
            "safety_margin": self.safety_margin,
            "safety_adjusted_total_seconds": self.safety_adjusted_total_seconds,
            "declared_runtime_limit_seconds": self.declared_runtime_limit_seconds,
            "peak_disk_bytes": self.peak_disk_bytes,
            "usable_disk_quota_bytes": self.usable_disk_quota_bytes,
            "remaining_disk_bytes": self.remaining_disk_bytes,
        }


def _resource_input_error(field_name: str) -> PipelineContractError:
    return PipelineContractError(
        "E_RESOURCE_INPUT_INVALID",
        f"Resource estimator field {field_name!r} is invalid.",
        context={"reason": "invalid_resource_input"},
        next_action="Provide finite, exact numeric admission inputs.",
    )


def _validated_nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _resource_input_error(field_name)
    return value


def _validated_positive_int(value: Any, field_name: str) -> int:
    result = _validated_nonnegative_int(value, field_name)
    if result == 0:
        raise _resource_input_error(field_name)
    return result


def _validated_positive_seconds(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _resource_input_error(field_name)
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise _resource_input_error(field_name)
    return result


def _steps(item_count: int, batch_size: int, epochs: int = 1) -> int:
    return math.ceil(item_count / batch_size) * epochs


def estimate_resource_budget(
    stages: Iterable[StageResourceInput],
    *,
    declared_runtime_limit_seconds: float,
    usable_disk_quota_bytes: int,
) -> ResourceBudgetEstimate:
    """Estimate worst-case primary plus one retry before any expensive work starts."""

    runtime_limit = _validated_positive_seconds(
        declared_runtime_limit_seconds, "declared_runtime_limit_seconds"
    )
    disk_quota = _validated_positive_int(usable_disk_quota_bytes, "usable_disk_quota_bytes")
    stage_inputs = list(stages)
    if not stage_inputs:
        raise _resource_input_error("stages")
    if any(not isinstance(stage, StageResourceInput) for stage in stage_inputs):
        raise _resource_input_error("stages")
    if any(not isinstance(stage.stage_id, SafeIdentifier) for stage in stage_inputs):
        raise _resource_input_error("stage_id")
    stage_inputs.sort(key=lambda stage: stage.stage_id.value)
    if len({stage.stage_id.value for stage in stage_inputs}) != len(stage_inputs):
        raise _resource_input_error("stage_id")

    estimates: list[StageResourceEstimate] = []
    for stage in stage_inputs:
        train_chunks = _validated_nonnegative_int(stage.train_chunks, "train_chunks")
        eval_chunks = _validated_nonnegative_int(stage.eval_chunks, "eval_chunks")
        inference_chunks = _validated_nonnegative_int(stage.inference_chunks, "inference_chunks")
        epochs = _validated_positive_int(stage.epochs, "epochs")
        primary_train_batch = _validated_positive_int(
            stage.primary_train_batch_size, "primary_train_batch_size"
        )
        primary_eval_batch = _validated_positive_int(
            stage.primary_eval_batch_size, "primary_eval_batch_size"
        )
        primary_inference_batch = _validated_positive_int(
            stage.primary_inference_batch_size, "primary_inference_batch_size"
        )
        primary_ga = _validated_positive_int(
            stage.primary_gradient_accumulation_steps,
            "primary_gradient_accumulation_steps",
        )
        retry_train_batch = _validated_positive_int(
            stage.retry_train_batch_size, "retry_train_batch_size"
        )
        retry_eval_batch = _validated_positive_int(
            stage.retry_eval_batch_size, "retry_eval_batch_size"
        )
        retry_inference_batch = _validated_positive_int(
            stage.retry_inference_batch_size, "retry_inference_batch_size"
        )
        retry_ga = _validated_positive_int(
            stage.retry_gradient_accumulation_steps,
            "retry_gradient_accumulation_steps",
        )
        configured_train = _validated_positive_seconds(
            stage.configured_train_seconds_per_step,
            "configured_train_seconds_per_step",
        )
        configured_eval = _validated_positive_seconds(
            stage.configured_eval_seconds_per_step,
            "configured_eval_seconds_per_step",
        )
        configured_inference = _validated_positive_seconds(
            stage.configured_inference_seconds_per_step,
            "configured_inference_seconds_per_step",
        )

        def timing(measured: Any, configured: float, field_name: str) -> float:
            return configured if measured is None else _validated_positive_seconds(measured, field_name)

        train_seconds_per_step = timing(
            stage.measured_train_seconds_per_step,
            configured_train,
            "measured_train_seconds_per_step",
        )
        eval_seconds_per_step = timing(
            stage.measured_eval_seconds_per_step,
            configured_eval,
            "measured_eval_seconds_per_step",
        )
        inference_seconds_per_step = timing(
            stage.measured_inference_seconds_per_step,
            configured_inference,
            "measured_inference_seconds_per_step",
        )
        checkpoint_bytes = _validated_nonnegative_int(stage.checkpoint_bytes, "checkpoint_bytes")
        artifact_bytes = _validated_nonnegative_int(stage.artifact_bytes, "artifact_bytes")
        workspace_bytes = _validated_nonnegative_int(stage.workspace_bytes, "workspace_bytes")

        primary_train_micro = _steps(train_chunks, primary_train_batch, epochs)
        primary_optimizer = math.ceil(primary_train_micro / primary_ga)
        primary_eval = _steps(eval_chunks, primary_eval_batch, epochs)
        primary_inference = _steps(inference_chunks, primary_inference_batch)
        retry_train_micro = _steps(train_chunks, retry_train_batch, epochs)
        retry_optimizer = math.ceil(retry_train_micro / retry_ga)
        retry_eval = _steps(eval_chunks, retry_eval_batch, epochs)
        retry_inference = _steps(inference_chunks, retry_inference_batch)
        primary_seconds = (
            primary_train_micro * train_seconds_per_step
            + primary_eval * eval_seconds_per_step
            + primary_inference * inference_seconds_per_step
        )
        retry_seconds = (
            retry_train_micro * train_seconds_per_step
            + retry_eval * eval_seconds_per_step
            + retry_inference * inference_seconds_per_step
        )
        estimates.append(
            StageResourceEstimate(
                stage_id=stage.stage_id.value,
                primary_train_micro_steps=primary_train_micro,
                primary_optimizer_steps=primary_optimizer,
                primary_eval_steps=primary_eval,
                primary_inference_steps=primary_inference,
                retry_train_micro_steps=retry_train_micro,
                retry_optimizer_steps=retry_optimizer,
                retry_eval_steps=retry_eval,
                retry_inference_steps=retry_inference,
                primary_seconds=primary_seconds,
                retry_seconds=retry_seconds,
                worst_case_seconds=primary_seconds + retry_seconds,
                checkpoint_bytes=checkpoint_bytes,
                artifact_bytes=artifact_bytes,
                workspace_bytes=workspace_bytes,
            )
        )

    worst_case_total = sum(stage.worst_case_seconds for stage in estimates)
    adjusted_total = worst_case_total * 1.30
    accumulated_artifacts = sum(stage.artifact_bytes for stage in estimates)
    peak_working = max(stage.checkpoint_bytes + stage.workspace_bytes for stage in estimates)
    peak_disk = accumulated_artifacts + peak_working
    if adjusted_total > runtime_limit:
        raise PipelineContractError(
            "E_RUNTIME_BUDGET",
            "Safety-adjusted worst-case runtime exceeds the declared limit.",
            context={"duration_ms": adjusted_total * 1000},
            next_action="Increase the declared runtime budget or reduce planned work explicitly.",
        )
    if peak_disk > disk_quota:
        raise PipelineContractError(
            "E_DISK_BUDGET",
            "Peak checkpoint/artifact/workspace disk exceeds usable quota.",
            context={"peak_bytes": peak_disk, "quota_bytes": disk_quota},
            next_action="Increase usable quota or reduce explicitly declared artifact sizes.",
        )
    return ResourceBudgetEstimate(
        stages=tuple(estimates),
        worst_case_total_seconds=worst_case_total,
        safety_margin=0.30,
        safety_adjusted_total_seconds=adjusted_total,
        declared_runtime_limit_seconds=runtime_limit,
        peak_disk_bytes=peak_disk,
        usable_disk_quota_bytes=disk_quota,
        remaining_disk_bytes=disk_quota - peak_disk,
    )


@dataclass(frozen=True)
class ModelInventoryItem:
    model_id: str
    revision: str
    role: str
    parameter_count: int | None
    artifact_hash: str
    quantization: str
    active: bool
    source: str = ""


def validate_model_budget(
    items: Iterable[ModelInventoryItem],
    limit: int = 9_000_000_000,
    warning: int = 8_500_000_000,
) -> dict[str, Any]:
    """Deduplicate active weight sets and enforce the hard parameter budget."""

    if (
        isinstance(limit, bool)
        or not isinstance(limit, int)
        or limit <= 0
        or isinstance(warning, bool)
        or not isinstance(warning, int)
        or warning <= 0
        or warning > limit
    ):
        raise PipelineContractError(
            "E_MODEL_BUDGET_CONFIG",
            "Model limit and warning must be positive exact integers with warning <= limit.",
            context={"reason": "invalid_model_budget_config"},
            next_action="Provide exact integer model parameter thresholds.",
        )
    inventory = list(items)
    for item in inventory:
        if not isinstance(item, ModelInventoryItem):
            raise PipelineContractError(
                "E_MODEL_INVENTORY_INVALID",
                "Every model inventory element must use the typed inventory schema.",
                context={"reason": "invalid_inventory_element"},
                next_action="Regenerate the inventory with ModelInventoryItem records.",
            )
        if not isinstance(item.active, bool):
            raise PipelineContractError(
                "E_MODEL_INVENTORY_ACTIVE",
                "Each model inventory active flag must be an exact boolean.",
                context={"reason": "invalid_active_flag"},
                next_action="Set every model inventory active field to true or false.",
            )
    active_items = [item for item in inventory if item.active]
    normalized_items: list[ModelInventoryItem] = []
    for item in active_items:
        string_fields = {
            "model_id": item.model_id,
            "revision": item.revision,
            "role": item.role,
            "artifact_hash": item.artifact_hash,
            "quantization": item.quantization,
            "source": item.source,
        }
        if any(not isinstance(value, str) for value in string_fields.values()):
            raise PipelineContractError(
                "E_MODEL_INVENTORY_INVALID",
                "Active model inventory string fields must have exact string types.",
                context={"reason": "invalid_inventory_string_field"},
                next_action="Regenerate the typed model inventory.",
            )
        normalized_strings = {name: value.strip() for name, value in string_fields.items()}
        artifact_hash = normalized_strings["artifact_hash"].casefold()
        source = normalized_strings["source"]
        if (
            any(not normalized_strings[name] for name in ("model_id", "revision", "role", "quantization", "source"))
            or re.fullmatch(r"[0-9a-f]{64}", artifact_hash) is None
        ):
            raise PipelineContractError(
                "E_MODEL_INVENTORY_INVALID",
                "Active model inventory requires non-empty identity fields and a full SHA-256 hash.",
                context={"reason": "missing_identity_or_invalid_sha256"},
                next_action="Record the immutable model source and 64-character SHA-256 weight-set hash.",
            )
        if (
            isinstance(item.parameter_count, bool)
            or not isinstance(item.parameter_count, int)
            or item.parameter_count <= 0
        ):
            raise PipelineContractError(
                "E_MODEL_SIZE_UNKNOWN",
                f"Active model {item.model_id!r} has no valid parameter count.",
                context={"model_id": item.model_id, "revision": item.revision, "role": item.role},
                next_action="Record a positive parameter_count in the model inventory.",
            )
        normalized_items.append(
            replace(
                item,
                model_id=normalized_strings["model_id"],
                revision=normalized_strings["revision"],
                role=normalized_strings["role"],
                artifact_hash=artifact_hash,
                quantization=normalized_strings["quantization"],
                source=source,
            )
        )

    hashes_by_identity: dict[tuple[str, str, str, str], set[str]] = {}
    for item in normalized_items:
        identity = (item.model_id, item.revision, item.quantization, item.source)
        hashes_by_identity.setdefault(identity, set()).add(item.artifact_hash)
    if any(len(hashes) > 1 for hashes in hashes_by_identity.values()):
        raise PipelineContractError(
            "E_MODEL_INVENTORY_CONFLICT",
            "One model identity maps to conflicting artifact hashes.",
            context={"reason": "conflicting_artifact_hash"},
            next_action="Regenerate the inventory from one immutable weight artifact.",
        )

    grouped: dict[str, list[ModelInventoryItem]] = {}
    for item in normalized_items:
        grouped.setdefault(item.artifact_hash, []).append(item)

    unique: list[ModelInventoryItem] = []
    for artifact_hash in sorted(grouped):
        group = grouped[artifact_hash]
        first = group[0]
        expected = (
            first.model_id,
            first.revision,
            first.parameter_count,
            first.quantization,
            first.source,
        )
        if any(
            (
                item.model_id,
                item.revision,
                item.parameter_count,
                item.quantization,
                item.source,
            )
            != expected
            for item in group[1:]
        ):
            raise PipelineContractError(
                "E_MODEL_INVENTORY_CONFLICT",
                "Duplicate artifact hash has contradictory inventory metadata.",
                context={
                    "artifact_hash": artifact_hash,
                    "model_id": first.model_id,
                    "reason": "contradictory_duplicate",
                },
                next_action="Correct the registry so one hash has one identity, size, and quantization.",
            )
        unique.append(first)

    total = sum(item.parameter_count for item in unique)
    if total > limit:
        raise PipelineContractError(
            "E_MODEL_OVER_9B",
            f"Active unique model weights total {total:,} parameters, above the {limit:,} limit.",
            context={"total_parameters": total, "limit": limit},
            next_action="Disable an optional model or reuse an already-counted weight set.",
        )
    warning_exceeded = total >= warning
    return {
        "unique_items": [asdict(item) for item in unique],
        "total_parameters": total,
        "limit": limit,
        "remaining": limit - total,
        "warning": warning_exceeded,
        "warning_threshold": warning,
        "warning_exceeded": warning_exceeded,
    }


def _error_code(kind: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", "_", str(kind).upper()).strip("_") or "SOURCE"
    return f"E_{normalized}_{suffix}"


class RunMode(str, Enum):
    FULL = "full"
    RESUME = "resume"
    INFERENCE_ONLY = "inference_only"


class SourceRole(str, Enum):
    INFERENCE_INPUT = "INFERENCE_INPUT"
    TRAIN_CORPUS = "TRAIN_CORPUS"
    RUNTIME_KB = "RUNTIME_KB"
    NER_BASE = "NER_BASE"
    FINAL_MODEL_ARTIFACT = "FINAL_MODEL_ARTIFACT"
    EMBEDDING_MODEL = "EMBEDDING_MODEL"
    QWEN_MODEL = "QWEN_MODEL"
    WHEELHOUSE = "WHEELHOUSE"
    RESUME_BUNDLE = "RESUME_BUNDLE"


class InstallMode(str, Enum):
    PREINSTALLED = "preinstalled"
    ONLINE_LOCKED = "online_locked"
    OFFLINE_WHEELHOUSE = "offline_wheelhouse"


class SourceRequirement(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    FORBIDDEN = "forbidden"


def source_role_requirement(
    run_mode: RunMode,
    role: SourceRole,
    *,
    install_mode: InstallMode,
    resume_bundle_has_ner_base: bool,
    final_artifact_self_contained: bool,
) -> SourceRequirement:
    if not isinstance(resume_bundle_has_ner_base, bool) or not isinstance(
        final_artifact_self_contained, bool
    ):
        raise PipelineContractError(
            "E_SOURCE_CONTROL_INVALID",
            "Source-role condition flags must be exact booleans.",
            context={"reason": "invalid_source_condition"},
            next_action="Set bundle and final-artifact sufficiency flags to true or false.",
        )
    run_mode = RunMode(run_mode)
    role = SourceRole(role)
    install_mode = InstallMode(install_mode)
    if role is SourceRole.WHEELHOUSE:
        return (
            SourceRequirement.REQUIRED
            if install_mode is InstallMode.OFFLINE_WHEELHOUSE
            else SourceRequirement.FORBIDDEN
        )
    matrix: dict[RunMode, dict[SourceRole, SourceRequirement]] = {
        RunMode.FULL: {
            SourceRole.INFERENCE_INPUT: SourceRequirement.REQUIRED,
            SourceRole.TRAIN_CORPUS: SourceRequirement.REQUIRED,
            SourceRole.RUNTIME_KB: SourceRequirement.REQUIRED,
            SourceRole.NER_BASE: SourceRequirement.REQUIRED,
            SourceRole.FINAL_MODEL_ARTIFACT: SourceRequirement.FORBIDDEN,
            SourceRole.EMBEDDING_MODEL: SourceRequirement.OPTIONAL,
            SourceRole.QWEN_MODEL: SourceRequirement.OPTIONAL,
            SourceRole.RESUME_BUNDLE: SourceRequirement.FORBIDDEN,
        },
        RunMode.RESUME: {
            SourceRole.INFERENCE_INPUT: SourceRequirement.REQUIRED,
            SourceRole.TRAIN_CORPUS: SourceRequirement.REQUIRED,
            SourceRole.RUNTIME_KB: SourceRequirement.REQUIRED,
            SourceRole.NER_BASE: (
                SourceRequirement.OPTIONAL
                if resume_bundle_has_ner_base
                else SourceRequirement.REQUIRED
            ),
            SourceRole.FINAL_MODEL_ARTIFACT: SourceRequirement.OPTIONAL,
            SourceRole.EMBEDDING_MODEL: SourceRequirement.OPTIONAL,
            SourceRole.QWEN_MODEL: SourceRequirement.OPTIONAL,
            SourceRole.RESUME_BUNDLE: SourceRequirement.REQUIRED,
        },
        RunMode.INFERENCE_ONLY: {
            SourceRole.INFERENCE_INPUT: SourceRequirement.REQUIRED,
            SourceRole.TRAIN_CORPUS: SourceRequirement.FORBIDDEN,
            SourceRole.RUNTIME_KB: SourceRequirement.REQUIRED,
            SourceRole.NER_BASE: (
                SourceRequirement.FORBIDDEN
                if final_artifact_self_contained
                else SourceRequirement.REQUIRED
            ),
            SourceRole.FINAL_MODEL_ARTIFACT: SourceRequirement.REQUIRED,
            SourceRole.EMBEDDING_MODEL: SourceRequirement.OPTIONAL,
            SourceRole.QWEN_MODEL: SourceRequirement.OPTIONAL,
            SourceRole.RESUME_BUNDLE: SourceRequirement.FORBIDDEN,
        },
    }
    return matrix[run_mode][role]


def _safe_path_ref(
    path: Path,
    trusted_roots: Mapping[str, str | os.PathLike[str]] | None,
) -> SafePathRef:
    canonical = path.resolve()
    canonical_text = str(canonical)
    path_hash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()
    matching_roots: list[tuple[int, str, Path, Path]] = []
    for alias, root_value in dict(trusted_roots or {}).items():
        safe_alias = SafeIdentifier(str(alias)).value
        root = Path(root_value).expanduser().resolve()
        try:
            relative = canonical.relative_to(root)
        except ValueError:
            continue
        matching_roots.append((len(root.parts), safe_alias, root, relative))
    if matching_roots:
        _, alias, _, relative = max(matching_roots, key=lambda item: item[0])
        relative_hash = hashlib.sha256(relative.as_posix().encode("utf-8")).hexdigest()[:16]
        return SafePathRef(alias, f"src-{relative_hash}", path_hash)
    return SafePathRef("untrusted", f"src-{path_hash[:16]}", path_hash)


@dataclass(frozen=True)
class SourceCandidateDecision:
    path: Path
    accepted: bool
    reason: EventReason
    path_ref: SafePathRef

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": {
                "root_alias": self.path_ref.root_alias,
                "source_id": self.path_ref.source_id,
                "path_hash": self.path_ref.path_hash,
            },
            "accepted": self.accepted,
            "reason": self.reason.value,
        }

    def as_event_context(self) -> dict[str, Any]:
        return {"path": self.path_ref, "accepted": self.accepted, "reason": self.reason}


@dataclass(frozen=True)
class SourceResolution:
    selected: Path
    decisions: tuple[SourceCandidateDecision, ...]


@dataclass(frozen=True)
class RoleSourceResolution:
    role: SourceRole
    requirement: SourceRequirement
    selected: Path | None
    decisions: tuple[SourceCandidateDecision, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role.value,
            "requirement": self.requirement.value,
            "selected": next(
                (decision.as_dict()["path"] for decision in self.decisions if decision.path == self.selected),
                None,
            ),
            "decisions": [decision.as_dict() for decision in self.decisions],
        }

    def as_event_context(self) -> dict[str, Any]:
        selected_ref = next(
            (
                decision.path_ref
                for decision in self.decisions
                if decision.accepted and decision.path == self.selected
            ),
            None,
        )
        return {
            "role": SafeIdentifier(self.role.value),
            "requirement": SafeIdentifier(self.requirement.value),
            "selected": selected_ref,
            "decisions": [decision.as_event_context() for decision in self.decisions],
        }


@dataclass(frozen=True)
class SourceRoleResolution:
    run_mode: RunMode
    roles: tuple[RoleSourceResolution, ...]

    def for_role(self, role: SourceRole) -> RoleSourceResolution:
        normalized = SourceRole(role)
        return next(item for item in self.roles if item.role is normalized)

    def as_dict(self) -> dict[str, Any]:
        return {"run_mode": self.run_mode.value, "roles": [item.as_dict() for item in self.roles]}

    def as_event_context(self) -> dict[str, Any]:
        return {
            "run_mode": SafeIdentifier(self.run_mode.value),
            "roles": [item.as_event_context() for item in self.roles],
        }


def _evaluate_candidate(
    path: Path,
    validator: Callable[[Path], bool],
    trusted_roots: Mapping[str, str | os.PathLike[str]] | None,
) -> SourceCandidateDecision:
    path_ref = _safe_path_ref(path, trusted_roots)
    if not path.exists():
        return SourceCandidateDecision(path, False, EventReason.SOURCE_MISSING, path_ref)
    try:
        accepted = bool(validator(path))
    except Exception:
        return SourceCandidateDecision(path, False, EventReason.SOURCE_VALIDATOR_ERROR, path_ref)
    reason = EventReason.SOURCE_ACCEPTED if accepted else EventReason.SOURCE_VALIDATOR_REJECTED
    return SourceCandidateDecision(path, accepted, reason, path_ref)


def _decision_context(decisions: Sequence[SourceCandidateDecision]) -> list[dict[str, Any]]:
    return [decision.as_event_context() for decision in decisions]


def _emit_source_resolution(
    logger: RuntimeEventLogger | None,
    *,
    phase: str,
    attempt: int,
    kind: str,
    selected: SafePathRef | None,
    decisions: Sequence[SourceCandidateDecision],
    error: PipelineContractError | None = None,
) -> None:
    if logger is None:
        return
    logger.emit(
        phase,
        "SOURCE_RESOLVED",
        "ERROR" if error is not None else "SUCCESS",
        attempt=attempt,
        context={
            "kind": SafeIdentifier(_error_code(kind, "").removeprefix("E_").strip("_").casefold()),
            "selected": selected,
            "decisions": _decision_context(decisions),
        },
        error=error,
    )


def resolve_unique_source(
    kind: str,
    override: str | os.PathLike[str] | None,
    candidates: Iterable[str | os.PathLike[str]],
    validator: Callable[[Path], bool],
    *,
    logger: RuntimeEventLogger | None = None,
    phase: str = "source_resolution",
    attempt: int = 1,
    return_decisions: bool = False,
    trusted_roots: Mapping[str, str | os.PathLike[str]] | None = None,
) -> Path | SourceResolution:
    """Resolve exactly one validated source, with strict override semantics."""

    if override is not None and str(override).strip():
        raw_override = Path(override).expanduser()
        try:
            selected = raw_override.resolve()
            decision = _evaluate_candidate(selected, validator, trusted_roots)
        except Exception:
            selected = raw_override
            decision = SourceCandidateDecision(
                raw_override,
                False,
                EventReason.SOURCE_RESOLUTION_ERROR,
                _safe_path_ref(raw_override.absolute(), trusted_roots),
            )
        decisions = (decision,)
        if not decision.accepted:
            contract_error = PipelineContractError(
                _error_code(kind, "OVERRIDE_INVALID"),
                f"The explicit {kind} override is missing or invalid.",
                context={"override": str(selected), "decisions": _decision_context(decisions)},
                next_action=f"Fix or remove the explicit {kind} override; no fallback was attempted.",
            )
            _emit_source_resolution(
                logger,
                phase=phase,
                attempt=attempt,
                kind=kind,
                selected=None,
                decisions=decisions,
                error=contract_error,
            )
            contract_error.source_decisions = decisions
            raise contract_error
        _emit_source_resolution(
            logger,
            phase=phase,
            attempt=attempt,
            kind=kind,
            selected=decision.path_ref,
            decisions=decisions,
        )
        resolution = SourceResolution(selected, decisions)
        return resolution if return_decisions else selected

    prepared: list[tuple[Path, SourceCandidateDecision | None]] = []
    for candidate in candidates:
        raw_path = Path(candidate).expanduser()
        try:
            prepared.append((raw_path.resolve(), None))
        except Exception:
            prepared.append(
                (
                    raw_path,
                    SourceCandidateDecision(
                        raw_path,
                        False,
                        EventReason.SOURCE_RESOLUTION_ERROR,
                        _safe_path_ref(raw_path.absolute(), trusted_roots),
                    ),
                )
            )
    prepared.sort(key=lambda item: (str(item[0]).casefold(), str(item[0])))
    paths = [path for path, _ in prepared]
    resolved: dict[str, Path] = {}
    seen_identities: set[str] = set()
    decisions_list: list[SourceCandidateDecision] = []
    for path, resolution_failure in prepared:
        if resolution_failure is not None:
            decisions_list.append(resolution_failure)
            continue
        identity = os.path.normcase(str(path))
        if identity in seen_identities:
            decisions_list.append(
                SourceCandidateDecision(
                    path,
                    False,
                    EventReason.SOURCE_DUPLICATE_PATH,
                    _safe_path_ref(path, trusted_roots),
                )
            )
            continue
        seen_identities.add(identity)
        decision = _evaluate_candidate(path, validator, trusted_roots)
        decisions_list.append(decision)
        if decision.accepted:
            resolved[identity] = path
    decisions = tuple(decisions_list)
    matches = sorted(resolved.values(), key=lambda path: (str(path).casefold(), str(path)))
    if not matches:
        contract_error = PipelineContractError(
            _error_code(kind, "MISSING"),
            f"No valid {kind} source was found.",
            context={"candidates": [str(path) for path in paths], "decisions": _decision_context(decisions)},
            next_action=f"Attach or explicitly configure exactly one valid {kind} source.",
        )
        _emit_source_resolution(
            logger,
            phase=phase,
            attempt=attempt,
            kind=kind,
            selected=None,
            decisions=decisions,
            error=contract_error,
        )
        contract_error.source_decisions = decisions
        raise contract_error
    if len(matches) > 1:
        contract_error = PipelineContractError(
            _error_code(kind, "AMBIGUOUS"),
            f"Multiple valid {kind} sources were found.",
            context={
                "candidates": [str(path) for path in matches],
                "decisions": _decision_context(decisions),
            },
            next_action=f"Set an explicit {kind} override to choose one source.",
        )
        _emit_source_resolution(
            logger,
            phase=phase,
            attempt=attempt,
            kind=kind,
            selected=None,
            decisions=decisions,
            error=contract_error,
        )
        contract_error.source_decisions = decisions
        raise contract_error
    selected = matches[0]
    _emit_source_resolution(
        logger,
        phase=phase,
        attempt=attempt,
        kind=kind,
        selected=_safe_path_ref(selected, trusted_roots),
        decisions=decisions,
    )
    resolution = SourceResolution(selected, decisions)
    return resolution if return_decisions else selected


def _resolve_role_input(
    role: SourceRole,
    requirement: SourceRequirement,
    override: str | os.PathLike[str] | None,
    candidates: Sequence[str | os.PathLike[str]],
    validator: Callable[[Path], bool] | None,
    trusted_roots: Mapping[str, str | os.PathLike[str]],
) -> RoleSourceResolution:
    if override is not None and str(override).strip():
        if validator is None:
            raise PipelineContractError(
                "E_SOURCE_VALIDATOR_MISSING",
                "A configured source role requires its validator.",
                context={"role": role.value},
                next_action="Provide the role-specific layout validator.",
            )
        resolution = resolve_unique_source(
            role.value,
            override,
            (),
            validator,
            return_decisions=True,
            trusted_roots=trusted_roots,
        )
        assert isinstance(resolution, SourceResolution)
        if resolution.decisions[0].path_ref.root_alias == "untrusted":
            error = PipelineContractError(
                "E_SOURCE_PATH_UNTRUSTED",
                "Resolved role source is outside all trusted roots.",
                context={"role": role.value},
                next_action="Attach the source below a configured trusted root.",
            )
            error.source_decisions = resolution.decisions
            raise error
        return RoleSourceResolution(role, requirement, resolution.selected, resolution.decisions)

    if not candidates:
        if requirement is SourceRequirement.REQUIRED:
            raise PipelineContractError(
                _error_code(role.value, "MISSING"),
                f"Required source role {role.value} is missing.",
                context={"role": role.value},
                next_action="Attach or override exactly one valid source for the required role.",
            )
        return RoleSourceResolution(role, requirement, None, ())
    if validator is None:
        raise PipelineContractError(
            "E_SOURCE_VALIDATOR_MISSING",
            "A discovered source role requires its validator.",
            context={"role": role.value},
            next_action="Provide the role-specific layout validator.",
        )

    prepared: list[tuple[Path, SourceCandidateDecision | None]] = []
    for candidate in candidates:
        raw_path = Path(candidate).expanduser()
        try:
            prepared.append((raw_path.resolve(), None))
        except Exception:
            prepared.append(
                (
                    raw_path,
                    SourceCandidateDecision(
                        raw_path,
                        False,
                        EventReason.SOURCE_RESOLUTION_ERROR,
                        _safe_path_ref(raw_path.absolute(), trusted_roots),
                    ),
                )
            )
    prepared.sort(key=lambda item: (str(item[0]).casefold(), str(item[0])))
    seen: set[str] = set()
    matches: list[Path] = []
    decisions: list[SourceCandidateDecision] = []
    for path, resolution_error in prepared:
        if resolution_error is not None:
            decisions.append(resolution_error)
            continue
        identity = os.path.normcase(str(path))
        if identity in seen:
            decisions.append(
                SourceCandidateDecision(
                    path,
                    False,
                    EventReason.SOURCE_DUPLICATE_PATH,
                    _safe_path_ref(path, trusted_roots),
                )
            )
            continue
        seen.add(identity)
        decision = _evaluate_candidate(path, validator, trusted_roots)
        decisions.append(decision)
        if decision.accepted:
            matches.append(path)
    if len(matches) > 1:
        error = PipelineContractError(
            _error_code(role.value, "AMBIGUOUS"),
            f"Source role {role.value} has multiple valid candidates.",
            context={"role": role.value, "decisions": tuple(decisions)},
            next_action="Set the role-specific override to exactly one source.",
        )
        error.source_decisions = tuple(decisions)
        raise error
    if not matches:
        if requirement is SourceRequirement.REQUIRED:
            error = PipelineContractError(
                _error_code(role.value, "MISSING"),
                f"Required source role {role.value} has no valid candidate.",
                context={"role": role.value, "decisions": tuple(decisions)},
                next_action="Attach exactly one valid source for the required role.",
            )
            error.source_decisions = tuple(decisions)
            raise error
        return RoleSourceResolution(role, requirement, None, tuple(decisions))
    selected = matches[0]
    selected_decision = next(decision for decision in decisions if decision.path == selected and decision.accepted)
    if selected_decision.path_ref.root_alias == "untrusted":
        error = PipelineContractError(
            "E_SOURCE_PATH_UNTRUSTED",
            "Resolved role source is outside all trusted roots.",
            context={"role": role.value},
            next_action="Attach the source below a configured trusted root.",
        )
        error.source_decisions = tuple(decisions)
        raise error
    return RoleSourceResolution(role, requirement, selected, tuple(decisions))


def resolve_source_roles(
    run_mode: RunMode,
    *,
    overrides: Mapping[SourceRole, str | os.PathLike[str] | None],
    candidates: Mapping[SourceRole, Iterable[str | os.PathLike[str]]],
    validators: Mapping[SourceRole, Callable[[Path], bool]],
    install_mode: InstallMode,
    resume_bundle_has_ner_base: bool,
    final_artifact_self_contained: bool,
    trusted_roots: Mapping[str, str | os.PathLike[str]],
    logger: RuntimeEventLogger | None = None,
    phase: str = "source_resolution",
    attempt: int = 1,
) -> SourceRoleResolution:
    """Resolve all roles, then emit one complete inventory before returning or raising."""

    mode = RunMode(run_mode)
    normalized_overrides = {SourceRole(role): value for role, value in overrides.items()}
    normalized_candidates = {
        SourceRole(role): tuple(values) for role, values in candidates.items()
    }
    normalized_validators = {SourceRole(role): validator for role, validator in validators.items()}
    requirements = {
        role: source_role_requirement(
            mode,
            role,
            install_mode=InstallMode(install_mode),
            resume_bundle_has_ner_base=resume_bundle_has_ner_base,
            final_artifact_self_contained=final_artifact_self_contained,
        )
        for role in SourceRole
    }

    role_results: list[RoleSourceResolution] = []
    errors: list[PipelineContractError] = []
    forbidden_errors: list[PipelineContractError] = []
    selected_by_identity: dict[str, SourceRole] = {}
    for role in SourceRole:
        requirement = requirements[role]
        if requirement is SourceRequirement.FORBIDDEN:
            forbidden_values: list[str | os.PathLike[str]] = []
            override = normalized_overrides.get(role)
            if override is not None and str(override).strip():
                forbidden_values.append(override)
            forbidden_values.extend(normalized_candidates.get(role, ()))
            forbidden_paths: list[Path] = []
            for value in forbidden_values:
                raw_path = Path(value).expanduser()
                try:
                    forbidden_paths.append(raw_path.resolve())
                except Exception:
                    forbidden_paths.append(raw_path.absolute())
            forbidden_paths.sort(key=lambda path: (str(path).casefold(), str(path)))
            decisions = tuple(
                SourceCandidateDecision(
                    path,
                    False,
                    EventReason.SOURCE_ROLE_FORBIDDEN,
                    _safe_path_ref(path, trusted_roots),
                )
                for path in forbidden_paths
            )
            role_results.append(RoleSourceResolution(role, requirement, None, decisions))
            if forbidden_paths:
                error = PipelineContractError(
                    "E_SOURCE_ROLE_FORBIDDEN",
                    f"Source role {role.value} is forbidden in run mode {mode.value}.",
                    context={"role": role.value, "reason": EventReason.SOURCE_ROLE_FORBIDDEN},
                    next_action="Detach the forbidden source or choose the correct run mode.",
                )
                error.source_decisions = decisions
                errors.append(error)
                forbidden_errors.append(error)
            continue
        try:
            role_result = _resolve_role_input(
                role,
                requirement,
                normalized_overrides.get(role),
                normalized_candidates.get(role, ()),
                normalized_validators.get(role),
                trusted_roots,
            )
        except PipelineContractError as error:
            decisions = tuple(getattr(error, "source_decisions", ()))
            role_result = RoleSourceResolution(role, requirement, None, decisions)
            errors.append(error)
        if role_result.selected is not None:
            identity = os.path.normcase(str(role_result.selected.resolve()))
            prior_role = selected_by_identity.get(identity)
            if prior_role is not None:
                errors.append(PipelineContractError(
                    "E_SOURCE_ROLE_COLLISION",
                    "One canonical path cannot satisfy multiple source roles.",
                    context={
                        "reason": EventReason.SOURCE_ROLE_COLLISION,
                        "roles": (
                            SafeIdentifier(prior_role.value),
                            SafeIdentifier(role.value),
                        ),
                    },
                    next_action="Attach distinct paths for each source role.",
                ))
            else:
                selected_by_identity[identity] = role
        role_results.append(role_result)
    resolution = SourceRoleResolution(mode, tuple(role_results))
    primary_error = forbidden_errors[0] if forbidden_errors else (errors[0] if errors else None)
    if primary_error is not None:
        primary_error.context.update(resolution.as_event_context())
    if logger is not None:
        logger.emit(
            phase,
            "SOURCE_RESOLVED",
            "ERROR" if primary_error is not None else "SUCCESS",
            attempt=attempt,
            context=resolution.as_event_context(),
            error=primary_error,
        )
    if primary_error is not None:
        raise primary_error
    return resolution


__all__ = [
    "DEFAULT_QWEN_MODEL_ID",
    "QWEN_7B_MODEL_ID",
    "RUNTIME_CONTROL_API_VERSION",
    "EventReason",
    "InstallMode",
    "ModelInventoryItem",
    "PipelineContractError",
    "QwenCudaOomError",
    "QwenProfile",
    "QwenRuntimeProbe",
    "ResourceBudgetEstimate",
    "ResourcePlan",
    "RoleSourceResolution",
    "RunMode",
    "RuntimeEventLogger",
    "SafeHash",
    "SafeIdentifier",
    "SafePathRef",
    "SourceCandidateDecision",
    "SourceResolution",
    "SourceRequirement",
    "SourceRole",
    "SourceRoleResolution",
    "StageResourceEstimate",
    "StageResourceInput",
    "atomic_write_json",
    "choose_resource_plan",
    "estimate_resource_budget",
    "execute_optional_qwen",
    "hardware_snapshot",
    "oom_retry_plan",
    "probe_qwen_runtime",
    "resolve_source_roles",
    "resolve_unique_source",
    "runtime_control_migration_report",
    "source_role_requirement",
    "validate_model_budget",
]
