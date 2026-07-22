"""Resource-safe runtime controls for the Kaggle pipeline.

This module deliberately has no import-time dependency on torch, transformers, or
psutil.  It is safe to import in CPU-only validation and notebook preflight code.
"""

from __future__ import annotations

import importlib
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
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


_EVENT_PREFIX = "[CLINICAL_PIPELINE] "
_GIB = 1024**3
_REDACTED = "[REDACTED]"
_SENSITIVE_KEY_PATTERN = re.compile(
    r"clinical|patient|document.*text|raw|text|token|secret|password|prompt|"
    r"credential|authorization|api.?key|content",
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
    "selected",
    "source",
    "total_gib",
    "total_parameters",
    "to_attempt",
    "train_batch_size",
    "warning_threshold",
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

DEFAULT_QWEN_MODEL_ID = "Qwen/Qwen2.5-3B-Instruct-AWQ"
QWEN_7B_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"


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


def _is_safe_context_key(key: str) -> bool:
    normalized = key.casefold()
    if _SENSITIVE_KEY_PATTERN.search(normalized):
        return False
    return normalized in _SAFE_CONTEXT_KEYS or normalized.endswith(_SAFE_CONTEXT_SUFFIXES)


def _safe_context_value(value: Any, *, key: str) -> Any:
    if not _is_safe_context_key(key):
        return _REDACTED
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _safe_context_value(item, key=str(item_key))
            for item_key, item in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_context_value(item, key=key) for item in value]
    return f"<{type(value).__name__}>"


def _safe_context(context: Mapping[str, Any] | None) -> dict[str, Any]:
    return {
        str(key): _safe_context_value(value, key=str(key))
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
        self._lock = threading.Lock()
        self._owner_pid = os.getpid()
        self._attempt_states: dict[tuple[str, int], str] = {}
        self._retry_transitions: set[tuple[str, int, int]] = set()
        self._aggregate_terminals: set[str] = set()

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
            return ("ATTEMPT_ERROR_RETRIABLE" if retriable else "ATTEMPT_ERROR", key)
        return None

    def _commit_lifecycle(self, transition: tuple[str, Any] | None) -> None:
        if transition is None:
            return
        action, key = transition
        if action == "ATTEMPT_STARTED":
            self._attempt_states[key] = "STARTED"
        elif action == "ATTEMPT_END":
            self._attempt_states[key] = "END"
        elif action == "ATTEMPT_ERROR":
            self._attempt_states[key] = "ERROR"
        elif action == "ATTEMPT_ERROR_RETRIABLE":
            self._attempt_states[key] = "ERROR_RETRIABLE"
        elif action == "RETRY_TRANSITION":
            self._retry_transitions.add(key)
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
        phase_value = _validated_identifier(phase, "phase")
        event_value = _validated_identifier(event, "event")
        status_value = _validated_identifier(status, "status")
        scope_value = _validated_identifier(scope, "scope")
        resources = hardware_snapshot()
        error_data = _error_payload(error)
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
class QwenProfile:
    model_id: str = DEFAULT_QWEN_MODEL_ID
    gpu_memory_utilization: float = 0.40
    max_model_len: int = 1024
    batch_ladder: tuple[int, ...] = (8, 4, 1)


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
    qwen_disabled_reason: str = ""


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
    kernel_probe: bool | None,
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
    if kernel_probe is not True:
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


def choose_resource_plan(
    snapshot: Mapping[str, Any],
    *,
    require_gpu: bool,
    qwen_requested: bool,
    fast_dev_run: bool = False,
    kernel_probe: bool | None = None,
    qwen_model_id: str | None = None,
    allow_qwen_7b_override: bool = False,
    min_host_ram_gib: float = 10.0,
    min_disk_free_gib: float = 15.0,
) -> ResourcePlan:
    """Validate the hardware budget and choose the single-GPU safe profile."""

    gpu = _gpu_details(snapshot)
    total = gpu.get("total_gib")
    free = gpu.get("free_gib")
    gpu_available = bool(gpu.get("available", total is not None))
    selected_qwen_model = str(qwen_model_id or DEFAULT_QWEN_MODEL_ID)

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

    total_measurement = _finite_measurement(total)
    free_measurement = _finite_measurement(free)
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
    host = host if isinstance(host, Mapping) else {}
    host_available = _finite_measurement(host.get("ram_available_gib"))
    disk_free = _finite_measurement(host.get("disk_free_gib"))
    if require_gpu and (host_available is None or host_available < float(min_host_ram_gib)):
        raise PipelineContractError(
            "E_RUNTIME_BUDGET",
            "Training host RAM admission failed.",
            context={
                "ram_available_gib": host_available,
                "min_host_ram_gib": float(min_host_ram_gib),
            },
            next_action="Free host RAM or attach a runtime with more memory.",
        )
    if require_gpu and (disk_free is None or disk_free < float(min_disk_free_gib)):
        raise PipelineContractError(
            "E_DISK_BUDGET",
            "Training disk admission failed.",
            context={
                "disk_free_gib": disk_free,
                "min_disk_free_gib": float(min_disk_free_gib),
            },
            next_action="Free working disk space or increase the declared disk quota.",
        )

    reason = ""
    if qwen_requested:
        reason = _qwen_disabled_reason(
            gpu,
            kernel_probe=kernel_probe,
            model_id=selected_qwen_model,
            allow_qwen_7b_override=allow_qwen_7b_override,
        )
    qwen_enabled = bool(qwen_requested and not reason)
    qwen_profile = QwenProfile(model_id=selected_qwen_model) if qwen_enabled else None
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
        qwen_disabled_reason=reason,
    )


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

    active_items = [item for item in items if item.active]
    normalized_items: list[ModelInventoryItem] = []
    for item in active_items:
        artifact_hash = item.artifact_hash.strip().casefold()
        source = item.source.strip()
        if not source or re.fullmatch(r"[0-9a-f]{64}", artifact_hash) is None:
            raise PipelineContractError(
                "E_MODEL_INVENTORY_INVALID",
                f"Active model {item.model_id!r} requires source and a full SHA-256 artifact hash.",
                context={
                    "model_id": item.model_id,
                    "revision": item.revision,
                    "role": item.role,
                    "reason": "missing_source_or_invalid_sha256",
                },
                next_action="Record the immutable model source and 64-character SHA-256 weight-set hash.",
            )
        if item.parameter_count is None or item.parameter_count <= 0:
            raise PipelineContractError(
                "E_MODEL_SIZE_UNKNOWN",
                f"Active model {item.model_id!r} has no valid parameter count.",
                context={"model_id": item.model_id, "revision": item.revision, "role": item.role},
                next_action="Record a positive parameter_count in the model inventory.",
            )
        normalized_items.append(replace(item, artifact_hash=artifact_hash, source=source))

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

    total = sum(int(item.parameter_count or 0) for item in unique)
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


@dataclass(frozen=True)
class SourceCandidateDecision:
    path: Path
    accepted: bool
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return {"path": str(self.path), "accepted": self.accepted, "reason": self.reason}


@dataclass(frozen=True)
class SourceResolution:
    selected: Path
    decisions: tuple[SourceCandidateDecision, ...]


def _evaluate_candidate(path: Path, validator: Callable[[Path], bool]) -> SourceCandidateDecision:
    if not path.exists():
        return SourceCandidateDecision(path, False, "missing")
    try:
        accepted = bool(validator(path))
    except Exception as error:
        return SourceCandidateDecision(path, False, f"validator_error:{type(error).__name__}")
    return SourceCandidateDecision(path, accepted, "accepted" if accepted else "validator_rejected")


def _decision_context(decisions: Sequence[SourceCandidateDecision]) -> list[dict[str, Any]]:
    return [decision.as_dict() for decision in decisions]


def _emit_source_resolution(
    logger: RuntimeEventLogger | None,
    *,
    phase: str,
    attempt: int,
    kind: str,
    selected: Path | None,
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
            "kind": kind,
            "selected": str(selected) if selected is not None else None,
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
) -> Path | SourceResolution:
    """Resolve exactly one validated source, with strict override semantics."""

    if override is not None and str(override).strip():
        raw_override = Path(override).expanduser()
        try:
            selected = raw_override.resolve()
            decision = _evaluate_candidate(selected, validator)
        except Exception as error:
            selected = raw_override
            decision = SourceCandidateDecision(
                raw_override,
                False,
                f"resolution_error:{type(error).__name__}",
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
            raise contract_error
        _emit_source_resolution(
            logger,
            phase=phase,
            attempt=attempt,
            kind=kind,
            selected=selected,
            decisions=decisions,
        )
        resolution = SourceResolution(selected, decisions)
        return resolution if return_decisions else selected

    prepared: list[tuple[Path, SourceCandidateDecision | None]] = []
    for candidate in candidates:
        raw_path = Path(candidate).expanduser()
        try:
            prepared.append((raw_path.resolve(), None))
        except Exception as error:
            prepared.append(
                (
                    raw_path,
                    SourceCandidateDecision(
                        raw_path,
                        False,
                        f"resolution_error:{type(error).__name__}",
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
            decisions_list.append(SourceCandidateDecision(path, False, "duplicate_path"))
            continue
        seen_identities.add(identity)
        decision = _evaluate_candidate(path, validator)
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
        raise contract_error
    selected = matches[0]
    _emit_source_resolution(
        logger,
        phase=phase,
        attempt=attempt,
        kind=kind,
        selected=selected,
        decisions=decisions,
    )
    resolution = SourceResolution(selected, decisions)
    return resolution if return_decisions else selected


__all__ = [
    "ModelInventoryItem",
    "PipelineContractError",
    "QwenProfile",
    "ResourcePlan",
    "RuntimeEventLogger",
    "SourceCandidateDecision",
    "SourceResolution",
    "DEFAULT_QWEN_MODEL_ID",
    "QWEN_7B_MODEL_ID",
    "atomic_write_json",
    "choose_resource_plan",
    "hardware_snapshot",
    "oom_retry_plan",
    "resolve_unique_source",
    "validate_model_budget",
]
