"""Resource-safe runtime controls for the Kaggle pipeline.

This module deliberately has no import-time dependency on torch, transformers, or
psutil.  It is safe to import in CPU-only validation and notebook preflight code.
"""

from __future__ import annotations

import importlib
import json
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
_SENSITIVE_KEYS = {
    "api_key",
    "clinical_text",
    "document_text",
    "password",
    "prompt",
    "raw_text",
    "secret",
    "text",
    "token",
    "tokens",
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
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.context = dict(context or {})
        self.next_action = next_action


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
    except BaseException:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return destination


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

    try:
        torch = importlib.import_module("torch")
        cuda = torch.cuda
        if bool(cuda.is_available()):
            device = int(cuda.current_device())
            properties = cuda.get_device_properties(device)
            free_bytes, total_bytes = cuda.mem_get_info(device)
            capability = cuda.get_device_capability(device)
            name = getattr(properties, "name", None)
            if name is None:
                name = cuda.get_device_name(device)
            gpu.update(
                {
                    "available": True,
                    "name": str(name),
                    "capability": ".".join(str(part) for part in capability),
                    "free_gib": _gib(free_bytes),
                    "total_gib": _gib(total_bytes),
                    "allocated_gib": _gib(cuda.memory_allocated(device)),
                    "reserved_gib": _gib(cuda.memory_reserved(device)),
                    "peak_gib": _gib(cuda.max_memory_allocated(device)),
                }
            )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        pass

    try:
        psutil = importlib.import_module("psutil")
        host["ram_available_gib"] = _gib(psutil.virtual_memory().available)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError, OSError):
        pass

    try:
        host["disk_free_gib"] = _gib(shutil.disk_usage(Path.cwd()).free)
    except OSError:
        pass
    return snapshot


def _json_safe(value: Any, *, key: str | None = None) -> Any:
    if key is not None and key.casefold() in _SENSITIVE_KEYS:
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(item_key): _json_safe(item, key=str(item_key)) for item_key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    return f"<{type(value).__name__}>"


def _error_payload(error: BaseException | Mapping[str, Any] | None) -> dict[str, Any] | None:
    if error is None:
        return None
    if isinstance(error, Mapping):
        return _json_safe(error)
    if isinstance(error, PipelineContractError):
        return {
            "code": error.code,
            "type": type(error).__name__,
            "message": error.message,
            "retriable": False,
            "next_action": error.next_action,
            "context": _json_safe(error.context),
        }
    return {
        "code": "E_UNEXPECTED",
        "type": type(error).__name__,
        "message": str(error),
        "retriable": False,
        "next_action": "Inspect diagnostics and fix the reported phase failure.",
    }


class RuntimeEventLogger:
    """Emit one-line structured events to stdout and an append-only JSONL file."""

    def __init__(self, run_id: str, jsonl_path: str | os.PathLike[str]) -> None:
        self.run_id = str(run_id)
        self.jsonl_path = Path(jsonl_path)
        self._lock = threading.Lock()

    def _emit(
        self,
        phase: str,
        event: str,
        status: str,
        *,
        attempt: int,
        context: Mapping[str, Any] | None,
        error: BaseException | Mapping[str, Any] | None,
        duration_ms: float | None,
    ) -> dict[str, Any]:
        resources = hardware_snapshot()
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "run_id": self.run_id,
            "phase": str(phase),
            "event": str(event),
            "status": str(status),
            "attempt": int(attempt),
            "duration_ms": duration_ms,
            "gpu": resources["gpu"],
            "host": resources["host"],
            "context": _json_safe(dict(context or {})),
        }
        error_data = _error_payload(error)
        if error_data is not None:
            payload["error"] = error_data
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

        with self._lock:
            self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            with self.jsonl_path.open("a", encoding="utf-8", newline="\n") as handle:
                handle.write(line + "\n")
                handle.flush()
            print(_EVENT_PREFIX + line, flush=True)
        return payload

    def emit(
        self,
        phase: str,
        event: str,
        status: str,
        *,
        attempt: int = 1,
        context: Mapping[str, Any] | None = None,
        error: BaseException | Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self._emit(
            phase,
            event,
            status,
            attempt=attempt,
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
                attempt=attempt,
                context=context,
                error=None,
                duration_ms=duration_ms,
            )


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
    qwen_profile: dict[str, Any] | None
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


def choose_resource_plan(
    snapshot: Mapping[str, Any],
    *,
    require_gpu: bool,
    qwen_requested: bool,
    fast_dev_run: bool = False,
) -> ResourcePlan:
    """Validate the hardware budget and choose the single-GPU safe profile."""

    gpu = _gpu_details(snapshot)
    total = gpu.get("total_gib")
    free = gpu.get("free_gib")
    gpu_available = bool(gpu.get("available", total is not None))

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

    if total is None or free is None or float(total) < 14.0 or float(free) < 12.0:
        raise PipelineContractError(
            "E_GPU_BUDGET",
            "GPU budget requires at least 14 GiB total and 12 GiB free VRAM.",
            context={"total_gib": total, "free_gib": free},
            next_action="Release GPU memory or attach a GPU that satisfies the preflight budget.",
        )

    p100 = _is_p100(gpu)
    qwen_enabled = bool(qwen_requested and not p100)
    qwen_profile = (
        {"gpu_memory_utilization": 0.40, "max_model_len": 2048, "batch_size": 16}
        if qwen_enabled
        else None
    )
    reason = "Qwen is disabled on NVIDIA P100 (compute capability 6.0)." if qwen_requested and p100 else ""
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


def validate_model_budget(
    items: Iterable[ModelInventoryItem],
    limit: int = 9_000_000_000,
    warning: int = 8_500_000_000,
) -> dict[str, Any]:
    """Deduplicate active weight sets and enforce the hard parameter budget."""

    unique: list[ModelInventoryItem] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        if not item.active:
            continue
        artifact_hash = item.artifact_hash.strip()
        key = ("hash", artifact_hash) if artifact_hash else ("model", item.model_id, item.revision)
        if key in seen:
            continue
        seen.add(key)
        if item.parameter_count is None or item.parameter_count <= 0:
            raise PipelineContractError(
                "E_MODEL_SIZE_UNKNOWN",
                f"Active model {item.model_id!r} has no valid parameter count.",
                context={"model_id": item.model_id, "revision": item.revision, "role": item.role},
                next_action="Record a positive parameter_count in the model inventory.",
            )
        unique.append(item)

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


def _accepted_candidate(path: Path, validator: Callable[[Path], bool]) -> bool:
    if not path.exists():
        return False
    try:
        return bool(validator(path))
    except (OSError, ValueError, TypeError):
        return False


def resolve_unique_source(
    kind: str,
    override: str | os.PathLike[str] | None,
    candidates: Iterable[str | os.PathLike[str]],
    validator: Callable[[Path], bool],
) -> Path:
    """Resolve exactly one validated source, with strict override semantics."""

    if override is not None and str(override).strip():
        selected = Path(override).expanduser().resolve()
        if not _accepted_candidate(selected, validator):
            raise PipelineContractError(
                _error_code(kind, "OVERRIDE_INVALID"),
                f"The explicit {kind} override is missing or invalid.",
                context={"override": str(selected)},
                next_action=f"Fix or remove the explicit {kind} override; no fallback was attempted.",
            )
        return selected

    resolved: dict[str, Path] = {}
    for candidate in candidates:
        path = Path(candidate).expanduser().resolve()
        key = str(path).casefold()
        if key not in resolved and _accepted_candidate(path, validator):
            resolved[key] = path
    matches = sorted(resolved.values(), key=lambda path: str(path).casefold())
    if not matches:
        raise PipelineContractError(
            _error_code(kind, "MISSING"),
            f"No valid {kind} source was found.",
            context={"candidates": []},
            next_action=f"Attach or explicitly configure exactly one valid {kind} source.",
        )
    if len(matches) > 1:
        raise PipelineContractError(
            _error_code(kind, "AMBIGUOUS"),
            f"Multiple valid {kind} sources were found.",
            context={"candidates": [str(path) for path in matches]},
            next_action=f"Set an explicit {kind} override to choose one source.",
        )
    return matches[0]


__all__ = [
    "ModelInventoryItem",
    "PipelineContractError",
    "ResourcePlan",
    "RuntimeEventLogger",
    "atomic_write_json",
    "choose_resource_plan",
    "hardware_snapshot",
    "oom_retry_plan",
    "resolve_unique_source",
    "validate_model_budget",
]
