from __future__ import annotations

import inspect
import json
import re
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar


T = TypeVar("T")


def iter_batches(items: Sequence[T], batch_size: int) -> Iterator[Sequence[T]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def parse_json_object(response_text: str) -> dict[str, Any] | None:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, re.IGNORECASE | re.DOTALL)
    if fenced:
        payload_text = fenced.group(1)
    else:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start < 0 or end < start:
            return None
        payload_text = response_text[start : end + 1]
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _accepted_parameters(callable_object: Any) -> set[str]:
    accepted: set[str] = set()
    for target in (callable_object, getattr(callable_object, "__init__", None)):
        if target is None:
            continue
        try:
            signature = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        accepted.update(
            name
            for name, parameter in signature.parameters.items()
            if parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
        )
    return accepted


def build_sampling_kwargs(
    sampling_params_class: Any,
    schema: dict[str, Any],
    *,
    temperature: float = 0.0,
    max_tokens: int = 100,
    structured_outputs_factory: Any | None = None,
    guided_decoding_factory: Any | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    schema_text = json.dumps(schema)
    accepted = _accepted_parameters(sampling_params_class)

    if structured_outputs_factory is None or guided_decoding_factory is None:
        try:
            from vllm import sampling_params as vllm_sampling_params
        except ImportError:
            vllm_sampling_params = None
        if vllm_sampling_params is not None:
            structured_outputs_factory = structured_outputs_factory or getattr(
                vllm_sampling_params, "StructuredOutputsParams", None
            )
            guided_decoding_factory = guided_decoding_factory or getattr(
                vllm_sampling_params, "GuidedDecodingParams", None
            )

    if "structured_outputs" in accepted and structured_outputs_factory is not None:
        kwargs["structured_outputs"] = structured_outputs_factory(json=schema_text)
    elif "guided_decoding" in accepted and guided_decoding_factory is not None:
        kwargs["guided_decoding"] = guided_decoding_factory(json=schema_text)
    return kwargs
