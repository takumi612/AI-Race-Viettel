from __future__ import annotations

import json
import logging
from typing import Any

from .vllm_compat import build_sampling_kwargs, iter_batches, parse_json_object


def _candidate_schema(candidate_ids: list[Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {"selected_id": {"enum": [*candidate_ids, None]}},
        "required": ["selected_id"],
    }


def _build_sampling_kwargs(
    sampling_params_class: Any,
    candidate_ids: list[Any],
    *,
    structured_outputs_factory: Any | None = None,
    guided_decoding_factory: Any | None = None,
) -> dict[str, Any]:
    return build_sampling_kwargs(
        sampling_params_class,
        _candidate_schema(candidate_ids),
        temperature=0.0,
        max_tokens=100,
        structured_outputs_factory=structured_outputs_factory,
        guided_decoding_factory=guided_decoding_factory,
    )


def _parse_selected_id(response_text: str, candidates: list[dict[str, Any]]) -> Any | None:
    payload = parse_json_object(response_text)
    if payload is None:
        return None
    selected_id = payload.get("selected_id")
    candidate_by_text = {str(item["candidate_id"]): item["candidate_id"] for item in candidates}
    return candidate_by_text.get(str(selected_id))


def _selection_warning_reason(response_text: str, candidates: list[dict[str, Any]]) -> str | None:
    payload = parse_json_object(response_text)
    if payload is None:
        return "invalid JSON"
    selected_id = payload.get("selected_id")
    if selected_id is None:
        return None
    if _parse_selected_id(response_text, candidates) is None:
        return "unknown selected_id"
    return None


class ClinicalLLMReranker:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct-AWQ",
        max_model_len: int = 1024,
        batch_size: int = 16,
        gpu_memory_utilization: float = 0.2,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.llm = None
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError("vllm is required for LLM reranking") from exc
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            quantization="awq" if "AWQ" in model_name else None,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,
        )

    def _build_prompt(
        self,
        context_text: str,
        entity_text: str,
        entity_type: str,
        candidates: list[dict[str, Any]],
    ) -> str:
        options_text = "".join(f"- ID: {c['candidate_id']} | Name: {c['name']}\n" for c in candidates)
        system_prompt = "You are a professional medical coding expert. Choose the best candidate code."
        user_prompt = (
            f'Clinical context:\n"""{context_text}"""\n\n'
            f"Entity: [{entity_text}] (Type: {entity_type})\n\n"
            f"Candidate codes:\n{options_text}\n"
            "Return JSON with `selected_id`, or null if no candidate fits."
        )
        return (
            f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
            f"<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

    def _build_json_schema(self, candidates: list[dict[str, Any]]) -> str:
        return json.dumps(_candidate_schema([candidate["candidate_id"] for candidate in candidates]))

    def rerank_batch(self, entity_queries: list[dict[str, Any]]) -> list[Any | None]:
        if not self.llm:
            raise RuntimeError("LLM is not initialized")
        from vllm import SamplingParams

        results: list[Any | None] = []
        for query_batch in iter_batches(entity_queries, self.batch_size):
            prompts = [
                self._build_prompt(
                    query["context_text"],
                    query["entity_text"],
                    query["entity_type"],
                    query["candidates"],
                )
                for query in query_batch
            ]
            sampling_params = [
                SamplingParams(
                    **_build_sampling_kwargs(
                        SamplingParams,
                        [candidate["candidate_id"] for candidate in query["candidates"]],
                    )
                )
                for query in query_batch
            ]
            outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            if len(outputs) != len(query_batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(query_batch)} rerank prompts")
            for query, output in zip(query_batch, outputs):
                generated_text = output.outputs[0].text
                selected_id = _parse_selected_id(generated_text, query["candidates"])
                warning_reason = _selection_warning_reason(generated_text, query["candidates"])
                if warning_reason is not None:
                    logging.warning(
                        "Could not parse a valid selected_id (%s) from LLM response: %s",
                        warning_reason,
                        generated_text[:300],
                    )
                results.append(selected_id)
        return results

    def destroy(self):
        import gc
        import torch

        try:
            import vllm.distributed.parallel_state as parallel_state

            is_initialized = getattr(parallel_state, "is_initialized", None)
            if is_initialized is None:
                is_initialized = getattr(parallel_state, "model_parallel_is_initialized", None)
            destroy_model_parallel = getattr(parallel_state, "destroy_model_parallel", None)
            if callable(is_initialized) and is_initialized() and callable(destroy_model_parallel):
                destroy_model_parallel()
        except (ImportError, AttributeError):
            pass
        del self.llm
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
