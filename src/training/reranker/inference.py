"""Lazy local Qwen QLoRA generation backend."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from src.training.reranker.data import build_reranker_prompt


class LocalTransformersReranker:
    def __init__(
        self,
        model_artifact: str | Path,
        *,
        project_root: str | Path,
        max_new_tokens: int = 64,
    ):
        self.model_artifact = Path(model_artifact).resolve()
        self.project_root = Path(project_root).resolve()
        if (
            isinstance(max_new_tokens, bool)
            or not isinstance(max_new_tokens, int)
            or max_new_tokens < 1
        ):
            raise ValueError("max_new_tokens must be positive")
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _runtime_config(self) -> dict[str, Any]:
        path = self.model_artifact / "runtime.json"
        if not path.is_file():
            raise ValueError(f"reranker runtime manifest is missing: {path}")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid reranker runtime manifest") from exc
        if set(value) != {"schema_version", "base_model"} or value[
            "schema_version"
        ] != 1:
            raise ValueError("invalid reranker runtime schema")
        base_model = value["base_model"]
        if not isinstance(base_model, str) or not base_model:
            raise ValueError("reranker runtime base_model is invalid")
        path_value = Path(base_model)
        if path_value.is_absolute() or ".." in path_value.parts:
            raise ValueError("reranker runtime base_model must be project-relative")
        return value

    def _load(self):
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        import torch
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        runtime = self._runtime_config()
        base_model = (self.project_root / runtime["base_model"]).resolve()
        if not base_model.is_dir():
            raise ValueError(f"local Qwen base model is missing: {base_model}")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            local_files_only=True,
            use_fast=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            local_files_only=True,
            quantization_config=quantization,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        model = PeftModel.from_pretrained(
            base,
            self.model_artifact,
            local_files_only=True,
        )
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        return model, tokenizer

    def generate(self, example: Mapping[str, Any]) -> str:
        model, tokenizer = self._load()
        prompt = build_reranker_prompt(example)
        inputs = tokenizer(prompt, return_tensors="pt")
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }
        output = model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        generated = output[0, inputs["input_ids"].shape[1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()


__all__ = ["LocalTransformersReranker"]
