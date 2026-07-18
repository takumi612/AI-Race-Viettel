"""Optional local LLM subset reranker with deterministic fallback."""

from __future__ import annotations

import json
import logging
import os
import sqlite3

from src.utils.paths import KB_DIR

LOGGER = logging.getLogger(__name__)


class LLMReranker:
    def __init__(
        self,
        use_llm: bool = False,
        api_url: str = "http://localhost:8000/v1",
        api_key: str = "token-viettel-race",
        timeout_seconds: float = 30.0,
    ):
        self.use_llm = bool(use_llm)
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.timeout_seconds = float(timeout_seconds)
        self.db_path = os.path.join(KB_DIR, "metadata.db")

    @staticmethod
    def parse_selected_codes(payload, allowed_codes) -> list[str]:
        if isinstance(payload, (bytes, bytearray)):
            payload = payload.decode("utf-8")
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid LLM JSON payload") from exc
        if not isinstance(payload, dict):
            raise ValueError("LLM payload must be an object")
        allowed = [str(code) for code in allowed_codes]
        allowed_set = set(allowed)
        if "selected_codes" in payload:
            selected = payload["selected_codes"]
            if not isinstance(selected, list) or any(not isinstance(code, str) for code in selected):
                raise ValueError("selected_codes must be a list of strings")
        elif "best_code" in payload:
            selected = [payload["best_code"]]
            if not isinstance(payload["best_code"], str):
                raise ValueError("best_code must be a string")
        else:
            raise ValueError("LLM payload contains no selected codes")
        if any(code not in allowed_set for code in selected):
            raise ValueError("selected code is outside the candidate pool")
        result = []
        for code in selected:
            if code not in result:
                result.append(code)
        return result

    def _get_candidate_descriptions(self, table_name: str, codes: list[str]) -> dict[str, str]:
        descriptions: dict[str, str] = {}
        if not codes:
            return descriptions
        try:
            with sqlite3.connect(self.db_path) as connection:
                placeholders = ",".join("?" for _ in codes)
                if table_name.lower() == "icd10":
                    rows = connection.execute(
                        f"SELECT code, name_vi, name_en FROM icd10 WHERE code IN ({placeholders})", codes
                    ).fetchall()
                    descriptions.update({row[0]: f"{row[1]} ({row[2]})" if row[2] else row[1] for row in rows})
                else:
                    rows = connection.execute(
                        f"SELECT rxcui, name FROM rxnorm WHERE rxcui IN ({placeholders})", codes
                    ).fetchall()
                    descriptions.update({row[0]: row[1] for row in rows})
        except sqlite3.Error:
            pass
        return descriptions

    def rerank(self, text_context: str, entity_text: str, entity_type: str, candidates: list) -> list:
        candidates = list(candidates or [])
        if not candidates or not self.use_llm:
            return candidates
        table_name = "icd10" if str(entity_type).upper().startswith("CH") else "rxnorm"
        descriptions = self._get_candidate_descriptions(table_name, candidates)
        context = "\n".join(
            f"{index + 1}. code={code}; description={descriptions.get(code, '')}"
            for index, code in enumerate(candidates)
        )
        payload = {
            "model": "Qwen2.5-7B-Instruct",
            "messages": [
                {"role": "system", "content": "Select only codes from the supplied candidate pool and return JSON."},
                {"role": "user", "content": f"Context: {text_context}\nEntity: {entity_text}\nCandidates:\n{context}"},
            ],
            "temperature": 0.1,
            "max_tokens": 50,
            "response_format": {"type": "json_object"},
        }
        try:
            import requests

            response = requests.post(
                f"{self.api_url}/chat/completions",
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            if response.status_code != 200:
                return candidates
            content = response.json()["choices"][0]["message"]["content"]
            selected = self.parse_selected_codes(content, candidates)
            return selected
        except Exception as exc:
            LOGGER.warning("LLM reranker fallback: %s", exc)
            return candidates


__all__ = ["LLMReranker"]
