from __future__ import annotations

from dataclasses import replace
from typing import Any

from .assertions import ClinicalLLMAssertionPredictor
from .entity_types import ASSERTION_ENTITY_TYPES
from .qwen_entity_validator import EntityValidationCounters, QwenEntityValidator
from .reranker import ClinicalLLMReranker
from .schema import EntityAnnotation


class RequiredQwenError(RuntimeError):
    pass


class RequiredQwenRefiner:
    def __init__(self, llm_engine: Any, *, batch_size: int = 64):
        self._llm_engine = llm_engine
        self.batch_size = batch_size
        self._entity_validator = QwenEntityValidator(llm_engine, batch_size=batch_size)
        self._entity_validation_counters = EntityValidationCounters()

    def validate_entities(
        self,
        entities: tuple[EntityAnnotation, ...],
        raw_text: str,
    ) -> tuple[EntityAnnotation, ...]:
        try:
            result = self._entity_validator.validate(entities, raw_text)
            self._entity_validation_counters.add(result.counters)
            return result.entities
        except RequiredQwenError:
            raise
        except Exception as exc:
            raise RequiredQwenError(str(exc)) from exc

    def validation_counters(self) -> dict[str, object]:
        return self._entity_validation_counters.to_dict()

    @staticmethod
    def _context(raw_text: str, entity: EntityAnnotation, window: int = 120) -> str:
        return raw_text[max(0, entity.start - window):min(len(raw_text), entity.end + window)]

    @staticmethod
    def _copy_entity(entity: EntityAnnotation) -> EntityAnnotation:
        return replace(
            entity,
            candidates=list(entity.candidates),
            assertions=list(entity.assertions),
            evidence=list(entity.evidence),
            ranked_candidates=[dict(candidate) for candidate in entity.ranked_candidates],
        )

    def _reranker(self) -> ClinicalLLMReranker:
        reranker = ClinicalLLMReranker.__new__(ClinicalLLMReranker)
        reranker.llm = self._llm_engine
        reranker.batch_size = self.batch_size
        return reranker

    def refine(
        self,
        entities: tuple[EntityAnnotation, ...],
        raw_text: str,
    ) -> tuple[EntityAnnotation, ...]:
        try:
            refined = [self._copy_entity(entity) for entity in entities]
            rerank_indices: list[int] = []
            rerank_queries: list[dict[str, Any]] = []
            for index, entity in enumerate(refined):
                if entity.type not in {"DISEASE", "DRUG"} or not entity.ranked_candidates:
                    continue
                if any("candidate_id" not in candidate for candidate in entity.ranked_candidates):
                    raise RequiredQwenError("ranked candidate is missing candidate_id")
                rerank_indices.append(index)
                rerank_queries.append(
                    {
                        "context_text": self._context(raw_text, entity),
                        "entity_text": entity.text,
                        "entity_type": entity.type,
                        "candidates": entity.ranked_candidates,
                    }
                )
            if rerank_queries:
                selected_ids = self._reranker().rerank_batch(rerank_queries, strict=True)
                if len(selected_ids) != len(rerank_indices):
                    raise RequiredQwenError("rerank response count mismatch")
                for index, selected_id in zip(rerank_indices, selected_ids):
                    refined[index].candidates = [] if selected_id is None else [str(selected_id)]

            assertion_indices = [
                index for index, entity in enumerate(refined) if entity.type in ASSERTION_ENTITY_TYPES
            ]
            if assertion_indices:
                assertion_queries = [
                    {"context": self._context(raw_text, refined[index]), "entity_text": refined[index].text}
                    for index in assertion_indices
                ]
                axes_values = ClinicalLLMAssertionPredictor(self._llm_engine).predict_batch(
                    assertion_queries,
                    batch_size=self.batch_size,
                    strict=True,
                )
                if len(axes_values) != len(assertion_indices):
                    raise RequiredQwenError("assertion response count mismatch")
                for index, axes in zip(assertion_indices, axes_values):
                    refined[index].assertions = axes.submission_labels()
            return tuple(refined)
        except RequiredQwenError:
            raise
        except Exception as exc:
            raise RequiredQwenError(str(exc)) from exc

    def destroy(self) -> None:
        engine = self._llm_engine
        self._llm_engine = None
        destroy = getattr(engine, "destroy", None)
        if callable(destroy):
            destroy()
