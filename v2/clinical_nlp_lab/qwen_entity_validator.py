"""Strict Qwen-backed validation for NER entity boundaries and types."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .entity_types import ENTITY_TYPE_TO_ID
from .schema import EntityAnnotation
from .vllm_compat import build_sampling_kwargs, iter_batches, parse_json_object


@dataclass(frozen=True, slots=True)
class EntityValidationDecision:
    action: Literal["keep", "drop", "trim"]
    relative_start: int | None = None
    relative_end: int | None = None
    entity_type: str | None = None


@dataclass(slots=True)
class EntityValidationCounters:
    query_count: int = 0
    keep: int = 0
    drop: int = 0
    trim: int = 0
    kb_bypass: int = 0
    before_type_counts: Counter[str] = field(default_factory=Counter)
    after_type_counts: Counter[str] = field(default_factory=Counter)
    before_length_buckets: Counter[str] = field(default_factory=Counter)
    after_length_buckets: Counter[str] = field(default_factory=Counter)
    max_before_length: int = 0
    max_after_length: int = 0
    _max_before_history: list[int] = field(default_factory=list, repr=False)
    _max_after_history: list[int] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict[str, object]:
        return {
            "query_count": self.query_count,
            "keep": self.keep,
            "drop": self.drop,
            "trim": self.trim,
            "kb_bypass": self.kb_bypass,
            "before_type_counts": dict(sorted(self.before_type_counts.items())),
            "after_type_counts": dict(sorted(self.after_type_counts.items())),
            "before_length_buckets": dict(sorted(self.before_length_buckets.items())),
            "after_length_buckets": dict(sorted(self.after_length_buckets.items())),
            "max_before_length": self.max_before_length,
            "max_after_length": self.max_after_length,
        }

    def copy(self) -> "EntityValidationCounters":
        return EntityValidationCounters(
            query_count=self.query_count,
            keep=self.keep,
            drop=self.drop,
            trim=self.trim,
            kb_bypass=self.kb_bypass,
            before_type_counts=Counter(self.before_type_counts),
            after_type_counts=Counter(self.after_type_counts),
            before_length_buckets=Counter(self.before_length_buckets),
            after_length_buckets=Counter(self.after_length_buckets),
            max_before_length=self.max_before_length,
            max_after_length=self.max_after_length,
            _max_before_history=list(self._max_before_history),
            _max_after_history=list(self._max_after_history),
        )

    def add(self, other: "EntityValidationCounters") -> "EntityValidationCounters":
        self.query_count += other.query_count
        self.keep += other.keep
        self.drop += other.drop
        self.trim += other.trim
        self.kb_bypass += other.kb_bypass
        self.before_type_counts.update(other.before_type_counts)
        self.after_type_counts.update(other.after_type_counts)
        self.before_length_buckets.update(other.before_length_buckets)
        self.after_length_buckets.update(other.after_length_buckets)
        self.max_before_length = max(self.max_before_length, other.max_before_length)
        self.max_after_length = max(self.max_after_length, other.max_after_length)
        self._max_before_history.extend(other._max_before_history or [other.max_before_length])
        self._max_after_history.extend(other._max_after_history or [other.max_after_length])
        return self

    def delta(self, previous: "EntityValidationCounters") -> "EntityValidationCounters":
        return EntityValidationCounters(
            query_count=self.query_count - previous.query_count,
            keep=self.keep - previous.keep,
            drop=self.drop - previous.drop,
            trim=self.trim - previous.trim,
            kb_bypass=self.kb_bypass - previous.kb_bypass,
            before_type_counts=_counter_delta(self.before_type_counts, previous.before_type_counts),
            after_type_counts=_counter_delta(self.after_type_counts, previous.after_type_counts),
            before_length_buckets=_counter_delta(
                self.before_length_buckets, previous.before_length_buckets
            ),
            after_length_buckets=_counter_delta(
                self.after_length_buckets, previous.after_length_buckets
            ),
            max_before_length=_delta_maximum(
                self.max_before_length,
                previous.max_before_length,
                self._max_before_history,
                previous._max_before_history,
            ),
            max_after_length=_delta_maximum(
                self.max_after_length,
                previous.max_after_length,
                self._max_after_history,
                previous._max_after_history,
            ),
        )


def _counter_delta(current: Counter[str], previous: Counter[str]) -> Counter[str]:
    return Counter({key: current[key] - previous[key] for key in set(current) | set(previous)})


def _delta_maximum(
    current: int,
    previous: int,
    current_history: list[int],
    previous_history: list[int],
) -> int:
    """Return the maximum contributed after a compatible cumulative snapshot."""
    if current_history[: len(previous_history)] == previous_history:
        contributions = current_history[len(previous_history) :]
        if contributions:
            return max(contributions)
        return 0
    return current if current > previous else 0


@dataclass(frozen=True, slots=True)
class EntityValidationResult:
    entities: tuple[EntityAnnotation, ...]
    decisions: tuple[EntityValidationDecision | None, ...]
    counters: EntityValidationCounters


def _length_bucket(length: int) -> str:
    if length <= 50:
        return "1-50"
    if length <= 100:
        return "51-100"
    return "101+"


def _validation_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "action": {"enum": ["keep", "drop", "trim"]},
            "relative_start": {"type": "integer"},
            "relative_end": {"type": "integer"},
            "entity_type": {"enum": list(ENTITY_TYPE_TO_ID)},
        },
        "required": ["action"],
        "additionalProperties": False,
    }


def _parse_decision(response_text: str, entity: EntityAnnotation) -> EntityValidationDecision:
    payload = parse_json_object(response_text)
    if payload is None:
        raise ValueError("malformed JSON entity validation response")
    allowed = {"action", "relative_start", "relative_end", "entity_type"}
    if set(payload) - allowed:
        raise ValueError("entity validation response has unsupported fields")
    action = payload.get("action")
    if action not in {"keep", "drop", "trim"}:
        raise ValueError("invalid entity validation action")
    if action != "trim":
        if set(payload) != {"action"}:
            raise ValueError("keep/drop decisions must only contain action")
        return EntityValidationDecision(action=action)

    if set(payload) != {"action", "relative_start", "relative_end", "entity_type"}:
        raise ValueError("trim decision must include relative boundaries and entity_type")
    relative_start = payload["relative_start"]
    relative_end = payload["relative_end"]
    entity_type = payload["entity_type"]
    if isinstance(relative_start, bool) or not isinstance(relative_start, int):
        raise ValueError("relative_start must be an integer")
    if isinstance(relative_end, bool) or not isinstance(relative_end, int):
        raise ValueError("relative_end must be an integer")
    if entity_type not in ENTITY_TYPE_TO_ID:
        raise ValueError("invalid entity type")
    if not (0 < relative_start < relative_end < len(entity.text)):
        raise ValueError("trim boundaries must be strictly inside the original entity")
    return EntityValidationDecision(
        action="trim",
        relative_start=relative_start,
        relative_end=relative_end,
        entity_type=entity_type,
    )


class QwenEntityValidator:
    def __init__(self, llm_engine: Any, *, batch_size: int = 64):
        self.llm = llm_engine
        self.batch_size = batch_size

    @staticmethod
    def _is_exact_kb(entity: EntityAnnotation) -> bool:
        if "proposal_kb_first" not in entity.evidence:
            return False
        return any(candidate.get("score") == 1.0 for candidate in entity.ranked_candidates)

    @staticmethod
    def _build_prompt(raw_text: str, entity: EntityAnnotation) -> str:
        return (
            "<|im_start|>system\n"
            "Ban la chuyen gia NER y khoa. Chi tra ve mot JSON hop le theo schema.\n"
            "<|im_end|>\n<|im_start|>user\n"
            "Hay giu, bo, hoac cat gon thuc the ve den mention nho nhat. Giai thich, "
            "nguyen nhan, yeu to nguy co va loi khuyen dieu tri khong thuoc mention nho nhat.\n"
            "Vi du hop le dai: 'ung thu phoi khong te bao nho giai doan IV' la mot mention "
            "ICD hop le neu toan bo cum la chan doan. Vi du khong hop le: 'viem phoi do nam lau' "
            "phai cat thanh 'viem phoi', bo phan giai thich nguyen nhan.\n\n"
            f'Van ban goc:\n"""{raw_text}"""\n\n'
            f"Thuc the: [{entity.text}]\n"
            f"Loai hien tai: {entity.type}\n"
            "Voi trim, relative_start va relative_end phai nam nghiem ngat ben trong text cua thuc the.\n"
            'Tra ve {"action":"keep"} hoac {"action":"drop"}; voi trim tra ve '
            '{"action":"trim","relative_start":0,"relative_end":4,"entity_type":"DISEASE"}.\n'
            "<|im_end|>\n<|im_start|>assistant\n"
        )

    @staticmethod
    def _record_before(counters: EntityValidationCounters, entity: EntityAnnotation) -> None:
        length = len(entity.text)
        counters.before_type_counts[entity.type] += 1
        counters.before_length_buckets[_length_bucket(length)] += 1
        counters.max_before_length = max(counters.max_before_length, length)

    @staticmethod
    def _record_after(counters: EntityValidationCounters, entity: EntityAnnotation) -> None:
        length = len(entity.text)
        counters.after_type_counts[entity.type] += 1
        counters.after_length_buckets[_length_bucket(length)] += 1
        counters.max_after_length = max(counters.max_after_length, length)

    @staticmethod
    def _apply_decision(
        entity: EntityAnnotation, decision: EntityValidationDecision, raw_text: str
    ) -> EntityAnnotation | None:
        entity.validate_offset(raw_text)
        if decision.action == "keep":
            return entity
        if decision.action == "drop":
            return None
        assert decision.relative_start is not None
        assert decision.relative_end is not None
        assert decision.entity_type is not None
        absolute_start = entity.start + decision.relative_start
        absolute_end = entity.start + decision.relative_end
        trimmed_text = entity.text[decision.relative_start : decision.relative_end]
        if not trimmed_text.strip():
            raise ValueError("trimmed entity cannot be whitespace only")
        if raw_text[absolute_start:absolute_end] != trimmed_text:
            raise ValueError("trimmed entity does not match raw-text offsets")
        return replace(
            entity,
            text=trimmed_text,
            type=decision.entity_type,
            position=(absolute_start, absolute_end),
            candidates=[],
            assertions=[],
            mention_head=None,
            ranked_candidates=[],
        )

    def validate(
        self, entities: tuple[EntityAnnotation, ...], raw_text: str
    ) -> EntityValidationResult:
        counters = EntityValidationCounters()
        decisions: list[EntityValidationDecision | None] = []
        to_query: list[EntityAnnotation] = []
        query_indexes: list[int] = []
        resolved: dict[int, EntityAnnotation | None] = {}
        for index, entity in enumerate(entities):
            entity.validate_offset(raw_text)
            self._record_before(counters, entity)
            if self._is_exact_kb(entity):
                counters.kb_bypass += 1
                decisions.append(None)
                resolved[index] = entity
            else:
                decisions.append(None)
                to_query.append(entity)
                query_indexes.append(index)

        counters.query_count = len(to_query)
        if not to_query:
            for entity in resolved.values():
                assert entity is not None
                self._record_after(counters, entity)
            return EntityValidationResult(
                tuple(entity for entity in resolved.values() if entity is not None),
                tuple(decisions),
                counters,
            )
        if self.llm is None:
            raise RuntimeError("LLM is not initialized")
        from vllm import SamplingParams

        for entity_batch, index_batch in zip(iter_batches(to_query, self.batch_size), iter_batches(query_indexes, self.batch_size)):
            prompts = [self._build_prompt(raw_text, entity) for entity in entity_batch]
            sampling_params = [
                SamplingParams(**build_sampling_kwargs(SamplingParams, _validation_schema()))
                for _ in entity_batch
            ]
            outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            if len(outputs) != len(entity_batch):
                raise ValueError(
                    f"vLLM returned {len(outputs)} outputs for {len(entity_batch)} entity validation prompts"
                )
            for entity, index, output in zip(entity_batch, index_batch, outputs):
                decision = _parse_decision(output.outputs[0].text, entity)
                decisions[index] = decision
                result = self._apply_decision(entity, decision, raw_text)
                if decision.action == "keep":
                    counters.keep += 1
                elif decision.action == "drop":
                    counters.drop += 1
                else:
                    counters.trim += 1
                resolved[index] = result

        for index, entity in enumerate(entities):
            result = resolved[index]
            if result is not None:
                self._record_after(counters, result)
        return EntityValidationResult(
            tuple(resolved[index] for index in range(len(entities)) if resolved[index] is not None),
            tuple(decisions),
            counters,
        )
