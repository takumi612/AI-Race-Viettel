from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .records import ClinicalRecord, parse_document_records
from .schema import ClinicalDocument, EntityAnnotation


@dataclass(frozen=True)
class SpanProposal:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    source: str
    ranked_candidates: tuple[dict[str, Any], ...] = ()
    candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class FinalModelBundle:
    ner_model: Any
    tokenizer: Any
    assertion_model: Any | None = None
    candidate_policy: Any | None = None
    kb_linker: Any | None = None
    qwen_reranker: Any | None = None


@dataclass(frozen=True)
class InferenceConfig:
    batch_size: int = 8
    max_length: int = 512
    stride: int = 128
    enable_qwen: bool = False
    enable_kb_recovery: bool = True


def merge_raw_span_proposals(
    proposals: Sequence[SpanProposal],
    records: Sequence[ClinicalRecord],
    raw_text: str = "",
) -> tuple[EntityAnnotation, ...]:
    if not proposals:
        return ()

    merged_entities: list[EntityAnnotation] = []
    for rec in records:
        rec_proposals = [
            p for p in proposals
            if rec.raw_start <= p.start and p.end <= rec.raw_end
        ]
        rec_proposals.sort(key=lambda p: (p.start, -(p.end - p.start), -p.confidence))
        selected: list[SpanProposal] = []
        for prop in rec_proposals:
            if prop.start < 0 or prop.end <= prop.start:
                continue
            if raw_text and raw_text[prop.start:prop.end] != prop.text:
                continue
            if any(max(prop.start, item.start) < min(prop.end, item.end) for item in selected):
                continue
            selected.append(prop)
        for proposal in selected:
            merged_entities.append(
                EntityAnnotation(
                    text=proposal.text,
                    type=proposal.entity_type,
                    position=(proposal.start, proposal.end),
                    candidates=list(proposal.candidate_ids),
                    confidence=proposal.confidence,
                    evidence=[f"proposal_{proposal.source}"],
                )
            )
    return tuple(merged_entities)


def _coerce_proposal(value: Any, source: str) -> SpanProposal:
    if isinstance(value, SpanProposal):
        return value
    if isinstance(value, EntityAnnotation):
        return SpanProposal(
            value.text,
            value.type,
            value.start,
            value.end,
            value.confidence,
            source,
            candidate_ids=tuple(value.candidates),
        )
    if isinstance(value, Mapping):
        position = value.get("position", (value.get("start"), value.get("end")))
        if not isinstance(position, (list, tuple)) or len(position) != 2:
            raise ValueError("proposal position must contain start and end")
        ranked = value.get("ranked_candidates", value.get("candidates", ()))
        ranked_candidates = tuple(dict(item) for item in ranked if isinstance(item, Mapping))
        return SpanProposal(
            text=str(value["text"]),
            entity_type=str(value.get("entity_type", value.get("type", ""))),
            start=int(position[0]),
            end=int(position[1]),
            confidence=float(value.get("confidence", 1.0)),
            source=str(value.get("source", source)),
            ranked_candidates=ranked_candidates,
            candidate_ids=tuple(str(item) for item in value.get("candidate_ids", ())),
        )
    raise TypeError(f"Unsupported proposal type: {type(value).__name__}")


def _call_ner(bundle: FinalModelBundle, raw_text: str, config: InferenceConfig) -> list[SpanProposal]:
    model = bundle.ner_model
    if model is None:
        return []
    method = next(
        (
            getattr(model, name, None)
            for name in ("propose", "predict_proposals", "detect")
            if callable(getattr(model, name, None))
        ),
        None,
    )
    if method is None and callable(model):
        method = model
    if method is None:
        return []
    try:
        values = method(raw_text, config)
    except TypeError:
        values = method(raw_text)
    return [_coerce_proposal(value, "ner") for value in (values or ())]


def _apply_candidate_policy(proposal: SpanProposal, policy: Any | None) -> SpanProposal:
    proposal = replace(proposal, ranked_candidates=proposal.ranked_candidates[:20])
    if policy is None or not proposal.ranked_candidates:
        return proposal
    try:
        if callable(getattr(policy, "apply", None)):
            selected = policy.apply(list(proposal.ranked_candidates))
        else:
            from .candidate_policy import apply_candidate_policy

            selected = apply_candidate_policy(proposal.ranked_candidates, policy)
        return replace(proposal, candidate_ids=tuple(str(item) for item in selected[:1]))
    except (TypeError, ValueError, KeyError):
        return replace(proposal, candidate_ids=())


def _apply_assertions(
    entities: tuple[EntityAnnotation, ...],
    raw_text: str,
    assertion_model: Any | None,
) -> tuple[EntityAnnotation, ...]:
    if assertion_model is None or not entities:
        return entities
    predictor = getattr(assertion_model, "predict", assertion_model)
    if not callable(predictor):
        return entities
    try:
        result = predictor(raw_text, entities)
    except TypeError:
        result = predictor(entities, raw_text)
    allowed = {"isNegated", "isHistorical", "isFamily"}
    for index, entity in enumerate(entities):
        if entity.type in {"LAB_NAME", "LAB_RESULT"}:
            entity.assertions = []
            continue
        if isinstance(result, Mapping):
            labels = result.get((entity.start, entity.end, entity.type), ())
        else:
            labels = result[index] if index < len(result) else ()
        entity.assertions = [str(label) for label in labels if str(label) in allowed]
    return entities


def infer_document(
    document_id: str,
    raw_text: str,
    bundle: FinalModelBundle,
    config: InferenceConfig,
) -> ClinicalDocument:
    records = parse_document_records(document_id, raw_text, ())
    proposals = _call_ner(bundle, raw_text, config)

    if config.enable_kb_recovery and bundle.kb_linker is not None:
        try:
            kb_proposals = bundle.kb_linker.scan_raw_text(raw_text)
            proposals.extend(_coerce_proposal(value, "kb_first") for value in (kb_proposals or ()))
        except (AttributeError, TypeError, ValueError):
            pass

    proposals = [_apply_candidate_policy(proposal, bundle.candidate_policy) for proposal in proposals]
    merged_entities = merge_raw_span_proposals(proposals, records, raw_text=raw_text)
    merged_entities = _apply_assertions(merged_entities, raw_text, bundle.assertion_model)

    if config.enable_qwen and bundle.qwen_reranker is not None:
        try:
            refined = bundle.qwen_reranker.refine(merged_entities, raw_text)
            if refined is not None:
                merged_entities = tuple(refined)
        except Exception:
            pass

    for entity in merged_entities:
        entity.validate_offset(raw_text)
    return ClinicalDocument(document_id=document_id, raw_text=raw_text, entities=list(merged_entities))
