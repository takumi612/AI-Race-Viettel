from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
from .schema import ClinicalDocument, EntityAnnotation
from .records import ClinicalRecord, parse_document_records


@dataclass(frozen=True)
class SpanProposal:
    text: str
    entity_type: str
    start: int
    end: int
    confidence: float
    source: str  # "ner" hoặc "kb_first"


@dataclass(frozen=True)
class FinalModelBundle:
    ner_model: Any
    tokenizer: Any
    assertion_model: Any | None = None
    candidate_policy: Any | None = None
    kb_linker: Any | None = None


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

    # Group proposals by record boundary
    merged_entities: list[EntityAnnotation] = []

    for rec in records:
        rec_proposals = [
            p for p in proposals
            if rec.raw_start <= p.start and p.end <= rec.raw_end
        ]

        if not rec_proposals:
            continue

        # Sort proposals by start ascending, length descending, confidence descending
        rec_proposals.sort(key=lambda p: (p.start, -(p.end - p.start), -p.confidence))

        selected: list[SpanProposal] = []
        for prop in rec_proposals:
            # Round-trip validation if raw_text is available
            if raw_text and raw_text[prop.start:prop.end] != prop.text:
                continue

            # Check overlap with already selected spans in this record
            overlap = False
            for sel in selected:
                if max(prop.start, sel.start) < min(prop.end, sel.end):
                    overlap = True
                    break

            if not overlap:
                selected.append(prop)

        for p in selected:
            entity = EntityAnnotation(
                text=p.text,
                type=p.entity_type,
                position=(p.start, p.end),
                confidence=p.confidence,
                evidence=[f"proposal_{p.source}"],
            )
            merged_entities.append(entity)

    return tuple(merged_entities)


def infer_document(
    document_id: str,
    raw_text: str,
    bundle: FinalModelBundle,
    config: InferenceConfig,
) -> ClinicalDocument:
    entities: list[EntityAnnotation] = []
    records = parse_document_records(document_id, raw_text, entities)

    proposals: list[SpanProposal] = []

    # KB-first recovery scanning
    if config.enable_kb_recovery and bundle.kb_linker is not None:
        try:
            kb_proposals = bundle.kb_linker.scan_raw_text(raw_text)
            proposals.extend(kb_proposals)
        except AttributeError:
            pass

    merged_entities = merge_raw_span_proposals(proposals, records, raw_text=raw_text)

    return ClinicalDocument(
        document_id=document_id,
        raw_text=raw_text,
        entities=list(merged_entities),
    )
