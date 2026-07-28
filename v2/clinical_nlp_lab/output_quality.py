"""Quality diagnostics that prevent publishing structurally valid model collapse."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from .data import load_input_documents
from .entity_span_policy import (
    SUSPICIOUS_GENERIC_SURFACES,
    is_suspicious_generic_surface,
    validate_entity_span,
)
from .schema import ClinicalDocument, EntityAnnotation


class OutputQualityError(ValueError):
    """Raised when inference output violates semantic safety gates."""


# Kept as a compatibility alias for existing diagnostics consumers.
_SUSPICIOUS_GENERIC_SURFACES = SUSPICIOUS_GENERIC_SURFACES

_CANDIDATE_ELIGIBLE_TYPES = {"DISEASE", "DRUG", "CHẨN_ĐOÁN", "THUỐC"}


def audit_output_documents(
    documents: Iterable[ClinicalDocument],
) -> dict[str, Any]:
    document_list = list(documents)
    type_counts: Counter[str] = Counter()
    coverages: list[float] = []
    boundary_errors = 0
    punctuation_only = 0
    whitespace_only = 0
    multiline = 0
    max_span_length = 0
    single_token_entities = 0
    suspicious_generic_spans = 0
    candidate_eligible = 0
    candidate_linked = 0

    for document in document_list:
        spans: list[tuple[int, int]] = []
        for entity in document.entities:
            entity.validate_offset(document.raw_text)
            type_counts[entity.type] += 1
            spans.append((entity.start, entity.end))
            max_span_length = max(max_span_length, entity.end - entity.start)
            if len(entity.text.split()) == 1:
                single_token_entities += 1
            if is_suspicious_generic_surface(entity.text):
                suspicious_generic_spans += 1
            if entity.type in _CANDIDATE_ELIGIBLE_TYPES:
                candidate_eligible += 1
                candidate_linked += bool(entity.candidates)
            violations = validate_entity_span(
                document.raw_text,
                entity.start,
                entity.end,
                entity.text,
                max_length=160,
            )
            if "punctuation_only" in violations:
                punctuation_only += 1
            if "whitespace_only" in violations:
                whitespace_only += 1
            if "multiline" in violations:
                multiline += 1
            boundary_errors += sum(
                reason in {"left_word_split", "right_word_split"}
                for reason in violations
            )

        covered = 0
        current_start: int | None = None
        current_end = 0
        for start, end in sorted(spans):
            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                covered += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            covered += current_end - current_start
        coverages.append(covered / max(1, len(document.raw_text)))

    entity_count = sum(type_counts.values())
    max_type_share = max(type_counts.values()) / entity_count if entity_count else 0.0
    return {
        "document_count": len(document_list),
        "entity_count": entity_count,
        "type_counts": dict(type_counts.most_common()),
        "max_type_share": round(max_type_share, 6),
        "mean_coverage": round(
            sum(coverages) / len(coverages) if coverages else 0.0, 6
        ),
        "boundary_error_count": boundary_errors,
        "punctuation_only_count": punctuation_only,
        "whitespace_only_count": whitespace_only,
        "multiline_count": multiline,
        "max_span_length": max_span_length,
        "single_token_entity_count": single_token_entities,
        "suspicious_generic_span_count": suspicious_generic_spans,
        "candidate_eligible_count": candidate_eligible,
        "candidate_linked_entity_count": candidate_linked,
        "candidate_link_rate": round(
            candidate_linked / candidate_eligible if candidate_eligible else 0.0,
            6,
        ),
    }


def enforce_output_quality(report: dict[str, Any]) -> None:
    violations: list[str] = []
    if float(report["mean_coverage"]) > 0.55:
        violations.append(f"coverage={report['mean_coverage']}")
    if int(report["entity_count"]) >= 50 and float(report["max_type_share"]) > 0.80:
        violations.append(f"type_share={report['max_type_share']}")
    if int(report["boundary_error_count"]) > 0:
        violations.append(f"boundary_errors={report['boundary_error_count']}")
    if int(report["punctuation_only_count"]) > 0:
        violations.append(f"punctuation_only={report['punctuation_only_count']}")
    if int(report.get("whitespace_only_count", 0)) > 0:
        violations.append(f"whitespace_only={report['whitespace_only_count']}")
    if int(report["multiline_count"]) > 0:
        violations.append(f"multiline={report['multiline_count']}")
    if int(report["max_span_length"]) > 160:
        violations.append(f"max_span_length={report['max_span_length']}")
    generic_count = int(report.get("suspicious_generic_span_count", 0))
    entity_count = int(report["entity_count"])
    if generic_count >= 3 and generic_count / max(1, entity_count) >= 0.01:
        violations.append(f"generic_spans={generic_count}")
    if violations:
        raise OutputQualityError(
            "Output quality gate rejected model collapse: " + ", ".join(violations)
        )


def audit_submission_directory(
    input_source: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    source_documents = {
        document.document_id: document for document in load_input_documents(input_source)
    }
    documents: list[ClinicalDocument] = []
    for document_id, source in source_documents.items():
        payload = json.loads(
            (Path(output_dir) / f"{document_id}.json").read_text(encoding="utf-8")
        )
        entities = [
            EntityAnnotation(
                text=str(item["text"]),
                type=str(item["type"]),
                position=(int(item["position"][0]), int(item["position"][1])),
                candidates=[str(value) for value in item.get("candidates", [])],
                assertions=[str(value) for value in item.get("assertions", [])],
            )
            for item in payload
        ]
        documents.append(ClinicalDocument(document_id, source.raw_text, entities))
    return audit_output_documents(documents)
