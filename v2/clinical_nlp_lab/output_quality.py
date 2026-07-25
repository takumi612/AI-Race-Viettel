"""Quality diagnostics that prevent publishing structurally valid model collapse."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable

from .data import load_input_documents
from .schema import ClinicalDocument, EntityAnnotation


class OutputQualityError(ValueError):
    """Raised when inference output violates semantic safety gates."""


def audit_output_documents(
    documents: Iterable[ClinicalDocument],
) -> dict[str, Any]:
    document_list = list(documents)
    type_counts: Counter[str] = Counter()
    coverages: list[float] = []
    boundary_errors = 0
    punctuation_only = 0
    multiline = 0
    max_span_length = 0

    for document in document_list:
        spans: list[tuple[int, int]] = []
        for entity in document.entities:
            entity.validate_offset(document.raw_text)
            type_counts[entity.type] += 1
            spans.append((entity.start, entity.end))
            max_span_length = max(max_span_length, entity.end - entity.start)
            if not any(character.isalnum() for character in entity.text):
                punctuation_only += 1
            if "\n" in entity.text or "\r" in entity.text:
                multiline += 1
            if (
                entity.start > 0
                and document.raw_text[entity.start - 1].isalnum()
                and document.raw_text[entity.start].isalnum()
            ):
                boundary_errors += 1
            if (
                entity.end < len(document.raw_text)
                and document.raw_text[entity.end - 1].isalnum()
                and document.raw_text[entity.end].isalnum()
            ):
                boundary_errors += 1

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
        "multiline_count": multiline,
        "max_span_length": max_span_length,
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
    if int(report["multiline_count"]) > 0:
        violations.append(f"multiline={report['multiline_count']}")
    if int(report["max_span_length"]) > 160:
        violations.append(f"max_span_length={report['max_span_length']}")
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
            )
            for item in payload
        ]
        documents.append(ClinicalDocument(document_id, source.raw_text, entities))
    return audit_output_documents(documents)
