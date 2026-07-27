"""Fixed natural validation and document-level NER evaluation helpers.

The 20 organizer-GT documents 181--200 are deliberately kept out of every
NER training selection.  This module is dependency-light so its contracts and
metrics can be exercised without a Transformer checkpoint.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import EntityAnnotation


REQUIRED_DATASET_FINGERPRINT = (
    "18a391e51786630b482bb500d5129eb102ae144450d7fc18b149f2799054f028"
)
NATURAL_VALIDATION_IDS = tuple(str(value) for value in range(181, 201))
OFFICIAL_TYPES = {
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
}
_EXPECTED_TYPE_COUNTS = {
    "CHẨN_ĐOÁN": 496,
    "THUỐC": 248,
    "TRIỆU_CHỨNG": 496,
    "TÊN_XÉT_NGHIỆM": 124,
    "KẾT_QUẢ_XÉT_NGHIỆM": 124,
}
_THRESHOLDS = (0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)
_LENGTH_BUCKETS = ("1-50", "51-100", "101+")


def _natural_key(value: str) -> tuple[int, str]:
    try:
        return (int(value), value)
    except ValueError:
        return (10**12, value)


def apply_natural_validation_partition(
    train_ids: Iterable[str], validation_ids: Iterable[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Remove natural IDs from train and prepend them to validation.

    Existing validation IDs remain available (notably synthetic validation),
    but duplicate IDs are removed deterministically.
    """
    existing_validation = tuple(str(value) for value in validation_ids)
    validation = tuple(
        dict.fromkeys((*NATURAL_VALIDATION_IDS, *existing_validation))
    )
    validation_set = set(validation)
    train = tuple(
        value
        for value in dict.fromkeys(str(item) for item in train_ids)
        if value not in validation_set
    )
    return train, validation


def _summary_length_bucket(length: int) -> str:
    if length <= 50:
        return "1-50"
    if length <= 100:
        return "51-100"
    return "101+"


def validate_natural_validation_summary(
    summary: Mapping[str, Any], dataset_fingerprint: str
) -> dict[str, object]:
    """Validate the immutable 181--200 inventory and return its manifest."""
    if dataset_fingerprint != REQUIRED_DATASET_FINGERPRINT:
        raise ValueError(
            "natural validation dataset fingerprint does not match the required dataset"
        )
    expected_ids = list(NATURAL_VALIDATION_IDS)
    if list(summary.get("document_ids", [])) != expected_ids:
        raise ValueError("natural validation document IDs must be exactly 181-200")
    if summary.get("entity_count") != 1488:
        raise ValueError("natural validation entity inventory must contain 1488 entities")
    if dict(summary.get("type_counts", {})) != _EXPECTED_TYPE_COUNTS:
        raise ValueError("natural validation type inventory does not match official types")
    if set(summary["type_counts"]) != OFFICIAL_TYPES:
        raise ValueError("natural validation types are not the five official types")
    if summary.get("over_50") != 120 or summary.get("over_100") != 5:
        raise ValueError("natural validation long-span inventory is invalid")
    if summary.get("max_length") != 110:
        raise ValueError("natural validation maximum span length is invalid")
    return {
        "schema_id": "clinical_nlp.natural_validation",
        "schema_version": 1,
        "dataset_fingerprint": dataset_fingerprint,
        "document_ids": expected_ids,
        "summary": dict(summary),
        "length_buckets": {
            "over_50": 120,
            "over_100": 5,
            "max_length": 110,
        },
    }


def build_natural_validation_manifest(
    dataset_root: str | Path, dataset_fingerprint: str
) -> dict[str, object]:
    """Scan the canonical input/gt layout and bind it to the known inventory."""
    # Fail before touching a path for a caller who has supplied a wrong data
    # binding; this makes accidental competition-input use impossible to hide.
    if dataset_fingerprint != REQUIRED_DATASET_FINGERPRINT:
        raise ValueError("natural validation dataset fingerprint is not approved")

    from .provenance import verify_dataset_provenance

    root = Path(dataset_root)
    verification = verify_dataset_provenance(root)
    if verification.dataset_fingerprint != dataset_fingerprint:
        raise ValueError("dataset fingerprint does not match verified dataset")
    input_dir = root / "input"
    gt_dir = root / "gt"
    documents: list[str] = []
    type_counts: Counter[str] = Counter()
    lengths: list[int] = []
    for document_id in NATURAL_VALIDATION_IDS:
        text_path = input_dir / f"{document_id}.txt"
        gt_path = gt_dir / f"{document_id}.json"
        if not text_path.is_file() or not gt_path.is_file():
            raise ValueError(f"natural validation requires input/GT pair for {document_id}")
        raw_text = text_path.read_text(encoding="utf-8")
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        entities = payload.get("entities") if isinstance(payload, dict) else payload
        if not isinstance(entities, list):
            raise ValueError(f"natural validation GT must be an entity list: {gt_path}")
        documents.append(document_id)
        for entity in entities:
            if not isinstance(entity, Mapping):
                raise ValueError(f"natural validation entity is invalid: {gt_path}")
            entity_type = entity.get("type")
            position = entity.get("position")
            if entity_type not in OFFICIAL_TYPES or not isinstance(position, list | tuple) or len(position) != 2:
                raise ValueError(f"natural validation entity schema is invalid: {gt_path}")
            start, end = int(position[0]), int(position[1])
            if not 0 <= start <= end <= len(raw_text):
                raise ValueError(f"natural validation entity offset is invalid: {gt_path}")
            type_counts[str(entity_type)] += 1
            lengths.append(end - start)
    summary = {
        "document_ids": documents,
        "entity_count": len(lengths),
        "type_counts": dict(sorted(type_counts.items())),
        "over_50": sum(length > 50 for length in lengths),
        "over_100": sum(length > 100 for length in lengths),
        "max_length": max(lengths, default=0),
    }
    return validate_natural_validation_summary(summary, dataset_fingerprint)


def _span_key(document_id: str, entity: EntityAnnotation) -> tuple[str, int, int, str]:
    return (str(document_id), entity.start, entity.end, entity.type)


def _score(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": tp,
        "predicted": predicted,
        "gold": gold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _flatten(
    documents: Mapping[str, Sequence[EntityAnnotation]],
) -> list[tuple[str, EntityAnnotation]]:
    return [
        (str(document_id), entity)
        for document_id in sorted(documents, key=_natural_key)
        for entity in sorted(
            documents[document_id],
            key=lambda item: (item.start, item.end, item.type, -item.confidence),
        )
    ]


def _overlap_true_positives(
    expected: list[tuple[str, EntityAnnotation]],
    predicted: list[tuple[str, EntityAnnotation]],
) -> int:
    used_gold: set[int] = set()
    matched = 0
    for document_id, entity in predicted:
        for index, (gold_document_id, gold_entity) in enumerate(expected):
            if index in used_gold:
                continue
            if document_id != gold_document_id or entity.type != gold_entity.type:
                continue
            if max(entity.start, gold_entity.start) < min(entity.end, gold_entity.end):
                used_gold.add(index)
                matched += 1
                break
    return matched


def _metric_report(
    expected_documents: Mapping[str, Sequence[EntityAnnotation]],
    predicted_documents: Mapping[str, Sequence[EntityAnnotation]],
    provenance: Mapping[str, str],
) -> dict[str, object]:
    expected = _flatten(expected_documents)
    predicted = _flatten(predicted_documents)
    gold_keys = {_span_key(document_id, entity) for document_id, entity in expected}
    predicted_keys = {_span_key(document_id, entity) for document_id, entity in predicted}
    exact = _score(len(gold_keys & predicted_keys), len(predicted_keys), len(gold_keys))
    overlap = _score(
        _overlap_true_positives(expected, predicted), len(predicted), len(expected)
    )

    def subset(
        predicate: Any,
    ) -> dict[str, object]:
        subset_expected: dict[str, list[EntityAnnotation]] = defaultdict(list)
        subset_predicted: dict[str, list[EntityAnnotation]] = defaultdict(list)
        for document_id, entity in expected:
            if predicate(document_id, entity):
                subset_expected[document_id].append(entity)
        for document_id, entity in predicted:
            if predicate(document_id, entity):
                subset_predicted[document_id].append(entity)
        return _metric_report_base(subset_expected, subset_predicted)

    types = sorted({entity.type for _, entity in [*expected, *predicted]})
    by_type = {entity_type: subset(lambda _doc, item, typ=entity_type: item.type == typ) for entity_type in types}
    sources = sorted({str(provenance.get(document_id, "unknown")) for document_id, _ in [*expected, *predicted]})
    by_provenance = {
        source: subset(lambda document_id, _item, value=source: str(provenance.get(document_id, "unknown")) == value)
        for source in sources
    }
    length_buckets = {
        bucket: subset(lambda _doc, item, value=bucket: _summary_length_bucket(item.end - item.start) == value)
        for bucket in _LENGTH_BUCKETS
    }
    return {
        "exact": exact,
        "overlap": overlap,
        "by_type": by_type,
        "by_provenance": by_provenance,
        "length_buckets": length_buckets,
    }


def _metric_report_base(
    expected_documents: Mapping[str, Sequence[EntityAnnotation]],
    predicted_documents: Mapping[str, Sequence[EntityAnnotation]],
) -> dict[str, object]:
    expected = _flatten(expected_documents)
    predicted = _flatten(predicted_documents)
    gold_keys = {_span_key(document_id, entity) for document_id, entity in expected}
    predicted_keys = {_span_key(document_id, entity) for document_id, entity in predicted}
    return {
        "exact": _score(len(gold_keys & predicted_keys), len(predicted_keys), len(gold_keys)),
        "overlap": _score(_overlap_true_positives(expected, predicted), len(predicted), len(expected)),
        "gold": len(expected),
        "predicted": len(predicted),
    }


def calibrate_document_entity_threshold(
    expected: Mapping[str, Sequence[EntityAnnotation]],
    predicted: Mapping[str, Sequence[EntityAnnotation]],
    provenance: Mapping[str, str],
) -> dict[str, object]:
    """Select a confidence threshold on exact typed document spans."""
    candidates: list[tuple[float, dict[str, object]]] = []
    for threshold in _THRESHOLDS:
        filtered = {
            str(document_id): [
                entity
                for entity in entities
                if float(entity.confidence) >= threshold
            ]
            for document_id, entities in predicted.items()
        }
        candidates.append((threshold, _metric_report(expected, filtered, provenance)))
    best_threshold, best = max(
        candidates,
        key=lambda item: (
            float(item[1]["exact"]["f1"]),
            float(item[1]["exact"]["precision"]),
            item[0],
        ),
    )
    return {
        "schema_id": "clinical_nlp.document_ner_calibration",
        "schema_version": 1,
        "objective": "document_exact_f1_precision_tiebreak",
        "confidence_threshold": best_threshold,
        **best,
        "thresholds": [
            {
                "confidence_threshold": threshold,
                "exact": report["exact"],
                "overlap": report["overlap"],
            }
            for threshold, report in candidates
        ],
    }


def _legacy_union_for_audit(
    entities: Iterable[EntityAnnotation], raw_text: str
) -> list[EntityAnnotation]:
    """Reproduce the pre-consensus overlap union strictly for training audits."""
    selected: list[EntityAnnotation] = []
    for entity in sorted(entities, key=lambda item: (item.type, item.start, item.end)):
        entity.validate_offset(raw_text)
        overlapping = [
            existing
            for existing in selected
            if existing.type == entity.type
            and existing.start < entity.end
            and entity.start < existing.end
        ]
        if not overlapping:
            selected.append(entity)
            continue
        start = min([entity.start, *(item.start for item in overlapping)])
        end = max([entity.end, *(item.end for item in overlapping)])
        owner = max([entity, *overlapping], key=lambda item: float(item.confidence))
        selected = [item for item in selected if item not in overlapping]
        selected.append(
            replace(
                owner,
                text=raw_text[start:end],
                position=(start, end),
                confidence=max(item.confidence for item in [entity, *overlapping]),
                evidence=sorted({value for item in [entity, *overlapping] for value in item.evidence}),
            )
        )
    return sorted(selected, key=lambda item: (item.start, item.end, item.type))


def compare_document_merge_strategies(
    expected: Mapping[str, Sequence[EntityAnnotation]],
    raw_chunk_predictions: Mapping[str, Sequence[EntityAnnotation]],
    provenance: Mapping[str, str],
    raw_texts: Mapping[str, str],
) -> dict[str, object]:
    """Compare legacy union and boundary consensus on identical raw chunks."""
    from .ner import merge_chunk_predictions

    legacy: dict[str, list[EntityAnnotation]] = {}
    consensus: dict[str, list[EntityAnnotation]] = {}
    for document_id in sorted(raw_chunk_predictions, key=_natural_key):
        raw_text = raw_texts.get(document_id)
        if raw_text is None:
            raise ValueError(f"missing raw text for document {document_id}")
        chunks = raw_chunk_predictions[document_id]
        legacy[document_id] = _legacy_union_for_audit(chunks, raw_text)
        consensus[document_id] = merge_chunk_predictions(chunks, raw_text)
    legacy_metrics = _metric_report(expected, legacy, provenance)
    consensus_metrics = _metric_report(expected, consensus, provenance)
    if (
        float(consensus_metrics["exact"]["precision"])
        <= float(legacy_metrics["exact"]["precision"])
        or float(consensus_metrics["exact"]["f1"])
        < float(legacy_metrics["exact"]["f1"])
    ):
        raise ValueError(
            "boundary consensus must improve exact precision without reducing exact F1"
        )
    return {
        "schema_id": "clinical_nlp.document_merge_comparison",
        "schema_version": 1,
        "legacy_union": legacy_metrics,
        "boundary_consensus": consensus_metrics,
    }
