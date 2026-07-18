from collections import Counter, defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence


ASSERTION_LABELS = ("isNegated", "isHistorical", "isHypothetical")


@dataclass(frozen=True)
class EntityMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    fbeta: float


@dataclass(frozen=True)
class AssertionMetrics:
    by_label: Mapping[str, EntityMetrics]
    macro_fbeta: float


@dataclass(frozen=True)
class CandidateMetrics:
    jaccard: float
    precision: float
    top1_hit_rate: float
    recall_at_20: float | None


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def _entity_key(entity: dict) -> tuple[int, int, str]:
    start, end = entity["position"]
    return int(start), int(end), str(entity["type"]).strip()


def _metrics_from_counts(tp: int, fp: int, fn: int, beta: float) -> EntityMetrics:
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return EntityMetrics(tp, fp, fn, precision, recall, fbeta(precision, recall, beta))


def score_exact_entities(gt: list[dict], pred: list[dict], beta: float = 0.5) -> EntityMetrics:
    gt_keys = Counter(_entity_key(entity) for entity in gt)
    pred_keys = Counter(_entity_key(entity) for entity in pred)
    tp = sum((gt_keys & pred_keys).values())
    fp = sum((pred_keys - gt_keys).values())
    fn = sum((gt_keys - pred_keys).values())
    return _metrics_from_counts(tp, fp, fn, beta)


def score_entities_by_type(gt, pred, beta: float = 0.5) -> dict[str, EntityMetrics]:
    entity_types = {str(entity["type"]).strip() for entity in [*gt, *pred]}
    return {
        entity_type: score_exact_entities(
            [entity for entity in gt if str(entity["type"]).strip() == entity_type],
            [entity for entity in pred if str(entity["type"]).strip() == entity_type],
            beta,
        )
        for entity_type in entity_types
    }


def _matched_entities(gt: Sequence[dict], pred: Sequence[dict]) -> list[tuple[dict, dict]]:
    gt_by_key: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    pred_by_key: dict[tuple[int, int, str], list[dict]] = defaultdict(list)
    for entity in gt:
        gt_by_key[_entity_key(entity)].append(entity)
    for entity in pred:
        pred_by_key[_entity_key(entity)].append(entity)
    return [
        (gt_entity, pred_entity)
        for key in gt_by_key.keys() & pred_by_key.keys()
        for gt_entity, pred_entity in zip(gt_by_key[key], pred_by_key[key])
    ]


def score_assertions(gt, pred, beta: float = 0.5) -> AssertionMetrics:
    matched = _matched_entities(gt, pred)
    by_label: dict[str, EntityMetrics] = {}
    for label in ASSERTION_LABELS:
        tp = fp = fn = 0
        for gt_entity, pred_entity in matched:
            expected = label in gt_entity.get("assertions", [])
            actual = label in pred_entity.get("assertions", [])
            tp += expected and actual
            fp += actual and not expected
            fn += expected and not actual
        by_label[label] = _metrics_from_counts(tp, fp, fn, beta)
    macro_fbeta = sum(metric.fbeta for metric in by_label.values()) / len(ASSERTION_LABELS)
    return AssertionMetrics(by_label=MappingProxyType(by_label), macro_fbeta=macro_fbeta)


def _codes(entity: dict) -> set[str]:
    return {str(code).strip().upper() for code in entity.get("candidates", []) if str(code).strip()}


def score_candidate_sets(gt, pred, retrieved: Mapping[tuple[int, int, str], Sequence[str]] | None = None) -> CandidateMetrics:
    matched = _matched_entities(gt, pred)
    code_pairs = [(gt_entity, pred_entity) for gt_entity, pred_entity in matched if _codes(gt_entity) or _codes(pred_entity)]
    jaccards = []
    tp = fp = 0
    top1_hits = top1_total = 0
    for gt_entity, pred_entity in code_pairs:
        expected = _codes(gt_entity)
        actual = _codes(pred_entity)
        union = expected | actual
        jaccards.append(len(expected & actual) / len(union) if union else 1.0)
        tp += len(expected & actual)
        fp += len(actual - expected)
        if expected:
            top1_total += 1
            top1_hits += int(bool(pred_entity.get("candidates")) and str(pred_entity["candidates"][0]).strip().upper() in expected)

    recall_at_20 = None
    if retrieved is not None:
        retrieved_hits = retrieved_total = 0
        for entity in gt:
            expected = _codes(entity)
            if not expected:
                continue
            retrieved_codes = {str(code).strip().upper() for code in retrieved.get(_entity_key(entity), [])[:20] if str(code).strip()}
            retrieved_hits += len(expected & retrieved_codes)
            retrieved_total += len(expected)
        recall_at_20 = retrieved_hits / retrieved_total if retrieved_total else 1.0

    return CandidateMetrics(
        jaccard=sum(jaccards) / len(jaccards) if jaccards else 1.0,
        precision=tp / (tp + fp) if tp + fp else 1.0,
        top1_hit_rate=top1_hits / top1_total if top1_total else 1.0,
        recall_at_20=recall_at_20,
    )
