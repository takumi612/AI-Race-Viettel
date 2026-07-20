from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .schema import EntityAnnotation


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def f1_from_counts(true_positive: int, false_positive: int, false_negative: int) -> dict[str, float | int]:
    precision = safe_divide(true_positive, true_positive + false_positive)
    recall = safe_divide(true_positive, true_positive + false_negative)
    f1 = safe_divide(2 * precision * recall, precision + recall)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def span_overlap(left: EntityAnnotation, right: EntityAnnotation) -> int:
    return max(0, min(left.end, right.end) - max(left.start, right.start))


def exact_span_metrics(gold: Iterable[EntityAnnotation], predicted: Iterable[EntityAnnotation]) -> dict[str, Any]:
    gold_keys = Counter((item.start, item.end, item.type) for item in gold)
    predicted_keys = Counter((item.start, item.end, item.type) for item in predicted)
    true_positive = sum((gold_keys & predicted_keys).values())
    return f1_from_counts(
        true_positive,
        sum(predicted_keys.values()) - true_positive,
        sum(gold_keys.values()) - true_positive,
    )


def relaxed_span_metrics(gold: Iterable[EntityAnnotation], predicted: Iterable[EntityAnnotation]) -> dict[str, Any]:
    gold_list = list(gold)
    predicted_list = list(predicted)
    used_gold: set[int] = set()
    true_positive = 0
    for prediction in sorted(predicted_list, key=lambda item: -item.confidence):
        best_index = None
        best_overlap = 0
        for index, target in enumerate(gold_list):
            if index in used_gold or target.type != prediction.type:
                continue
            overlap = span_overlap(target, prediction)
            if overlap > best_overlap:
                best_overlap = overlap
                best_index = index
        if best_index is not None and best_overlap > 0:
            used_gold.add(best_index)
            true_positive += 1
    return f1_from_counts(
        true_positive,
        len(predicted_list) - true_positive,
        len(gold_list) - true_positive,
    )


def jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def _levenshtein_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            substitution = previous[right_index - 1] + (left_value != right_value)
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    return _levenshtein_distance(reference_words, hypothesis_words) / len(reference_words)


def _strict_match(gold: list[EntityAnnotation], predicted: list[EntityAnnotation]) -> list[tuple[EntityAnnotation, EntityAnnotation]]:
    predictions_by_key: dict[tuple[int, int, str], list[EntityAnnotation]] = {}
    for item in predicted:
        predictions_by_key.setdefault((item.start, item.end, item.type), []).append(item)
    matches: list[tuple[EntityAnnotation, EntityAnnotation]] = []
    for target in gold:
        key = (target.start, target.end, target.type)
        bucket = predictions_by_key.get(key, [])
        if bucket:
            matches.append((target, bucket.pop(0)))
    return matches


def _approximate_match(gold: list[EntityAnnotation], predicted: list[EntityAnnotation]) -> list[tuple[EntityAnnotation, EntityAnnotation]]:
    used: set[int] = set()
    matches: list[tuple[EntityAnnotation, EntityAnnotation]] = []
    for target in gold:
        best_index = None
        best_score = 0.0
        for index, candidate in enumerate(predicted):
            if index in used or candidate.type != target.type:
                continue
            overlap = span_overlap(target, candidate)
            union = max(target.end, candidate.end) - min(target.start, candidate.start)
            overlap_score = overlap / union if union else 0.0
            text_score = max(0.0, 1.0 - word_error_rate(target.text, candidate.text))
            score = 0.65 * overlap_score + 0.35 * text_score
            if score > best_score:
                best_score = score
                best_index = index
        if best_index is not None and best_score >= 0.35:
            used.add(best_index)
            matches.append((target, predicted[best_index]))
    return matches


def competition_score(
    gold: Iterable[EntityAnnotation], predicted: Iterable[EntityAnnotation], approximate: bool = False
) -> dict[str, Any]:
    gold_list = list(gold)
    predicted_list = list(predicted)
    matches = _approximate_match(gold_list, predicted_list) if approximate else _strict_match(gold_list, predicted_list)
    text_scores: list[float] = []
    assertion_scores: list[float] = []
    candidate_scores: list[float] = []
    candidate_weights: list[int] = []
    for target, candidate in matches:
        text_scores.append(max(0.0, 1.0 - word_error_rate(target.text, candidate.text)))
        assertion_scores.append(jaccard(target.assertions, candidate.assertions))
        candidate_scores.append(jaccard(target.candidates, candidate.candidates))
        candidate_weights.append(max(1, len(target.candidates)))
    text_score = safe_divide(sum(text_scores), len(gold_list))
    assertions_score = safe_divide(sum(assertion_scores), len(gold_list))
    weighted_candidate_sum = sum(score * weight for score, weight in zip(candidate_scores, candidate_weights))
    total_candidate_weight = sum(max(1, len(item.candidates)) for item in gold_list)
    candidates_score = safe_divide(weighted_candidate_sum, total_candidate_weight)
    final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
    return {
        "matching": "approximate_greedy" if approximate else "strict_exact_position_type",
        "gold_entities": len(gold_list),
        "predicted_entities": len(predicted_list),
        "matched_entities": len(matches),
        "text_score": round(text_score, 6),
        "assertions_score": round(assertions_score, 6),
        "candidates_score": round(candidates_score, 6),
        "final_score": round(final_score, 6),
        "is_official": False,
        "limitation": "Organizer matching and WER details were not provided; this evaluator is provisional.",
    }
