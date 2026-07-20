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


def evaluate_benchmark(
    eval_input: str | Path,
    eval_gt_dir: str | Path,
    pipeline: Any,
    run_root: str | Path,
) -> dict[str, Any]:
    """
    Đánh giá bóc tách theo 3 Stage (NER, Retrieval Top-K, Linking) và xuất báo cáo phân tích lỗi (Error Diagnostics).
    """
    import json
    from pathlib import Path
    from .schema import write_json

    eval_input = Path(eval_input)
    eval_gt_dir = Path(eval_gt_dir)
    run_root = Path(run_root)
    eval_output_dir = run_root / "eval_output"
    diagnostics_dir = run_root / "diagnostics"
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    # 1. Run Pipeline Inference
    from .pipeline import run_inference
    run_inference(
        eval_input,
        eval_output_dir,
        pipeline.artifact_dir,
        create_zip=False,
        ner_model_dir=pipeline.trans_detector.model_dir if pipeline.trans_detector else None,
    )

    gt_files = list(eval_gt_dir.glob("*.json"))
    
    # Stage 1 (NER) Stats
    total_gt_entities = 0
    total_pred_entities = 0
    correct_ner_exact = 0
    boundary_mismatch_count = 0
    
    # Stage 2 (Retrieval) Stats
    retrieval_eval_count = 0
    top5_hits = 0
    top10_hits = 0
    
    # Stage 3 (Linking) Stats
    correct_linking_strict = 0

    # Error Diagnostics Categories
    error_analysis = {
        "missed_entities": [],
        "boundary_mismatches": [],
        "retrieval_misses": [],
        "wrong_code_links": [],
        "spurious_entities": [],
    }

    per_type_stats: dict[str, dict[str, int]] = {}

    for gt_file in gt_files:
        doc_id = gt_file.stem
        pred_file = eval_output_dir / f"{doc_id}.json"
        if not pred_file.is_file():
            continue

        gt_data = json.loads(gt_file.read_text(encoding="utf-8"))
        pred_data = json.loads(pred_file.read_text(encoding="utf-8"))

        gt_ents = gt_data.get("entities", [])
        pred_ents = pred_data.get("entities", [])

        total_gt_entities += len(gt_ents)
        total_pred_entities += len(pred_ents)

        pred_exact_map = {(e["start"], e["end"], e["label"]): e for e in pred_ents}
        gt_exact_map = {(e["start"], e["end"], e["label"]): e for e in gt_ents}

        # Track matched predictions
        matched_pred_indices = set()

        for ge in gt_ents:
            etype = ge.get("label", "UNKNOWN")
            if etype not in per_type_stats:
                per_type_stats[etype] = {"gt": 0, "correct_ner": 0, "correct_code": 0}
            per_type_stats[etype]["gt"] += 1

            gt_code = ge.get("code")
            gt_text = ge.get("text", "")
            gt_key = (ge["start"], ge["end"], ge["label"])

            # --- STAGE 1: NER EVALUATION ---
            if gt_key in pred_exact_map:
                correct_ner_exact += 1
                per_type_stats[etype]["correct_ner"] += 1
                pe = pred_exact_map[gt_key]
                pred_idx = pred_ents.index(pe)
                matched_pred_indices.add(pred_idx)

                # --- STAGE 3: LINKING EVALUATION ---
                pred_code = pe.get("code")
                if gt_code and gt_code == pred_code:
                    correct_linking_strict += 1
                    per_type_stats[etype]["correct_code"] += 1
                elif gt_code and pred_code != gt_code:
                    error_analysis["wrong_code_links"].append({
                        "doc_id": doc_id,
                        "text": gt_text,
                        "label": etype,
                        "gt_code": gt_code,
                        "pred_code": pred_code,
                    })
            else:
                # Check for boundary mismatch (overlapping span)
                overlapping_preds = [
                    (idx, pe) for idx, pe in enumerate(pred_ents)
                    if pe["label"] == ge["label"]
                    and max(0, min(ge["end"], pe["end"]) - max(ge["start"], pe["start"])) > 0
                ]
                if overlapping_preds:
                    boundary_mismatch_count += 1
                    best_idx, pe = overlapping_preds[0]
                    matched_pred_indices.add(best_idx)
                    error_analysis["boundary_mismatches"].append({
                        "doc_id": doc_id,
                        "gt_text": gt_text,
                        "pred_text": pe.get("text", ""),
                        "gt_offsets": [ge["start"], ge["end"]],
                        "pred_offsets": [pe["start"], pe["end"]],
                        "label": etype,
                    })
                else:
                    error_analysis["missed_entities"].append({
                        "doc_id": doc_id,
                        "text": gt_text,
                        "offsets": [ge["start"], ge["end"]],
                        "label": etype,
                        "gt_code": gt_code,
                    })

            # --- STAGE 2: RETRIEVAL TOP-K EVALUATION ---
            if gt_code and hasattr(pipeline, "icd10_index") and hasattr(pipeline, "rxnorm_index"):
                retrieval_eval_count += 1
                index = pipeline.icd10_index if etype in {"DISEASE", "SYMPTOM", "LAB_RESULT"} else pipeline.rxnorm_index
                try:
                    cands = index.retrieve(gt_text, top_k=10)
                    cand_codes = [str(c.get("candidate_id") or c.get("code")) for c in cands]
                    if str(gt_code) in cand_codes[:5]:
                        top5_hits += 1
                    if str(gt_code) in cand_codes[:10]:
                        top10_hits += 1
                    else:
                        error_analysis["retrieval_misses"].append({
                            "doc_id": doc_id,
                            "text": gt_text,
                            "label": etype,
                            "gt_code": gt_code,
                            "top_candidates": [c.get("name") for c in cands[:3]],
                        })
                except Exception:
                    pass

        # Spurious entities (False Positives)
        for idx, pe in enumerate(pred_ents):
            if idx not in matched_pred_indices:
                error_analysis["spurious_entities"].append({
                    "doc_id": doc_id,
                    "text": pe.get("text", ""),
                    "offsets": [pe["start"], pe["end"]],
                    "label": pe.get("label"),
                    "pred_code": pe.get("code"),
                })

    # Metrics calculation
    ner_precision = safe_divide(correct_ner_exact, total_pred_entities)
    ner_recall = safe_divide(correct_ner_exact, total_gt_entities)
    ner_f1 = safe_divide(2 * ner_precision * ner_recall, ner_precision + ner_recall)
    
    top5_recall = safe_divide(top5_hits, retrieval_eval_count)
    top10_recall = safe_divide(top10_hits, retrieval_eval_count)
    linking_acc = safe_divide(correct_linking_strict, correct_ner_exact)

    report = {
        "documents_evaluated": len(gt_files),
        "stage1_ner": {
            "gt_entities": total_gt_entities,
            "pred_entities": total_pred_entities,
            "exact_matched": correct_ner_exact,
            "boundary_mismatches": boundary_mismatch_count,
            "precision": round(ner_precision, 4),
            "recall": round(ner_recall, 4),
            "f1": round(ner_f1, 4),
        },
        "stage2_retrieval": {
            "evaluated_entities": retrieval_eval_count,
            "top5_hit_count": top5_hits,
            "top10_hit_count": top10_hits,
            "top5_recall": round(top5_recall, 4),
            "top10_recall": round(top10_recall, 4),
        },
        "stage3_linking": {
            "strict_linking_matched": correct_linking_strict,
            "strict_linking_accuracy": round(linking_acc, 4),
        },
        "per_type_breakdown": per_type_stats,
        "error_counts": {
            "missed_entities": len(error_analysis["missed_entities"]),
            "boundary_mismatches": len(error_analysis["boundary_mismatches"]),
            "retrieval_misses": len(error_analysis["retrieval_misses"]),
            "wrong_code_links": len(error_analysis["wrong_code_links"]),
            "spurious_entities": len(error_analysis["spurious_entities"]),
        }
    }

    # Save Error Diagnostics Report
    write_json(diagnostics_dir / "benchmark_error_analysis.json", error_analysis)
    write_json(diagnostics_dir / "benchmark_summary.json", report)

    # Print Formatted Output
    print("\n" + "="*60)
    print(f"📊 STAGE-BY-STAGE BENCHMARK EVALUATION RESULTS ({len(gt_files)} docs)")
    print("="*60)
    print("🔹 STAGE 1: NER (Entity Recognition)")
    print(f"   - Total GT / Pred Entities : {total_gt_entities} / {total_pred_entities}")
    print(f"   - Precision                : {ner_precision:.2%}")
    print(f"   - Recall                   : {ner_recall:.2%}")
    print(f"   - F1-Score                 : {ner_f1:.2%}")
    print(f"   - Boundary Mismatches      : {boundary_mismatch_count}")
    print("-" * 60)
    print("🔹 STAGE 2: CANDIDATE RETRIEVAL (BM25 + FAISS)")
    print(f"   - Evaluated GT Entities    : {retrieval_eval_count}")
    print(f"   - Top-5 Recall  (Hit@5)    : {top5_recall:.2%} ({top5_hits}/{retrieval_eval_count})")
    print(f"   - Top-10 Recall (Hit@10)   : {top10_recall:.2%} ({top10_hits}/{retrieval_eval_count})")
    print("-" * 60)
    print("🔹 STAGE 3: LLM RERANKER & LINKING")
    print(f"   - Strict Linking Accuracy  : {linking_acc:.2%} ({correct_linking_strict}/{correct_ner_exact})")
    print("-" * 60)
    print("📋 ERROR DIAGNOSTICS SUMMARY")
    print(f"   - Lỗi bỏ sót thực thể (Missed)      : {len(error_analysis['missed_entities'])}")
    print(f"   - Lỗi lệch ranh giới (Boundary)    : {len(error_analysis['boundary_mismatches'])}")
    print(f"   - Lỗi Candidate sót (Retrieval Miss): {len(error_analysis['retrieval_misses'])}")
    print(f"   - Lỗi Reranker gán sai mã (Wrong Code): {len(error_analysis['wrong_code_links'])}")
    print(f"   - Lỗi bắt nhầm rác (Spurious FP)   : {len(error_analysis['spurious_entities'])}")
    print("="*60)
    print(f"💾 Detailed error analysis saved to: {diagnostics_dir / 'benchmark_error_analysis.json'}\n")

    return report
