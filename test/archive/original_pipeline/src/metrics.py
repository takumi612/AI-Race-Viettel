#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import io

# Ép terminal Windows hiển thị UTF-8 tránh lỗi ký tự tiếng Việt
if sys.platform.startswith('win'):
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except AttributeError:
        pass  # Đề phòng môi trường không hỗ trợ buffer (vd: test suite giả lập)

"""
Evaluation Engine for AI Race 2026 - Bài 2 (Ontological Reasoning in Medical Knowledge Retrieval)
Developed to match BTC's scoring guidelines as detailed in the competition specs.
Includes robust optimizations for lowercase matching, whitespace stripping, and handling edge cases.

LƯU Ý QUAN TRỌNG (giả định không được đề bài quy định rõ):
Đề bài KHÔNG mô tả cách ghép cặp (matching) 1 khái niệm ground-truth với 1 khái niệm
prediction khi số lượng 2 bên lệch nhau. Engine này tự chọn cách ghép: bipartite
greedy matching theo "type" trùng khớp + IoU vị trí ký tự >= iou_threshold (mặc định 0.5).
J_assertions(i)/J_candidates(i) của mỗi sample được tính bằng TRUNG BÌNH Jaccard trên
từng cặp khái niệm đã ghép (entity-level), KHÔNG phải Jaccard trên tập phẳng gộp toàn
sample. Đây là cách hiểu hợp lý nhất theo tinh thần đề bài (xem note "sai loại bị tính
2 lần, 0 điểm cả 3 metric"), nhưng nếu BTC dùng ngưỡng IoU khác hoặc cách ghép khác thì
điểm tự chấm ở đây có thể lệch so với điểm thật trên hệ thống. iou_threshold được expose
ra để dễ đo độ nhạy của điểm số theo giả định này (xem --iou-threshold trong evaluate.py).
"""

# Các loại khái niệm được xét assertions / candidates theo đúng mục 3.2 Output của đề bài
ASSERTION_TYPES = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"}
CANDIDATE_TYPES = {"CHẨN_ĐOÁN", "THUỐC"}

# Ngưỡng IoU mặc định dùng để ghép cặp GT-Pred (giả định, xem lưu ý phía trên)
DEFAULT_IOU_THRESHOLD = 0.5


def calculate_iou(pos1, pos2):
    """
    Calculate Intersection over Union (IoU) of two character position intervals.
    pos1, pos2: list or tuple of [start, end] (0-indexed, exclusive end)
    """
    s1, e1 = pos1
    s2, e2 = pos2
    inter = max(0, min(e1, e2) - max(s1, s2))
    union = (e1 - s1) + (e2 - s2) - inter
    if union == 0:
        return 0.0
    return inter / union


def calculate_wer(ref_text, hyp_text, case_insensitive=True):
    """
    Calculate Word Error Rate (WER) between two text strings using Levenshtein distance at word level.
    Splits text by whitespace. By default, ignores case to align with medical concept matching.
    """
    if case_insensitive:
        ref_text = ref_text.lower()
        hyp_text = hyp_text.lower()
        
    ref_words = ref_text.strip().split()
    hyp_words = hyp_text.strip().split()
    
    if not ref_words:
        return float(len(hyp_words))
    
    n = len(ref_words)
    m = len(hyp_words)
    
    # DP table for Levenshtein distance
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
        
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1   # Substitution
                )
    return dp[n][m] / n


def greedy_match(gts, preds, iou_threshold=0.5):
    """
    Perform greedy bipartite matching between Ground Truths and Predictions.
    Matching criteria:
    - Must have the identical 'type'
    - Character position IoU must be >= iou_threshold
    
    gts, preds: list of dictionaries representing entities.
    Returns:
    - matches: list of tuples (gt_idx, pred_idx)
    - unmatched_gts: list of gt indices that did not match
    - unmatched_preds: list of pred indices that did not match
    """
    gt_indexed = [(i, gt) for i, gt in enumerate(gts)]
    pred_indexed = [(j, pred) for j, pred in enumerate(preds)]
    
    candidates = []
    for i, gt in gt_indexed:
        gt_type = str(gt.get("type", "")).strip()
        for j, pred in pred_indexed:
            pred_type = str(pred.get("type", "")).strip()
            if gt_type and gt_type == pred_type:
                iou = calculate_iou(gt["position"], pred["position"])
                if iou >= iou_threshold:
                    candidates.append((iou, i, j))
                    
    # Sort candidates by IoU descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    matched_gts = set()
    matched_preds = set()
    matches = []
    
    for iou, i, j in candidates:
        if i not in matched_gts and j not in matched_preds:
            matched_gts.add(i)
            matched_preds.add(j)
            matches.append((i, j))
            
    unmatched_gts = [i for i, _ in gt_indexed if i not in matched_gts]
    unmatched_preds = [j for j, _ in pred_indexed if j not in matched_preds]
    
    return matches, unmatched_gts, unmatched_preds


def calculate_jaccard(gt_list, pred_list):
    """
    Calculate Jaccard similarity between two lists of items.
    Edge cases defined by BTC:
    - Both empty: Jaccard = 1.0
    - One empty, other not: Jaccard = 0.0
    """
    gt_set = set(gt_list)
    pred_set = set(pred_list)
    if not gt_set and not pred_set:
        return 1.0
    if not gt_set or not pred_set:
        return 0.0
    inter = len(gt_set.intersection(pred_set))
    union = len(gt_set.union(pred_set))
    return inter / union


def _candidate_weight(gt_sample):
    """
    W(i) = sum_{k in sample i, k thuộc CANDIDATE_TYPES} (len(ground_truth_candidates(k)) + 1)
    Chỉ phụ thuộc ground truth, tách riêng để có thể tái sử dụng khi 1 sample bị lỗi
    (xem evaluate_dataset) mà vẫn tính đúng trọng số của sample đó.
    """
    weight = 0
    for gt in gt_sample:
        if gt.get("type") in CANDIDATE_TYPES:
            weight += len(gt.get("candidates", []) or []) + 1
    return weight


def evaluate_sample(gt_sample, pred_sample, verbose=False, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    Evaluate a single sample containing multiple entities.
    Returns:
    - sample_text_score (float)
    - sample_assertions_score (float)
    - sample_candidates_jaccard (float)
    - weight (int) - weight of the sample based on number of candidate codes in ground truth
    """
    # 1. Bipartite matching (xem lưu ý về giả định matching ở đầu file)
    matches, unmatched_gts, unmatched_preds = greedy_match(gt_sample, pred_sample, iou_threshold=iou_threshold)
    
    if verbose:
        print(f"  Matches: {len(matches)} | Unmatched GT (FN): {len(unmatched_gts)} | Unmatched Pred (FP): {len(unmatched_preds)}")
        for gt_idx, pred_idx in matches:
            print(f"    Match: GT '{gt_sample[gt_idx]['text']}' <-> Pred '{pred_sample[pred_idx]['text']}' | Type: {gt_sample[gt_idx]['type']}")
        for gt_idx in unmatched_gts:
            print(f"    FN (Missing GT): '{gt_sample[gt_idx]['text']}' | Type: {gt_sample[gt_idx]['type']}")
        for pred_idx in unmatched_preds:
            print(f"    FP (Extra Pred): '{pred_sample[pred_idx]['text']}' | Type: {pred_sample[pred_idx]['type']}")

    # 2. Text Score (WER)
    total_text_score = 0.0
    for gt_idx, pred_idx in matches:
        wer = calculate_wer(gt_sample[gt_idx]["text"], pred_sample[pred_idx]["text"], case_insensitive=True)
        total_text_score += max(0.0, 1.0 - wer)
        
    union_size = len(gt_sample) + len(pred_sample) - len(matches)
    if union_size == 0:
        sample_text_score = 1.0
    else:
        sample_text_score = total_text_score / union_size
        
    # 3. Assertions Score (Jaccard similarity with corresponding entities)
    # Only applicable to: CHẨN_ĐOÁN, THUỐC, TRIỆU_CHỨNG
    # Filter matched / unmatched entities that support assertions
    assertion_matches = [(g, p) for g, p in matches if gt_sample[g]["type"] in ASSERTION_TYPES]
    assertion_fns = [g for g in unmatched_gts if gt_sample[g]["type"] in ASSERTION_TYPES]
    assertion_fps = [p for p in unmatched_preds if pred_sample[p]["type"] in ASSERTION_TYPES]
    
    total_assertion_jaccard = 0.0
    for g, p in assertion_matches:
        # Standardize strings by stripping whitespace
        gt_assertions = [lbl.strip() for lbl in gt_sample[g].get("assertions", []) if lbl.strip()]
        pred_assertions = [lbl.strip() for lbl in pred_sample[p].get("assertions", []) if lbl.strip()]
        j = calculate_jaccard(gt_assertions, pred_assertions)
        total_assertion_jaccard += j
        
    assertion_union_size = len(assertion_matches) + len(assertion_fns) + len(assertion_fps)
    if assertion_union_size == 0:
        sample_assertions_score = 1.0
    else:
        sample_assertions_score = total_assertion_jaccard / assertion_union_size
    
    # 4. Candidates Score (Jaccard similarity with corresponding entities)
    # Only applicable to: CHẨN_ĐOÁN, THUỐC
    candidate_matches = [(g, p) for g, p in matches if gt_sample[g]["type"] in CANDIDATE_TYPES]
    candidate_fns = [g for g in unmatched_gts if gt_sample[g]["type"] in CANDIDATE_TYPES]
    candidate_fps = [p for p in unmatched_preds if pred_sample[p]["type"] in CANDIDATE_TYPES]
    
    total_candidate_jaccard = 0.0
    for g, p in candidate_matches:
        # Standardize standard codes by stripping and converting to uppercase (e.g. "k21.9" -> "K21.9")
        gt_candidates = [code.strip().upper() for code in gt_sample[g].get("candidates", []) if code.strip()]
        pred_candidates = [code.strip().upper() for code in pred_sample[p].get("candidates", []) if code.strip()]
        j = calculate_jaccard(gt_candidates, pred_candidates)
        total_candidate_jaccard += j
        
    candidate_union_size = len(candidate_matches) + len(candidate_fns) + len(candidate_fps)
    if candidate_union_size == 0:
        sample_candidates_jaccard = 1.0
    else:
        sample_candidates_jaccard = total_candidate_jaccard / candidate_union_size
        
    # Weight of this sample for dataset candidates aggregation
    # W(i) = sum_{k in sample i} (len(ground_truth_candidates(k)) + 1)
    weight = _candidate_weight(gt_sample)
            
    return sample_text_score, sample_assertions_score, sample_candidates_jaccard, weight


def evaluate_dataset(gt_dataset, pred_dataset, verbose=False, iou_threshold=DEFAULT_IOU_THRESHOLD):
    """
    Evaluate dataset-wide scores.
    gt_dataset, pred_dataset: list of list of dictionaries representing entities per sample
    iou_threshold: ngưỡng IoU dùng để ghép cặp GT-Pred (xem lưu ý về giả định ở đầu file)
    """
    n_samples = len(gt_dataset)
    if n_samples == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    total_text = 0.0
    total_assertions = 0.0
    weighted_candidates_sum = 0.0
    total_weight = 0
    
    for i in range(n_samples):
        if verbose:
            print(f"--- Evaluating Sample {i+1}/{n_samples} ---")
        gt_s = gt_dataset[i]
        pred_s = pred_dataset[i]

        # Cách ly lỗi: nếu 1 sample có dữ liệu bất thường (VD: prediction sai
        # cấu trúc lọt qua bước validate ở evaluate.py) thì chỉ sample đó bị
        # tính 0 điểm, KHÔNG làm sập toàn bộ batch chấm điểm của các đội khác.
        try:
            text_s, assert_s, cand_jaccard_s, weight_s = evaluate_sample(
                gt_s, pred_s, verbose=verbose, iou_threshold=iou_threshold
            )
        except Exception as e:
            print(f"  [WARNING] Lỗi khi chấm sample {i+1}: {e!r}. Gán 0 điểm cho sample này.")
            text_s, assert_s, cand_jaccard_s = 0.0, 0.0, 0.0
            weight_s = _candidate_weight(gt_s)
        
        total_text += text_s
        total_assertions += assert_s
        
        weighted_candidates_sum += cand_jaccard_s * weight_s
        total_weight += weight_s
        
        if verbose:
            print(f"  Text Score: {text_s:.4f} | Assertion Score: {assert_s:.4f} | Candidates Jaccard: {cand_jaccard_s:.4f} (Weight: {weight_s})\n")
        
    final_text_score = total_text / n_samples
    final_assertions_score = total_assertions / n_samples
    
    if total_weight == 0:
        # Check if the predictions contain any candidate predictions at all
        pred_has_candidates = False
        for pred_s in pred_dataset:
            for pred in pred_s:
                if pred.get("type") in CANDIDATE_TYPES and pred.get("candidates"):
                    pred_has_candidates = True
                    break
        final_candidates_score = 0.0 if pred_has_candidates else 1.0
    else:
        final_candidates_score = weighted_candidates_sum / total_weight
        
    final_score = 0.3 * final_text_score + 0.3 * final_assertions_score + 0.4 * final_candidates_score
    
    return final_text_score, final_assertions_score, final_candidates_score, final_score


def run_tests():
    """
    Run automated unit tests to verify the correctness of the evaluation engine.
    """
    print("=================== RUNNING ENGINE TESTS ===================")
    
    # Ca kiểm thử 1: Khớp hoàn hảo
    print("\n[Test Case 1] Perfect Match (Ground Truth == Prediction)")
    gt_s1 = [
        {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "position": [58, 83], "assertions": ["isHistorical"], "candidates": ["308135"]},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [196, 198], "assertions": []}
    ]
    pred_s1 = [
        {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "position": [58, 83], "assertions": ["isHistorical"], "candidates": ["308135"]},
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [196, 198], "assertions": []}
    ]
    text, assertions, candidates, final = evaluate_dataset([gt_s1], [pred_s1], verbose=True)
    assert abs(final - 1.0) < 1e-6, f"Perfect match score should be 1.0, got {final}"
    print(f"PASSED. Scores -> Text: {text:.4f}, Assertions: {assertions:.4f}, Candidates: {candidates:.4f}, Final: {final:.4f}")
    
    # Ca kiểm thử 2: Lệch vị trí nhẹ nhưng IoU >= 0.5 và thiếu một số từ (WER > 0)
    print("\n[Test Case 2] Word level mismatch (WER > 0) with IoU >= 0.5")
    # Ground truth: amlodipine 10 mg po daily (5 words)
    # Pred: amlodipine 10 mg (3 words), position overlaps heavily
    gt_s2 = [
        {"text": "amlodipine 10 mg po daily", "type": "THUỐC", "position": [58, 83], "assertions": ["isHistorical"], "candidates": ["308135"]}
    ]
    pred_s2 = [
        {"text": "amlodipine 10 mg", "type": "THUỐC", "position": [58, 74], "assertions": ["isHistorical"], "candidates": ["308135"]}
    ]
    # IoU: 16/25 = 0.64 (matches)
    # WER: deletions = 2, total = 5 -> WER = 0.4. Text score = 1.0 - 0.4 = 0.6.
    # Assertions should match perfectly (both "isHistorical" for matched THUỐC) -> 1.0.
    # Candidates should match perfectly (both ["308135"] for matched THUỐC) -> 1.0.
    text, assertions, candidates, final = evaluate_dataset([gt_s2], [pred_s2], verbose=True)
    assert abs(text - 0.6) < 1e-6, f"Expected text score 0.6, got {text}"
    assert abs(assertions - 1.0) < 1e-6, f"Expected assertions score 1.0, got {assertions}"
    assert abs(candidates - 1.0) < 1e-6, f"Expected candidates score 1.0, got {candidates}"
    print(f"PASSED. Text score exact match with Levenshtein-WER formula. Final: {final:.4f}")
    
    # Ca kiểm thử 3: Sai nhãn loại thực thể (type)
    print("\n[Test Case 3] Entity type mismatch (counts as FP + FN)")
    gt_s3 = [
        {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [196, 198], "assertions": []}
    ]
    pred_s3 = [
        {"text": "ho", "type": "CHẨN_ĐOÁN", "position": [196, 198], "assertions": [], "candidates": ["K21.9"]}
    ]
    # Type mismatch -> Matches: 0. Unmatched GT = 1, Unmatched Pred = 1. Union size = 2.
    # Text Score = 0 / 2 = 0.0.
    # Assertions: matched = 0, GT FN = 1 (TRIỆU_CHỨNG), Pred FP = 1 (CHẨN_ĐOÁN). Union = 2. Score = 0.0.
    # Candidates: matched = 0, GT FN = 0 (TRIỆU_CHỨNG, ignored), Pred FP = 1 (CHẨN_ĐOÁN, has candidates). Union = 1. Score = 0.0. Weight = 0.
    text, assertions, candidates, final = evaluate_dataset([gt_s3], [pred_s3], verbose=True)
    assert abs(text - 0.0) < 1e-6, f"Expected text score 0.0, got {text}"
    assert abs(assertions - 0.0) < 1e-6, f"Expected assertions score 0.0, got {assertions}"
    assert abs(candidates - 0.0) < 1e-6, f"Expected candidates score 0.0, got {candidates}"
    print(f"PASSED. Mismatch type penalized correctly. Scores -> Text: {text:.4f}, Assertions: {assertions:.4f}, Candidates: {candidates:.4f}, Final: {final:.4f}")

    # Ca kiểm thử 4: Hoán đổi gán mã candidates (Thử thách của Jaccard định vị tuple)
    print("\n[Test Case 4] Swapped candidates code check")
    gt_s4 = [
        {"text": "Bệnh A", "type": "CHẨN_ĐOÁN", "position": [10, 20], "assertions": [], "candidates": ["A01"]},
        {"text": "Bệnh B", "type": "CHẨN_ĐOÁN", "position": [30, 40], "assertions": [], "candidates": ["B02"]}
    ]
    pred_s4 = [
        {"text": "Bệnh A", "type": "CHẨN_ĐOÁN", "position": [10, 20], "assertions": [], "candidates": ["B02"]},
        {"text": "Bệnh B", "type": "CHẨN_ĐOÁN", "position": [30, 40], "assertions": [], "candidates": ["A01"]}
    ]
    # Matched A<->A, B<->B.
    # Candidates matched A: GT ["A01"] vs Pred ["B02"] -> Jaccard = 0.0
    # Candidates matched B: GT ["B02"] vs Pred ["A01"] -> Jaccard = 0.0
    # Average Candidates score = 0.0. Weight = (1+1) + (1+1) = 4.
    text, assertions, candidates, final = evaluate_dataset([gt_s4], [pred_s4], verbose=True)
    assert abs(candidates - 0.0) < 1e-6, f"Swapped candidates should get 0.0, got {candidates}"
    print(f"PASSED. Candidates code swap correctly computed as 0.0 Jaccard. Scores -> Text: {text:.4f}, Assertions: {assertions:.4f}, Candidates: {candidates:.4f}, Final: {final:.4f}")

    # Ca kiểm thử 5: Case-insensitive và Whitespace stripping check
    print("\n[Test Case 5] Case-insensitive and whitespace stripping check")
    gt_s5 = [
        {"text": "Amlodipine 10 MG", "type": "THUỐC", "position": [58, 74], "assertions": [" isHistorical "], "candidates": [" 308135 "]}
    ]
    pred_s5 = [
        {"text": "amlodipine 10 mg", "type": "THUỐC", "position": [58, 74], "assertions": ["isHistorical"], "candidates": ["308135"]}
    ]
    text, assertions, candidates, final = evaluate_dataset([gt_s5], [pred_s5], verbose=True)
    assert abs(text - 1.0) < 1e-6, f"Expected text score 1.0, got {text}"
    assert abs(assertions - 1.0) < 1e-6, f"Expected assertions score 1.0, got {assertions}"
    assert abs(candidates - 1.0) < 1e-6, f"Expected candidates score 1.0, got {candidates}"
    print(f"PASSED. Case and spaces normalized successfully. Scores -> Text: {text:.4f}, Assertions: {assertions:.4f}, Candidates: {candidates:.4f}, Final: {final:.4f}")

    # Ca kiểm thử 6: iou_threshold có thể cấu hình được
    print("\n[Test Case 6] Configurable iou_threshold")
    # IoU thực tế = 0.5 đúng bằng ngưỡng mặc định -> match khi threshold <= 0.5,
    # KHÔNG match khi threshold > 0.5 (VD 0.6)
    gt_s6 = [{"text": "sot", "type": "TRIỆU_CHỨNG", "position": [0, 10], "assertions": []}]
    pred_s6 = [{"text": "sot", "type": "TRIỆU_CHỨNG", "position": [5, 15], "assertions": []}]
    # inter = 5, union = 15 -> IoU = 1/3 ~ 0.333
    text_low, _, _, _ = evaluate_dataset([gt_s6], [pred_s6], iou_threshold=0.3)
    text_high, _, _, _ = evaluate_dataset([gt_s6], [pred_s6], iou_threshold=0.5)
    assert text_low > 0.0, "Với threshold 0.3 (thấp hơn IoU thực tế) phải match được"
    assert text_high == 0.0, "Với threshold 0.5 (cao hơn IoU thực tế ~0.33) không được match"
    print(f"PASSED. threshold=0.3 -> text_score={text_low:.4f} (match) | threshold=0.5 -> text_score={text_high:.4f} (không match)")

    # Ca kiểm thử 7: 1 sample lỗi dữ liệu không được làm sập cả batch
    print("\n[Test Case 7] Lỗi ở 1 sample không làm sập toàn bộ batch")
    gt_ok = [{"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2], "assertions": []}]
    pred_ok = [{"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2], "assertions": []}]
    pred_broken = [{"text": "sot", "type": "TRIỆU_CHỨNG"}]  # thiếu "position" -> lỗi khi tính IoU
    text, assertions, candidates, final = evaluate_dataset(
        [gt_ok, gt_ok], [pred_ok, pred_broken], verbose=True
    )
    # Sample 1 đạt 1.0, sample 2 lỗi -> gán 0.0. Trung bình text_score phải là 0.5, không phải crash.
    assert abs(text - 0.5) < 1e-6, f"Kỳ vọng text_score trung bình 0.5 (1 sample OK, 1 sample lỗi=0), got {text}"
    print(f"PASSED. Batch vẫn chạy hết dù 1 sample lỗi. Scores -> Text: {text:.4f}, Assertions: {assertions:.4f}, Candidates: {candidates:.4f}, Final: {final:.4f}")

    print("\n=================== ALL TESTS PASSED SUCCESSFULLY ===================")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        print("Usage: python metrics.py test")
        run_tests()
