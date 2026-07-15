#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
evaluate.py — CLI Evaluation Runner for AI Race 2026 - Bài 2

Chạy đánh giá metrics giữa 2 thư mục:
  - data/dev/:     Chứa các file JSON nhãn chuẩn (N.json)
  - data/output/:  Chứa các file JSON dự đoán   (N.json)

Usage (chạy từ thư mục gốc dự án):
    python src/evaluate.py
    python src/evaluate.py --verbose
    python src/evaluate.py --file 11
    python src/evaluate.py --gt data/dev/ --pred data/output/
"""

import os
import json
import argparse
import sys
import io

# Ép terminal Windows hiển thị UTF-8 tránh lỗi ký tự tiếng Việt
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Hỗ trợ import metrics.py trong cùng thư mục
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import metrics

REQUIRED_KEYS = ("text", "type", "position")
LIST_IF_PRESENT_KEYS = ("assertions", "candidates")

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def validate_entities(data, fname):
    """
    Kiểm tra 1 file JSON output có đúng cấu trúc theo mục 3.2 Output của đề bài không:
      - root phải là list
      - mỗi phần tử phải là dict, có đủ "text" (str), "type" (str), "position" (list 2 số)
      - "assertions"/"candidates" nếu có phải là list
    """
    if not isinstance(data, list):
        return False, f"{fname}: root phải là list, nhận kiểu '{type(data).__name__}'"

    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            return False, f"{fname}[{idx}]: mỗi khái niệm phải là object/dict"

        for key in REQUIRED_KEYS:
            if key not in item:
                return False, f"{fname}[{idx}]: thiếu field bắt buộc '{key}'"

        if not isinstance(item.get("text"), str):
            return False, f"{fname}[{idx}]: field 'text' phải là string"

        if not isinstance(item.get("type"), str) or not item["type"].strip():
            return False, f"{fname}[{idx}]: field 'type' phải là string không rỗng"

        pos = item.get("position")
        if not (
            isinstance(pos, (list, tuple))
            and len(pos) == 2
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in pos)
        ):
            return False, f"{fname}[{idx}]: field 'position' phải là list gồm 2 số, nhận {pos!r}"

        for key in LIST_IF_PRESENT_KEYS:
            if key in item and not isinstance(item[key], list):
                return False, f"{fname}[{idx}]: field '{key}' phải là list nếu có mặt"

    return True, None

def run_evaluation(
    gt_dir: str,
    pred_dir: str,
    verbose: bool = False,
    single_file: str = None,
    iou_threshold: float = metrics.DEFAULT_IOU_THRESHOLD,
):
    gt_dir = os.path.abspath(gt_dir)
    pred_dir = os.path.abspath(pred_dir)

    if not os.path.isdir(gt_dir):
        print(f"[ERROR] Thư mục GT không tồn tại: {gt_dir}")
        sys.exit(1)
    if not os.path.isdir(pred_dir):
        print(f"[ERROR] Thư mục Pred không tồn tại: {pred_dir}")
        sys.exit(1)

    def _sort_key(fname):
        stem = os.path.splitext(fname)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    gt_files = sorted(
        [f for f in os.listdir(gt_dir) if f.endswith(".json")],
        key=_sort_key,
    )

    if not gt_files:
        print(f"[ERROR] Không tìm thấy file JSON nào trong thư mục GT: {gt_dir}")
        sys.exit(1)

    if single_file:
        fname = single_file if single_file.endswith(".json") else single_file + ".json"
        gt_files = [f for f in gt_files if f == fname]
        if not gt_files:
            print(f"[ERROR] Không tìm thấy file '{fname}' trong {gt_dir}")
            sys.exit(1)

    gt_list = []
    pred_list = []
    missing_preds = []
    invalid_preds = []

    for fname in gt_files:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)

        try:
            gt_data = load_json(gt_path)
        except json.JSONDecodeError as e:
            print(f"[ERROR] File GT '{fname}' không phải JSON hợp lệ: {e}")
            sys.exit(1)

        is_valid, err = validate_entities(gt_data, fname)
        if not is_valid:
            print(f"[ERROR] File GT '{fname}' sai cấu trúc: {err}")
            sys.exit(1)

        if not os.path.exists(pred_path):
            print(f"[WARNING] Thiếu file dự đoán: {pred_path} — gán prediction = [] (điểm 0)")
            missing_preds.append(fname)
            pred_data = []
        else:
            try:
                pred_data = load_json(pred_path)
            except json.JSONDecodeError as e:
                print(f"[WARNING] File dự đoán '{fname}' không phải JSON hợp lệ ({e}) — gán prediction = [] (điểm 0)")
                invalid_preds.append(fname)
                pred_data = []
            else:
                is_valid, err = validate_entities(pred_data, fname)
                if not is_valid:
                    print(f"[WARNING] File dự đoán '{fname}' sai cấu trúc ({err}) — gán prediction = [] (điểm 0)")
                    invalid_preds.append(fname)
                    pred_data = []

        gt_list.append(gt_data)
        pred_list.append(pred_data)

    # Chạy metrics
    print(f"\n{'='*60}")
    print(f"  Evaluation: {len(gt_list)} mẫu")
    print(f"  GT dir       : {gt_dir}")
    print(f"  Pred dir     : {pred_dir}")
    print(f"  IoU threshold: {iou_threshold}  (giả định ghép cặp GT-Pred, xem metrics.py)")
    if missing_preds:
        print(f"  [WARNING] Thiếu {len(missing_preds)} file dự đoán: {missing_preds}")
    if invalid_preds:
        print(f"  [WARNING] {len(invalid_preds)} file dự đoán sai cấu trúc (đã gán điểm 0): {invalid_preds}")
    print(f"{'='*60}\n")

    text_s, assert_s, cand_s, final_s = metrics.evaluate_dataset(
        gt_list, pred_list, verbose=verbose, iou_threshold=iou_threshold
    )

    print(f"\n{'='*60}")
    print(f"  text_score       (WER-based)      = {text_s:.4f}  (trọng số 30%)")
    print(f"  assertions_score (Jaccard)         = {assert_s:.4f}  (trọng số 30%)")
    print(f"  candidates_score (Weighted Jaccard)= {cand_s:.4f}  (trọng số 40%)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  FINAL SCORE                        = {final_s:.4f}")
    print(f"{'='*60}\n")

    return text_s, assert_s, cand_s, final_s

def run_sweep(gt_dir: str, pred_dir: str, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]):
    """
    Chạy sweep qua nhiều ngưỡng IoU khác nhau và in ra bảng so sánh
    """
    gt_dir = os.path.abspath(gt_dir)
    pred_dir = os.path.abspath(pred_dir)

    print(f"\n{'='*70}")
    print(f"  SWEEPING IoU THRESHOLDS")
    print(f"  GT dir  : {gt_dir}")
    print(f"  Pred dir: {pred_dir}")
    print(f"{'='*70}")

    def _sort_key(fname):
        stem = os.path.splitext(fname)[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    gt_files = sorted(
        [f for f in os.listdir(gt_dir) if f.endswith(".json")],
        key=_sort_key,
    )

    if not gt_files:
        print(f"[ERROR] Không tìm thấy file JSON nào trong thư mục GT: {gt_dir}")
        sys.exit(1)

    gt_list = []
    pred_list = []

    for fname in gt_files:
        gt_path = os.path.join(gt_dir, fname)
        pred_path = os.path.join(pred_dir, fname)
        try:
            gt_data = load_json(gt_path)
            is_valid, err = validate_entities(gt_data, fname)
            if not is_valid:
                print(f"[ERROR] File GT '{fname}' sai cấu trúc: {err}")
                sys.exit(1)
        except Exception as e:
            print(f"[ERROR] Lỗi đọc file GT '{fname}': {e}")
            sys.exit(1)

        if not os.path.exists(pred_path):
            pred_data = []
        else:
            try:
                pred_data = load_json(pred_path)
                is_valid, err = validate_entities(pred_data, fname)
                if not is_valid:
                    pred_data = []
            except Exception:
                pred_data = []

        gt_list.append(gt_data)
        pred_list.append(pred_data)

    print(f"Loaded {len(gt_list)} samples. Running evaluations...")
    print(f"\n| IoU Thresh | Text Score | Assert Score | Candidate Score | FINAL SCORE |")
    print(f"| :---: | :---: | :---: | :---: | :---: |")

    for th in thresholds:
        t_s, a_s, c_s, f_s = metrics.evaluate_dataset(gt_list, pred_list, verbose=False, iou_threshold=th)
        print(f"| {th:.2f} | {t_s:.4f} | {a_s:.4f} | {c_s:.4f} | **{f_s:.4f}** |")
    print(f"{'='*70}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AI Race 2026 Bài 2 — so sánh output/ với groundtruth/"
    )
    parser.add_argument("--gt", default="data/dev/", help="Đường dẫn thư mục chứa file JSON ground truth (mặc định: data/dev/)")
    parser.add_argument("--pred", default="data/output/", help="Đường dẫn thư mục chứa file JSON dự đoán (mặc định: data/output/)")
    parser.add_argument("--verbose", action="store_true", help="In chi tiết từng mẫu")
    parser.add_argument("--file", default=None, help="Chỉ đánh giá một file cụ thể (vd: --file 11)")
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=metrics.DEFAULT_IOU_THRESHOLD,
        help="Ngưỡng IoU vị trí dùng để ghép cặp khái niệm GT-Pred",
    )
    parser.add_argument("--sweep", action="store_true", help="Chạy quét qua nhiều ngưỡng IoU khác nhau (0.3 -> 0.8)")
    args = parser.parse_args()

    if args.sweep:
        run_sweep(args.gt, args.pred)
    else:
        run_evaluation(
            args.gt,
            args.pred,
            verbose=args.verbose,
            single_file=args.file,
            iou_threshold=args.iou_threshold,
        )

if __name__ == "__main__":
    main()
