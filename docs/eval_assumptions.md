# Eval Assumptions — Giai đoạn 0 (Evaluation Engine)

Tài liệu này ghi lại tất cả các giả định kỹ thuật đã chọn cho script `metrics.py`.
Mọi thay đổi về thuật toán đánh giá **phải** được cập nhật tại đây trước khi áp dụng.

---

## 1. Cách tính `position` (Kết luận đã xác thực)

**Quy tắc:** `position = [start, end]` được tính theo **ký tự Unicode codepoint** (không phải byte UTF-8).

```python
# Chuẩn:
start = text.find(entity_text)          # 0-indexed, đếm ký tự
end   = start + len(entity_text)        # vị trí ngay SAU ký tự cuối (Python slicing)

# Xác nhận:
assert text[start:end] == entity_text   # luôn đúng
```

**Lưu ý đã kiểm tra:**
- Dấu cách ` `, xuống dòng `\n`, tab `\t` đều được đếm vào index bình thường.
- File input dùng **LF only** (không phải CRLF) — mỗi `\n` = 1 ký tự.
- Ký tự tiếng Việt tổ hợp (như `ề`, `ổ`, `ượ`) = 1 codepoint trong UTF-8 NFC normalized.
- Đề bài ví dụ `amlodipine 10 mg po daily` tại `[58, 83]`: xác nhận `len = 25 = end - start` ✓

---

## 2. Thuật toán Ghép cặp (Entity Matching)

**Lựa chọn:** Greedy Bipartite Matching dựa trên IoU vị trí + cùng `type`.

**Quy trình:**
1. Với mọi cặp (GT, Pred) trong cùng sample: tính IoU trên khoảng `[start, end]`.
2. Lọc các cặp có `IoU ≥ 0.5` **và** cùng `type`.
3. Sắp xếp danh sách cặp theo `IoU` giảm dần.
4. Greedy assign: lấy cặp IoU cao nhất còn khả dụng (chưa bị gán).
5. Unmatched GT → FN (điểm 0). Unmatched Pred → FP (điểm 0).

**Lý do chọn ngưỡng IoU = 0.5:**
- Ngưỡng tiêu chuẩn trong NER evaluation.
- Chưa được xác nhận bằng leaderboard thật — **cần kiểm chứng ở bước 3.6**.
- Giả định có thể cần điều chỉnh về 0.3 hoặc 0.7 tùy kết quả leaderboard.

**Quy tắc type mismatch (từ đề bài):**
> *"đoán đúng text nhưng sai type thì khái niệm bị tính 2 lần, mỗi lần 0 điểm"*

→ Overlap vị trí nhưng khác `type` = **không ghép cặp** → cả 2 bị phạt (FP + FN).

---

## 3. Tính `text_score` qua WER

**Công thức:**
```
text_score = Σ(1 - WER(i)) / N_samples
```
Với mỗi sample i:
```
score_text(i) = Σ max(0, 1 - WER(matched_pair)) / (N_gt + N_pred - N_matched)
```

**WER** = Levenshtein distance ở cấp độ **từ** (split by whitespace), **case-insensitive**.

**Xử lý FP/FN:** Mỗi thực thể không match đóng góp `WER = 1.0` → điểm 0 vào mẫu số.

---

## 4. Tính `assertions_score`

**Áp dụng với:** `CHẨN_ĐOÁN`, `THUỐC`, `TRIỆU_CHỨNG`.  
**Không áp dụng:** `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`.

**Cách tính Jaccard cho từng cặp đã match:**
```
J(gt_assertions, pred_assertions)
```

**Tính điểm trung bình mẫu i:**
```
assertions_score(i) = Σ J(matched pair) / (N_assertion_matches + N_fn + N_fp)
```
Trong đó FN, FP chỉ đếm các thực thể thuộc loại có hỗ trợ assertions.

**Trường hợp biên Jaccard:**
- Cả 2 rỗng → J = 1.0 (thực thể không có assertion nào, đoán rỗng là đúng)
- Một bên rỗng → J = 0.0
- Giao/Hợp thông thường → |Giao| / |Hợp|

---

## 5. Tính `candidates_score`

**Áp dụng với:** `CHẨN_ĐOÁN`, `THUỐC`.

**Cách tính Jaccard cho từng cặp đã match:**
```
J(gt_candidates, pred_candidates)
```
*Mã candidate được chuẩn hóa: strip() + upper() trước khi so khớp.*

**Trọng số mẫu i:**
```
W(i) = Σ_k (len(gt_candidates(k)) + 1)
```
Tổng chạy qua tất cả thực thể `CHẨN_ĐOÁN`/`THUỐC` trong mẫu i (cả GT lẫn GT không match).

**Tính điểm toàn dataset:**
```
candidates_score = Σ_i [J_candidates(i) * W(i)] / Σ_i W(i)
```

---

## 6. Tính `final_score`

```
final_score = 0.3 * text_score + 0.3 * assertions_score + 0.4 * candidates_score
```

---

## 7. Kết quả kiểm thử Unit Test (`metrics.py test`)

| Test Case | Mô tả | Kết quả |
|:---|:---|:---:|
| 1 | Perfect match (GT == Pred) | ✅ Final = 1.0000 |
| 2 | WER > 0, IoU ≥ 0.5 (thiếu từ cuối) | ✅ Text = 0.6000 |
| 3 | Sai `type` (FP + FN đồng thời) | ✅ Final = 0.0000 |
| 4 | Hoán đổi mã candidates | ✅ Candidates = 0.0000 |
| 5 | Case-insensitive + whitespace strip | ✅ Final = 1.0000 |

---

## 8. Kết quả đánh giá 10 mẫu dev thực tế

Chạy `python evaluate.py --gt groundtruth/ --pred output/` trên 10 mẫu dev (Average Case):

```
text_score        = 0.9009
assertions_score  = 0.8476
candidates_score  = 0.7742
FINAL SCORE       = 0.8342
```

---

## 9. Mục tiếp theo cần thực hiện (Bước 3.6)

> ⚠️ **Bắt buộc khi có Baseline chạy được (Giai đoạn 0.5):**
>
> Nộp thử 1 lần lên Leaderboard → ghi nhận điểm thật → so sánh với điểm `evaluate.py` tự chấm trên đúng output đã nộp.
>
> **Nếu lệch nhiều (> 0.05):** Xem xét điều chỉnh:
> - Ngưỡng IoU (thử 0.3 và 0.7 thay vì 0.5)
> - Cách tính WER đa-concept trong 1 sample
> - Giả định về schema động của JSON output

---

## 10. Ghi chú về cách đếm position (phát hiện từ phân tích)

Văn bản mẫu đề bài `[58, 83]` cho `"amlodipine 10 mg po daily"`:
- Python `str.find()` tính ra `[56, 81]` với chuỗi rút gọn tay nhập
- Lệch 2 ký tự do văn bản gốc trong file mẫu có thêm ký tự đệm (BOM + `\n` hoặc thêm space)
- **Kết luận:** `end - start = len(entity_text) = 25` → cách đếm CHARACTERS (codepoints) là đúng
- **Mô hình tìm position thực tế:** cách tốt nhất là dùng `text.find(entity_text)` trên văn bản gốc sau khi NER trả về chuỗi văn bản. Nếu entity_text bị WER lệch nhẹ, dùng sliding window để tìm vị trí xấp xỉ tốt nhất.
