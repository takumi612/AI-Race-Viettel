# Nhật ký thực nghiệm (Experiment Log) — AI Race 2026

Bảng này ghi nhận chi tiết lịch sử thử nghiệm, các thông số cấu hình và so sánh đối chiếu giữa điểm tự chấm cục bộ (`metrics.py`) với điểm Leaderboard chính thức của BTC.

## Bảng theo dõi thực nghiệm

| Phiên bản | NER Model | Retrieval Config | Reranker / Threshold (ε) | Điểm tự chấm (`metrics.py`) | Điểm Leaderboard thật | Chênh lệch (Local - LB) | Ghi chú & Bài học kinh nghiệm |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **v0.5** | LLM Few-shot (Qwen) | BM25s (Chỉ ICD-10) | Top-1 BM25 | *Text:* <br> *Assertion:* <br> *Candidate:* <br> **Final:** | | | Phiên bản baseline đầu tiên để test tích hợp, định dạng JSON và đo rate limit cooldown. |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

## Hướng dẫn sử dụng:
1. **Bắt buộc** ghi nhận dòng mới ngay sau khi nhận kết quả từ Leaderboard của BTC.
2. Cột **Chênh lệch** là chìa khóa để hiệu chỉnh ngưỡng IoU và thuật toán WER của `metrics.py` (Mục 3.6 của Kế hoạch).
3. Đính kèm các phát hiện lỗi chi tiết trong cột **Ghi chú & Bài học kinh nghiệm**.
