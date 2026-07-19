# Giai đoạn 3 - Data và Baseline

## Trạng thái

Hoàn thành. Data loader, validator, EDA, section detector và dictionary/rule baseline đã chạy trên 100 input thật.

## Đã hoàn thành

- Loader hỗ trợ ZIP hoặc thư mục text, giữ nguyên raw text và sắp xếp document ID tự nhiên.
- Validator kiểm tra offset, duplicate/overlap, type/assertion/candidate distribution và dataset fingerprint.
- Section detector rule/regex tạo `section_name/start/end/text` và kiểm tra slice raw text.
- Baseline dictionary + generic clinical rules chạy offline, không fit trên private input.
- EDA report phân biệt rõ `not_scored` khi thiếu ground truth.

## Bằng chứng

- `reports/stage_03_eda.json`.
- 100 documents loaded; input validator không có lỗi.
- Train annotation documents found: 0; annotated validator không tạo metric giả.
- Mọi baseline span được kiểm tra `raw_text[start:end] == text`.
- Unit tests section offsets, repeated mentions và bad-offset validator đều pass.

## Lựa chọn tối ưu

Dùng rule/dictionary baseline làm fallback vì không có annotation; không tạo pseudo-label và không dùng private input để fit TF-IDF/alias/threshold.

## File chính

- `clinical_nlp_lab/data.py`
- `clinical_nlp_lab/text.py`
- `clinical_nlp_lab/ner.py`
- `tools/run_stage3_baseline.py`

