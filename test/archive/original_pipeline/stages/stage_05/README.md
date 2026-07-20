# Giai đoạn 5 - Clinical Context và Assertion

## Trạng thái

Hoàn thành implementation và rule validation; supervised multi-task training/threshold tuning bị khóa do thiếu assertion annotations.

## Đã hoàn thành

- Assertion axes nội bộ: polarity, temporality, certainty, experiencer.
- Rule cues cho phủ định, tiền sử, planned/resolved, uncertainty và experiencer family.
- Section feature và context window quanh entity.
- Official assertion mapping: phủ định, tiền sử và người nhà map sang ba label được repository validator cho phép.
- Assertion dataset builder và multi-task XLM-R model factory trong `training.py`.

## Bằng chứng

- `reports/stage_05_clinical_context.json`.
- Rule predictor chạy trên toàn bộ internal baseline entities; offset errors = 0.
- `artifacts/assertion_mapping.json` ghi trạng thái `CONFIRMED_FROM_REPOSITORY_VALIDATOR`.
- Không tune threshold hoặc báo cáo assertion Jaccard/macro-F1 khi không có gold labels.

## Lựa chọn tối ưu

Dùng hybrid rules làm active fallback vì các cue phủ định/tiền sử/người nhà có thể kiểm chứng trực tiếp; multi-task XLM-R vẫn sẵn sàng bật khi annotation đủ.

## File chính

- `clinical_nlp_lab/assertions.py`
- `clinical_nlp_lab/training.py`
- `tools/run_stage5_context.py`
