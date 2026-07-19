# Giai đoạn 8 - Integration, Evaluation và Submission

## Trạng thái

Hoàn thành. End-to-end inference đã chạy trên toàn bộ 100 input và tạo `output.zip` đúng cấu trúc.

## Đã hoàn thành

- Integrated `run_inference(...)` pipeline.
- Sectioning, dictionary/rule NER, boundary refinement, assertion axes, ICD-10/RxNorm routing, relation diagnostics.
- Official schema conversion chỉ dùng năm key; entity type chưa xác nhận bị drop có log.
- JSON validation, filename validation, UTF-8, NaN check và ZIP CRC/structure validation.
- Save/load artifact reload và deterministic equivalence test.
- Strict/approximate evaluator interface; không báo score khi thiếu gold.

## Bằng chứng

- `reports/stage_08_integration.json`.
- 100/100 JSON parse được.
- 100/100 output members trong `output/1.json` ... `output/100.json`.
- Nested `output/output/` không tồn tại.
- Offset errors = 0.
- Reload document `1` cho output nội bộ tương đương trước/sau reload.
- `output.zip` CRC hợp lệ.

## Giải thích submission rỗng

`artifacts/entity_type_mapping.json` đang ở trạng thái `UNCONFIRMED_NO_TRAIN_ANNOTATIONS`, nên 845 internal entities được giữ trong diagnostics nhưng không bị gán official type đoán mò. Vì vậy 100 JSON submission là `[]`, hợp lệ schema nhưng không phải performance claim.

## File chính

- `clinical_nlp_lab/pipeline.py`
- `tools/run_pipeline.py`
- `tools/run_stage8_integration.py`
- `output.zip`

