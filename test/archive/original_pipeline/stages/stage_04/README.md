# Giai đoạn 4 - Entity Extraction

## Trạng thái

Hoàn thành phần triển khai và kiểm thử interface; supervised training được khóa có chủ đích vì không có annotated train/validation data.

## Đã hoàn thành

- BIO label-map builder và character-span -> BIO conversion.
- Fast-tokenizer contract với `return_offsets_mapping=True` và `return_overflowing_tokens=True`.
- Document-level split trước chunking, không split theo chunk.
- Sliding-window overlap (`max_length=512`, `stride=128`).
- BIO prediction -> raw character span reconstruction.
- Chunk merge, overlap resolution và boundary refinement.
- Optional XLM-R `Trainer` path khi `torch`/`transformers` và annotations tồn tại.

## Bằng chứng

- `reports/stage_04_entity_extraction.json`.
- Unit tests BIO, reconstruction, sliding windows, overlap và offset invariant đều pass.
- Workspace có 0 annotated documents; training result ghi rõ `trained=false` và không phát sinh metric giả.
- Fallback detector giai đoạn 3 đã xác nhận 845 spans, 0 offset errors trên 100 input.

## Lựa chọn tối ưu

Giữ XLM-RoBERTa-base + BIO làm supervised default theo contract, nhưng active inference dùng dictionary/rule detector cho tới khi có nhãn. Đây là lựa chọn duy nhất vừa giữ code train thật vừa không tạo dữ liệu/score giả.

## File chính

- `clinical_nlp_lab/ner.py`
- `clinical_nlp_lab/training.py`
- `tools/run_stage4_ner.py`

