# Giai đoạn 7 - Relation Extraction

## Trạng thái

Hoàn thành rule baseline và diagnostics; classifier supervised chưa chạy vì relation labels không tồn tại.

## Đã hoàn thành

- Entity-pair generation có cùng câu, type compatibility và khoảng cách tối đa.
- Rule relations cho drug-condition, condition-symptom và lab-condition/symptom.
- Interface negative sampling và entity-pair classifier examples.
- Diagnostics relation output riêng, không phá JSON submission.

## Bằng chứng

- `reports/stage_07_relations.json`.
- Rule extractor chạy trên 100 input và ghi số cặp/quan hệ thực tế.
- Unit test `DRUG_TREATS_CONDITION` pass.
- `artifacts/relation_mapping.json` xác nhận `submission_enabled=false`.

## Lựa chọn tối ưu

Rule baseline là lựa chọn an toàn khi không có relation labels; không hard-code các relation ví dụ thành official submission labels.

## File chính

- `clinical_nlp_lab/relations.py`
- `clinical_nlp_lab/training.py`
- `tools/run_stage7_relations.py`

