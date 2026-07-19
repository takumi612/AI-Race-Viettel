# Giai đoạn 6 - Entity Linking

## Trạng thái

Hoàn thành baseline linking và reranking offline trên ICD-10/RxNorm artifacts.

## Đã hoàn thành

- Type routing: DISEASE -> ICD-10, DRUG -> RxNorm; không dùng chung candidate index.
- Exact alias lookup, fuzzy SequenceMatcher, character n-gram retrieval.
- Weighted lexical reranking và threshold/output-k config.
- Medication attribute parser: drug name, strength, route, frequency, dose form.
- Candidate IDs giữ nguyên ontology IDs; không trả candidate cho type chưa có ontology.
- Semantic embedding/cross-encoder là optional path, không bắt buộc ở baseline.

## Bằng chứng

- `reports/stage_06_entity_linking.json`.
- Ontology self-lookup sanity trên 1.000 aliases, top-1 recall được ghi rõ là sanity check, không phải competition score.
- 100 input được route và validate offsets; không fit trên private input.
- Candidate sizes: 12.137 ICD-10 và 56.053 RxNorm.

## Lựa chọn tối ưu

Dùng lexical + character reranking làm baseline vì chạy offline, không cần thêm model/dependency và phù hợp RAM/VRAM; embedding/reranker chỉ bật sau khi có validation evidence.

## File chính

- `clinical_nlp_lab/linking.py`
- `artifacts/icd10/`
- `artifacts/rxnorm/`
- `tools/run_stage6_linking.py`

