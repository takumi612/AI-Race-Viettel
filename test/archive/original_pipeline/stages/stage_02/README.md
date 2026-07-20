# Giai đoạn 2 - Knowledge-base Preprocessing

## Trạng thái

Hoàn thành. Artifacts được build trực tiếp từ hai nguồn ontology thật và đã kiểm tra save/load.

## Đã hoàn thành

- ICD-10 loader tìm header theo tên, tách canonical code khỏi marker `†/*`, giữ song ngữ và provenance.
- RxNorm streaming parser đọc trực tiếp `rrf/RXNCONSO.RRF` trong ZIP với filter cấu hình.
- RXNREL streaming parser chỉ giữ bốn relation cần thiết từ nguồn `RXNORM`.
- Cache deterministic `jsonl.gz`, metadata, SHA-256 và build report.
- Runtime mappings/config/threshold/model-status artifacts.

## Bằng chứng định lượng

| Artifact | Records | Kích thước | Kiểm tra |
|---|---:|---:|---|
| ICD-10 dictionary | 12.137 | 630.718 byte | 12.137 unique IDs |
| RxNorm dictionary | 56.053 | 1.444.746 byte | 56.053 unique RXCUI |
| RxNorm relation cache | 490.074 | 2.123.808 byte | Stream parse thành công |

- Build toàn bộ mất 41,45 giây trong runtime hiện tại.
- Load ICD-10: 0,171 giây.
- Load RxNorm: 0,465 giây.
- Stream relation cache: 1,45 giây.
- Source SHA-256 khớp audit giai đoạn 1.
- `artifacts/kb_build_report.json` xác nhận `save_load_passed=true`.

## Lựa chọn tối ưu

Chọn deterministic compressed JSONL thay Parquet/SQLite ở baseline vì runtime hiện tại không có `pyarrow`, trong khi cache chỉ khoảng 4,2 MB, load nhanh và không cần dependency bổ sung. Giao diện iterator cho phép đổi backend sau này mà không đổi pipeline.

## File chính

- `clinical_nlp_lab/kb.py`
- `tools/build_knowledge_bases.py`
- `artifacts/icd10/`
- `artifacts/rxnorm/`

