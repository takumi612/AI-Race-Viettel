# PROJECT STATE

## Goal

Xây dựng lab Clinical NLP end-to-end chạy được trên Colab/Kaggle, bao phủ NER, assertion/context, ICD-10/RxNorm linking, relation baseline, evaluation, artifact reload và submission generation.

## Current stage

**Tất cả 9 giai đoạn: COMPLETED (2026-07-19).**

## Completion evidence

- Audit dữ liệu: `reports/PHASE1_DATA_AUDIT.md`, `reports/phase1_audit.json`.
- Knowledge bases: `reports/kb_build_report.json` và `artifacts/*`.
- Stage reports: `reports/stage_03_eda.json` đến `reports/stage_08_integration.json`.
- Notebook: `medical_information_extraction_lab.ipynb` chạy tuần tự thành công, 32 code cells, không lỗi.
- Test suite: 27/27 tests pass bằng `python -m unittest discover -s tests -p 'test_*.py' -v`.
- Inference: 100 input documents, 100 JSON output, 842 submission entities, `output.zip` hợp lệ 100 members, CRC/structure pass.
- Official repository validator: 0 validation errors trên 100 output files.
- Offset invariant: 0 lỗi; reload equivalence: pass; private-test fitting: false.

## Completed

- Đọc yêu cầu trong `Lab Clinical NLP End-to-End.docx` thành project contract.
- Kiểm kê dữ liệu thật: `input.zip` có 100 UTF-8 documents, ID 1-100, không có annotation.
- Kiểm tra ICD-10: 12,219 rows, 12,137 canonical candidates; giữ metadata marker dagger/asterisk.
- Quét streaming RxNorm: 1,202,603 RXNCONSO rows, 7,423,180 RXNREL rows; tạo 56,053 candidates và 490,074 selected relations.
- Hoàn tất loader/validator/EDA/sectioning, raw-offset mapping, BIO/chunking/reconstruction và overlap resolution.
- Có dictionary/rule NER fallback: 845 internal entities, không có offset error.
- Có assertion hybrid với polarity, temporality, certainty, experiencer.
- Có ICD-10/RxNorm type-routed linking, medication parser và weighted lexical reranking.
- Có relation baseline theo sentence/type/distance; relation chỉ nằm trong diagnostics vì official submission schema không có relation field.
- Có evaluator strict/relaxed, artifact save/load, pipeline integration và output ZIP validation.
- Tạo README tổng, README riêng cho từng stage, requirements, SPEC, DECISIONS, manifest và notebook duy nhất.

## In progress

Không còn công việc dang dở trong phạm vi prompt.

## Next stage

Không có. Chỉ cần bổ sung annotation train/validation thật để bật supervised XLM-R và chấm competition score.

## Chosen architecture

- Adapter-driven data layer; giữ raw text bất biến; position end-exclusive.
- Rule-based sectioning bằng character ranges.
- XLM-RoBERTa-base + BIO/sliding window là supervised path có guard; hiện fallback dictionary/rule do không có annotation.
- Assertion hybrid: clinical rules + optional multi-task XLM-R heads.
- ICD-10 và RxNorm dùng linker/index tách biệt.
- Retrieval: exact + fuzzy + character n-gram; semantic reranker là lớp tùy chọn có benchmark.
- Relation baseline rule-based; classifier chỉ dùng khi có labels.
- Artifact-first inference; không train lại khi tạo submission.
- Official schema adapter theo từng entity type; mapping được xác nhận từ validator của repository.

## Available data

- `input.zip`: 100 documents UTF-8, ID 1-100, không có train/validation annotation.
- `ICD10.xlsx`: sheet `ICD10` có 12,219 records, 12,137 canonical candidates.
- `RxNorm_full_07062026.zip`: 1,202,603 RXNCONSO rows và 7,423,180 RXNREL rows.

## Labels and schema status

Official entity labels và assertions đã được xác nhận từ `src/validation/submission.py`: `CHẨN_ĐOÁN`, `THUỐC`, `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`; assertions gồm `isNegated`, `isHistorical`, `isFamily`. Schema key thay đổi theo type. Relation không có trong submission schema và vẫn chỉ lưu tại `diagnostics/`.

## Created files and artifacts

- Core package: `clinical_nlp_lab/` (config, schema, text, data, KB, NER, linking, assertions, relations, evaluation, training, artifacts, pipeline).
- Tools: `tools/build_knowledge_bases.py`, `tools/run_stage*.py`, `tools/build_notebook.py`, `tools/execute_notebook.py`, `tools/update_manifest.py` và audit tool.
- Tests: `tests/test_core.py`, `tests/test_kb_and_pipeline.py`.
- Docs: `README.md`, `stages/stage_01/README.md` ... `stages/stage_09/README.md`, `SPEC.md`, `DECISIONS.md`.
- Notebook: `medical_information_extraction_lab.ipynb`.
- Runtime artifacts: ICD-10/RxNorm JSONL.GZ caches, metadata/checksums, mappings, thresholds, model status và KB build report.
- Outputs: `output/1.json` ... `output/100.json`, `diagnostics/`, `run_summary.json`, `integration_report.json`, `output.zip`.
- Machine inventory: `ARTIFACT_MANIFEST.json`.

## Stage checkpoints

1. Stage 1 - contract, data audit và decision log: completed.
2. Stage 2 - ICD-10/RxNorm preprocessing, streaming build, checksums, save/load: completed.
3. Stage 3 - loader, validator, EDA, sectioning, split và baseline: completed.
4. Stage 4 - BIO/alignment/chunking, reconstruction, refinement, guarded trainer: completed.
5. Stage 5 - assertion/context features, hybrid predictor, official assertion mapping: completed.
6. Stage 6 - type-routed linking, medication parser, lexical reranking: completed.
7. Stage 7 - relation baseline, negative-sampling interface, diagnostics guard: completed.
8. Stage 8 - integration, evaluator, output ZIP, validation, reload: completed.
9. Stage 9 - notebook, README từng stage, requirements, manifest, final verification: completed.

## Known issues and assumptions

1. Thiếu train/validation annotation và ontology candidate cho symptom/lab; vì vậy supervised score và official competition score chưa thể tính.
2. `ICD10.xlsx` có duplicate `J13` và một English label thiếu ở `A97`; loader giữ evidence và deterministic deduplication.
3. 858 ICD-10 codes có dagger/asterisk markers; canonical code tách marker nhưng metadata vẫn bảo tồn.
4. Giả định `input.zip` là unlabeled inference/private-test candidate; không fit dictionary, threshold hay model trên đó.
5. Relation không đưa vào submission vì official schema hiện không có relation field.

## Critical invariants

- Giữ nguyên `raw_text`; `raw_text[start:end]` phải bằng entity text.
- Disease route vào ICD-10; drug route vào RxNorm RXCUI; không route sai ontology.
- Assertion xét polarity, temporality, certainty, experiencer.
- Không thêm field ngoài schema chính thức; không gọi external API; không hard-code output.
- Không fit trên private test hoặc dùng validation annotation để tạo training dictionary.
- Notebook chạy từ trên xuống dưới; inference reload artifacts mà không train lại.
- `output.zip` phải có đúng `output/<id>.json` và parse được.
