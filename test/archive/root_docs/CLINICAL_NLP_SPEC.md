# SPEC - Clinical NLP End-to-End Lab

Phiên bản: 1.0  
Ngày chốt: 2026-07-19  
Trạng thái: yêu cầu bất biến của dự án

## 1. Mục tiêu

Xây dựng một notebook duy nhất `medical_information_extraction_lab.ipynb` có thể chạy tuần tự trên Google Colab hoặc Kaggle, trong giới hạn GPU phổ thông 12-16 GB VRAM, không gọi API thương mại hoặc API y tế trực tuyến. Hệ thống chuyển văn bản lâm sàng tự do thành entity JSON có span, type, assertions và candidate chuẩn hóa; đồng thời có baseline relation extraction nội bộ.

Deliverables cuối dự án:

- `medical_information_extraction_lab.ipynb`
- `README.md`
- `requirements.txt`
- `SPEC.md`
- `PROJECT_STATE.md`
- `DECISIONS.md`
- `ARTIFACT_MANIFEST.json`
- `output.zip` theo đúng cấu trúc khi chạy inference

## 2. Schema submission

Mỗi tài liệu đầu vào tạo một JSON array. Mỗi entity chỉ có các key chính thức sau, trừ khi ban tổ chức cung cấp schema khác:

```json
{
  "text": "span nguyên văn",
  "type": "official_label",
  "candidates": ["standard_code"],
  "assertions": ["official_assertion"],
  "position": [0, 12]
}
```

Ràng buộc:

- `position` tạm hiểu là `[start, end)` theo Python slicing.
- `raw_text[start:end] == entity["text"]` phải luôn đúng.
- Không thêm key ngoài schema.
- Nếu không có entity, output là `[]`.
- Relation không được thêm vào submission nếu schema chính thức không có trường relation.
- ZIP phải có `output/<document_id>.json`, không có lớp `output/output/`.

## 3. Nhiệm vụ nghiệp vụ

- Clinical named entity recognition và type classification.
- Boundary/attribute refinement, gồm thuốc và giá trị xét nghiệm.
- Assertion/context theo bốn trục nội bộ: polarity, temporality, certainty, experiencer.
- Disease linking sang ICD-10.
- Drug linking sang RxNorm RXCUI.
- Candidate linking cho các type khác chỉ theo ontology/schema được xác nhận.
- Clinical relation extraction ít nhất ở mức baseline.
- Conversion về official schema và submission validation.

## 4. Metric

Metric mục tiêu:

```text
final_score = 0.3 * text_score
            + 0.3 * assertions_score
            + 0.4 * candidates_score
```

- `text_score`: dựa trên `1 - WER`.
- `assertions_score`: Jaccard.
- `candidates_score`: Jaccard có trọng số theo số candidate ground truth.
- Do quy tắc entity matching và WER chưa được cung cấp, dự án phải có cả strict evaluator và approximate evaluator, đồng thời ghi rõ giả định.

## 5. Pipeline bất biến

```text
Raw clinical document
-> document validation
-> section/sentence segmentation
-> entity detection
-> character span reconstruction
-> boundary/attribute refinement
-> assertion/context prediction
-> type-routed entity linking
-> relation extraction
-> conflict resolution
-> official schema conversion
-> submission validation
-> output.zip
```

Mỗi module phải nêu input, output, phương pháp, công nghệ, code, evaluation, save/load artifact và unit test nhỏ. Module phải có thể thay thế độc lập.

## 6. Công nghệ mặc định

- Data/validation: `pathlib`, `json`, `zipfile`, `pandas` hoặc `polars`, `pydantic` hoặc validator tùy chỉnh.
- Sectioning: regex, heading dictionary, newline-aware splitter và character-range mapping.
- NER: `xlm-roberta-base`, Hugging Face Transformers/Datasets, PyTorch, BIO, fast-tokenizer offset mapping, sliding window.
- Assertion: XLM-R shared encoder + multi-head classifier, kết hợp rule cues khi cần.
- ICD-10: bilingual exact lookup, RapidFuzz, character TF-IDF, optional multilingual embeddings/FAISS và reranker.
- RxNorm: streaming RRF parser, medication attribute parser, exact/fuzzy/character/semantic retrieval, optional relation-aware reranker.
- Relation: rule baseline; XLM-R entity-pair classifier chỉ khi có nhãn.
- Evaluation: `jiwer`, scikit-learn metrics và custom matching/Jaccard.

Mọi thay đổi khỏi mặc định phải có bằng chứng validation hoặc ràng buộc tài nguyên và được ghi trong `DECISIONS.md`.

## 7. Dữ liệu đã xác nhận ở giai đoạn 1

- `input.zip`: 100 file text UTF-8, ID 1-100, không có annotation.
- `ICD10.xlsx`: sheet `ICD10`, header hàng 3, 12,219 bản ghi.
- `RxNorm_full_07062026.zip`: RxNorm release 2026-07-06; có `rrf/RXNCONSO.RRF` và `rrf/RXNREL.RRF`.
- Không có train/validation ground truth trong workspace tại thời điểm chốt SPEC.

Chi tiết kiểm kê nằm trong `reports/PHASE1_DATA_AUDIT.md` và `reports/phase1_audit.json`.

## 8. Data leakage và split

- Chia train/validation theo document trước mọi chunking hoặc feature fitting.
- Không để chunk cùng document xuất hiện ở cả train và validation.
- Không đưa validation annotation vào training dictionary.
- Không fit TF-IDF, embeddings, aliases, ranker hoặc thresholds trên private test.
- Ontology ICD-10/RxNorm được phép index cho inference; supervised artifacts học từ annotation phải chỉ dùng training fold.
- `input.zip` hiện tại chỉ dùng cho kiểm tra I/O và inference plumbing, không dùng để học tham số.

## 9. Offset và Unicode

- Luôn giữ riêng `raw_text` và `model_text`.
- Không gọi `strip`, `lower`, whitespace replacement hoặc Unicode normalization trực tiếp trên raw text rồi tái sử dụng offset.
- Nếu model dùng normalized text, phải có mapping hai chiều raw index và normalized index.
- Mọi entity cuối phải được slice trực tiếp từ `raw_text`.

## 10. Artifacts và reproducibility

- Cố định seed cho Python, NumPy và PyTorch.
- Lưu config, library versions, label mappings, thresholds, tokenizer, model, indexes, metrics và decision log.
- Inference phải load hoàn toàn từ `artifacts/` mà không train lại.
- Sau save phải xóa object khỏi RAM, reload và xác nhận output trước/sau tương đương.
- Artifact manifest phải ghi path, producer, consumer, size, inference requirement, rebuild method và checksum khi phù hợp.

## 11. Notebook contract

- Có config tập trung và `FAST_DEV_RUN`.
- Có train, validation, inference và submission generation khi ground truth tồn tại.
- Không hard-code output, label hoặc đường dẫn máy cá nhân.
- Notebook hợp lệ theo `nbformat.validate` và không chứa API key/token.
- Các cell theo dependency order và chạy tuần tự từ trên xuống dưới.

## 12. Điều chưa được phép tự bịa

- Official entity/assertion/relation label sets.
- Ontology cho triệu chứng/xét nghiệm.
- Candidate requirement theo type.
- Entity matching/WER chính thức.
- Relation field trong submission.
- Kết quả validation hoặc benchmark khi không có annotation.

Các mục này phải là config/adapter có thể cập nhật sau organizer confirmation.

## 13. Tiêu chí hoàn thành

Dự án chỉ hoàn thành khi đủ deliverables, notebook chạy tuần tự, có toàn bộ pipeline yêu cầu, tạo đúng `output.zip`, reload inference artifacts thành công và không vi phạm critical invariants.

