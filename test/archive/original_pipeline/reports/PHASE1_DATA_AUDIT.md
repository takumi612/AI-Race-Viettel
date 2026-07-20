# Giai đoạn 1 - Contract và Data Audit

Ngày audit: 2026-07-19  
Phạm vi: `input.zip`, `ICD10.xlsx`, `RxNorm_full_07062026.zip`  
Trạng thái huấn luyện: **chưa huấn luyện mô hình**

## 1. Kết luận điều hành

- Ba tài nguyên dự kiến đều tồn tại và đọc được.
- `input.zip` là một tập gồm đúng 100 tài liệu văn bản, đánh số liên tục từ `1.txt` đến `100.txt`. Không có annotation, train split hoặc validation split.
- Vì không có ground truth, chưa thể xác định nhãn entity, assertion, relation; chưa thể tạo train/validation split, benchmark mô hình, tune threshold hoặc báo cáo metric có ý nghĩa.
- `ICD10.xlsx` có sheet chính `ICD10`, đúng bốn header cốt lõi cần cho linker song ngữ. Dữ liệu có một dòng trùng, một tên tiếng Anh thiếu, và 858 mã mang ký hiệu dagger/asterisk cần tách khỏi mã chuẩn nhưng giữ làm metadata.
- Gói RxNorm là Full Update Release ngày 2026-07-06. Có đủ `rrf/RXNCONSO.RRF` và `rrf/RXNREL.RRF`; cả hai được quét trực tiếp trong ZIP bằng streaming, không giải nén toàn bộ và không nạp thành DataFrame lớn.
- Kiến trúc mặc định vẫn giữ XLM-RoBERTa-base + BIO cho NER, assertion hybrid, linker ICD-10/RxNorm tách biệt và relation rule-based. Các module supervised chỉ được kích hoạt khi có annotation hợp lệ.

## 2. Kiểm kê file

| File | Kích thước | SHA-256 |
|---|---:|---|
| `input.zip` | 85,874 byte | `46fe4a578b2c4478faa7c570b218218f539c0bbf1ea409168ae67a14ad86ca35` |
| `ICD10.xlsx` | 2,011,137 byte | `4737e36d65698de987b718485c4f2c7dab8c8b71fdd263854798c27ee89e528e` |
| `RxNorm_full_07062026.zip` | 259,313,098 byte | `53523ee9f1fcd7ee426698edf566aedebe548a6ec8cc372c41271fc5b28e784c` |

Không phát hiện đường dẫn ZIP traversal hoặc tên member trùng trong hai archive.

## 3. Audit `input.zip`

### 3.1 Cấu trúc và encoding

- 101 members: một thư mục `input/` và 100 file `.txt`.
- ID liên tục từ 1 đến 100; không thiếu và không trùng ID.
- CRC của ZIP hợp lệ.
- 100/100 file decode strict UTF-8 thành công; không có BOM.
- Tất cả file dùng LF; không có NUL hoặc control character bất thường.
- 100/100 văn bản ở Unicode NFC.
- Không có file rỗng, whitespace-only hoặc nội dung trùng hoàn toàn.

### 3.2 Thống kê tài liệu

| Chỉ tiêu | Min | P25 | Median | P75 | P95 | Max | Mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Byte | 182 | 797.0 | 1,582.5 | 2,204.5 | 3,832.3 | 5,702 | 1,709.48 |
| Ký tự | 136 | 611.75 | 1,222.5 | 1,741.0 | 2,942.7 | 4,428 | 1,323.36 |
| Dòng | 3 | 14.0 | 29.5 | 36.25 | 60.05 | 140 | 29.64 |

### 3.3 Schema và nhãn

Không tìm thấy file `.json`, annotation, manifest nhãn, train split hoặc validation split. Do đó:

- entity labels thực tế: chưa xác định;
- assertion labels thực tế: chưa xác định;
- relation labels thực tế: chưa xác định;
- ontology cho triệu chứng/xét nghiệm: chưa xác định;
- quy tắc entity matching và cách tính WER của ban tổ chức: chưa xác định.

`input.zip` được xem là **unlabeled inference/private-test candidate**. Nó chỉ được dùng để kiểm tra loader, encoding, offset, I/O và submission plumbing; không được dùng để fit TF-IDF, học alias, train model hoặc tune threshold.

## 4. Audit `ICD10.xlsx`

### 4.1 Cấu trúc workbook

- 14 sheets.
- Sheet chính: `ICD10`, 12,222 hàng x 22 cột.
- Header ở hàng 3; dữ liệu bắt đầu từ hàng 4.
- Có đủ: `MÃ BỆNH`, `MÃ BỆNH KHÔNG DẤU`, `DISEASE NAME`, `TÊN BỆNH`.

### 4.2 Chất lượng dữ liệu sheet `ICD10`

- 12,219 bản ghi không rỗng.
- 12,218 mã hiển thị duy nhất.
- 12,137 mã không dấu chấm duy nhất.
- Một dòng trùng chính xác: mã `J13` tại hàng 4238 và 4239.
- Một tên tiếng Anh thiếu: mã `A97` tại hàng 456; tên tiếng Việt là `Sốt xuất huyết Dengue`.
- 858 mã hiển thị có ký hiệu nghiệp vụ: 82 dagger (`†`) và 776 asterisk (`*`).
- Sau khi tách `†`/`*` khỏi mã và bỏ dấu chấm, không còn mismatch với cột `MÃ BỆNH KHÔNG DẤU`; mã chuẩn hóa đều khớp pattern ICD-10 đang dùng.
- Có 82 nhóm mã không dấu chấm trùng, mỗi nhóm có hai dòng; đây chủ yếu là biến thể mã có/không có marker, không nên xóa mù quáng.

### 4.3 Quy tắc loader được chốt

- Tìm cột theo header đã chuẩn hóa, không dùng chỉ số cột cố định.
- Giữ nguyên `display_code` có marker và tạo `canonical_code` bằng cách tách marker `†`/`*`.
- Giữ cả mã có dấu chấm và không dấu chấm làm aliases.
- Gộp tên Việt/Anh; cho phép thiếu một ngôn ngữ.
- Deduplicate có kiểm soát theo `canonical_code` + tên; ghi lại marker và provenance thay vì bỏ mất dòng.
- Không suy diễn tên tiếng Anh còn thiếu từ private test.

## 5. Audit `RxNorm_full_07062026.zip`

### 5.1 Release và archive

- Release: RxNorm Full Update Release ngày 2026-07-06.
- 44 members; tổng dung lượng giải nén 1,828,225,005 byte.
- Có cả full dataset và biến thể `prescribe/`.
- `rrf/RXNCONSO.RRF`: 131,620,308 byte chưa nén.
- `rrf/RXNREL.RRF`: 527,774,099 byte chưa nén.

### 5.2 `RXNCONSO.RRF`

- 1,202,603 dòng.
- 18 trường trên mọi dòng; không có dòng sai field count và không có lỗi UTF-8.
- Tất cả dòng có `LAT=ENG`.
- `SUPPRESS`: 807,980 `N`, 387,468 `O`, 7,155 `E`.
- Với filter mặc định `LAT=ENG`, `SAB=RXNORM`, `SUPPRESS=N`, TTY thuộc `{IN, PIN, MIN, BN, SCD, SBD, GPCK, BPCK, DF, DFG}`: 56,053 dòng và 56,053 RXCUI duy nhất.

| TTY sau filter | Số dòng |
|---|---:|
| SCD | 17,552 |
| IN | 14,648 |
| SBD | 9,696 |
| BN | 5,110 |
| MIN | 3,841 |
| PIN | 3,643 |
| BPCK | 740 |
| GPCK | 653 |
| DF | 126 |
| DFG | 44 |

### 5.3 `RXNREL.RRF`

- 7,423,180 dòng.
- 16 trường trên mọi dòng; không có dòng sai field count và không có lỗi UTF-8.
- Quan hệ cần cho medication reranking đều tồn tại:

| RELA | Số dòng |
|---|---:|
| `has_ingredient` | 355,165 |
| `has_dose_form` | 135,757 |
| `tradename_of` | 118,543 |
| `consists_of` | 116,818 |

### 5.4 Quy tắc preprocessing được chốt

- Đọc trực tiếp member trong ZIP bằng `zipfile` + streaming parser.
- Không giải nén toàn bộ archive và không nạp toàn bộ RRF vào RAM.
- Chỉ parse `RXNREL` khi build relation cache; baseline retrieval có thể chạy chỉ với `RXNCONSO`.
- Ghi cache Parquet hoặc SQLite theo chunk; lưu cấu hình filter và release date cùng artifact.
- Mọi alias/index học từ annotation phải tách khỏi ontology cache để ngăn leakage.

## 6. Kiến trúc đề xuất

1. **Data contract layer:** adapter cho unlabeled text và annotated JSON; validator bắt buộc giữ `raw_text` bất biến và kiểm tra offset end-exclusive.
2. **Segmentation:** heading dictionary + regex + newline-aware sentence splitter; mọi segment giữ start/end trên raw text.
3. **NER:** XLM-RoBERTa-base token classification, BIO, fast-tokenizer offsets, sliding window và document-level split. GlobalPointer/span model chỉ xét khi annotation chứng minh nested/overlap đáng kể.
4. **Boundary refinement:** confidence + regex y tế + type-specific rules; luôn tái kiểm tra slice raw text.
5. **Assertion/context:** rule-based cues + XLM-R multi-task heads cho polarity, temporality, certainty, experiencer; ánh xạ về official labels khi labels thật có sẵn.
6. **Entity linking:** ICD-10 và RxNorm dùng hai pipeline/index riêng. Exact/fuzzy/character retrieval là baseline; embedding/cross-encoder là lớp nâng cao có thể bật/tắt.
7. **Relation:** rule-based theo cùng câu, type compatibility và khoảng cách; lưu riêng trong diagnostics nếu submission schema không có relation.
8. **Evaluation:** strict matcher theo document + exact position + type và approximate matcher theo type + overlap + text similarity; không báo metric giả khi thiếu ground truth.
9. **Inference/submission:** chỉ load artifact đã lưu, validate đúng năm key chính thức, bảo đảm không có thư mục `output/output/`.

## 7. Baseline so với phương án nâng cao

| Module | Baseline mặc định | Nâng cao | Điều kiện chuyển |
|---|---|---|---|
| NER | XLM-R BIO | GlobalPointer/span NER | Annotation cho thấy nested/overlap và validation cải thiện |
| Assertion | Rule cues + XLM-R multi-task | Classifier/rules chuyên biệt theo type/section | Có đủ nhãn và macro-F1/Jaccard tốt hơn |
| ICD-10 | Exact + RapidFuzz + char TF-IDF | Multilingual embeddings + cross-encoder | Recall@k/reranking tăng trên validation |
| RxNorm | Parser thuốc + exact/fuzzy/char retrieval | Embedding + relation-aware reranker | Cải thiện candidate Jaccard/MRR trong giới hạn VRAM |
| Relation | Rule-based | XLM-R entity-pair classifier | Có relation labels và negative sampling đáng tin cậy |
| Post-processing | Deterministic confidence rules | Calibrated learned resolver | Có đủ validation data, không leakage |

## 8. Vấn đề chưa rõ cần organizer xác nhận

1. Train/validation annotations nằm ở đâu và schema chính xác là gì?
2. Danh sách official entity, assertion và relation labels?
3. Candidate bắt buộc cho type nào ngoài bệnh và thuốc?
4. Ontology dùng cho triệu chứng và kết quả xét nghiệm?
5. End offset có phải exclusive như invariant Python slice không?
6. Ban tổ chức ghép entity và tính WER chính xác thế nào?
7. Quan hệ có thuộc submission chính thức không?
8. JSON array order có ảnh hưởng metric không?
9. Giới hạn RAM, thời gian inference và chính sách download model trên Colab/Kaggle?
10. Giới hạn 9B áp dụng cho từng model hay tổng pipeline?

## 9. Gate sang giai đoạn 2

Giai đoạn 1 hoàn tất. Có thể bắt đầu preprocessing knowledge base mà không cần annotation, nhưng các giai đoạn supervised (split, NER, assertion, relation classifier, threshold tuning và benchmark) vẫn bị khóa cho đến khi có train annotations hợp lệ.

