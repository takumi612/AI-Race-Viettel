# Precision-First Hybrid Clinical NLP Pipeline Design

**Date:** 2026-07-18  
**Status:** Approved for implementation planning  
**Scope:** Viettel AI Race 2026 — Bài 2, Vòng 1

## 1. Mục tiêu

Viết lại baseline hiện tại thành một pipeline hybrid ưu tiên precision nhưng vẫn kiểm soát recall, đồng thời bám sát metric chính thức của cuộc thi. Thay đổi tập trung vào năm điểm:

1. Đánh giá bằng dữ liệu đáng tin cậy, không tune trên pseudo-ground-truth.
2. Chunking thích ứng theo cấu trúc và ngữ cảnh bệnh án, giữ nguyên character offset.
3. NER tách mention detection khỏi contextual type classification.
4. Retriever kết hợp BM25 và semantic score bằng trọng số có tổng bằng 1, ưu tiên BM25.
5. Candidate selector xuất tập mã nhỏ, có confidence, tối ưu Jaccard thay vì mặc định Top-5.

## 2. Ngoài phạm vi

Giai đoạn này không:

- Thay toàn bộ NER bằng XLM-R/CRF.
- Fine-tune LLM hoặc embedding model mới.
- Dùng API thương mại bên ngoài.
- Tối ưu riêng theo từng file trong 100 input công khai.
- Dùng holdout để chọn tham số hoặc sửa rule.

Neural NER chỉ được xem xét sau khi hybrid baseline đã được đo trên tập nhãn đáng tin cậy và các lỗi precision chính đã được xử lý.

## 3. Phân loại dữ liệu và chống leakage

### 3.1. Nguồn dữ liệu

- `data/input/`: 100 file input công khai dùng để inference và tạo bài nộp; không có nhãn chính thức để tune.
- `data/dev/input/1.txt` đến `100.txt` và `data/dev/gt/1.json` đến `100.json`: nhãn tự tạo; chỉ dùng cho error discovery và regression case được xác minh thủ công.
- `data/dev/input/101.txt` đến `200.txt` và `data/dev/gt/101.json` đến `200.json`: 100 nhãn được cung cấp; là nguồn metric đáng tin cậy.

### 3.2. Chia tập đáng tin cậy

- Development pool: file `101–180`.
- Final holdout: file `181–200`.
- Development pool dùng 5-fold cross-validation theo file để tune `alpha`, NER thresholds, assertion thresholds và candidate selection thresholds.
- Final holdout chỉ chạy sau khi cấu hình đã được khoá. Không sửa rule hoặc tham số dựa trên lỗi của holdout trong cùng vòng phát triển.

Mọi báo cáo phải ghi rõ kết quả đến từ pseudo-GT, cross-validation hay final holdout.

## 4. Metric và nguyên tắc chọn cấu hình

### 4.1. NER

NER được đánh giá theo exact character span và exact `type`.

Metric chính để tune threshold là `F0.5`:

```text
F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)
```

`F0.5` làm precision có trọng số lớn hơn recall. Hệ thống đồng thời phải báo cáo:

- Micro precision, recall và F0.5 toàn bộ entity.
- Precision, recall và F0.5 theo từng type.
- Số FP và FN theo section.
- Relaxed overlap metric chỉ dùng chẩn đoán lỗi, không dùng chọn cấu hình.

### 4.2. Assertion

Assertion được báo cáo bằng macro precision, recall và F0.5 cho từng nhãn:

- `isNegated`
- `isHistorical`
- `isFamily`

Không dùng accuracy tổng vì phần lớn entity có assertion rỗng.

### 4.3. Candidate mapping

Metric chọn candidate selector là:

- Candidate Jaccard theo công thức chấm hiện có.
- Candidate precision.
- Top-1 hit rate và retrieval Recall@20 chỉ dùng để tách lỗi retrieval khỏi lỗi selector.

Retriever được phép có Recall@20 cao, nhưng output cuối không được mặc định chứa toàn bộ Top-5.

### 4.4. Điểm toàn pipeline

Điểm cuối vẫn được báo cáo theo công thức cuộc thi. Trong từng stage, threshold được tune bằng metric phù hợp với stage đó. Một cấu hình chỉ được chấp nhận khi:

- F0.5 hoặc candidate Jaccard tốt hơn baseline trên cross-validation; và
- Điểm toàn pipeline trung bình trên cross-validation không giảm.

Nếu hai cấu hình có điểm toàn pipeline chênh không quá `0.005`, chọn cấu hình có precision cao hơn.

## 5. Kiến trúc mục tiêu

```text
Raw clinical document
    -> Section-aware adaptive chunker
    -> Mention detector
    -> Contextual type classifier and reject gate
    -> Span resolver and deduplicator
    -> Assertion analyzer
    -> Query normalizer
    -> BM25 Top-20 + Semantic Top-20
    -> Normalized weighted score fusion
    -> Clinical candidate validation
    -> Precision-first candidate selector
    -> Submission schema and offset validator
    -> JSON output
```

Mỗi component phải có interface độc lập và kiểm thử được mà không cần chạy toàn pipeline.

## 6. Adaptive clinical chunking

### 6.1. Nguyên tắc

- Không thay đổi text gốc trước khi tính offset.
- Mỗi chunk mang `text`, `start`, `end`, `section_type` và `header_text`.
- `document_text[chunk.start:chunk.end]` phải bằng `chunk.text`.
- Chunk boundary ưu tiên section, bullet, newline và sentence boundary trước token window.

### 6.2. Section detection

Chunker nhận diện các nhóm section tối thiểu:

- Patient demographics.
- Past medical history.
- Pre-admission medications.
- Current symptoms/history of present illness.
- Examination and assessment.
- Laboratory results.
- Imaging and procedures.
- Treatment/current medications.
- Unknown/general section.

Section detection dùng heading patterns và layout signal như dòng ngắn, đánh số, dấu `:` và bullet. Heading patterns nằm trong một data file có provenance, không nằm rải rác trong code.

### 6.3. Chia chunk

1. Chia document theo section.
2. Trong mỗi section, chia theo bullet hoặc sentence/newline.
3. Ghép các đơn vị nhỏ liên tiếp cho đến tối đa 384 model tokens.
4. Nếu một đơn vị vượt giới hạn, dùng sliding token window 384 tokens với overlap 64 tokens.
5. Boundary được dịch về sentence/newline gần nhất khi có thể.

Thông số 384/64 là mặc định cấu hình; giá trị thực tế được benchmark nhưng không tune trên final holdout.

### 6.4. Entity-centered context

- `THUỐC`: bullet/dòng chứa entity cộng section header; header có thể nằm ở chunk trước nhưng phải thuộc cùng section.
- `CHẨN_ĐOÁN` và `TRIỆU_CHỨNG`: câu chứa entity cộng tối đa một câu lân cận trong cùng section; ưu tiên câu sau, nếu không có mới dùng câu trước.
- `TÊN_XÉT_NGHIỆM` và `KẾT_QUẢ_XÉT_NGHIỆM`: cùng dòng hoặc cùng bullet, không mở rộng sang section khác.
- Assertion analyzer luôn nhận section type và header để nhận diện lịch sử hoặc gia đình.

## 7. Precision-first hybrid NER

### 7.1. Mention detection

Mention detector tạo span candidates từ:

- ICD-10/RxNorm database lexicon.
- Curated synonym and abbreviation files có provenance.
- High-precision medication patterns.
- High-precision laboratory name/value patterns.
- Symptom and diagnosis lexicon đã được audit.

Một span có thể mang nhiều type candidates. Dictionary match không tự quyết định type cuối.

### 7.2. Contextual type classification

Type scorer sử dụng các feature:

- `section_type` và heading.
- Từ khóa trước/sau span.
- Đơn vị xét nghiệm và cấu trúc `name: value`.
- Liều lượng, đường dùng, tần suất và động từ dùng thuốc.
- Độ dài span, exact lexicon match và lexical specificity.
- Type candidates do từng knowledge source đề xuất.

Ví dụ `creatinine` trong laboratory section với giá trị/đơn vị được ưu tiên `TÊN_XÉT_NGHIỆM`; trong medication section chỉ được xem là thuốc nếu có tín hiệu dùng thuốc đủ mạnh.

### 7.3. Reject gate

- Mỗi type có threshold riêng.
- Span chung chung hoặc một token như `yếu`, `đau`, `viêm`, `loét` cần context score cao hơn multi-word exact match.
- Nếu best type score dưới threshold, span bị reject thay vì cố gán type.
- Nếu hai type gần nhau hơn ambiguity margin, ưu tiên reject trừ khi section rule giải quyết được xung đột.

### 7.4. Span resolution

- Ưu tiên exact multi-word span và confidence cao.
- Không cho entity là substring nằm bên trong từ dài hơn.
- Entity từ overlapping token windows được deduplicate theo absolute offsets.
- Hai prediction cùng `(start, end, type)` chỉ giữ prediction confidence cao nhất.
- Overlap khác type được giải quyết bằng confidence, section consistency và specificity; không phụ thuộc thứ tự duyệt dictionary.

## 8. Assertion redesign

- Assertion scope không cắt bỏ negation cue trong các cấu trúc `không có`, `chưa phát hiện`, `phủ nhận có`.
- `Pre-admission medications` và `Past medical history` tạo prior cho `isHistorical`.
- Family history section tạo prior cho `isFamily`, nhưng phải dừng khi context quay lại bệnh nhân.
- Rule output deterministic và có confidence để có thể tune threshold.
- Assertion không áp dụng cho test name/result trong output schema.

## 9. Retriever score fusion

### 9.1. Candidate generation

- BM25 lấy Top-20 cùng raw BM25 score.
- FAISS lấy Top-20 cùng cosine/IP score.
- Candidate pool là hợp của hai danh sách theo code.
- Code không tồn tại trong knowledge base bị loại trước fusion.

### 9.2. Score normalization

Mỗi component được chuẩn hoá độc lập trong phạm vi query:

```text
normalized_score = (score - min_score) / (max_score - min_score)
```

Nếu component chỉ có một kết quả hoặc mọi score bằng nhau, các kết quả hiện diện nhận `1.0`. Candidate không xuất hiện trong component nhận `0.0`.

### 9.3. Weighted fusion

```text
fusion_score = alpha * bm25_normalized
             + (1 - alpha) * semantic_normalized
```

Ràng buộc:

- `0 <= alpha <= 1`.
- Tổng trọng số luôn bằng 1.
- Giá trị mặc định `alpha = 0.75`.
- Grid tune trên development folds: `0.60, 0.70, 0.75, 0.80, 0.90`.
- Tie-break lần lượt theo BM25 rank, semantic rank, rồi code để deterministic.

Không thêm exact-match bonus ngoài công thức làm tổng trọng số vượt 1. Exact match được thể hiện qua BM25 score và candidate selector.

## 10. Precision-first candidate selector

Candidate selector nhận ranked candidates với fusion score, component scores và clinical validation result.

- Xuất Top-1 khi score vượt confidence threshold hoặc margin Top-1/Top-2 vượt margin threshold.
- Xuất Top-2 khi hai candidate gần nhau, đều hợp lệ và configuration cho thấy Top-2 tăng Jaccard.
- Không xuất fixed Top-5.
- Cho phép `[]` khi mọi candidate không hợp lệ hoặc dưới minimum confidence.
- LLM reranker, nếu bật, phải trả về tập con của candidate pool; chỉ đổi thứ tự không được xem là reranking thành công.
- Threshold có thể khác giữa ICD-10 và RxNorm.

## 11. Clinical validation và code integrity

- Không tạo entity có text không xuất hiện trong input.
- Dual dagger/asterisk code chỉ được thêm vào candidate set của entity gốc khi code đã được normalize, tồn tại trong KB và việc thêm mã cải thiện metric trên development folds.
- Sau khi clinical filter loại candidate, selector xét tiếp candidate pool để điền kết quả; không trả danh sách thiếu chỉ vì lọc sau Top-5.
- Mọi output candidate phải qua existence check theo type.
- Historical RxNorm mapping phải được gọi thật hoặc bị loại khỏi tuyên bố/tài liệu.

## 12. Hardcode audit

### 12.1. Giữ trong code

- Tên năm entity types.
- Ba assertion labels.
- Schema contract và công thức metric.

### 12.2. Chuyển thành cấu hình

- `alpha`.
- Retrieval internal top-k.
- Chunk token limit và overlap.
- Per-type NER thresholds.
- Ambiguity margin.
- Candidate confidence và margin thresholds.
- LLM enable flag và timeout.

### 12.3. Chuyển thành data có provenance hoặc loại bỏ

- `CLINICAL_MEDICATIONS`.
- `CLINICAL_DIAGNOSES`.
- `SYMPTOMS` và `TEST_NAMES` viết trực tiếp trong Python.
- `blacklist_common`.
- Static override mappings.
- Section headings và assertion triggers.
- Default preference cho Oral Tablet/Capsule.

Verified override schema tối thiểu gồm normalized term, type, codes, source và note. Override validator từ chối:

- Code không tồn tại trong KB.
- RxNorm code có tên không tương thích với term.
- ICD mapping mâu thuẫn rõ với mô tả `with/without`, giới tính hoặc loại bệnh.
- Mapping không có provenance.

Các đường dẫn tuyệt đối như `D:\AI Race Viettel\...` phải được thay bằng `PROJECT_ROOT` hoặc CLI argument.

## 13. Error handling và submission safety

- Lỗi một file phải tạo đúng file JSON `[]` và ghi structured error log.
- Trước khi đóng gói, validator kiểm tra đủ 100 file, JSON parse, dynamic schema, allowed type/assertion, integer offsets, exact text slice và candidate existence.
- Packager luôn tạo `output.zip` chứa thư mục `output/` và chỉ dùng artefact vừa được validate.
- Không duy trì đồng thời một zip mới sai cấu trúc và một zip cũ đúng cấu trúc.

## 14. Testing strategy

### 14.1. Unit tests

- Chunk offsets và section boundaries.
- Chunk overlap deduplication.
- `creatinine` lab-versus-drug disambiguation.
- Không nhận `yếu` trong `yếu tố`.
- Negation scope cho `không có`, `chưa phát hiện`, `phủ nhận có`.
- Historical assertion trong pre-admission medication section.
- BM25/semantic normalization.
- Fusion weights luôn có tổng bằng 1.
- Candidate selector Top-1/Top-2/reject behavior.
- Override integrity với các mapping sai đã phát hiện.
- Không sinh dual-code entity giả.
- Submission schema, offsets và zip layout.

### 14.2. Integration tests

- Chạy pipeline trên sample chính thức trong đề.
- Chạy cross-validation trên `101–180`, báo cáo per-fold và mean/std.
- So sánh baseline và pipeline mới bằng exact NER F0.5, assertion macro F0.5, candidate Jaccard và final score.
- Chạy final holdout `181–200` đúng một lần sau khi khoá cấu hình.

### 14.3. Regression data

Pseudo-GT chỉ được chuyển thành regression case sau khi entity và label của case đó được kiểm tra thủ công. Không dùng toàn bộ pseudo-GT làm oracle.

## 15. Rollout order

1. Khoá evaluator và trusted split.
2. Thêm submission validator/packager và code integrity checks.
3. Thêm adaptive chunker.
4. Tách mention detection khỏi contextual type classification.
5. Sửa assertion scope và section priors.
6. Thay RRF hiện tại bằng normalized weighted fusion.
7. Thêm calibrated candidate selector.
8. Audit hardcode và override data.
9. Chạy cross-validation, khoá config, rồi chạy holdout.
10. Rerun 100 public inputs và tạo một submission artefact duy nhất.

## 16. Điều kiện hoàn thành

- Không có output entity mà `text != document[start:end]`.
- Không có candidate không tồn tại trong KB tương ứng.
- Không tạo entity giả để chứa dual code.
- Zip mới nhất có đúng `output/1.json` đến `output/100.json`.
- Tất cả unit/integration tests pass.
- Cross-validation báo cáo precision, recall, F0.5, Jaccard và final score với mean/std.
- Pipeline mới không giảm cross-validation final score và cải thiện metric precision-first tương ứng so với baseline chạy lại trên cùng split.
- Final holdout được báo cáo riêng, không dùng để tiếp tục tune trong cùng vòng phát triển.

