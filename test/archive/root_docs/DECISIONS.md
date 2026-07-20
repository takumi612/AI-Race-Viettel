# DECISIONS

## D-001 - Thực hiện theo stage gate

**Decision ID:** D-001  
**Stage:** 1 - Contract và data audit  
**Module:** Project execution  
**Default method:** Thực hiện lần lượt chín giai đoạn và cập nhật state sau mỗi giai đoạn.  
**Chosen method:** Hoàn tất giai đoạn 1, chưa chuyển sang preprocessing/training trong cùng batch.  
**Reason:** Project contract yêu cầu không tạo toàn bộ dự án trong một phản hồi quá lớn và phải cập nhật trạng thái trước giai đoạn mới.  
**Evidence:** Prompt nguồn xác định thứ tự bắt buộc và cấm train ở giai đoạn 1.  
**Alternatives rejected:** Tạo ngay notebook hoàn chỉnh với dữ liệu/nhãn giả.  
**Impact:** Audit và contract có thể kiểm tra độc lập; tiến độ chậm hơn nhưng không mất kiểm soát context.  
**Rollback plan:** Không cần rollback; stage tiếp theo chỉ bắt đầu sau khi state được đọc lại.

## D-002 - Xem `input.zip` là unlabeled inference candidate

**Decision ID:** D-002  
**Stage:** 1  
**Module:** Data loading / leakage control  
**Default method:** Kiểm tra train/test schema và label thực tế.  
**Chosen method:** Chỉ dùng `input.zip` để audit I/O, encoding, offsets và submission plumbing; cấm mọi fitting/tuning trên tập này.  
**Reason:** Archive chỉ chứa 100 file `.txt`, không có JSON hoặc annotation.  
**Evidence:** 101 ZIP members gồm một thư mục và 100 text files; `annotation_files_found=0`.  
**Alternatives rejected:** Suy diễn weak labels từ ontology hoặc dùng input text để fit TF-IDF/aliases.  
**Impact:** Không thể train/benchmark supervised modules cho tới khi có annotation, nhưng tránh leakage.  
**Rollback plan:** Khi organizer cung cấp train annotations, thêm annotated-loader adapter và document-level split mà không thay đổi raw-text contract.

## D-003 - Giữ NER mặc định XLM-R + BIO ở trạng thái conditional

**Decision ID:** D-003  
**Stage:** 1  
**Module:** Clinical entity detection  
**Default method:** XLM-RoBERTa-base token classification + BIO + sliding window.  
**Chosen method:** Giữ phương pháp mặc định trong SPEC nhưng chưa instantiate/train; GlobalPointer/span NER chỉ benchmark khi có nhãn nested/overlap.  
**Reason:** Không có annotation để đo nested rate, exact-span F1 hoặc tokenizer alignment errors.  
**Evidence:** Detected entity labels rỗng và không có ground truth.  
**Alternatives rejected:** Chuyển ngay sang GlobalPointer theo phỏng đoán hoặc huấn luyện trên pseudo-label.  
**Impact:** Kiến trúc phù hợp contract nhưng không tạo metric giả.  
**Rollback plan:** Nếu validation chứng minh BIO không phù hợp, ghi decision mới và thay module qua interface span detector.

## D-004 - Canonicalize ICD-10 nhưng bảo tồn marker

**Decision ID:** D-004  
**Stage:** 1  
**Module:** ICD-10 knowledge base  
**Default method:** Dùng mã có/không dấu chấm và tên Việt/Anh làm aliases.  
**Chosen method:** Tách `†`/`*` khỏi `canonical_code`, giữ `display_code`, marker và provenance làm metadata; deduplicate có kiểm soát.  
**Reason:** 858 mã hiển thị mang marker; bỏ marker giúp mã khớp cột không dấu chấm, nhưng xóa cả dòng sẽ làm mất ý nghĩa dagger/asterisk.  
**Evidence:** 82 dagger, 776 asterisk; canonical mismatch count bằng 0 sau khi tách marker.  
**Alternatives rejected:** Giữ marker trong candidate ID; hoặc deduplicate mù theo mã không dấu chấm.  
**Impact:** Linker xuất canonical ICD-10, đồng thời vẫn có metadata phục vụ diagnostics.  
**Rollback plan:** Nếu schema chính thức yêu cầu marker, xuất qua formatter riêng từ metadata mà không rebuild raw workbook.

## D-005 - Parse RxNorm trực tiếp trong ZIP bằng streaming

**Decision ID:** D-005  
**Stage:** 1  
**Module:** RxNorm knowledge base  
**Default method:** `zipfile` + streaming/chunked parser, cache Parquet hoặc SQLite.  
**Chosen method:** Giữ mặc định; không giải nén toàn bộ và không load RRF vào một DataFrame lớn.  
**Reason:** Archive giải nén 1.83 GB; RXNREL có hơn 7.4 triệu dòng.  
**Evidence:** `RXNCONSO.RRF` 131.6 MB/1,202,603 dòng; `RXNREL.RRF` 527.8 MB/7,423,180 dòng.  
**Alternatives rejected:** Giải nén toàn bộ; `pandas.read_csv` một lần cho toàn file.  
**Impact:** Phù hợp RAM Colab/Kaggle và cho phép cache tái sử dụng.  
**Rollback plan:** Có thể đổi backend cache giữa Parquet và SQLite qua config nếu benchmark I/O yêu cầu.

## D-006 - Dùng filter RxNorm bảo thủ cho baseline

**Decision ID:** D-006  
**Stage:** 1  
**Module:** RxNorm candidate dictionary  
**Default method:** `LAT=ENG`, `SAB=RXNORM`, `SUPPRESS=N`, TTY cấu hình gồm IN/PIN/MIN/BN/SCD/SBD/GPCK/BPCK/DF/DFG.  
**Chosen method:** Giữ filter mặc định và lưu release/filter cùng cache.  
**Reason:** Filter tạo candidate set nhỏ, có ngữ nghĩa rõ và tránh suppressed/source-specific noise ở baseline.  
**Evidence:** 56,053 dòng và 56,053 RXCUI duy nhất sau filter.  
**Alternatives rejected:** Index toàn bộ 1,202,603 rows ngay từ đầu; chỉ giữ ingredient và bỏ clinical drug/brand/dose form.  
**Impact:** Baseline nhẹ; alias enrichment từ nguồn khác chỉ thêm sau benchmark.  
**Rollback plan:** Mở rộng SAB/TTY bằng config và rebuild cache nếu validation Recall@k thấp.

## D-007 - Relation chỉ là diagnostics cho tới khi schema xác nhận

**Decision ID:** D-007  
**Stage:** 1  
**Module:** Clinical relation extraction / submission  
**Default method:** Rule baseline và không thêm relation nếu official schema không có trường này.  
**Chosen method:** Giữ rule-based relation module nội bộ; xuất `diagnostics/relations.json`, không thêm key vào entity JSON.  
**Reason:** Không có relation labels hoặc official submission schema trong dữ liệu hiện tại.  
**Evidence:** `input.zip` không có annotation; prompt chỉ đưa schema entity năm key.  
**Alternatives rejected:** Hard-code relation labels ví dụ vào submission.  
**Impact:** Đáp ứng yêu cầu module relation mà không phá schema.  
**Rollback plan:** Khi schema xác nhận relation, thêm schema converter/version mới và unit tests tương ứng.

## D-008 - Evaluator ở trạng thái provisional

**Decision ID:** D-008  
**Stage:** 1  
**Module:** Evaluation  
**Default method:** Strict matcher + approximate competition matcher.  
**Chosen method:** Giữ cả hai evaluator trong thiết kế; chưa tính metric và chưa tune threshold.  
**Reason:** Chưa có ground truth và chưa biết entity matching/WER chính thức.  
**Evidence:** Không có train/validation annotations hoặc evaluator từ ban tổ chức.  
**Alternatives rejected:** Báo cáo score từ rule baseline trên private test hoặc tự bịa ground truth.  
**Impact:** Tránh metric sai; interface vẫn sẵn sàng để cập nhật.  
**Rollback plan:** Thay approximate matcher bằng implementation chính thức khi organizer công bố.

## D-009 - Dùng deterministic JSONL.GZ cho ontology cache

**Decision ID:** D-009  
**Stage:** 2  
**Module:** Knowledge-base persistence  
**Default method:** Parquet hoặc SQLite cache.  
**Chosen method:** Gzip-compressed deterministic JSON Lines với metadata/checksum riêng.  
**Reason:** Runtime kiểm thử không có `pyarrow`; JSONL.GZ hỗ trợ streaming, không thêm dependency và vẫn nhỏ/nhanh với candidate set đã lọc.  
**Evidence:** ICD-10 cache 630.718 byte load 0,171 giây; RxNorm cache 1.444.746 byte load 0,465 giây; relation cache 2.123.808 byte stream trong 1,45 giây.  
**Alternatives rejected:** Cài thêm dependency chỉ để dùng Parquet; SQLite làm tăng độ phức tạp schema/migration cho baseline read-mostly.  
**Impact:** Artifact portable, deterministic và chạy offline; không có predicate pushdown như Parquet.  
**Rollback plan:** Giữ cùng record schema/iterator và thay backend bằng Parquet/SQLite nếu benchmark ở scale lớn chứng minh cần thiết.

## D-010 - Rule/dictionary baseline là active fallback khi thiếu annotation

**Decision ID:** D-010  
**Stage:** 3  
**Module:** Data baseline / entity detection  
**Default method:** XLM-R NER supervised với document-level split.  
**Chosen method:** Chạy ontology phrase matching + generic clinical rules; giữ XLM-R train interface nhưng không activate khi train annotation count bằng 0.  
**Reason:** Không có ground truth để tạo BIO labels hoặc đánh giá exact-span F1.  
**Evidence:** `reports/stage_03_eda.json`: 100 documents, 0 annotated train documents, 845 internal spans và 0 offset errors.  
**Alternatives rejected:** Pseudo-label từ private text hoặc báo cáo token F1 không có gold labels.  
**Impact:** Inference plumbing hoạt động offline; supervised model sẽ tự bật khi train adapter tìm thấy annotation.  
**Rollback plan:** Khi annotation xuất hiện, chạy document split rồi benchmark BIO và span model trên validation.

## D-011 - Giữ XLM-R Trainer nhưng khóa execution khi thiếu annotation

**Decision ID:** D-011  
**Stage:** 4  
**Module:** Clinical entity detection  
**Default method:** XLM-RoBERTa-base + `AutoModelForTokenClassification` + BIO + sliding window.  
**Chosen method:** Implement đầy đủ trainer/alignment/reconstruction; runtime hiện tại chỉ chạy fallback vì `torch`, `transformers` và annotations đều thiếu.  
**Reason:** Đảm bảo notebook có code train thật nhưng không tải model hoặc bịa nhãn/metric trong môi trường không đủ dependency.  
**Evidence:** `reports/stage_04_entity_extraction.json`: 0 annotations, trainer `trained=false`, sliding windows và BIO contracts được kiểm thử.  
**Alternatives rejected:** Cài model/đặt nhãn giả chỉ để làm notebook có vẻ đã train.  
**Impact:** Khi Colab/Kaggle có dependencies và train data, cùng interface sẽ train; local offline vẫn chạy baseline.  
**Rollback plan:** Nếu nested span F1 trên validation không đạt, thay detector sau interface bằng GlobalPointer/span implementation và ghi decision mới.

## D-012 - Hybrid assertion rules là active context predictor

**Decision ID:** D-012  
**Stage:** 5  
**Module:** Assertion và clinical context  
**Default method:** Shared XLM-R encoder với các head polarity/temporality/certainty/experiencer và rule cues.  
**Chosen method:** Rule cues + section/context features chạy active; multi-task model factory giữ sẵn cho annotated training.  
**Reason:** Không có assertion labels để fit heads hoặc tune validation Jaccard.  
**Evidence:** 845 internal entities, 0 offset errors; `reports/stage_05_clinical_context.json` ghi rõ phân bố nội bộ và `threshold_tuning=not_run`.  
**Alternatives rejected:** Gán mọi entity là `isHistorical` hoặc tự đặt official assertion names.  
**Impact:** Phủ định, tiền sử, người nhà và uncertainty được xử lý minh bạch trong diagnostics; submission không chứa label chưa xác nhận.  
**Rollback plan:** Khi có labels, train multi-task heads, tune từng axis trên validation và cập nhật assertion mapping artifact.

## D-013 - Lexical/character reranking là active linker

**Decision ID:** D-013  
**Stage:** 6  
**Module:** ICD-10/RxNorm candidate retrieval  
**Default method:** Exact, fuzzy, character TF-IDF/embedding và cross-encoder reranking.  
**Chosen method:** Exact alias + SequenceMatcher + character n-gram similarity + weighted lexical reranking; embedding/cross-encoder giữ optional.  
**Reason:** Baseline phải offline, gọn và chạy được trong runtime không có sentence-transformers/FAISS/scipy.  
**Evidence:** 1.000/1.000 ontology alias sanity top-1 hits; trên input 250 disease và 212 drug spans được route/link, 0 offset errors. Đây không phải competition recall.  
**Alternatives rejected:** Download model hoặc fit semantic index trên private text.  
**Impact:** Candidate linking có artifact-free fallback và có thể thay backend mà không đổi schema.  
**Rollback plan:** Khi train/validation labels có, benchmark embedding/cross-encoder và thay reranker nếu Recall@k/MRR tăng.

## D-014 - Relation baseline diagnostics-only

**Decision ID:** D-014  
**Stage:** 7  
**Module:** Clinical relation extraction  
**Default method:** Rule baseline, sau đó entity-pair classifier nếu có labels.  
**Chosen method:** Rule baseline với cùng câu, type pair và distance; lưu relation trong diagnostics, không đưa vào submission.  
**Reason:** Official relation schema và labels chưa có.  
**Evidence:** `reports/stage_07_relations.json`: 13.330 candidate pairs, 30 rule predictions, `relation_in_submission=false`.  
**Alternatives rejected:** Thêm relation keys vào JSON hoặc gán các tên relation ví dụ thành official labels.  
**Impact:** Có module relation và diagnostics mà không phá schema chính thức.  
**Rollback plan:** Bật schema converter/classifier khi organizer xác nhận relation field và labels.

## D-015 - Official schema adapter được xác nhận từ repository validator

**Decision ID:** D-015  
**Stage:** 8  
**Module:** Official schema conversion / submission  
**Default method:** Xuất mọi entity sau NER và map type/assertion sang official schema.  
**Chosen method:** Map theo `src/validation/submission.py`: `DISEASE -> CHẨN_ĐOÁN`, `DRUG -> THUỐC`, `SYMPTOM -> TRIỆU_CHỨNG`, `LAB_RESULT -> KẾT_QUẢ_XÉT_NGHIỆM`; assertions phủ định/tiền sử/gia đình map sang ba label được validator cho phép. Field được serialize theo từng entity type.  
**Reason:** Repository đích do người dùng chỉ định đã chứa validator chính thức, là bằng chứng cụ thể hơn giả định schema năm key đồng nhất ban đầu.  
**Evidence:** Validator của repository báo 0 lỗi trên 100 JSON; `reports/stage_08_integration.json` ghi 842 submission entities, 0 offset errors và ZIP 100 members/CRC pass.  
**Alternatives rejected:** Giữ 100 mảng rỗng dù schema đã có trong repository; xuất internal labels `DISEASE/DRUG/SYMPTOM`.  
**Impact:** Submission có dự đoán baseline thực và vẫn schema-safe; 3 `PATIENT_INFO` chưa có official type bị drop có log.  
**Rollback plan:** Chỉ cập nhật mapping/schema adapter và rerun `tools/run_pipeline.py` nếu ban tổ chức phát hành validator mới.
