# Thiết kế huấn luyện theo curriculum và metadata-first

**Phiên bản:** 2.0
**Ngày cập nhật:** 2026-07-23
**Trạng thái:** Chờ người dùng duyệt trước khi lập implementation plan

## 1. Mục đích

Tài liệu này đặc tả thiết kế đích cho quá trình huấn luyện và suy luận của pipeline trích xuất thông tin y khoa trong cuộc thi AI Race Viettel. Thiết kế mở rộng pipeline Kaggle end-to-end hiện tại bằng sáu cải tiến hướng đến tối ưu metric:

1. huấn luyện NER ba giai đoạn có nhận biết nguồn dữ liệu;
2. xử lý ngữ cảnh theo hướng metadata-first mà không thay đổi văn bản gốc;
3. giám sát owner-window an toàn tại biên chunk;
4. mô hình phân loại assertion độc lập và cơ chế khôi phục entity theo hướng KB-first;
5. đánh giá out-of-fold trên dữ liệu BTC và baseline/ablation bắt buộc;
6. kiểm soát phiên bản dataset, KB và báo cáo audit bằng fingerprint.

Thiết kế phải sử dụng 2.000 tài liệu synthetic v2 để tăng độ phủ, nhưng không để văn phong hoặc phân bố template của dữ liệu synthetic lấn át 100 tài liệu có nhãn đáng tin cậy từ ban tổ chức (BTC).

## 2. Contract của bài thi và mục tiêu tối ưu

Pipeline trích xuất năm loại entity:

- chẩn đoán;
- thuốc;
- triệu chứng;
- tên xét nghiệm;
- kết quả xét nghiệm.

Chỉ xuất ba trường assertion chính thức:

- `isNegated`;
- `isHistorical`;
- `isFamily`.

Candidate ICD-10 chỉ được gắn cho entity chẩn đoán. Candidate RxNorm chỉ được gắn cho entity thuốc. Các loại entity còn lại không có candidate ontology.

Trong nhãn BTC đã quan sát, mỗi entity chẩn đoán hoặc thuốc có tối đa một candidate ở output cuối. Nhiều entity có danh sách candidate rỗng; đây là nhãn abstention hợp lệ, không phải lỗi cần tự động điền. Vì vậy, cấu hình submission mặc định vẫn là `candidate_output_k = 1`. Giới hạn này không áp dụng cho bước retrieval: tập candidate nội bộ vẫn có thể cấu hình, mặc định lấy tối đa 20 candidate trước khi lọc và rerank.

Evaluator local chỉ là công cụ đại diện để phát triển, chưa được xác nhận là mã chấm chính thức của BTC. Token-level F1 và công thức tổng hợp 30/30/40 hiện tại vẫn được báo cáo để chẩn đoán, nhưng việc chọn mô hình phải ưu tiên chất lượng entity ở cấp tài liệu và báo cáo riêng chất lượng assertion cùng entity linking. Mọi kết luận tối ưu metric phải ghi rõ đây là tối ưu trên proxy metric cho đến khi có evaluator chính thức hoặc phản hồi leaderboard.

## 3. Chính sách dữ liệu

Corpus chuẩn là `data_v2/Training_data/synthetic_train_v2`.

| ID | Nguồn gốc | Vai trò |
| --- | --- | --- |
| 1–100 | Input của BTC với GT được khôi phục/tự sinh | Chỉ cách ly và audit; không dùng để fit mô hình hoặc hiệu chỉnh threshold |
| 101–200 | Input của BTC với GT bị leak do BTC cung cấp | Corpus thật có nhãn đáng tin cậy |
| 201–2200 | Synthetic v2 | Tăng độ phủ, warm-up biểu diễn, khái niệm hiếm, đa dạng thể loại và regularization |

Không loại bỏ 2.000 tài liệu synthetic. Tất cả phải đủ điều kiện tham gia Stage 1 và Stage 2. Stage 3 chỉ dùng một tập replay đã chọn để bước thích nghi cuối bám theo ngôn ngữ của BTC thay vì văn phong synthetic.

Các file input và GT gốc không được thay đổi trong quá trình huấn luyện. Metadata dẫn xuất, split, feature và kết quả calibration được lưu thành artifact sidecar.

### 3.1. Quality gate cho synthetic v2

Profile hiện tại của 2.000 tài liệu synthetic là baseline chất lượng tối thiểu, không phải mục tiêu để overfit:

- 12 document genre bệnh viện, mỗi genre chiếm khoảng 1/12 corpus;
- độ dài 336–492 từ, trung bình khoảng 413,8 từ/tài liệu;
- 400 tài liệu long-tail, tương đương 20%;
- tối thiểu 411 mã ICD-10 và 414 mã RxNorm khác nhau;
- không có fixed exact line xuất hiện trong toàn bộ 2.000 tài liệu;
- không có lỗi offset, schema hoặc candidate ngoài raw KB;
- không có vi phạm rõ ràng về tuổi, giới và bối cảnh sản/nhi khoa trong quality gate hiện tại.

Một lần sinh lại hoặc sửa dataset không được làm giảm các chỉ số trên quá 5% tương đối, trừ khi có biên bản giải thích rằng việc giảm độ phủ giúp loại bỏ nhãn sai. Không ép mọi tài liệu đạt đúng một độ dài hoặc số entity cố định.

### 3.2. Phiên bản và độ tin cậy của báo cáo

Mọi dataset report phải chứa:

- fingerprint của toàn bộ cặp input/GT;
- hash của manifest và KB artifact;
- thời điểm tạo, phiên bản generator/validator và phạm vi document ID;
- trạng thái `current`, `stale` hoặc `archived`.

Report có fingerprint không khớp dataset hiện tại không được dùng làm bằng chứng nghiệm thu. Báo cáo cũ phải được chuyển sang vùng archive hoặc hiển thị rõ trạng thái `stale`. Quality gate phải thất bại nếu hai báo cáo cùng nhận là `current` nhưng chứa thống kê mâu thuẫn.

### 3.3. Audit nhãn độc lập

Trước huấn luyện chính thức, bắt buộc có hai lớp xác minh:

1. validator tất định kiểm tra toàn bộ schema, offset, assertion, candidate và tính nhất quán lâm sàng có thể biểu diễn bằng contract;
2. ít nhất hai agent độc lập audit một mẫu phân tầng của phiên bản dataset hiện tại.

Mẫu agent audit phải phủ đủ 12 genre, standard/long-tail, năm loại entity, ba assertion, candidate rỗng và các candidate hiếm. Hai agent không được tự động sửa GT. Bất đồng được ghi vào review queue; chỉ thay đổi nhãn khi có bằng chứng từ raw text, raw KB và quy tắc contract. ID 1–100 vẫn bị cách ly kể cả khi một phần nhãn vượt qua audit.

### 3.4. Development split và out-of-fold

Một holdout cố định 20 tài liệu không đủ ổn định và dễ bị overfit khi đồng thời chọn NER, assertion, linking và metadata. Development giữ riêng khoảng 10 tài liệu làm blind challenge, sau đó sử dụng grouped 5-fold out-of-fold (OOF) trên 90 tài liệu BTC còn lại:

- mỗi fold có khoảng 72 tài liệu organizer train và 18 tài liệu organizer validation;
- mỗi tài liệu trong 90 tài liệu OOF xuất hiện trong validation đúng một lần;
- 80% synthetic train và 20% `synthetic_holdout` được chia deterministic theo exact/near-duplicate group;
- Stage 1 synthetic checkpoint được dùng chung giữa các fold khi dataset/config hash trùng nhau;
- Stage 2–3, assertion và calibration được thực hiện riêng cho từng fold.

`blind_challenge_slice` khoảng 10 tài liệu organizer được stratify trước khi tạo OOF. Slice này không được dùng để train, chọn epoch, threshold, ablation hay metadata. Nó chỉ được chạy một lần sau khi cấu hình đã khóa để phát hiện overfit OOF; nếu dùng kết quả slice để đổi cấu hình, phải bắt đầu lại một experiment manifest mới.

Group cứng chỉ được tạo từ exact duplicate, near-duplicate có độ tương đồng vượt threshold đã ghi trong config và đóng băng trước split, hoặc cùng một record/patient bị tách thành nhiều file. Không nối transitive toàn bộ corpus chỉ vì hai tài liệu dùng chung một bệnh hoặc thuốc phổ biến. Template fingerprint và surface entity được dùng để stratify, đo overlap và phân tích seen/unseen; surface overlap phổ biến không bị cấm tuyệt đối.

Mỗi fold cần bảo toàn tốt nhất có thể phân bố loại entity, assertion, genre, candidate rỗng, long-tail và độ dài tài liệu. Splitter phải kiểm tra không có hard-group leakage và phải tạo được 5 fold không rỗng trên 90 tài liệu OOF; nếu không tạo được, pipeline dừng với báo cáo group gây nghẽn thay vì âm thầm quay về random split.

OOF prediction của 90 tài liệu được ghép lại để tính metric, calibration threshold và error analysis. Blind challenge chỉ được xem sau khi cấu hình đã khóa. Synthetic holdout chỉ dùng để chẩn đoán độ phủ và hiện tượng ghi nhớ; kết quả synthetic không được phép lấn át suy giảm trên OOF organizer.

### 3.5. Quy trình final-fit

Sau khi chọn được curriculum, threshold và các hyperparameter khác, quá trình final-fit sử dụng toàn bộ 100 tài liệu thật đáng tin cậy. Điểm số được tạo trên chính 100 tài liệu này sau final-fit không được báo cáo như một kết quả validation không thiên lệch.

Lần chạy final-fit sử dụng các quyết định đã được chọn và đóng băng trước khi xem kết quả final-fit:

- số epoch đã chọn, điểm kết thúc từng stage và learning rate;
- tỷ lệ sampling;
- threshold assertion;
- threshold và margin của linking.

Final-fit không early-stop hoặc chọn checkpoint dựa trên 100 tài liệu đang được dùng để fit. Nó chạy đúng lịch huấn luyện được chọn từ OOF và lưu trạng thái cuối của stage theo cấu hình. Threshold assertion/linking được chọn từ OOF prediction; không calibration lại trên prediction in-sample của final model.

Artifact cuối phải lưu cả metric của development split và thông tin rằng mô hình được giao đã được fit tiếp bằng toàn bộ 100 nhãn thật đáng tin cậy.

## 4. Mô hình ngữ cảnh metadata-first

Không viết lại văn bản y khoa gốc vì offset trong output phải tham chiếu trực tiếp đến tài liệu ban đầu. Ngữ cảnh được biểu diễn trong sidecar record với các trường sau khi có thể phát hiện:

- `document_id`;
- `source_bucket`: `quarantine`, `organizer` hoặc `synthetic`;
- `document_genre`;
- `template_group` và `near_duplicate_group`;
- `record_spans` và `patient_block_id` cho từng hồ sơ/bệnh nhân trong file;
- span của section và loại section đã chuẩn hóa;
- vai trò người nói hoặc người viết;
- chủ thể/experiencer, ví dụ bệnh nhân hoặc người nhà;
- `primary_surfaces` và nhóm tần suất của surface, chỉ dùng cho stratification/diagnostic;
- cờ long-tail và ontology coverage;
- content hash và phiên bản bộ sinh dữ liệu.

Cho phép dùng rule nhận diện tiêu đề section vì chúng xác định cấu trúc tài liệu. Regex nội dung cho bệnh, thuốc hoặc triệu chứng không được dùng làm bộ sinh nhãn chính. Bộ phân tích giá trị xét nghiệm có thể được giữ làm lớp làm giàu dữ liệu có cấu trúc, với điều kiện kết quả của nó không bị coi là ground truth mặc định.

Metadata có thể được sử dụng cho:

- chia dữ liệu theo group;
- sampling có nhận biết nguồn;
- xác định ngữ cảnh assertion;
- chẩn đoán prediction;
- token ngữ cảnh đặc biệt tùy chọn, chỉ khi ablation trên organizer OOF chứng minh có cải thiện.

Trong tài liệu này, `real_holdout` ở các mô tả cũ được thay bằng organizer validation fold tương ứng hoặc tập OOF gộp. Baseline không chèn metadata token vào encoder. Quyết định này giúp giữ nguyên offset và tránh biến metadata được phát hiện chưa chính xác thành đầu vào bắt buộc của mô hình.

Record boundary có hai mức confidence:

- high-confidence: delimiter/định danh bệnh nhân rõ ràng và cấu trúc span hợp lệ;
- uncertain: không đủ bằng chứng để hard split.

Chunking và assertion context không được vượt qua high-confidence record boundary. Với boundary uncertain, hệ thống giữ nguyên tài liệu để tránh cắt sai nhưng phải ghi diagnostic và chặn assertion cue tại delimiter gần nhất có thể xác minh.

## 5. Chunking an toàn tại biên

Cửa sổ tokenizer mặc định:

- `max_length = 512` token, bao gồm special token;
- overlap/stride mục tiêu là 128 token.

Tokenization thực hiện theo record span high-confidence trước, sau đó mới tạo cửa sổ overlap trong từng record. Nếu không có record span đáng tin cậy, toàn tài liệu là một record fallback. Mỗi feature phải giữ document ID, `patient_block_id`, raw character offset, biên record, biên cửa sổ và metadata nguồn.

### 5.1. Giám sát owner-window

Mỗi gold entity được gán cho đúng một owner window. Window hợp lệ phải chứa trọn vẹn span entity. Owner là window hợp lệ làm cực đại khoảng cách token nhỏ hơn giữa entity với hai biên trái/phải có thể sử dụng. Nếu bằng nhau, chọn window có index nhỏ hơn.

Nhãn huấn luyện tuân theo các quy tắc:

1. owner window nhận đầy đủ nhãn BIO của entity;
2. các bản sao của entity trong những window overlap khác được mask bằng `-100` khi tính loss;
3. window chỉ chứa một phần entity sẽ mask các token entity nhìn thấy bằng `-100`;
4. token không thuộc entity vẫn là nhãn `O` hợp lệ, trừ khi bị mask bởi quy tắc special token hoặc padding của tokenizer;
5. entity không được phép bị chia bởi một record boundary; trường hợp này làm validation thất bại và yêu cầu sửa boundary metadata;
6. nếu không có window nào chứa trọn entity, preprocessing phải dừng đối với tài liệu đó và ghi diagnostic, thay vì âm thầm chuyển entity thành `O`.

Cơ chế này đảm bảo mỗi entity chỉ đóng góp vào loss một lần, overlap không làm thay đổi tần suất lớp và không tạo supervision sai ở biên.

### 5.2. Hợp nhất khi inference

Prediction được chiếu về raw character offset. Các prediction trùng hoàn toàn được gộp trước. Span cùng loại và overlap được hợp nhất bằng confidence đã calibration và độ đầy đủ tại biên. Không merge prediction qua hai `patient_block_id` khác nhau. Xung đột khác loại được xử lý tất định theo độ tin cậy của nguồn, confidence và độ đầy đủ của span; mọi phương án bị loại phải được ghi vào diagnostic.

## 6. Curriculum NER ba giai đoạn

Backbone NER tiếp tục là XLM-R, trừ khi một thí nghiệm có kiểm soát trên organizer OOF chứng minh mô hình khác tốt hơn. Hai mươi epoch là giới hạn an toàn tuyệt đối, không phải số epoch mục tiêu.

### 6.1. Stage 1: warm-up bằng synthetic

Mục tiêu: học năm loại entity, biến thể từ vựng rộng, thể loại tài liệu y khoa, surface ICD/RxNorm hiếm và hành vi tại biên.

- Dữ liệu: phần synthetic train ở development; toàn bộ 2.000 tài liệu synthetic ở final-fit.
- Giới hạn epoch: 3, khoảng dự kiến 1–3.
- Khoảng learning rate ban đầu: `2e-5` đến `3e-5`.
- Tín hiệu lựa chọn ở development: metric entity trên synthetic và điều kiện không suy giảm trên organizer OOF.
- Output: `stage1_synthetic_checkpoint`.

### 6.2. Stage 2: huấn luyện trộn cân bằng theo nguồn

Mục tiêu: căn chỉnh biểu diễn theo cách viết của BTC trong khi vẫn giữ độ phủ synthetic.

- Dữ liệu: 80 tài liệu thật dùng để train cùng phần synthetic train ở development; toàn bộ 100 tài liệu thật cùng 2.000 tài liệu synthetic ở final-fit.
- Đơn vị sampling: các chunk có nhãn owner-window, được nhóm theo nguồn và tài liệu.
- Tỷ lệ tiếp xúc mục tiêu trong mỗi epoch-equivalent: 30–40% chunk organizer và 60–70% chunk synthetic.
- Giới hạn epoch: 2, khoảng dự kiến 1–2.
- Sampler không được tạo cân bằng bằng cách sao chép file hoặc sửa corpus.
- Output: `stage2_mixed_checkpoint`.

### 6.3. Stage 3: thích nghi với dữ liệu BTC và synthetic replay

Mục tiêu: tối đa hóa độ phù hợp với ngôn ngữ và phong cách gán nhãn của BTC, đồng thời tránh quên thảm họa đối với khái niệm hiếm.

- Dữ liệu ở development: 80 tài liệu thật dùng để train cùng replay được chọn từ phần synthetic train.
- Dữ liệu ở final-fit: toàn bộ 100 tài liệu thật cùng replay được chọn bằng chính sách đã đóng băng từ development.
- Tỷ lệ replay: 15–20% ví dụ của Stage 3.
- Replay ưu tiên surface entity hiếm, khái niệm ICD/RxNorm long-tail, genre hiếm, tổ hợp assertion hiếm và ví dụ khó tại biên.
- Khoảng learning rate ban đầu: `5e-6` đến `1e-5`.
- Giới hạn ở development: 8 epoch; early stopping dự kiến dừng sớm hơn.
- Final-fit dùng số epoch Stage 3 đã được chọn trong development và không đánh giá trên 100 tài liệu đang dùng để fit.
- Output: `final_ner_model`.

### 6.4. Early stopping và chọn checkpoint

Checkpoint ở development được chọn bằng prediction cấp tài liệu trên organizer OOF:

`selection_score = 0.70 * exact_entity_f1 + 0.20 * overlap_entity_f1 + 0.10 * macro_type_f1`

Yêu cầu:

- exact-span F1 được tính micro-average trên các entity;
- overlap F1 sử dụng ghép cặp một-một và một overlap threshold được ghi rõ;
- macro type F1 là trung bình đều của năm loại entity;
- token-level F1 chỉ phục vụ chẩn đoán;
- patience áp dụng trên selection score ở từng fold, không phải training loss;
- checkpoint không được chọn nếu validation schema hoặc offset thất bại.

Báo cáo phải lưu mean, standard deviation và worst-fold cho metric tổng; đồng thời lưu metric theo từng loại, từng genre, surface đã thấy/chưa thấy, candidate rỗng, long-tail và số lỗi boundary. Không được chọn cấu hình dựa trên một fold tốt bất thường.

### 6.5. Baseline và ablation bắt buộc

Trước khi chốt final-fit, chạy cùng một OOF split và seed policy cho tối thiểu các cấu hình:

1. pipeline NER hiện tại;
2. NER + owner-window;
3. curriculum ba stage;
4. curriculum + assertion classifier;
5. curriculum + KB-first recovery;
6. đầy đủ pipeline có calibration và reranker tùy chọn.

Mỗi cấu hình phải chạy qua cùng một decoder, overlap merge, assertion interface và candidate policy trước khi báo cáo exact entity F1, assertion F1, candidate accuracy/coverage, KB-first incremental precision/recall, runtime và peak memory. Token-level F1 của baseline chỉ là diagnostic, không được so trực tiếp với entity-level F1 của cấu hình mới. Thành phần mới chỉ được giữ nếu cải thiện OOF selection score hoặc cải thiện một metric phụ quan trọng mà không làm metric chính giảm quá ngưỡng đã định trong config. Search budget, seed, thứ tự thử, git commit, dataset/KB fingerprint và danh sách document của từng fold phải được ghi vào `experiment_manifest.json` để tránh overfit OOF.

## 7. Mô hình assertion

Assertion được xử lý bởi một classifier multi-label riêng, thay vì regex nội dung diện rộng.

Classifier chỉ nhận các entity type được contract cho phép assertion: `DISEASE`, `DRUG` và `SYMPTOM`. `LAB_NAME` và `LAB_RESULT` không được tạo assertion output hoặc dùng như nhãn assertion dương tính. Với mỗi entity được phát hiện hoặc gold entity thuộc nhóm hợp lệ, classifier nhận:

- cửa sổ ngữ cảnh cục bộ có đánh dấu entity;
- metadata section và experiencer nếu có;
- loại entity;
- ngữ cảnh câu/mệnh đề gốc.

Mô hình dự đoán xác suất độc lập cho `isNegated`, `isHistorical` và `isFamily`; các nhãn có thể đồng thời cùng đúng nếu raw text hỗ trợ. Với mỗi entity hợp lệ, absence của một assertion trong GT là nhãn âm cho axis đó, không phải missing label. Threshold của từng nhãn được calibration riêng trên OOF organizer prediction, có báo cáo precision/recall/F1 và prevalence. Ví dụ assertion synthetic cung cấp regularization và độ phủ, nhưng ví dụ organizer nhận trọng số sampling cao hơn.

Để phù hợp tài nguyên Kaggle, assertion classifier có thể tái sử dụng một bản encoder NER đã freeze hoặc dùng classification head/adapter nhẹ. Phương án được chọn dựa trên assertion F1 ở OOF organizer và chi phí runtime. Sau final-fit, không calibration lại threshold trên prediction in-sample.

Rule chỉ được dùng làm fallback cấu trúc có precision cao và phục vụ diagnostic. Rule không được suy luận assertion từ keyword không giới hạn phạm vi trên toàn tài liệu. Mọi cue của rule phải bị giới hạn trong mệnh đề/câu và section tương ứng.

## 8. Ontology retrieval, khôi phục entity và linking

Quá trình phát hiện entity có hai đường recall độc lập:

1. prediction của NER;
2. phrase/alias retrieval theo hướng KB-first trên ICD-10 và RxNorm.

Nhánh KB-first tìm alias chuẩn hóa khớp chính xác trước, sau đó mới xét phương án lexical/fuzzy hoặc semantic có kiểm soát. Nhánh này có thể đề xuất mention bệnh hoặc thuốc mà NER bỏ sót, nhưng không được trực tiếp ép mention vào output. Proposal phải qua xác minh span, kiểm tra type, confidence threshold, giải quyết xung đột và candidate calibration. Alias một token, acronym mơ hồ, stopword y khoa và phrase có ranh giới chữ số/chữ cái không hợp lệ phải qua boundary/ambiguity filter; không được quét mọi tên KB rồi xuất tất cả.

Với mỗi mention chẩn đoán hoặc thuốc được chấp nhận:

1. lexical và semantic retrieval tạo tối đa 20 candidate nội bộ;
2. phần head của mention thuốc được tách khỏi hàm lượng, đường dùng, dạng bào chế và tần suất;
3. bộ lọc tất định loại các entry sai type hoặc không hợp lệ trong KB;
4. reranker sắp xếp tập candidate còn lại;
5. áp dụng minimum score, top-1/top-2 margin và abstention threshold;
6. xuất tối đa một candidate.

Candidate contract có hai lớp định danh:

- `canonical_id`: ID dùng trong index và linking;
- `official_display_id`: chuỗi candidate cần xuất nếu contract/evaluator yêu cầu giữ dạng hiển thị, ví dụ mã ICD có dấu `*`.

Candidate rỗng trong GT BTC là nhãn abstention hợp lệ. Non-empty candidate của mọi entity organizer dùng cho calibration phải tồn tại trong raw KB và runtime artifact; coverage bắt buộc là 100% trước khi train. Nếu raw KB có ID nhưng runtime artifact thiếu, pipeline dừng ở bước build KB và báo danh sách thiếu, không chuyển thành warning.

Chất lượng retrieval được đo riêng bằng recall@1, recall@5, recall@10, top-1 accuracy, coverage và accuracy có điều kiện trên các trường hợp không abstain. Nhánh KB-first còn phải báo cáo incremental precision/recall ở cấp mention so với NER-only, số false positive trên tài liệu không có entity tương ứng và tỷ lệ candidate rỗng. Threshold được chọn trên OOF organizer prediction và đóng băng trước final-fit.

Qwen là thành phần tùy chọn, chỉ dùng để rerank candidate mơ hồ hoặc fallback cho assertion. Qwen chỉ được chọn từ candidate hợp lệ đã cung cấp. Thiếu weights, JSON sai, timeout hoặc lỗi CUDA phải quay về hành vi tất định và không được làm dừng pipeline.

## 9. Điều phối notebook Kaggle

Notebook chuẩn tiếp tục là `v2/medical_information_extraction_kaggle.ipynb` và hỗ trợ:

- `RUN_MODE = "full"`: validate dữ liệu, train mọi stage, calibration các thành phần development khi phù hợp, chạy inference và đóng gói artifact;
- `RUN_MODE = "resume"`: tiếp tục từ stage hoàn tất gần nhất có manifest và checkpoint vượt qua kiểm tra toàn vẹn;
- `RUN_MODE = "inference_only"`: load artifact cuối và tạo output cuộc thi mà không train.

Notebook thực hiện tuần tự các pha logic:

1. khám phá môi trường, dataset và KB;
2. validation tất định, fingerprint và tạo manifest;
3. trích xuất metadata, record boundary và tạo grouped OOF split;
4. tạo feature owner-window;
5. audit độc lập phiên bản dataset hiện tại;
6. Stage 1 warm-up bằng synthetic, cache theo dataset/config hash;
7. Stage 2 huấn luyện trộn cân bằng theo nguồn ở từng fold;
8. Stage 3 thích nghi với organizer ở từng fold;
9. train và calibration assertion theo OOF;
10. linking retrieval, KB coverage và calibration threshold theo OOF;
11. baseline/ablation và chốt experiment manifest;
12. chạy final-fit khi được bật;
13. smoke test reload checkpoint;
14. inference, validate output và đóng gói artifact.

Mỗi stage và mỗi fold hoàn tất phải ghi một stage manifest theo cách atomic. Chỉ được resume khi manifest khớp dataset fingerprint, KB hash, phiên bản code/config, định danh tokenizer, label mapping, OOF split và checkpoint inventory. Stage 1 cache không được dùng lại nếu synthetic partition hoặc owner-window config thay đổi.

## 10. Artifact và khả năng quan sát

Lần chạy tối thiểu phải tạo:

- `stage1_synthetic_checkpoint/`;
- `stage2_mixed_checkpoint/`;
- `final_ner_model/`;
- `assertion_model/`;
- `candidate_calibration.json`;
- `kb_coverage_report.json`;
- `metadata_manifest.jsonl`;
- `split_manifest.json`;
- `oof_predictions.jsonl`;
- `experiment_manifest.json`;
- `audit_manifest.json`;
- `training_history.json`;
- `evaluation_report.json`;
- `run_manifest.json`;
- `diagnostics/run_summary.json`;
- `output.zip`;
- `trained_ner_artifacts.zip`.

Checkpoint optimizer trung gian có thể bị xóa sau khi checkpoint được chọn đã reload thành công và archive cuối đã được xác minh. Model archive bàn giao phải giữ mọi file cần thiết để reload offline, preprocessing, label mapping, assertion inference và candidate calibration.

Diagnostic phải phân biệt:

- entity có nguồn từ NER và entity có nguồn từ KB-first;
- entity bị loại bởi xung đột overlap/type;
- quyết định từ assertion classifier và từ fallback;
- linking abstention cùng nguyên nhân;
- mức tiếp xúc dữ liệu theo từng nguồn trong huấn luyện;
- runtime và peak memory của từng stage;
- hiệu năng trên surface chưa thấy hoặc hiếm.

Mọi metric phải phân biệt ít nhất ba cấp:

- document-level;
- record/patient-block-level;
- entity/mention-level.

Tất cả report phải ghi số document, số record, số entity và số chunk tham gia metric để không nhầm “nhiều chunk” với “nhiều bằng chứng độc lập”.

## 11. Xử lý lỗi

Pipeline phải dừng sớm trong các trường hợp:

- cặp input/GT không hợp lệ;
- lỗi schema hoặc offset;
- candidate không tồn tại trong KB đi kèm;
- non-empty gold candidate của organizer không có trong runtime KB artifact;
- report hiện tại có fingerprint không khớp dataset/KB;
- rò rỉ hard-group giữa train và OOF fold;
- không thể tạo đủ 5 grouped OOF fold;
- thiếu `record_spans` khi assertion hoặc chunking yêu cầu hard boundary;
- entity không thể nằm trọn trong bất kỳ window nào theo cấu hình;
- checkpoint được chọn không thể load;
- resume manifest không khớp;
- output sai định dạng hoặc cấu trúc ZIP không hợp lệ.

Nếu semantic retrieval hoặc Qwen tùy chọn bị lỗi, pipeline hạ cấp về hành vi lexical/tất định và ghi warning. Các lỗi tùy chọn này không làm vô hiệu một lần chạy vốn đáp ứng đúng contract.

Ngược lại, thiếu gold candidate trong runtime artifact, stale report hoặc không tạo được grouped OOF là lỗi chặn, không phải warning.

## 12. Chiến lược xác minh

Quá trình triển khai tuân theo test-driven development. Các tầng kiểm thử bắt buộc:

1. **Unit test**: phân tích metadata, record boundary, gán owner-window, mask partial span, source-aware sampling, metric cấp tài liệu, assertion threshold, cắt candidate, abstention và fallback tất định.
2. **Integration test**: bàn giao checkpoint giữa ba stage, grouped OOF 5-fold, tính toàn vẹn khi resume, reload mô hình, inference end-to-end, schema output, offset và candidate hợp lệ.
3. **Data test**: không có lỗi pairing/schema/offset/KB, không có hard-group leakage, đủ 5 OOF fold, đúng số lượng theo nguồn và đúng phân bố replay.
4. **Fast development smoke test**: dataset local nhỏ hoàn tất mọi stage logic mà không tải mô hình lớn.
5. **Kaggle acceptance run**: một lần `Run All` thực tế trên GPU tạo weights có thể reload và các archive output vượt qua kiểm tra CRC.

Hai agent review độc lập là bắt buộc trước lần train chính thức. Kết quả review mang tính tư vấn và không được tự động rewrite GT; kiểm tra tất định, raw text/raw KB và nhãn organizer đáng tin cậy vẫn là nguồn chuẩn để quyết định review queue.

## 13. Tiêu chí nghiệm thu

Thiết kế chỉ được coi là triển khai thành công khi thỏa mãn toàn bộ điều kiện:

- ID 1–100 bị loại khỏi fitting và calibration.
- Toàn bộ 2.000 tài liệu synthetic đủ điều kiện tham gia Stage 1 và Stage 2.
- Quality gate synthetic giữ 12 genre, khoảng 400 long-tail, độ dài trung bình khoảng 413,8 từ và độ phủ ontology theo baseline hiện tại, trong biên giảm tương đối tối đa 5% nếu không có biên bản giải thích.
- Mô hình cuối sử dụng toàn bộ 100 tài liệu có nhãn organizer đáng tin cậy sau khi các quyết định development đã được đóng băng.
- Không gold entity nào đóng góp NER loss trong nhiều hơn một window overlap.
- Partial entity không bao giờ bị chuyển thành supervision `O`.
- Việc chọn mô hình sử dụng metric entity cấp tài liệu trên organizer OOF 5-fold, có mean/std/worst-fold.
- Assertion threshold và candidate abstention threshold được calibration trên OOF organizer đáng tin cậy, không calibration lại trên final-fit in-sample prediction.
- Grouped OOF tạo đủ 5 fold và không có hard-group leakage.
- `synthetic_holdout` chiếm 20% synthetic theo split deterministic và `blind_challenge_slice` được giữ ngoài mọi quyết định tuning.
- Baseline và sáu cấu hình ablation được ghi trong `experiment_manifest.json`; thành phần mới không được làm giảm metric chính quá ngưỡng đã định.
- KB-first incremental precision/recall được báo cáo riêng so với NER-only.
- Mọi non-empty gold candidate organizer có mặt trong runtime KB artifact trước khi train.
- Report hiện tại có fingerprint khớp dataset, manifest và KB; report cũ được đánh dấu `stale` hoặc `archived`.
- Hai agent độc lập đã audit mẫu phân tầng của đúng phiên bản dataset đang train.
- Regex nội dung chung cho bệnh, thuốc hoặc triệu chứng không tồn tại trong bộ phát hiện chính.
- Retrieval nội bộ giữ nhiều candidate trong khi output cuối xuất tối đa một candidate.
- Mọi candidate được xuất đều tồn tại trong KB ICD-10/RxNorm đi kèm.
- Mọi unit test và integration test local đều pass.
- Mô hình đã lưu có thể reload và tái tạo inference đúng schema.
- `output.zip` và model archive vượt qua kiểm tra file inventory và CRC.
- Chỉ tuyên bố thành công trên Kaggle sau khi một lần `Run All` thực tế đã được audit.

## 14. Ngoài phạm vi

- Coi GT được khôi phục của ID 1–100 là nhãn huấn luyện đáng tin cậy.
- Thay NER bằng danh sách phrase regex.
- Train Qwen làm bộ phát hiện entity chính.
- Gán candidate ICD-10/RxNorm cho triệu chứng hoặc entity xét nghiệm khi contract cuộc thi không yêu cầu.
- Viết lại tài liệu gốc hoặc dùng offset của văn bản đã chuẩn hóa trong submission.
- Tuyên bố điểm validation cuối không thiên lệch sau khi đã fit trên toàn bộ 100 tài liệu organizer đáng tin cậy.
- Mở rộng sang relation extraction trước khi các thành phần entity, assertion và linking ổn định.

## 15. Quan hệ với các đặc tả hiện có

Tài liệu này tinh chỉnh và được ưu tiên áp dụng cho curriculum training, xử lý metadata, chunk supervision, assertion và khôi phục KB-first. Các đặc tả sau vẫn có hiệu lực ở những điểm không mâu thuẫn với tài liệu này:

- `docs/superpowers/specs/2026-07-22-kaggle-end-to-end-clinical-pipeline-design.md`;
- `docs/superpowers/specs/2026-07-22-synthetic-train-v2-design.md`.

Đặc tả này không yêu cầu sinh lại corpus synthetic v2. Chỉ thay đổi dataset khi validation hoặc semantic audit phát hiện một lỗi cụ thể.

## 16. Tự đánh giá thiết kế sau khi phản biện

### 16.1. Các vấn đề đã được xử lý

| Vấn đề | Quyết định trong spec |
| --- | --- |
| Holdout 20 tài liệu dễ overfit và splitter có thể không tạo được split | Grouped 5-fold OOF, hard-group chỉ cho duplicate/near-duplicate/record cùng nguồn, fail-closed nếu không tạo đủ fold |
| Context có thể chảy giữa nhiều bệnh nhân trong một file | `record_spans`, `patient_block_id`, chunk/assertion/merge không vượt high-confidence boundary |
| GT candidate rỗng bị hiểu nhầm là lỗi | Candidate rỗng được định nghĩa là abstention hợp lệ |
| Runtime KB thiếu gold candidate hoặc làm mất dạng ICD display | Coverage 100% trước train và tách `canonical_id`/`official_display_id` |
| Audit report stale và số liệu mâu thuẫn | Fingerprint, trạng thái report và invalidation stale |
| Khẳng định “tối ưu metric” khi chỉ có proxy evaluator | Gắn nhãn proxy, OOF mean/std/worst-fold, baseline/ablation bắt buộc |
| KB-first tăng recall nhưng có thể làm tăng false positive | Đo incremental precision/recall ở cấp mention và lọc ambiguity/boundary |
| Yêu cầu agent verify chưa đủ mạnh | Hai agent độc lập audit bắt buộc trước train chính thức |

### 16.2. Các rủi ro chấp nhận được

| Rủi ro | Lý do chấp nhận | Cách theo dõi |
| --- | --- | --- |
| Chưa có evaluator chính thức của BTC | Chưa có nguồn chuẩn trong workspace hiện tại | Ghi rõ proxy metric trong report; thay evaluator khi BTC cung cấp |
| 5-fold OOF làm tăng chi phí GPU | 100 tài liệu BTC ít, Stage 1 được cache và chỉ Stage 2–3 chạy theo fold | Log runtime/peak memory từng fold; chỉ fallback 3-fold khi có phê duyệt và được ghi rõ trong experiment manifest |
| OOF threshold có thể lệch nhẹ so với final model | Final-fit dùng lịch và threshold đã đóng băng, không tự calibration in-sample | Theo dõi calibration drift ở smoke test và không tuyên bố score unbiased |
| Qwen/semantic retrieval có thể không chạy trên Kaggle | Đây là nhánh tùy chọn | Deterministic lexical fallback và warning bắt buộc |

### 16.3. Kết luận tự đánh giá

Sau các bổ sung trên, không còn lỗ hổng HIGH đã biết trong thiết kế dữ liệu, split, chunking, candidate contract, audit hoặc chọn metric. Spec đủ chặt để lập implementation plan. Tuy nhiên, spec chưa phải bằng chứng pipeline đã hoạt động: vẫn cần triển khai, chạy toàn bộ test, xác nhận candidate coverage thực tế và audit một phiên Kaggle `Run All` trước khi tuyên bố hoàn thành.
