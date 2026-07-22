# Thiết kế huấn luyện theo curriculum và metadata-first

## 1. Mục đích

Tài liệu này đặc tả thiết kế đích cho quá trình huấn luyện và suy luận của pipeline trích xuất thông tin y khoa trong cuộc thi AI Race Viettel. Thiết kế mở rộng pipeline Kaggle end-to-end hiện tại bằng bốn cải tiến hướng đến tối ưu metric:

1. huấn luyện NER ba giai đoạn có nhận biết nguồn dữ liệu;
2. xử lý ngữ cảnh theo hướng metadata-first mà không thay đổi văn bản gốc;
3. giám sát owner-window an toàn tại biên chunk;
4. mô hình phân loại assertion độc lập và cơ chế khôi phục entity theo hướng KB-first.

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

Trong nhãn BTC đã quan sát, mỗi entity chẩn đoán hoặc thuốc có tối đa một candidate ở output cuối. Vì vậy, cấu hình submission mặc định vẫn là `candidate_output_k = 1`. Giới hạn này không áp dụng cho bước retrieval: tập candidate nội bộ vẫn có thể cấu hình, mặc định lấy tối đa 20 candidate trước khi lọc và rerank.

Evaluator local chỉ là công cụ đại diện để phát triển, chưa được xác nhận là mã chấm chính thức của BTC. Token-level F1 và công thức tổng hợp 30/30/40 hiện tại vẫn được báo cáo để chẩn đoán, nhưng việc chọn mô hình phải ưu tiên chất lượng entity ở cấp tài liệu và báo cáo riêng chất lượng assertion cùng entity linking.

## 3. Chính sách dữ liệu

Corpus chuẩn là `data_v2/Training_data/synthetic_train_v2`.

| ID | Nguồn gốc | Vai trò |
| --- | --- | --- |
| 1–100 | Input của BTC với GT được khôi phục/tự sinh | Chỉ cách ly và audit; không dùng để fit mô hình hoặc hiệu chỉnh threshold |
| 101–200 | Input của BTC với GT bị leak do BTC cung cấp | Corpus thật có nhãn đáng tin cậy |
| 201–2200 | Synthetic v2 | Tăng độ phủ, warm-up biểu diễn, khái niệm hiếm, đa dạng thể loại và regularization |

Không loại bỏ 2.000 tài liệu synthetic. Tất cả phải đủ điều kiện tham gia Stage 1 và Stage 2. Stage 3 chỉ dùng một tập replay đã chọn để bước thích nghi cuối bám theo ngôn ngữ của BTC thay vì văn phong synthetic.

Các file input và GT gốc không được thay đổi trong quá trình huấn luyện. Metadata dẫn xuất, split, feature và kết quả calibration được lưu thành artifact sidecar.

### 3.1. Development split

Việc chọn hyperparameter và checkpoint sử dụng một development split cố định:

- 80 tài liệu thật đáng tin cậy để huấn luyện;
- 20 tài liệu thật đáng tin cậy làm `real_holdout`;
- một phần synthetic dùng để huấn luyện và một `synthetic_holdout` được chia theo group để chẩn đoán.

Quá trình chia dữ liệu phải nhận biết group. Document ID, template family, nhóm surface chính đã chuẩn hóa và nhóm near-duplicate không được xuất hiện ở cả train lẫn holdout. Trong giới hạn của các ràng buộc group, real split cần bảo toàn độ phủ theo loại entity và nhãn assertion tốt nhất có thể.

Real holdout là tập chính để chọn mô hình. Synthetic holdout chỉ dùng để chẩn đoán độ phủ và hiện tượng ghi nhớ; kết quả trên synthetic holdout không được phép lấn át sự suy giảm trên real holdout.

### 3.2. Quy trình final-fit

Sau khi chọn được curriculum, threshold và các hyperparameter khác, quá trình final-fit sử dụng toàn bộ 100 tài liệu thật đáng tin cậy. Điểm số được tạo trên chính 100 tài liệu này sau final-fit không được báo cáo như một kết quả validation không thiên lệch.

Lần chạy final-fit sử dụng các quyết định đã được chọn và đóng băng trong giai đoạn development:

- số epoch đã chọn, điểm kết thúc từng stage và learning rate;
- tỷ lệ sampling;
- threshold assertion;
- threshold và margin của linking.

Final-fit không early-stop hoặc chọn checkpoint dựa trên 100 tài liệu đang được dùng để fit. Nó chạy đúng lịch huấn luyện đã chọn ở development và lưu trạng thái cuối của stage theo cấu hình.

Artifact cuối phải lưu cả metric của development split và thông tin rằng mô hình được giao đã được fit tiếp bằng toàn bộ 100 nhãn thật đáng tin cậy.

## 4. Mô hình ngữ cảnh metadata-first

Không viết lại văn bản y khoa gốc vì offset trong output phải tham chiếu trực tiếp đến tài liệu ban đầu. Ngữ cảnh được biểu diễn trong sidecar record với các trường sau khi có thể phát hiện:

- `document_id`;
- `source_bucket`: `quarantine`, `organizer` hoặc `synthetic`;
- `document_genre`;
- `template_group` và `near_duplicate_group`;
- span của section và loại section đã chuẩn hóa;
- vai trò người nói hoặc người viết;
- chủ thể/experiencer, ví dụ bệnh nhân hoặc người nhà;
- cờ long-tail và ontology coverage;
- content hash và phiên bản bộ sinh dữ liệu.

Cho phép dùng rule nhận diện tiêu đề section vì chúng xác định cấu trúc tài liệu. Regex nội dung cho bệnh, thuốc hoặc triệu chứng không được dùng làm bộ sinh nhãn chính. Bộ phân tích giá trị xét nghiệm có thể được giữ làm lớp làm giàu dữ liệu có cấu trúc, với điều kiện kết quả của nó không bị coi là ground truth mặc định.

Metadata có thể được sử dụng cho:

- chia dữ liệu theo group;
- sampling có nhận biết nguồn;
- xác định ngữ cảnh assertion;
- chẩn đoán prediction;
- token ngữ cảnh đặc biệt tùy chọn, chỉ khi ablation trên `real_holdout` chứng minh có cải thiện.

Baseline không chèn metadata token vào encoder. Quyết định này giúp giữ nguyên offset và tránh biến metadata được phát hiện chưa chính xác thành đầu vào bắt buộc của mô hình.

## 5. Chunking an toàn tại biên

Cửa sổ tokenizer mặc định:

- `max_length = 512` token, bao gồm special token;
- overlap/stride mục tiêu là 128 token.

Mỗi feature phải giữ document ID, raw character offset, biên cửa sổ và metadata nguồn.

### 5.1. Giám sát owner-window

Mỗi gold entity được gán cho đúng một owner window. Window hợp lệ phải chứa trọn vẹn span entity. Owner là window hợp lệ làm cực đại khoảng cách token nhỏ hơn giữa entity với hai biên trái/phải có thể sử dụng. Nếu bằng nhau, chọn window có index nhỏ hơn.

Nhãn huấn luyện tuân theo các quy tắc:

1. owner window nhận đầy đủ nhãn BIO của entity;
2. các bản sao của entity trong những window overlap khác được mask bằng `-100` khi tính loss;
3. window chỉ chứa một phần entity sẽ mask các token entity nhìn thấy bằng `-100`;
4. token không thuộc entity vẫn là nhãn `O` hợp lệ, trừ khi bị mask bởi quy tắc special token hoặc padding của tokenizer;
5. nếu không có window nào chứa trọn entity, preprocessing phải dừng đối với tài liệu đó và ghi diagnostic, thay vì âm thầm chuyển entity thành `O`.

Cơ chế này đảm bảo mỗi entity chỉ đóng góp vào loss một lần, overlap không làm thay đổi tần suất lớp và không tạo supervision sai ở biên.

### 5.2. Hợp nhất khi inference

Prediction được chiếu về raw character offset. Các prediction trùng hoàn toàn được gộp trước. Span cùng loại và overlap được hợp nhất bằng confidence đã calibration và độ đầy đủ tại biên. Xung đột khác loại được xử lý tất định theo độ tin cậy của nguồn, confidence và độ đầy đủ của span; mọi phương án bị loại phải được ghi vào diagnostic.

## 6. Curriculum NER ba giai đoạn

Backbone NER tiếp tục là XLM-R, trừ khi một thí nghiệm có kiểm soát trên real holdout chứng minh mô hình khác tốt hơn. Hai mươi epoch là giới hạn an toàn tuyệt đối, không phải số epoch mục tiêu.

### 6.1. Stage 1: warm-up bằng synthetic

Mục tiêu: học năm loại entity, biến thể từ vựng rộng, thể loại tài liệu y khoa, surface ICD/RxNorm hiếm và hành vi tại biên.

- Dữ liệu: phần synthetic train ở development; toàn bộ 2.000 tài liệu synthetic ở final-fit.
- Giới hạn epoch: 3, khoảng dự kiến 1–3.
- Khoảng learning rate ban đầu: `2e-5` đến `3e-5`.
- Tín hiệu lựa chọn ở development: metric entity trên synthetic và điều kiện không suy giảm trên real holdout.
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

Checkpoint ở development được chọn bằng prediction cấp tài liệu trên `real_holdout`:

`selection_score = 0.70 * exact_entity_f1 + 0.20 * overlap_entity_f1 + 0.10 * macro_type_f1`

Yêu cầu:

- exact-span F1 được tính micro-average trên các entity;
- overlap F1 sử dụng ghép cặp một-một và một overlap threshold được ghi rõ;
- macro type F1 là trung bình đều của năm loại entity;
- token-level F1 chỉ phục vụ chẩn đoán;
- patience áp dụng trên selection score, không phải training loss;
- checkpoint không được chọn nếu validation schema hoặc offset thất bại.

Metric theo từng loại, từng genre, surface đã thấy/chưa thấy và số lỗi boundary phải được lưu để phân tích.

## 7. Mô hình assertion

Assertion được xử lý bởi một classifier multi-label riêng, thay vì regex nội dung diện rộng.

Với mỗi entity được phát hiện hoặc gold entity, classifier nhận:

- cửa sổ ngữ cảnh cục bộ có đánh dấu entity;
- metadata section và experiencer nếu có;
- loại entity;
- ngữ cảnh câu/mệnh đề gốc.

Mô hình dự đoán xác suất độc lập cho `isNegated`, `isHistorical` và `isFamily`. Threshold của từng nhãn được calibration riêng trên real holdout đáng tin cậy. Ví dụ assertion synthetic cung cấp regularization và độ phủ, nhưng ví dụ organizer nhận trọng số sampling cao hơn.

Để phù hợp tài nguyên Kaggle, assertion classifier có thể tái sử dụng một bản encoder NER đã freeze hoặc dùng classification head/adapter nhẹ. Phương án được chọn dựa trên assertion F1 ở real holdout và chi phí runtime.

Rule chỉ được dùng làm fallback cấu trúc có precision cao và phục vụ diagnostic. Rule không được suy luận assertion từ keyword không giới hạn phạm vi trên toàn tài liệu. Mọi cue của rule phải bị giới hạn trong mệnh đề/câu và section tương ứng.

## 8. Ontology retrieval, khôi phục entity và linking

Quá trình phát hiện entity có hai đường recall độc lập:

1. prediction của NER;
2. phrase/alias retrieval theo hướng KB-first trên ICD-10 và RxNorm.

Nhánh KB-first tìm alias chuẩn hóa khớp chính xác trước, sau đó mới xét phương án lexical/fuzzy hoặc semantic có kiểm soát. Nhánh này có thể đề xuất mention bệnh hoặc thuốc mà NER bỏ sót, nhưng không được trực tiếp ép mention vào output. Proposal phải qua xác minh span, kiểm tra type, confidence threshold, giải quyết xung đột và candidate calibration.

Với mỗi mention chẩn đoán hoặc thuốc được chấp nhận:

1. lexical và semantic retrieval tạo tối đa 20 candidate nội bộ;
2. phần head của mention thuốc được tách khỏi hàm lượng, đường dùng, dạng bào chế và tần suất;
3. bộ lọc tất định loại các entry sai type hoặc không hợp lệ trong KB;
4. reranker sắp xếp tập candidate còn lại;
5. áp dụng minimum score, top-1/top-2 margin và abstention threshold;
6. xuất tối đa một candidate.

Chất lượng retrieval được đo riêng bằng recall@1, recall@5, recall@10, top-1 accuracy, coverage và accuracy có điều kiện trên các trường hợp không abstain. Threshold được chọn trên real holdout và đóng băng trước final-fit.

Qwen là thành phần tùy chọn, chỉ dùng để rerank candidate mơ hồ hoặc fallback cho assertion. Qwen chỉ được chọn từ candidate hợp lệ đã cung cấp. Thiếu weights, JSON sai, timeout hoặc lỗi CUDA phải quay về hành vi tất định và không được làm dừng pipeline.

## 9. Điều phối notebook Kaggle

Notebook chuẩn tiếp tục là `v2/medical_information_extraction_kaggle.ipynb` và hỗ trợ:

- `RUN_MODE = "full"`: validate dữ liệu, train mọi stage, calibration các thành phần development khi phù hợp, chạy inference và đóng gói artifact;
- `RUN_MODE = "resume"`: tiếp tục từ stage hoàn tất gần nhất có manifest và checkpoint vượt qua kiểm tra toàn vẹn;
- `RUN_MODE = "inference_only"`: load artifact cuối và tạo output cuộc thi mà không train.

Notebook thực hiện tuần tự các pha logic:

1. khám phá môi trường, dataset và KB;
2. validation tất định và tạo manifest;
3. trích xuất metadata và tạo development split theo group;
4. tạo feature owner-window;
5. Stage 1 warm-up bằng synthetic;
6. Stage 2 huấn luyện trộn cân bằng theo nguồn;
7. Stage 3 thích nghi với organizer;
8. train và calibration assertion;
9. linking retrieval và calibration threshold;
10. chạy final-fit khi được bật;
11. smoke test reload checkpoint;
12. inference, validate output và đóng gói artifact.

Mỗi stage hoàn tất phải ghi một stage manifest theo cách atomic. Chỉ được resume khi manifest khớp dataset hash, phiên bản code/config, định danh tokenizer, label mapping và checkpoint inventory.

## 10. Artifact và khả năng quan sát

Lần chạy tối thiểu phải tạo:

- `stage1_synthetic_checkpoint/`;
- `stage2_mixed_checkpoint/`;
- `final_ner_model/`;
- `assertion_model/`;
- `candidate_calibration.json`;
- `metadata_manifest.jsonl`;
- `split_manifest.json`;
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

## 11. Xử lý lỗi

Pipeline phải dừng sớm trong các trường hợp:

- cặp input/GT không hợp lệ;
- lỗi schema hoặc offset;
- candidate không tồn tại trong KB đi kèm;
- rò rỉ group giữa train và holdout;
- entity không thể nằm trọn trong bất kỳ window nào theo cấu hình;
- checkpoint được chọn không thể load;
- resume manifest không khớp;
- output sai định dạng hoặc cấu trúc ZIP không hợp lệ.

Nếu semantic retrieval hoặc Qwen tùy chọn bị lỗi, pipeline hạ cấp về hành vi lexical/tất định và ghi warning. Các lỗi tùy chọn này không làm vô hiệu một lần chạy vốn đáp ứng đúng contract.

## 12. Chiến lược xác minh

Quá trình triển khai tuân theo test-driven development. Các tầng kiểm thử bắt buộc:

1. **Unit test**: phân tích metadata, gán owner-window, mask partial span, source-aware sampling, metric cấp tài liệu, assertion threshold, cắt candidate, abstention và fallback tất định.
2. **Integration test**: bàn giao checkpoint giữa ba stage, tính toàn vẹn khi resume, reload mô hình, inference end-to-end, schema output, offset và candidate hợp lệ.
3. **Data test**: không có lỗi pairing/schema/offset/KB, không có split leakage, đúng số lượng theo nguồn và đúng phân bố replay.
4. **Fast development smoke test**: dataset local nhỏ hoàn tất mọi stage logic mà không tải mô hình lớn.
5. **Kaggle acceptance run**: một lần `Run All` thực tế trên GPU tạo weights có thể reload và các archive output vượt qua kiểm tra CRC.

Các agent review độc lập có thể audit ngữ nghĩa nhãn và thay đổi triển khai khi được yêu cầu. Kết quả review mang tính tư vấn; kiểm tra tất định và nhãn organizer đáng tin cậy vẫn là nguồn chuẩn.

## 13. Tiêu chí nghiệm thu

Thiết kế chỉ được coi là triển khai thành công khi thỏa mãn toàn bộ điều kiện:

- ID 1–100 bị loại khỏi fitting và calibration.
- Toàn bộ 2.000 tài liệu synthetic đủ điều kiện tham gia Stage 1 và Stage 2.
- Mô hình cuối sử dụng toàn bộ 100 tài liệu có nhãn organizer đáng tin cậy sau khi các quyết định development đã được đóng băng.
- Không gold entity nào đóng góp NER loss trong nhiều hơn một window overlap.
- Partial entity không bao giờ bị chuyển thành supervision `O`.
- Việc chọn mô hình sử dụng metric entity cấp tài liệu trên real holdout.
- Assertion threshold và candidate abstention threshold được calibration trên real holdout đáng tin cậy.
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
