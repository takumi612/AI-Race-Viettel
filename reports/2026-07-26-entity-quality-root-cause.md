# Báo cáo root cause và hardening chất lượng entity — 2026-07-26

## Kết luận

Điểm `11.6164` không còn thấp vì notebook chạy smoke mode hoặc thiếu epoch. Notebook đã chạy đủ Stage 1/2/3/final fit với lần lượt `5/3/6/3` epoch, nhưng WER vẫn là `87.0806`. Nguyên nhân chính là pipeline tối ưu sai metric và không có lớp precision control khi suy luận trên dữ liệu khác miền.

## Bằng chứng từ lần chạy đã chấm

- 100/100 file JSON và CRC của ZIP hợp lệ.
- 4.245 entity, trung bình 42,45 entity/hồ sơ.
- Entity phủ trung bình 41,1757% toàn bộ raw text.
- 969 entity chỉ có một token.
- 93 span là từ chức năng/generic như `và`, `không`, `lúc`, `khoa`, `thể`, `phẫu`, `xét`.
- Chỉ 305/2.352 entity thuộc loại cần ICD/RxNorm có candidate, tương đương 12,9677%.
- Assertion `isHistorical` xuất hiện 1.653 lần, cao bất thường so với phân phối train.
- Offset, schema và cấu trúc ZIP đều đúng. Đây là lỗi ngữ nghĩa/model selection, không phải lỗi đóng gói.

Quality gate mới chạy lại trên chính ZIP này cho kết quả:

```text
REJECT: generic_spans=93
```

## Root causes đã xác nhận

### 1. Checkpoint được chọn bằng token-level BIO F1

Stage 2/3 báo metric gần `0.9998`, nhưng metric cũ chỉ so từng token. Nó không phản ánh đầy đủ việc span bị cắt sai boundary, tạo thêm entity hoặc gán nhầm type — các lỗi bị chấm rất nặng theo WER/entity matching của cuộc thi.

### 2. Final fit không có validation

Lần chạy cũ có `validation_chunks=0`, `best_metric=null`, `best_checkpoint=null`. Final fit đưa cả validation vào train, sau đó xuất checkpoint cuối mà không kiểm tra khả năng tổng quát hóa.

### 3. NER inference không có confidence threshold

`TransformerNERDetector` lấy argmax của mọi token và phát hành mọi span BIO, kể cả span confidence thấp. Domain shift biến các từ thông thường thành entity giả và không có calibration nào ngăn chúng đi vào submission.

### 4. KB-first recovery tin mọi alias với confidence 0.99

Alias ngắn hoặc mơ hồ như `tin`, `ho`, `phẫu`, `bệnh` có thể tạo entity trực tiếp. Nhiều code dùng chung alias còn tạo nhiều proposal chồng nhau; proposal đầu tiên thắng theo thứ tự thay vì được rerank có chủ đích.

Ngoài ra, span do NER phát hiện không được đưa qua candidate linker. Chỉ span do KB exact-scan tự tạo mới có ranked pool, nên Qwen không thể rerank phần lớn entity `DISEASE/DRUG`.

### 5. Dữ liệu synthetic bị template/surface collapse

- 2.000 tài liệu synthetic chỉ có khoảng 962 surface entity duy nhất.
- 100 organizer GT có 3.404 surface duy nhất nhưng chỉ 93 surface giao với synthetic.
- Output có 2.705 surface duy nhất; 2.657 chưa xuất hiện trong synthetic lẫn organizer GT.
- Vì vậy validation cùng phân phối/template rất dễ đạt gần tuyệt đối nhưng không đại diện cho input vòng thi dạng hỏi đáp và bệnh án tự nhiên.

## Thay đổi đã triển khai

### Entity-level checkpoint selection

- Thêm exact typed BIO span precision/recall/F1 trên validation windows.
- Chọn checkpoint bằng `entity_f1`, không dùng token F1 làm objective chính.
- Tìm confidence threshold trên validation; nếu F1 hòa, chọn threshold cao hơn.
- Ghi artifact versioned `ner_calibration.json` cạnh checkpoint.

### Validation-safe final fit

- Final fit chỉ train trên train partitions.
- Organizer validation và synthetic validation tiếp tục được giữ riêng.
- Final checkpoint có `best_metric`, `best_checkpoint` và calibration thực.

### Precision control ở inference

- Checkpoint mới dùng threshold đã calibrate.
- Legacy checkpoint không có artifact dùng fallback bảo thủ `0.85`.
- Artifact sai schema hoặc threshold ngoài `[0,1]` làm pipeline dừng.

### KB recovery an toàn

- Loại alias một token quá ngắn/generic.
- Loại các analyte dễ bị RxNorm gán nhầm thành thuốc: `glucose`, `creatinine`, `protein`, `prothrombin`.
- Alias chung nhiều code được gom thành một ranked candidate pool.
- Candidate mơ hồ được giữ cho Qwen rerank; deterministic policy không tự chọn bừa.
- Mọi span `DISEASE/DRUG` từ NER được đưa qua lexical candidate ranking trước policy/Qwen.
- Candidate pool luôn có display `name` hợp lệ cho Qwen prompt; thiếu tên không còn làm required-Qwen phase crash.

### Semantic output gate

Diagnostics mới gồm:

- `single_token_entity_count`
- `suspicious_generic_span_count`
- `candidate_eligible_count`
- `candidate_linked_entity_count`
- `candidate_link_rate`
- `ner_confidence_threshold`

Output collapse có từ ba generic span trở lên và chiếm ít nhất 1% entity sẽ bị dừng trước khi tạo submission ZIP.

## Qwen

Qwen vẫn chỉ được điều khiển bằng đúng một cờ trong notebook:

```python
ENABLE_QWEN_RERANKER = False
```

Đổi thành `True` để chạy Qwen. Không có config thứ hai và `config.json` không thể ghi đè. Khi bật, lỗi init/generation/JSON/cleanup đều làm phase thất bại thay vì fallback im lặng.

## Verification

```text
425 passed, 2 skipped
Notebook: 29 cells, 15 code cells, 13 phases, syntax valid
```

## Việc bắt buộc để có điểm mới

Code và notebook đã được harden, nhưng không thể tính điểm hidden-test tại local vì không có ground truth của 100 hồ sơ vòng thi. Cần chạy lại notebook canonical trên Kaggle GPU, tải `output.zip` mới và nộp lại. Trước khi nộp, xác nhận:

1. `fast_dev_run` là `False`.
2. `validation_chunks > 0` ở final fit.
3. `best_metric` là entity F1 và không còn gần 1.0 một cách phi thực tế.
4. `ner_confidence_threshold` xuất hiện trong diagnostics.
5. `suspicious_generic_span_count` không kích hoạt gate.
6. Nếu dùng Qwen, đặt duy nhất `ENABLE_QWEN_RERANKER=True` và kiểm tra `qwen_status=COMPLETED`.
