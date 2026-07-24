# Thiết kế notebook Kaggle chỉ suy luận từ dataset kết quả

## Mục tiêu

Tạo một notebook Kaggle mới bằng tiếng Việt, nạp checkpoint NER và các
artifact runtime từ dataset kết quả đã attach, xử lý một dataset input riêng,
sau đó tạo file nộp bài mới mà không thực hiện bất kỳ bước huấn luyện nào.

Notebook train hiện tại `train-ai-race-v2-32-8.ipynb` phải được giữ nguyên.

## Sản phẩm bàn giao

- `train-ai-race-v2-32-8-inference-only.ipynb`
- `KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md`

Notebook và runbook đều dùng tiếng Việt cho nội dung hướng dẫn người dùng.
Tên biến, API Python, đường dẫn và thông báo kỹ thuật có thể giữ tiếng Anh khi
điều đó giúp khớp với thư viện hoặc log Kaggle.

Quy trình sử dụng cuối cùng:

1. Attach dataset kết quả.
2. Attach dataset input mới.
3. Bật GPU và Internet trong Kaggle.
4. Import notebook rồi chọn **Run All**.
5. Tải `/kaggle/working/output.zip`.

## Dataset kết quả trên Kaggle

Kaggle có thể cung cấp dữ liệu theo một trong hai dạng:

1. Thư mục đã giải nén, có:
   - `training_artifacts/ner_model/`
   - `AI-Race-Viettel/v2/clinical_nlp_lab/`
   - `AI-Race-Viettel/v2/artifacts/`
2. File `results.zip` chứa các đường dẫn trên.

Notebook ưu tiên cây thư mục đã giải nén hoàn chỉnh. Nếu chỉ có
`results.zip`, notebook chỉ giải nén checkpoint, mã runtime và knowledge base
cần thiết vào `/kaggle/working/inference_runtime`.

Checkpoint hợp lệ tối thiểu phải có:

- `model.safetensors`
- `config.json`
- các file tokenizer mà Hugging Face Transformers có thể nạp

Runtime còn cần package `clinical_nlp_lab` và artifact ICD-10/RxNorm đi kèm.

## Dataset input mới

Notebook hỗ trợ cả:

1. Thư mục `input/` đã giải nén, chứa ít nhất một file
   `<document-id>.txt`.
2. File `input.zip` chứa thư mục `input/` hoặc các file `.txt`.

Cơ chế tự tìm kiếm phải loại trừ dataset kết quả, dữ liệu train, output cũ,
diagnostics, checkpoint và thư mục làm việc. Hai biến override ở đầu notebook
cho phép chỉ định đường dẫn chính xác khi Kaggle có nhiều ứng viên.

Không cần attach annotation hoặc ground truth.

## Phương án được chọn

Notebook dùng cơ chế **tự phát hiện hai dạng dữ liệu**: ưu tiên thư mục Kaggle
đã giải nén nhưng vẫn hỗ trợ ZIP dự phòng. Cách này bền vững hơn hard-code tên
dataset và không phụ thuộc vào việc Kaggle có tự giải nén archive hay không.

## Luồng chạy

1. Khởi tạo cấu hình, logging, đường dẫn Kaggle và biến override.
2. Tìm và xác thực đúng một nguồn kết quả.
3. Dùng trực tiếp cây đã giải nén hoặc chỉ giải nén các thành phần runtime cần
   thiết.
4. Đưa package runtime vào `sys.path`.
5. Tìm và xác thực đúng một nguồn input mới.
6. Dùng trực tiếp thư mục `input/` hoặc giải nén `input.zip` vào thư mục làm
   việc tách biệt.
7. Chỉ cài các dependency inference còn thiếu.
8. Nạp checkpoint NER đã train và các artifact retrieval/linking.
9. Chạy pipeline suy luận cho từng văn bản.
10. Xác thực JSON đầu ra và tạo
    `/kaggle/working/output.zip`.
11. Ghi manifest gồm nguồn dữ liệu, checkpoint, số input, số output và
    `training_skipped: true`.

Notebook không được gọi `Trainer`, `.train()`,
`train_ner_subprocess.py`, không chia train/validation và không đóng gói
checkpoint mới.

## Quy ước output

Archive cuối cùng có đúng một JSON cho mỗi file input:

```text
output.zip
└── output/
    ├── <document-id-1>.json
    └── <document-id-2>.json
```

Notebook không sao chép `output.zip` hoặc thư mục `output/` lịch sử từ
dataset kết quả. Mỗi lần chạy luôn tạo dự đoán mới cho dataset input vừa
attach.

## Nội dung runbook

`KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md` phải hướng dẫn bằng tiếng Việt:

1. Cách tạo/upload dataset chứa `results.zip` hoặc cây đã giải nén.
2. Cách tạo/upload dataset chứa `input.zip` hoặc `input/*.txt`.
3. Cách import notebook và attach cả hai dataset bằng **Add Input**.
4. Cách bật GPU, bật Internet và chọn **Run All**.
5. Cách kiểm tra `run_manifest.json` có
   `"training_skipped": true`.
6. Cách lưu version và tải `output.zip`.
7. Cách dùng biến override khi có nhiều dataset.
8. Cách xử lý lỗi thiếu checkpoint, không tìm thấy input, thiếu dependency,
   CUDA OOM, sai schema và nhầm output cũ.

## Xử lý lỗi và an toàn

Notebook dừng sớm với thông báo rõ ràng khi:

- không tìm thấy nguồn kết quả hoàn chỉnh;
- có nhiều nguồn kết quả nhưng chưa dùng override;
- thiếu model, tokenizer, runtime package hoặc knowledge base;
- không tìm thấy input hợp lệ;
- có nhiều nguồn input nhưng chưa dùng override;
- input trỏ vào results, train, diagnostics, checkpoint, output hoặc working;
- document ID trùng nhau;
- prediction JSON sai schema;
- tên hoặc số lượng output không khớp input.

Giải nén ZIP phải từ chối đường dẫn tuyệt đối và thành phần `..`, đồng thời
chỉ giải nén các prefix dự kiến.

## Kiểm thử và xác minh

Xác minh local phải kiểm tra:

- notebook là JSON notebook hợp lệ;
- mọi code cell đều compile;
- output và execution count trong notebook đã được xóa;
- không code cell nào gọi train hoặc đóng gói checkpoint;
- có cả nhánh thư mục đã giải nén và nhánh ZIP;
- có kiểm tra model, runtime, knowledge base, input và output;
- có kiểm tra cấu trúc ZIP và một-output-cho-mỗi-input;
- nội dung hiển thị của notebook và runbook là tiếng Việt;
- runbook bao phủ đầy đủ quy trình Kaggle;
- notebook train gốc không thay đổi.

Smoke test dùng fixture thư mục/ZIP nhỏ cho logic discovery và extraction.
Inference đầy đủ được kiểm chứng trên Kaggle vì checkpoint thật khoảng
1,1 GB và cần GPU/runtime mục tiêu.
