# Hướng dẫn chạy Kaggle inference-only

Notebook sử dụng:
`train-ai-race-v2-32-8-inference-only.ipynb`

Notebook này chỉ nạp checkpoint đã train và chạy suy luận. Notebook không
chia dữ liệu train, không gọi `Trainer`, không fine-tune và không tạo
checkpoint mới.

## 1. Chuẩn bị dataset kết quả

Dataset trong ảnh của bạn đã đúng cấu trúc. Sau khi Kaggle giải nén, notebook
chấp nhận cây thư mục:

```text
/kaggle/input/<results-dataset-slug>/
└── results/
    ├── artifacts/
    │   ├── config.json
    │   ├── entity_type_mapping.json
    │   ├── assertion_mapping.json
    │   ├── relation_mapping.json
    │   ├── thresholds.json
    │   ├── icd10/
    │   │   └── icd10_dictionary.jsonl.gz
    │   └── rxnorm/
    │       ├── rxnorm_dictionary.jsonl.gz
    │       └── rxnorm_relations.jsonl.gz
    └── training_artifacts/
        └── ner_model/
            ├── config.json
            ├── model.safetensors
            ├── tokenizer.json
            └── tokenizer_config.json
```

Dataset không cần chứa thư mục `AI-Race-Viettel` hoặc source code. Notebook
tự clone source code đã kiểm chứng từ GitHub vào
`/kaggle/working/AI-Race-Viettel`.

Notebook cũng hỗ trợ dataset chứa nguyên file `results.zip`. Trong trường hợp
đó, archive cần chứa `artifacts/` và
`training_artifacts/ner_model/` ở cấp gốc.

Kaggle đôi khi tự giải nén file `.jsonl.gz` thành `.jsonl`. Notebook có bước
chuẩn hóa để nén lại các knowledge base này trước khi chạy pipeline.

## 2. Chuẩn bị dataset input mới

Tạo một Kaggle Dataset riêng chỉ chứa dữ liệu cần dự đoán. Có thể dùng một
trong hai cấu trúc:

```text
input/
├── 1.txt
├── 2.txt
└── 3.txt
```

hoặc:

```text
input.zip
└── input/
    ├── 1.txt
    ├── 2.txt
    └── 3.txt
```

Mỗi document ID phải duy nhất. Không đặt dữ liệu train, ground truth,
diagnostics hoặc output cũ trong dataset input.

## 3. Import notebook và attach dataset

1. Vào **Kaggle → Code → New Notebook**.
2. Chọn **File → Import Notebook**.
3. Upload `train-ai-race-v2-32-8-inference-only.ipynb`.
4. Trong panel bên phải, chọn **Add Input**.
5. Attach dataset kết quả chứa cây `results/` hoặc `results.zip`.
6. Chọn **Add Input** lần nữa và attach dataset input mới chứa
   `input/` hoặc `input.zip`.
7. Không attach dataset train cũ nếu không cần thiết, vì có thể tạo nhiều ứng
   viên input và khiến notebook chủ động dừng để tránh chọn nhầm.

## 4. Cấu hình Kaggle

Trong **Settings**:

- **Accelerator:** có thể để **None** để chạy CPU.
- **Internet:** bật **ON** để clone code và cài dependency retrieval còn
  thiếu.
- Chọn **Run All**, không chạy bắt đầu từ cell giữa notebook.

Notebook mặc định chạy CPU và không tải Qwen:

```python
ENABLE_QWEN_RERANKER = False
INSTALL_VLLM = ENABLE_QWEN_RERANKER
REQUIRE_GPU = False
```

Log `CUDA unavailable; CPU mode enabled` là thông báo bình thường, không phải
lỗi.

Notebook clone source code trực tiếp từ nhánh đã push:

```text
codex/kaggle-end-to-end-pipeline
```

Notebook không dùng mã commit pin. Vì vậy cần bảo đảm nhánh này đã được push
lên GitHub trước khi chạy Kaggle.

## 5. Chỉ định đường dẫn khi notebook thấy nhiều dataset

Thông thường giữ nguyên:

```python
PROJECT_ROOT_OVERRIDE = ""
RESULTS_ZIP_OVERRIDE = ""
INPUT_SOURCE_OVERRIDE = ""
```

Nếu có nhiều dataset, đặt đường dẫn tuyệt đối trong cell cấu hình đầu tiên.
Mặc dù tên biến là `RESULTS_ZIP_OVERRIDE`, biến này nhận cả thư mục đã giải
nén:

```python
RESULTS_ZIP_OVERRIDE = "/kaggle/input/<results-dataset-slug>/results"
INPUT_SOURCE_OVERRIDE = "/kaggle/input/<input-dataset-slug>/input"
```

Nếu dataset vẫn giữ file ZIP:

```python
RESULTS_ZIP_OVERRIDE = "/kaggle/input/<results-dataset-slug>/results.zip"
INPUT_SOURCE_OVERRIDE = "/kaggle/input/<input-dataset-slug>/input.zip"
```

Không cần đặt `PROJECT_ROOT_OVERRIDE` trong quy trình thông thường.

## 6. Dấu hiệu preflight đã qua

Trong log, notebook phải in một dictionary tương tự:

```text
{
  "config_compatibility": "validated",
  "source_branch": "codex/kaggle-end-to-end-pipeline",
  "model_type": "xlm-roberta",
  "model_transformers_version": "5.14.1",
  "label_count": 11
}
```

Điều này xác nhận notebook đã:

- tìm đúng `results/artifacts`;
- tìm đúng `results/training_artifacts/ner_model`;
- merge và kiểm tra `config.json`;
- kiểm tra model là XLM-R token classification;
- kiểm tra đủ 11 nhãn BIO;
- kiểm tra knowledge base ICD-10/RxNorm.

Sau đó notebook sẽ tìm input, kiểm tra CPU/GPU, cài dependency còn thiếu mà
không thay `torch`/`transformers`, load checkpoint và chạy inference.

## 7. Lấy kết quả

Sau khi **Run All** hoàn tất, kiểm tra:

```text
/kaggle/working/output.zip
/kaggle/working/run_manifest.json
/kaggle/working/diagnostics/
```

Mở `run_manifest.json` và xác nhận tối thiểu:

```json
{
  "training_skipped": true,
  "config_compatibility": "validated",
  "source_branch": "codex/kaggle-end-to-end-pipeline"
}
```

Chọn **Save Version → Save & Run All**. Sau khi version hoàn tất, mở tab
**Output** và tải `/kaggle/working/output.zip`.

Không dùng `output.zip` lịch sử nằm trong dataset kết quả. Notebook luôn tạo
file mới tại `/kaggle/working/output.zip`.

## 8. Trạng thái tương thích đã kiểm tra

Các artifact hiện tại đã được kiểm tra local bằng code của nhánh
`codex/kaggle-end-to-end-pipeline`:

- `config.json` được nạp và merge đúng;
- knowledge base đọc được 12.137 bản ghi ICD-10 và 56.053 bản ghi RxNorm;
- mapping tiếng Việt đúng UTF-8;
- tokenizer có vocab size 250.002;
- checkpoint có 199 tensor và 11 nhãn;
- model 1,1 GB load thành công;
- smoke inference nhận được `DISEASE` và `DRUG`.

`model_status.json` trong dataset có thể còn ghi NER chưa train. Đây là
metadata cũ từ bước build knowledge base. Pipeline inference không dùng cờ
này khi thư mục `training_artifacts/ner_model` tồn tại; checkpoint thật vẫn
được ưu tiên và được preflight trực tiếp.

Một số khóa mới như `enable_regex_fallback` hoặc
`thresholds.candidate_min_margin` có thể không nằm trong config cũ. Hàm
`load_config` merge các giá trị mặc định tương thích, nên không cần sửa
dataset chỉ vì thiếu các khóa này.

## 9. Xử lý lỗi thường gặp

### Không tìm thấy results

Kiểm tra dataset có đúng một trong các nguồn:

- `results/artifacts` và `results/training_artifacts/ner_model`;
- hoặc `results.zip`.

Nếu có nhiều nguồn, đặt `RESULTS_ZIP_OVERRIDE`.

### Không tìm thấy input

Đảm bảo dataset input có ít nhất một file `.txt` trong `input/` hoặc
`input.zip`. Nếu attach nhiều dataset có thư mục `input`, đặt
`INPUT_SOURCE_OVERRIDE`.

### Thiếu knowledge base sau khi Kaggle giải nén

Kiểm tra `icd10_dictionary.jsonl`/`.jsonl.gz` và
`rxnorm_dictionary.jsonl`/`.jsonl.gz`. Notebook hỗ trợ cả bản `.jsonl` Kaggle
đã tự giải nén và bản `.jsonl.gz`.

### Lỗi Git clone

Đảm bảo **Internet = ON**. Không cần thêm source code vào dataset kết quả.

### Thiếu `bm25s`, `faiss`, `sentence-transformers` hoặc `vllm`

Bật Internet, restart session và chạy lại từ cell đầu. Notebook mặc định cài
các dependency inference còn thiếu bằng `pip --no-deps`, tránh ghi đè bộ
`torch`/`transformers` do Kaggle cung cấp.

Nếu log có `Offending backend: keras_nlp` hoặc
`No module named transformers.tokenization_utils_tokenizers`, session đã bị
một lần cài pip cũ làm hỏng. Chọn **Restart Session** rồi chạy lại notebook
mới từ cell đầu; không chạy tiếp cell 5 trong session đó.

### CUDA Out Of Memory

Lỗi này chỉ áp dụng khi bạn chủ động bật GPU/Qwen:

```python
ENABLE_QWEN_RERANKER = True
REQUIRE_GPU = True
```

Thử giảm:

```python
QWEN_GPU_MEMORY_UTILIZATION = 0.40
QWEN_MAX_MODEL_LEN = 3072
QWEN_BATCH_SIZE = 32
```

Nếu chỉ cần kiểm tra pipeline không dùng Qwen:

```python
ENABLE_QWEN_RERANKER = False
```

Tắt Qwen có thể làm giảm chất lượng reranking/assertion so với cấu hình đầy
đủ.

### Sai schema hoặc thiếu output

Tải `/kaggle/working/diagnostics/` và đọc document ID báo lỗi. Notebook chủ
động không tạo bài nộp nếu số JSON, tên file hoặc schema không khớp input.
