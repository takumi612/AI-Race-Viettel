# Kaggle runbook: train và tạo output.zip

Notebook: `medical_information_extraction_kaggle.ipynb`

## 1. Tạo Kaggle Dataset chứa dữ liệu

Tạo một Dataset private, ví dụ `ai-race-clinical-data`, với cấu trúc:

```text
ai-race-clinical-data/
├── input.zip                         # hoặc input/<id>.txt
└── synthetic_train_v1/
    ├── input/<id>.txt
    └── gt/<id>.json
```

Cũng có thể dùng layout train trực tiếp:

```text
train/001.txt
train/001.json
```

Entity train dùng internal type `DISEASE`, `DRUG`, `SYMPTOM`, `LAB_RESULT`
hoặc `PATIENT_INFO`; `position` là `[start, end]` với `end` exclusive và phải
thỏa `raw_text[start:end] == text`.

## 2. Tạo notebook Kaggle

1. Vào **Kaggle → Code → New Notebook**.
2. Chọn **File → Import Notebook** và upload
   `medical_information_extraction_kaggle.ipynb`.
3. Trong panel **Input**, chọn **Add Input** và attach Dataset ở bước 1.
4. Trong **Settings**, chọn một **GPU accelerator**.
5. Bật **Internet** để notebook clone GitHub và tải `xlm-roberta-base`.
6. Chọn **Run All**.

Mặc định không cần sửa cell. Notebook tự tìm `input.zip`, `input/*.txt`,
`train/` hoặc `synthetic_train_v1/` bên dưới `/kaggle/input`.

## 3. Chạy khi Internet bị tắt

Attach thêm hai Kaggle Dataset:

- Dataset code chứa `clinical_nlp_lab/`, `artifacts/` và requirements.
- Dataset model chứa toàn bộ file Hugging Face của `xlm-roberta-base`.

Nếu auto-discovery không tìm thấy, đặt trong cell đầu:

```python
PROJECT_ROOT_OVERRIDE = "/kaggle/input/<code-dataset>"
MODEL_NAME_OR_PATH_OVERRIDE = "/kaggle/input/<model-dataset>/xlm-roberta-base"
```

## 4. Tùy chỉnh training

Mặc định notebook train đầy đủ:

```python
FAST_DEV_RUN = False
REQUIRE_TRAINING_DATA = True
REQUIRE_GPU = True
```

Để kiểm tra pipeline nhanh, đổi `FAST_DEV_RUN = True`. Không dùng checkpoint
smoke-test để nộp bài.

Các hyperparameter chính nằm trong cell **Training configuration**:

```python
NER_EPOCHS = 3
TRAIN_BATCH_SIZE = 4
LEARNING_RATE = 2e-5
```

Nếu GPU hết bộ nhớ, giảm `TRAIN_BATCH_SIZE` xuống `2` hoặc `1`.

## 5. File đầu ra

Sau Run All, các file nằm trong `/kaggle/working`:

```text
/kaggle/working/output.zip
/kaggle/working/trained_ner_artifacts.zip
/kaggle/working/training_artifacts/ner_model/
/kaggle/working/diagnostics/
/kaggle/working/run_manifest.json
```

`output.zip` dùng checkpoint NER vừa train nếu training thành công. Chọn
**Save Version → Save & Run All** để Kaggle lưu notebook outputs, sau đó tải
`output.zip` và `trained_ner_artifacts.zip` từ tab Output.

## 6. Các lỗi thường gặp

- `No annotated training data found`: Dataset chưa có `train/` hoặc cặp
  `synthetic_train_v1/input + gt`.
- `GPU is required`: chưa bật accelerator trong Settings.
- Không tải được model: bật Internet hoặc attach model Dataset và đặt
  `MODEL_NAME_OR_PATH_OVERRIDE`.
- CUDA out of memory: giảm batch size; giữ `max_length=512` và `stride=128`.
- Offset validation failed: sửa annotation để `raw_text[start:end] == text`.
