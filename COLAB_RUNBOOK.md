# Runbook: upload notebook và Run all trên Google Colab

Notebook: `medical_information_extraction_lab.ipynb`  
Git branch: `Pipeline_colab`

## 1. Chuẩn bị Google Drive một lần

Tạo cấu trúc sau:

```text
MyDrive/
└── AI-Race-Viettel/
    ├── data/
    │   ├── input/                         # một hoặc nhiều file <id>.txt
    │   ├── train/                         # tùy chọn: cặp <id>.txt + <id>.json
    │   ├── synthetic_train_v1/            # tùy chọn thay train/
    │   │   ├── input/<id>.txt
    │   │   └── gt/<id>.json
    │   ├── ICD10.xlsx                     # tùy chọn nếu artifacts cache đã có
    │   └── RxNorm_full_07062026.zip       # tùy chọn nếu artifacts cache đã có
    └── output/                            # notebook tự tạo
```

Thay cho `data/input/`, có thể upload `data/input.zip`. Không cần cả hai.

Nhánh Git đã chứa cache ICD-10/RxNorm trong `artifacts/`, nên inference bình thường không cần upload hai knowledge source raw. Chỉ cần raw files khi muốn rebuild knowledge bases.

## 2. Chuẩn bị annotation để train

Notebook hỗ trợ hai layout:

```text
data/train/001.txt
data/train/001.json
```

hoặc:

```text
data/synthetic_train_v1/input/001.txt
data/synthetic_train_v1/gt/001.json
```

Entity JSON dùng `position: [start, end]`, trong đó `end` exclusive và phải thỏa `raw_text[start:end] == text`. Xem thêm `train/README.md`.

Nếu không có annotation, notebook vẫn chạy inference nhưng cell NER training sẽ ghi `trained=false` và không tạo checkpoint giả.

## 3. Chạy Colab

1. Mở Colab và chọn **Runtime > Change runtime type > GPU**.
2. Upload `medical_information_extraction_lab.ipynb`.
3. Chọn **Runtime > Run all**.
4. Chấp nhận quyền mount Google Drive.

Không cần sửa cell nếu dùng đúng cấu trúc Drive ở trên. Notebook mặc định:

- clone `https://github.com/takumi612/AI-Race-Viettel.git` nhánh `Pipeline_colab`;
- cài `requirements-colab.txt`;
- tìm dữ liệu tại `MyDrive/AI-Race-Viettel/data/`;
- train với `FAST_DEV_RUN=False`;
- lưu checkpoint tại `MyDrive/clinical-nlp-training-artifacts/ner_model/`;
- lưu kết quả tại `MyDrive/AI-Race-Viettel/output/output.zip`.

## 4. Chạy thử nhanh

Đổi trong cell đầu:

```python
FAST_DEV_RUN = True
```

Khi smoke-test thành công, đổi lại `False` và Run all.

## 5. Dữ liệu nằm ở vị trí khác

Sửa cell `Production data and output resolver`:

```python
DATA_ROOT_OVERRIDE = "/content/drive/MyDrive/thu_muc_du_lieu"
OUTPUT_ARCHIVE_OVERRIDE = "/content/drive/MyDrive/thu_muc_ket_qua/output.zip"
```

Hoặc dùng các biến chi tiết trong bootstrap: `INPUT_ZIP_OVERRIDE`, `TRAIN_DIR_OVERRIDE`, `ICD10_PATH_OVERRIDE`, `RXNORM_ZIP_OVERRIDE` và `TRAINING_OUTPUT_DIR_OVERRIDE`.

## 6. Kết quả mong đợi

Cell cuối phải in:

```text
submission_schema_valid: True
output_zip: /content/drive/MyDrive/AI-Race-Viettel/output/output.zip
```

ZIP phải có cấu trúc:

```text
output.zip
└── output/
    ├── 1.json
    ├── 2.json
    └── ...
```

## Lưu ý về dữ liệu hiện tại

Repository không chứa private input hoặc annotation train thật. Official schema đã được đối chiếu từ `src/validation/submission.py` của chính repository và notebook áp dụng mapping tương ứng. Nếu không có annotation, notebook vẫn tạo submission bằng baseline artifacts/rule-dictionary nhưng không thể báo supervised validation score; chất lượng cần được đánh giá bằng ground truth trước khi nộp thi.
