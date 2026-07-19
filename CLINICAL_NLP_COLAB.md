# Clinical NLP notebook trên Colab

Nhánh `Pipeline_colab` bổ sung một pipeline offline, độc lập với pipeline hiện có trong `src/`.

## Chạy

1. Mở `medical_information_extraction_lab.ipynb` bằng Colab.
2. Trong cell bootstrap đầu tiên, đặt:

```python
GITHUB_REPO_URL = "https://github.com/takumi612/AI-Race-Viettel.git"
MOUNT_GOOGLE_DRIVE = True
FAST_DEV_RUN = True   # smoke-test; đổi False để train đầy đủ
```

3. Đặt annotation trong `train/`, hoặc dùng `TRAIN_DIR_OVERRIDE` trỏ tới thư mục trên Drive.
4. Nếu dữ liệu không nằm trong repo, dùng `INPUT_ZIP_OVERRIDE`, `ICD10_PATH_OVERRIDE` và `RXNORM_ZIP_OVERRIDE`.
5. Checkpoint được lưu mặc định tại `MyDrive/clinical-nlp-training-artifacts/ner_model/`.

## Annotation

Xem [`train/README.md`](train/README.md). Loader hỗ trợ cặp `.txt` + `.json` hoặc JSON record có `raw_text`.

## Thành phần được thêm

- `clinical_nlp_lab/`: loader, KB, NER, assertion, linking, relation, evaluator và pipeline.
- `artifacts/`: cache ICD-10/RxNorm và mapping runtime.
- `tools/`: build KB, chạy stage, build/execute notebook.
- `stages/`, `reports/`, `tests/clinical_nlp_lab/`: tài liệu, bằng chứng và test.

Không commit dataset/private input, raw RxNorm ZIP, output, diagnostics, checkpoint hoặc `test_artifacts/`.
