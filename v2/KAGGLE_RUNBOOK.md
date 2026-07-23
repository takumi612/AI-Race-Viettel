# Kaggle runbook: clinical training → inference → artifacts

Notebook chính: `medical_information_extraction_kaggle.ipynb`.

Notebook đã nhúng đúng runtime v2 và các KB artifact nhỏ cần thiết, nên không
phụ thuộc vào một checkout GitHub `main` có thể cũ. Có thể attach thêm code
Dataset để debug, nhưng không bắt buộc.

## Chuẩn bị Dataset

Attach một Kaggle Dataset có cấu trúc:

```text
ai-race-clinical-data/
├── input/                         # hoặc input.zip: dữ liệu cần dự đoán
└── synthetic_train_v2/
    ├── input/<id>.txt
    ├── gt/<id>.json
    └── reports/dataset_manifest.jsonl
```

`synthetic_train_v2` hiện gồm 2.200 hồ sơ. Manifest đánh dấu 1–100 là
quarantine (không dùng train), 101–200 là organizer GT giữ nguyên, và 2.000
mẫu synthetic đủ điều kiện huấn luyện.

## Runtime Kaggle

1. Bật GPU accelerator.
2. Bật Internet nếu chưa attach sẵn model XLM-R và embedding model.
3. Import `v2/medical_information_extraction_kaggle.ipynb`.
4. Nếu notebook không tự tìm đúng Dataset, đặt trong cell đầu:

```python
INPUT_SOURCE_OVERRIDE = "/kaggle/input/ai-race-clinical-data/input"
TRAIN_SOURCE_OVERRIDE = "/kaggle/input/ai-race-clinical-data/synthetic_train_v2"
```

Notebook tự cài các package thiếu từ `requirements-kaggle.txt`. File này yêu
cầu `transformers`, `accelerate>=1.1.0`, `sentencepiece`, `safetensors`,
`bm25s`, `faiss-cpu` và `sentence-transformers`.

## Run All và các bước được thực hiện

1. Kiểm tra input/GT và manifest.
2. Chỉ nạp các document `train_eligible=true`.
3. Grouped train/validation split theo template và clinical surface.
4. Fine-tune XLM-R NER, có gradient accumulation, mixed precision trên GPU,
   early stopping và token-level evaluation.
5. Lưu `training_artifacts/ner_model/model.safetensors`, tokenizer,
   `training_result.json` và `split_manifest.json`.
6. Đóng gói `trained_ner_artifacts.zip`.
7. Reload checkpoint vừa train để inference.
8. Entity linking dùng KB ICD-10/RxNorm. Embedding retrieval được ưu tiên;
   nếu không tải được model semantic, pipeline tự chuyển sang lexical BM25/
   alias retrieval và vẫn tạo output hợp lệ.
9. Qwen reranker là tầng tùy chọn. Nếu lỗi VRAM hoặc model, deterministic
   pipeline vẫn tiếp tục và ghi rõ `llm_fallback_reason`.
10. Validate offset, schema, số lượng file và CRC của `output.zip`.

## File đầu ra

Trong `/kaggle/working` cần có:

```text
output.zip                         # output/<document_id>.json
trained_ner_artifacts.zip          # checkpoint + tokenizer + training result
training_artifacts/ner_model/      # checkpoint gốc
training_artifacts/training_result.json
training_artifacts/split_manifest.json
run_manifest.json
diagnostics/run_summary.json
```

Notebook sẽ dừng nếu thiếu checkpoint sau khi train, thiếu output JSON, sai
offset, sai schema hoặc `output.zip` có member lỗi CRC.

## Báo cáo sự cố khi Kaggle lỗi

Nếu `Run All` gặp lỗi hoặc không thể hoàn thành, vui lòng gửi lại cho agent tối thiểu các thông tin/file sau:

1. `run_manifest.json` và `LATEST.json` (nếu đã được khởi tạo);
2. File JSONL log của phase phát sinh lỗi (trong thư mục `logs/` hoặc console output);
3. `resource_plan.json` và traceback đầy đủ từ cell bị dừng.

Agent sẽ phân tích nguyên nhân lỗi chính xác từ log, đề xuất bản sửa code và hướng dẫn cách `resume` hoặc chạy lại an toàn.

## Smoke test cục bộ

Có thể kiểm tra trước khi upload Kaggle:

```powershell
$env:PYTHONPATH = "D:\AI Race Viettel\v2"
python v2/scripts/train_ner_subprocess.py `
  --train-source "D:\AI Race Viettel\data_v2\Training_data\synthetic_train_v2" `
  --output-dir "D:\AI Race Viettel\scratch\local_training_smoke" `
  --config-path "D:\AI Race Viettel\v2\artifacts\config.json" `
  --model-source "D:\AI Race Viettel\results\training_artifacts\ner_model" `
  --fast-dev-run True
```

Sau đó dùng notebook để chạy inference và kiểm tra `output.zip`.
