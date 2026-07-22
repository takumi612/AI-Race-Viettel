# Kaggle runbook v2 (Inference-Only): Hybrid Retrieval & LLM Reranker

Notebook: `medical_information_extraction_inference_kaggle.ipynb`

Notebook này chỉ thực hiện **suy luận (Inference)** từ checkpoint NER và tập từ điển Artifacts có sẵn. Notebook **không** thực hiện huấn luyện (train) hay fine-tune lại mô hình.

## 1. Yêu cầu Môi trường (Cực kỳ quan trọng)

Do v2 sử dụng Hybrid Retrieval (FAISS) và LLM Reranker (vLLM Qwen2.5-7B), hệ thống Kaggle của bạn phải đáp ứng:
1. **Bật Internet (Internet = ON):** Để tải các thư viện bổ sung như `bm25s`, `faiss-cpu`, `sentence-transformers`, `vllm` và tự động clone mã nguồn từ GitHub.
2. **Bật GPU (Accelerator = T4 x2 hoặc P100):** Bắt buộc để chạy NER checkpoint và LLM Reranker. Khuyến nghị dùng T4 x2.

## 2. Tạo 2 Kaggle Datasets riêng biệt

Tạo 2 Dataset private riêng biệt để tránh việc vô tình nhầm lẫn giữa dữ liệu artifacts cũ và bài nộp mới:

1. **Dataset 1: Model Checkpoint & Artifacts**
   Tạo Dataset chứa file `results.zip` (hoặc thư mục `results/` đã tự động giải nén). File zip này chứa checkpoint mô hình NER và từ điển:
   ```text
   results/
   ├── training_artifacts/
   │   └── ner_model/
   │       ├── config.json
   │       ├── model.safetensors
   │       └── tokenizer.json
   └── artifacts/
       ├── icd10/
       └── rxnorm/
   ```
   *Lưu ý:* Không đưa mã nguồn Python, notebook, tập test hay Git repository vào Dataset này.

2. **Dataset 2: Dữ liệu cần suy luận (Inference Input)**
   Tạo Dataset chứa file `input.zip` hoặc thư mục `input/` gồm các văn bản `.txt`:
   ```text
   inference-input-data/
   └── input.zip                         # Hoặc input/<id>.txt
   ```
   *Lưu ý:* Chỉ chứa các văn bản cần dự đoán, không kèm dữ liệu train hoặc kết quả nộp cũ.

### 2.3. Cơ chế Ghi đè Đường dẫn Thủ công (Override Variables)
Để tránh rủi ro notebook tự động quét nhầm thư mục dữ liệu khác khi bạn đính kèm nhiều Datasets trên Kaggle, bạn có thể thiết lập trực tiếp các biến Override ở Cell 1 của Notebook:
```python
RESULTS_ZIP_OVERRIDE = "/kaggle/input/my-checkpoint-dataset/results.zip"
INPUT_SOURCE_OVERRIDE = "/kaggle/input/my-inference-dataset/input.zip"
```
Khi các biến này được gán đường dẫn tuyệt đối, notebook sẽ bỏ qua việc quét tự động và chỉ sử dụng đúng nguồn được chỉ định.

## 3. Quy trình thực thi 3 Stage trong Notebook

Notebook sẽ bỏ qua bước huấn luyện (`training_skipped: true`) và thực thi trực tiếp Pipeline 3 Stage như sau:

- **Stage 1: NER & Assertion**
  - Khôi phục `TransformerNERDetector` từ checkpoint được attach.
  - Trích xuất toàn bộ thực thể từ các file văn bản đầu vào.
  - $\rightarrow$ Xóa `TransformerNERDetector` & gọi `torch.cuda.empty_cache()`.
- **Stage 2: Hybrid Retrieval**
  - Load `bm25s` và `faiss` index vào RAM từ tập từ điển `artifacts/`.
  - Lọc Top-10 ứng viên cho tất cả thực thể.
  - $\rightarrow$ Xóa Index & gọi `gc.collect()`.
- **Stage 3: LLM Reranker**
  - Load `Qwen2.5-7B-Instruct-AWQ` vào GPU qua vLLM engine.
  - Đọc ngữ cảnh + Top-10 ứng viên $\rightarrow$ Quyết định mã chính xác.
  - $\rightarrow$ Tắt vLLM & xuất kết quả.

## 4. Cấu hình & Tạo notebook Kaggle

1. Vào **Kaggle → Code → New Notebook**.
2. Chọn **File → Import Notebook** và upload `medical_information_extraction_inference_kaggle.ipynb` từ thư mục `v2`.
3. Trong panel **Input**, chọn **Add Input** và attach cả 2 Datasets ở bước 2 (Dataset chứa `results.zip` và Dataset chứa `input.zip` / `input/*.txt`).
4. Trong **Settings**, chọn **GPU accelerator** và bật **Internet**.
5. Cấu hình mặc định Qwen Reranker sử dụng `Qwen/Qwen2.5-7B-Instruct-AWQ` (`gpu_memory_utilization = 0.5`, `max_model_len = 4096`, `batch_size = 64`).
6. Chọn **Run All**. Không chạy riêng lẻ từng cell ở cuối: các cell đầu tiên cần thực thi để tìm archive, cài đặt thư viện và khôi phục checkpoint.

## 5. File đầu ra & Xác minh

Sau khi **Run All** hoàn tất, các file kết quả sẽ nằm tại `/kaggle/working`:

```text
/kaggle/working/output.zip
/kaggle/working/run_manifest.json
/kaggle/working/diagnostics/
```

**Các bước xác minh:**
1. Mở `/kaggle/working/run_manifest.json` và xác nhận thuộc tính `"training_skipped": true`. Nếu không có hoặc có giá trị `false`, dừng lại vì phiên chạy chưa đạt chuẩn inference-only.
2. Chọn **Save Version → Save & Run All** để Kaggle lưu lại outputs của notebook.
3. Tải file `/kaggle/working/output.zip` mới nhất từ tab Output để nộp thi.

## 6. Các lỗi thường gặp ở phiên bản Inference

- **Không tìm thấy Dataset hoặc chọn sai file ZIP**: Nếu attach nhiều Dataset chứa artifacts, hãy xóa các Dataset không liên quan. Hoặc thiết lập `RESULTS_ZIP_OVERRIDE` trong cell cấu hình đầu tiên chỉ định chính xác đường dẫn `/kaggle/input/<dataset>/results.zip`.
- **Thiếu file trong Model Checkpoint**: Đảm bảo file `results.zip` chứa đầy đủ `training_artifacts/ner_model/config.json`, `model.safetensors`, `tokenizer.json` và thư mục `artifacts/` chứa từ điển ICD-10/RxNorm.
- **Không tìm thấy dữ liệu đầu vào (No input found)**: Đảm bảo đã attach Dataset chứa file `input.zip` hoặc thư mục `input/*.txt` chứa ít nhất một văn bản không rỗng.
- **ImportError: No module named 'vllm' / 'faiss'**: Kiểm tra lại xem đã bật **Internet** trên Kaggle chưa. Bật Internet, restart session và chọn Run All để notebook tự động cài đặt dependencies.
- **CUDA Out Of Memory ở Stage 3**: Pipeline tự giải phóng mô hình NER trước khi load Qwen. Nếu bị OOM trên GPU T4, hãy giảm `gpu_memory_utilization` từ `0.5` xuống `0.4` hoặc giảm `max_model_len`.
- **Lỗi Validation / Schema của output.zip**: Tải thư mục `/kaggle/working/diagnostics` để kiểm tra danh sách tài liệu bị lỗi, đảm bảo văn bản đầu vào đúng mã hóa UTF-8 trước khi thực hiện chạy lại.

