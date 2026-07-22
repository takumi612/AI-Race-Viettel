# Kaggle runbook v2: Hybrid Retrieval & LLM Reranker

Notebook: `medical_information_extraction_kaggle.ipynb`

Đây là phiên bản **tối ưu hóa bộ nhớ (Memory Optimized)** chia Pipeline làm 3 Stage xử lý độc lập để bạn có thể chạy được các Model/Index "khổng lồ" mà không bị tràn RAM/VRAM của Kaggle.

## 1. Yêu cầu Môi trường (Cực kỳ quan trọng)

Do v2 sử dụng Hybrid Retrieval (FAISS) và LLM Reranker (vLLM Qwen2.5-7B), hệ thống Kaggle của bạn phải đáp ứng:
1. **Bật Internet (Internet = ON):** Để tải các thư viện bổ sung như `bm25s`, `faiss-cpu`, `sentence-transformers`, `vllm`.
2. **Bật GPU (Accelerator = T4 x2 hoặc P100):** Bắt buộc để chạy LLM Reranker và NER. Khuyến nghị dùng T4 x2.

## 2. Tạo Kaggle Dataset & Cấu hình Đường dẫn Dữ liệu

### 2.1. Cấu trúc Dataset chuẩn
Tạo một Dataset private, ví dụ `ai-race-clinical-data`, với cấu trúc:

```text
ai-race-clinical-data/
├── input.zip                         # hoặc input/<id>.txt (Dữ liệu cần dự đoán)
└── synthetic_train_v1/               # Dữ liệu huấn luyện
    ├── input/<id>.txt
    └── gt/<id>.json
```

Cũng có thể dùng layout train trực tiếp:

```text
train/001.txt
train/001.json
```

### 2.2. Cơ chế Tự động Tìm kiếm Dữ liệu (Auto-Discovery) & Chỉ định Đường dẫn (Override)
- **Tự động tìm kiếm (Auto-Discovery):** Notebook tự động quét trong `/kaggle/input/` để phát hiện dữ liệu huấn luyện (`TRAIN_SOURCE`) và dữ liệu dự đoán (`INPUT_SOURCE`).
- **Ghi đè thủ công (Override):** Khi attach nhiều Dataset hoặc muốn đảm bảo **tính toàn vẹn dữ liệu (Data Integrity)** (tránh trường hợp notebook quét nhầm thư mục chứa tập test làm dữ liệu train), bạn nên điền trực tiếp đường dẫn tuyệt đối vào cell 1 của Notebook:
  ```python
  INPUT_SOURCE_OVERRIDE = "/kaggle/input/ai-race-clinical-data/input.zip"
  TRAIN_SOURCE_OVERRIDE = "/kaggle/input/ai-race-clinical-data/synthetic_train_v1"
  ```
- **Lưu ý:** Khi `TRAIN_SOURCE_OVERRIDE` hoặc `INPUT_SOURCE_OVERRIDE` được thiết lập, notebook sẽ bỏ qua hoàn toàn logic tự động quét và chỉ nạp đúng đường dẫn được chỉ định.

## 3. Quy trình thực thi 3 Stage trong Notebook

Sau khi load dữ liệu và train xong mô hình NER (nếu cần), phần Inference sẽ chạy tuần tự như sau:

- **Stage 1: NER & Assertion** 
  - Load `TransformerNERDetector` vào GPU.
  - Cắt toàn bộ thực thể.
  - $\rightarrow$ Xóa `TransformerNERDetector` & gọi `torch.cuda.empty_cache()`.
- **Stage 2: Hybrid Retrieval**
  - Load `bm25s` và `faiss` index vào RAM. 
  - Lọc Top-10 ứng viên cho tất cả thực thể.
  - $\rightarrow$ Xóa Index & gọi `gc.collect()`.
- **Stage 3: LLM Reranker**
  - Load `Qwen2.5-7B-Instruct-AWQ` vào GPU qua vLLM engine.
  - Đọc ngữ cảnh + Top-10 ứng viên $\rightarrow$ Quyết định mã chính xác.
  - $\rightarrow$ Tắt vLLM & xuất kết quả.

## 4. Quản lý Thư viện Phụ thuộc (Dependencies Bootstrap)

1. **Tự động cài đặt (Internet = ON):** 
   - Biến `INSTALL_MISSING_DEPENDENCIES` trong Notebook dùng để điều khiển việc tự động cài các gói còn thiếu (`bm25s`, `faiss-cpu`, `sentence-transformers`, `vllm`).
   - Notebook được thiết kế để chỉ cài thêm các package truy vấn còn thiếu mà **KHÔNG nâng cấp hoặc ghi đè** bộ thư viện PyTorch/Transformers gốc của Kaggle, tránh gây lỗi xung đột phiên bản CUDA.
2. **Chế độ Offline (Internet = OFF):**
   - Nếu chạy trong môi trường không có Internet, bạn cần tạo 1 Kaggle Dataset chứa sẵn các file wheel (`.whl`) của `bm25s`, `faiss-cpu`, `sentence-transformers`, `vllm` và đính kèm vào notebook.

## 5. Quy trình tạo và chạy Notebook Kaggle

1. Vào **Kaggle → Code → New Notebook**.
2. Chọn **File → Import Notebook** và upload `medical_information_extraction_kaggle.ipynb` từ thư mục `v2`.
3. Trong panel **Input**, chọn **Add Input** và attach Dataset ở bước 2.
4. Trong **Settings**, chọn **GPU accelerator** và bật **Internet**.
5. Đặt `INSTALL_MISSING_DEPENDENCIES = True` (nếu môi trường Kaggle thiếu `bm25s`/`sentence-transformers`) hoặc điền `TRAIN_SOURCE_OVERRIDE` / `INPUT_SOURCE_OVERRIDE` nếu muốn chỉ định chính xác nguồn dữ liệu.
6. Chọn **Run All**.

## 6. File đầu ra

Sau Run All, các file nằm trong `/kaggle/working`:

```text
/kaggle/working/output.zip
/kaggle/working/trained_ner_artifacts.zip
/kaggle/working/run_manifest.json
/kaggle/working/diagnostics/run_summary.json
```

Chọn **Save Version → Save & Run All** để Kaggle lưu notebook outputs, sau đó tải `output.zip` từ tab Output để nộp thi.

## 7. Các lỗi thường gặp ở v2

- **ImportError / Missing dependencies**: Kiểm tra đã bật Internet trên Kaggle hay chưa. Nếu thiếu `bm25s` hoặc `sentence-transformers`, đảm bảo `INSTALL_MISSING_DEPENDENCIES = True` hoặc cài đặt gói offline.
- **CUDA Out Of Memory ở Stage 3**: Pipeline phải giải phóng NER, encoder và index retrieval trước khi tải Qwen. Nếu vẫn OOM trên T4, giảm `gpu_memory_utilization` từ `0.5` xuống `0.4` hoặc giảm `max_model_len`; không tăng lên `0.95`.
- **Lỗi không tìm thấy mô hình NER**: Nếu bạn không attach tập train (chỉ có tập input) và cũng không có checkpoint nào, Stage 1 sẽ bỏ qua việc trích xuất và output ra file JSON rỗng.

