# Kaggle runbook v2: Hybrid Retrieval & LLM Reranker

Notebook: `medical_information_extraction_kaggle.ipynb`

Đây là phiên bản **tối ưu hóa bộ nhớ (Memory Optimized)** chia Pipeline làm 3 Stage xử lý độc lập để bạn có thể chạy được các Model/Index "khổng lồ" mà không bị tràn RAM/VRAM của Kaggle.

## 1. Yêu cầu Môi trường (Cực kỳ quan trọng)

Do v2 sử dụng Hybrid Retrieval (FAISS) và LLM Reranker (vLLM Qwen2.5-7B), hệ thống Kaggle của bạn phải đáp ứng:
1. **Bật Internet (Internet = ON):** Để tải các thư viện bổ sung như `bm25s`, `faiss-cpu`, `sentence-transformers`, `vllm`.
2. **Bật GPU (Accelerator = T4 x2 hoặc P100):** Bắt buộc để chạy LLM Reranker và NER. Khuyến nghị dùng T4 x2.

## 2. Tạo Kaggle Dataset chứa dữ liệu

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

## 4. Tạo notebook Kaggle

1. Vào **Kaggle → Code → New Notebook**.
2. Chọn **File → Import Notebook** và upload `medical_information_extraction_kaggle.ipynb` từ thư mục `v2`.
3. Trong panel **Input**, chọn **Add Input** và attach Dataset ở bước 2.
4. Trong **Settings**, chọn **GPU accelerator** và bật **Internet**.
5. Bootstrap cài requirements train/retrieval (`bm25s`, `faiss-cpu`, `sentence-transformers`) và kiểm tra/cài `vllm==0.25.1` riêng cho inference. Không cần cài `autoawq`: vLLM đọc checkpoint AWQ trực tiếp.
6. Chọn **Run All**.

## 5. File đầu ra

Sau Run All, các file nằm trong `/kaggle/working`:

```text
/kaggle/working/output.zip
/kaggle/working/trained_ner_artifacts.zip
```

Chọn **Save Version → Save & Run All** để Kaggle lưu notebook outputs, sau đó tải `output.zip` từ tab Output để nộp thi.

## 6. Các lỗi thường gặp ở v2

- **ImportError: No module named 'vllm' / 'faiss'**: Bạn quên bật Internet trên Kaggle. Hoặc Kaggle đang thiếu ổ đĩa để tải, bạn có thể tạo 1 dataset riêng chứa các `.whl` offline.
- **CUDA Out Of Memory ở Stage 3**: Pipeline phải giải phóng NER, encoder và index retrieval trước khi tải Qwen. Nếu vẫn OOM trên T4, giảm `gpu_memory_utilization` từ `0.5` xuống `0.4` hoặc giảm `max_model_len`; không tăng lên `0.95`.
- **Lỗi không tìm thấy mô hình NER**: Nếu bạn không attach tập train (chỉ có tập input) và cũng không có checkpoint nào, Stage 1 sẽ bỏ qua việc trích xuất và output ra file JSON rỗng.
