# AI Race 2026 — Bài 2: Ontological Reasoning in Medical Knowledge Retrieval

Hệ thống xử lý văn bản y tế tự động (Clinical NLP) phục vụ cuộc thi **Viettel AI Race 2026 (Vòng 1 - Sơ loại)**. Pipeline thực hiện trích xuất thực thể y khoa (NER) từ bệnh án tiếng Việt tự do, chuẩn hóa thuộc tính (Assertion), truy xuất và ánh xạ (Entity Normalization) sang mã danh mục chuẩn quốc tế:
- Bệnh/Chẩn đoán $\rightarrow$ **ICD-10** (Danh mục Bộ Y Tế)
- Thuốc $\rightarrow$ **RxNorm** (CSDL chuẩn Mỹ UMLS)

---

## Modular training trên Colab T4

Repo đã có pipeline huấn luyện tách module cho XLM-R NER, BGE-M3 LoRA và
Qwen2.5-7B QLoRA. Tất cả trainer chỉ đọc build dữ liệu có fingerprint, chặn
pseudo-GT/holdout khỏi gradient, hỗ trợ checkpoint resume và yêu cầu review
precision-first trước khi promote artifact.

Runbook đầy đủ: [docs/training/MODEL_TRAINING_COLAB.md](docs/training/MODEL_TRAINING_COLAB.md).
Nền tảng data/split: [docs/training/DATA_FOUNDATION.md](docs/training/DATA_FOUNDATION.md).

```bash
python -m pip install -r requirements-train.txt
python -m pytest tests/training -q
python -m src.training.ner.train --help
python -m src.training.embedding.train --help
python -m src.training.reranker.train --help
```

Lưu ý: code trainer đã hoàn chỉnh, nhưng không được bắt đầu train production
cho tới khi synthetic 2.000 mẫu pass QA và các code ontology còn thiếu trong
trusted 101–180 được xử lý có provenance.

## Trusted benchmark policy

Configuration selection uses only trusted development IDs **101–180** with deterministic folds. IDs **1–100** are self-generated pseudo-GT and may be reported only as `untrusted`; they never select thresholds or retrieval weights. Holdout IDs **181–200** are evaluated once with a SHA-256-verified locked configuration.

```bash
python -m src.evaluation.benchmark --dev-pool --baseline --output reports/baseline-dev.json
python -m src.evaluation.benchmark --dev-pool --alphas 0.60 0.70 0.75 0.80 0.90 --output reports/precision-first-cv.json --write-locked-config reports/locked-config.json
python -m src.evaluation.benchmark --holdout --locked-config reports/locked-config.json --output reports/final-holdout.json
python src/pipeline/main.py --input data/input --output data/output --config reports/locked-config.json
```

The benchmark refuses an invalid or reused holdout lock. Neural NER remains optional; the current production path uses the validated baseline extractor and precision-first selector.

## 📂 Cấu trúc dự án

```text
d:\AI Race Viettel\
├── docs/                      # Tài liệu thiết kế & phân tích đề bài
│   ├── Kế hoạch triển khai.md  # Master Plan v6 (Chốt giải pháp kỹ thuật)
│   ├── Phân tích đề bài.md     # Tóm tắt nghiệp vụ & luật chơi
│   ├── eval_assumptions.md    # Chi tiết cách đếm position & công thức chấm điểm
│   └── experiment_log.md      # Nhật ký thực nghiệm (Tracking mô hình & điểm)
├── data/                      # Quản lý dữ liệu tập trung (Đã gitignored tệp lớn)
│   ├── kb/                    # CSDL tri thức y khoa (ICD-10 SQLite, RxNorm context)
│   ├── raw/                   # Dữ liệu synthetic thô phục vụ training (BIO format)
│   ├── processed/             # Dữ liệu đã gán nhãn & phân chia train/val
│   ├── dev/                   # Tập dữ liệu kiểm thử cục bộ (Validation Set)
│   ├── input/                 # 100 file .txt đầu vào tập test công khai của BTC
│   └── output/                # Kết quả JSON dự đoán của pipeline
├── src/                       # Mã nguồn chính
│   ├── data_generation/       # Module sinh dữ liệu huấn luyện lâm sàng
│   ├── ner/                   # Mô hình trích xuất thực thể (Backbone XLM-R)
│   ├── assertion/             # Phân tích thuộc tính (isNegated, isFamily, isHistorical)
│   ├── retrieval/             # Truy xuất candidates (BM25s + FAISS Flat IP)
│   ├── ranking/               # Tái xếp hạng candidates bằng LLM (Qwen) + XGrammar
│   ├── validation/            # Bộ lọc kiểm duyệt lâm sàng (clinical_validator.py)
│   ├── pipeline/              # Bộ điều phối (Orchestrator) kết nối toàn pipeline
│   ├── utils/
│   │   ├── paths.py           # Single Source of Truth cho đường dẫn dự án
│   │   └── setup_db.py        # Thiết lập cơ sở dữ liệu SQLite y khoa
│   ├── metrics.py             # Engine tự chấm điểm cục bộ (Passed 7 unit tests)
│   └── evaluate.py            # CLI chạy chấm điểm & sweep IoU tự động
```

---

## 🛠️ Hướng dẫn thiết lập (Setup)

### 1. Khởi tạo môi trường ảo Python 3.10+

Khuyến nghị sử dụng `venv` hoặc `conda` để tránh xung đột thư viện:

```bash
# Sử dụng venv
python -m venv venv
source venv/Scripts/activate  # Trên Windows
source venv/bin/activate      # Trên Linux/macOS

# Hoặc sử dụng Conda
conda create -n airace python=3.10 -y
conda activate airace
```

### 2. Cài đặt các thư viện cần thiết

```bash
pip install -r requirements.txt
# Hoặc cài đặt trực tiếp các thư viện core:
pip install bm25s scipy rapidfuzz pandas openpyxl
```

---

## 💾 Hướng dẫn tải dữ liệu từ Google Drive

Toàn bộ CSDL chuẩn, mô hình, và tệp bổ trợ lớn đã được upload tại thư mục [Google Drive của dự án](https://drive.google.com/drive/folders/1d3DdQEJHjfuHSPX65Ld-mEYxqu6gI9tK?usp=sharing).

### 1. Cài đặt công cụ `gdown`
Để tải nhanh thư mục từ Drive qua Command Line:
```bash
pip install gdown
```

### 2. Tải toàn bộ thư mục dữ liệu y khoa vào `data/kb/`
Chạy lệnh sau từ thư mục gốc của dự án để kéo CSDL chuẩn (`metadata.db`, `ICD10.xlsx`, `icd10_context.txt`, `icd10_dictionary.json`...):

```bash
gdown --folder "https://drive.google.com/drive/folders/1d3DdQEJHjfuHSPX65Ld-mEYxqu6gI9tK" -O data/kb/ --remaining-ok
```

*Lưu ý: Nếu quá trình tải từ Drive báo lỗi quota hoặc file quá lớn, bạn có thể tải thủ công qua trình duyệt và đặt các file vào đúng thư mục `data/kb/`.*

### Cấu trúc dữ liệu mong đợi sau khi tải:
```text
data/kb/
├── ICD10.xlsx           # Excel gốc ICD-10 của Bộ Y Tế
├── metadata.db          # SQLite chứa 25,123 mã ICD-10 và luật lâm sàng
├── icd10_context.txt    # Context text phẳng cho RAG
├── icd10_dictionary.json# Dictionary JSON phẳng cho LLM
└── (RxNorm files...)    # Dữ liệu RxNorm (Tải sau 17/07)
```

---

## 📈 Kiểm thử và Đánh giá cục bộ

Chúng ta có một Engine tự chấm điểm hoàn toàn độc lập, khớp cấu trúc công thức của BTC giúp đánh giá chất lượng mô hình trước khi nộp.

### 1. Chạy Unit Test cho thuật toán chấm điểm
Trước khi chạy đánh giá thực tế, hãy chắc chắn Engine chấm điểm hoạt động chính xác:
```bash
python src/metrics.py test
```
*Kết quả mong đợi: `=================== ALL TESTS PASSED SUCCESSFULLY ===================`*

### 2. Chạy đánh giá (Evaluation CLI)
So sánh kết quả dự đoán trong `data/output/` với nhãn chuẩn trong `data/dev/`:
```bash
python src/evaluate.py --gt data/dev/ --pred data/output/
```

### 3. Chạy Sweep tự động IoU tìm ngưỡng ghép cặp của BTC
Do BTC không công bố ngưỡng IoU dùng để ghép cặp khái niệm đoán đúng với nhãn thật, sử dụng tính năng `--sweep` sau khi nộp thử baseline để so khớp:
```bash
python src/evaluate.py --sweep
```
*Kết quả sẽ xuất ra bảng so sánh điểm chi tiết ứng với các ngưỡng IoU từ `0.3` đến `0.8` để bạn đối chiếu ngay với điểm Leaderboard thật.*

---

## 📖 Tài liệu cần đọc trước khi phát triển

1. **[Master Plan v6 (Kế hoạch triển khai.md)](docs/Kế hoạch triển khai.md)**: Đọc để hiểu kiến trúc 3 tầng (NER XLM-R $\rightarrow$ Hybrid Retrieval $\rightarrow$ LLM Reranker), bảng phân công nhiệm vụ và lộ trình triển khai hàng ngày.
2. **[Phân tích đề bài.md](docs/Phân tích đề bài.md)**: Hiểu các yêu cầu nghiệp vụ, định dạng JSON động đầu ra của 5 loại thực thể y tế.
3. **[eval_assumptions.md](docs/eval_assumptions.md)**: Tài liệu hóa các giả định toán học về cách tính Position (Unicode codepoint), WER, Jaccard và ghép cặp Greedy.
4. **[experiment_log.md](docs/experiment_log.md)**: Điền thông số và điểm số vào đây sau mỗi lượt nộp bài để theo dõi tiến độ cải tiến mô hình.
