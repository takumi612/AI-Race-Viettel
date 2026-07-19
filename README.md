# AI Race 2026 — Bài 2: Ontological Reasoning in Medical Knowledge Retrieval

Hệ thống xử lý văn bản y tế tự động (Clinical NLP) phục vụ cuộc thi **Viettel AI Race 2026 (Vòng 1 - Sơ loại)**. Pipeline thực hiện trích xuất thực thể y khoa (NER) từ bệnh án tiếng Việt tự do, chuẩn hóa thuộc tính (Assertion), truy xuất và ánh xạ (Entity Normalization) sang mã danh mục chuẩn quốc tế:
- Bệnh/Chẩn đoán $\rightarrow$ **ICD-10** (Danh mục Bộ Y Tế)
- Thuốc $\rightarrow$ **RxNorm** (CSDL chuẩn Mỹ UMLS)

---

## One-click Clinical NLP notebook trên Colab

Nếu mục tiêu là chạy pipeline end-to-end và sinh `output.zip`, mở
[`medical_information_extraction_lab.ipynb`](medical_information_extraction_lab.ipynb)
trên Colab rồi chọn **Runtime → Run all**. Notebook tự mount Drive, clone đúng
nhánh `Pipeline_colab`, cài `requirements-colab.txt`, tìm input thật và lưu kết
quả tại `MyDrive/AI-Race-Viettel/output/output.zip`.

Chuẩn bị dữ liệu một lần theo
[`COLAB_RUNBOOK.md`](COLAB_RUNBOOK.md). Không cần upload input/model vào Git;
dataset, output, checkpoint và raw RxNorm lớn được giữ trên Drive. Nếu có
annotation, đặt cặp `.txt` + `.json` trong `data/train/` hoặc layout
`data/synthetic_train_v1/input` + `data/synthetic_train_v1/gt` để notebook train
trước khi inference. Nếu không có annotation, notebook chạy baseline artifact
và không bịa supervised score.

Chi tiết các giai đoạn và bằng chứng runtime nằm trong `stages/stage_01` đến
`stages/stage_09`, `reports/final_verification.json` và
`CLINICAL_NLP_PROJECT_STATE.md`.

---

## Modular training trên Colab T4

Repo đã có pipeline huấn luyện tách module cho XLM-R NER, BGE-M3 LoRA và
Qwen2.5-7B QLoRA. Tất cả trainer chỉ đọc build dữ liệu có fingerprint, chặn
pseudo-GT/holdout khỏi gradient, hỗ trợ checkpoint resume và yêu cầu review
precision-first trước khi promote artifact.

Runbook đầy đủ: [docs/training/MODEL_TRAINING_COLAB.md](docs/training/MODEL_TRAINING_COLAB.md).
Nền tảng data/split: [docs/training/DATA_FOUNDATION.md](docs/training/DATA_FOUNDATION.md).
Tải Qwen và train QLoRA:
[docs/training/QWEN_QLORA_COLAB.md](docs/training/QWEN_QLORA_COLAB.md).

### Quick start trên Colab

Repo GitHub không chứa dataset hoặc model lớn. Trên một runtime Colab mới,
phải clone code rồi tải toàn bộ `data` canonical từ Drive **trước** khi build
hoặc train:

```bash
%cd /content
!git clone --branch develop --single-branch https://github.com/takumi612/AI-Race-Viettel.git
%cd /content/AI-Race-Viettel
!python -m pip install -U "gdown>=6.0.0"
!python -m gdown "https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link" \
  --folder -O "/content/AI-Race-Viettel/data/"
```

Giữ dấu `/` cuối output để `gdown` ghi các thư mục `dev`, `kb`, `models`,
`synthetic_train_v1` trực tiếp bên trong `data`, không tạo `data/data`.
`gdown` 6 không dùng tùy chọn cũ `--remaining-ok`.

Chạy cell fail-fast sau ngay khi download xong:

```python
from pathlib import Path
import hashlib
import json
import sqlite3

data = Path("/content/AI-Race-Viettel/data")
synthetic = data / "synthetic_train_v1"
required = [synthetic, data / "dev", data / "models", data / "kb" / "metadata.db"]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Drive download incomplete; missing: {missing}")

input_count = len(list((synthetic / "input").glob("*.txt")))
gt_count = len(list((synthetic / "gt").glob("*.json")))
if (input_count, gt_count) != (2000, 2000):
    raise RuntimeError(f"Synthetic count mismatch: input={input_count}, gt={gt_count}")

qa = json.loads(
    (synthetic / "qa" / "validation_report.json").read_text(encoding="utf-8")
)
if qa.get("passed") is not True:
    raise RuntimeError("Synthetic QA did not pass")

expected = "66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a"
actual = hashlib.sha256((synthetic / "manifest.jsonl").read_bytes()).hexdigest()
if actual != expected:
    raise RuntimeError(f"Manifest SHA-256 mismatch: {actual}")

db = data / "kb" / "metadata.db"
if db.stat().st_size == 0:
    raise RuntimeError("metadata.db is empty")
with sqlite3.connect(f"file:{db.as_posix()}?mode=ro", uri=True) as connection:
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }
required_tables = {"icd10", "rxnorm"}
if not required_tables.issubset(tables):
    raise RuntimeError(f"metadata.db missing tables: {sorted(required_tables - tables)}")

bge_weights = data / "models" / "bge-m3" / "model.safetensors"
if not bge_weights.is_file():
    raise FileNotFoundError(f"BGE-M3 weights missing: {bge_weights}")

qwen = data / "models" / "Qwen2.5-7B-Instruct"
qwen_revision = "a09a35458c702b33eeacc393d103063234e8bc28"
qwen_index = qwen / "model.safetensors.index.json"
qwen_shards = []
if qwen_index.is_file():
    qwen_shards = sorted(
        set(json.loads(qwen_index.read_text(encoding="utf-8"))["weight_map"].values())
    )
qwen_ready = (
    (qwen / "config.json").is_file()
    and (qwen / "tokenizer.json").is_file()
    and len(qwen_shards) == 4
    and all((qwen / shard).is_file() and (qwen / shard).stat().st_size > 0 for shard in qwen_shards)
    and (qwen / "HF_REVISION.txt").is_file()
    and (qwen / "HF_REVISION.txt").read_text(encoding="utf-8").strip()
    == qwen_revision
)
print("DATA CHECK PASSED")
print("Qwen reranker ready:", qwen_ready)
```

`Qwen reranker ready: False` không chặn build dữ liệu, NER hoặc BGE, nhưng
phải bổ sung model `data/models/Qwen2.5-7B-Instruct` trước stage QLoRA vì config
đặt `local_files_only: true`.

Tải revision Qwen đã khóa trực tiếp trên Colab:

```bash
!python -m pip install -U "huggingface_hub>=0.34,<2"
!hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir /content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct
```

Sau download phải kiểm tra đủ 4 weight shard và ghi `HF_REVISION.txt` theo
[hướng dẫn Qwen/QLoRA](docs/training/QWEN_QLORA_COLAB.md). Base weights nằm ở
`data/models/Qwen2.5-7B-Instruct`; adapter sau train nằm ở
`artifacts/training/reranker/<run-name>/final`.

Sau khi cell in `DATA CHECK PASSED`:

```bash
!python -m pip install -r requirements-train.txt
!python -m pytest tests/training -q
!python -m src.training.build_datasets \
  --config configs/training/data.yaml \
  --project-root .
```

Dataset synthetic canonical hiện đã đồng bộ:

- Path: `D:\AI Race Viettel\data\synthetic_train_v1`
- 2.000 cặp input/ground-truth, seed `20260719`
- Manifest SHA-256:
  `66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a`
- QA: `passed=true`, source/ontology validation: 0 lỗi
- Codex task: `codex://threads/019f7475-c529-7113-87ee-530fcb4eac16`

Synthetic gate đã đạt. Production build hiện vẫn cố ý dừng ở 8 candidate thuộc
trusted 101–180 chưa tồn tại trong `metadata.db`; cần xử lý có provenance trước
khi bắt đầu train.

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
│   ├── synthetic_train_v1/    # 2.000 cặp synthetic canonical đã QA
│   ├── dev/                   # Trusted dev 101–180 và holdout 181–200
│   ├── kb/                    # metadata.db, ICD-10/RxNorm và retrieval indexes
│   ├── models/                # Base model local cho BGE-M3/Qwen
│   ├── training/              # Dataset build có fingerprint, được sinh lại
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

Code trên GitHub không chứa dataset hoặc model lớn. Nguồn canonical là thư mục
[Google Drive `data`](https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link).
Folder phải được chia sẻ ở chế độ “Anyone with the link” để Colab tải được.

### 1. Cài đặt công cụ `gdown`

Yêu cầu `gdown>=6.0.0` để tải đệ quy thư mục có hơn 50 file:

```bash
python -m pip install -U "gdown>=6.0.0"
```

### 2. Tải toàn bộ dữ liệu vào `data/`

Chạy từ thư mục gốc của một fresh clone. Dấu `/` cuối `data/` là có chủ ý:

```bash
python -m gdown "https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link" --folder -O "data/"
```

Không chạy download vào một `data` đã có dataset khác. Hãy dùng fresh clone
hoặc backup dữ liệu cũ trước. Nếu Drive quota chặn `gdown`, tải thủ công toàn bộ
nội dung folder và giữ nguyên cấu trúc bên dưới.

### Cấu trúc dữ liệu mong đợi sau khi tải:

```text
data/
├── synthetic_train_v1/
│   ├── input/           # đúng 2.000 file .txt
│   ├── gt/              # đúng 2.000 file .json
│   ├── qa/validation_report.json
│   └── manifest.jsonl
├── dev/
├── kb/metadata.db
├── models/bge-m3/
├── models/Qwen2.5-7B-Instruct/  # bắt buộc trước QLoRA
├── input/               # chỉ inference
└── output/              # kết quả inference hiện có
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
