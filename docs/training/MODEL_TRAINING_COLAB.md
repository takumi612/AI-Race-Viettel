# Huấn luyện modular precision-first trên Google Colab miễn phí

Tài liệu này là runbook cho pipeline đã triển khai trong `src/training`. Mỗi
model dùng một Colab GPU runtime riêng; không giữ XLM-R, BGE-M3 và Qwen đồng
thời trong VRAM.

## 1. Điều kiện bắt buộc trước khi train

Không train nếu một trong các gate sau chưa đạt:

- `data/synthetic_train_v1/qa/validation_report.json` chưa có `"passed": true`;
- synthetic chưa đủ đúng 2.000 cặp dữ liệu;
- `data/kb/metadata.db` rỗng, thiếu bảng hoặc sai namespace;
- trusted IDs 101–180 còn code không tồn tại trong ontology;
- `data/training/manifests/build.json` chưa được tạo thành công.

`data/input` chỉ dùng cho inference/nộp bài. IDs 1–100 là pseudo-GT, không được
đưa vào gradient. IDs 181–200 là holdout, không được dùng chọn model, threshold
hay epoch.

Dataset synthetic canonical hiện tại có 2.000 cặp, seed `20260719`, QA
`passed=true` và manifest SHA-256
`66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a`.
Nguồn canonical thuộc task
`codex://threads/019f7475-c529-7113-87ee-530fcb4eac16`.

Production build đã qua synthetic nhưng còn chặn 8 candidate trusted ontology
được liệt kê trong [DATA_FOUNDATION.md](DATA_FOUNDATION.md). Không hardcode các
code đó vào DB chỉ để vượt validation.

## 2. Nguồn data và nơi lưu artifact

GitHub không chứa dataset/model lớn. Nguồn `data` canonical được chia sẻ tại:

```text
https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link
```

Sau khi tải, repo Colab phải có cấu trúc:

```text
/content/AI-Race-Viettel/data/
├── synthetic_train_v1/
│   ├── input/                    # 2.000 .txt
│   ├── gt/                       # 2.000 .json
│   ├── qa/validation_report.json
│   └── manifest.jsonl
├── dev/
├── kb/metadata.db
├── models/
│   ├── bge-m3/
│   └── Qwen2.5-7B-Instruct/      # bắt buộc trước QLoRA
└── input/                        # chỉ inference
```

MyDrive chỉ dùng để giữ build/checkpoint qua lần reset runtime:

```text
/content/drive/MyDrive/AI-Race-Viettel/
├── data/training_builds/<build-id>/
└── artifacts/
```

## 3. Khởi tạo một runtime Colab

Chọn `Runtime > Change runtime type > T4 GPU`. Dùng một fresh runtime để tránh
trộn dataset cũ, sau đó clone code và tải `data` trước khi cài dependency
training:

```bash
%cd /content
!git clone --branch develop --single-branch https://github.com/takumi612/AI-Race-Viettel.git
%cd /content/AI-Race-Viettel
!python -m pip install -U pip
!python -m pip install -U "gdown>=6.0.0"
!python -m gdown "https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link" \
  --folder -O "/content/AI-Race-Viettel/data/"
```

Phải giữ dấu `/` cuối output. `gdown` 6 tải folder lớn đệ quy và không còn tùy
chọn `--remaining-ok`. Lệnh trên ghi `dev`, `kb`, `models` và
`synthetic_train_v1` trực tiếp trong `data`; không tạo `data/data`.

Kiểm tra download bằng cell fail-fast sau:

```python
from pathlib import Path
import hashlib
import json
import sqlite3

repo = Path("/content/AI-Race-Viettel")
repo_data = repo / "data"
synthetic = repo_data / "synthetic_train_v1"
required = [
    synthetic,
    repo_data / "dev",
    repo_data / "models",
    repo_data / "kb" / "metadata.db",
]
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

db = repo_data / "kb" / "metadata.db"
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

bge_weights = repo_data / "models" / "bge-m3" / "model.safetensors"
if not bge_weights.is_file():
    raise FileNotFoundError(f"BGE-M3 weights missing: {bge_weights}")

qwen = repo_data / "models" / "Qwen2.5-7B-Instruct"
qwen_ready = (
    (qwen / "config.json").is_file()
    and (qwen / "tokenizer.json").is_file()
    and any(qwen.glob("*.safetensors"))
)
print("DATA CHECK PASSED")
print("Qwen reranker ready:", qwen_ready)
```

Không tiếp tục nếu chưa thấy `DATA CHECK PASSED`. Giá trị
`Qwen reranker ready: False` không chặn build/NER/BGE, nhưng chặn QLoRA vì
`configs/training/reranker_qwen25_7b_qlora.yaml` đặt `local_files_only: true`.

Cài dependency và kiểm tra GPU:

```bash
!python -m pip install -r requirements-train.txt
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Mount MyDrive chỉ để lưu artifact. Cell này không thay thế hoặc symlink source
dataset:

```python
from google.colab import drive

drive.mount("/content/drive")
drive_root = Path("/content/drive/MyDrive/AI-Race-Viettel")
drive_artifacts = drive_root / "artifacts"
drive_artifacts.mkdir(parents=True, exist_ok=True)
repo_artifacts = repo / "artifacts"
if repo_artifacts.is_symlink():
    if repo_artifacts.resolve() != drive_artifacts.resolve():
        raise RuntimeError(f"Wrong artifacts symlink: {repo_artifacts.resolve()}")
elif repo_artifacts.exists():
    if any(repo_artifacts.iterdir()):
        raise RuntimeError(f"Refusing to replace non-empty directory: {repo_artifacts}")
    repo_artifacts.rmdir()
    repo_artifacts.symlink_to(drive_artifacts, target_is_directory=True)
else:
    repo_artifacts.symlink_to(drive_artifacts, target_is_directory=True)
```

## 4. Build dữ liệu một lần

```bash
!python -m pytest tests/training -q
!python -m src.training.build_datasets \
  --config configs/training/data.yaml \
  --project-root .
```

Build local trước để giữ atomic rename. Production build hiện còn chặn 8
candidate trusted ontology; không bypass gate. Chỉ copy lên MyDrive sau khi
manifest hợp lệ:

```python
import json
import shutil

local_training = repo_data / "training"
manifest = json.loads(
    (local_training / "manifests" / "build.json").read_text(encoding="utf-8")
)
drive_data = drive_root / "data"
archive = drive_data / "training_builds" / manifest["build_id"]
if archive.exists():
    raise FileExistsError(f"Training build already exists: {archive}")
archive.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(local_training, archive)
print("BUILD_ID =", manifest["build_id"])
```

## 5. Quy tắc stage và resume

Thứ tự chung:

1. `synthetic`;
2. năm lần `trusted-fold`, mỗi lần khởi tạo từ cùng synthetic checkpoint;
3. chọn hyperparameter/epoch bằng out-of-fold exact F0.5, precision là
   tiebreaker và phải đạt recall floor;
4. `trusted-final` khởi tạo lại từ synthetic checkpoint và train toàn bộ
   101–180;
5. promote artifact final sau khi review metric.

`--resume` chỉ tiếp tục đúng cùng run, config, dataset build và initial
checkpoint fingerprint. Nó không có nghĩa là bắt đầu stage mới.

## 6. XLM-R NER

Dry-run:

```bash
!python -m src.training.ner.train \
  --stage synthetic \
  --dry-run
```

Train synthetic:

```bash
!python -m src.training.ner.train \
  --stage synthetic \
  --run-dir artifacts/training/ner/synthetic-candidate
```

Resume sau khi Colab ngắt:

```bash
!python -m src.training.ner.train \
  --stage synthetic \
  --run-dir artifacts/training/ner/synthetic-candidate \
  --resume
```

Trusted fold, thay `0` lần lượt bằng `0..4`:

```bash
!python -m src.training.ner.train \
  --stage trusted-fold \
  --fold 0 \
  --initial-checkpoint artifacts/training/ner/synthetic-candidate/final \
  --run-dir artifacts/training/ner/trusted-fold-0-candidate
```

Final:

```bash
!python -m src.training.ner.train \
  --stage trusted-final \
  --initial-checkpoint artifacts/training/ner/synthetic-candidate/final \
  --run-dir artifacts/training/ner/trusted-final-candidate
```

Metric chính là exact-span, exact-type F0.5. Không chọn checkpoint bằng
synthetic validation.

## 7. BGE-M3 LoRA và FAISS

Synthetic:

```bash
!python -m src.training.embedding.train \
  --stage synthetic \
  --run-dir artifacts/training/embedding/synthetic-candidate
```

Trusted fold:

```bash
!python -m src.training.embedding.train \
  --stage trusted-fold \
  --fold 0 \
  --initial-checkpoint artifacts/training/embedding/synthetic-candidate/final \
  --run-dir artifacts/training/embedding/trusted-fold-0-candidate
```

Final:

```bash
!python -m src.training.embedding.train \
  --stage trusted-final \
  --initial-checkpoint artifacts/training/embedding/synthetic-candidate/final \
  --run-dir artifacts/training/embedding/trusted-final-candidate
```

Sinh lại embedding bằng đúng adapter final. Batch mặc định là 32 trên GPU; nếu
T4 OOM, giảm xuống 16 hoặc 8:

```bash
!python -m src.retrieval.generate_embeddings \
  --model BGE-M3 \
  --table icd10 \
  --model-path artifacts/training/embedding/trusted-final-candidate/final \
  --output-dir artifacts/index-build \
  --db data/kb/metadata.db \
  --batch-size 16

!python -m src.retrieval.generate_embeddings \
  --model BGE-M3 \
  --table rxnorm \
  --model-path artifacts/training/embedding/trusted-final-candidate/final \
  --output-dir artifacts/index-build \
  --db data/kb/metadata.db \
  --batch-size 16
```

Build hai index có manifest ràng buộc adapter, DB, vectors, codes và FAISS:

```bash
!python -m src.retrieval.build_faiss_index \
  --model BGE-M3 \
  --table icd10 \
  --embedding-dir artifacts/index-build \
  --output-dir artifacts/indexes/icd10-bge-m3 \
  --adapter-dir artifacts/training/embedding/trusted-final-candidate/final \
  --base-model data/models/bge-m3 \
  --db data/kb/metadata.db

!python -m src.retrieval.build_faiss_index \
  --model BGE-M3 \
  --table rxnorm \
  --embedding-dir artifacts/index-build \
  --output-dir artifacts/indexes/rxnorm-bge-m3 \
  --adapter-dir artifacts/training/embedding/trusted-final-candidate/final \
  --base-model data/models/bge-m3 \
  --db data/kb/metadata.db
```

Runtime từ chối semantic index nếu DB hoặc adapter không đúng fingerprint và
fail closed về BM25. Fusion mặc định:

```text
score = 0.75 * normalized_BM25 + 0.25 * normalized_semantic
```

## 8. Freeze candidate pool

Chỉ freeze sau khi BGE adapter và cả hai index đã khóa:

```bash
!python -m src.training.reranker.train \
  --freeze-candidates \
  --icd-index-dir artifacts/indexes/icd10-bge-m3 \
  --rxnorm-index-dir artifacts/indexes/rxnorm-bge-m3 \
  --embedding-model-artifact artifacts/training/embedding/trusted-final-candidate/final \
  --alpha 0.75 \
  --internal-top-k 20 \
  --candidate-top-k 10
```

Output mặc định là `artifacts/training/reranker/frozen`. Nếu gold nằm ngoài
top-k, record được ghi là retrieval miss và không được chèn gold vào pool.

## 9. Qwen2.5-7B QLoRA

Synthetic:

```bash
!python -m src.training.reranker.train \
  --stage synthetic \
  --run-dir artifacts/training/reranker/synthetic-candidate
```

Trusted fold:

```bash
!python -m src.training.reranker.train \
  --stage trusted-fold \
  --fold 0 \
  --initial-checkpoint artifacts/training/reranker/synthetic-candidate/final \
  --run-dir artifacts/training/reranker/trusted-fold-0-candidate
```

Final:

```bash
!python -m src.training.reranker.train \
  --stage trusted-final \
  --initial-checkpoint artifacts/training/reranker/synthetic-candidate/final \
  --run-dir artifacts/training/reranker/trusted-final-candidate
```

QLoRA dùng NF4 4-bit, FP16, all-linear LoRA, sequence 1.024 và micro-batch 1.
Loss chỉ tính trên completion; prompt labels là `-100`. Artifact không hợp lệ
nếu raw out-of-pool rate khác 0.

## 10. Review và promote artifact

Trainer tạo candidate, không tự tuyên bố model tốt. Sau khi gom năm fold và
chọn cấu hình precision-first, chuẩn bị JSON metric đã review. Ví dụ:

```json
{
  "precision": 0.91,
  "recall": 0.78,
  "f0_5": 0.88,
  "selection_source": "trusted_oof_5fold"
}
```

Promote final NER:

```bash
!python -m src.training.promote \
  --run-dir artifacts/training/ner/trusted-final-candidate \
  --artifact-dir artifacts/locked/ner-v1 \
  --metrics-json reports/ner-oof-selected.json \
  --status locked
```

Lặp lại cho embedding và reranker. Nếu không truyền `--metrics-json`, CLI đọc
`training_metrics.json` trong run. Với `trusted-final`, nên truyền báo cáo OOF
đã dùng để chọn cấu hình thay vì train loss.

## 11. Cấu hình inference GPU

Ví dụ phần config cần bật sau khi artifact/index đã được khóa:

```json
{
  "retrieval": {
    "alpha": 0.75,
    "internal_top_k": 20,
    "embedding_model_type": "BGE-M3",
    "embedding_model_artifact": "artifacts/locked/embedding-v1/final",
    "icd_index_artifact": "artifacts/indexes/icd10-bge-m3",
    "rxnorm_index_artifact": "artifacts/indexes/rxnorm-bge-m3"
  },
  "ner": {
    "mode": "hybrid",
    "model_artifact": "artifacts/locked/ner-v1",
    "model_threshold": 0.7
  },
  "reranker": {
    "enabled": true,
    "backend": "local_transformers",
    "model_artifact": "artifacts/locked/reranker-v1",
    "max_new_tokens": 64,
    "timeout_seconds": 30
  }
}
```

Không bật model chỉ vì train xong. Trước hết phải chạy trusted OOF benchmark,
khóa threshold/config, sau đó mới đánh giá holdout 181–200 đúng một lần.

## 12. Kiểm tra cuối

```bash
!python -m pytest tests/training -q
!python -m pytest -q
!python src/metrics.py test
!python scripts/audit_overrides.py --scan-paths src scripts
!python scripts/audit_overrides.py \
  --db data/kb/metadata.db \
  --overrides src/resources/verified_overrides.json
!python -m src.training.ner.train --help
!python -m src.training.embedding.train --help
!python -m src.training.reranker.train --help
!python -m src.training.promote --help
```

Sau khi inference:

```bash
!python src/pipeline/main.py \
  --input data/input \
  --output data/output \
  --config reports/locked-config.json
```

Đánh giá trusted/holdout bằng `src.evaluation.benchmark` và locked config như
phần Trusted benchmark policy trong README; không chấm lẫn `data/input` với
`data/dev`. Không chạy holdout lặp lại để tune. Không xóa dữ liệu qua symlink
Drive.
