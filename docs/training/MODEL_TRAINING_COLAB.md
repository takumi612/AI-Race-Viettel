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

Tại trạng thái dữ liệu đã audit gần nhất, build còn chặn 8 candidate ontology
được liệt kê trong [DATA_FOUNDATION.md](DATA_FOUNDATION.md). Không hardcode các
code đó vào DB chỉ để vượt validation.

## 2. Cấu trúc Drive

```text
MyDrive/AI-Race-Viettel/
├── data/
│   ├── synthetic_train_v1/
│   ├── dev/
│   ├── kb/metadata.db
│   ├── models/
│   │   ├── bge-m3/
│   │   └── Qwen2.5-7B-Instruct/
│   └── training_builds/<build-id>/
└── artifacts/
```

Không tạo đường dẫn `data/models/models`. Nguồn chính xác của model là:

```text
/content/drive/MyDrive/AI-Race-Viettel/data/models
```

## 3. Khởi tạo một runtime Colab

Chọn `Runtime > Change runtime type > T4 GPU`, sau đó:

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
%cd /content
!git clone <GITHUB_REPOSITORY_URL> AI-Race-Viettel
%cd /content/AI-Race-Viettel
!git checkout develop
!python -m pip install -U pip
!python -m pip install -r requirements-train.txt
```

Kiểm tra GPU và phiên bản:

```bash
!nvidia-smi
!python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Tạo link an toàn. Đoạn này không xóa thư mục Drive và từ chối ghi đè dữ liệu
không phải placeholder:

```python
from pathlib import Path
import shutil

repo = Path("/content/AI-Race-Viettel")
drive_root = Path("/content/drive/MyDrive/AI-Race-Viettel")
drive_data = drive_root / "data"
repo_data = repo / "data"
repo_data.mkdir(parents=True, exist_ok=True)

def link_directory(src: Path, dst: Path) -> None:
    if not src.is_dir():
        raise FileNotFoundError(f"Missing Drive directory: {src}")
    if dst.is_symlink():
        if dst.resolve() != src.resolve():
            raise RuntimeError(f"Wrong symlink: {dst} -> {dst.resolve()}")
        return
    if dst.exists():
        unexpected = {item.name for item in dst.iterdir()} - {".gitkeep"}
        if unexpected:
            raise RuntimeError(f"Refusing to replace non-placeholder: {dst}")
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src, target_is_directory=True)

def link_file(src: Path, dst: Path) -> None:
    if not src.is_file() or src.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if dst.resolve() != src.resolve():
            raise RuntimeError(f"Wrong symlink: {dst} -> {dst.resolve()}")
        return
    if dst.exists():
        raise RuntimeError(f"Refusing to overwrite: {dst}")
    dst.symlink_to(src)

link_directory(drive_data / "models", repo_data / "models")
link_file(drive_data / "kb" / "metadata.db", repo_data / "kb" / "metadata.db")

drive_artifacts = drive_root / "artifacts"
drive_artifacts.mkdir(parents=True, exist_ok=True)
link_directory(drive_artifacts, repo / "artifacts")
```

Ở runtime build data đầu tiên, link thêm nguồn:

```python
link_directory(
    drive_data / "synthetic_train_v1",
    repo_data / "synthetic_train_v1",
)
link_directory(drive_data / "dev", repo_data / "dev")
```

Ở runtime model sau này, link build đã khóa:

```python
build_id = "<BUILD_ID_FROM_MANIFEST>"
link_directory(
    drive_data / "training_builds" / build_id,
    repo_data / "training",
)
```

## 4. Build dữ liệu một lần

```bash
!python -m pytest tests/training -q
!python -m src.training.build_datasets \
  --config configs/training/data.yaml \
  --project-root .
```

Build local trước để giữ atomic rename. Chỉ copy lên Drive sau khi manifest hợp
lệ:

```python
import json

local_training = repo_data / "training"
manifest = json.loads(
    (local_training / "manifests" / "build.json").read_text(encoding="utf-8")
)
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
