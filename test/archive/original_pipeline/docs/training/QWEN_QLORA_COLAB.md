# Tải Qwen2.5-7B và train QLoRA trên Colab

Hướng dẫn này dùng cho stage reranker của pipeline sau khi dataset training,
BGE adapter, hai FAISS index và frozen candidate pool đã được tạo thành công.
Không chạy Qwen trực tiếp trên `data/input`.

## 1. Model và đường dẫn đã khóa

- Hugging Face repo: `Qwen/Qwen2.5-7B-Instruct`
- Revision:
  `a09a35458c702b33eeacc393d103063234e8bc28`
- Runtime path bắt buộc:
  `/content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct`
- Config sử dụng path trên:
  `configs/training/reranker_qwen25_7b_qlora.yaml`
- Weights gốc: 4 shard safetensors, khoảng 15,23 GB.

Config đặt `local_files_only: true`. Trainer không tự tải model từ Internet và
sẽ dừng nếu runtime path phía trên chưa tồn tại.

## 2. Tải model trên Colab

Chạy sau khi clone repo và tải folder `data` theo README:

```bash
%cd /content/AI-Race-Viettel
!python -m pip install -U "huggingface_hub>=0.34,<2"
!hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --dry-run
```

Dry-run phải liệt kê `model-00001-of-00004.safetensors` đến
`model-00004-of-00004.safetensors`. Sau đó tải thật vào đúng local directory:

```bash
!hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \
  --local-dir /content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct
```

Model public không bắt buộc token. Nếu bị rate-limit, thêm `HF_TOKEN` vào
Colab Secrets rồi đăng nhập mà không ghi token vào notebook:

```python
from google.colab import userdata
from huggingface_hub import login

token = userdata.get("HF_TOKEN")
if token:
    login(token=token, add_to_git_credential=False)
```

## 3. Kiểm tra weights và ghi provenance

Cell này kiểm tra đủ mọi shard được tham chiếu bởi index, không chỉ kiểm tra
folder tồn tại:

```python
from pathlib import Path
import json

model_dir = Path(
    "/content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct"
)
revision = "a09a35458c702b33eeacc393d103063234e8bc28"
required = [
    model_dir / "config.json",
    model_dir / "tokenizer.json",
    model_dir / "tokenizer_config.json",
    model_dir / "model.safetensors.index.json",
]
missing = [str(path) for path in required if not path.is_file()]
if missing:
    raise FileNotFoundError(f"Qwen files missing: {missing}")

index = json.loads(
    (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
)
shards = sorted(set(index["weight_map"].values()))
missing_shards = [
    name
    for name in shards
    if not (model_dir / name).is_file() or (model_dir / name).stat().st_size == 0
]
if missing_shards:
    raise FileNotFoundError(f"Qwen shards missing/empty: {missing_shards}")
if len(shards) != 4:
    raise RuntimeError(f"Expected 4 Qwen shards, found {len(shards)}")

(model_dir / "HF_REVISION.txt").write_text(revision + "\n", encoding="utf-8")
print("QWEN CHECK PASSED")
print("revision:", revision)
print("shards:", shards)
print("weight GiB:", round(sum((model_dir / name).stat().st_size for name in shards) / 2**30, 2))
```

Không train nếu cell chưa in `QWEN CHECK PASSED`.

## 4. Lưu base weights trên MyDrive

Colab xóa local disk khi reset runtime. Sau lần tải đầu, copy model đã kiểm tra
vào MyDrive:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount("/content/drive")
source = Path(
    "/content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct"
)
persistent = Path(
    "/content/drive/MyDrive/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct"
)
if persistent.exists():
    raise FileExistsError(
        f"Persistent Qwen already exists; verify it instead of overwriting: {persistent}"
    )
persistent.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source, persistent)
print("Saved Qwen to", persistent)
```

Đường dẫn MyDrive trên phải được đồng bộ vào folder `data` public của dự án.
Khi đó lệnh `gdown` trong README sẽ tải Qwen cùng dataset. Nếu chưa đồng bộ,
copy từ MyDrive về local ở runtime mới:

```python
from pathlib import Path
import shutil

persistent = Path(
    "/content/drive/MyDrive/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct"
)
local = Path(
    "/content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct"
)
if not persistent.is_dir():
    raise FileNotFoundError(f"Persistent Qwen missing: {persistent}")
if local.exists():
    raise FileExistsError(f"Local Qwen already exists; verify it: {local}")
local.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(persistent, local)
print("Restored Qwen to", local)
```

Không tạo `data/models/models` và không lưu model vào Git.

## 5. Điều kiện trước QLoRA

Chọn T4 GPU và kiểm tra runtime:

```bash
!python -m pip install -r requirements-train.txt
!nvidia-smi
!python -c "import torch, bitsandbytes; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), bitsandbytes.__version__)"
```

Các artifact sau phải tồn tại:

```text
data/training/manifests/build.json
artifacts/training/embedding/trusted-final-candidate/final/
artifacts/indexes/icd10-bge-m3/
artifacts/indexes/rxnorm-bge-m3/
artifacts/training/reranker/frozen/manifest.json
```

Nếu frozen candidate pool chưa có, tạo bằng BM25-first fusion đã khóa:

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

## 6. Train QLoRA

Dry-run xác nhận model path, dataset build và frozen candidate fingerprint:

```bash
!python -m src.training.reranker.train \
  --stage synthetic \
  --dry-run
```

Train synthetic:

```bash
!python -m src.training.reranker.train \
  --stage synthetic \
  --run-dir artifacts/training/reranker/synthetic-candidate
```

Nếu Colab ngắt giữa chừng, chạy lại đúng run với `--resume`:

```bash
!python -m src.training.reranker.train \
  --stage synthetic \
  --run-dir artifacts/training/reranker/synthetic-candidate \
  --resume
```

Train một trusted fold từ synthetic adapter:

```bash
!python -m src.training.reranker.train \
  --stage trusted-fold \
  --fold 0 \
  --initial-checkpoint artifacts/training/reranker/synthetic-candidate/final \
  --run-dir artifacts/training/reranker/trusted-fold-0-candidate
```

Sau khi chọn epoch/config bằng OOF, train trusted final:

```bash
!python -m src.training.reranker.train \
  --stage trusted-final \
  --initial-checkpoint artifacts/training/reranker/synthetic-candidate/final \
  --run-dir artifacts/training/reranker/trusted-final-candidate
```

Config hiện dùng QLoRA NF4 4-bit, double quantization, FP16, sequence length
1.024, micro-batch 1, gradient accumulation 16 và gradient checkpointing. Đây
là cấu hình dành cho T4 16 GB; không tăng batch/sequence đồng thời.

## 7. Weights đầu ra nằm ở đâu

Base model không bị sửa. Trainer chỉ lưu PEFT/LoRA adapter và tokenizer:

```text
artifacts/training/reranker/synthetic-candidate/final/
artifacts/training/reranker/trusted-fold-0-candidate/final/
artifacts/training/reranker/trusted-final-candidate/final/
```

Nếu đã chạy cell mount artifact trong
[MODEL_TRAINING_COLAB.md](MODEL_TRAINING_COLAB.md), toàn bộ `artifacts/` là
symlink sang MyDrive nên checkpoint được giữ tự động. Inference final cần cả:

1. base weights tại `data/models/Qwen2.5-7B-Instruct`;
2. adapter tại `artifacts/training/reranker/trusted-final-candidate/final`.

Không promote adapter chỉ vì train xong; phải review OOF F0.5, precision,
recall floor và out-of-pool rate theo runbook chính.
