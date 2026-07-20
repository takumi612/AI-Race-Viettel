# Nền tảng dữ liệu huấn luyện precision-first

Tài liệu này mô tả Phase A của kiến trúc huấn luyện modular: đọc và kiểm
tra nguồn, khóa split, tạo seed data cho NER/BGE-M3/Qwen và ghi manifest có
fingerprint. Phase A **không tự huấn luyện model**; các training loop NER,
BGE-M3 adapter và Qwen QLoRA thuộc các phase tiếp theo.

## 1. Chính sách dữ liệu đã khóa

| Nguồn | ID/số lượng | Vai trò | Được dùng gradient? |
|---|---:|---|---|
| Synthetic đã QA | 2.000 | 1.700 train + 300 validation theo group | Có |
| Pseudo-GT | 1–100 | Chỉ diagnostic, không được build | Không |
| Trusted GT | 101–180 | 5 fold, mỗi fold 16 hồ sơ | Có, theo fold |
| Final holdout | 181–200 | Đánh giá sau khi khóa config | Không |
| `data/input` | 100 public inputs | Inference/submission cuối | Không |

Các invariant fail-closed:

- Chỉ nhận synthetic ở `data/synthetic_train_v1`; mọi thư mục có hậu tố
  `.failed-validation` bị từ chối trước khi đọc file.
- Synthetic phải có đúng 2.000 cặp `.txt`/`.json` và
  `qa/validation_report.json` phải chứa `"passed": true`.
- Span dùng Unicode half-open `[start, end)` và phải thỏa
  `text[start:end] == entity.text`.
- Chỉ `CHẨN_ĐOÁN` được mang ICD-10 và chỉ `THUỐC` được mang RxNorm; mã
  không có trong `metadata.db` làm build thất bại.
- Split theo `profile_id`/`split_group`, đồng thời chặn cùng content hash
  xuất hiện ở hai partition.
- Pseudo-GT 1–100 không được load nên không thể vô tình lọt vào seed data.
- Reranker seed chỉ có ground truth; chưa tạo candidate pool cho tới khi
  retriever và fingerprint của retriever được khóa.

## 2. Trạng thái dữ liệu cục bộ cần xử lý

Lần audit gần nhất trên `data/dev` đã tìm và xử lý có audit:

- 37 entity trùng hoàn toàn được loại ở source adapter;
- 1 candidate chỉ chứa khoảng trắng được đổi thành không có candidate;
- 8 candidate trong trusted IDs 101–180 không tồn tại trong
  `data/kb/metadata.db` hiện tại và **vẫn bị chặn**:

| Namespace | ID | Mã |
|---|---:|---|
| ICD-10 | 118 | `S06.20` |
| ICD-10 | 125 | `B18.10` |
| ICD-10 | 172 | `M05.80` |
| ICD-10 | 173 | `M10.00` (2 entity) |
| RxNorm | 143 | `2586402` |
| RxNorm | 157 | `20577` |
| RxNorm | 158 | `678` |

Không nên thêm các mã này vào DB bằng hardcode. Cần đối chiếu nguồn ontology:
nếu mã GT sai thì sửa GT có provenance; nếu DB thiếu thì rebuild DB từ nguồn
chuẩn rồi chạy lại audit. NER span vẫn dùng được, nhưng mã sai không được đưa
vào embedding/reranker gradient.

Synthetic canonical đã được promote và xác minh tại
`D:\AI Race Viettel\data\synthetic_train_v1`:

- đúng 2.000 cặp `input/*.txt` và `gt/*.json`;
- deterministic seed `20260719`;
- `qa/validation_report.json`: `passed=true`, không error/warning;
- manifest SHA-256:
  `66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a`;
- source loader + ontology validator của training: 2.000 record, 0 finding;
- split seed `20260719`: 1.700 train, 300 validation, group overlap bằng 0;
- projection: 2.000 NER record, 2.736 embedding seed và 2.736 reranker seed.

Task sinh dữ liệu canonical:
`codex://threads/019f7475-c529-7113-87ee-530fcb4eac16`.

## 3. Chạy local

Từ root của repository:

```powershell
python -m pip install -r requirements-train.txt
python -m pytest tests/training -q
python -m src.training.build_datasets `
  --config configs/training/data.yaml `
  --project-root .
```

Synthetic hiện đã qua gate. Lần chạy production đã đi qua toàn bộ 2.000
synthetic record và dừng ở 8 mã trusted ontology phía trên. Đây là hành vi
mong muốn của pipeline precision-first.

Nếu `data/training` đã tồn tại, kiểm tra
`data/training/manifests/build.json` trước. Chỉ sau đó mới thay thế nguyên tử:

```powershell
python -m src.training.build_datasets `
  --config configs/training/data.yaml `
  --project-root . `
  --replace
```

`--allow-non-production-count` chỉ dành cho fixture/smoke test. Flag được ghi
vào manifest và không thể thay đổi ranh giới trusted/holdout.

## 4. Output

```text
data/training/
├── canonical/records.jsonl
├── splits/assignments.jsonl
├── ner/records.jsonl
├── embedding/seeds.jsonl
├── reranker/seeds.jsonl
└── manifests/build.json
```

Manifest chứa:

- normalized config SHA-256;
- SHA-256 code thực tế của `src/training` và dependency/config;
- Git commit, trạng thái dirty và status SHA-256;
- fingerprint riêng cho synthetic/trusted/holdout;
- `metadata.db` SHA-256;
- split SHA-256;
- count và SHA-256 của từng artifact.

Build ghi vào `data/.training-build-<uuid>` rồi đọc lại toàn bộ JSONL để xác
nhận count/hash. Output cũ chỉ được đổi tên sau khi temporary build đã hợp lệ;
replace thất bại sẽ rollback output cũ.

## 5. Google Colab miễn phí

GitHub chỉ chứa code. Clone repo rồi tải toàn bộ `data` canonical từ
[Drive của dự án](https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link)
trước khi cài dependency training:

```bash
%cd /content
!git clone --branch develop --single-branch https://github.com/takumi612/AI-Race-Viettel.git
%cd /content/AI-Race-Viettel
!python -m pip install -U "gdown>=6.0.0"
!python -m gdown "https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link" \
  --folder -O "/content/AI-Race-Viettel/data/"
```

Output phải có dấu `/` cuối. Với `gdown` 6, không thêm tùy chọn cũ
`--remaining-ok`. Fresh clone sau download phải có cấu trúc:

```text
data/
├── synthetic_train_v1/
│   ├── input/                 # 2.000 .txt
│   ├── gt/                    # 2.000 .json
│   ├── qa/validation_report.json
│   └── manifest.jsonl
├── dev/
├── kb/metadata.db
├── models/bge-m3/
├── models/Qwen2.5-7B-Instruct/  # cần cho QLoRA
└── input/                      # chỉ inference cuối
```

Không download vào một checkout đã có dataset khác. Không đặt output là
`data/models`, vì nguồn Drive đã chứa cả folder `models` và sẽ tạo sai đường
dẫn `data/models/models`.

Cell preflight dưới đây phải in `DATA CHECK PASSED` trước khi build:

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

`Qwen reranker ready: False` chỉ chặn stage QLoRA. Config reranker dùng
`local_files_only: true`, vì vậy phải bổ sung đúng local model trước stage đó.
Xem [QWEN_QLORA_COLAB.md](QWEN_QLORA_COLAB.md) để tải revision đã khóa, kiểm
tra đủ shard, lưu base weights và chạy QLoRA.

Cài dependency, test và chạy build:

```bash
!python -m pip install -r requirements-train.txt
!python -m pytest tests/training -q
!python -m src.training.build_datasets \
  --config configs/training/data.yaml \
  --project-root .
```

Build hiện vẫn dừng có chủ ý ở 8 candidate trusted ontology đã liệt kê phía
trên. Chỉ sau khi xử lý chúng có provenance và build manifest hợp lệ mới lưu
output/checkpoint lên MyDrive. Mount Drive ở bước này, không dùng symlink làm
nguồn dataset:

```python
from google.colab import drive
import json
import shutil

drive.mount("/content/drive")
drive_data = Path("/content/drive/MyDrive/AI-Race-Viettel/data")
local_output = repo / "data" / "training"
manifest = json.loads(
    (local_output / "manifests" / "build.json").read_text(encoding="utf-8")
)
drive_output = (
    drive_data / "training_builds" / manifest["build_id"]
)
if drive_output.exists():
    raise FileExistsError(f"Build already archived: {drive_output}")
drive_output.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(local_output, drive_output)
print(drive_output)
```

Checkpoint NER/BGE/Qwen phải được copy hoặc ghi vào một thư mục versioned trên
MyDrive để tồn tại qua lần reset runtime và resume đúng manifest.

## 6. Verification trước khi bàn giao

```powershell
python -m pytest tests/training -q
python -m pytest -q
python src/metrics.py test
python scripts/audit_overrides.py --scan-paths src scripts
python scripts/audit_overrides.py `
  --db data/kb/metadata.db `
  --overrides src/resources/verified_overrides.json
git diff --check
git status --short
```

`tests/` không nằm trong production path scan vì test config cố ý chứa các
đường dẫn tuyệt đối giả để xác nhận loader từ chối chúng.

Một Phase A build chỉ được coi là production khi không dùng
`--allow-non-production-count`, report synthetic pass, mọi namespace code hợp
lệ và manifest/artifact round-trip thành công.
