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

Synthetic hiện chỉ được dùng khi session sinh dữ liệu đã hoàn tất validation
và thư mục kết quả được promote thành `data/synthetic_train_v1`. Không đổi tên
thư mục `.failed-validation` chỉ để vượt gate.

## 3. Chạy local

Từ root của repository:

```powershell
python -m pip install -r requirements-train.txt
python -m pytest tests/training -q
python -m src.training.build_datasets `
  --config configs/training/data.yaml `
  --project-root .
```

Lần chạy đầu từ chối nếu synthetic chưa pass hoặc 8 mã ontology phía trên
chưa được giải quyết. Đây là hành vi mong muốn của pipeline precision-first.

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

Clone code sạch và mount Drive:

```python
from google.colab import drive

drive.mount("/content/drive")
```

```bash
%cd /content
!git clone <GITHUB_REPOSITORY_URL> AI-Race-Viettel
%cd /content/AI-Race-Viettel
!git checkout develop
!python -m pip install -r requirements-train.txt
```

Drive nên có cấu trúc:

```text
MyDrive/AI-Race-Viettel/data/
├── synthetic_train_v1/
├── dev/
├── kb/metadata.db
├── models/                 # dùng ở phase training model
└── input/                  # chỉ dùng inference cuối
```

Đoạn setup an toàn dưới đây không tạo đường dẫn lặp
`data/models/models`. Nó chỉ thay placeholder `.gitkeep`; nếu đích có dữ liệu
khác thì dừng:

```python
from pathlib import Path
import shutil

repo = Path("/content/AI-Race-Viettel")
drive_data = Path("/content/drive/MyDrive/AI-Race-Viettel/data")
repo_data = repo / "data"
repo_data.mkdir(parents=True, exist_ok=True)

def link_source_directory(name: str) -> None:
    src = drive_data / name
    dst = repo_data / name
    if not src.is_dir():
        raise FileNotFoundError(f"Missing Drive directory: {src}")
    if dst.is_symlink():
        if dst.resolve() != src.resolve():
            raise RuntimeError(f"Wrong existing symlink: {dst} -> {dst.resolve()}")
        return
    if dst.exists():
        unexpected = {item.name for item in dst.iterdir()} - {".gitkeep"}
        if unexpected:
            raise RuntimeError(f"Refusing to replace non-placeholder: {dst}")
        shutil.rmtree(dst)
    dst.symlink_to(src, target_is_directory=True)

link_source_directory("synthetic_train_v1")
link_source_directory("dev")

src_db = drive_data / "kb" / "metadata.db"
dst_db = repo_data / "kb" / "metadata.db"
if not src_db.is_file() or src_db.stat().st_size == 0:
    raise FileNotFoundError(f"metadata.db missing or empty: {src_db}")
dst_db.parent.mkdir(parents=True, exist_ok=True)
if dst_db.is_symlink():
    if dst_db.resolve() != src_db.resolve():
        raise RuntimeError(f"Wrong metadata.db symlink: {dst_db}")
elif dst_db.exists():
    raise RuntimeError(f"Refusing to overwrite existing file: {dst_db}")
else:
    dst_db.symlink_to(src_db)
```

Chạy build:

```bash
!python -m pytest tests/training -q
!python -m src.training.build_datasets \
  --config configs/training/data.yaml \
  --project-root .
```

Output được build trên disk Colab để giữ atomic rename. Sau khi manifest hợp
lệ, copy sang thư mục versioned trên Drive:

```python
import json

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

Không xóa thư mục qua symlink trong notebook. Checkpoint NER/BGE/Qwen ở các
phase sau phải ghi trực tiếp vào thư mục versioned trên Drive và resume theo
manifest.

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
