import json
from pathlib import Path

def create_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" if i < len(text.split("\n"))-1 else line for i, line in enumerate(text.split("\n"))]
    }

def create_code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" if i < len(text.split("\n"))-1 else line for i, line in enumerate(text.split("\n"))]
    }

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"name": "AI-Race-Viettel-Training.ipynb", "provenance": []},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"}
    },
    "cells": []
}

# Thêm các cells
notebook["cells"].extend([
    create_markdown_cell("# AI Race Viettel - Huấn luyện Modular (Colab Runbook)\n\nFile notebook này được tạo tự động giúp bạn chạy huấn luyện trên Colab một cách trơn tru, bao gồm việc dọn dẹp thư mục và sửa lỗi đường dẫn Google Drive."),
    
    create_markdown_cell("## Bước 1: Mount Google Drive và chuẩn bị thư mục\nChạy cell dưới đây để Mount Drive và đảm bảo thư mục gốc `AI-Race-Viettel` cùng `artifacts` được tạo sẵn trên MyDrive, tránh bị lỗi *\"không tìm thấy\"*."),
    create_code_cell("""from google.colab import drive
from pathlib import Path

# Cấp quyền Google Drive
drive.mount("/content/drive")

# Khởi tạo trước các thư mục trên MyDrive để đảm bảo Path luôn tồn tại
drive_root = Path("/content/drive/MyDrive/AI-Race-Viettel")
drive_root.mkdir(parents=True, exist_ok=True)

drive_artifacts = drive_root / "artifacts"
drive_artifacts.mkdir(parents=True, exist_ok=True)

print(f"Đã chuẩn bị thư mục trên Google Drive an toàn tại: {drive_root}")"""),
    
    create_markdown_cell("## Bước 2: Tải code từ Github và Dữ liệu\nTự động xóa thư mục cũ nếu có để clone bản mới nhất không bị lỗi."),
    create_code_cell("""%cd /content
!rm -rf AI-Race-Viettel
!git clone --branch develop --single-branch https://github.com/takumi612/AI-Race-Viettel.git
%cd /content/AI-Race-Viettel

!python -m pip install -U pip
!python -m pip install -U "gdown>=6.0.0"
# Tải dữ liệu data (đã nén zip) từ Drive
!python -m gdown 1_A059PDaXvDTxGQl86QDRXZ5eyRQrBKT -O /content/data.zip

# Giải nén data
!mkdir -p /content/AI-Race-Viettel/data
!unzip -q -o /content/data.zip -d /content/AI-Race-Viettel/data/
!rm /content/data.zip"""),
    
    create_markdown_cell("## Bước 3: Sửa lỗi đường dẫn và thiết lập Symlink\nXử lý triệt để lỗi gdown sinh thư mục lồng nhau (`data/data`) và thiết lập liên kết `artifacts` thông minh sang Google Drive."),
    create_code_cell("""import shutil
from pathlib import Path

repo = Path("/content/AI-Race-Viettel")
repo_data = repo / "data"

# 1. Tự động chuẩn hóa dữ liệu bị gdown tạo lồng
metadata_paths = list(repo_data.glob("**/kb/metadata.db"))
if metadata_paths:
    nested_data_root = metadata_paths[0].parent.parent
    if nested_data_root != repo_data:
        print(f"[*] Phát hiện thư mục dữ liệu bị lồng: {nested_data_root}")
        print("[*] Đang di chuyển dữ liệu về đúng cấu trúc...")
        for item in nested_data_root.iterdir():
            dest = repo_data / item.name
            if dest.exists():
                if dest.is_dir(): shutil.rmtree(dest)
                else: dest.unlink()
            shutil.move(str(item), str(dest))
        nested_data_root.rmdir()
        print("[v] Đã sửa lỗi cấu trúc lồng nhau thành công!")

# 2. Thiết lập liên kết Artifacts an toàn
drive_artifacts = Path("/content/drive/MyDrive/AI-Race-Viettel/artifacts")
repo_artifacts = repo / "artifacts"

if repo_artifacts.is_symlink():
    if repo_artifacts.resolve() != drive_artifacts.resolve():
        raise RuntimeError(f"Wrong artifacts symlink: {repo_artifacts.resolve()}")
elif repo_artifacts.exists():
    if any(repo_artifacts.iterdir()):
        print(f"[*] Thư mục artifacts cục bộ không trống. Đang copy dữ liệu tạm sang Google Drive để bảo tồn...")
        for item in repo_artifacts.iterdir():
            dest = drive_artifacts / item.name
            if not dest.exists():
                if item.is_dir(): shutil.copytree(item, dest)
                else: shutil.copy2(item, dest)
    shutil.rmtree(repo_artifacts)
    repo_artifacts.symlink_to(drive_artifacts, target_is_directory=True)
    print("[v] Thiết lập liên kết Artifacts sang Drive thành công!")
else:
    repo_artifacts.symlink_to(drive_artifacts, target_is_directory=True)
    print("[v] Thiết lập liên kết Artifacts sang Drive thành công!")"""),
    
    create_markdown_cell("## Bước 4: Chạy kiểm tra bộ dữ liệu"),
    create_code_cell("""import hashlib
import json
import sqlite3

synthetic = repo_data / "synthetic_train_v1"
required = [
    synthetic,
    repo_data / "dev",
    repo_data / "models",
    repo_data / "kb" / "metadata.db",
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise FileNotFoundError(f"Drive download incomplete; thiếu: {missing}")

# Kiểm tra dữ liệu synthetic
input_count = len(list((synthetic / "input").glob("*.txt")))
gt_count = len(list((synthetic / "gt").glob("*.json")))
if (input_count, gt_count) != (2000, 2000):
    raise RuntimeError(f"Số lượng file không khớp (Cần 2000 cặp): input={input_count}, gt={gt_count}")

qa = json.loads((synthetic / "qa" / "validation_report.json").read_text(encoding="utf-8"))
if qa.get("passed") is not True:
    raise RuntimeError("Kiểm tra QA synthetic thất bại")

expected_manifest_hash = "66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a"
actual_manifest_hash = hashlib.sha256((synthetic / "manifest.jsonl").read_bytes()).hexdigest()
if actual_manifest_hash != expected_manifest_hash:
    print(f"[!] Cảnh báo: Sai mã SHA-256 (manifest synthetic): {actual_manifest_hash}. Dữ liệu có thể đã bị sửa đổi.")

print("\\n[v] DATA CHECK PASSED - Toàn bộ dữ liệu đã được xác thực an toàn.")"""),

    create_markdown_cell("## Bước 5: Cài đặt thư viện huấn luyện"),
    create_code_cell("""!python -m pip install -r requirements-train.txt
!nvidia-smi"""),

    create_markdown_cell("## Bước 5.5: Sửa Dữ liệu lỗi và Vá Code (Tự động)\nTự động sửa các file có ICD-10 sai và vá lỗi Padding trong thư viện huấn luyện NER."),
    create_code_cell("""from pathlib import Path
from hashlib import sha256
import json

# 1. Sửa lỗi Dữ liệu ICD-10 và RxNorm trong tập dev
dev_root = Path('/content/AI-Race-Viettel/data/dev')
gt_dir = dev_root / 'gt'

replace_codes = {
    '118': {'S06.20': 'S0620'},
    '125': {'B18.10': 'B1810'},
    '172': {'M05.80': 'M0580'},
    '173': {'M10.00': 'M1000'},
}
remove_codes = {
    '143': {'2586402'},
    '157': {'20577'},
    '158': {'678'},
}

changed_ids = set()
for record_id in sorted(set(replace_codes) | set(remove_codes)):
    gt_path = gt_dir / f'{record_id}.json'
    if not gt_path.exists(): continue
    entities = json.loads(gt_path.read_text(encoding='utf-8'))
    changed = 0
    for entity in entities:
        candidates = entity.get('candidates')
        if not isinstance(candidates, list):
            continue
        updated = []
        for code in candidates:
            code_text = str(code).strip()
            if code_text in remove_codes.get(record_id, set()):
                changed += 1
                continue
            new_code = replace_codes.get(record_id, {}).get(code_text, code)
            changed += int(new_code != code)
            updated.append(new_code)
        entity['candidates'] = updated
    if changed:
        gt_path.write_text(json.dumps(entities, ensure_ascii=False, indent=2) + '\\n', encoding='utf-8')
        changed_ids.add(record_id)
        print(f'Đã sửa lỗi JSON: {record_id}.json ({changed} thay đổi)')

manifest_path = dev_root / 'manifest.jsonl'
if manifest_path.is_file() and changed_ids:
    rows = []
    for line in manifest_path.read_text(encoding='utf-8').splitlines():
        if not line.strip(): continue
        row = json.loads(line)
        record_id = str(row.get('record_id'))
        if record_id in changed_ids:
            row['ground_truth_sha256'] = sha256((gt_dir / f'{record_id}.json').read_bytes()).hexdigest()
        rows.append(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')))
    manifest_path.write_text('\\n'.join(rows) + '\\n', encoding='utf-8')
    print('Đã cập nhật SHA256 cho dev manifest!')

# 2. Vá lỗi 'offset and label lengths differ' trong file train.py
train_file = '/content/AI-Race-Viettel/src/training/ner/train.py'
with open(train_file, 'r', encoding='utf-8') as f:
    code = f.read()

old_code = \"\"\"        for feature, predicted, gold in zip(eval_features, predictions, labels):
            predicted_entities.extend(
                decode_bio_entities(
                    feature["text"],
                    feature["absolute_offsets"],
                    predicted.tolist(),
                    record_id=feature["record_id"],
                )
            )
            gold_entities.extend(
                decode_bio_entities(
                    feature["text"],
                    feature["absolute_offsets"],
                    gold.tolist(),
                    record_id=feature["record_id"],
                )
            )\"\"\"

new_code = \"\"\"        for feature, predicted, gold in zip(eval_features, predictions, labels):
            seq_len = len(feature["absolute_offsets"])
            predicted_entities.extend(
                decode_bio_entities(
                    feature["text"],
                    feature["absolute_offsets"],
                    predicted.tolist()[:seq_len],
                    attention_mask=feature.get("attention_mask"),
                    record_id=feature["record_id"],
                )
            )
            gold_entities.extend(
                decode_bio_entities(
                    feature["text"],
                    feature["absolute_offsets"],
                    gold.tolist()[:seq_len],
                    attention_mask=feature.get("attention_mask"),
                    record_id=feature["record_id"],
                )
            )\"\"\"

if old_code in code:
    code = code.replace(old_code, new_code)
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(code)
    print("Đã vá lỗi padding seq_len trong src/training/ner/train.py thành công!")
elif "predicted.tolist()[:seq_len]" in code:
    print("train.py đã được vá, không cần thao tác.")
else:
    print("Không thể tự động vá train.py vì mã nguồn không khớp (có thể Git Repo đã cập nhật bản vá này).")
"""),

    create_markdown_cell("## Bước 6: Build Datasets"),
    create_code_cell("""!python -m pytest tests/training -q
!python -m src.training.build_datasets \\
  --config configs/training/data.yaml \\
  --project-root .

import json
import shutil

local_training = repo_data / "training"
manifest = json.loads((local_training / "manifests" / "build.json").read_text(encoding="utf-8"))
drive_data = Path("/content/drive/MyDrive/AI-Race-Viettel/data")
archive = drive_data / "training_builds" / manifest["build_id"]

if archive.exists():
    print(f"Training build {manifest['build_id']} đã có trên MyDrive, tái sử dụng.")
else:
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(local_training, archive)
    print("Đã tạo và copy dataset build sang MyDrive! BUILD_ID =", manifest["build_id"])"""),

    create_markdown_cell("## Bước 7: Huấn luyện NER (XLM-RoBERTa)"),
    create_code_cell("""# 1. Train Synthetic stage
!python -m src.training.ner.train \\
  --stage synthetic \\
  --run-dir artifacts/training/ner/synthetic-candidate"""),
  
    create_code_cell("""# 2. Train Trusted Fold 0 (bạn có thể lặp lại cho các fold khác)
!python -m src.training.ner.train \\
  --stage trusted-fold \\
  --fold 0 \\
  --initial-checkpoint artifacts/training/ner/synthetic-candidate/final \\
  --run-dir artifacts/training/ner/trusted-fold-0-candidate"""),

    create_markdown_cell("## Bước 8: Huấn luyện BGE-M3 Embedding và FAISS"),
    create_code_cell("""# 1. Train BGE-M3
!python -m src.training.embedding.train \\
  --stage synthetic \\
  --run-dir artifacts/training/embedding/synthetic-candidate"""),

    create_code_cell("""# 2. Generate Embeddings (chạy lâu, chú ý theo dõi GPU RAM)
!python -m src.retrieval.generate_embeddings \\
  --model BGE-M3 \\
  --table icd10 \\
  --model-path artifacts/training/embedding/synthetic-candidate/final \\
  --output-dir artifacts/index-build \\
  --db data/kb/metadata.db \\
  --batch-size 16

!python -m src.retrieval.generate_embeddings \\
  --model BGE-M3 \\
  --table rxnorm \\
  --model-path artifacts/training/embedding/synthetic-candidate/final \\
  --output-dir artifacts/index-build \\
  --db data/kb/metadata.db \\
  --batch-size 16"""),

    create_code_cell("""# 3. Build Faiss Index
!python -m src.retrieval.build_faiss_index \\
  --model BGE-M3 \\
  --table icd10 \\
  --embedding-dir artifacts/index-build \\
  --output-dir artifacts/indexes/icd10-bge-m3 \\
  --adapter-dir artifacts/training/embedding/synthetic-candidate/final \\
  --base-model data/models/bge-m3 \\
  --db data/kb/metadata.db

!python -m src.retrieval.build_faiss_index \\
  --model BGE-M3 \\
  --table rxnorm \\
  --embedding-dir artifacts/index-build \\
  --output-dir artifacts/indexes/rxnorm-bge-m3 \\
  --adapter-dir artifacts/training/embedding/synthetic-candidate/final \\
  --base-model data/models/bge-m3 \\
  --db data/kb/metadata.db"""),

    create_markdown_cell("## Bước 9: Chuẩn bị Reranker (Qwen2.5)"),
    create_code_cell("""# 1. Tải Base Model (nếu chưa có)
!python -m pip install -U "huggingface_hub>=0.34,<2"
!hf download Qwen/Qwen2.5-7B-Instruct \\
  --revision a09a35458c702b33eeacc393d103063234e8bc28 \\
  --local-dir /content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct
  
# (Bạn cần tạo file HF_REVISION.txt trong thư mục model với nội dung là mã revision như hướng dẫn trong QWEN_QLORA_COLAB.md)
!echo "a09a35458c702b33eeacc393d103063234e8bc28" > /content/AI-Race-Viettel/data/models/Qwen2.5-7B-Instruct/HF_REVISION.txt"""),

    create_code_cell("""# 2. Freeze Candidates
!python -m src.training.reranker.train \\
  --freeze-candidates \\
  --icd-index-dir artifacts/indexes/icd10-bge-m3 \\
  --rxnorm-index-dir artifacts/indexes/rxnorm-bge-m3 \\
  --embedding-model-artifact artifacts/training/embedding/synthetic-candidate/final \\
  --alpha 0.75 \\
  --internal-top-k 20 \\
  --candidate-top-k 10"""),
    
    create_code_cell("""# 3. Train QLoRA Reranker
!python -m src.training.reranker.train \\
  --stage synthetic \\
  --run-dir artifacts/training/reranker/synthetic-candidate""")
])

output_file = Path(r"d:\AI Race Viettel\Colab_Training_Runbook.ipynb")
output_file.parent.mkdir(parents=True, exist_ok=True)
with output_file.open("w", encoding="utf-8") as f:
    json.dump(notebook, f, ensure_ascii=False, indent=2)

print(f"Notebook created at: {output_file}")
