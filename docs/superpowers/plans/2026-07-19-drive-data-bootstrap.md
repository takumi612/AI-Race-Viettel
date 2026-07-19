# Drive Data Bootstrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Làm cho hướng dẫn Colab tải dataset canonical từ Google Drive trước khi build và train, với verification đủ chặt để phát hiện download thiếu hoặc sai cấu trúc.

**Architecture:** README cung cấp quick start duy nhất; hai runbook giải thích chi tiết cùng một luồng. `gdown>=6.0.0` tải Drive folder trực tiếp vào `data/`; một cell Python fail-fast kiểm tra cấu trúc, số mẫu, QA, manifest SHA-256 và SQLite trước khi chạy code training.

**Tech Stack:** Markdown, Bash/Colab magics, Python 3.10+, gdown 6.x, pathlib, hashlib, json, sqlite3, pytest.

## Global Constraints

- Drive URL: `https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link`.
- Download target: `/content/AI-Race-Viettel/data`; không tạo `data/data`.
- Expected synthetic count: 2.000 `.txt` và 2.000 `.json`.
- Expected manifest SHA-256: `66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a`.
- `data/input` không được dùng cho training.
- Không stage hoặc commit nội dung trong `data/`.

---

### Task 1: Cập nhật README quick start

**Files:**
- Modify: `README.md`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: Drive URL, canonical fingerprint và GitHub remote đã khóa.
- Produces: luồng copy/paste từ clone đến build dataset.

- [x] **Step 1:** Thay phần Colab đầu README bằng thứ tự clone → `gdown>=6` → download → verify → install → test/build.
- [x] **Step 2:** Thay link Drive cũ và hướng dẫn chỉ tải `data/kb` bằng nguồn toàn bộ `data` canonical.
- [x] **Step 3:** Cập nhật cây thư mục theo cấu trúc thật: `synthetic_train_v1/{input,gt,qa}`, `dev`, `kb`, `models`, `input`, `output`.
- [x] **Step 4:** Chạy `rg` để xác nhận README không còn Drive ID cũ hoặc placeholder clone URL.
- [x] **Step 5:** Ignore `data/synthetic_train_v1/` để `git add .` không đưa dataset canonical lên GitHub.

### Task 2: Đồng bộ hai runbook Colab

**Files:**
- Modify: `docs/training/DATA_FOUNDATION.md`
- Modify: `docs/training/MODEL_TRAINING_COLAB.md`

**Interfaces:**
- Consumes: quick start Task 1.
- Produces: setup chi tiết, cell verification và quy tắc lưu checkpoint lên MyDrive.

- [x] **Step 1:** Thay setup symlink nguồn data bằng download trực tiếp từ Drive public.
- [x] **Step 2:** Giữ mount MyDrive chỉ cho `artifacts/` và `training_builds/`, không dùng làm điều kiện để đọc source dataset.
- [x] **Step 3:** Thêm cell Python chạy fail-fast với `Path`, `hashlib`, `json`, `sqlite3`; kiểm tra folder, count, QA, fingerprint, DB size và bảng DB.
- [x] **Step 4:** Đảm bảo build/trainer chỉ xuất hiện sau verification.

### Task 3: Verification, commit và push

**Files:**
- Verify: `README.md`
- Verify: `docs/training/DATA_FOUNDATION.md`
- Verify: `docs/training/MODEL_TRAINING_COLAB.md`

**Interfaces:**
- Consumes: ba tài liệu đã đồng bộ.
- Produces: commit đã kiểm tra trên nhánh `develop` và push lên `origin/develop`.

- [x] **Step 1:** Chạy cell verification tương đương trên dataset local hiện có; expected: 2.000/2.000, QA true, đúng SHA-256, DB có bảng.
- [x] **Step 2:** Kiểm tra CLI `gdown>=6` và khả năng liệt kê Drive folder bằng `--folder --json` nếu folder đã public; không tải toàn bộ 11+ GB chỉ để test docs.
- [x] **Step 3:** Chạy `python -m pytest tests/training -q`, `git diff --check`, scan Drive IDs/path và kiểm tra mọi code fence.
- [x] **Step 4:** Stage đúng các file tài liệu/plan; xác nhận `data/synthetic_train_v1` không staged.
- [ ] **Step 5:** Commit và `git push origin develop`; chỉ báo thành công khi remote push trả exit code 0.

### Task 4: Hướng dẫn Qwen2.5-7B QLoRA

**Files:**
- Create: `docs/training/QWEN_QLORA_COLAB.md`
- Modify: `README.md`
- Modify: `docs/training/DATA_FOUNDATION.md`
- Modify: `docs/training/MODEL_TRAINING_COLAB.md`

**Interfaces:**
- Consumes: Hugging Face model `Qwen/Qwen2.5-7B-Instruct` revision `a09a35458c702b33eeacc393d103063234e8bc28`.
- Produces: local base model tại `data/models/Qwen2.5-7B-Instruct` và PEFT adapter tại `artifacts/training/reranker/<run>/final`.

- [x] **Step 1:** Ghi lệnh `hf download --dry-run` và download thật với revision/local-dir cố định.
- [x] **Step 2:** Thêm cell kiểm tra config, tokenizer, safetensors index, đủ 4 shard và provenance marker `HF_REVISION.txt`.
- [x] **Step 3:** Ghi luồng copy base weights sang MyDrive và restore về local mà không ghi đè dữ liệu có sẵn.
- [x] **Step 4:** Ghi prerequisites, freeze candidates, dry-run, synthetic, resume, trusted-fold và trusted-final commands.
- [x] **Step 5:** Liên kết hướng dẫn từ README/runbook và kiểm tra toàn bộ Python cells, local links, CLI help/dry-run.
