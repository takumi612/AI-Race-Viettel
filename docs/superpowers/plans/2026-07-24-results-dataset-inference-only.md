# Kế hoạch triển khai notebook Kaggle chỉ suy luận

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Tạo notebook Kaggle inference-only và runbook tiếng Việt ở thư mục gốc, dùng dataset kết quả đã train cùng một dataset input mới để sinh `output.zip` mà không train lại.

**Architecture:** Tái sử dụng generator inference đã có tại `v2/tools/build_kaggle_inference_notebook.py`, bổ sung hợp đồng kiểm thử cho artifact mới ở thư mục gốc và Việt hóa toàn bộ phần hướng dẫn trong notebook. Logic Python hiện có tiếp tục hỗ trợ cả thư mục Kaggle đã tự giải nén lẫn file ZIP, trong khi runbook độc lập mô tả quy trình attach hai dataset và Run All.

**Tech Stack:** Python 3.12, Jupyter notebook JSON, `zipfile`, `pathlib`, pytest, Hugging Face Transformers và pipeline `clinical_nlp_lab` hiện có.

## Global Constraints

- Tạo `train-ai-race-v2-32-8-inference-only.ipynb`.
- Tạo `KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md`.
- Giữ nguyên `train-ai-race-v2-32-8.ipynb`.
- Nội dung hướng dẫn trong notebook và runbook dùng tiếng Việt.
- Không gọi `Trainer`, `.train()`, `train_ner_subprocess.py`, chia train/validation hoặc đóng gói checkpoint mới.
- Chấp nhận cả cây kết quả/input đã giải nén và `results.zip`/`input.zip`.
- Luôn tạo mới `/kaggle/working/output.zip` và ghi `training_skipped: true`.

---

### Task 1: Mở rộng hợp đồng kiểm thử cho artifact ở thư mục gốc

**Files:**
- Modify: `v2/tests/test_inference_notebook.py`
- Test: `v2/tests/test_inference_notebook.py`

**Interfaces:**
- Consumes: `build_notebook() -> dict[str, Any]` từ generator hiện có.
- Produces: kiểm thử bắt buộc notebook gốc tồn tại, đồng bộ với generator, không train và có hướng dẫn tiếng Việt.

- [ ] **Step 1: Viết test thất bại cho notebook đích**

Thêm hằng số và các test:

```python
WORKSPACE_ROOT = ROOT.parent
ROOT_NOTEBOOK = WORKSPACE_ROOT / "train-ai-race-v2-32-8-inference-only.ipynb"
TRAINING_NOTEBOOK = WORKSPACE_ROOT / "train-ai-race-v2-32-8.ipynb"


def test_root_inference_notebook_is_generated_from_the_canonical_builder():
    notebook = json.loads(ROOT_NOTEBOOK.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("inference_builder_root", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert notebook == module.build_notebook()


def test_root_inference_notebook_is_vietnamese_and_never_trains():
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in json.loads(ROOT_NOTEBOOK.read_text(encoding="utf-8"))["cells"]
    )
    for phrase in ("suy luận", "dataset kết quả", "Run All", "output.zip"):
        assert phrase in source
    for forbidden in ("train_ner_subprocess.py", "Trainer(", ".train()"):
        assert forbidden not in source


def test_training_notebook_remains_present():
    assert TRAINING_NOTEBOOK.is_file()
```

- [ ] **Step 2: Chạy test và xác nhận RED**

Run:

```powershell
python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
```

Expected: FAIL vì `train-ai-race-v2-32-8-inference-only.ipynb` chưa tồn tại.

- [ ] **Step 3: Commit test RED**

```powershell
git add -- v2/tests/test_inference_notebook.py
git commit -m "test: require Vietnamese inference notebook artifact"
```

### Task 2: Việt hóa generator và sinh notebook inference-only

**Files:**
- Modify: `v2/tools/build_kaggle_inference_notebook.py`
- Modify: `v2/medical_information_extraction_inference_kaggle.ipynb`
- Create: `train-ai-race-v2-32-8-inference-only.ipynb`
- Test: `v2/tests/test_inference_notebook.py`

**Interfaces:**
- Consumes: logic discovery, selective extraction, runtime validation và `run_inference` đang có trong generator.
- Produces: `build_notebook() -> dict[str, Any]` với markdown tiếng Việt và hai notebook JSON đồng bộ.

- [ ] **Step 1: Việt hóa các markdown cell và thông báo hướng dẫn**

Giữ nguyên code inference, nhưng thay phần hướng dẫn người dùng bằng các nhãn:

```python
markdown_cell(
    """# Suy luận Clinical NLP trên Kaggle

Notebook này nạp checkpoint đã train từ dataset kết quả và tạo dự đoán mới.
Notebook không huấn luyện hoặc fine-tune mô hình.

Trước khi chọn **Run All**, hãy attach dataset kết quả, dataset input, bật GPU
và bật Internet."""
)
```

Các tiêu đề bước dùng:

```text
1. Cấu hình runtime và tìm dataset kết quả
2. Khôi phục checkpoint và artifact suy luận
3. Tìm dữ liệu input mới
4. Cài dependency suy luận và kiểm tra GPU
5. Chạy suy luận từ checkpoint đã đóng gói
6. Kiểm tra bài nộp và ghi manifest
7. Tải kết quả
```

- [ ] **Step 2: Sinh cả notebook canonical và notebook bàn giao**

Run:

```powershell
python v2/tools/build_kaggle_inference_notebook.py --output v2/medical_information_extraction_inference_kaggle.ipynb
python v2/tools/build_kaggle_inference_notebook.py --output train-ai-race-v2-32-8-inference-only.ipynb
```

Expected: hai file là notebook JSON sạch, không có output thực thi.

- [ ] **Step 3: Chạy test và xác nhận GREEN**

Run:

```powershell
python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 4: Commit generator và notebook**

```powershell
git add -- v2/tools/build_kaggle_inference_notebook.py v2/medical_information_extraction_inference_kaggle.ipynb train-ai-race-v2-32-8-inference-only.ipynb
git commit -m "feat: add Vietnamese Kaggle inference notebook"
```

### Task 3: Viết Kaggle runbook tiếng Việt và xác minh cuối

**Files:**
- Create: `KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md`
- Modify: `v2/tests/test_inference_notebook.py`
- Test: `v2/tests/test_inference_notebook.py`

**Interfaces:**
- Consumes: notebook `train-ai-race-v2-32-8-inference-only.ipynb` và hợp đồng hai dataset.
- Produces: hướng dẫn thao tác Kaggle hoàn chỉnh cùng regression test cho các bước bắt buộc.

- [ ] **Step 1: Viết test thất bại cho runbook**

```python
ROOT_RUNBOOK = WORKSPACE_ROOT / "KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md"


def test_vietnamese_runbook_covers_the_complete_kaggle_workflow():
    text = ROOT_RUNBOOK.read_text(encoding="utf-8")
    for phrase in (
        "results.zip",
        "input.zip",
        "Add Input",
        "GPU",
        "Internet",
        "Run All",
        "training_skipped",
        "output.zip",
        "RESULTS_ZIP_OVERRIDE",
        "INPUT_SOURCE_OVERRIDE",
    ):
        assert phrase in text
```

- [ ] **Step 2: Chạy test và xác nhận RED**

Run:

```powershell
python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
```

Expected: FAIL vì `KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md` chưa tồn tại.

- [ ] **Step 3: Viết runbook**

Runbook phải có các phần cụ thể:

```text
1. Chuẩn bị dataset kết quả
2. Chuẩn bị dataset input mới
3. Import notebook và Add Input
4. Bật GPU/Internet và Run All
5. Kiểm tra run_manifest.json
6. Save Version và tải output.zip
7. Override đường dẫn khi có nhiều dataset
8. Xử lý lỗi thường gặp
```

Đường dẫn override phải khớp chính xác tên biến trong notebook sau khi đọc
source generator; test dùng đúng tên đó.

- [ ] **Step 4: Chạy kiểm thử tập trung và toàn bộ**

Run:

```powershell
python -m pytest v2/tests/test_inference_notebook.py -q -p no:cacheprovider
python -m pytest v2/tests -q -p no:cacheprovider
python v2/tools/build_kaggle_inference_notebook.py --output scratch/generated_inference_check.ipynb
```

Expected: toàn bộ test PASS; notebook kiểm tra sinh thành công.

- [ ] **Step 5: Kiểm tra artifact và notebook train gốc**

Run:

```powershell
git diff --check
git status --short -- train-ai-race-v2-32-8.ipynb train-ai-race-v2-32-8-inference-only.ipynb KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md
```

Expected: notebook train gốc không có thay đổi mới; chỉ notebook inference và
runbook xuất hiện trong phạm vi bàn giao.

- [ ] **Step 6: Commit runbook và test**

```powershell
git add -- KAGGLE_INFERENCE_ONLY_RUNBOOK_VI.md v2/tests/test_inference_notebook.py
git commit -m "docs: add Vietnamese Kaggle inference runbook"
```
