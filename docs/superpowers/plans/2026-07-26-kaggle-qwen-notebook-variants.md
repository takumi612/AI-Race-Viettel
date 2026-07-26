# Kaggle Qwen Notebook Variants Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generate clean Kaggle training notebooks for an explicitly enabled Qwen runtime and a stable no-Qwen runtime.

**Architecture:** Extend the canonical notebook builder with a boolean variant input. The enabled variant injects an explicit CUDA 12.9 vLLM bootstrap and an early import check; the unable-Qwen variant omits all vLLM installation/import code. Both variants share the same training, inference, quality, and packaging cells.

**Tech Stack:** Python 3.12, Jupyter notebook JSON, argparse, pytest, vLLM 0.25.1 CUDA 12.9 wheel, PyTorch CUDA 12.9 index.

## Global Constraints

- Output `ai-race-training-v2-enableQwen.ipynb` and `ai-race-training-v2-unableQwen.ipynb` at repository root.
- Both notebooks default to `RUN_MODE="full"` and contain no saved outputs or execution counts.
- Only the enabled variant may install or import vLLM.
- The enabled variant must reject an unusable vLLM runtime before phase execution.
- Preserve recursive `synthetic_train_v2` and inference-input discovery.
- Preserve all 13 canonical phases and the existing error notebook.

---

### Task 1: Add failing variant contracts

**Files:**
- Create: `v2/tests/test_kaggle_notebook_variants.py`
- Test: `v2/tests/test_kaggle_notebook_variants.py`

**Interfaces:**
- Consumes: dynamically loaded `v2/tools/build_kaggle_notebook.py`
- Produces: required API `build_notebook(enable_qwen_reranker: bool = False) -> dict[str, Any]`

- [ ] **Step 1: Write failing tests**

```python
def test_enabled_variant_uses_cuda_129_and_import_checks_vllm():
    source = code_source(build_notebook(enable_qwen_reranker=True))
    assert "ENABLE_QWEN_RERANKER = True" in source
    assert "vllm-0.25.1+cu129" in source
    assert "https://download.pytorch.org/whl/cu129" in source
    assert "from vllm import LLM" in source
    assert "--no-deps" not in source


def test_unable_variant_omits_vllm_runtime():
    source = code_source(build_notebook(enable_qwen_reranker=False))
    assert "ENABLE_QWEN_RERANKER = False" in source
    assert "vllm" not in source.lower()
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest v2/tests/test_kaggle_notebook_variants.py -q`

Expected: FAIL because `build_notebook` does not accept `enable_qwen_reranker`.

- [ ] **Step 3: Commit the failing contracts**

```powershell
git add -- v2/tests/test_kaggle_notebook_variants.py
git commit -m "test: define Kaggle Qwen notebook variants"
```

### Task 2: Implement variant-aware notebook generation

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py`
- Modify: `v2/tests/test_kaggle_observability.py`
- Test: `v2/tests/test_kaggle_notebook_variants.py`
- Test: `v2/tests/test_kaggle_observability.py`

**Interfaces:**
- Produces: `build_qwen_bootstrap(enabled: bool) -> str`
- Produces: `build_notebook(enable_qwen_reranker: bool = False) -> dict[str, Any]`
- Produces CLI: `--qwen-mode {enabled,unable}`

- [ ] **Step 1: Add the minimal builder implementation**

Use a setup-source marker replacement so braces in the existing generated Python remain unchanged:

```python
def build_qwen_bootstrap(enabled: bool) -> str:
    if not enabled:
        return ""
    return '''VLLM_CUDA129_WHEEL = (
    "https://github.com/vllm-project/vllm/releases/download/v0.25.1/"
    "vllm-0.25.1+cu129-cp38-abi3-manylinux_2_28_x86_64.whl"
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade",
     VLLM_CUDA129_WHEEL, "--extra-index-url", "https://download.pytorch.org/whl/cu129"],
    check=True,
)
importlib.invalidate_caches()
try:
    from vllm import LLM as _VLLM_IMPORT_CHECK
except Exception as exc:
    raise RuntimeError(
        "Qwen is enabled but the CUDA 12.9 vLLM runtime failed to import. "
        "Restart with the unableQwen notebook or inspect the CUDA driver."
    ) from exc
'''
```

Replace `__ENABLE_QWEN_RERANKER__` with `repr(enable_qwen_reranker)` and `__QWEN_BOOTSTRAP__` with the returned bootstrap. Add `--qwen-mode`, map `enabled` to `True`, and pass it to `build_notebook`.

- [ ] **Step 2: Update observability tests to assert the new explicit stack instead of the removed `--no-deps` branch**

```python
assert "vllm-0.25.1+cu129" in enabled_source
assert "--no-deps" not in enabled_source
assert "vllm" not in unable_source.lower()
```

- [ ] **Step 3: Run focused tests to verify GREEN**

Run: `python -m pytest v2/tests/test_kaggle_notebook_variants.py v2/tests/test_kaggle_observability.py v2/tests/test_single_toggle_qwen.py -q`

Expected: all tests PASS.

- [ ] **Step 4: Commit the builder change**

```powershell
git add -- v2/tools/build_kaggle_notebook.py v2/tests/test_kaggle_observability.py
git commit -m "feat: generate Kaggle Qwen notebook variants"
```

### Task 3: Generate and verify import-ready notebooks

**Files:**
- Modify: `v2/medical_information_extraction_kaggle.ipynb`
- Create: `ai-race-training-v2-enableQwen.ipynb`
- Create: `ai-race-training-v2-unableQwen.ipynb`

**Interfaces:**
- Consumes: builder CLI `--qwen-mode {enabled,unable}`
- Produces: two import-ready user notebooks

- [ ] **Step 1: Generate the unable-Qwen canonical and user notebook**

```powershell
python v2/tools/build_kaggle_notebook.py --qwen-mode unable --output v2/medical_information_extraction_kaggle.ipynb
python v2/tools/build_kaggle_notebook.py --qwen-mode unable --output ai-race-training-v2-unableQwen.ipynb
```

- [ ] **Step 2: Generate the enabled-Qwen user notebook**

```powershell
python v2/tools/build_kaggle_notebook.py --qwen-mode enabled --output ai-race-training-v2-enableQwen.ipynb
```

- [ ] **Step 3: Validate notebook hygiene and variant contracts**

Run: `python -m pytest v2/tests/test_kaggle_notebook_variants.py v2/tests/test_notebook_orchestrator_contract.py v2/tests/test_kaggle_observability.py v2/tests/test_single_toggle_qwen.py v2/tests/test_required_qwen_phase.py -q`

Expected: all tests PASS, all code cells parse, all outputs are empty, and all execution counts are null.

- [ ] **Step 4: Run the builder validation on both outputs**

```powershell
python v2/tools/build_kaggle_notebook.py --source ai-race-training-v2-enableQwen.ipynb --output C:\tmp\enableQwen-verified.ipynb
python v2/tools/build_kaggle_notebook.py --source ai-race-training-v2-unableQwen.ipynb --output C:\tmp\unableQwen-verified.ipynb
```

Expected: both reports contain `"valid": true`, `"phase_count": 13`, and `"syntax_errors": []`.

- [ ] **Step 5: Commit generated notebooks**

```powershell
git add -- v2/medical_information_extraction_kaggle.ipynb ai-race-training-v2-enableQwen.ipynb ai-race-training-v2-unableQwen.ipynb
git commit -m "feat: publish Kaggle Qwen notebook variants"
```
