# Single-Toggle Required Qwen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the canonical end-to-end Kaggle notebook so `ENABLE_QWEN_RERANKER` is the only enable switch, and an enabled-but-unavailable Qwen run fails before packaging instead of silently falling back.

**Architecture:** Carry the notebook boolean through `RunConfig` into phase 12. Phase 12 owns Qwen construction and cleanup, injects a strict adapter into `FinalModelBundle`, and raises required-Qwen errors. Preserve ranked candidate pools as runtime-only metadata through merge so the adapter can rerank real candidates without changing the official submission schema.

**Tech Stack:** Python 3.11+, dataclasses, pytest, Hugging Face/Transformers, optional vLLM, Jupyter notebook JSON, existing `ClinicalLLMReranker` and `ClinicalLLMAssertionPredictor`.

## Global Constraints

- Work only on `codex/kaggle-end-to-end-pipeline` at the latest checked-in commit.
- Do not reset, checkout, or overwrite unrelated dirty files.
- `ENABLE_QWEN_RERANKER` in the notebook is the only user-facing Qwen enable boolean.
- Disabled Qwen must not import or initialize vLLM.
- Enabled Qwen must fail the run on initialization, generation, parse, validation, or cleanup errors.
- Training remains independent of Qwen.
- Official output keys and raw-text offsets must remain unchanged.
- Every production change follows RED → GREEN → REFACTOR with a watched failing test first.
- Tests must not download Qwen weights or require a GPU.

---

### Task 1: Add the single-toggle transport contract

**Files:**
- Modify: `v2/clinical_nlp_lab/orchestration.py:32-51`
- Modify: `v2/tools/build_kaggle_notebook.py:262-277`
- Modify: `v2/tests/test_orchestration_contract.py`
- Modify: `v2/tests/test_kaggle_observability.py`
- Test: `v2/tests/test_single_toggle_qwen.py`

**Interfaces:**
- `RunConfig.enable_qwen_reranker: bool = False`
- The generated notebook constructs `RunConfig(enable_qwen_reranker=ENABLE_QWEN_RERANKER)`.
- Phase runners consume `config.enable_qwen_reranker`; no phase reads an enable value from `config.json`.

- [ ] **Step 1: Write the failing transport tests**

Add tests that:

```python
def test_run_config_defaults_qwen_disabled():
    assert RunConfig().enable_qwen_reranker is False


def test_builder_passes_notebook_toggle_into_run_config_source():
    source = Path("v2/tools/build_kaggle_notebook.py").read_text(encoding="utf-8")
    assert "enable_qwen_reranker=ENABLE_QWEN_RERANKER" in source
    assert "KAGGLE_OPTIONS" not in source
```

Also assert that a `config.json` containing `enable_qwen=True` cannot change a
`RunConfig(enable_qwen_reranker=False)` value.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
python -m pytest v2/tests/test_single_toggle_qwen.py v2/tests/test_orchestration_contract.py v2/tests/test_kaggle_observability.py -q -p no:cacheprovider
```

Expected: fail because `RunConfig` has no field, the builder does not pass the
field, and `KAGGLE_OPTIONS` still exists.

- [ ] **Step 3: Implement the minimal transport change**

Add the boolean field to `RunConfig`, pass it from the notebook builder, remove
the unused `KAGGLE_OPTIONS` compatibility block, and leave all model/runtime
parameter constants unchanged.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command. Expected: all focused transport tests pass.

- [ ] **Step 5: Regenerate the canonical notebook**

Run the checked-in generator with an explicit output path, then assert the
generated notebook contains one assignment and one `RunConfig` argument:

```powershell
python v2/tools/build_kaggle_notebook.py --output v2/medical_information_extraction_kaggle.ipynb
python -m pytest v2/tests/test_kaggle_observability.py -q -p no:cacheprovider
```

- [ ] **Step 6: Commit**

```powershell
git add -- v2/clinical_nlp_lab/orchestration.py v2/tools/build_kaggle_notebook.py v2/tests/test_orchestration_contract.py v2/tests/test_kaggle_observability.py v2/tests/test_single_toggle_qwen.py v2/medical_information_extraction_kaggle.ipynb
git commit -m "feat: make notebook Qwen toggle authoritative"
```

---

### Task 2: Preserve ranked candidate pools through inference

**Files:**
- Modify: `v2/clinical_nlp_lab/schema.py:27-36`
- Modify: `v2/clinical_nlp_lab/inference.py:11-81,105-134`
- Modify: `v2/tests/test_inference_data_flow.py`
- Test: `v2/tests/test_single_toggle_qwen.py`

**Interfaces:**
- Add a runtime-only `ranked_candidates` field to `EntityAnnotation`.
- `EntityAnnotation.to_submission()` must continue serializing only the official
  `candidates` IDs.
- `merge_raw_span_proposals()` copies each selected proposal's ranked pool into
  the runtime entity.

- [ ] **Step 1: Write the failing candidate-pool tests**

Add a test that creates a `SpanProposal` with two ranked candidate dictionaries,
merges it, and asserts:

```python
assert entity.ranked_candidates[0]["candidate_id"] == "C1"
assert "ranked_candidates" not in entity.to_submission("DISEASE", [])
```

Add an inference test proving the pool survives NER plus KB merge and remains
bounded by the configured top-k.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```powershell
python -m pytest v2/tests/test_single_toggle_qwen.py v2/tests/test_inference_data_flow.py -q -p no:cacheprovider
```

Expected: fail because `EntityAnnotation` has no runtime pool and merge drops
`SpanProposal.ranked_candidates`.

- [ ] **Step 3: Implement the minimal runtime-only field**

Add the field with a safe empty default, copy ranked dictionaries during merge,
and keep `to_submission()` unchanged. Do not add the field to official JSON
or ZIP output.

- [ ] **Step 4: Run the tests and verify GREEN**

Run the same focused pytest command. Expected: all candidate-pool tests pass.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/schema.py v2/clinical_nlp_lab/inference.py v2/tests/test_inference_data_flow.py v2/tests/test_single_toggle_qwen.py
git commit -m "feat: preserve runtime candidate pools for reranking"
```

---

### Task 3: Add a strict Qwen refiner adapter

**Files:**
- Create: `v2/clinical_nlp_lab/qwen_refiner.py`
- Modify: `v2/clinical_nlp_lab/reranker.py:45-132`
- Modify: `v2/clinical_nlp_lab/assertions.py:74-135`
- Test: `v2/tests/test_qwen_refiner_required.py`
- Modify: `v2/tests/test_reranker_compatibility.py`

**Interfaces:**
- Create `RequiredQwenRefiner` with:

```python
class RequiredQwenRefiner:
    def __init__(self, llm_engine: Any, *, batch_size: int = 64): ...
    def refine(
        self,
        entities: tuple[EntityAnnotation, ...],
        raw_text: str,
    ) -> tuple[EntityAnnotation, ...]: ...
    def destroy(self) -> None: ...
```

- Create a strict Qwen error type, for example
  `RequiredQwenError(RuntimeError)`.
- Add strict behavior to reranker/assertion parsing so invalid JSON, unknown
  IDs, invalid assertion enums, and response-count mismatches raise instead of
  silently defaulting.
- Valid `selected_id: null` remains an explicit abstention.

- [ ] **Step 1: Write failing adapter tests**

Cover:

```python
def test_refiner_selects_only_id_from_preserved_pool():
    ...


def test_refiner_accepts_null_as_abstention():
    ...


def test_refiner_rejects_unknown_candidate_id():
    ...


def test_refiner_rejects_malformed_qwen_response():
    ...


def test_refiner_preserves_text_type_and_offsets():
    ...


def test_refiner_destroy_is_called_and_errors_are_visible():
    ...
```

Use a fake engine whose `generate()` returns lightweight response objects; do
not import vLLM in unit tests.

- [ ] **Step 2: Run tests and verify RED**

Run:

```powershell
python -m pytest v2/tests/test_qwen_refiner_required.py v2/tests/test_reranker_compatibility.py -q -p no:cacheprovider
```

Expected: fail because the adapter and strict parsing mode do not exist.

- [ ] **Step 3: Implement the minimal adapter**

Build rerank queries from each entity's raw context and runtime candidate pool.
Use the existing vLLM compatibility helpers and existing assertion predictor.
Validate every returned ID and enum before mutating an entity. Keep all entity
text, type, and positions immutable.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same command. Expected: all strict adapter tests pass without model
downloads or GPU access.

- [ ] **Step 5: Run the existing vLLM compatibility suite**

```powershell
python -m pytest v2/tests/test_reranker_compatibility.py v2/tests/test_assertion_vllm_compat.py -q -p no:cacheprovider
```

Expected: existing legacy behavior remains compatible when strict mode is not
requested.

- [ ] **Step 6: Commit**

```powershell
git add -- v2/clinical_nlp_lab/qwen_refiner.py v2/clinical_nlp_lab/reranker.py v2/clinical_nlp_lab/assertions.py v2/tests/test_qwen_refiner_required.py v2/tests/test_reranker_compatibility.py
git commit -m "feat: add strict required-Qwen refiner"
```

---

### Task 4: Wire phase 12, hard-fail semantics, and observability

**Files:**
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py:591-633`
- Modify: `v2/clinical_nlp_lab/runtime_bundle.py:119-129`
- Modify: `v2/clinical_nlp_lab/inference.py:206-238`
- Modify: `v2/tests/test_kaggle_phase_runners.py`
- Test: `v2/tests/test_required_qwen_phase.py`

**Interfaces:**
- Phase 12 reads only `config.enable_qwen_reranker`.
- `load_final_model_bundle(..., qwen_reranker=refiner)` receives the adapter
  only when enabled.
- `InferenceConfig.enable_qwen` equals the same transport boolean.
- Required-Qwen exceptions propagate out of phase 12.
- Run summary contains `qwen_requested`, `qwen_initialized`, model name,
  query counts, abstentions, and `qwen_status`.

- [ ] **Step 1: Write failing phase tests**

Add tests that:

1. pass `RunConfig(enable_qwen_reranker=False)` with a config file containing
   `enable_qwen=True` and assert Qwen is not constructed;
2. pass `RunConfig(enable_qwen_reranker=True)` and assert the refiner is
   constructed and injected;
3. make construction raise `ImportError` and assert phase 12 raises;
4. make inference raise `RequiredQwenError` and assert phase 12 raises;
5. assert phase 13 is not invoked after required-Qwen failure;
6. assert cleanup runs when inference fails.

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest v2/tests/test_required_qwen_phase.py v2/tests/test_kaggle_phase_runners.py -q -p no:cacheprovider
```

Expected: fail because phase 12 currently reads `config.json`, never
constructs Qwen, swallows inference exceptions, and does not expose Qwen
status.

- [ ] **Step 3: Implement phase-12 lifecycle**

When disabled, use the current deterministic bundle path. When enabled:

1. import and construct `ClinicalLLMReranker`;
2. construct `RequiredQwenRefiner`;
3. pass it to `load_final_model_bundle`;
4. execute inference;
5. record successful status;
6. destroy the engine in `finally`;
7. re-raise any required-Qwen exception.

Remove the `load_config(...).get("enable_qwen", False)` branch. Do not catch
required-Qwen errors in `infer_document`; disabled behavior may retain its
existing no-op path.

- [ ] **Step 4: Run tests and verify GREEN**

Run the same focused phase test command. Expected: all required-Qwen lifecycle
tests pass.

- [ ] **Step 5: Run inference regression tests**

```powershell
python -m pytest v2/tests/test_inference_data_flow.py v2/tests/test_candidate_policy.py v2/tests/test_kaggle_phase_runners.py -q -p no:cacheprovider
```

Expected: deterministic disabled behavior and official output contracts remain
green.

- [ ] **Step 6: Commit**

```powershell
git add -- v2/clinical_nlp_lab/kaggle_phases.py v2/clinical_nlp_lab/runtime_bundle.py v2/clinical_nlp_lab/inference.py v2/tests/test_required_qwen_phase.py v2/tests/test_kaggle_phase_runners.py
git commit -m "feat: require Qwen when notebook toggle is enabled"
```

---

### Task 5: Regenerate notebook and verify the complete contract

**Files:**
- Modify: `v2/medical_information_extraction_kaggle.ipynb`
- Modify: `v2/tests/test_inference_notebook.py`
- Modify: `v2/tests/test_kaggle_observability.py`
- Test: `v2/tests/test_single_toggle_qwen.py`

- [ ] **Step 1: Write failing generated-notebook assertions**

Assert:

```python
assert source.count("ENABLE_QWEN_RERANKER =") == 1
assert "enable_qwen_reranker=ENABLE_QWEN_RERANKER" in source
assert "KAGGLE_OPTIONS" not in source
assert 'loaded_cfg.get("enable_qwen"' not in source
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
python -m pytest v2/tests/test_single_toggle_qwen.py v2/tests/test_inference_notebook.py v2/tests/test_kaggle_observability.py -q -p no:cacheprovider
```

Expected: fail against the current generated notebook because it still
contains `KAGGLE_OPTIONS` and does not pass the boolean into `RunConfig`.

- [ ] **Step 3: Regenerate the canonical notebook**

Run the checked-in builder and verify the notebook JSON parses:

```powershell
python v2/tools/build_kaggle_notebook.py
python -m json.tool v2/medical_information_extraction_kaggle.ipynb > $env:TEMP\medical_information_extraction_kaggle.checked.json
```

- [ ] **Step 4: Run the complete local suite**

```powershell
python -m pytest v2/tests -q -p no:cacheprovider
```

Expected: all tests pass without downloading large models or requiring CUDA.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/medical_information_extraction_kaggle.ipynb v2/tests/test_inference_notebook.py v2/tests/test_kaggle_observability.py v2/tests/test_single_toggle_qwen.py
git commit -m "test: verify single-toggle required-Qwen notebook contract"
```

- [ ] **Step 6: Final verification**

Run:

```powershell
git diff --check HEAD~1..HEAD
git status --short --branch
```

Confirm that only the intended implementation commits touched tracked files,
the pre-existing unrelated dirty files remain untouched, and the final
notebook contains one authoritative enable switch.
