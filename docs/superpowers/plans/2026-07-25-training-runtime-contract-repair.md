# Training–Runtime Contract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the Kaggle training/runtime contract so chunking, curriculum, assertion typing, KB recovery, post-processing, and quality validation produce a retrainable high-precision pipeline.

**Architecture:** A shared native-overflow window contract supplies NER and explicit assertion metadata. Canonical type IDs and stage specifications cross process boundaries as data, while inference applies boundary-aware recovery and measurable quality gates before publication.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, pytest, Kaggle notebooks.

## Global Constraints

- Preserve `data_v2/Training_data/synthetic_train_v2`; do not rewrite source annotations.
- Keep raw offsets half-open and exactly round-trippable to source text.
- Use `DISEASE=0, DRUG=1, SYMPTOM=2, LAB_NAME=3, LAB_RESULT=4` everywhere.
- Follow red-green-refactor for every production behavior change.
- Do not touch unrelated dirty workspace files.

---

### Task 1: Canonical owner-window and assertion contract

**Files:**
- Modify: `v2/clinical_nlp_lab/examples.py`
- Modify: `v2/clinical_nlp_lab/collation.py`
- Create: `v2/clinical_nlp_lab/entity_types.py`
- Test: `v2/tests/test_owner_window_native_overflow.py`
- Test: `v2/tests/test_assertion_type_contract.py`

**Interfaces:**
- Produces: `ENTITY_TYPE_TO_ID`, `ASSERTION_ENTITY_TYPES`, and explicit `OwnedEntity` metadata on `TokenWindow`.
- Consumes: fast tokenizer overflow encodings with `overflow_to_sample_mapping`.

- [ ] **Step 1: Write failing native-overflow tests**

Create a deterministic fake fast tokenizer returning two overflow windows,
each with literal special-token IDs and offsets. Assert both windows preserve
special tokens, offsets become absolute, and one entity has exactly one owner.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest v2/tests/test_owner_window_native_overflow.py -q`

Expected: FAIL because the current builder tokenizes once and manually slices.

- [ ] **Step 3: Implement the shared window contract**

Call the tokenizer with `truncation=True`, `max_length`, `stride`,
`return_offsets_mapping=True`, and `return_overflowing_tokens=True`. Store
explicit `(entity_id, entity_type, token_start, token_end, assertions)` owner
metadata per window.

- [ ] **Step 4: Write and verify failing canonical-type tests**

Assert literal IDs for all five types and assert LAB_NAME/LAB_RESULT masks are
false while DISEASE/DRUG/SYMPTOM masks are true.

Run: `python -m pytest v2/tests/test_assertion_type_contract.py -q`

Expected: FAIL because the collator derives type IDs from BIO positions.

- [ ] **Step 5: Implement canonical collation and verify GREEN**

Use owned entity metadata directly; remove positional assertion lookup and
BIO-derived type IDs.

Run: `python -m pytest v2/tests/test_owner_window_native_overflow.py v2/tests/test_assertion_type_contract.py -q`

Expected: PASS.

### Task 2: Make curriculum specifications executable

**Files:**
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py`
- Modify: `v2/scripts/train_ner_subprocess.py`
- Modify: `v2/clinical_nlp_lab/sampling.py`
- Test: `v2/tests/test_kaggle_curriculum_execution.py`

**Interfaces:**
- Consumes: `StageSpec.to_dict()` serialized in each stage-input manifest.
- Produces: stage selections and training arguments that honor max epochs,
  learning rate, organizer fraction, and replay fraction.

- [ ] **Step 1: Write failing stage behavior tests**

Assert stage 2 and stage 3 selections differ, and literal stage specs produce
the declared epochs and learning rates in resolved subprocess settings.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest v2/tests/test_kaggle_curriculum_execution.py -q`

Expected: FAIL because stages 2/3 currently share IDs and subprocess settings
come from global config.

- [ ] **Step 3: Pass and enforce StageSpec**

Embed the stage spec in the input manifest, validate it against
`plan_curriculum("full")`, resolve training arguments from that spec, and use
deterministic source-bucket sampling for the declared fractions.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest v2/tests/test_kaggle_curriculum_execution.py v2/tests/test_curriculum_state.py -q`

Expected: PASS.

### Task 3: Held-out assertion training and calibration

**Files:**
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py`
- Modify: `v2/clinical_nlp_lab/assertion_model.py`
- Test: `v2/tests/test_assertion_calibration_contract.py`

**Interfaces:**
- Consumes: fixed document train/validation partitions and canonical entity IDs.
- Produces: thresholds fitted from final validation logits only.

- [ ] **Step 1: Write failing calibration tests**

Use extreme logits to assert finite probabilities, and use disjoint literal
train/validation document IDs to assert calibration consumes validation
examples only.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest v2/tests/test_assertion_calibration_contract.py -q`

Expected: FAIL from unstable exponential and lack of held-out calibration API.

- [ ] **Step 3: Implement stable held-out calibration**

Use a stable sigmoid, train the head only on train windows, evaluate once after
the final epoch on validation windows, and reject empty-positive calibration
axes with an explicit conservative threshold policy.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest v2/tests/test_assertion_calibration_contract.py v2/tests/test_assertion_features.py -q`

Expected: PASS without overflow warnings.

### Task 4: Boundary-aware KB and inference post-processing

**Files:**
- Modify: `v2/clinical_nlp_lab/runtime_bundle.py`
- Modify: `v2/clinical_nlp_lab/ner.py`
- Modify: `v2/clinical_nlp_lab/inference.py`
- Test: `v2/tests/test_kb_word_boundaries.py`
- Test: `v2/tests/test_inference_quality_filter.py`

**Interfaces:**
- Produces: standalone-token exact KB matches and refined, confidence-prioritized
  non-overlapping proposals.

- [ ] **Step 1: Write failing KB tests**

For raw text containing `ho`, `hoặc`, and `khoa học`, assert only the standalone
`ho` receives candidate R05.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest v2/tests/test_kb_word_boundaries.py -q`

Expected: FAIL because substring search emits all occurrences.

- [ ] **Step 3: Implement normalized boundary checks**

Require alphanumeric boundaries around normalized aliases and retain exact raw
offset round trips.

- [ ] **Step 4: Write failing inference filter tests**

Assert punctuation-only, mid-word, excessive multiline, and implausibly long
proposals are rejected; assert a short high-confidence bounded span beats a
longer overlapping low-confidence span.

- [ ] **Step 5: Implement refinement and verify GREEN**

Run: `python -m pytest v2/tests/test_kb_word_boundaries.py v2/tests/test_inference_quality_filter.py v2/tests/test_primary_inference_path.py -q`

Expected: PASS.

### Task 5: Dataset and output quality gates

**Files:**
- Modify: `v2/clinical_nlp_lab/dataset_quality.py`
- Create: `v2/clinical_nlp_lab/output_quality.py`
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py`
- Test: `v2/tests/test_training_dataset_contract.py`
- Test: `v2/tests/test_output_quality_gate.py`

**Interfaces:**
- Produces: serializable training/output audit reports and explicit gate errors.

- [ ] **Step 1: Write failing quality-gate tests**

Assert the current dataset audit accepts all required types and valid assertion
scope. Assert a synthetic output with 83% coverage and 89% one-type share is
rejected, while a sparse balanced fixture is accepted.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest v2/tests/test_training_dataset_contract.py v2/tests/test_output_quality_gate.py -q`

Expected: FAIL because no output collapse gate exists.

- [ ] **Step 3: Implement reports and phase enforcement**

Compute counts, shares, coverage, boundary errors, multiline spans, and span
lengths. Save diagnostics before publishing and raise a contract error on
pathological thresholds.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest v2/tests/test_training_dataset_contract.py v2/tests/test_output_quality_gate.py -q`

Expected: PASS.

### Task 6: Stage observability, notebook regeneration, and full verification

**Files:**
- Modify: `v2/scripts/train_ner_subprocess.py`
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py`
- Modify: `v2/tools/build_kaggle_notebook.py`
- Regenerate: `v2/medical_information_extraction_kaggle.ipynb`
- Modify: `v2/KAGGLE_RUNBOOK.md`
- Test: `v2/tests/test_kaggle_phase_results.py`
- Test: `v2/tests/test_kaggle_notebook_contract.py`

**Interfaces:**
- Produces: per-stage metrics JSON and notebook-visible result summaries.

- [ ] **Step 1: Write failing observability tests**

Assert stage results expose train/validation window counts, configured epochs,
learning rate, train loss, evaluation metric, and best checkpoint. Assert phase
6 calls its 32-document count `sample_window_count`.

- [ ] **Step 2: Verify RED**

Run: `python -m pytest v2/tests/test_kaggle_phase_results.py v2/tests/test_kaggle_notebook_contract.py -q`

Expected: FAIL because current phase results expose only checkpoint hashes and
an ambiguous `window_count`.

- [ ] **Step 3: Publish metrics and regenerate notebook**

Write a compact `training_result.json`, include it in phase results, update the
runbook, and regenerate the notebook using the repository builder.

- [ ] **Step 4: Run targeted and full verification**

Run:

```powershell
$env:PYTHONPATH='v2'
python -m pytest v2/tests/test_owner_window_native_overflow.py v2/tests/test_assertion_type_contract.py v2/tests/test_kaggle_curriculum_execution.py v2/tests/test_assertion_calibration_contract.py v2/tests/test_kb_word_boundaries.py v2/tests/test_inference_quality_filter.py v2/tests/test_training_dataset_contract.py v2/tests/test_output_quality_gate.py v2/tests/test_kaggle_phase_results.py v2/tests/test_kaggle_notebook_contract.py -q
python -m pytest v2/tests -q
```

Expected: all tests pass, with only the existing environment-dependent skips.

- [ ] **Step 5: Review diff and commit**

Review only the files listed in this plan, then commit on
`codex/kaggle-end-to-end-pipeline` without staging unrelated workspace files.
