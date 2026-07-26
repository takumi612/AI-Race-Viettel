# Entity Quality Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace misleading token-only checkpoint selection and unsafe inference recovery with entity-level calibration and precision-first safeguards that directly target the competition's WER/entity matching failure mode.

**Architecture:** Training will calculate BIO span metrics and a confidence threshold from held-out windows, persist the calibration beside the checkpoint, and inference will enforce that threshold. Exact KB recovery will group aliases into candidate pools and reject short or ambiguous aliases before they can outrank NER. Phase diagnostics and quality gates will expose entity confidence, suspicious spans, candidate coverage, and calibration provenance.

**Tech Stack:** Python 3.10, PyTorch, Hugging Face Transformers, NumPy, pytest, nbformat-based deterministic notebook builder.

## Global Constraints

- Preserve raw-text offsets; no normalization may alter submission positions.
- Keep `ENABLE_QWEN_RERANKER` as the only user-facing Qwen enable switch.
- Qwen remains a strict required phase when enabled and is not loaded when disabled.
- Use entity-level metrics for model selection; token metrics remain diagnostics only.
- Do not use hidden competition input for training or fitting.
- Implement every behavior change test-first and regenerate the canonical notebook deterministically.

---

### Task 1: Entity-level BIO metrics and NER confidence calibration

**Files:**
- Modify: `v2/clinical_nlp_lab/training.py`
- Modify: `v2/scripts/train_ner_subprocess.py`
- Create: `v2/tests/test_ner_calibration.py`
- Modify: `v2/tests/test_training_helpers.py`

**Interfaces:**
- Produces: `compute_bio_span_metrics(eval_prediction) -> dict[str, float]` with `entity_precision`, `entity_recall`, `entity_f1`, and `ner_confidence_threshold`.
- Produces: `write_ner_calibration(path, metrics)` storing a versioned threshold artifact.
- Consumes: Hugging Face `(logits, labels)` evaluation payloads with padding label `-100` and `O` label `0`.

- [ ] **Step 1: Write failing tests for exact BIO span scoring**

Add literal logits/labels where token accuracy is high but a wrong `B/I` boundary makes entity F1 zero, and assert the new metric reports that discrepancy.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_ner_calibration.py tests/test_training_helpers.py -q -p no:cacheprovider`

Expected: FAIL because `compute_bio_span_metrics` and the calibration artifact do not exist.

- [ ] **Step 3: Implement BIO span extraction and deterministic threshold search**

Extract typed spans per evaluation window, score exact matches, search a fixed confidence grid, break F1 ties toward the higher threshold, and return token metrics only as secondary diagnostics.

- [ ] **Step 4: Select checkpoints using entity F1 and persist calibration**

Change Trainer's `metric_for_best_model` from `f1` to `entity_f1`, call the same metric on the selected checkpoint, and write `ner_calibration.json` into `ner_model`.

- [ ] **Step 5: Run focused tests and commit**

Run: `python -m pytest tests/test_ner_calibration.py tests/test_training_helpers.py tests/test_kaggle_phase_results.py -q -p no:cacheprovider`

Commit: `fix: select NER checkpoints by calibrated entity F1`

### Task 2: Apply NER calibration during inference

**Files:**
- Modify: `v2/clinical_nlp_lab/ner.py`
- Modify: `v2/clinical_nlp_lab/runtime_bundle.py`
- Modify: `v2/clinical_nlp_lab/inference.py`
- Create: `v2/tests/test_ner_confidence_filter.py`

**Interfaces:**
- Consumes: `ner_model/ner_calibration.json` with `schema_id`, `schema_version`, `confidence_threshold`, and validation metrics.
- Produces: `TransformerNERDetector.confidence_threshold: float` and filters only complete decoded entities after chunk merging.

- [ ] **Step 1: Write failing detector tests**

Use real `EntityAnnotation` instances to assert that spans below the persisted threshold are removed while equal/above-threshold spans and offsets are preserved.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_ner_confidence_filter.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement strict artifact loading and confidence filtering**

Reject malformed thresholds, use a conservative documented fallback for legacy checkpoints, and expose the loaded value in runtime diagnostics.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_ner_confidence_filter.py tests/test_primary_inference_path.py tests/test_inference_data_flow.py -q -p no:cacheprovider`

Commit: `fix: enforce calibrated NER confidence at inference`

### Task 3: Make KB-first recovery precision-safe and Qwen-ready

**Files:**
- Modify: `v2/clinical_nlp_lab/runtime_bundle.py`
- Modify: `v2/clinical_nlp_lab/inference.py`
- Create: `v2/tests/test_kb_recovery_precision.py`
- Modify: `v2/tests/test_retrieval_resources.py`

**Interfaces:**
- Produces: one `SpanProposal` per `(entity_type, alias, raw span)` with a ranked candidate pool.
- Consumes: ICD-10/RxNorm records containing `candidate_id` and aliases.

- [ ] **Step 1: Write failing tests for unsafe and ambiguous aliases**

Assert that generic one-token aliases such as `và`, `tin`, and `phẫu` never become entities; unique clinical aliases remain recoverable; and an alias shared by multiple codes produces one ranked pool rather than competing 0.99-confidence spans.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_kb_recovery_precision.py tests/test_retrieval_resources.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement alias safety and grouped recovery**

Normalize once, require clinically meaningful token length, reject a fixed Vietnamese/English function-word set, group candidate IDs deterministically, and leave ambiguous pools unselected for the policy while preserving them for Qwen.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_kb_recovery_precision.py tests/test_retrieval_resources.py tests/test_required_qwen_phase.py -q -p no:cacheprovider`

Commit: `fix: prevent unsafe KB aliases from creating entities`

### Task 4: Add semantic output diagnostics and fail unsafe submissions

**Files:**
- Modify: `v2/clinical_nlp_lab/output_quality.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py`
- Create: `v2/tests/test_output_semantic_quality.py`
- Modify: `v2/tests/test_kaggle_observability.py`

**Interfaces:**
- Produces: quality fields `single_token_entity_count`, `suspicious_generic_span_count`, `candidate_eligible_count`, and `candidate_link_rate`.
- Consumes: final submission entities plus raw text; diagnostics must not change the submission schema.

- [ ] **Step 1: Write failing quality-gate tests**

Build a literal output containing generic entities such as `và`, `không`, and `lúc`, assert the audit counts them, and assert the gate rejects a repeated generic-span collapse while accepting valid short clinical concepts such as `sốt`.

- [ ] **Step 2: Run the focused tests and confirm RED**

Run: `python -m pytest tests/test_output_semantic_quality.py tests/test_kaggle_observability.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement diagnostics and phase reporting**

Add precision-oriented semantic counts, candidate coverage by eligible type, calibration threshold provenance, and actionable error text without altering output JSON records.

- [ ] **Step 4: Run focused tests and commit**

Run: `python -m pytest tests/test_output_semantic_quality.py tests/test_kaggle_observability.py tests/test_required_qwen_phase.py -q -p no:cacheprovider`

Commit: `feat: gate semantically collapsed submission output`

### Task 5: Regenerate notebook and verify end-to-end contracts

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py` only if new diagnostics must be surfaced.
- Regenerate: `v2/medical_information_extraction_kaggle.ipynb`
- Modify: `reports/2026-07-26-entity-quality-root-cause.md`

**Interfaces:**
- Consumes: all production changes from Tasks 1-4.
- Produces: deterministic canonical Kaggle notebook and final evidence report.

- [ ] **Step 1: Regenerate the notebook**

Run: `python tools/build_kaggle_notebook.py`

- [ ] **Step 2: Run focused integration tests**

Run: `python -m pytest tests/test_notebook_orchestrator_contract.py tests/test_kaggle_observability.py tests/test_required_qwen_phase.py -q -p no:cacheprovider`

- [ ] **Step 3: Run the complete suite**

Run: `python -m pytest tests -q -p no:cacheprovider`

- [ ] **Step 4: Validate generated notebook and diff hygiene**

Run notebook syntax/determinism validation and `git diff --check`.

- [ ] **Step 5: Write the final root-cause and rerun guidance report**

Record the measured 100-file output statistics, explain why the old 0.999 metric was misleading, list the fixed safeguards, and state that a fresh Kaggle GPU run plus leaderboard submission is required to measure the hidden-test score.

- [ ] **Step 6: Commit**

Commit: `docs: document entity quality hardening evidence`
