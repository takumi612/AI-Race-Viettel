# Centralized Entity Span Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task with verification checkpoints.

**Goal:** Ensure every entity emitted by inference, Qwen mutation, submission serialization, and output auditing obeys one shared span/surface contract.

**Architecture:** Add a dependency-light primitive validator returning structured violation codes. Replace duplicated checks in proposal merge, Qwen trim, submission validation, and output audit with this validator. Invalid Qwen trims preserve the original valid entity and record fallback reasons.

**Tech Stack:** Python 3.11+, dataclasses, `re`, pytest, existing `clinical_nlp_lab` package.

## Global Constraints

- Do not blacklist medically meaningful special characters.
- Do not weaken output-quality thresholds.
- Do not change Qwen/vLLM/GPU configuration, training, or datasets.
- Preserve existing public output-quality report keys.
- Keep exact raw-text offsets.
- Write failing tests before production code for each behavior.

---

### Task 1: Add the shared span policy

**Files:**
- Create: `v2/clinical_nlp_lab/entity_span_policy.py`
- Create: `v2/tests/test_entity_span_policy.py`

**Interfaces:**
- `validate_entity_span(raw_text: str, start: int, end: int, text: str, *, max_length: int | None = None) -> tuple[str, ...]`
- `is_suspicious_generic_surface(text: str) -> bool`
- `SUSPICIOUS_GENERIC_SURFACES: frozenset[str]`

- [ ] **Step 1: Write failing table-driven tests** for valid medical punctuation and every violation code.
- [ ] **Step 2: Run `pytest v2/tests/test_entity_span_policy.py -q` and confirm import/behavior failures.**
- [ ] **Step 3: Implement the compiled regex and raw-text boundary checks.**
- [ ] **Step 4: Run the focused policy tests and confirm they pass.**

### Task 2: Enforce the policy in Qwen trim

**Files:**
- Modify: `v2/clinical_nlp_lab/qwen_entity_validator.py`
- Modify: `v2/tests/test_qwen_entity_validator.py`

**Interfaces:**
- Keep existing `EntityValidationDecision` and result APIs.
- Add `trim_fallback_reasons` to `EntityValidationCounters`.

- [ ] **Step 1: Add failing tests for punctuation-only trim fallback, fallback diagnostics, and valid special-character trims.**
- [ ] **Step 2: Run the focused tests and confirm the new tests fail.**
- [ ] **Step 3: Replace the private `output_quality` import with the shared policy; validate original and proposed spans; preserve originals on hard violations or generic surfaces; update counter arithmetic.**
- [ ] **Step 4: Run `pytest v2/tests/test_qwen_entity_validator.py -q`.**

### Task 3: Replace proposal and output-quality duplication

**Files:**
- Modify: `v2/clinical_nlp_lab/inference.py`
- Modify: `v2/clinical_nlp_lab/output_quality.py`
- Modify: `v2/tests/test_output_quality_gate.py`
- Modify: `v2/tests/test_output_semantic_quality.py`

- [ ] **Step 1: Add failing consistency tests proving proposal filtering and audit classify the same invalid fixtures.**
- [ ] **Step 2: Run focused inference/output tests and confirm failure.**
- [ ] **Step 3: Refactor both consumers to aggregate shared violation codes while preserving report keys and thresholds.**
- [ ] **Step 4: Run focused inference/output tests and confirm pass.**

### Task 4: Preserve the schema/output-policy boundary

**Files:**
- Test: `v2/tests/test_schema_validation.py`
- Test: `v2/tests/test_preflight.py`

- [ ] **Step 1: Add a regression test proving schema validation accepts organizer multiline and 162-character entities while retaining exact-offset checks.**
- [ ] **Step 2: Run the focused schema test and confirm it fails if output-only policy leaked into schema validation.**
- [ ] **Step 3: Keep output-only policy out of `validate_submission_payload()` and enforce it in inference/output quality.**
- [ ] **Step 4: Run schema and preflight tests and confirm pass.**

### Task 5: Full regression verification

**Files:**
- No additional production files.

- [ ] **Step 1: Run `pytest v2/tests/test_entity_span_policy.py v2/tests/test_qwen_entity_validator.py v2/tests/test_output_quality_gate.py v2/tests/test_output_semantic_quality.py v2/tests/test_schema_validation.py v2/tests/test_preflight.py -q`.**
- [ ] **Step 2: Run `pytest v2/tests -q`.**
- [ ] **Step 3: Run `python -m compileall v2/clinical_nlp_lab`.**
- [ ] **Step 4: Inspect `git diff --check` and review changed files.**
- [ ] **Step 5: Commit implementation and tests with `fix: centralize entity span validation`.**
