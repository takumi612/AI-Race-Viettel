# Contract-First Kaggle Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Connect the existing data, model, inference, and runtime modules into a truthful CPU-testable Kaggle pipeline without local model training.

**Architecture:** Keep the canonical raw-offset contracts as the source of truth. Add narrow integration boundaries: training-contract preparation, final-model inference, and a phase orchestrator with injectable phase runners. The notebook calls the orchestrator API and contains no copied business logic.

**Tech Stack:** Python 3.10+, PyTorch for tensor contracts, pytest, JSON/JSONL, SHA-256, standard-library atomic filesystem operations.

## Global Constraints

- Do not run local training, model download, checkpoint reload, or local end-to-end acceptance.
- Keep `max_length=512` and `stride=128` as the owner-window contract.
- Raw text remains authoritative for every output offset.
- Candidate retrieval may keep at most 20 internal candidates and emits at most one candidate.
- Qwen remains optional and cannot invalidate deterministic output.
- Resume must reject incompatible dataset/config/checkpoint hashes.
- Existing user untracked files and unrelated modifications must remain untouched.

## Task 1: Training contract integration

**Files:**
- Modify: `v2/clinical_nlp_lab/training.py`
- Modify: `v2/clinical_nlp_lab/curriculum.py`
- Create: `v2/tests/test_training_contract.py`

- [x] Write failing tests for owner-window batch preparation, deterministic stage manifests, and hash-guarded resume.
- [x] Implement `TrainingContract`, `build_training_contract`, and `StageManifest` helpers without loading a model.
- [x] Route token-classification feature preparation through owner windows and the existing collator boundary while retaining Trainer-compatible labels.
- [x] Run focused tests and the existing owner-window/curriculum tests.

## Task 2: Final model primary path

**Files:**
- Modify: `v2/clinical_nlp_lab/assertions.py`
- Modify: `v2/clinical_nlp_lab/candidate_policy.py`
- Modify: `v2/clinical_nlp_lab/inference.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py`
- Create: `v2/tests/test_primary_inference_path.py`

- [x] Write failing tests proving NER proposals, KB recovery, assertion masking, candidate policy, and raw-offset validation compose deterministically.
- [x] Implement dependency-injected NER/assertion/linker adapters in `infer_document`.
- [x] Enforce candidate top-20/internal and output-at-most-one at the integration boundary.
- [x] Run focused inference tests without model weights.

## Task 3: Runtime orchestrator and artifact lifecycle

**Files:**
- Modify: `v2/clinical_nlp_lab/orchestration.py`
- Create: `v2/tests/test_orchestration_contract.py`

- [x] Write failing tests for 13 phase order, `full`/`resume`/`inference_only`, atomic staging, terminal events, and stale resume rejection.
- [x] Implement phase dispatch with injectable runners, JSONL lifecycle events, immutable run directories, and atomic `LATEST.json` publication.
- [x] Make missing phase runners fail closed; real training/inference hooks remain explicit for Kaggle.
- [x] Run focused orchestrator tests.

## Task 4: Notebook and documentation handoff

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py`
- Modify: `v2/KAGGLE_RUNBOOK.md`
- Modify: `v2/PIPELINE_VI.md`
- Modify: `docs/superpowers/plans/2026-07-23-contract-first-resource-safe-kaggle-execution.md`

- [x] Write failing source-contract tests requiring notebook API calls and no embedded training implementation.
- [x] Make the canonical notebook call the orchestrator API while preserving deterministic generation.
- [x] Document three run modes, artifact handoff, resume rules, and the fact that Kaggle `Run All` remains user-owned.
- [x] Update the canonical plan with a verified status section; Kaggle acceptance remains externally pending.
- [x] Run the full CPU suite, notebook generator determinism, and `git diff --check`.

## Continuation verification — 2026-07-23

- [x] Curriculum manifests are carried into `TrainingContract` for resume validation.
- [x] Assertion encoder binding is frozen and remains in eval mode during head training.
- [x] Persisted candidate calibration is loaded by the production policy boundary.
- [x] Final-bundle inference writes schema/offset-validated JSON and CRC-checked `output.zip`.
- [x] Missing phase runners fail closed with an atomic error artifact; no false `PASS`.
- [x] Fresh suite: `347 passed, 2 skipped`.
- [x] Bind built-in Kaggle phase runners, distributed stage training subprocess, final head/inference/package phases.
- [x] Generate 13 observable notebook phase cells with JSON progress and traceback boundaries.
- [ ] Obtain a real Kaggle `Run All` artifact; this cannot be verified locally.
