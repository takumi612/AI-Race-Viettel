# Metadata-First Curriculum Training Design

## 1. Purpose

This specification defines the target training and inference design for the Viettel AI Race clinical information extraction pipeline. It extends the existing end-to-end Kaggle design with four metric-oriented improvements:

1. source-aware three-stage NER training;
2. metadata-first context handling without changing raw text;
3. boundary-safe owner-window supervision;
4. independently trained assertion classification and KB-first entity recovery.

The design must use the 2,000 synthetic v2 records for coverage without allowing their writing style or template distribution to dominate the 100 trusted organizer-labelled records.

## 2. Competition contract and optimization target

The pipeline extracts five entity types:

- diagnosis;
- drug;
- symptom;
- laboratory test name;
- laboratory test result.

Only the official assertion fields are emitted:

- `isNegated`;
- `isHistorical`;
- `isFamily`.

ICD-10 candidates are attached only to diagnosis entities. RxNorm candidates are attached only to drug entities. Other entity types have no ontology candidates.

The observed organizer labels contain at most one final candidate per diagnosis or drug entity. Therefore, the submission default remains `candidate_output_k = 1`. This does not limit retrieval: the internal candidate pool remains configurable, with a default of 20 candidates before filtering and reranking.

The local evaluator is a development proxy, not confirmed organizer scoring code. Token-level F1 and the current 30/30/40 local composite are reported for diagnostics, but model selection must prioritize document-level entity quality and separately report assertion and linking quality.

## 3. Data policy

The canonical corpus is `data_v2/Training_data/synthetic_train_v2`.

| IDs | Origin | Role |
| --- | --- | --- |
| 1-100 | Organizer inputs with reconstructed/self-generated GT | Quarantine and audit only; excluded from model fitting and threshold calibration |
| 101-200 | Organizer inputs with organizer-provided leaked GT | Trusted real-labelled corpus |
| 201-2200 | Synthetic v2 | Coverage, representation warm-up, rare concepts, genre diversity, and regularization |

The 2,000 synthetic records are not discarded. They must all be eligible for Stage 1 and Stage 2 training. Stage 3 uses a selected replay subset so that final adaptation follows organizer language rather than synthetic style.

The existing raw input and GT files remain immutable during training. Derived metadata, splits, features, and calibration results are written as sidecar artifacts.

### 3.1 Development split

Hyperparameter and checkpoint selection use a fixed development split:

- 80 trusted real documents for training;
- 20 trusted real documents for `real_holdout`;
- a grouped synthetic training partition and grouped `synthetic_holdout` for diagnostics.

Splitting is group-aware. Document ID, template family, normalized primary surface group, and near-duplicate group must not cross a train/holdout boundary. The real split should preserve entity-type and assertion-label coverage as far as group constraints allow.

The real holdout is the primary model-selection set. The synthetic holdout diagnoses coverage and memorization but cannot override a degradation on the real holdout.

### 3.2 Final-fit protocol

After selecting the curriculum schedule, thresholds, and other hyperparameters, a final fit uses all 100 trusted real documents. No score produced on these 100 documents after final fitting is reported as an unbiased validation score.

The final-fit run uses the frozen decisions selected during development:

- selected epoch counts, stage endpoints, and learning rates;
- sampling proportions;
- assertion thresholds;
- linking thresholds and margins;

Final fit does not early-stop or select a checkpoint against the 100 trusted documents used for fitting. It follows the development-selected schedule and saves the configured final stage endpoint.

The final artifact records both the development split metrics and the fact that the delivered model was subsequently fitted on all trusted real labels.

## 4. Metadata-first context model

Raw clinical text is never rewritten because output offsets must refer to the original document. Context is represented in a sidecar record with the following fields where detectable:

- `document_id`;
- `source_bucket`: `quarantine`, `organizer`, or `synthetic`;
- `document_genre`;
- `template_group` and `near_duplicate_group`;
- section spans and normalized section types;
- speaker or author role;
- experiencer/subject, such as patient or family member;
- long-tail and ontology-coverage flags;
- content hash and builder version.

Section-heading rules are permitted because they identify document structure. Content regexes for diseases, drugs, or symptoms are not primary label generators. Laboratory value parsers may remain as structured enrichment, provided their detections are not treated as unquestioned ground truth.

Metadata can be used for:

- group-aware splitting;
- source-aware sampling;
- assertion context;
- prediction diagnostics;
- optional special context tokens only if ablation on `real_holdout` shows a gain.

The baseline does not inject metadata tokens into the encoder. This avoids changing text offsets and prevents noisy metadata detection from becoming a required model input.

## 5. Boundary-safe chunking

The default tokenizer window is:

- `max_length = 512` tokens including special tokens;
- overlap/stride target of 128 tokens.

Every feature retains the document ID, raw character offsets, window bounds, and source metadata.

### 5.1 Owner-window supervision

Each gold entity is assigned to exactly one owner window. Eligible windows must contain the complete entity span. The owner is the eligible window that maximizes the entity's minimum token distance from the left and right usable boundaries. Ties are resolved by the earliest window index.

Training labels follow these rules:

1. the owner window receives the complete BIO labels for the entity;
2. copies of that entity in other overlapping windows are masked with `-100` for loss computation;
3. a window containing only part of an entity masks the visible entity tokens with `-100`;
4. non-entity tokens remain valid `O` supervision unless masked by tokenizer special/padding rules;
5. if no window contains a complete entity, preprocessing fails for that document and records a diagnostic rather than silently converting it to `O`.

This makes every entity contribute once, prevents overlap from changing class frequency, and removes false boundary supervision.

### 5.2 Inference merge

Predictions are projected to raw character offsets. Exact duplicates are collapsed first. Same-type overlapping spans are merged using calibrated confidence and boundary completeness. Cross-type conflicts are resolved deterministically by source reliability, confidence, and span completeness, with all discarded alternatives recorded in diagnostics.

## 6. Three-stage NER curriculum

The NER backbone remains XLM-R unless a controlled real-holdout experiment justifies a replacement. Twenty epochs is a hard safety cap, not a target duration.

### 6.1 Stage 1: synthetic warm-up

Purpose: learn the five entity types, broad lexical variation, clinical genres, rare ICD/RxNorm surfaces, and boundary behavior.

- Data: synthetic training partition during development; all 2,000 synthetic records during final fit.
- Epoch cap: 3, with an expected range of 1-3.
- Initial learning-rate search range: `2e-5` to `3e-5`.
- Selection signal during development: synthetic entity metrics plus a no-regression check on the real holdout.
- Output: `stage1_synthetic_checkpoint`.

### 6.2 Stage 2: source-balanced mixed training

Purpose: align representations with organizer writing while retaining synthetic coverage.

- Data: 80 trusted real training documents plus the synthetic training partition during development; all 100 trusted real documents plus all 2,000 synthetic records during final fit.
- Sampling unit: owner-labelled chunks grouped by source and document.
- Target sampling exposure: 30-40% organizer chunks and 60-70% synthetic chunks per epoch-equivalent.
- Epoch cap: 2, with an expected range of 1-2.
- The sampler must not achieve balance by copying files or modifying the corpus.
- Output: `stage2_mixed_checkpoint`.

### 6.3 Stage 3: organizer adaptation with synthetic replay

Purpose: maximize fit to organizer language and annotation style without catastrophic forgetting of rare concepts.

- Data during development: 80 trusted real training documents plus replay selected from synthetic training data.
- Data during final fit: all 100 trusted real documents plus replay selected using the frozen development policy.
- Replay exposure: 15-20% of Stage 3 examples.
- Replay prioritizes rare entity surfaces, long-tail ICD/RxNorm concepts, rare genres, rare assertion combinations, and boundary-hard examples.
- Initial learning-rate search range: `5e-6` to `1e-5`.
- Development epoch cap: 8; early stopping is expected to terminate earlier.
- Final fit uses the Stage 3 epoch count selected during development and does not evaluate against the fitted 100 trusted documents.
- Output: `final_ner_model`.

### 6.4 Early stopping and checkpoint selection

Development checkpoint selection uses document-level predictions on `real_holdout`:

`selection_score = 0.70 * exact_entity_f1 + 0.20 * overlap_entity_f1 + 0.10 * macro_type_f1`

Requirements:

- exact-span F1 is micro-averaged across entities;
- overlap F1 uses one-to-one matching and a documented overlap threshold;
- macro type F1 averages the five entity types equally;
- token-level F1 is diagnostic only;
- patience is applied to the selection score, not training loss;
- a checkpoint cannot be selected if output schema or offset validation fails.

Per-type metrics, per-genre metrics, seen/unseen-surface metrics, and boundary-error counts are persisted for analysis.

## 7. Assertion model

Assertions are handled by a separate multi-label classifier instead of broad content regexes.

For each detected or gold entity, the classifier receives:

- an entity-marked local context window;
- section and experiencer metadata when available;
- the entity type;
- the raw sentence/clause context.

It predicts independent probabilities for `isNegated`, `isHistorical`, and `isFamily`. Thresholds are calibrated per label on the trusted real holdout. Synthetic assertion examples provide regularization and coverage, but organizer examples receive higher sampling weight.

To fit Kaggle resources, the assertion classifier may reuse a frozen copy of the NER encoder or use a lightweight classification head/adapter. The chosen variant must be based on real-holdout assertion F1 and runtime cost.

Rules are restricted to high-precision structural fallback and diagnostics. They must not infer assertions from unscoped keywords across a full document. Any rule cue is bounded by clause/sentence and section scope.

## 8. Ontology retrieval, recovery, and linking

Entity discovery has two independent recall paths:

1. NER predictions;
2. KB-first phrase and alias retrieval over ICD-10 and RxNorm.

The KB-first path searches normalized exact aliases first, then controlled lexical/fuzzy or semantic alternatives. It can propose a disease or drug mention that NER missed, but it cannot directly force the mention into output. Proposals pass span verification, type checks, confidence thresholds, conflict resolution, and candidate calibration.

For every accepted diagnosis or drug mention:

1. lexical and semantic retrieval produce up to 20 internal candidates;
2. drug mention heads are separated from strength, route, form, and frequency;
3. deterministic filters remove type-incompatible or invalid KB entries;
4. a reranker orders the remaining pool;
5. minimum score, top-1/top-2 margin, and abstention thresholds are applied;
6. at most one candidate is emitted.

Retrieval quality is measured separately with recall@1, recall@5, recall@10, top-1 accuracy, coverage, and accuracy conditional on non-abstention. Thresholds are selected on the real holdout and frozen for final fit.

Qwen is optional and limited to ambiguous candidate reranking or assertion fallback. It may only choose from supplied valid candidates. Missing weights, invalid JSON, timeout, or CUDA failure must fall back to deterministic behavior and must not abort the pipeline.

## 9. Kaggle notebook orchestration

The canonical notebook remains `v2/medical_information_extraction_kaggle.ipynb` and supports:

- `RUN_MODE = "full"`: validate data, train all stages, calibrate development components where applicable, run inference, and package artifacts;
- `RUN_MODE = "resume"`: resume from the latest complete stage whose manifest and checkpoint pass integrity checks;
- `RUN_MODE = "inference_only"`: load final artifacts and generate competition output without training.

The notebook executes these logical phases:

1. environment, dataset, and KB discovery;
2. deterministic validation and manifest construction;
3. metadata extraction and group-aware development split;
4. owner-window feature construction;
5. Stage 1 synthetic warm-up;
6. Stage 2 source-balanced mixed training;
7. Stage 3 organizer adaptation;
8. assertion training/calibration;
9. linking retrieval and threshold calibration;
10. final-fit path when enabled;
11. checkpoint reload smoke test;
12. inference, output validation, and artifact packaging.

Each completed stage writes an atomic stage manifest. Resume is allowed only when the manifest matches dataset hashes, code/config version, tokenizer identity, label mapping, and checkpoint inventory.

## 10. Artifacts and observability

The run produces, at minimum:

- `stage1_synthetic_checkpoint/`;
- `stage2_mixed_checkpoint/`;
- `final_ner_model/`;
- `assertion_model/`;
- `candidate_calibration.json`;
- `metadata_manifest.jsonl`;
- `split_manifest.json`;
- `training_history.json`;
- `evaluation_report.json`;
- `run_manifest.json`;
- `diagnostics/run_summary.json`;
- `output.zip`;
- `trained_ner_artifacts.zip`.

Intermediate optimizer checkpoints may be deleted after the selected checkpoint is safely reloaded and the final archive is verified. Delivered model archives must retain all files required for offline reload, preprocessing, label mapping, assertion inference, and candidate calibration.

Diagnostics must distinguish:

- NER-origin versus KB-first-origin entities;
- entities removed by overlap/type conflict;
- assertion classifier versus fallback decisions;
- linking abstentions and their reasons;
- per-source training exposure;
- stage runtime and peak memory;
- unseen/rare-surface performance.

## 11. Failure handling

The pipeline fails early for:

- invalid input/GT pairs;
- schema or offset errors;
- candidates absent from the bundled KB;
- train/holdout group leakage;
- an entity that cannot be fully represented by any configured window;
- an unloadable selected checkpoint;
- mismatched resume manifests;
- malformed output or invalid ZIP structure.

Optional semantic retrieval or Qwen failures degrade to lexical/deterministic behavior and are recorded as warnings. They do not invalidate an otherwise correct run.

## 12. Verification strategy

Implementation follows test-driven development. Required test layers are:

1. **Unit tests**: metadata parsing, owner-window assignment, partial-span masking, source-aware sampling, document-level metrics, assertion thresholds, retrieval truncation, abstention, and deterministic fallbacks.
2. **Integration tests**: three-stage checkpoint handoff, resume integrity, model reload, end-to-end inference, output schema, offsets, and candidate validity.
3. **Data tests**: zero pairing/schema/offset/KB errors, zero split leakage, source counts, and replay-distribution checks.
4. **Fast development smoke test**: a tiny local dataset completes all logical stages without downloading a large model.
5. **Kaggle acceptance run**: a real GPU `Run All` produces reloadable weights and CRC-valid output archives.

Independent review agents may audit label semantics and implementation changes when requested. Their review is advisory; deterministic checks and trusted organizer labels remain the source of truth.

## 13. Acceptance criteria

The design is implemented successfully only when all of the following hold:

- IDs 1-100 are excluded from fitting and calibration.
- All 2,000 synthetic records are eligible for Stage 1 and Stage 2.
- The final model uses all 100 trusted organizer-labelled records after development decisions are frozen.
- No gold entity contributes NER loss in more than one overlapping window.
- Partial entities are never converted to `O` supervision.
- Model selection uses document-level entity metrics on the real holdout.
- Assertion thresholds and candidate abstention thresholds are calibrated on trusted real holdout data.
- Generic disease/drug/symptom content regexes are absent from the primary detector.
- Internal retrieval keeps multiple candidates while final output emits at most one.
- Every emitted candidate exists in the bundled ICD-10/RxNorm KB.
- Local unit and integration tests pass.
- Saved models reload and reproduce schema-valid inference.
- `output.zip` and model archives pass file inventory and CRC validation.
- Kaggle success is claimed only after a real `Run All` execution has been audited.

## 14. Non-goals

- Treating reconstructed GT from IDs 1-100 as trusted training labels.
- Replacing NER with regex phrase lists.
- Training Qwen as the primary entity detector.
- Assigning ICD-10/RxNorm candidates to symptoms or laboratory entities when the competition contract does not require them.
- Rewriting raw documents or using normalized text offsets in submissions.
- Claiming an unbiased final validation score after fitting on all 100 trusted organizer-labelled records.
- Expanding into relation extraction before entity, assertion, and linking components are stable.

## 15. Relationship to existing specifications

This document refines and takes precedence for curriculum training, metadata handling, chunk supervision, assertions, and KB-first recovery. The following existing specifications remain valid where they do not conflict with this document:

- `docs/superpowers/specs/2026-07-22-kaggle-end-to-end-clinical-pipeline-design.md`;
- `docs/superpowers/specs/2026-07-22-synthetic-train-v2-design.md`.

The synthetic v2 corpus itself is not regenerated by this specification. Dataset changes are required only when validation or semantic audit identifies a concrete error.
