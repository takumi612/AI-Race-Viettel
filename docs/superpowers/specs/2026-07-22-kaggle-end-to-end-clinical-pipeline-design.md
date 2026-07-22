# Kaggle End-to-End Clinical NLP Pipeline Design

## Goal

Make `v2/` a reproducible end-to-end Kaggle workflow that validates and splits the training corpus, fine-tunes a clinical NER checkpoint on a single 16 GB Kaggle GPU, runs ontology-assisted inference, produces schema-valid competition output, and packages reloadable weights plus evidence-rich reports.

## Runtime target

- Lowest supported accelerator: one Kaggle T4 or P100 with 16 GB VRAM.
- A second T4 may be used when available, but correctness cannot depend on multi-GPU execution.
- The training path must not require Qwen or vLLM.
- Qwen reranking is optional and must degrade to deterministic ranking without aborting inference.
- The canonical entry point is `v2/medical_information_extraction_kaggle.ipynb` with a single `Run All` flow.

## Source-of-truth data

The maintained corpus is `data_v2/Training_data/synthetic_train_v2` with paired `input/<id>.txt` and `gt/<id>.json` files.

- IDs 1-100 are organizer inputs with reconstructed labels and must remain auditable separately.
- IDs 101-200 contain organizer-provided ground truth and must never be silently rewritten.
- IDs 201-2200 are generated training documents and may be replaced when validation detects template duplication, clinical inconsistency, invalid ontology candidates, or assertion errors.

Every build must emit a manifest recording document origin, genre, long-tail status, template family, primary ICD/RxNorm candidates, and content hashes. Dataset validation must reject missing pairs, invalid UTF-8, invalid offsets, invalid output keys, assertions on unsupported entity types, missing candidates for disease/drug entities, and candidates absent from the bundled ICD-10/RxNorm knowledge bases.

## Evaluation-safe splitting

Random document splitting is not acceptable because generated documents share templates and ontology concepts. The splitter will group documents by template family and normalized primary entity surfaces. It will stratify by source bucket, genre, entity type, and long-tail status where possible.

The workflow reports two validation slices:

1. `real_holdout`: organizer-origin documents excluded from training for that run.
2. `synthetic_holdout`: generated documents whose template groups are excluded from training.

No document ID, template group, or normalized primary surface group may cross train and validation. The split manifest and leakage audit are persisted with the weights.

## Preprocessing and NER

Raw text remains immutable so submission offsets always index the original document. Unicode normalization and alias normalization are derived views only.

Documents are segmented by detected clinical sections and then tokenized into overlapping model windows. Every feature stores document ID and raw offsets. Predictions from overlapping windows are merged deterministically.

XLM-R is fine-tuned for `DISEASE`, `DRUG`, `SYMPTOM`, `LAB_NAME`, and `LAB_RESULT`. Model selection uses entity-level exact-span F1, with token-level metrics retained only as diagnostics. The saved model must include `model.safetensors`, tokenizer files, label mappings, training configuration, split manifest, training history, and evaluation report.

Ontology phrase scanning runs independently of NER. It is a recall layer, not a gate: an entity can originate from NER, an exact ICD/RxNorm alias, or both. Generic symptom and patient-info regexes are removed from the primary detector. Lab regex remains only as a structured parser for value and unit enrichment.

## Entity merge and assertions

Overlaps are resolved by source reliability, type compatibility, span completeness, and confidence. Exact ontology spans and high-confidence NER spans outrank generic parsers. Conflicting detections remain visible in diagnostics.

Assertion cues operate within sentence/clause and section boundaries. The rule layer must not treat `man tinh` as historical, `nguoi nha` alone as family experiencer, or `theo doi` alone as diagnostic uncertainty. Rule assertions are deterministic fallback behavior. Qwen assertion resolution is optional and only handles ambiguous cases.

## ICD-10 and RxNorm linking

Only disease and drug entities receive competition candidates. Retrieval combines lexical and semantic ranking over the bundled knowledge bases and returns a configurable pool, default 20.

Drug queries use the medication mention head; strength, route, form, and frequency are parsed separately and used for filtering/reranking. Candidate selection applies a real score threshold and top-1/top-2 margin. Low-confidence mentions may abstain instead of receiving a fabricated mapping.

`candidate_top_k` controls the retrieval/reranking pool. `candidate_output_k` controls the final submission and is enforced after every path, including deterministic fallback and Qwen failure. The default final output is one candidate.

Qwen may only select an ID present in the supplied candidate pool. Invalid JSON, unknown IDs, timeout, CUDA failure, or missing vLLM triggers deterministic fallback and is recorded in the run summary.

## Kaggle workflow and artifacts

The training notebook performs:

1. deterministic input and training-data discovery;
2. corpus and knowledge-base validation;
3. grouped train/validation split plus leakage audit;
4. XLM-R training with early stopping and best-checkpoint selection;
5. checkpoint reload smoke test;
6. retrieval-index construction;
7. NER, merge, assertion, linking, and optional Qwen inference;
8. schema, offset, file-count, ZIP structure, and CRC validation;
9. artifact packaging and run-manifest generation.

Required outputs under `/kaggle/working`:

- `output.zip` containing exactly `output/<document_id>.json`;
- `trained_ner_artifacts.zip` containing a reloadable checkpoint and training evidence;
- `training_artifacts/ner_model/model.safetensors`;
- `training_result.json`;
- `evaluation_report.json`;
- `split_manifest.json`;
- `run_manifest.json`;
- `diagnostics/run_summary.json`.

The notebook must fail early for invalid data or an unloadable checkpoint. Optional reranking failures must not invalidate otherwise valid deterministic output.

## Verification and acceptance

- Dataset validator reports zero pairing, UTF-8, schema, offset, assertion, and ontology-candidate errors.
- Leakage audit reports zero group intersections between train and validation.
- New behavior is developed test-first, including split isolation, candidate truncation, abstention, regex demotion, assertion scope, and Qwen fallback.
- All local unit/integration tests terminate without downloading large models.
- A CPU fast-dev fixture trains or uses a tiny local model fixture, reloads the checkpoint, runs inference, validates every output, and packages both ZIP files.
- The canonical notebook and generated notebook are byte-equivalent, all code cells parse, and notebook tests verify all required artifacts.
- Kaggle execution is not considered proven until the downloaded manifest, training metrics, checkpoint inventory, output inventory, and ZIP CRC are audited.

## Non-goals

- Training Qwen or making Qwen the primary NER model.
- Assigning ICD/RxNorm candidates to symptom or laboratory entities when the competition schema does not allow them.
- Making relation extraction a scored production dependency before NER, assertion, and entity-linking metrics are reliable.
