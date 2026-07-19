# Model Training Pipelines Implementation Plan

**Goal:** Complete Colab-T4-ready NER, BGE-M3 LoRA, and Qwen2.5-7B QLoRA
training pipelines on top of the validated Phase A data foundation.

**Architecture:** All trainers consume only fingerprinted `data/training`
artifacts, share strict run/artifact manifests and precision-first selection,
and lazy-load GPU libraries. Each model trains in a separate Colab runtime and
can resume from a Drive checkpoint. Holdout is rejected by every gradient-data
selector.

**Tech stack:** Python 3.10+, PyTorch, Transformers 4.57-compatible APIs, PEFT,
Sentence Transformers Trainer, datasets, bitsandbytes on Linux/Colab, pytest.

## Task 1: Shared run, metric, and artifact contracts

Create:

- `src/training/metrics.py`
- `src/training/artifacts.py`
- `tests/training/test_training_runtime.py`

Requirements:

- Exact precision/recall/F0.5 with configurable recall floor.
- Precision is the first tiebreaker inside score tolerance.
- Run config and dataset build manifest fingerprints are immutable.
- Resume must match task, base model, config hash, and dataset build ID.
- Final artifact promotion is atomic and status is one of candidate,
  validated, locked.

TDD gate:

```bash
python -m pytest tests/training/test_training_runtime.py -q
```

Commit: `feat: add precision-first training runtime`

## Task 2: NER BIO preparation and constrained decoding

Create:

- `src/training/ner/__init__.py`
- `src/training/ner/bio.py`
- `src/training/ner/data.py`
- `tests/training/test_ner_training.py`

Requirements:

- Five competition entity types and deterministic BIO label IDs.
- Fast-tokenizer offset alignment; special/padding labels are `-100`.
- Overflow chunks preserve absolute offsets and never train on a partial entity.
- Invalid `I-*` transitions are converted to `B-*`.
- Decoding returns exact document slices and deduplicates overlap windows.
- Dataset selectors expose only synthetic train/validation or trusted folds;
  pseudo and holdout are rejected.

TDD gate:

```bash
python -m pytest tests/training/test_ner_training.py -q
```

Commit: `feat: prepare and decode NER training data`

## Task 3: XLM-R trainer and model runtime

Create:

- `src/training/ner/config.py`
- `src/training/ner/train.py`
- `src/ner/model_extractor.py`
- `configs/training/ner_xlmr_base.yaml`
- `tests/training/test_ner_trainer.py`

Modify:

- `src/config.py`
- `src/pipeline/main.py`

Requirements:

- Strict config; default `xlm-roberta-base`, max length 384, stride 64.
- Stages: synthetic, trusted-fold, trusted-final.
- Transformers are imported lazily; `--dry-run` validates data/model paths
  without downloading or allocating a model.
- Exact span F0.5 compute callback, FP16 on CUDA, gradient accumulation,
  checkpoint resume, early stopping.
- Runtime modes rule/model/hybrid; missing or fingerprint-mismatched model
  fails closed to rule in hybrid mode.

TDD gate:

```bash
python -m pytest tests/training/test_ner_trainer.py -q
```

Commit: `feat: train and serve XLM-R clinical NER`

## Task 4: BGE-M3 examples, hard negatives, trainer, and index contract

Create:

- `src/training/embedding/__init__.py`
- `src/training/embedding/data.py`
- `src/training/embedding/config.py`
- `src/training/embedding/train.py`
- `src/training/embedding/index_manifest.py`
- `configs/training/embedding_bge_m3_lora.yaml`
- `tests/training/test_embedding_training.py`

Requirements:

- Resolve positive descriptions read-only from the correct DB namespace.
- Mine wrong BM25/semantic results only after split; never insert the gold code.
- Exclude holdout and retrieval misses from positive ranking metrics.
- SentenceTransformerTrainer with triplet records and PEFT LoRA.
- Recall@1/5/10, MRR@10, nDCG@10, downstream F0.5.
- Index manifest must match base model, adapter, DB, and embedding fingerprint.
- BM25-first alpha stays 0.75 by default and weights sum to one.

TDD gate:

```bash
python -m pytest tests/training/test_embedding_training.py -q
```

Commit: `feat: train BGE-M3 retrieval adapter`

## Task 5: Frozen candidates and Qwen QLoRA

Create:

- `src/training/reranker/__init__.py`
- `src/training/reranker/data.py`
- `src/training/reranker/config.py`
- `src/training/reranker/train.py`
- `src/training/reranker/inference.py`
- `configs/training/reranker_qwen25_7b_qlora.yaml`
- `tests/training/test_reranker_training.py`

Modify:

- `src/ranking/llm_reranker.py`
- `src/config.py`

Requirements:

- Candidate dataset requires a frozen retriever fingerprint.
- Gold outside top-k is a retrieval miss and is never injected into candidates.
- Prompt target is JSON only and selects zero to two in-pool codes.
- 4-bit NF4, FP16 compute, PEFT LoRA on all linear layers, micro-batch one.
- Completion-only labels mask prompt tokens with `-100`.
- Invalid JSON/out-of-pool rates are reported; out-of-pool must be zero.
- Local transformers backend is lazy and deterministic fallback remains.

TDD gate:

```bash
python -m pytest tests/training/test_reranker_training.py -q
```

Commit: `feat: train Qwen subset reranker`

## Task 6: Colab commands and final verification

Create:

- `docs/training/MODEL_TRAINING_COLAB.md`

Modify:

- `requirements-train.txt`
- `README.md`

Verification:

```bash
python -m pytest tests/training -q
python -m pytest -q
python src/metrics.py test
python scripts/audit_overrides.py --scan-paths src scripts
python scripts/audit_overrides.py \
  --db data/kb/metadata.db \
  --overrides src/resources/verified_overrides.json
python -m src.training.ner.train --help
python -m src.training.embedding.train --help
python -m src.training.reranker.train --help
git diff --check
```

Commit: `docs: document modular model training`
