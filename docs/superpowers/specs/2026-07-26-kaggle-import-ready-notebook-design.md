# Kaggle Import-Ready Notebook Design

## Goal

Produce a clean Kaggle notebook that can be imported and run from a stopped
session without inheriting stale runtime state or the incompatible vLLM CUDA 13
dependency.

## Source and output

- Generate from the canonical end-to-end notebook builder.
- Keep the canonical generated notebook synchronized.
- Publish a separate user-facing file named
  `ai-race-training-v2-kaggle-fixed.ipynb` at the repository root.
- Preserve the existing error notebook for diagnosis and comparison.

## Runtime behavior

- Default to `RUN_MODE="full"` so a new Kaggle session starts from phase 1.
- Default `ENABLE_QWEN_RERANKER=False`.
- When Qwen is disabled, do not install or import vLLM.
- Preserve the complete training, inference, quality-gate, and packaging phases.
- Retain recursive discovery of the uploaded `synthetic_train_v2` training data
  and the separate inference `input` directory.

## Notebook hygiene

- Remove all execution counts, stream output, tracebacks, and stale cell state.
- Keep standard notebook metadata and a Python 3 kernelspec.
- Include a concise pre-run note describing the required Kaggle dataset layout,
  GPU/Internet requirements, and the disabled-Qwen compatibility choice.

## Verification

- Add or update source-level notebook contract tests before changing the builder.
- Regenerate the notebook deterministically.
- Validate notebook JSON and compile every code cell as Python after removing
  notebook-only command syntax where applicable.
- Run focused Kaggle notebook, Qwen-toggle, and orchestration contract tests.
- Confirm the user-facing copy is byte-identical to the verified canonical
  notebook and contains no saved outputs or execution counts.
