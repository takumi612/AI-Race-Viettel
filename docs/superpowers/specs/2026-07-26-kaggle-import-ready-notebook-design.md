# Kaggle Import-Ready Notebook Design

## Goal

Produce two clean Kaggle notebooks that can be imported and run from a stopped
session without inheriting stale runtime state. One notebook must run the full
pipeline with Qwen enabled on an explicitly compatible CUDA stack; the other is
a stable no-Qwen fallback.

## Source and output

- Generate from the canonical end-to-end notebook builder.
- Keep the canonical generated notebook synchronized.
- Publish two user-facing files at the repository root:
  - `ai-race-training-v2-enableQwen.ipynb`
  - `ai-race-training-v2-unableQwen.ipynb`
- Preserve the existing error notebook for diagnosis and comparison.

## Runtime behavior

- Default to `RUN_MODE="full"` so a new Kaggle session starts from phase 1.
- Set `ENABLE_QWEN_RERANKER=True` only in the `enableQwen` notebook.
- Install an explicit CUDA 12.9 vLLM wheel and its aligned PyTorch dependency
  stack before importing torch in the `enableQwen` notebook. Do not install the
  CUDA 13 PyPI wheel with `--no-deps`.
- Import-check vLLM during bootstrap in the `enableQwen` notebook and fail with
  an actionable compatibility message before training if the runtime is not
  usable.
- Set `ENABLE_QWEN_RERANKER=False` in the `unableQwen` notebook and do not
  install or import vLLM in that variant.
- Preserve the complete training, inference, quality-gate, and packaging phases.
- Retain recursive discovery of the uploaded `synthetic_train_v2` training data
  and the separate inference `input` directory.

## Notebook hygiene

- Remove all execution counts, stream output, tracebacks, and stale cell state.
- Keep standard notebook metadata and a Python 3 kernelspec.
- Include a concise pre-run note describing the required Kaggle dataset layout,
  GPU/Internet requirements, and the selected Qwen runtime contract.

## Verification

- Add or update source-level notebook contract tests before changing the builder.
- Assert that the enabled variant carries the single Qwen toggle, explicit CUDA
  12.9 installation contract, and early vLLM import check.
- Assert that the unable-Qwen variant contains neither vLLM installation nor a
  reachable vLLM import.
- Regenerate the notebook deterministically.
- Validate notebook JSON and compile every code cell as Python after removing
  notebook-only command syntax where applicable.
- Run focused Kaggle notebook, Qwen-toggle, and orchestration contract tests.
- Confirm both user-facing notebooks contain no saved outputs or execution
  counts and differ only where their documented Qwen runtime contracts require.
