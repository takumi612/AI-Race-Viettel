# Final fix report — 2026-07-26

## Scope

- Mapped required-Qwen assertion axes to the supported internal labels only:
  `NEGATED -> isNegated`, `HISTORICAL -> isHistorical`, and
  `FAMILY -> isFamily`. Neutral/certainty axes emit no submission label.
- Added an integration regression test that sends the refined entity through
  the artifact assertion mapping and real submission validator.
- Made the training notebook's single `ENABLE_QWEN_RERANKER` assignment guard
  a checked `vllm==0.25.1` installation with `--no-deps`, cache invalidation,
  and a hard import-resolution failure. Regenerated the notebook.

## TDD evidence

Before the production changes, the new and updated focused tests failed with:

- Qwen assertions emitted `polarity:NEGATED` rather than `isNegated`; the real
  submission path rejected the resulting serialized `"None"` assertion.
- The generated notebook had no guarded vLLM installation branch.

After the production changes:

- `python -m pytest tests/test_qwen_refiner_required.py tests/test_kaggle_observability.py -q -p no:cacheprovider`
  — 17 passed.
- `python -m pytest tests/test_qwen_refiner_required.py tests/test_required_qwen_phase.py tests/test_single_toggle_qwen.py tests/test_kaggle_observability.py tests/test_notebook_orchestrator_contract.py -q -p no:cacheprovider`
  — 31 passed.
- `python -m pytest tests -q -p no:cacheprovider` — 398 passed, 2 skipped.
- Notebook regeneration validation — 29 cells, 15 code cells, 13 phases;
  syntax validation passed.
- `git diff --check` — passed.

## Concerns

No code-level concerns. The repository had unrelated existing deletion and
untracked files; they were not changed or included in the fix commit.
