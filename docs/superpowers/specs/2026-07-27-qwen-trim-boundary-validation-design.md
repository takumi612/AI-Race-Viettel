# Qwen Trim Boundary Validation

## Problem

Phase 12 rejects Qwen-enabled inference with `boundary_errors=606`.
NER and KB proposals enter Qwen with valid raw-text and token boundaries, but
the Qwen entity-validation schema currently requires every `trim` decision to
move both the start and end offsets strictly inside the original entity. This
forbids common one-sided trims and can force character-level cuts inside words.
The trim application verifies only raw-text slice equality, so invalid word
boundaries survive until the final output-quality gate.

## Intended behavior

- Permit left-only, right-only, and two-sided trims.
- Require a `trim` to make the entity strictly shorter than the original.
- Accept a trimmed span only when both new boundaries do not split an
  alphanumeric run.
- When Qwen returns a character-aligned but word-boundary-invalid trim, preserve
  the original entity. The original has already passed proposal boundary
  validation, so this fallback does not introduce a new offset error or discard
  an otherwise valid entity.
- Keep the final output-quality gate unchanged.

## Implementation

Update `clinical_nlp_lab/qwen_entity_validator.py`:

1. Relax the trim schema to allow `relative_start=0` and
   `relative_end=len(entity.text)`.
2. Reject a no-op trim where both original boundaries are retained.
3. Validate the absolute start and end using the same alphanumeric-boundary
   rule as inference/output-quality validation.
4. Return the original entity for a boundary-invalid trim.

No training, model, dataset, or quality-threshold behavior changes.

## Tests

Update `tests/test_qwen_entity_validator.py` with regression coverage for:

- right-only trim;
- left-only trim;
- valid two-sided trim;
- no-op trim rejection;
- left boundary split fallback;
- right boundary split fallback.

Run the focused Qwen validator and output-quality tests, followed by the full
test suite. The original Kaggle training phases do not need to be rerun; after
the updated branch is available, Phase 12 can be resumed from existing
checkpoints.
