# Centralized Entity Span Validation

## 1. Problem

Phase 12 has failed at the final output-quality gate for two different consequences
of the same architectural defect:

- `boundary_errors=606`: Qwen trims could split an alphanumeric run.
- `punctuation_only=1`: Qwen could trim an otherwise valid entity down to a
  punctuation-only substring such as `"-"`.

Qwen does not invent the punctuation character. It returns relative offsets into
the original entity, and the current implementation accepts the selected raw-text
slice when its offsets match and it does not split a word.

The pipeline currently implements the definition of a valid entity span separately
in:

- proposal filtering in `inference.py`;
- trim application in `qwen_entity_validator.py`;
- final diagnostics in `output_quality.py`;
- basic offset checking in `schema.py`.

These implementations enforce different subsets of the contract. Consequently, an
invalid transformation can survive inference and fail only after the expensive
Qwen pass has completed.

## 2. Objective

Create one dependency-light, reusable validation policy for entity surfaces and
raw-text boundaries. Every transformation that emits an entity must use the same
policy, while the final output-quality gate remains an independent enforcement
boundary over the same validation results.

This change must prevent punctuation-only, whitespace-only, multiline, offset
mismatch, out-of-range, word-splitting, and overlong output entities without
blanket-rejecting medically meaningful punctuation.

## 3. Validation Contract

Add a shared module, `clinical_nlp_lab/entity_span_policy.py`, which accepts
primitive values (`raw_text`, `start`, `end`, and `text`) rather than importing
`EntityAnnotation`. This avoids a circular dependency with `schema.py`.

The module returns structured violation codes rather than a single boolean:

- `invalid_range`: `0 <= start < end <= len(raw_text)` is false;
- `offset_mismatch`: `raw_text[start:end] != text`;
- `whitespace_only`: the surface contains no non-whitespace character;
- `punctuation_only`: the surface contains no alphanumeric character;
- `multiline`: the surface contains `\r` or `\n`;
- `left_word_split`: the start boundary splits an alphanumeric run;
- `right_word_split`: the end boundary splits an alphanumeric run;
- `too_long`: the span exceeds the configured output limit.

The punctuation rule is positive rather than a blacklist: a valid surface must
contain at least one Unicode alphanumeric character. Special characters remain
valid inside clinical mentions such as `COVID-19`, `38.5°C`, `SpO2 95%`,
`HBsAg (+)`, and `T3/T4`.

Suspicious generic surfaces are a separate semantic policy. Move the existing
surface set out of the private `output_quality.py` constant into the shared module
and expose a normalized helper. This removes the current dependency inversion in
which the Qwen validator imports a private constant from the final quality gate.

## 4. Enforcement Points

### 4.1 Proposal merge

Replace `_is_valid_proposal_boundary()`'s duplicated checks with the shared
validator. Invalid model or KB proposals are filtered before becoming
`EntityAnnotation` objects.

### 4.2 Qwen trim

After converting Qwen relative offsets to an absolute proposed span, run the shared
validator before constructing the replacement entity.

If the proposed trim has any hard violation or becomes a suspicious generic
surface:

- preserve the original entity;
- do not attach metadata based on the rejected text/type;
- record a fallback counter grouped by violation reason.

The original entity is itself checked with the shared hard policy before Qwen is
called. An invalid original entity is a pipeline contract violation and must fail
locally instead of being silently preserved.

Explicit Qwen `drop` decisions continue to drop the entity. Malformed JSON,
unsupported types, invalid relative ranges, and response-count mismatches remain
hard Qwen contract errors.

### 4.3 Submission schema boundary

Keep `validate_submission_payload()` limited to the official schema, types, keys,
and exact offsets. The preflight reuses this function for organizer ground truth,
which legitimately contains multiline and 162-character entities. Output-only
quality rules must therefore remain in inference mutation boundaries and the final
output audit rather than changing the shared ground-truth schema contract.

### 4.4 Output quality

Refactor `audit_output_documents()` to aggregate the shared violation codes instead
of reimplementing punctuation, multiline, word-boundary, and length checks.
Preserve the existing report keys and thresholds for compatibility with Kaggle
diagnostics.

The final gate is not relaxed or removed. It remains the defense-in-depth check
that no producer bypassed the shared contract.

## 5. Diagnostics

Extend `EntityValidationCounters` with a `trim_fallback_reasons` counter. For a
rejected trim, increment one entry for every applicable violation code, plus a
separate `suspicious_generic` reason where relevant.

This counter must be included in `to_dict()`, `copy()`, `add()`, and `delta()` so
the Phase 12 diagnostics identify why Qwen proposals were rejected without
inspecting every model response.

The existing `trim` count continues to represent Qwen trim decisions. The fallback
counter distinguishes accepted trims from trim decisions that safely preserved the
original entity.

## 6. Testing

Use table-driven contract tests covering:

- valid Vietnamese text;
- valid medical punctuation (`COVID-19`, `38.5°C`, `HBsAg (+)`, `T3/T4`);
- punctuation-only surfaces (`-`, `...`, `/()`, `_`);
- whitespace-only text;
- multiline text;
- invalid and empty ranges;
- raw-text mismatch;
- left and right word splits;
- configured maximum length.

Add integration regressions proving that:

1. proposal filtering and output auditing classify the same fixtures consistently;
2. a Qwen trim to `"-"` preserves the original entity and records
   `punctuation_only`;
3. boundary-invalid and generic trims use the same fallback path;
4. valid one-sided and two-sided trims still succeed;
5. submission schema validation continues to accept valid organizer multiline and
   over-160-character ground truth;
6. the complete output-quality report retains its existing public keys.

Run the focused entity-policy, Qwen-validator, inference, schema, and output-quality
tests, followed by the full `v2` test suite.

## 7. Non-goals

- Do not blacklist all special characters.
- Do not change the Qwen model, GPU configuration, or vLLM attention backend.
- Do not weaken quality thresholds to make Phase 12 pass.
- Do not retrain NER or modify the dataset.
- Do not automatically delete an original entity merely because Qwen proposed an
  invalid trim.

## 8. Expected Result

An invalid Qwen substring is rejected immediately at the transformation boundary,
the original valid entity is retained, and Phase 12 diagnostics explain the
fallback. Because proposal creation, Qwen mutation, submission validation, and the
final audit share one contract, adding a new hard entity invariant requires one
policy change and its contract tests rather than independent patches throughout
the pipeline.
