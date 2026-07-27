# Precision-First NER Boundary and Qwen Validation Design

## 1. Objective

Improve the Kaggle clinical information extraction pipeline's hidden-set score by
reducing false-positive entities and overlong entity boundaries without breaking
raw-text offsets, exact KB matches, candidate linking, assertions, or the required
`output.zip` structure.

The primary optimization target is entity extraction quality because the latest
submission has `WER=85.0468`, while candidate and assertion quality can only be
scored meaningfully after the entity span and type are correct.

## 2. Evidence and Root Cause

The latest run produced 3,191 entities across 100 documents with no offset or ZIP
structure errors. Forty predicted spans are longer than 100 characters, or 1.25%
of all output entities.

The 2,200-document training corpus contains 36,588 entities:

| Source bucket | Documents | Entities | More than 50 chars | More than 100 chars | Maximum chars |
|---|---:|---:|---:|---:|---:|
| Reconstructed | 100 | 7,224 | 624 | 22 | 153 |
| Organizer GT | 100 | 7,224 | 637 | 32 | 162 |
| Synthetic | 2,000 | 22,140 | 1,729 | 0 | 99 |

Long ground-truth spans are therefore real, but spans above 100 characters come
from the reconstructed and organizer buckets rather than the 2,000 synthetic
documents. They are usually complete ICD descriptions in explicit diagnostic
templates. The hidden inference documents instead contain questions, explanations,
risk factors, and free-form prose. The model transfers the learned long-diagnosis
pattern to clauses that are not entity mentions.

There is also a document-level decoding defect. `merge_chunk_predictions()`
currently unions overlapping same-type predictions with `min(start)` and
`max(end)`, then assigns the maximum confidence of the contributors. Conflicting
window boundaries can therefore create a span wider than either reliable boundary.
NER calibration is computed per owner window and does not measure this post-merge
behavior, so the validation score can remain near 1.0 while final document spans
are poor.

## 3. Design Principles

1. Do not impose a global character limit. Valid ICD descriptions can exceed 100
   characters.
2. Preserve exact raw offsets. Every emitted entity text must equal
   `raw_text[start:end]`.
3. Treat exact KB evidence as stronger than model-only evidence.
4. Use Qwen to validate uncertain model proposals, not to invent unconstrained
   offsets.
5. Measure the same document-level representation that is submitted.
6. Never train or fit on the competition inference input.
7. Fail explicitly when required Qwen output is malformed; do not silently publish
   an unvalidated submission under the Qwen-enabled notebook.

## 4. Proposed Inference Architecture

The production inference order will be:

1. Transformer NER emits raw span proposals with confidence and owner-window
   evidence.
2. Exact KB scanning emits immutable high-precision disease and drug proposals.
3. A boundary-consensus merger resolves duplicate and conflicting window proposals.
4. The Qwen entity validator reviews model-derived entities in raw-text context.
5. Candidate retrieval and policy run on the validated text, type, and offsets.
6. The deterministic assertion head runs on validated entities.
7. Qwen candidate reranking and assertion refinement run on the validated entities.
8. Submission validation, diagnostics, and packaging run unchanged at the contract
   boundary.

This ordering ensures that candidate and assertion metadata never remain attached
to an entity whose text, type, or position was changed by validation.

## 5. Boundary-Consensus Merge

The merger must never widen conflicting predictions merely because they overlap.
It will group proposals by document and entity type, then apply these rules:

- Identical boundaries are deduplicated and their independent evidence is retained.
- Near-identical boundaries are treated as agreement only when one boundary differs
  by at most one tokenizer edge and neither candidate crosses a clause or sentence
  delimiter absent from the other candidate.
- For agreeing predictions, select one observed boundary; do not synthesize a union
  boundary.
- For conflicting overlapping predictions, select the proposal with the strongest
  evidence ordering: exact KB match, independent window agreement, calibrated span
  confidence, then shorter boundary as the deterministic tie-breaker.
- Different entity types do not merge. Existing proposal arbitration decides which
  overlapping type survives using evidence and confidence.

The resulting entity must always be one of the observed raw-text boundaries. This
eliminates union-created spans and makes the behavior testable without a model.

## 6. Qwen Entity Validator

### 6.1 Scope

Exact KB spans with score 1.0 bypass entity validation and remain immutable.
Transformer-derived entities are sent to Qwen in bounded surrounding context.
Validation covers all official entity types rather than using length alone because
short false positives also contribute to WER.

### 6.2 Response Contract

For each input entity, Qwen must return exactly one structured decision:

- `keep`: preserve text, type, and position.
- `drop`: remove the proposal.
- `trim`: return a relative start and end fully contained within the original span,
  plus an allowed entity type.

Qwen may not extend a span, create a new entity, return free-form offsets, or emit an
unknown type. A `trim` decision is accepted only when:

- `0 <= relative_start < relative_end <= len(original_text)`;
- the proposed substring is non-whitespace and contains an alphanumeric character;
- converting the relative range to raw offsets reproduces the exact substring;
- the resulting type belongs to the configured internal entity type set.

Any response-count mismatch, malformed JSON, invalid action, out-of-range boundary,
or offset mismatch raises `RequiredQwenError` in the Qwen-enabled notebook.

### 6.3 Prompt Requirements

The prompt will distinguish a clinical entity mention from surrounding explanation,
causality, risk factors, treatment advice, and historical narrative. It will ask for
the smallest self-contained mention supported by the text. It will include concise
positive and negative Vietnamese examples, including a long valid ICD phrase and a
long invalid explanatory clause.

## 7. Training and Validation Changes

Window-level BIO metrics remain useful for training diagnostics but no longer serve
as the sole model-selection evidence.

The pipeline will add a fixed, provenance-bound natural-language validation slice
from organizer GT. Documents `181` through `200`, inclusive, form this slice and
are excluded from all training and final-fit phases. On the current dataset this is
20 documents containing 1,488 entities, all five official entity types, 120 spans
longer than 50 characters, and five spans longer than 100 characters. The validation
manifest binds these IDs to dataset fingerprint
`18a391e51786630b482bb500d5129eb102ae144450d7fc18b149f2799054f028`.
Preflight fails rather than silently selecting a different benchmark when the
fingerprint or required document inventory changes.

Validation will report:

- exact typed entity precision, recall, and F1 after document-level merge;
- type-matched overlap precision, recall, and F1;
- results by provenance bucket and entity type;
- results for spans of 1-50, 51-100, and more than 100 characters;
- count of merged entities whose boundary was not present in any input proposal,
  which must be zero;
- calibrated threshold and the metric used to select it.

Qwen validation quality is measured separately using deterministic fixture cases in
the test suite. The competition inference input is never labeled, trained on, or
used to fit thresholds.

## 8. Notebook and Observability Changes

The canonical Kaggle notebook remains a thin orchestrator. Business logic stays in
`clinical_nlp_lab`.

Phase 12 will report these additional fields:

- entity count before and after boundary consensus;
- Qwen entity validation query count;
- counts for `keep`, `drop`, and `trim`;
- counts by entity type before and after validation;
- span-length buckets before and after validation;
- maximum span length before and after validation;
- immutable exact-KB bypass count;
- Qwen entity validation status.

Diagnostics for each document will preserve the original proposal, decision,
validated entity, and evidence code without recording model internals or sensitive
runtime errors. Phase 12 must not package output unless all validated offsets pass.

The notebook will continue to expose one `ENABLE_QWEN_RERANKER` toggle. When true,
entity validation, candidate reranking, and assertion refinement are all required.
When false, all three Qwen operations are disabled and the deterministic boundary
merger remains active.

## 9. Error Handling

- Missing final checkpoint, fitted heads, KB artifacts, or validation manifest is a
  hard failure before inference.
- Invalid Qwen entity output is a hard Phase 12 failure in Qwen-enabled mode.
- Qwen cleanup errors remain chained to the original inference error.
- A proposed trim that violates raw offsets is rejected rather than repaired.
- Empty post-validation documents are valid submission files containing `[]`.
- ZIP construction still requires exactly one `output/<id>.json` member per input
  document, numeric ordering, valid CRC, and no extra members.

## 10. Testing Strategy

Implementation follows test-driven development.

Unit tests will cover:

- identical and near-identical window proposals;
- conflicting proposals that previously produced a union boundary;
- evidence ordering and deterministic tie-breaking;
- long valid KB entities that must survive;
- Qwen `keep`, `drop`, and valid `trim` decisions;
- invalid relative offsets, types, actions, JSON, and response counts;
- relinking and assertion execution after a trim or type change;
- Qwen-disabled deterministic behavior;
- Phase 12 counters and strict failure behavior.

Integration tests will cover one multi-window document through inference and one
small synthetic Kaggle phase through packaging. Existing offset, ZIP, Qwen-required,
candidate, assertion, and notebook contract tests must continue to pass.

## 11. Acceptance Criteria

The change is complete when all of the following hold:

1. No document-level entity boundary is wider than every contributing proposal.
2. Every submitted entity passes exact raw-offset validation.
3. Exact KB matches are not dropped or trimmed by Qwen validation.
4. Candidate linking and assertions operate on the final validated entity.
5. Natural-language validation reports post-merge exact and overlap metrics.
6. Post-merge exact precision improves over the current merger on the fixed
   organizer validation slice, while exact F1 does not decrease.
7. The Qwen validator passes deterministic contract tests for all decisions and
   failure modes.
8. Phase 12 exposes complete before/after and Qwen decision diagnostics.
9. The full test suite passes.
10. A Kaggle Run All produces a structurally valid 100-record `output.zip` with zero
    offset errors and a completed Qwen entity-validation status.

Hidden leaderboard improvement is the intended outcome but cannot be asserted
locally because hidden ground truth is unavailable. The new local acceptance gates
are designed to prevent the misleading near-perfect window metric that accompanied
the previous low-scoring submissions.

## 12. Out of Scope

- Creating labels from the competition inference input.
- Replacing the Transformer NER model with full-document generative extraction.
- Changing the official submission schema or scoring implementation.
- Hard-coding per-document corrections for the current 100 input files.
- Regenerating the entire 2,000-document synthetic corpus in this change.
