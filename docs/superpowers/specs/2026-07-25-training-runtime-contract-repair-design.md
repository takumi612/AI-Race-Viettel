# Training–Runtime Contract Repair Design

## Problem

The Kaggle training run publishes structurally valid output whose semantic
quality has collapsed: predictions cover most of each document, LAB_NAME
dominates, SYMPTOM disappears, short KB aliases match inside words, and
assertion labels are assigned with incompatible train/runtime entity-type IDs.

The training dataset itself passes the current provenance, offset, overlap,
ordering, type-distribution, and assertion-distribution audits. The repair must
therefore preserve `data_v2/Training_data/synthetic_train_v2` and correct the
code that consumes it.

## Contract

### Windows

Training and inference must use the tokenizer's native overflow mechanism with
the same `max_length` and `stride`. Every window must contain the tokenizer's
normal special-token layout. Raw offsets remain absolute document offsets.
Each gold entity is owned by exactly one containing window; tokens belonging
to an entity in another overlapping window are masked from NER loss.

`TokenWindow` carries explicit owned entity metadata. Assertion collation must
never infer an entity's semantic type from the numeric position of a BIO label.

### Entity types and assertions

One canonical mapping is shared by training, artifact publication, and runtime:

```text
DISEASE=0, DRUG=1, SYMPTOM=2, LAB_NAME=3, LAB_RESULT=4
```

Only DISEASE, DRUG, and SYMPTOM are assertion-bearing types. Assertion examples
are split by document into train and validation sets. Thresholds are calibrated
only from final validation logits with a numerically stable sigmoid.

### Curriculum

Every training subprocess receives the selected `StageSpec` values. Stage 1,
stage 2, stage 3, and final-fit must use their declared epoch and learning-rate
values. Stage sampling must honor organizer and replay fractions instead of
giving stages 2 and 3 identical selections. Final-fit remains bounded to its
declared two epochs and must not silently fall back to the global 20-epoch
configuration.

### Inference

KB exact recovery requires normalized token boundaries. Short aliases such as
`ho` may match only as standalone tokens, never inside `hoặc`, `khoa`, or
similar words.

NER chunk proposals are boundary-refined before cross-source merge. Invalid
fragments (punctuation-only, mid-word boundaries, implausibly long multiline
spans) are rejected. Overlap resolution prioritizes higher-confidence,
well-bounded spans instead of blindly preferring the longest span at a start
offset.

### Quality gates and observability

Before publishing output, diagnostics must include entity count, type counts,
type shares, document coverage, mid-word boundaries, punctuation-only spans,
multiline spans, and maximum span length. Publishing fails on pathological
collapse, including excessive mean coverage or a single predicted type
dominating the complete run.

The Kaggle notebook phase result must expose actual per-stage train/validation
window counts, epochs, learning rate, train loss, evaluation metric, and best
checkpoint. The phase-06 sample window count remains explicitly labeled as a
sample diagnostic.

## Data policy

Do not rewrite the 2,200 training inputs or annotations. Add contract audits
that fail before training if offsets, ordering, overlap, supported types,
assertion scope, required type presence, or extreme class imbalance violate the
training contract. Dataset regeneration is a separate operation and is not
part of this repair.

## Verification

Regression tests cover:

- native-overflow window special tokens and absolute offsets;
- one-owner behavior across overlapping windows;
- canonical entity-type IDs and assertion masks;
- stage-specific epoch, learning-rate, and selection behavior;
- held-out assertion calibration and stable sigmoid;
- word-boundary KB recovery;
- inference boundary refinement and quality-gate rejection;
- dataset audit acceptance for the current `synthetic_train_v2`;
- notebook generation and existing orchestration contracts.

The full `v2/tests` suite must pass before notebook regeneration and again after
all generated artifacts are updated.
