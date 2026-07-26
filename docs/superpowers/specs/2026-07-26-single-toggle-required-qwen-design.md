# Single-Toggle Required Qwen Design

## Status and scope

This design applies to the latest end-to-end Kaggle path on
`codex/kaggle-end-to-end-pipeline`, whose canonical entry point is
`v2/medical_information_extraction_kaggle.ipynb`.

It does not switch inference back to the legacy `run_inference` path and does
not change the inference-only notebook. It supersedes earlier end-to-end
design clauses that described Qwen failure as an optional deterministic
fallback. Under this design, Qwen remains optional only while disabled. Once
requested, it is a required phase.

## Goal

Make the single notebook boolean `ENABLE_QWEN_RERANKER` the only user-facing
source of truth for whether Qwen is used. When the value is `True`, the
end-to-end run must initialize and execute Qwen reranking and assertion
refinement or fail the run with a clear error. When the value is `False`, the
existing deterministic inference path must run without importing or loading
vLLM.

## Configuration contract

The first notebook configuration cell owns the only enable switch:

```python
ENABLE_QWEN_RERANKER = False
```

The notebook passes this value directly into:

```python
RunConfig(enable_qwen_reranker=ENABLE_QWEN_RERANKER)
```

`RunConfig.enable_qwen_reranker` is transport state, not a second source of
configuration. Phase 12 must consume that field directly. It must not read
`enable_qwen`, `enable_qwen_reranker`, or any equivalent boolean from
`config.json`, environment variables, `KAGGLE_OPTIONS`, model artifacts, or a
second notebook cell.

The existing model/runtime parameters remain implementation settings:

- model: `Qwen/Qwen2.5-7B-Instruct-AWQ`;
- GPU memory utilization: `0.50`;
- maximum model length: `4096`;
- batch size: `64`.

They do not independently enable Qwen.

## Runtime architecture

The required data flow is:

```text
ENABLE_QWEN_RERANKER
    -> RunConfig.enable_qwen_reranker
    -> phase_12_inference
    -> RequiredQwenRefiner
    -> FinalModelBundle.qwen_reranker
    -> infer_document
```

Phase 12 owns the Qwen lifecycle. It must:

1. inspect `RunConfig.enable_qwen_reranker`;
2. leave `FinalModelBundle.qwen_reranker=None` when disabled;
3. construct the Qwen engine and required refiner before document inference
   when enabled;
4. inject the refiner into `load_final_model_bundle`;
5. run inference with `InferenceConfig.enable_qwen` set to the same
   `RunConfig` value;
6. destroy the Qwen engine in a `finally` block.

No broad exception handler may convert a required-Qwen error into a
deterministic success.

## Candidate-pool preservation

The current bundle path reduces ranked candidate dictionaries to final
candidate IDs before Qwen executes. That makes genuine reranking impossible.

The inference contract must preserve a runtime-only ranked candidate pool for
each merged disease or drug entity:

- each pool entry contains the candidate ID, display name, system, and score;
- the pool is capped by the existing candidate top-k policy;
- the pool is never serialized as an extra submission key;
- the selected output still obeys `candidate_output_k=1`;
- Qwen may select only an ID present in that entity's preserved pool;
- a valid `selected_id: null` means explicit abstention;
- an unknown ID is a required-Qwen error.

The runtime-only pool may be represented by a dedicated transient field on
`EntityAnnotation` or an equivalent sidecar keyed by document ID, position,
and entity type. The implementation must not hide the pool inside the
official `candidates` submission field.

## Required Qwen adapter

The existing `ClinicalLLMReranker` exposes `rerank_batch`, while
`FinalModelBundle` expects a `refine(entities, raw_text)` interface. A focused
adapter must bridge these APIs rather than changing the end-to-end path back
to the legacy pipeline.

The adapter must:

1. build batched reranking requests from entity context and preserved
   candidate pools;
2. call `ClinicalLLMReranker.rerank_batch`;
3. replace each eligible entity's final candidate with the selected in-pool
   ID or an empty list for a valid abstention;
4. build assertion-refinement requests for assertion-eligible entity types;
5. call `ClinicalLLMAssertionPredictor.predict_batch`;
6. map returned axes to the existing internal assertion labels expected by
   submission serialization;
7. return entities with unchanged text, type, and raw offsets.

The adapter must reject result-count mismatches, malformed responses, unknown
candidate IDs, invalid assertion enums, and offset changes.

## Failure semantics

`ENABLE_QWEN_RERANKER=False`:

- vLLM and Qwen modules are not imported or initialized;
- deterministic NER, assertion, candidate policy, and packaging behavior stay
  unchanged;
- the run summary reports `qwen_requested=false`.

`ENABLE_QWEN_RERANKER=True`:

- Qwen initialization and execution are mandatory;
- missing vLLM, missing or unloadable model weights, unsupported
  quantization, CUDA OOM, initialization failure, generation failure,
  timeout, malformed JSON, unknown candidate ID, assertion parse failure,
  result-count mismatch, and cleanup failure all fail phase 12;
- phase 13 packaging and final `PASS` status are not reached;
- the raised error identifies the Qwen stage and original exception type;
- partially generated output is not presented as a successful submission.

A valid Qwen abstention is not an error. It produces no candidate for that
entity while preserving all other valid entity fields.

## Observability

Phase 12 and `diagnostics/run_summary.json` must expose:

- `qwen_requested`;
- `qwen_initialized`;
- `qwen_model_name`;
- `qwen_rerank_query_count`;
- `qwen_assertion_query_count`;
- `qwen_abstention_count`;
- `qwen_status`, limited to `DISABLED`, `COMPLETED`, or `FAILED`.

When Qwen is enabled, a successful phase requires:

```text
qwen_requested = true
qwen_initialized = true
qwen_status = COMPLETED
```

Failure diagnostics must not contain access tokens, Hugging Face credentials,
raw environment values, or full model responses.

## Notebook generation

`v2/tools/build_kaggle_notebook.py` remains the source generator for
`v2/medical_information_extraction_kaggle.ipynb`.

The generated notebook must:

- expose exactly one `ENABLE_QWEN_RERANKER` assignment;
- set vLLM installation from the same boolean;
- pass the boolean exactly once into `RunConfig`;
- contain no compatibility comment that merely mentions an unused
  `enable_qwen_reranker` call;
- contain no `KAGGLE_OPTIONS` enable value;
- regenerate deterministically from the checked-in builder.

## Testing strategy

All production changes are developed test-first.

Unit tests must prove:

1. `RunConfig` defaults Qwen to disabled.
2. The generated notebook has exactly one enable assignment and passes it to
   `RunConfig`.
3. Phase 12 does not import or instantiate Qwen when disabled.
4. Phase 12 constructs and injects the refiner when enabled.
5. `config.json` cannot override the notebook/`RunConfig` boolean.
6. Ranked candidate pools survive merge without appearing as submission
   keys.
7. Qwen can select only a supplied candidate.
8. Valid null selection abstains.
9. Invalid JSON, unknown IDs, count mismatches, initialization errors,
   generation errors, assertion errors, and cleanup errors propagate as
   phase failures.
10. Qwen does not alter entity text, type, or offsets.
11. Qwen is destroyed on both success and failure.
12. A required-Qwen failure prevents phase 13 and final `PASS`.

Tests must use fake lightweight engines and must not download Qwen weights or
require a GPU.

## Acceptance criteria

- Changing only `ENABLE_QWEN_RERANKER=False` to `True` is sufficient to
  request Qwen in the canonical notebook.
- There is no second enable boolean in runtime artifacts or inference code.
- Disabled behavior remains deterministic and passes the existing test suite.
- Enabled behavior demonstrably initializes the Qwen adapter and runs at
  least one applicable reranking or assertion query when applicable entities
  exist.
- Any required-Qwen failure stops the run and prevents a successful package.
- Successful output remains schema-valid with exact raw-text offsets and at
  most one candidate per eligible entity.
- The generated notebook and its builder agree on the single-toggle contract.

## Non-goals

- Training or fine-tuning Qwen.
- Using Qwen as the primary NER model.
- Returning candidates outside the deterministic retrieval pool.
- Making the inference-only notebook use the end-to-end phase orchestrator.
- Fixing the independent NER metric, data leakage, oversampling, or candidate
  retrieval-quality issues identified by the broader pipeline audit.
