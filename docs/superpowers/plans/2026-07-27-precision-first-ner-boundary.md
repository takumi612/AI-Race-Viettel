# Precision-First NER Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent union-created and semantically overlong NER spans, validate model-only entities with Qwen before linking/assertion, and expose trustworthy document-level validation and Kaggle diagnostics.

**Architecture:** Transformer NER remains the proposal generator and exact KB recovery remains immutable. A consensus merger chooses only observed boundaries, a strict Qwen validator may keep/drop/trim model proposals, and candidate/assertion metadata is calculated only after entity validation. Natural validation documents 181–200 are bound by dataset fingerprint and evaluated after document-level merge.

**Tech Stack:** Python 3.10+, pytest, PyTorch/Transformers, vLLM 0.25.1, Qwen2.5-7B-Instruct-AWQ, Jupyter/Kaggle.

**Local pytest prerequisite:** when running from the repository root, first set
`$env:PYTHONPATH = (Join-Path (Get-Location) 'v2')`. The package is source-rooted
under `v2/` and root pytest collection otherwise cannot import `clinical_nlp_lab`.

## Global Constraints

- Do not train, fit, calibrate, or create labels from the competition inference input.
- Do not impose a global character-length cap on entities.
- Every emitted entity must satisfy `entity.text == raw_text[entity.start:entity.end]`.
- Exact KB proposals with score `1.0` bypass Qwen entity validation and remain immutable.
- Qwen may keep, drop, or trim inside the original span; it may not extend a span or invent an entity.
- Invalid Qwen JSON, action, type, response count, or relative offset must become `RequiredQwenError` in Qwen-enabled mode.
- Candidate retrieval and assertion prediction must use the final validated text, type, and position.
- All new automated tests are CPU-only unit tests; do not add component or integration tests.
- Kaggle Run All is performed by the user after handoff; runtime errors are handled from the returned cell output and stack trace.

## File Structure

- `v2/clinical_nlp_lab/ner.py`: raw window prediction evidence and consensus boundary selection.
- `v2/clinical_nlp_lab/qwen_entity_validator.py`: Qwen decision schema, parsing, prompt construction, offset-safe application, and pure counters.
- `v2/clinical_nlp_lab/qwen_refiner.py`: required-Qwen facade for entity validation followed by existing candidate/assertion refinement.
- `v2/clinical_nlp_lab/inference.py`: production ordering: propose, merge, validate, relink, assert, refine metadata.
- `v2/clinical_nlp_lab/natural_validation.py`: fixed organizer validation contract for documents 181–200.
- `v2/clinical_nlp_lab/training.py`: pure document-level threshold calibration and metric reporting.
- `v2/scripts/train_ner_subprocess.py`: post-training document-level evaluation and calibration artifact publication.
- `v2/clinical_nlp_lab/kaggle_phases.py`: fixed validation manifest, Phase 12 counters, and summary fields.
- `v2/tools/build_kaggle_notebook.py`: smoke-test checklist in generated notebooks.
- `v2/KAGGLE_RUNBOOK.md`: user-run smoke instructions and required error evidence.
- `v2/tests/test_chunk_boundary_consensus.py`: boundary merge unit tests.
- `v2/tests/test_qwen_entity_validator.py`: Qwen keep/drop/trim and invalid-response unit tests.
- `v2/tests/test_inference_data_flow.py`: unit tests for post-validation metadata reset and ordering using fakes.
- `v2/tests/test_natural_validation.py`: fixed manifest and partition unit tests.
- `v2/tests/test_document_ner_metrics.py`: pure document-level calibration and report unit tests.
- Existing relevant tests are updated only where public result fields or notebook text change.

---

### Task 1: Replace Union-Based Chunk Merge with Boundary Consensus

**Files:**
- Create: `v2/tests/test_chunk_boundary_consensus.py`
- Modify: `v2/clinical_nlp_lab/ner.py:258-332`

**Interfaces:**
- Consumes: per-window `EntityAnnotation` values produced by `bio_predictions_to_spans()`.
- Produces: `merge_chunk_predictions(chunk_predictions, raw_text) -> list[EntityAnnotation]`, where every returned boundary was present in an input prediction and evidence records independent window support.

- [x] **Step 1: Write failing boundary-consensus unit tests**

```python
from clinical_nlp_lab.ner import merge_chunk_predictions
from clinical_nlp_lab.schema import EntityAnnotation


def entity(raw: str, start: int, end: int, confidence: float, window: int):
    return EntityAnnotation(
        text=raw[start:end],
        type="DISEASE",
        position=(start, end),
        confidence=confidence,
        evidence=[f"transformer_window:{window}"],
    )


def test_conflicting_overlap_never_creates_union_boundary():
    raw = "0123456789" * 8
    merged = merge_chunk_predictions(
        [entity(raw, 10, 40, 0.97, 0), entity(raw, 30, 60, 0.96, 1)], raw
    )
    assert [(item.start, item.end) for item in merged] in ([(10, 40)], [(30, 60)])
    assert (10, 60) not in {(item.start, item.end) for item in merged}


def test_identical_boundaries_accumulate_window_evidence():
    raw = "Bệnh nhân được chẩn đoán viêm phổi cộng đồng."
    start = raw.index("viêm phổi cộng đồng")
    end = start + len("viêm phổi cộng đồng")
    merged = merge_chunk_predictions(
        [entity(raw, start, end, 0.91, 0), entity(raw, start, end, 0.95, 1)], raw
    )
    assert len(merged) == 1
    assert merged[0].position == (start, end)
    assert merged[0].confidence == 0.95
    assert set(merged[0].evidence) >= {"transformer_window:0", "transformer_window:1"}


def test_near_boundaries_choose_one_observed_boundary_without_crossing_clause_mark():
    raw = "Tiền sử viêm phổi; hiện không khó thở."
    start = raw.index("viêm phổi")
    exact_end = start + len("viêm phổi")
    merged = merge_chunk_predictions(
        [entity(raw, start, exact_end, 0.94, 0), entity(raw, start, exact_end + 1, 0.93, 1)], raw
    )
    assert len(merged) == 1
    assert merged[0].position in {(start, exact_end), (start, exact_end + 1)}
    assert ";" not in merged[0].text


def test_long_observed_boundary_is_not_removed_by_length():
    raw = "x " + ("chẩn đoán hợp lệ " * 8).strip() + "."
    start, end = 2, len(raw) - 1
    merged = merge_chunk_predictions([entity(raw, start, end, 0.99, 0)], raw)
    assert len(merged) == 1
    assert len(merged[0].text) > 100
```

- [x] **Step 2: Run the tests and verify RED**

Run: `python -m pytest v2/tests/test_chunk_boundary_consensus.py -q`

Expected: the conflicting-overlap test fails because current code creates `(10, 60)`.

- [x] **Step 3: Add window evidence at decode time**

In `TransformerNERDetector.detect()`, append the owner-window identifier before merging:

```python
for entity in decoded:
    entity.evidence = sorted(set(entity.evidence + [f"transformer_window:{chunk_index}"]))
chunk_entities.extend(decoded)
```

- [x] **Step 4: Implement observed-boundary consensus**

Replace union construction with grouping and deterministic selection:

```python
def _window_support(entity: EntityAnnotation) -> int:
    return len({item for item in entity.evidence if item.startswith("transformer_window:")})


def _boundary_rank(entity: EntityAnnotation) -> tuple[int, float, int, int, int]:
    return (
        _window_support(entity),
        float(entity.confidence),
        -(entity.end - entity.start),
        -entity.start,
        -entity.end,
    )
```

Deduplicate identical `(start, end, type)` values by merging evidence and taking maximum confidence. Implement `_near_equivalent(left, right, raw_text)` as start/end deltas of at most one character with no `.;:\n\r` delimiter in the symmetric boundary difference. Near-equivalent predictions may combine window evidence but must select one observed boundary. Other overlapping same-type predictions are conflicts: select the higher `_boundary_rank()` value without combining support and never construct `min(start), max(end)`. Pass the selected observed entities to `resolve_overlaps()`.

- [x] **Step 5: Verify GREEN and related regressions**

Run:

```powershell
python -m pytest v2/tests/test_chunk_boundary_consensus.py -q
python -m pytest v2/tests/test_ner_policy.py v2/tests/test_ner_confidence_filter.py v2/tests/test_inference_data_flow.py -q
```

Expected: all tests pass.

- [x] **Step 6: Commit Task 1**

```powershell
git add -- v2/clinical_nlp_lab/ner.py v2/tests/test_chunk_boundary_consensus.py
git commit -m "fix: select consensus NER chunk boundaries"
```

---

### Task 2: Add Strict Qwen Entity Validation

**Files:**
- Create: `v2/clinical_nlp_lab/qwen_entity_validator.py`
- Create: `v2/tests/test_qwen_entity_validator.py`
- Modify: `v2/clinical_nlp_lab/qwen_refiner.py:1-75`

**Interfaces:**
- Produces: `EntityValidationDecision`, `EntityValidationCounters`, `EntityValidationResult`, and `QwenEntityValidator.validate(entities, raw_text) -> EntityValidationResult`.
- Produces: `RequiredQwenRefiner.validate_entities(entities, raw_text) -> tuple[EntityAnnotation, ...]` and `validation_counters() -> dict[str, object]`.
- Preserves: existing `RequiredQwenRefiner.refine()` behavior for candidate reranking and assertion refinement.

- [x] **Step 1: Write failing keep/drop/trim unit tests**

Use the existing fake-vLLM pattern from `v2/tests/test_qwen_refiner_required.py`:

```python
class FakeEngine:
    def __init__(self, responses):
        self.responses = list(responses)
        self.generate_calls = 0

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.generate_calls += 1
        batch = self.responses.pop(0)
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]) for text in batch]


@pytest.fixture
def fake_engine(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=FakeSamplingParams))
    return lambda responses: FakeEngine([responses])


RAW_TEXT = "Bệnh nhân có nguy cơ viêm phổi do nằm lâu."


def ner_entity() -> EntityAnnotation:
    return EntityAnnotation(
        text="nguy cơ viêm phổi do nằm lâu",
        type="DISEASE",
        position=(13, 41),
        candidates=["J18.9"],
        assertions=["isHistorical"],
        evidence=["proposal_ner"],
    )


def kb_entity(score: float) -> EntityAnnotation:
    start = RAW_TEXT.index("viêm phổi")
    end = start + len("viêm phổi")
    return EntityAnnotation(
        text=RAW_TEXT[start:end],
        type="DISEASE",
        position=(start, end),
        evidence=["proposal_kb_first"],
        ranked_candidates=[{"candidate_id": "J18.9", "name": "Viêm phổi", "score": score}],
    )


def test_trim_translates_relative_offsets_and_clears_metadata(fake_engine):
    original = ner_entity()
    engine = fake_engine(['{"action":"trim","relative_start":8,"relative_end":17,"entity_type":"DISEASE"}'])
    result = QwenEntityValidator(engine).validate((original,), RAW_TEXT)
    assert result.entities[0].text == "viêm phổi"
    assert result.entities[0].position == (21, 30)
    assert result.entities[0].candidates == []
    assert result.entities[0].assertions == []


@pytest.mark.parametrize("response", [
    "not json",
    '{"action":"extend","entity_type":"DISEASE"}',
    '{"action":"trim","relative_start":-1,"relative_end":4,"entity_type":"DISEASE"}',
    '{"action":"trim","relative_start":0,"relative_end":999,"entity_type":"DISEASE"}',
    '{"action":"trim","relative_start":0,"relative_end":4,"entity_type":"UNKNOWN"}',
])
def test_invalid_entity_decision_is_rejected(fake_engine, response):
    with pytest.raises(ValueError):
        QwenEntityValidator(fake_engine([response])).validate((ner_entity(),), RAW_TEXT)


def test_exact_kb_entity_bypasses_qwen(fake_engine):
    entity = kb_entity(score=1.0)
    engine = fake_engine([])
    result = QwenEntityValidator(engine).validate((entity,), RAW_TEXT)
    assert result.entities == (entity,)
    assert engine.generate_calls == 0
    assert result.counters.kb_bypass == 1
```

Also test `keep`, `drop`, response-count mismatch, whitespace-only trim, and exact raw-offset round trip.

- [x] **Step 2: Run the tests and verify RED**

Run: `python -m pytest v2/tests/test_qwen_entity_validator.py -q`

Expected: import failure because `qwen_entity_validator.py` does not exist.

- [x] **Step 3: Implement decision and counter dataclasses**

```python
@dataclass(frozen=True, slots=True)
class EntityValidationDecision:
    action: Literal["keep", "drop", "trim"]
    relative_start: int | None = None
    relative_end: int | None = None
    entity_type: str | None = None


@dataclass(slots=True)
class EntityValidationCounters:
    query_count: int = 0
    keep: int = 0
    drop: int = 0
    trim: int = 0
    kb_bypass: int = 0
    before_type_counts: Counter[str] = field(default_factory=Counter)
    after_type_counts: Counter[str] = field(default_factory=Counter)
    before_length_buckets: Counter[str] = field(default_factory=Counter)
    after_length_buckets: Counter[str] = field(default_factory=Counter)
    max_before_length: int = 0
    max_after_length: int = 0
```

Add `to_dict()`, `add()`, and `delta()` as pure deterministic methods.

- [x] **Step 4: Implement strict prompt, schema, parsing, and application**

Use `build_sampling_kwargs()`, `iter_batches()`, and `parse_json_object()` from `vllm_compat.py`. The schema allows only `keep`, `drop`, and `trim`; `trim` must include relative boundaries and an entity type from `ENTITY_TYPE_TO_ID`.

The prompt must state in Vietnamese that explanations, causes, risk factors, and treatment advice are not part of the smallest entity mention. Include one long valid ICD example and one invalid explanatory-clause example. Apply trim only after validating containment and `raw_text[absolute_start:absolute_end]` equality.

- [x] **Step 5: Add the required-Qwen facade**

In `RequiredQwenRefiner.__init__`, instantiate `QwenEntityValidator`. Add:

```python
def validate_entities(self, entities, raw_text):
    try:
        result = self._entity_validator.validate(entities, raw_text)
        self._entity_validation_counters.add(result.counters)
        return result.entities
    except RequiredQwenError:
        raise
    except Exception as exc:
        raise RequiredQwenError(str(exc)) from exc


def validation_counters(self):
    return self._entity_validation_counters.to_dict()
```

Do not call `validate_entities()` inside the legacy `refine()` method; Task 3 places it at the correct point in inference.

- [x] **Step 6: Verify GREEN and existing Qwen tests**

Run:

```powershell
python -m pytest v2/tests/test_qwen_entity_validator.py -q
python -m pytest v2/tests/test_qwen_refiner_required.py v2/tests/test_required_qwen_phase.py -q
```

Expected: all tests pass without CUDA, model downloads, or network access.

- [x] **Step 7: Commit Task 2**

```powershell
git add -- v2/clinical_nlp_lab/qwen_entity_validator.py v2/clinical_nlp_lab/qwen_refiner.py v2/tests/test_qwen_entity_validator.py
git commit -m "feat: validate NER entities with strict Qwen decisions"
```

---

### Task 3: Reorder Inference Around Validated Entities

**Files:**
- Modify: `v2/clinical_nlp_lab/inference.py:1-265`
- Modify: `v2/tests/test_inference_data_flow.py`

**Interfaces:**
- Produces: `_link_validated_entities(entities, linker, policy) -> tuple[EntityAnnotation, ...]`.
- Changes: `infer_document()` order to merge before Qwen entity validation and link/assert only afterward.
- Preserves: `FinalModelBundle` and `InferenceConfig` public constructors.

- [x] **Step 1: Write a failing unit test for trim then relink/assert**

```python
def test_trimmed_entity_is_relinked_and_asserted_using_final_boundary():
    raw = "Nguy cơ viêm phổi do nằm lâu."
    calls = []

    class Ner:
        def detect(self, _raw):
            return [EntityAnnotation(
                text="Nguy cơ viêm phổi do nằm lâu",
                type="DISEASE",
                position=(0, 28),
                confidence=0.98,
                evidence=["transformer_window:0"],
            )]

    class Refiner:
        def validate_entities(self, entities, _raw):
            calls.append("validate")
            return (replace(entities[0], text="viêm phổi", position=(8, 17), candidates=[], assertions=[]),)
        def refine(self, entities, _raw):
            calls.append("metadata")
            return entities

    class Linker:
        def rank_candidates(self, entity_type, mention, limit=20):
            calls.append(("link", entity_type, mention))
            return [{"candidate_id": "J18.9", "name": "Viêm phổi", "score": 1.0}]

    class Assertion:
        def predict(self, _raw, entities):
            calls.append(("assert", entities[0].position))
            return {(8, 17, "DISEASE"): ["isHistorical"]}

    class AcceptTopPolicy:
        def apply(self, ranked):
            return [ranked[0]["candidate_id"]]

    document = infer_document(
        "1", raw,
        FinalModelBundle(Ner(), None, Assertion(), AcceptTopPolicy(), Linker(), Refiner()),
        InferenceConfig(enable_qwen=True, enable_kb_recovery=False),
    )
    assert document.entities[0].text == "viêm phổi"
    assert document.entities[0].candidates == ["J18.9"]
    assert calls.index("validate") < calls.index(("link", "DISEASE", "viêm phổi"))
    assert calls.index(("assert", (8, 17))) < calls.index("metadata")


def test_exact_kb_span_over_160_characters_is_not_filtered():
    raw = "bệnh " * 40
    proposal = SpanProposal(
        raw.rstrip(), "DISEASE", 0, len(raw.rstrip()), 0.99, "kb_first",
        ranked_candidates=({"candidate_id": "Z99", "name": "Long ICD", "score": 1.0},),
    )
    records = [ClinicalRecord("1", "1:record-0001", 0, len(raw), (0,))]
    merged = merge_raw_span_proposals([proposal], records, raw_text=raw)
    assert len(merged) == 1
    assert len(merged[0].text) > 160
```

- [x] **Step 2: Run the test and verify RED**

Run: `python -m pytest v2/tests/test_inference_data_flow.py::test_trimmed_entity_is_relinked_and_asserted_using_final_boundary -q`

Expected: failure because current inference links before Qwen refinement and has no `validate_entities()` call.

- [x] **Step 3: Implement source-priority proposal arbitration**

Extend `SpanProposal` with `support_count: int = 1`. Rank record proposals by:

```python
def _proposal_rank(proposal: SpanProposal):
    exact_kb = int(proposal.source == "kb_first" and any(
        float(item.get("score", 0.0)) == 1.0 for item in proposal.ranked_candidates
    ))
    return (exact_kb, proposal.support_count, proposal.confidence, -(proposal.end - proposal.start))
```

Use descending rank in `merge_raw_span_proposals()` so an exact KB span beats an overlapping Transformer span without using length as a global filter.

Remove the current `len(text) > 160` rejection from `_is_valid_proposal_boundary()`. Keep rejection of empty, punctuation-only, multiline, mid-word, and cross-record spans.

- [x] **Step 4: Implement post-validation linking**

Add `_link_validated_entities()` that converts each validated entity to a `SpanProposal`, calls `_attach_ranked_candidates()` and `_apply_candidate_policy()`, and returns copied entities containing only the new ranked/selected candidates.

- [x] **Step 5: Reorder `infer_document()`**

Implement this exact sequence:

```python
proposals = _call_ner(bundle, raw_text, config)
if config.enable_kb_recovery and bundle.kb_linker is not None:
    proposals.extend(kb_recovery_proposals)
merged_entities = merge_raw_span_proposals(proposals, records, raw_text=raw_text)
if config.enable_qwen and bundle.qwen_reranker is not None:
    merged_entities = tuple(bundle.qwen_reranker.validate_entities(merged_entities, raw_text))
merged_entities = _link_validated_entities(
    merged_entities, bundle.kb_linker, bundle.candidate_policy
)
merged_entities = _apply_assertions(merged_entities, raw_text, bundle.assertion_model)
if config.enable_qwen and bundle.qwen_reranker is not None:
    merged_entities = tuple(bundle.qwen_reranker.refine(merged_entities, raw_text))
```

Exact KB entities bypass validation inside `QwenEntityValidator` but still pass through the same downstream linking/assertion flow.

- [x] **Step 6: Verify GREEN and related unit tests**

Run:

```powershell
python -m pytest v2/tests/test_inference_data_flow.py -q
python -m pytest v2/tests/test_primary_inference_path.py v2/tests/test_required_qwen_phase.py v2/tests/test_single_toggle_qwen.py -q
```

Expected: all tests pass.

- [x] **Step 7: Commit Task 3**

```powershell
git add -- v2/clinical_nlp_lab/inference.py v2/tests/test_inference_data_flow.py
git commit -m "refactor: link metadata after Qwen entity validation"
```

---

### Task 4: Publish Qwen Entity Validation Diagnostics

**Files:**
- Modify: `v2/clinical_nlp_lab/qwen_entity_validator.py`
- Modify: `v2/clinical_nlp_lab/qwen_refiner.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py:256-390`
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py:591-731`
- Modify: `v2/tests/test_qwen_entity_validator.py`
- Modify: existing assertions in `v2/tests/test_required_qwen_phase.py`

**Interfaces:**
- Consumes: `RequiredQwenRefiner.validation_counters()`.
- Produces Phase 12 fields: `qwen_entity_query_count`, `qwen_entity_keep_count`, `qwen_entity_drop_count`, `qwen_entity_trim_count`, `qwen_entity_kb_bypass_count`, before/after type counts, before/after length buckets, and maximum lengths.

- [x] **Step 1: Write failing pure counter unit tests**

```python
def test_validation_counter_add_and_delta_are_deterministic():
    before = EntityValidationCounters(query_count=2, keep=1, drop=1)
    increment = EntityValidationCounters(query_count=3, keep=1, trim=2, kb_bypass=4)
    after = before.copy().add(increment)
    assert after.delta(before).to_dict() == increment.to_dict()
```

Add a length-bucket test for `1-50`, `51-100`, and `101+`.

- [x] **Step 2: Run the tests and verify RED**

Run: `python -m pytest v2/tests/test_qwen_entity_validator.py -q`

Expected: failure until `copy()`, `add()`, `delta()`, bucket counts, and max lengths are implemented.

- [x] **Step 3: Add cumulative and per-document snapshots**

Expose `validation_counters()` as a deep copied dictionary. In `run_inference_with_bundle()`, snapshot counters immediately before and after each `infer_document()` call and write the delta under `qwen_entity_validation` in that document's diagnostic JSON.

- [x] **Step 4: Add Phase 12 summary fields**

After `run_inference_with_bundle()` returns, merge `bundle.qwen_reranker.validation_counters()` into `qwen_summary`. Keep existing rerank/assertion counters and `qwen_status` unchanged. Qwen-disabled mode emits zeros and empty dictionaries.

- [x] **Step 5: Verify GREEN and existing observability tests**

Run:

```powershell
python -m pytest v2/tests/test_qwen_entity_validator.py -q
python -m pytest v2/tests/test_required_qwen_phase.py v2/tests/test_kaggle_observability.py v2/tests/test_output_semantic_quality.py -q
```

Expected: all tests pass. No new Phase 12 integration test is added.

- [x] **Step 6: Commit Task 4**

```powershell
git add -- v2/clinical_nlp_lab/qwen_entity_validator.py v2/clinical_nlp_lab/qwen_refiner.py v2/clinical_nlp_lab/pipeline.py v2/clinical_nlp_lab/kaggle_phases.py v2/tests/test_qwen_entity_validator.py v2/tests/test_required_qwen_phase.py
git commit -m "feat: report Qwen entity validation decisions"
```

---

### Task 5: Bind Natural Validation and Calibrate on Document-Level Spans

**Files:**
- Create: `v2/clinical_nlp_lab/natural_validation.py`
- Create: `v2/tests/test_natural_validation.py`
- Create: `v2/tests/test_document_ner_metrics.py`
- Modify: `v2/clinical_nlp_lab/kaggle_phases.py:170-276, 402-423`
- Modify: `v2/clinical_nlp_lab/training.py:81-176`
- Modify: `v2/clinical_nlp_lab/ner.py:212-332`
- Modify: `v2/scripts/train_ner_subprocess.py:104-289`
- Modify: relevant expected fields in `v2/tests/test_kaggle_phase_runners.py` and `v2/tests/test_kaggle_phase_results.py`

**Interfaces:**
- Produces: `NATURAL_VALIDATION_IDS == ("181", ..., "200")`.
- Produces: `validate_natural_validation_summary(summary, dataset_fingerprint) -> dict[str, object]` and `build_natural_validation_manifest(dataset_root, dataset_fingerprint) -> dict[str, object]`.
- Produces: `apply_natural_validation_partition(train_ids, validation_ids) -> tuple[tuple[str, ...], tuple[str, ...]]`.
- Produces: `calibrate_document_entity_threshold(expected, predicted, provenance) -> dict[str, object]`.
- Produces: `compare_document_merge_strategies(expected, raw_chunk_predictions, provenance, raw_texts) -> dict[str, object]`, containing `legacy_union` and `boundary_consensus` metrics from the same predictions.
- Produces: `TransformerNERDetector.predict_chunks(raw_text) -> list[EntityAnnotation]` for audit/calibration; `detect()` remains the filtered production API.

- [x] **Step 1: Write failing natural-validation unit tests**

```python
def test_natural_validation_partition_is_fixed_and_disjoint():
    train, validation = apply_natural_validation_partition(
        tuple(str(value) for value in range(101, 221)),
        ("501", "502"),
    )
    assert validation == ("181", "182", "183", "184", "185", "186", "187", "188", "189", "190",
                          "191", "192", "193", "194", "195", "196", "197", "198", "199", "200",
                          "501", "502")
    assert not set(train) & set(validation)


def test_manifest_rejects_wrong_dataset_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="fingerprint"):
        build_natural_validation_manifest(tmp_path, "wrong")


def test_natural_validation_summary_binds_expected_inventory():
    summary = {
        "document_ids": [str(value) for value in range(181, 201)],
        "entity_count": 1488,
        "type_counts": {
            "CHẨN_ĐOÁN": 496,
            "THUỐC": 248,
            "TRIỆU_CHỨNG": 496,
            "TÊN_XÉT_NGHIỆM": 124,
            "KẾT_QUẢ_XÉT_NGHIỆM": 124,
        },
        "over_50": 120,
        "over_100": 5,
        "max_length": 110,
    }
    manifest = validate_natural_validation_summary(summary, REQUIRED_DATASET_FINGERPRINT)
    assert manifest["schema_id"] == "clinical_nlp.natural_validation"
    assert manifest["summary"] == summary
```

`build_natural_validation_manifest()` scans the real dataset layout and delegates all contract checks to the pure `validate_natural_validation_summary()` function tested above.

- [x] **Step 2: Write failing document-metric unit tests**

```python
def entity(start: int, end: int, entity_type: str, confidence: float = 1.0):
    return EntityAnnotation(
        text="x" * (end - start),
        type=entity_type,
        position=(start, end),
        confidence=confidence,
    )


def test_document_threshold_uses_exact_span_f1_with_precision_tiebreak():
    gold = {"181": [entity(10, 20, "DISEASE")]}
    predicted = {"181": [
        entity(10, 20, "DISEASE", confidence=0.95),
        entity(30, 45, "DISEASE", confidence=0.70),
    ]}
    report = calibrate_document_entity_threshold(gold, predicted, {"181": "organizer"})
    assert report["confidence_threshold"] == 0.95
    assert report["exact"]["precision"] == 1.0
    assert report["exact"]["recall"] == 1.0
    assert report["length_buckets"]["1-50"]["gold"] == 1


def test_consensus_merge_beats_legacy_union_on_same_chunk_predictions():
    raw = "x" * 80
    gold = {"181": [entity(10, 40, "DISEASE")]}
    chunks = {"181": [
        entity(10, 40, "DISEASE", confidence=0.97),
        entity(30, 60, "DISEASE", confidence=0.96),
    ]}
    report = compare_document_merge_strategies(gold, chunks, {"181": "organizer"}, {"181": raw})
    assert report["legacy_union"]["exact"]["precision"] == 0.0
    assert report["boundary_consensus"]["exact"]["precision"] == 1.0
```

Also test type breakdown, overlap metrics, `51-100`, `101+`, and deterministic JSON-compatible output.

- [x] **Step 3: Run both test files and verify RED**

Run:

```powershell
python -m pytest v2/tests/test_natural_validation.py v2/tests/test_document_ner_metrics.py -q
```

Expected: import failures for the new interfaces.

- [x] **Step 4: Implement the fixed validation contract**

In `natural_validation.py`, define:

```python
REQUIRED_DATASET_FINGERPRINT = "18a391e51786630b482bb500d5129eb102ae144450d7fc18b149f2799054f028"
NATURAL_VALIDATION_IDS = tuple(str(value) for value in range(181, 201))
OFFICIAL_TYPES = {"CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"}
```

The manifest validates 20 input/GT pairs, 1,488 entities, all five official types, 120 spans longer than 50 characters, five spans longer than 100, and maximum length 110. It writes schema ID/version, document IDs, fingerprint, counts, and length buckets.

- [x] **Step 5: Bind the validation IDs into every training stage**

Phase 5 writes `artifacts/splits/natural_validation.json`. `_write_stage_input()` removes 181–200 from every selected train list and uses them as the organizer validation component, while retaining the existing synthetic validation IDs. Phase 11 uses the same fixed IDs for assertion validation and excludes them from head training.

Do not mutate the source dataset and do not use the competition inference directory.

- [x] **Step 6: Implement pure document-level calibration**

Evaluate thresholds `(0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95, 0.99)`. Select maximum exact typed F1; break ties by precision, then higher threshold. Report exact and type-matched overlap metrics overall, by type, by provenance, and by length bucket.

Add a training-only `_legacy_union_for_audit()` that reproduces the old `min(start)/max(end)` behavior without being imported by production inference. `compare_document_merge_strategies()` runs legacy union and the new consensus merger on the same raw chunk predictions, reports both metric sets, and raises when consensus exact precision is not higher or exact F1 decreases on the fixed natural validation slice.

- [x] **Step 7: Expose unfiltered chunk predictions and publish post-merge metrics**

Refactor `TransformerNERDetector.detect()`:

```python
def predict_chunks(self, raw_text: str) -> list[EntityAnnotation]:
    # tokenize, infer, decode, attach transformer_window evidence; no threshold filter

def detect(self, raw_text: str) -> list[EntityAnnotation]:
    merged = merge_chunk_predictions(self.predict_chunks(raw_text), raw_text)
    return filter_entities_by_confidence(merged, self.confidence_threshold)
```

After the selected checkpoint is saved on the main process, release the Trainer model GPU allocation, instantiate `TransformerNERDetector` with threshold `0.0`, collect raw chunk predictions for the fixed validation documents, compare legacy and consensus merge strategies, calibrate the consensus predictions using the pure function, overwrite `ner_calibration.json` with objective `document_exact_f1_precision_tiebreak`, and write `document_validation_report.json` beside the model.

Add these fields to `training_result.json`: `document_entity_precision`, `document_entity_recall`, `document_entity_f1`, `document_overlap_f1`, and `document_ner_confidence_threshold`.

- [x] **Step 8: Verify GREEN and training contract regressions**

Run:

```powershell
python -m pytest v2/tests/test_natural_validation.py v2/tests/test_document_ner_metrics.py -q
python -m pytest v2/tests/test_kaggle_phase_runners.py v2/tests/test_kaggle_phase_results.py v2/tests/test_ner_calibration.py v2/tests/test_training_helpers.py -q
```

Expected: all tests pass without loading an actual Transformer checkpoint.

- [x] **Step 9: Commit Task 5**

```powershell
git add -- v2/clinical_nlp_lab/natural_validation.py v2/clinical_nlp_lab/kaggle_phases.py v2/clinical_nlp_lab/training.py v2/clinical_nlp_lab/ner.py v2/scripts/train_ner_subprocess.py v2/tests/test_natural_validation.py v2/tests/test_document_ner_metrics.py v2/tests/test_kaggle_phase_runners.py v2/tests/test_kaggle_phase_results.py
git commit -m "feat: calibrate NER on fixed natural validation spans"
```

---

### Task 6: Update Generated Notebooks and User Smoke Checklist

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py`
- Modify: `v2/tests/test_kaggle_notebook_variants.py`
- Modify: `v2/KAGGLE_RUNBOOK.md`
- Regenerate: `v2/medical_information_extraction_kaggle.ipynb`
- Regenerate: `ai-race-training-v2-enableQwen.ipynb`
- Regenerate: `ai-race-training-v2-unableQwen.ipynb`

**Interfaces:**
- Produces: generated notebook markdown naming required Qwen entity validation and the Phase 12 smoke fields.
- Preserves: 13 canonical phases and the single `ENABLE_QWEN_RERANKER` toggle.

- [x] **Step 1: Write failing notebook text unit tests**

```python
def test_enabled_notebook_documents_entity_validation_smoke_fields():
    notebook = _load_builder().build_notebook(enable_qwen_reranker=True)
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"]
        if cell["cell_type"] == "markdown"
    )
    assert "qwen_entity_validation" in markdown
    assert "keep / drop / trim" in markdown
    assert "Nếu lỗi, lưu cell output và stack trace" in markdown
```

- [x] **Step 2: Run the test and verify RED**

Run: `python -m pytest v2/tests/test_kaggle_notebook_variants.py -q`

Expected: the new markdown assertions fail.

- [x] **Step 3: Add the smoke checklist to the builder and runbook**

The checklist tells the user to verify all 13 phases, `qwen_status=COMPLETED`, entity validation counters, 100 output JSON files, valid ZIP/CRC, zero offset errors, and before/after length reports. The error handoff requires the failing cell output, complete traceback, Phase 12 result when present, `diagnostics/qwen_summary.json`, and `diagnostics/output_quality.json`.

- [x] **Step 4: Regenerate the notebook artifacts**

Run from `D:\AI Race Viettel`:

```powershell
python v2/tools/build_kaggle_notebook.py --qwen-mode enabled --output v2/medical_information_extraction_kaggle.ipynb
python v2/tools/build_kaggle_notebook.py --qwen-mode enabled --output ai-race-training-v2-enableQwen.ipynb
python v2/tools/build_kaggle_notebook.py --qwen-mode unable --output ai-race-training-v2-unableQwen.ipynb
```

- [x] **Step 5: Verify generated notebooks**

Run:

```powershell
python -m pytest v2/tests/test_kaggle_notebook_variants.py v2/tests/test_inference_notebook.py -q
```

Expected: all tests pass; each notebook has 13 phases and no code-cell syntax errors.

- [x] **Step 6: Commit Task 6**

```powershell
git add -- v2/tools/build_kaggle_notebook.py v2/tests/test_kaggle_notebook_variants.py v2/KAGGLE_RUNBOOK.md v2/medical_information_extraction_kaggle.ipynb ai-race-training-v2-enableQwen.ipynb ai-race-training-v2-unableQwen.ipynb
git commit -m "docs: add Kaggle NER validation smoke checklist"
```

---

### Task 7: Local Verification and Handoff

**Files:**
- No production changes unless verification exposes a defect.
- Update the implementation plan checkboxes as tasks complete.

**Interfaces:**
- Produces: a locally verified branch, updated Kaggle notebook, and exact user smoke-test instructions.

- [x] **Step 1: Run all newly added unit tests**

```powershell
python -m pytest v2/tests/test_chunk_boundary_consensus.py v2/tests/test_qwen_entity_validator.py v2/tests/test_natural_validation.py v2/tests/test_document_ner_metrics.py -q
```

Expected: all pass on CPU with no network access.

- [x] **Step 2: Run existing tests related to changed modules**

```powershell
$env:PYTHONPATH = (Join-Path (Get-Location) 'v2')
python -m pytest v2/tests/test_inference_data_flow.py v2/tests/test_primary_inference_path.py v2/tests/test_ner_calibration.py v2/tests/test_ner_confidence_filter.py v2/tests/test_required_qwen_phase.py v2/tests/test_qwen_refiner_required.py v2/tests/test_single_toggle_qwen.py v2/tests/test_kaggle_phase_runners.py v2/tests/test_kaggle_phase_results.py v2/tests/test_kaggle_observability.py v2/tests/test_kaggle_notebook_variants.py v2/tests/test_inference_notebook.py -q
```

Expected: all pass. These are existing regression tests, not a new integration suite.

- [x] **Step 3: Run static and repository checks**

```powershell
python -m compileall -q v2/clinical_nlp_lab v2/scripts v2/tools
git diff --check
git status --short
```

Expected: Python compilation succeeds, no whitespace errors, and only intentional files remain modified.

- [x] **Step 4: Inspect acceptance evidence**

Confirm locally:

- conflicting chunk predictions never produce a union boundary;
- long exact-KB unit fixture survives;
- Qwen keep/drop/trim and all invalid responses are covered;
- post-trim candidate/assertion metadata is cleared and recalculated;
- validation IDs 181–200 are absent from every stage train list;
- document calibration uses exact span F1 with precision tie-break;
- notebooks remain 13-phase and expose the user smoke checklist.

- [x] **Step 5: Commit any final plan/checklist-only adjustment**

If verification required no code adjustment, do not create an empty commit. If only tracked documentation changed:

```powershell
git add -- docs/superpowers/plans/2026-07-27-precision-first-ner-boundary.md
git commit -m "docs: record precision-first NER verification"
```

- [x] **Step 6: Hand off Kaggle smoke test to the user**

Provide the enabled notebook path, branch/commit, Run All prerequisites, and the exact error bundle to return if it fails. Do not claim Kaggle runtime success until the user reports the result.
