# Qwen Trim Boundary Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Qwen entity trimming from creating word-boundary-invalid submission spans while supporting valid one-sided trims.

**Architecture:** Keep validation within `QwenEntityValidator`, where Qwen decisions become domain entities. Relax relative trim bounds, reject no-op trims, and apply the same alphanumeric boundary invariant used by inference and the final quality gate; invalid boundary trims fall back to the already-valid original entity.

**Tech Stack:** Python 3.10+, dataclasses, pytest, vLLM-compatible guided JSON schema.

## Global Constraints

- Permit left-only, right-only, and two-sided trims.
- Require a `trim` to make the entity strictly shorter than the original.
- Never publish a trimmed span that splits an alphanumeric run.
- Preserve the original entity for a word-boundary-invalid Qwen trim.
- Keep the final output-quality gate unchanged.
- Do not change training, models, datasets, or quality thresholds.

---

### Task 1: Support one-sided trim decisions

**Files:**
- Modify: `v2/clinical_nlp_lab/qwen_entity_validator.py:150-223`
- Test: `v2/tests/test_qwen_entity_validator.py:91-142`

**Interfaces:**
- Consumes: `EntityAnnotation.text`, `_validation_schema(entity)`, and `_parse_decision(response_text, entity)`.
- Produces: trim decisions satisfying `0 <= relative_start < relative_end <= len(entity.text)` while excluding `(0, len(entity.text))`.

- [ ] **Step 1: Replace the strict-both-sides tests with failing one-sided and no-op tests**

```python
def test_guided_schema_allows_one_sided_trim_boundaries():
    entity = ner_entity()
    schema = _validation_schema(entity)
    trim = next(
        variant
        for variant in schema["oneOf"]
        if variant["properties"]["action"]["enum"] == ["trim"]
    )

    assert trim["properties"]["relative_start"]["minimum"] == 0
    assert trim["properties"]["relative_end"]["maximum"] == len(entity.text)


@pytest.mark.parametrize(
    ("relative_start", "relative_end"),
    [
        (0, len("risk of pneumonia")),
        (ner_entity().text.index("pneumonia"), len(ner_entity().text)),
    ],
)
def test_parse_decision_allows_one_sided_trim(relative_start, relative_end):
    entity = ner_entity()
    decision = _parse_decision(
        json.dumps(
            {
                "action": "trim",
                "relative_start": relative_start,
                "relative_end": relative_end,
                "entity_type": "DISEASE",
            }
        ),
        entity,
    )

    assert decision.relative_start == relative_start
    assert decision.relative_end == relative_end


def test_parse_decision_rejects_no_op_trim():
    entity = ner_entity()
    response = json.dumps(
        {
            "action": "trim",
            "relative_start": 0,
            "relative_end": len(entity.text),
            "entity_type": "DISEASE",
        }
    )

    with pytest.raises(ValueError, match="shorten"):
        _parse_decision(response, entity)
```

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```powershell
python -m pytest -q tests/test_qwen_entity_validator.py -k "one_sided or no_op"
```

Working directory: `D:\AI Race Viettel\v2`

Expected: schema and one-sided parsing assertions fail because the current code requires both offsets to be strictly internal.

- [ ] **Step 3: Implement the minimal schema and parser changes**

Change the trim schema bounds to:

```python
"relative_start": {
    "type": "integer",
    "minimum": 0,
    "maximum": len(entity.text) - 1,
},
"relative_end": {
    "type": "integer",
    "minimum": 1,
    "maximum": len(entity.text),
},
```

Change parser validation to:

```python
if not (0 <= relative_start < relative_end <= len(entity.text)):
    raise ValueError("trim boundaries must stay within the original entity")
if relative_start == 0 and relative_end == len(entity.text):
    raise ValueError("trim must shorten the original entity")
```

Update the prompt so it says either boundary may be retained, but `trim` must shorten the entity. Use a right-only example with `relative_start=0`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

```powershell
python -m pytest -q tests/test_qwen_entity_validator.py -k "schema or prompt or decision or trim"
```

Expected: all selected tests pass.

- [ ] **Step 5: Commit Task 1**

```powershell
git add -- v2/clinical_nlp_lab/qwen_entity_validator.py v2/tests/test_qwen_entity_validator.py
git commit -m "fix: allow one-sided Qwen entity trims"
```

---

### Task 2: Fall back when a trim splits a word

**Files:**
- Modify: `v2/clinical_nlp_lab/qwen_entity_validator.py:282-310`
- Test: `v2/tests/test_qwen_entity_validator.py`

**Interfaces:**
- Consumes: parsed `EntityValidationDecision`, original `EntityAnnotation`, and document `raw_text`.
- Produces: a trimmed `EntityAnnotation` only for word-boundary-valid spans; otherwise returns the unchanged original entity.

- [ ] **Step 1: Write failing regression tests for left and right boundary splits**

```python
@pytest.mark.parametrize(
    ("relative_start", "relative_end"),
    [
        (1, len(ner_entity().text)),
        (0, len(ner_entity().text) - 1),
    ],
)
def test_word_boundary_invalid_trim_preserves_original_entity(
    fake_engine, relative_start, relative_end
):
    original = ner_entity()
    response = json.dumps(
        {
            "action": "trim",
            "relative_start": relative_start,
            "relative_end": relative_end,
            "entity_type": "DISEASE",
        }
    )

    result = QwenEntityValidator(fake_engine([response])).validate(
        (original,), RAW_TEXT
    )

    assert result.entities == (original,)
    assert result.entities[0].position == original.position
    assert result.entities[0].candidates == original.candidates
    assert result.entities[0].assertions == original.assertions
```

Retain `test_trim_translates_relative_offsets_and_clears_metadata` as the valid two-sided trim control.

- [ ] **Step 2: Run the regression tests and verify RED**

```powershell
python -m pytest -q tests/test_qwen_entity_validator.py -k "word_boundary_invalid"
```

Expected: both cases fail because `_apply_decision()` currently returns the character-trimmed entity.

- [ ] **Step 3: Implement the minimal post-trim boundary check**

After raw-text slice validation and before `replace(...)`, add:

```python
left_splits_word = (
    absolute_start > 0
    and raw_text[absolute_start - 1].isalnum()
    and raw_text[absolute_start].isalnum()
)
right_splits_word = (
    absolute_end < len(raw_text)
    and raw_text[absolute_end - 1].isalnum()
    and raw_text[absolute_end].isalnum()
)
if left_splits_word or right_splits_word:
    return entity
```

Do not change quality-gate thresholds and do not snap offsets.

- [ ] **Step 4: Run focused validator and output-quality tests**

```powershell
python -m pytest -q tests/test_qwen_entity_validator.py tests/test_output_quality_gate.py tests/test_output_semantic_quality.py
```

Expected: all focused tests pass.

- [ ] **Step 5: Run the full test suite**

```powershell
python -m pytest -q
```

Expected: zero failures.

- [ ] **Step 6: Review the diff for scope**

```powershell
git diff --check
git diff -- v2/clinical_nlp_lab/qwen_entity_validator.py v2/tests/test_qwen_entity_validator.py
```

Expected: no whitespace errors; only validator and regression-test changes.

- [ ] **Step 7: Commit Task 2**

```powershell
git add -- v2/clinical_nlp_lab/qwen_entity_validator.py v2/tests/test_qwen_entity_validator.py
git commit -m "fix: reject Qwen trims that split words"
```
