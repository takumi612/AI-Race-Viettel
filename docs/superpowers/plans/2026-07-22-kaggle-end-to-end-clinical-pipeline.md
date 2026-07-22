# Kaggle End-to-End Clinical Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a validated training corpus and a single-GPU Kaggle workflow that trains, reloads, evaluates, runs inference, and packages valid submission output plus reusable weights.

**Architecture:** Keep `v2/` as the maintained runtime. Add explicit data contracts and grouped splitting, train XLM-R with entity-level evidence, combine NER with ontology retrieval, enforce deterministic candidate policy and robust fallback, and generate the canonical Kaggle notebook from tested Python source.

**Tech Stack:** Python 3.11, PyTorch, Hugging Face Transformers, XLM-R, BM25S, FAISS, SentenceTransformers, optional vLLM/Qwen, pytest, Jupyter notebook JSON.

## Global Constraints

- Run training and deterministic inference on one Kaggle T4 or P100 with 16 GB VRAM.
- Keep IDs 101-200 organizer ground truth immutable.
- Preserve raw text and exclusive-end character offsets.
- Do not require Qwen/vLLM for training or valid output generation.
- Only disease and drug entities receive ICD-10/RxNorm candidates.
- Enforce `candidate_output_k=1` after deterministic and Qwen paths.
- Every production behavior change starts with a failing test.

---

### Task 1: Dataset contract, manifest, and leakage-safe split

**Files:**
- Create: `v2/clinical_nlp_lab/dataset_quality.py`
- Modify: `v2/clinical_nlp_lab/data.py`
- Modify: `scripts/validate_synthetic_train_v2.py`
- Test: `v2/tests/test_dataset_quality.py`
- Test: `v2/tests/test_grouped_split.py`

**Interfaces:**
- Produces: `DatasetRecord`, `build_dataset_manifest()`, `validate_dataset_contract()`, `grouped_train_validation_split()` and `audit_split_leakage()`.
- Consumes: `ClinicalDocument`, bundled ICD/RxNorm candidate IDs, and optional `genre_manifest.json` metadata.

- [ ] **Step 1: Write failing tests for candidate/schema validation and immutable organizer labels**

```python
def test_contract_rejects_unknown_disease_candidate():
    report = validate_dataset_contract([document_with_candidate("UNKNOWN")], {"I10"}, set())
    assert report["is_valid"] is False
    assert report["errors"][0]["code"] == "unknown_icd_candidate"

def test_organizer_gt_hashes_are_preserved(tmp_path):
    before = hash_ground_truth_range(SOURCE_GT, range(101, 201))
    validate_dataset_contract(load_annotated_documents(SOURCE), ICD_IDS, RX_IDS)
    assert hash_ground_truth_range(SOURCE_GT, range(101, 201)) == before
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest v2/tests/test_dataset_quality.py -q -p no:cacheprovider`

Expected: collection/import failure because `dataset_quality.py` does not exist.

- [ ] **Step 3: Implement manifest and contract validation**

```python
@dataclass(frozen=True, slots=True)
class DatasetRecord:
    document_id: str
    source_bucket: str
    template_group: str
    genre: str
    long_tail: bool
    primary_surfaces: tuple[str, ...]
    sha256: str

def validate_dataset_contract(documents, icd_ids, rxnorm_ids):
    errors = []
    # validate offsets, exact official fields, assertion eligibility,
    # required disease/drug candidates, and ontology membership
    return {"is_valid": not errors, "errors": errors}
```

- [ ] **Step 4: Write failing grouped-split leakage tests**

```python
def test_grouped_split_never_crosses_template_or_surface_groups():
    train, validation, manifest = grouped_train_validation_split(DOCUMENTS, RECORDS, 0.2, 42)
    audit = audit_split_leakage(train, validation, RECORDS)
    assert audit == {"document_ids": [], "template_groups": [], "surface_groups": []}
```

- [ ] **Step 5: Implement deterministic grouped splitting and reports**

```python
def grouped_train_validation_split(documents, records, validation_fraction=0.2, seed=42):
    groups = build_connected_groups(records, keys=("template_group", "primary_surfaces"))
    train_ids, validation_ids = allocate_groups(groups, validation_fraction, seed)
    return select(documents, train_ids), select(documents, validation_ids), build_split_manifest(...)
```

- [ ] **Step 6: Run focused tests and dataset validator**

Run: `python -m pytest v2/tests/test_dataset_quality.py v2/tests/test_grouped_split.py tests/test_build_synthetic_train_v2.py tests/test_synthetic_train_v2_generator.py -q -p no:cacheprovider`

Run: `python scripts/validate_synthetic_train_v2.py`

Expected: all tests pass; validator reports 2,200 paired documents and zero contract errors.

---

### Task 2: Preprocessing, NER metrics, and detector policy

**Files:**
- Modify: `v2/clinical_nlp_lab/training.py`
- Modify: `v2/clinical_nlp_lab/ner.py`
- Modify: `v2/clinical_nlp_lab/text.py`
- Test: `v2/tests/test_ner_policy.py`
- Test: `v2/tests/test_entity_metrics.py`

**Interfaces:**
- Produces: `compute_entity_metrics()`, source-aware `resolve_overlaps()`, and `DictionaryRuleEntityDetector(..., enable_generic_regex=False)`.
- Consumes: token offsets, raw text, BIO predictions, and detector evidence.

- [ ] **Step 1: Write failing tests proving symptom/patient regexes are disabled by default**

```python
def test_generic_symptom_and_patient_regex_are_not_primary_detectors():
    detector = DictionaryRuleEntityDetector([], [])
    assert detector.detect("Nam 45 tuổi, sốt và ho") == []
```

- [ ] **Step 2: Verify RED, then add an explicit fallback flag and lab parser separation**

```python
def __init__(..., enable_generic_regex: bool = False):
    self.enable_generic_regex = enable_generic_regex

def detect(self, raw_text):
    entities = self._dictionary_entities(raw_text)
    if self.enable_generic_regex:
        entities.extend(self._fallback_regex_entities(raw_text))
    return resolve_overlaps(entities, raw_text)
```

- [ ] **Step 3: Write failing exact-span entity metric tests**

```python
def test_entity_metrics_penalize_boundary_and_type_errors():
    metrics = compute_entity_metrics(PREDICTED, EXPECTED)
    assert metrics["exact_f1"] == 0.5
    assert metrics["overlap_f1"] == 1.0
```

- [ ] **Step 4: Implement entity-level metrics and source-aware overlap ranking**

```python
SOURCE_PRIORITY = {"ontology_exact_phrase": 3, "transformer_bio": 2, "structured_lab_parser": 1}

def compute_entity_metrics(predicted_by_document, expected_by_document):
    return exact_and_overlap_micro_scores(predicted_by_document, expected_by_document)
```

- [ ] **Step 5: Run focused NER/training tests**

Run: `python -m pytest v2/tests/test_ner_policy.py v2/tests/test_entity_metrics.py v2/tests/test_training_helpers.py -q -p no:cacheprovider`

Expected: all tests pass without loading a remote transformer checkpoint.

---

### Task 3: Candidate policy and medication query parsing

**Files:**
- Modify: `v2/clinical_nlp_lab/retrieval.py`
- Modify: `v2/clinical_nlp_lab/linking.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py`
- Test: `v2/tests/test_candidate_policy.py`

**Interfaces:**
- Produces: `CandidatePolicy`, `apply_candidate_policy()`, and medication-head retrieval.
- Consumes: ranked candidate dictionaries, configured minimum score, minimum margin, and output K.

- [ ] **Step 1: Write failing tests for output truncation, abstention, and medication-head query**

```python
def test_candidate_output_k_is_enforced_without_qwen():
    policy = CandidatePolicy(min_score=0.0, min_margin=0.0, output_k=1)
    assert apply_candidate_policy(RANKED, policy) == ["A"]

def test_candidate_policy_abstains_below_threshold():
    policy = CandidatePolicy(min_score=0.5, min_margin=0.0, output_k=1)
    assert apply_candidate_policy([{"candidate_id": "A", "score": 0.2}], policy) == []

def test_drug_retrieval_uses_mention_head():
    linker.retrieve("DRUG", "metformin 500 mg uống ngày 2 lần", mention_head="metformin")
    assert INDEX.last_query == "metformin"
```

- [ ] **Step 2: Verify RED and implement candidate policy**

```python
@dataclass(frozen=True, slots=True)
class CandidatePolicy:
    min_score: float
    min_margin: float
    output_k: int

def apply_candidate_policy(ranked, policy):
    if not ranked or ranked[0]["score"] < policy.min_score:
        return []
    if len(ranked) > 1 and ranked[0]["score"] - ranked[1]["score"] < policy.min_margin:
        return []
    return [item["candidate_id"] for item in ranked[:policy.output_k]]
```

- [ ] **Step 3: Wire policy into deterministic and post-Qwen output paths**

```python
entity.candidates = apply_candidate_policy(ranked, self.candidate_policy)
# after optional reranking
entity.candidates = entity.candidates[: self.candidate_policy.output_k]
```

- [ ] **Step 4: Run focused retrieval/pipeline tests**

Run: `python -m pytest v2/tests/test_candidate_policy.py v2/tests/test_retrieval_resources.py -q -p no:cacheprovider`

Expected: all tests pass and no model download occurs.

---

### Task 4: Assertion scope and optional-Qwen fallback

**Files:**
- Modify: `v2/clinical_nlp_lab/assertions.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py`
- Test: `v2/tests/test_assertion_scope.py`
- Test: `v2/tests/test_pipeline_qwen_fallback.py`

**Interfaces:**
- Produces: clause-scoped deterministic assertions and non-fatal Qwen fallback diagnostics.
- Consumes: raw text, entity span, detected sections, and optional LLM outputs.

- [ ] **Step 1: Write failing assertion scope regression tests**

```python
@pytest.mark.parametrize("text,entity_text,unexpected", [
    ("Người nhà đưa bệnh nhân vào viện. Bệnh nhân khó thở.", "khó thở", "isFamily"),
    ("Đái tháo đường mạn tính đang điều trị.", "Đái tháo đường", "isHistorical"),
    ("Theo dõi tại khoa. Chẩn đoán tăng huyết áp.", "tăng huyết áp", "isPossible"),
])
def test_broad_cues_do_not_leak(text, entity_text, unexpected):
    assert unexpected not in predict_for_mention(text, entity_text)
```

- [ ] **Step 2: Implement sentence/clause-scoped cues and semantic family patterns**

```python
def assertion_context(raw_text, start, end):
    return containing_clause(raw_text, start, end)

FAMILY_RELATION_PATTERN = re.compile(r"(?iu)(?:mẹ|cha|bố|anh|chị|em)\s+(?:của\s+)?bệnh\s+nhân|tiền\s+sử\s+gia\s+đình")
```

- [ ] **Step 3: Write failing Qwen-exception fallback test**

```python
def test_qwen_failure_preserves_deterministic_output(monkeypatch, fixture_runtime):
    monkeypatch.setattr(ClinicalLLMReranker, "rerank_batch", raising_cuda_error)
    summary = run_inference(enable_qwen_reranker=True, **fixture_runtime)
    assert summary["llm_reranker_enabled"] is False
    assert summary["llm_fallback_reason"].startswith("RuntimeError")
    assert Path(summary["zip_path"]).is_file()
```

- [ ] **Step 4: Replace hard failure with cleanup plus deterministic fallback**

```python
except Exception as exc:
    llm_fallback_reason = f"{type(exc).__name__}: {exc}"
    llm_reranker_enabled = False
    llm_assertion_enabled = False
finally:
    if reranker is not None:
        reranker.destroy()
```

- [ ] **Step 5: Run assertion and fallback tests**

Run: `python -m pytest v2/tests/test_assertion_scope.py v2/tests/test_pipeline_qwen_fallback.py v2/tests/test_assertion_vllm_compat.py v2/tests/test_reranker_compatibility.py -q -p no:cacheprovider`

Expected: all tests pass.

---

### Task 5: Training evidence, checkpoint reload, and artifact packaging

**Files:**
- Modify: `v2/clinical_nlp_lab/training.py`
- Modify: `v2/scripts/train_ner_subprocess.py`
- Create: `v2/clinical_nlp_lab/run_artifacts.py`
- Test: `v2/tests/test_training_artifacts.py`

**Interfaces:**
- Produces: `training_result.json`, `evaluation_report.json`, `split_manifest.json`, checkpoint inventory, reload smoke-test result, and `trained_ner_artifacts.zip`.
- Consumes: grouped split, Trainer state/history, best checkpoint, and tokenizer.

- [ ] **Step 1: Write failing artifact-contract tests**

```python
def test_training_bundle_requires_reloadable_checkpoint(tmp_path):
    bundle = package_training_artifacts(tmp_path, tmp_path / "weights.zip")
    assert set(bundle.required_members) <= set(bundle.zip_members)

def test_missing_model_safetensors_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="model.safetensors"):
        validate_checkpoint_inventory(tmp_path)
```

- [ ] **Step 2: Implement checkpoint inventory and ZIP validation**

```python
REQUIRED_CHECKPOINT_FILES = {"config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"}

def validate_checkpoint_inventory(model_dir):
    missing = REQUIRED_CHECKPOINT_FILES - {p.name for p in Path(model_dir).iterdir()}
    if missing:
        raise ValueError(f"Missing checkpoint files: {sorted(missing)}")
```

- [ ] **Step 3: Record grouped split, entity metrics, training history, and reload result**

```python
training_result = {
    "trained": True,
    "best_metric": trainer.state.best_metric,
    "best_checkpoint": trainer.state.best_model_checkpoint,
    "reload_smoke_test": reload_smoke_test(output_path),
    "checkpoint_inventory": validate_checkpoint_inventory(output_path),
}
```

- [ ] **Step 4: Run training artifact tests**

Run: `python -m pytest v2/tests/test_training_artifacts.py v2/tests/test_training_helpers.py -q -p no:cacheprovider`

Expected: all tests pass using local fixture files only.

---

### Task 6: Canonical Kaggle notebook and one-click flow

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py`
- Regenerate: `v2/medical_information_extraction_kaggle.ipynb`
- Modify: `v2/KAGGLE_RUNBOOK.md`
- Test: `v2/tests/test_kaggle_training_notebook.py`

**Interfaces:**
- Produces: canonical notebook code cells and all required `/kaggle/working` artifacts.
- Consumes: validated data, training subprocess, inference API, and artifact-packaging API.

- [ ] **Step 1: Write failing notebook contract tests**

```python
def test_training_notebook_requires_all_success_artifacts():
    source = notebook_source()
    for name in REQUIRED_OUTPUTS:
        assert f'assert_output("{name}")' in source

def test_generated_notebook_matches_builder():
    assert NOTEBOOK.read_text(encoding="utf-8") == render_notebook(build_notebook())
```

- [ ] **Step 2: Update builder cells for grouped split, validation, reload, and packaging**

```python
REQUIRED_OUTPUTS = (
    "output.zip", "trained_ner_artifacts.zip", "training_result.json",
    "evaluation_report.json", "split_manifest.json", "run_manifest.json",
    "diagnostics/run_summary.json",
)
```

- [ ] **Step 3: Regenerate notebook with the maintained builder**

Run: `python v2/tools/build_kaggle_notebook.py --output v2/medical_information_extraction_kaggle.ipynb`

Expected: command exits 0 and reports the output path.

- [ ] **Step 4: Parse every code cell and run notebook tests**

Run: `python -m pytest v2/tests/test_kaggle_training_notebook.py v2/tests/test_kaggle_observability.py -q -p no:cacheprovider`

Expected: all tests pass; notebook cells parse without execution output.

---

### Task 7: End-to-end verifier and completion evidence

**Files:**
- Create: `v2/tools/verify_kaggle_run.py`
- Create: `v2/tests/test_end_to_end_fixture.py`
- Modify: `v2/README.md`

**Interfaces:**
- Produces: a machine-readable audit of downloaded Kaggle artifacts.
- Consumes: run directory or results ZIP containing manifests, checkpoint, diagnostics, and submission ZIP.

- [ ] **Step 1: Write failing verifier tests for valid and corrupted bundles**

```python
def test_verifier_accepts_complete_run(complete_run):
    assert verify_kaggle_run(complete_run)["is_valid"] is True

def test_verifier_rejects_crc_or_missing_weight(corrupt_run):
    report = verify_kaggle_run(corrupt_run)
    assert report["is_valid"] is False
```

- [ ] **Step 2: Implement artifact, manifest, output, schema, offset, and CRC verification**

```python
def verify_kaggle_run(run_root):
    checks = [check_required_files(...), check_checkpoint(...), check_output_zip(...), check_manifests(...)]
    return {"is_valid": all(item["passed"] for item in checks), "checks": checks}
```

- [ ] **Step 3: Run full local verification suite with durations**

Run: `python -m pytest v2/tests tests -q --durations=10 -p no:cacheprovider`

Expected: all tests pass and no test hangs on network/model download.

- [ ] **Step 4: Run dataset and notebook verification commands**

Run: `python scripts/validate_synthetic_train_v2.py`

Run: `python v2/tools/build_kaggle_notebook.py --output v2/_generated_check.ipynb`

Expected: dataset reports zero errors; notebook builds and all code cells parse.

- [ ] **Step 5: Inspect diff and record unverified external gate honestly**

Run: `git diff --check`

Run: `git status --short`

Expected: no whitespace errors and only task-owned files are modified. A real Kaggle GPU run remains an external acceptance gate until its downloaded artifacts pass `verify_kaggle_run.py`.
