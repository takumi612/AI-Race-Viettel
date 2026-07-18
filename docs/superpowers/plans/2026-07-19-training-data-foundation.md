# Training Data Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the validated, fingerprinted, leakage-resistant data foundation and task-specific seed datasets required by the NER, BGE-M3, and Qwen training pipelines.

**Architecture:** Source input/GT pairs are converted into immutable canonical records, validated against `metadata.db`, assigned deterministic synthetic/trusted/holdout splits, and projected into NER, embedding, and reranker seed JSONL files. A single atomic CLI build writes manifests and refuses failed synthetic datasets, pseudo-label training, holdout leakage, or mismatched spans.

**Tech Stack:** Python 3.10+, stdlib dataclasses/json/hashlib/sqlite3/pathlib, PyYAML 6, pytest 8.

## Global Constraints

- Train target is a free Google Colab T4 16 GB runtime; no GPU dependency is required for this phase.
- `data/input/` is never read as training data.
- Pseudo-label IDs 1–100 are excluded from gradient-training outputs.
- Trusted IDs 101–180 are available for five-fold refinement.
- Holdout IDs 181–200 are emitted only as holdout and never as train/validation.
- A source directory ending in `.failed-validation` is always rejected.
- Synthetic input must contain exactly 2,000 text/GT pairs before a production build.
- Unicode spans use half-open `[start, end)` and must match the source text exactly.
- All generated artifacts include source, database, split, config, and Git fingerprints.
- Generated outputs are written atomically and are ignored by Git.

---

## Planned File Structure

```text
src/training/
├── __init__.py                 # Public data-foundation exports
├── contracts.py                # Canonical and task-seed dataclasses
├── fingerprints.py             # Stable SHA-256 helpers
├── sources.py                  # Viettel input/GT source reader
├── validation.py               # Schema, span, code, and source gates
├── splits.py                   # Synthetic split and trusted fold builder
├── projections.py              # NER/embedding/reranker seed projections
└── build_datasets.py           # Atomic orchestration and CLI
configs/training/data.yaml      # Default paths, IDs, counts, and seed
requirements-train.txt          # Training-only dependencies
tests/training/
├── test_contracts.py
├── test_fingerprints.py
├── test_sources_validation.py
├── test_splits.py
├── test_projections.py
└── test_build_datasets.py
```

## Task 1: Canonical contracts and fingerprints

**Files:**

- Create: `src/training/__init__.py`
- Create: `src/training/contracts.py`
- Create: `src/training/fingerprints.py`
- Create: `tests/training/test_contracts.py`
- Create: `tests/training/test_fingerprints.py`

**Interfaces:**

- Produces: `CanonicalEntity.from_mapping(mapping, text) -> CanonicalEntity`
- Produces: `CanonicalRecord.to_mapping() -> dict[str, object]`
- Produces: `stable_json_sha256(value) -> str`
- Produces: `sha256_file(path) -> str`
- Produces: `fingerprint_files(paths, root) -> str`

- [ ] **Step 1: Write failing contract tests**

```python
def test_entity_requires_exact_half_open_span():
    text = "Dùng metformin hằng ngày"
    entity = CanonicalEntity.from_mapping(
        {
            "text": "metformin",
            "type": "THUỐC",
            "position": [5, 14],
            "assertions": [],
            "candidates": ["6809"],
        },
        text,
    )
    assert entity.start == 5
    assert entity.end == 14
    assert entity.codes == ("6809",)


def test_entity_rejects_mismatched_span():
    with pytest.raises(ValueError, match="span text mismatch"):
        CanonicalEntity.from_mapping(
            {"text": "metformin", "type": "THUỐC", "position": [0, 9]},
            "Dùng metformin",
        )
```

- [ ] **Step 2: Run contract tests and verify failure**

Run:

```bash
python -m pytest tests/training/test_contracts.py -q
```

Expected: collection fails because `src.training.contracts` does not exist.

- [ ] **Step 3: Implement immutable contracts**

```python
ALLOWED_ENTITY_TYPES = frozenset({
    "CHẨN_ĐOÁN",
    "THUỐC",
    "TRIỆU_CHỨNG",
    "TÊN_XÉT_NGHIỆM",
    "KẾT_QUẢ_XÉT_NGHIỆM",
})
ALLOWED_ASSERTIONS = frozenset({"isNegated", "isHistorical", "isFamily"})


@dataclass(frozen=True)
class CanonicalEntity:
    text: str
    entity_type: str
    start: int
    end: int
    assertions: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object], source_text: str):
        entity_text = mapping.get("text")
        entity_type = mapping.get("type")
        position = mapping.get("position")
        if not isinstance(entity_text, str) or not entity_text:
            raise ValueError("entity text must be a non-empty string")
        if entity_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"unsupported entity type: {entity_type!r}")
        if (
            not isinstance(position, list)
            or len(position) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in position)
        ):
            raise ValueError("position must contain two integer offsets")
        start, end = position
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError("position is outside source text")
        if source_text[start:end] != entity_text:
            raise ValueError("span text mismatch")
        assertions_raw = mapping.get("assertions", [])
        codes_raw = mapping.get("candidates", [])
        if not isinstance(assertions_raw, list) or any(
            value not in ALLOWED_ASSERTIONS for value in assertions_raw
        ):
            raise ValueError("invalid assertions")
        if not isinstance(codes_raw, list) or any(
            isinstance(value, bool)
            or not isinstance(value, (str, int))
            or not str(value).strip()
            for value in codes_raw
        ):
            raise ValueError("invalid candidates")
        return cls(
            text=entity_text,
            entity_type=entity_type,
            start=start,
            end=end,
            assertions=tuple(sorted(set(assertions_raw))),
            codes=tuple(dict.fromkeys(str(value).strip() for value in codes_raw)),
        )
```

`CanonicalRecord` contains `record_id`, `source`, `trust_tier`, `text`, `entities`, `sha256`, `split_group`, and immutable metadata. `to_mapping()` serializes entity type back to the competition field name `type` and codes to `candidates`.

- [ ] **Step 4: Write and run fingerprint tests**

```python
def test_stable_json_hash_ignores_mapping_order():
    assert stable_json_sha256({"a": 1, "b": 2}) == stable_json_sha256(
        {"b": 2, "a": 1}
    )


def test_file_set_hash_is_root_relative(tmp_path):
    first = tmp_path / "a.txt"
    first.write_text("hello", encoding="utf-8")
    assert fingerprint_files([first], tmp_path) == fingerprint_files(
        [first], tmp_path
    )
```

Run:

```bash
python -m pytest tests/training/test_contracts.py tests/training/test_fingerprints.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/__init__.py src/training/contracts.py src/training/fingerprints.py tests/training/test_contracts.py tests/training/test_fingerprints.py
git commit -m "feat: add canonical training data contracts"
```

## Task 2: Source reader and validation gates

**Files:**

- Create: `src/training/sources.py`
- Create: `src/training/validation.py`
- Create: `tests/training/test_sources_validation.py`

**Interfaces:**

- Consumes: `CanonicalEntity`, `CanonicalRecord`, `sha256_file`
- Produces: `SourceSpec`
- Produces: `load_source_records(spec) -> tuple[CanonicalRecord, ...]`
- Produces: `ValidationFinding`
- Produces: `validate_records(records, db_path) -> tuple[ValidationFinding, ...]`
- Produces: `require_valid_source(spec, records, db_path) -> None`

- [ ] **Step 1: Write failing source-gate tests**

```python
def test_failed_validation_directory_is_never_read(tmp_path):
    root = tmp_path / "synthetic.failed-validation"
    (root / "input").mkdir(parents=True)
    (root / "gt").mkdir()
    with pytest.raises(ValueError, match="failed-validation"):
        load_source_records(
            SourceSpec(
                name="synthetic",
                root=root,
                trust_tier="synthetic_validated",
                expected_ids=("0001",),
            )
        )


def test_source_rejects_missing_pair(tmp_path):
    root = make_source(tmp_path, {"0001": ("text", [])})
    (root / "gt" / "0001.json").unlink()
    with pytest.raises(ValueError, match="missing GT"):
        load_source_records(source_spec(root, ("0001",)))
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/training/test_sources_validation.py -q
```

Expected: collection fails because source modules do not exist.

- [ ] **Step 3: Implement paired source loading**

`SourceSpec` defines:

```python
@dataclass(frozen=True)
class SourceSpec:
    name: str
    root: Path
    trust_tier: str
    expected_ids: tuple[str, ...]
    manifest_path: Path | None = None
```

`load_source_records`:

- Rejects root names ending in `.failed-validation`.
- Requires `input/` and `gt/`.
- Requires exactly one `.txt` and one `.json` for every expected ID.
- Rejects unexpected IDs for synthetic production builds.
- Loads optional manifest JSONL metadata keyed by `record_id`.
- Builds `split_group` from manifest `profile_id`, falling back to record ID.
- Preserves missing `assertions` and `candidates` as empty tuples.

- [ ] **Step 4: Add SQLite namespace validation tests**

```python
def test_validator_checks_code_namespace(tmp_path):
    db = build_metadata_db(tmp_path, icd=("I10",), rxnorm=("6809",))
    record = canonical_record(
        entities=[
            entity("tăng huyết áp", "CHẨN_ĐOÁN", codes=("6809",)),
            entity("metformin", "THUỐC", codes=("I10",)),
        ]
    )
    findings = validate_records((record,), db)
    assert {finding.code for finding in findings} == {
        "unknown_icd10_code",
        "unknown_rxnorm_code",
    }
```

- [ ] **Step 5: Implement validation gates and run tests**

Validation covers:

- Duplicate record IDs.
- Record text hash mismatch.
- Empty text.
- Span overlap/crossing.
- Exact span match.
- Allowed entity/assertion values.
- ICD candidates only for `CHẨN_ĐOÁN`.
- RxNorm candidates only for `THUỐC`.
- Code existence in read-only SQLite.
- Validation report must exist and contain `"passed": true` for synthetic sources.

Run:

```bash
python -m pytest tests/training/test_sources_validation.py -q
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/training/sources.py src/training/validation.py tests/training/test_sources_validation.py
git commit -m "feat: validate training data sources"
```

## Task 3: Deterministic synthetic split and trusted folds

**Files:**

- Create: `src/training/splits.py`
- Create: `tests/training/test_splits.py`

**Interfaces:**

- Consumes: `CanonicalRecord`
- Produces: `SplitAssignment`
- Produces: `build_synthetic_split(records, validation_size, seed) -> tuple[SplitAssignment, ...]`
- Produces: `build_trusted_folds(records, folds, seed) -> tuple[SplitAssignment, ...]`
- Produces: `assert_no_split_leakage(records, assignments) -> None`

- [ ] **Step 1: Write failing deterministic/group tests**

```python
def test_synthetic_split_is_exact_deterministic_and_group_safe():
    records = tuple(
        canonical_record(
            record_id=f"{index:04d}",
            split_group=f"group-{index // 2}",
            metadata={"family": "rare" if index < 6 else "rich"},
        )
        for index in range(10)
    )
    first = build_synthetic_split(records, validation_size=4, seed=7)
    second = build_synthetic_split(tuple(reversed(records)), validation_size=4, seed=7)
    assert first == second
    assert sum(item.split == "synthetic_validation" for item in first) == 4
    assert_no_split_leakage(records, first)


def test_trusted_folds_exclude_pseudo_and_holdout():
    records = tuple(canonical_record(record_id=str(index)) for index in range(1, 201))
    assignments = build_trusted_folds(records, folds=5, seed=11)
    assert {item.record_id for item in assignments} == {
        str(index) for index in range(101, 181)
    }
    assert sorted(Counter(item.fold for item in assignments).values()) == [16] * 5
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/training/test_splits.py -q
```

Expected: collection fails because `src.training.splits` does not exist.

- [ ] **Step 3: Implement split assignment**

```python
@dataclass(frozen=True, order=True)
class SplitAssignment:
    record_id: str
    split: str
    fold: int | None = None
```

Synthetic algorithm:

1. Sort records and groups independent of input order.
2. Build group-level family/type/assertion signatures.
3. Order groups by SHA-256 of `seed + group_id`.
4. Select whole groups for validation while preserving target coverage.
5. Use singleton groups to reach exact validation size.
6. Raise a clear error when exact size is impossible without splitting a group.

Trusted folds:

1. Select only numeric IDs 101–180.
2. Stratify by entity-type/assertion signature.
3. Deterministically round-robin records into five folds.
4. Require 16 records per fold.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/training/test_splits.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/splits.py tests/training/test_splits.py
git commit -m "feat: add deterministic training splits"
```

## Task 4: Task-specific seed projections

**Files:**

- Create: `src/training/projections.py`
- Create: `tests/training/test_projections.py`

**Interfaces:**

- Consumes: canonical records and split assignments
- Produces: `project_ner_records(records: Sequence[CanonicalRecord], assignments: Mapping[str, SplitAssignment]) -> tuple[dict[str, object], ...]`
- Produces: `project_embedding_seeds(records: Sequence[CanonicalRecord], assignments: Mapping[str, SplitAssignment]) -> tuple[dict[str, object], ...]`
- Produces: `project_reranker_seeds(records: Sequence[CanonicalRecord], assignments: Mapping[str, SplitAssignment]) -> tuple[dict[str, object], ...]`

- [ ] **Step 1: Write failing projection tests**

```python
def test_ner_projection_keeps_all_entities_and_offsets():
    record = canonical_record(
        text="Sốt, dùng metformin.",
        entities=(
            entity("Sốt", "TRIỆU_CHỨNG", 0, 3),
            entity("metformin", "THUỐC", 10, 19, codes=("6809",)),
        ),
    )
    projected = project_ner_records((record,), assignments_for(record))
    assert projected[0]["entities"][1]["position"] == [10, 19]


def test_embedding_projection_keeps_only_coded_diagnosis_and_drug():
    projected = project_embedding_seeds((record_with_coded_and_uncoded_entities(),), assignments)
    assert {item["positive_codes"][0] for item in projected} == {"I10", "6809"}


def test_reranker_seed_has_no_fabricated_candidate_pool():
    seed = project_reranker_seeds((coded_record(),), assignments)[0]
    assert seed["ground_truth_codes"] == ["I10"]
    assert "candidates" not in seed
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/training/test_projections.py -q
```

Expected: collection fails because `src.training.projections` does not exist.

- [ ] **Step 3: Implement projections**

NER output:

```json
{
  "record_id":"synthetic-0001",
  "text":"Tăng huyết áp.",
  "entities":[{
    "text":"Tăng huyết áp",
    "type":"CHẨN_ĐOÁN",
    "position":[0,14],
    "assertions":[],
    "candidates":["I10"]
  }],
  "split":"synthetic_train"
}
```

Embedding seed:

```json
{
  "example_id":"record:entity-index",
  "record_id":"synthetic-0001",
  "query":"Tăng huyết áp",
  "context":"Tăng huyết áp.",
  "entity_type":"CHẨN_ĐOÁN",
  "positive_codes":["I10"],
  "split":"synthetic_train"
}
```

Reranker seed:

```json
{
  "example_id":"record:entity-index",
  "record_id":"synthetic-0001",
  "context":"Tăng huyết áp.",
  "entity_text":"Tăng huyết áp",
  "entity_type":"CHẨN_ĐOÁN",
  "assertions":[],
  "ground_truth_codes":["I10"],
  "split":"synthetic_train"
}
```

Reranker candidates are deliberately absent until Phase D freezes a retriever fingerprint.

- [ ] **Step 4: Run tests**

Run:

```bash
python -m pytest tests/training/test_projections.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/projections.py tests/training/test_projections.py
git commit -m "feat: project task training datasets"
```

## Task 5: Atomic build orchestration and CLI

**Files:**

- Create: `src/training/build_datasets.py`
- Create: `configs/training/data.yaml`
- Create: `requirements-train.txt`
- Create: `tests/training/test_build_datasets.py`
- Modify: `.gitignore`

**Interfaces:**

- Consumes: source, validation, split, projection, and fingerprint interfaces
- Produces: `DatasetBuildConfig.from_mapping(mapping) -> DatasetBuildConfig`
- Produces: `build_training_datasets(config) -> Path`
- Produces CLI: `python -m src.training.build_datasets --config PATH`

- [ ] **Step 1: Write failing atomic-build tests**

```python
def test_build_refuses_failed_synthetic_and_preserves_existing_output(tmp_path):
    output = tmp_path / "training"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    config = build_config(
        synthetic_root=tmp_path / "synthetic.failed-validation",
        output_root=output,
    )
    with pytest.raises(ValueError, match="failed-validation"):
        build_training_datasets(config)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_build_writes_manifest_and_task_jsonl(valid_build_config):
    output = build_training_datasets(valid_build_config)
    assert (output / "manifests" / "build.json").is_file()
    assert (output / "splits" / "assignments.jsonl").is_file()
    assert (output / "ner" / "records.jsonl").is_file()
    assert (output / "embedding" / "seeds.jsonl").is_file()
    assert (output / "reranker" / "seeds.jsonl").is_file()
```

- [ ] **Step 2: Run and verify failure**

Run:

```bash
python -m pytest tests/training/test_build_datasets.py -q
```

Expected: collection fails because build orchestration does not exist.

- [ ] **Step 3: Implement configuration and atomic build**

Default YAML:

```yaml
schema_version: 1
seed: 20260719
synthetic:
  root: data/synthetic_train_v1
  expected_count: 2000
  validation_size: 300
trusted:
  root: data/dev
  first_id: 101
  last_id: 180
  folds: 5
holdout:
  root: data/dev
  first_id: 181
  last_id: 200
database: data/kb/metadata.db
output: data/training
```

Atomic algorithm:

1. Resolve project-relative paths.
2. Reject machine-specific paths in config.
3. Load and validate all sources before creating output.
4. Write to `data/.training-build-<uuid>`.
5. Write canonical, split, NER, embedding, reranker, and manifest JSONL.
6. Re-open generated files and verify counts/fingerprints.
7. Refuse to replace an existing output unless `--replace` is explicit.
8. Rename the validated temporary directory to the final output.
9. On failure, preserve the previous final output and remove only the exact temporary directory.

- [ ] **Step 4: Add CLI and dependency**

`requirements-train.txt`:

```text
-r requirements.txt
PyYAML>=6,<7
```

CLI arguments:

```text
--config configs/training/data.yaml
--project-root PATH
--replace
--allow-non-production-count
```

`--allow-non-production-count` exists only for fixtures/smoke builds and is recorded in the manifest. It cannot change trusted or holdout ID boundaries.

- [ ] **Step 5: Ignore generated training outputs**

Append:

```gitignore
data/training/
data/.training-build-*/
```

- [ ] **Step 6: Run CLI tests**

Run:

```bash
python -m pytest tests/training/test_build_datasets.py -q
python -m src.training.build_datasets --help
```

Expected: tests pass and CLI exits zero with documented arguments.

- [ ] **Step 7: Commit**

```bash
git add src/training/build_datasets.py configs/training/data.yaml requirements-train.txt tests/training/test_build_datasets.py .gitignore
git commit -m "feat: build training datasets atomically"
```

## Task 6: Full verification and operator documentation

**Files:**

- Create: `docs/training/DATA_FOUNDATION.md`
- Modify: `README_TASKS_7_10.md`

**Interfaces:**

- Documents the exact local and Colab build commands.
- Documents source protection and expected output manifests.

- [ ] **Step 1: Write operator guide**

The guide includes:

```bash
pip install -r requirements-train.txt
python -m src.training.build_datasets \
  --config configs/training/data.yaml \
  --project-root .
```

Colab paths are supplied by a copied config that points to mounted Drive symlinks; business logic remains unchanged.

- [ ] **Step 2: Run focused training-data tests**

Run:

```bash
python -m pytest tests/training -q
```

Expected: all Phase A tests pass.

- [ ] **Step 3: Run project regression suite**

Run:

```bash
python -m pytest -q
python src/metrics.py test
```

Expected: all existing and new tests pass.

- [ ] **Step 4: Audit paths and repository state**

Run:

```bash
python scripts/audit_overrides.py --scan-paths src scripts
python scripts/audit_overrides.py \
  --db data/kb/metadata.db \
  --overrides src/resources/verified_overrides.json
git diff --check
git status --short
```

Expected: no machine-specific production path finding, override audit passes,
no whitespace errors, and only intended documentation changes before the final
commit. Tests are excluded from the path scan because absolute-path rejection
fixtures intentionally contain machine-specific example strings.

- [ ] **Step 5: Commit**

```bash
git add docs/training/DATA_FOUNDATION.md README_TASKS_7_10.md
git commit -m "docs: document training data foundation"
```
