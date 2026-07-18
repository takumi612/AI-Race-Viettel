# Precision-First Hybrid Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current rule-heavy baseline with a precision-first hybrid pipeline that uses trusted evaluation, adaptive clinical chunking, contextual NER, normalized BM25/semantic fusion, calibrated candidate selection, and submission-safe output.

**Architecture:** Keep the existing modular pipeline and SQLite/FAISS assets, but introduce small testable components around it. Detection proposes mentions; context resolves type or rejects; BM25 and semantic retrieval expose comparable scores; a selector emits only the smallest justified candidate set. Structural validation is mandatory before packaging.

**Tech Stack:** Python 3.10+, pytest, SQLite, bm25s, NumPy, FAISS, PyTorch, sentence-transformers, existing ICD-10/RxNorm assets.

## Global Constraints

- Do not use external commercial APIs during inference.
- Self-hosted models must remain at or below 9B parameters.
- `data/input/` contains the 100 public inference files used only to produce the submission; it has no trusted labels for tuning.
- `data/dev/input/` contains the matching development texts for pseudo-label IDs 1–100 and supplied-label IDs 101–200.
- `data/dev/gt/1.json` through `100.json` are pseudo-labels and must not tune metrics or thresholds.
- A pseudo-label may become a regression oracle only after manual verification recorded with `verified_by`, `verified_at`, and `source_file`; structural invariants over public input do not require label trust.
- `data/dev/gt/101.json` through `180.json` form the development pool; `181.json` through `200.json` are the untouched final holdout.
- NER threshold selection uses exact-span exact-type F0.5, with precision weighted more heavily than recall.
- Retrieval fusion must be `alpha * BM25 + (1 - alpha) * semantic`, with weights summing to exactly 1.
- The default `alpha` is `0.75`; the only tuning grid is `0.60, 0.70, 0.75, 0.80, 0.90`.
- No rule may depend on a public input file ID or copy a public output label.
- Every entity must satisfy `document[start:end] == entity["text"]`.
- Production code changes follow RED -> GREEN -> REFACTOR; do not add behavior without first observing the relevant test fail.

---

## File Structure

**Create**

- `requirements.txt` — reproducible runtime and test dependencies.
- `pytest.ini` — test discovery and UTF-8 defaults.
- `src/evaluation/__init__.py` — evaluation package marker.
- `src/evaluation/precision_metrics.py` — exact entity and F-beta metrics.
- `src/evaluation/trusted_split.py` — immutable development/holdout IDs.
- `src/evaluation/benchmark.py` — cross-validation and locked-holdout CLI.
- `src/config.py` — validated dataclass configuration.
- `src/chunking/__init__.py` — chunking package marker.
- `src/chunking/clinical_chunker.py` — offset-safe adaptive section chunking.
- `src/ner/types.py` — mention/type decision contracts.
- `src/ner/type_resolver.py` — contextual type scorer and reject gate.
- `src/ner/lexicon_loader.py` — validated loader for provenance-bearing clinical terms.
- `src/retrieval/types.py` — scored retrieval contracts.
- `src/retrieval/score_fusion.py` — normalization and weighted fusion.
- `src/retrieval/candidate_selector.py` — Top-1/Top-2/reject policy.
- `src/validation/override_validator.py` — hardcode and KB integrity audit.
- `src/validation/submission.py` — output/schema/offset/code validator and packager.
- `src/resources/section_patterns.json` — section headings with provenance.
- `src/resources/type_rules.json` — contextual type features and generic-term rules.
- `src/resources/clinical_lexicon.json` — medication, diagnosis, symptom, and laboratory terms with source/status metadata.
- `src/resources/assertion_rules.json` — assertion cues, scope boundaries, section priors, and provenance.
- `src/resources/clinical_validation_rules.json` — dose-form and demographic validation rules with provenance.
- `src/resources/verified_overrides.json` — provenance-bearing override schema.
- `scripts/audit_overrides.py` — CLI for override integrity.
- `scripts/package_submission.py` — validate then create the only submission zip.
- `tests/conftest.py` — shared paths and small fixtures.
- `tests/test_precision_metrics.py`
- `tests/test_config.py`
- `tests/test_submission.py`
- `tests/test_clinical_chunker.py`
- `tests/test_type_resolver.py`
- `tests/test_assertion_analyzer.py`
- `tests/test_score_fusion.py`
- `tests/test_candidate_selector.py`
- `tests/test_override_validator.py`
- `tests/test_pipeline_regressions.py`

**Modify**

- `src/ner/extractor.py` — return mention candidates and resolve types contextually.
- `src/assertion/rule_based.py` — section-aware assertion scope.
- `src/retrieval/bm25_retriever.py` — expose raw scores.
- `src/retrieval/hybrid_retriever.py` — normalized fusion and scored retrieval.
- `src/validation/clinical_validator.py` — candidate predicate; remove synthetic entities.
- `src/ranking/llm_reranker.py` — require a subset, not a reorder-only response.
- `src/pipeline/main.py` — integrate components and fail-safe output.
- `src/evaluate.py` — print precision-first metrics and trusted split labels.
- `src/utils/setup_rxnorm.py` — derive default archive paths from project configuration.
- `src/retrieval/eval_recall.py` — require explicit CLI data paths; remove machine-specific defaults.
- `scripts/create_gt_all.py` — require explicit input/output paths and label generated annotations as pseudo-GT.
- `scripts/aggregate_data.py` — require explicit source/output paths.
- `README.md` — reproducible setup, benchmark, validation, and packaging commands.

---

### Task 1: Establish Test Infrastructure and Trusted Metrics

**Files:**

- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `src/evaluation/__init__.py`
- Create: `src/evaluation/precision_metrics.py`
- Create: `src/evaluation/trusted_split.py`
- Create: `tests/conftest.py`
- Create: `tests/test_precision_metrics.py`

**Interfaces:**

- Produces: `fbeta(precision: float, recall: float, beta: float = 0.5) -> float`
- Produces: `score_exact_entities(gt: list[dict], pred: list[dict], beta: float = 0.5) -> EntityMetrics`
- Produces: `score_entities_by_type(gt, pred, beta=0.5) -> dict[str, EntityMetrics]`
- Produces: `score_assertions(gt, pred, beta=0.5) -> AssertionMetrics`
- Produces: `score_candidate_sets(gt, pred, retrieved=None) -> CandidateMetrics`
- Produces: `development_ids() -> tuple[int, ...]` and `holdout_ids() -> tuple[int, ...]`

- [ ] **Step 1: Add dependency and pytest configuration**

Create `requirements.txt`:

```text
numpy>=1.26,<3
openpyxl>=3.1,<4
bm25s>=0.2,<0.3
faiss-cpu>=1.8,<2
torch>=2.2,<3
sentence-transformers>=3,<6
transformers>=4.44,<6
requests>=2.31,<3
pytest>=8,<9
```

Create `pytest.ini`:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
addopts = -ra
```

Create `tests/conftest.py` so later tasks do not depend on undeclared local paths:

```python
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def metadata_db(project_root: Path) -> Path:
    path = project_root / "data" / "kb" / "metadata.db"
    if not path.exists():
        pytest.skip(f"metadata DB is unavailable: {path}")
    return path
```

- [ ] **Step 2: Write failing precision metric tests**

```python
# tests/test_precision_metrics.py
from src.evaluation.precision_metrics import (
    fbeta,
    score_assertions,
    score_candidate_sets,
    score_entities_by_type,
    score_exact_entities,
)
from src.evaluation.trusted_split import development_ids, holdout_ids


def test_f05_weights_precision_more_than_recall():
    high_precision = fbeta(precision=0.9, recall=0.5, beta=0.5)
    high_recall = fbeta(precision=0.5, recall=0.9, beta=0.5)
    assert high_precision > high_recall


def test_exact_entity_metric_penalizes_wrong_type_twice():
    gt = [{"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]}]
    pred = [{"text": "ho", "type": "CHẨN_ĐOÁN", "position": [0, 2]}]
    result = score_exact_entities(gt, pred)
    assert (result.tp, result.fp, result.fn) == (0, 1, 1)
    assert result.fbeta == 0.0


def test_exact_entity_metric_penalizes_duplicate_predictions():
    entity = {"text": "ho", "type": "TRIỆU_CHỨNG", "position": [0, 2]}
    result = score_exact_entities([entity], [entity, dict(entity)])
    assert (result.tp, result.fp, result.fn) == (1, 1, 0)
    assert result.precision == 0.5


def test_trusted_split_is_disjoint_and_immutable():
    assert development_ids() == tuple(range(101, 181))
    assert holdout_ids() == tuple(range(181, 201))
    assert set(development_ids()).isdisjoint(holdout_ids())


def test_per_type_and_assertion_metrics_expose_precision_errors():
    gt = [{
        "text": "ho",
        "type": "TRIỆU_CHỨNG",
        "position": [0, 2],
        "assertions": ["isNegated"],
    }]
    pred = [{
        "text": "ho",
        "type": "TRIỆU_CHỨNG",
        "position": [0, 2],
        "assertions": ["isHistorical"],
    }]
    assert score_entities_by_type(gt, pred)["TRIỆU_CHỨNG"].precision == 1.0
    assertions = score_assertions(gt, pred)
    assert assertions.by_label["isNegated"].fn == 1
    assert assertions.by_label["isHistorical"].fp == 1


def test_candidate_metrics_separate_selector_precision_from_retrieval_recall():
    gt = [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [0, 13], "candidates": ["I10"]}]
    pred = [{"text": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "position": [0, 13], "candidates": ["I10", "I11"]}]
    retrieved = {(0, 13, "CHẨN_ĐOÁN"): ["I10", "I11", "I12"]}
    metrics = score_candidate_sets(gt, pred, retrieved=retrieved)
    assert metrics.jaccard == 0.5
    assert metrics.precision == 0.5
    assert metrics.top1_hit_rate == 1.0
    assert metrics.recall_at_20 == 1.0
```

- [ ] **Step 3: Run tests and verify RED**

Run:

```powershell
python -m pytest tests/test_precision_metrics.py -v
```

Expected: collection fails because `src.evaluation.precision_metrics` and `trusted_split` do not exist.

- [ ] **Step 4: Implement the minimum metric contracts**

```python
# src/evaluation/precision_metrics.py
from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class EntityMetrics:
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    fbeta: float


def fbeta(precision: float, recall: float, beta: float = 0.5) -> float:
    if beta <= 0:
        raise ValueError("beta must be positive")
    beta_sq = beta * beta
    denominator = beta_sq * precision + recall
    if denominator == 0:
        return 0.0
    return (1 + beta_sq) * precision * recall / denominator


def _entity_key(entity: dict) -> tuple[int, int, str]:
    start, end = entity["position"]
    return int(start), int(end), str(entity["type"]).strip()


def score_exact_entities(gt: list[dict], pred: list[dict], beta: float = 0.5) -> EntityMetrics:
    gt_keys = Counter(_entity_key(entity) for entity in gt)
    pred_keys = Counter(_entity_key(entity) for entity in pred)
    tp = sum((gt_keys & pred_keys).values())
    fp = sum((pred_keys - gt_keys).values())
    fn = sum((gt_keys - pred_keys).values())
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    return EntityMetrics(tp, fp, fn, precision, recall, fbeta(precision, recall, beta))
```

Add immutable `AssertionMetrics` and `CandidateMetrics` dataclasses. `score_entities_by_type()` calls the same exact `(start, end, type)` logic per entity type. `score_assertions()` computes TP/FP/FN for each allowed assertion only on exact-span/exact-type matched entities, then macro-averages the three label F0.5 values. `score_candidate_sets()` aligns exact-span/exact-type code-bearing entities and reports mean Jaccard, micro candidate precision, Top-1 hit rate, and retrieval Recall@20 separately; an absent retrieved pool yields `recall_at_20=None` rather than a fabricated value.

The benchmark added in Task 9 must also aggregate exact-entity FP/FN by `section_type` and report relaxed span overlap strictly under a `diagnostic` key. Neither diagnostic value may participate in configuration selection.

```python
# src/evaluation/trusted_split.py
_DEVELOPMENT_IDS = tuple(range(101, 181))
_HOLDOUT_IDS = tuple(range(181, 201))


def development_ids() -> tuple[int, ...]:
    return _DEVELOPMENT_IDS


def holdout_ids() -> tuple[int, ...]:
    return _HOLDOUT_IDS
```

- [ ] **Step 5: Run tests and verify GREEN**

Run `python -m pytest tests/test_precision_metrics.py -v`.

Expected: 6 tests pass.

- [ ] **Step 6: Commit Task 1**

```powershell
git add requirements.txt pytest.ini src/evaluation tests/conftest.py tests/test_precision_metrics.py
git commit -m "test: establish trusted precision metrics"
```

---

### Task 2: Centralize Configuration and Audit Overrides

**Files:**

- Create: `src/config.py`
- Create: `src/resources/verified_overrides.json`
- Create: `src/validation/override_validator.py`
- Create: `scripts/audit_overrides.py`
- Create: `tests/test_config.py`
- Create: `tests/test_override_validator.py`
- Modify: `src/pipeline/main.py`
- Modify: `src/utils/setup_rxnorm.py`
- Modify: `src/retrieval/eval_recall.py`
- Modify: `scripts/create_gt_all.py`
- Modify: `scripts/aggregate_data.py`

**Interfaces:**

- Produces: `PROJECT_ROOT`, `DATA_DIR`, `KB_DIR`, `PipelineConfig`, `RetrievalConfig`, `ChunkingConfig`, `NERConfig`, `AssertionConfig`, `CandidateSelectionConfig`, `RerankerConfig`
- Produces: `PipelineConfig.from_mapping(values) -> PipelineConfig` and `PipelineConfig.to_dict() -> dict`
- Produces: `validate_override_entries(entries: list[dict], db_path: str) -> list[str]`
- Produces: `find_machine_specific_paths(paths: list[Path]) -> list[PathFinding]`

- [ ] **Step 1: Write failing configuration tests**

```python
# tests/test_config.py
import pytest
from src.config import NERConfig, PipelineConfig, RerankerConfig, RetrievalConfig


def test_retrieval_weights_sum_to_one():
    config = RetrievalConfig(alpha=0.75)
    assert config.bm25_weight == 0.75
    assert config.semantic_weight == 0.25
    assert config.bm25_weight + config.semantic_weight == pytest.approx(1.0)


def test_alpha_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        RetrievalConfig(alpha=1.1)


def test_default_pipeline_is_precision_first():
    config = PipelineConfig()
    assert config.retrieval.alpha == 0.75
    assert config.ner.beta == 0.5
    assert set(config.ner.per_type_thresholds) == {
        "CHẨN_ĐOÁN", "TRIỆU_CHỨNG", "THUỐC", "TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"
    }
    assert config.retrieval.internal_top_k == 20
    assert config.assertion.negated_threshold == 0.70
    assert config.selection.load_historical_rxnorm is False
    assert config.reranker.enabled is False
    assert config.reranker.timeout_seconds == 30.0


def test_non_positive_reranker_timeout_is_rejected():
    with pytest.raises(ValueError):
        RerankerConfig(timeout_seconds=0)


def test_threshold_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        NERConfig(default_threshold=1.01)


def test_config_mapping_rejects_unknown_keys_and_preserves_weight_invariant():
    config = PipelineConfig.from_mapping({"retrieval": {"alpha": 0.80}})
    assert config.retrieval.bm25_weight + config.retrieval.semantic_weight == pytest.approx(1.0)
    with pytest.raises(ValueError, match="unknown"):
        PipelineConfig.from_mapping({"retrieval": {"bonus": 0.20}})
```

- [ ] **Step 2: Run configuration tests and verify RED**

Run `python -m pytest tests/test_config.py -v`.

Expected: import fails because `src.config` does not exist.

- [ ] **Step 3: Implement validated dataclass configuration**

```python
# src/config.py
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
KB_DIR = DATA_DIR / "kb"


@dataclass(frozen=True)
class RetrievalConfig:
    alpha: float = 0.75
    internal_top_k: int = 20
    hierarchical_expansion: bool = False

    def __post_init__(self):
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if self.internal_top_k < 1:
            raise ValueError("internal_top_k must be positive")

    @property
    def bm25_weight(self) -> float:
        return self.alpha

    @property
    def semantic_weight(self) -> float:
        return 1.0 - self.alpha


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 384
    overlap_tokens: int = 64

    def __post_init__(self):
        if self.max_tokens < 1 or not 0 <= self.overlap_tokens < self.max_tokens:
            raise ValueError("chunk token bounds are invalid")


@dataclass(frozen=True)
class NERConfig:
    beta: float = 0.5
    ambiguity_margin: float = 0.15
    default_threshold: float = 0.70
    per_type_thresholds: Mapping[str, float] = field(default_factory=lambda: {
        "CHẨN_ĐOÁN": 0.72,
        "TRIỆU_CHỨNG": 0.76,
        "THUỐC": 0.72,
        "TÊN_XÉT_NGHIỆM": 0.72,
        "KẾT_QUẢ_XÉT_NGHIỆM": 0.75,
    })

    def __post_init__(self):
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if not 0.0 <= self.ambiguity_margin <= 1.0:
            raise ValueError("ambiguity_margin must be in [0, 1]")
        if not 0.0 <= self.default_threshold <= 1.0:
            raise ValueError("default_threshold must be in [0, 1]")
        frozen_thresholds = MappingProxyType(dict(self.per_type_thresholds))
        if any(not 0.0 <= value <= 1.0 for value in frozen_thresholds.values()):
            raise ValueError("per-type thresholds must be in [0, 1]")
        object.__setattr__(self, "per_type_thresholds", frozen_thresholds)


@dataclass(frozen=True)
class AssertionConfig:
    negated_threshold: float = 0.70
    historical_threshold: float = 0.70
    family_threshold: float = 0.70

    def __post_init__(self):
        values = (self.negated_threshold, self.historical_threshold, self.family_threshold)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("assertion thresholds must be in [0, 1]")


@dataclass(frozen=True)
class CandidateSelectionConfig:
    icd_min_score: float = 0.55
    rxnorm_min_score: float = 0.60
    top1_margin: float = 0.12
    top2_margin: float = 0.04
    load_historical_rxnorm: bool = False

    def __post_init__(self):
        values = (self.icd_min_score, self.rxnorm_min_score, self.top1_margin, self.top2_margin)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("candidate thresholds must be in [0, 1]")


@dataclass(frozen=True)
class RerankerConfig:
    enabled: bool = False
    timeout_seconds: float = 30.0

    def __post_init__(self):
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class PipelineConfig:
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    ner: NERConfig = field(default_factory=NERConfig)
    assertion: AssertionConfig = field(default_factory=AssertionConfig)
    selection: CandidateSelectionConfig = field(default_factory=CandidateSelectionConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
```

Implement `from_mapping()` by explicitly constructing each nested dataclass and rejecting unknown keys at every level; never use an unchecked recursive `setattr`. `to_dict()` must serialize mappings and nested dataclasses to plain JSON-compatible values so Task 9 can hash and lock the exact configuration.

- [ ] **Step 4: Run configuration tests and verify GREEN**

Run `python -m pytest tests/test_config.py -v`.

Expected: 6 tests pass.

- [ ] **Step 5: Write a failing machine-specific path regression test**

Append to `tests/test_config.py`:

```python
from src.validation.override_validator import find_machine_specific_paths


def test_absolute_path_audit_reports_file_line_and_value(tmp_path):
    source_path = tmp_path / "bad_runtime.py"
    source_path.write_text('DATA_DIR = r"D:\\\\private-data"\n', encoding="utf-8")
    findings = find_machine_specific_paths([source_path])
    assert [(item.path, item.line_number, item.value) for item in findings] == [
        (source_path, 1, r"D:\\private-data")
    ]
```

- [ ] **Step 6: Run the path regression and verify RED**

Run `python -m pytest tests/test_config.py::test_absolute_path_audit_reports_file_line_and_value -v`.

Expected: import fails because the path-audit API does not exist.

- [ ] **Step 7: Remove runtime path hardcoding**

Implement `find_machine_specific_paths()` as the reusable behavior behind `scripts/audit_overrides.py --scan-paths`. In `setup_rxnorm.py`, derive `DEFAULT_ZIP_PATH = KB_DIR / "rxnorm_full_01052026.zip"` and accept `setup_rxnorm(zip_path: str | Path = DEFAULT_ZIP_PATH)`. In `eval_recall.py`, `create_gt_all.py`, and `aggregate_data.py`, replace module-level absolute paths with required `argparse` inputs or project-relative defaults from `src.config`; validate paths before processing. `create_gt_all.py` must call generated labels `pseudo-GT` in its CLI help, logs, and output metadata so they cannot be mistaken for the supplied labels 101–200.

Run `python -m pytest tests/test_config.py -v`, then `python scripts/audit_overrides.py --scan-paths src scripts`.

Expected: all configuration/path-audit tests pass and the repository scan reports no machine-specific path.

- [ ] **Step 8: Write failing override integrity tests**

```python
# tests/test_override_validator.py
from src.validation.override_validator import validate_override_entries


def test_known_wrong_rxnorm_mapping_is_rejected(metadata_db):
    entries = [{
        "term": "ketorolac",
        "type": "THUỐC",
        "codes": ["6809"],
        "source": "legacy",
        "note": "known bad mapping",
    }]
    errors = validate_override_entries(entries, str(metadata_db))
    assert any("6809" in error and "metformin" in error for error in errors)


def test_override_requires_provenance(metadata_db):
    entries = [{"term": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "codes": ["I10"]}]
    errors = validate_override_entries(entries, str(metadata_db))
    assert any("source" in error for error in errors)


def test_pipeline_does_not_load_quarantined_legacy_overrides(project_root):
    source = (project_root / "src/pipeline/main.py").read_text(encoding="utf-8")
    assert 'KB_DIR, "override_dict.json"' not in source
```

- [ ] **Step 9: Run override tests and verify RED**

Run `python -m pytest tests/test_override_validator.py -v`.

Expected: import fails because the validator does not exist.

- [ ] **Step 10: Implement KB existence and name compatibility validation**

Implement `validate_override_entries()` with read-only SQLite connections. For RxNorm, require each code to exist and require either the normalized term to occur in an RxNorm name or `note` to contain `verified brand alias`. For ICD, require code existence and provenance. Return errors; do not raise after the first entry.

```python
def validate_override_entries(entries: list[dict], db_path: str) -> list[str]:
    errors = []
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        for index, entry in enumerate(entries):
            for field in ("term", "type", "codes", "source", "note"):
                if field not in entry:
                    errors.append(f"entry[{index}] missing {field}")
            if errors and any(error.startswith(f"entry[{index}]") for error in errors):
                continue
            for code in entry["codes"]:
                error = _validate_code(conn, entry, str(code).strip())
                if error:
                    errors.append(f"entry[{index}] {error}")
    return errors
```

Create `verified_overrides.json` with schema version 1 and an initially empty `entries` list. Treat `data/kb/override_dict.json` as quarantined legacy evidence: do not load it during inference and do not silently migrate its entries. The pipeline may load only `verified_overrides.json`. The audit must explicitly report the known wrong `ketorolac -> 6809` and `nitroglycerin -> 7417` mappings in the legacy file; do not replace them with guessed codes.

- [ ] **Step 11: Add audit CLI and verify GREEN**

The CLI supports the verified schema plus `--legacy-format` for forensic audit, prints every error, and exits `1` on errors or `0` on success. A failing audit of the quarantined legacy file is expected and must never block using an empty/clean verified file.

Run:

```powershell
python scripts/audit_overrides.py --db data/kb/metadata.db --overrides src/resources/verified_overrides.json
python scripts/audit_overrides.py --db data/kb/metadata.db --overrides data/kb/override_dict.json --legacy-format
python -m pytest tests/test_override_validator.py -v
```

Expected: CLI exit 0 for the verified file, non-zero for the quarantined legacy file with the two known wrong mappings listed, and tests pass.

- [ ] **Step 12: Commit Task 2**

```powershell
git add src/config.py src/resources/verified_overrides.json src/validation/override_validator.py src/utils/setup_rxnorm.py src/retrieval/eval_recall.py src/pipeline/main.py scripts/audit_overrides.py scripts/create_gt_all.py scripts/aggregate_data.py tests/test_config.py tests/test_override_validator.py
git commit -m "refactor: centralize precision-first configuration"
```

---

### Task 3: Add Submission Validation and Safe Packaging

**Files:**

- Create: `src/validation/submission.py`
- Create: `scripts/package_submission.py`
- Create: `tests/test_submission.py`
- Modify: `src/pipeline/main.py:187-209`

**Interfaces:**

- Produces: `validate_prediction(text: str, entities: list[dict], db_path: str | None) -> list[str]`
- Produces: `validate_output_directory(input_dir: str, output_dir: str, expected_ids: range) -> list[str]`
- Produces: `package_submission(output_dir: str, zip_path: str) -> None`
- Produces: `write_failure_output(output_path, error_log_path, file_id, error) -> None`

- [ ] **Step 1: Write failing schema, offset, and zip tests**

```python
import json
import zipfile

from src.validation.submission import package_submission, validate_prediction, write_failure_output


def test_validator_rejects_text_offset_mismatch(tmp_path):
    errors = validate_prediction(
        "bệnh nhân ho",
        [{"text": "sốt", "type": "TRIỆU_CHỨNG", "position": [10, 12], "assertions": []}],
        db_path=None,
    )
    assert any("text slice" in error for error in errors)


def test_packager_puts_json_under_output_folder(tmp_path):
    output = tmp_path / "predictions"
    output.mkdir()
    for file_id in range(1, 101):
        (output / f"{file_id}.json").write_text("[]", encoding="utf-8")
    zip_path = tmp_path / "output.zip"
    package_submission(str(output), str(zip_path))
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.namelist()[0].startswith("output/")
        assert set(archive.namelist()) == {f"output/{i}.json" for i in range(1, 101)}


def test_failure_writer_creates_empty_output_and_structured_log(tmp_path):
    output_path = tmp_path / "35.json"
    log_path = tmp_path / "errors.jsonl"
    write_failure_output(output_path, log_path, "35", ValueError("bad offset"))
    assert json.loads(output_path.read_text(encoding="utf-8")) == []
    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert record == {
        "file_id": "35",
        "error_type": "ValueError",
        "message": "bad offset",
    }
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_submission.py -v`.

Expected: import fails because `src.validation.submission` does not exist.

- [ ] **Step 3: Implement dynamic schema and exact offset validation**

Use one schema map keyed by type. Require integer offsets, allowed assertions, exact keys, and exact text slice. When `db_path` is supplied, verify ICD and RxNorm candidates in the corresponding table.

```python
SCHEMA_KEYS = {
    "CHẨN_ĐOÁN": {"text", "type", "position", "assertions", "candidates"},
    "THUỐC": {"text", "type", "position", "assertions", "candidates"},
    "TRIỆU_CHỨNG": {"text", "type", "position", "assertions"},
    "TÊN_XÉT_NGHIỆM": {"text", "type", "position"},
    "KẾT_QUẢ_XÉT_NGHIỆM": {"text", "type", "position"},
}
ALLOWED_ASSERTIONS = {"isNegated", "isHistorical", "isFamily"}
```

- [ ] **Step 4: Implement fail-safe packaging**

`package_submission()` validates that files `1.json` through `100.json` exist before opening the destination zip. Write each member with `arcname=f"output/{file_id}.json"`. Write to `<zip>.tmp` first, then atomically replace the destination only after success.

- [ ] **Step 5: Make pipeline write `[]` on per-file failure**

Implement `write_failure_output()` using UTF-8 JSON and append-only JSON Lines. In `BaselinePipeline.run()`, compute `output_path` before `try`; in `except`, delegate to the helper with file ID, error class, and message. Do not include a traceback in the public output and do not leave a missing JSON.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_submission.py -v
python scripts/package_submission.py --input data/input --output data/output --zip output.zip --validate-only
```

Expected: tests pass; validator reports existing offset errors before packaging and refuses to overwrite `output.zip`.

- [ ] **Step 7: Commit Task 3**

```powershell
git add src/validation/submission.py scripts/package_submission.py tests/test_submission.py src/pipeline/main.py
git commit -m "feat: validate and package submissions safely"
```

---

### Task 4: Implement Offset-Safe Adaptive Clinical Chunking

**Files:**

- Create: `src/chunking/__init__.py`
- Create: `src/chunking/clinical_chunker.py`
- Create: `src/resources/section_patterns.json`
- Create: `tests/test_clinical_chunker.py`

**Interfaces:**

- Produces: `ClinicalChunk(text: str, start: int, end: int, section_type: str, header_text: str)`
- Produces: `ClinicalChunker.chunk(document: str) -> list[ClinicalChunk]`
- Produces: `ClinicalChunker.context_for_span(document, chunk, start, end, entity_type) -> str`

- [ ] **Step 1: Write failing chunking tests**

```python
from src.chunking.clinical_chunker import ClinicalChunker
from src.config import ChunkingConfig


def test_chunk_offsets_always_slice_original_text():
    text = "1. Tiền sử bệnh\nThuốc trước khi nhập viện\n- metoprolol 25mg po bid\n\n2. Kết quả xét nghiệm\ncreatinine: 1.2 mg/dL"
    chunks = ClinicalChunker().chunk(text)
    assert chunks
    assert all(text[c.start:c.end] == c.text for c in chunks)


def test_pre_admission_medication_section_is_preserved():
    text = "Thuốc trước khi nhập viện\n- metoprolol 25mg po bid"
    chunk = ClinicalChunker().chunk(text)[0]
    assert chunk.section_type == "pre_admission_medications"
    assert chunk.header_text == "Thuốc trước khi nhập viện"


def test_long_section_uses_overlap_without_losing_offsets():
    text = "Triệu chứng hiện tại\n" + "ho khó thở " * 1000
    chunker = ClinicalChunker(ChunkingConfig(max_tokens=50, overlap_tokens=10))
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    assert chunks[1].start < chunks[0].end
    assert all(text[c.start:c.end] == c.text for c in chunks)


def test_lab_context_does_not_cross_into_medication_section():
    text = "Thuốc hiện tại\n- metoprolol 25mg\nKết quả xét nghiệm\ncreatinine: 1.2 mg/dL"
    chunker = ClinicalChunker()
    chunks = chunker.chunk(text)
    start = text.index("creatinine")
    lab_chunk = next(chunk for chunk in chunks if chunk.start <= start < chunk.end)
    context = chunker.context_for_span(
        text, lab_chunk, start, start + len("creatinine"), "TÊN_XÉT_NGHIỆM"
    )
    assert "creatinine: 1.2 mg/dL" in context
    assert "metoprolol" not in context
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_clinical_chunker.py -v`.

Expected: import fails because the chunker does not exist.

- [ ] **Step 3: Add section pattern data**

Create JSON entries with `section_type`, `patterns`, and `source`. Include the exact headings already present in trusted input, but patterns must be generic phrases rather than file-specific text.

```json
{
  "version": 1,
  "sections": [
    {"section_type": "demographics", "patterns": ["thông tin bệnh nhân", "hành chính"], "source": "clinical document heading"},
    {"section_type": "past_medical_history", "patterns": ["tiền sử bệnh"], "source": "clinical document heading"},
    {"section_type": "pre_admission_medications", "patterns": ["thuốc trước khi nhập viện"], "source": "clinical document heading"},
    {"section_type": "current_symptoms", "patterns": ["triệu chứng hiện tại", "tiền sử bệnh hiện tại"], "source": "clinical document heading"},
    {"section_type": "examination_assessment", "patterns": ["khám lâm sàng", "đánh giá"], "source": "clinical document heading"},
    {"section_type": "laboratory", "patterns": ["kết quả xét nghiệm", "xét nghiệm"], "source": "clinical document heading"},
    {"section_type": "imaging", "patterns": ["chẩn đoán hình ảnh"], "source": "clinical document heading"},
    {"section_type": "treatment_current_medications", "patterns": ["điều trị", "thuốc hiện tại"], "source": "clinical document heading"},
    {"section_type": "family_history", "patterns": ["tiền sử gia đình"], "source": "clinical document heading"}
  ]
}
```

- [ ] **Step 4: Implement chunk dataclass, section spans, and token windows**

Use regex `finditer()` on line starts to obtain absolute section boundaries; when patterns overlap, choose the longest matching heading before assigning the section type. Unmatched text uses `section_type="unknown"`. Split section content into line/bullet units while retaining `match.start()` offsets. Use whitespace token spans when no model tokenizer is injected. Group units to the token limit, then create overlapping windows only for oversized units.

Do not call `.strip()` when assigning `ClinicalChunk.text`; trim by adjusting `start` and `end`, then slice the original document.

Implement entity-centered context without changing entity offsets: medication receives its bullet/line plus section header; diagnosis/symptom receives its sentence plus at most one neighboring sentence inside the same section; test name/result receives only its line/bullet. Never cross a section boundary.

- [ ] **Step 5: Run tests and verify GREEN**

Run `python -m pytest tests/test_clinical_chunker.py -v`.

Expected: all tests pass.

- [ ] **Step 6: Commit Task 4**

```powershell
git add src/chunking src/resources/section_patterns.json tests/test_clinical_chunker.py
git commit -m "feat: add adaptive clinical chunking"
```

---

### Task 5: Separate Mention Detection from Contextual Type Resolution

**Files:**

- Create: `src/ner/types.py`
- Create: `src/ner/type_resolver.py`
- Create: `src/ner/lexicon_loader.py`
- Create: `src/resources/type_rules.json`
- Create: `src/resources/clinical_lexicon.json`
- Create: `tests/test_type_resolver.py`
- Modify: `src/ner/extractor.py:158-381`

**Interfaces:**

- Produces: `MentionCandidate(text, start, end, candidate_types, sources, exact)`
- Produces: `TypeDecision(entity_type: str | None, confidence: float, scores: dict[str, float], reason: str)`
- Produces: `ClinicalLexicon.load(path) -> tuple[ClinicalTerm, ...]`
- Produces: `ContextualTypeResolver.resolve(mention, document, chunk) -> TypeDecision`
- Changes: `BaselineExtractor(..., clinical_lexicon_path=None).extract_entities(text, chunks=None) -> list[dict]`

- [ ] **Step 1: Write failing contextual disambiguation tests**

```python
import pytest

from src.chunking.clinical_chunker import ClinicalChunk, ClinicalChunker
from src.ner.extractor import BaselineExtractor
from src.ner.lexicon_loader import ClinicalLexicon
from src.ner.type_resolver import ContextualTypeResolver
from src.ner.types import MentionCandidate


def test_creatinine_in_lab_section_is_not_drug():
    text = "Kết quả xét nghiệm\ncreatinine: 1.2 mg/dL"
    chunk = ClinicalChunker().chunk(text)[0]
    start = text.index("creatinine")
    mention = MentionCandidate("creatinine", start, start + len("creatinine"), frozenset({"THUỐC", "TÊN_XÉT_NGHIỆM"}), frozenset({"rxnorm", "lab"}), True)
    decision = ContextualTypeResolver().resolve(mention, text, chunk)
    assert decision.entity_type == "TÊN_XÉT_NGHIỆM"


def test_generic_word_is_not_extracted_inside_longer_word():
    extractor = BaselineExtractor(load_database=False)
    assert not any(entity["text"].lower() == "yếu" for entity in extractor.extract_entities("Các yếu tố nguy cơ"))


def test_ambiguous_low_confidence_span_is_rejected():
    mention = MentionCandidate("loét", 0, 4, frozenset({"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}), frozenset({"lexicon"}), True)
    chunk = ClinicalChunk("loét", 0, 4, "unknown", "")
    decision = ContextualTypeResolver().resolve(mention, "loét", chunk)
    assert decision.entity_type is None


def test_extractor_uses_injected_clinical_lexicon(tmp_path):
    path = tmp_path / "clinical_lexicon.json"
    path.write_text(
        '{"schema_version": 1, "entries": ['
        '{"term": "xét nghiệm zeta", "type": "TÊN_XÉT_NGHIỆM", '
        '"source": "verified-test", "status": "verified"}'
        ']}',
        encoding="utf-8",
    )
    extractor = BaselineExtractor(load_database=False, clinical_lexicon_path=path)
    entities = extractor.extract_entities("Kết quả xét nghiệm\nxét nghiệm zeta: 1.2")
    assert any(entity["text"] == "xét nghiệm zeta" for entity in entities)


def test_lexicon_requires_source_and_status(tmp_path):
    path = tmp_path / "lexicon.json"
    path.write_text(
        '{"schema_version": 1, "entries": [{"term": "ho", "type": "TRIỆU_CHỨNG"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source|status"):
        ClinicalLexicon.load(path)


def test_lexicon_deduplicates_normalized_term_and_type(tmp_path):
    path = tmp_path / "lexicon.json"
    path.write_text(
        '{"schema_version": 1, "entries": ['
        '{"term": "Ho", "type": "TRIỆU_CHỨNG", "source": "legacy", "status": "unverified"},'
        '{"term": "ho", "type": "TRIỆU_CHỨNG", "source": "legacy", "status": "unverified"}'
        ']}',
        encoding="utf-8",
    )
    assert len(ClinicalLexicon.load(path)) == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_type_resolver.py -v`.

Expected: imports fail because mention/type contracts do not exist.

- [ ] **Step 3: Implement immutable contracts**

```python
@dataclass(frozen=True)
class MentionCandidate:
    text: str
    start: int
    end: int
    candidate_types: frozenset[str]
    sources: frozenset[str]
    exact: bool


@dataclass(frozen=True)
class TypeDecision:
    entity_type: str | None
    confidence: float
    scores: dict[str, float]
    reason: str
```

- [ ] **Step 4: Externalize clinical lexicons with provenance**

Move the existing `CLINICAL_MEDICATIONS`, `CLINICAL_DIAGNOSES`, `CLINICAL_SYMPTOMS`, and `TEST_NAMES` entries mechanically into `clinical_lexicon.json`. Each entry must have `term`, `type`, `source`, and `status`; mark legacy, unverified terms with `status: "unverified"` instead of silently treating them as ground truth. The loader validates schema, normalizes duplicates, and supplies a lower source prior for unverified entries. `BaselineExtractor` loads only the configured/injected lexicon resource; no lexicon entry may bypass contextual thresholds or the ambiguity reject gate.

Move generic blacklist behavior into explicit penalties in `type_rules.json`; a generic token may only survive when contextual evidence raises it above the configured per-type threshold. Implement the malformed-entry and normalized-duplicate behavior already covered by Step 1 before wiring the loader into the extractor.

- [ ] **Step 5: Move scoring constants into `type_rules.json`**

Include section priors, medication signals, laboratory units, route terms, generic terms, and source-confidence weights. Per-type thresholds and ambiguity margin come from `NERConfig` so development calibration can serialize them into the locked configuration. The resolver loads rule data once. No file-specific phrase is allowed.

- [ ] **Step 6: Implement type scoring and reject gate**

Start all candidate types at their exact/source prior, add section and local-signal weights, then normalize scores to `[0, 1]` using `1 / (1 + exp(-raw_score))`. Reject when best score is below its per-type threshold or the best/second-best gap is below ambiguity margin.

- [ ] **Step 7: Refactor Trie output without changing dictionary loading yet**

Change `TrieMatcher.search_in_text()` to aggregate every type at the same absolute span. Return `MentionCandidate` objects. Remove type priority from mention detection. Keep word-boundary enforcement before candidate creation.

`BaselineExtractor.extract_entities()` obtains chunks, resolves every mention, rejects `None`, and performs confidence-based overlap resolution. Drug dose expansion must adjust `end` and set `text = document[start:end]` without `.strip()` mismatch.

- [ ] **Step 8: Run focused and existing extractor tests**

Run:

```powershell
python -m pytest tests/test_type_resolver.py -v
python src/ner/extractor.py
```

Expected: contextual tests pass; extractor self-check runs without offset errors.

- [ ] **Step 9: Commit Task 5**

```powershell
git add src/ner/types.py src/ner/type_resolver.py src/ner/lexicon_loader.py src/ner/extractor.py src/resources/type_rules.json src/resources/clinical_lexicon.json tests/test_type_resolver.py
git commit -m "refactor: resolve NER types from clinical context"
```

---

### Task 6: Make Assertions Section-Aware and Scope-Safe

**Files:**

- Create: `src/resources/assertion_rules.json`
- Modify: `src/assertion/rule_based.py:3-82`
- Create: `tests/test_assertion_analyzer.py`
- Modify: `src/pipeline/main.py:74-77`

**Interfaces:**

- Produces: `AssertionAnalyzer.score(full_text, start_idx, end_idx, section_type="unknown", header_text="") -> dict[str, float]`
- Changes: `AssertionAnalyzer.analyze(full_text, start_idx, end_idx, section_type="unknown", header_text="") -> list[str]`

- [ ] **Step 1: Write failing assertion scope tests**

```python
import json

import pytest

from src.assertion.rule_based import AssertionAnalyzer


@pytest.mark.parametrize("text, entity", [
    ("Bệnh nhân không có ho", "ho"),
    ("Chưa phát hiện viêm phổi", "viêm phổi"),
    ("Bệnh nhân phủ nhận có đau ngực", "đau ngực"),
])
def test_negation_phrase_keeps_cue(text, entity):
    start = text.index(entity)
    result = AssertionAnalyzer().analyze(text, start, start + len(entity))
    assert "isNegated" in result


def test_pre_admission_medication_is_historical():
    text = "Thuốc trước khi nhập viện\n- metoprolol 25mg po bid"
    start = text.index("metoprolol")
    result = AssertionAnalyzer().analyze(
        text, start, start + len("metoprolol 25mg po bid"),
        section_type="pre_admission_medications",
        header_text="Thuốc trước khi nhập viện",
    )
    assert "isHistorical" in result


def test_assertion_analyzer_uses_injected_rule_resource(project_root, tmp_path):
    default_path = project_root / "src/resources/assertion_rules.json"
    rules = json.loads(default_path.read_text(encoding="utf-8"))
    rules["negation_cues"].append("tuyệt đối vắng")
    custom_path = tmp_path / "assertion_rules.json"
    custom_path.write_text(json.dumps(rules, ensure_ascii=False), encoding="utf-8")
    text = "Bệnh nhân tuyệt đối vắng đau ngực"
    start = text.index("đau ngực")
    result = AssertionAnalyzer(rules_path=custom_path).analyze(
        text, start, start + len("đau ngực")
    )
    assert "isNegated" in result


def test_assertion_analyzer_exposes_calibratable_scores():
    text = "Bệnh nhân không có ho"
    start = text.index("ho")
    scores = AssertionAnalyzer().score(text, start, start + len("ho"))
    assert 0.0 <= scores["isNegated"] <= 1.0
    assert scores["isNegated"] >= scores["isHistorical"]


def test_family_prior_stops_when_context_returns_to_patient():
    text = "Tiền sử gia đình: mẹ tăng huyết áp. Bệnh nhân hiện đau ngực."
    start = text.index("đau ngực")
    result = AssertionAnalyzer().analyze(
        text,
        start,
        start + len("đau ngực"),
        section_type="family_history",
        header_text="Tiền sử gia đình",
    )
    assert "isFamily" not in result
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_assertion_analyzer.py -v`.

Expected: at least the three compound-negation tests and historical-section test fail.

- [ ] **Step 3: Replace token terminators with phrase-aware scope**

Create `assertion_rules.json` with version, cue lists, confidence weights, scope boundaries, section priors, patient-return cues, and a `source` field for every rule group. Load it once in `AssertionAnalyzer`. `score()` returns a deterministic `[0, 1]` confidence for each allowed assertion. `analyze()` applies the three thresholds from `PipelineConfig.assertion`. Find the nearest punctuation/bullet boundary before the entity. Match compound negation patterns from the resource before applying termination terms. Do not treat `có` or `phát hiện` as a terminator when it belongs to `không có`, `phủ nhận có`, or `chưa phát hiện`. Suppress the family prior for spans after a patient-return cue in the same section.

Add section priors:

```python
if section_type in {"pre_admission_medications", "past_medical_history"}:
    assertions.add("isHistorical")
if section_type == "family_history":
    assertions.add("isFamily")
```

Return assertions in the deterministic order `isNegated`, `isHistorical`, `isFamily`.

- [ ] **Step 4: Pass section context from pipeline**

When processing an entity, locate its containing `ClinicalChunk` and pass `section_type/header_text` to `analyze()`.

- [ ] **Step 5: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_assertion_analyzer.py -v
python src/assertion/rule_based.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 6**

```powershell
git add src/assertion/rule_based.py src/resources/assertion_rules.json src/pipeline/main.py tests/test_assertion_analyzer.py
git commit -m "fix: make assertion scope section aware"
```

---

### Task 7: Replace Rank-Only RRF with Normalized Weighted Fusion

**Files:**

- Create: `src/retrieval/types.py`
- Create: `src/retrieval/score_fusion.py`
- Create: `tests/test_score_fusion.py`
- Modify: `src/retrieval/bm25_retriever.py:74-89`
- Modify: `src/retrieval/hybrid_retriever.py:118-321`

**Interfaces:**

- Produces: `ComponentCandidate(code: str, score: float, rank: int)`
- Produces: `RetrievedCandidate(code, fusion_score, bm25_score, semantic_score, bm25_rank, semantic_rank)`
- Produces: `minmax_scores(candidates) -> dict[str, float]`
- Produces: `fuse_candidates(bm25, semantic, alpha, valid_codes=None) -> list[RetrievedCandidate]`
- Produces: `BM25Retriever.retrieve_scored(query, top_k) -> list[ComponentCandidate]`
- Produces: `HybridRetriever.retrieve_scored(query, top_k=None) -> list[RetrievedCandidate]`
- Preserves: `HybridRetriever.retrieve(query, top_k=5) -> list[str]` as a compatibility wrapper.

- [ ] **Step 1: Write failing score fusion tests**

```python
import pytest

from src.retrieval.score_fusion import fuse_candidates, minmax_scores
from src.retrieval.types import ComponentCandidate


def test_fusion_weights_sum_to_one_and_favor_bm25():
    bm25 = [ComponentCandidate("A", 10.0, 0), ComponentCandidate("B", 5.0, 1)]
    semantic = [ComponentCandidate("B", 0.95, 0), ComponentCandidate("C", 0.90, 1)]
    fused = fuse_candidates(bm25, semantic, alpha=0.75)
    by_code = {item.code: item for item in fused}
    assert by_code["A"].fusion_score == pytest.approx(0.75)
    assert by_code["C"].fusion_score <= 0.25


def test_alpha_boundaries_are_exact_component_modes():
    bm25 = [ComponentCandidate("A", 2.0, 0)]
    semantic = [ComponentCandidate("B", 0.9, 0)]
    assert fuse_candidates(bm25, semantic, alpha=1.0)[0].code == "A"
    assert fuse_candidates(bm25, semantic, alpha=0.0)[0].code == "B"


def test_equal_component_scores_are_deterministic():
    values = [ComponentCandidate("B", 1.0, 0), ComponentCandidate("A", 1.0, 1)]
    assert minmax_scores(values) == {"B": 1.0, "A": 1.0}


def test_invalid_kb_codes_are_removed_before_fusion():
    bm25 = [ComponentCandidate("VALID", 2.0, 0), ComponentCandidate("MISSING", 1.0, 1)]
    fused = fuse_candidates(bm25, [], alpha=0.75, valid_codes={"VALID"})
    assert [candidate.code for candidate in fused] == ["VALID"]
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_score_fusion.py -v`.

Expected: imports fail because scored retrieval types do not exist.

- [ ] **Step 3: Implement scored contracts and pure fusion functions**

Validate `alpha` in `[0, 1]`. Normalize code strings and remove candidates outside `valid_codes` before computing component minima/maxima. Normalize only candidates present in each component. Missing candidates receive `0.0`. Sort by descending fusion score, then ascending BM25 rank, semantic rank, and code.

```python
fusion = alpha * bm25_normalized.get(code, 0.0) + (1.0 - alpha) * semantic_normalized.get(code, 0.0)
```

- [ ] **Step 4: Expose BM25 scores**

Add `retrieve_scored()` using both arrays returned by `bm25s.BM25.retrieve()`. Convert each result into `ComponentCandidate`. Keep `retrieve()` as `[item.code for item in retrieve_scored(...)]`.

- [ ] **Step 5: Expose FAISS scores and integrate fusion**

Convert FAISS distances/indices to deduplicated, higher-is-better `ComponentCandidate` values. Retrieve Top-20 from each component, pass the known ICD/RxNorm code set into `fuse_candidates()`, and apply the configured alpha. Remove rank-only RRF plus every exact-match or dose-form score bonus outside the normalized formula. Slice only after fusion. Disable implicit ICD/RxNorm hierarchical insertion unless `hierarchical_expansion=True`.

- [ ] **Step 6: Run focused retrieval tests**

Run:

```powershell
python -m pytest tests/test_score_fusion.py -v
python -c "from src.retrieval.hybrid_retriever import HybridRetriever; print(HybridRetriever('icd10').retrieve('tăng huyết áp', 5))"
```

Expected: tests pass; smoke query returns codes.

- [ ] **Step 7: Commit Task 7**

```powershell
git add src/retrieval/types.py src/retrieval/score_fusion.py src/retrieval/bm25_retriever.py src/retrieval/hybrid_retriever.py tests/test_score_fusion.py
git commit -m "feat: fuse normalized BM25 and semantic scores"
```

---

### Task 8: Add Precision-First Candidate Selection and Safe Clinical Validation

**Files:**

- Create: `src/retrieval/candidate_selector.py`
- Create: `src/resources/clinical_validation_rules.json`
- Create: `tests/test_candidate_selector.py`
- Modify: `src/validation/clinical_validator.py:196-281`
- Modify: `src/ranking/llm_reranker.py:71-166`
- Modify: `src/pipeline/main.py:79-183`

**Interfaces:**

- Produces: `CandidateSelector.select(entity_type, ranked, is_valid) -> list[str]`
- Produces: `ClinicalValidator.is_candidate_valid(entity, code, patient_info) -> bool`
- Produces: `dose_form_is_compatible(drug_text, rxcui_name, rules) -> bool`
- Produces: `LLMReranker.parse_selected_codes(payload, allowed_codes) -> list[str]`
- Changes: LLM reranker returns a validated subset of input candidates or the unchanged fallback list.

- [ ] **Step 1: Write failing selector tests**

```python
import pytest

from src.ranking.llm_reranker import LLMReranker
from src.retrieval.candidate_selector import CandidateSelector
from src.retrieval.types import RetrievedCandidate
from src.validation.clinical_validator import ClinicalValidator, dose_form_is_compatible


def candidate(code: str, score: float) -> RetrievedCandidate:
    return RetrievedCandidate(
        code=code,
        fusion_score=score,
        bm25_score=score,
        semantic_score=0.0,
        bm25_rank=0,
        semantic_rank=None,
    )


def test_selector_returns_top1_for_clear_margin():
    ranked = [candidate("A", 0.90), candidate("B", 0.60)]
    assert CandidateSelector().select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A"]


def test_selector_returns_top2_only_for_close_valid_scores():
    ranked = [candidate("A", 0.80), candidate("B", 0.78), candidate("C", 0.50)]
    assert CandidateSelector().select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A", "B"]


def test_selector_rejects_all_invalid_or_low_confidence_codes():
    ranked = [candidate("A", 0.40), candidate("B", 0.30)]
    assert CandidateSelector().select("THUỐC", ranked, lambda _: True) == []


def test_dual_code_check_never_creates_new_entity(metadata_db):
    clinical_validator = ClinicalValidator(db_path=str(metadata_db))
    entities = [{"text": "gout", "type": "CHẨN_ĐOÁN", "position": [0, 4], "assertions": [], "candidates": ["M10.9"]}]
    result = clinical_validator.check_dual_codes(entities)
    assert len(result) == 1
    assert result[0]["text"] == "gout"


def test_historical_rxnorm_mapping_is_opt_in(metadata_db):
    clinical_validator = ClinicalValidator(
        db_path=str(metadata_db),
        load_historical_rxnorm=False,
    )
    assert clinical_validator.rxnorm_mapping == {}


def test_dose_form_validation_uses_supplied_rules():
    rules = {
        "route_groups": [{
            "name": "custom",
            "mention_terms": ["đường zeta"],
            "rxnorm_terms": ["zeta form"],
        }]
    }
    assert dose_form_is_compatible("thuốc đường zeta", "ingredient zeta form", rules)
    assert not dose_form_is_compatible("thuốc đường zeta", "ingredient oral tablet", rules)


def test_llm_reranker_accepts_only_a_subset_of_input_candidates():
    assert LLMReranker.parse_selected_codes(
        '{"selected_codes": ["A"]}', ["A", "B"]
    ) == ["A"]
    with pytest.raises(ValueError, match="candidate pool"):
        LLMReranker.parse_selected_codes(
            '{"selected_codes": ["FOREIGN"]}', ["A", "B"]
        )
```

- [ ] **Step 2: Run tests and verify RED**

Run `python -m pytest tests/test_candidate_selector.py -v`.

Expected: selector imports fail and synthetic-entity regression fails.

- [ ] **Step 3: Implement selector policy**

Filter invalid candidates before applying thresholds. Use ICD/RxNorm minimum scores from config. Return Top-1 when margin is at least `top1_margin`; Top-2 only when both exceed minimum and their margin is at most `top2_margin`; otherwise return Top-1 if it exceeds minimum. Never return more than two codes.

- [ ] **Step 4: Refactor clinical validation into a predicate**

Move sex, age, and dose-form decisions into `is_candidate_valid()`. Keep `check_and_fix_candidates()` as a compatibility wrapper. `check_dual_codes()` must stop creating entities; initially return entities unchanged while logging dual-code metadata for analysis.

Make the 372k-entry historical RxNorm mapping explicitly opt-in through `CandidateSelectionConfig.load_historical_rxnorm`. Do not query or retain that table when the flag is false. Historical CUIs may only expand the retrieved candidate pool before validation and selection; they may never be appended directly to output. Remove README or log claims that the mapping affects inference when the flag is disabled.

Move dose-form terms and contradiction rules to `clinical_validation_rules.json`, with a `source` field per rule group, and route all decisions through the pure `dose_form_is_compatible()` API so resource injection is testable. Delete the `oral tablet`/`oral capsule` preference from `HybridRetriever`: it is an out-of-formula bonus and violates the normalized fusion contract. Dose form may reject a clinically contradictory candidate, but may not add an uncalibrated score bonus.

- [ ] **Step 5: Require LLM subset semantics**

Instantiate and call the reranker only when `PipelineConfig.reranker.enabled` is true, and pass `timeout_seconds` from configuration. Accept either `{"selected_codes": [...]}` or the legacy `best_code`, validate every returned code belongs to the input pool, and return only the selected subset. If parsing, timeout, or validation fails, return the original ranked list for the deterministic selector to handle.

- [ ] **Step 6: Integrate scored retrieval and selector in pipeline**

For diagnosis and medication entities:

1. Retrieve internal Top-20 scored candidates.
2. Apply clinical predicate.
3. Optionally ask LLM for a subset.
4. Run deterministic candidate selector.
5. Assign at most two codes to output.

Remove the per-candidate SQLite exact-match loop in `main.py`; weighted BM25 handles lexical strength and the current block silently fails because `sqlite3` is not imported.

- [ ] **Step 7: Run tests and verify GREEN**

Run:

```powershell
python -m pytest tests/test_candidate_selector.py -v
python -m pytest tests/test_score_fusion.py tests/test_submission.py -v
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 8**

```powershell
git add src/retrieval/candidate_selector.py src/resources/clinical_validation_rules.json src/validation/clinical_validator.py src/ranking/llm_reranker.py src/pipeline/main.py tests/test_candidate_selector.py
git commit -m "feat: select precision-first candidate subsets"
```

---

### Task 9: Integrate Benchmarking, Calibrate on Trusted Data, and Package

**Files:**

- Create: `src/evaluation/benchmark.py`
- Create: `tests/test_pipeline_regressions.py`
- Modify: `src/evaluate.py`
- Modify: `README.md`

**Interfaces:**

- Produces CLI: `python -m src.evaluation.benchmark --dev-pool --alphas 0.60 0.70 0.75 0.80 0.90`
- Produces CLI: `python -m src.evaluation.benchmark --holdout --locked-config <path>`
- Produces JSON report with split label, config, per-type metrics, candidate Jaccard, final score, mean, and standard deviation.
- Produces: `required_metric_paths() -> frozenset[str]` as the stable report contract.

- [ ] **Step 1: Write failing pipeline regression tests**

```python
import pytest

from src.evaluation.benchmark import required_metric_paths
from src.pipeline.main import BaselinePipeline


@pytest.fixture(scope="module")
def pipeline() -> BaselinePipeline:
    return BaselinePipeline()


@pytest.fixture(scope="module")
def public_input_35(project_root):
    return (project_root / "data" / "input" / "35.txt").read_text(encoding="utf-8")


def test_benchmark_report_contract_covers_stage_metrics():
    assert required_metric_paths() == frozenset({
        "entity.micro.precision",
        "entity.micro.recall",
        "entity.micro.f0_5",
        "entity.by_type",
        "entity.errors_by_section",
        "assertion.by_label",
        "assertion.macro_f0_5",
        "candidates.jaccard",
        "candidates.precision",
        "candidates.top1_hit_rate",
        "retrieval.recall_at_20",
        "diagnostic.relaxed_overlap",
        "final_score",
    })


def test_official_medication_example_preserves_exact_offset(pipeline):
    text = "Danh sách thuốc trước nhập viện. amlodipine 10 mg po daily"
    entities = pipeline.process_text(text)
    medication = next(entity for entity in entities if entity["text"].startswith("amlodipine"))
    start, end = medication["position"]
    assert text[start:end] == medication["text"]
    assert "isHistorical" in medication["assertions"]
    assert len(medication["candidates"]) <= 2


def test_no_prediction_contains_synthetic_text(public_input_35, pipeline):
    entities = pipeline.process_text(public_input_35)
    assert all(public_input_35[e["position"][0]:e["position"][1]] == e["text"] for e in entities)
    assert all(not e["text"].startswith("Biểu hiện lâm sàng của") for e in entities)
```

- [ ] **Step 2: Run regression tests and verify RED**

Run `python -m pytest tests/test_pipeline_regressions.py -v`.

Expected: current pipeline API/behavior fails at least one regression.

- [ ] **Step 3: Add `process_text()` and complete integration**

Change construction to `BaselinePipeline(config: PipelineConfig | None = None)` and ensure every component receives the same validated configuration. Extract text processing from `process_file()` into `process_text(text: str) -> list[dict]`; `process_file()` only reads a file and delegates. Add `--input`, `--output`, and `--config` CLI arguments. Load JSON with `PipelineConfig.from_mapping()` and fail before inference on unknown/invalid fields. This makes integration tests independent of filesystem writes and ensures the locked configuration is the one actually used for public inference.

- [ ] **Step 4: Implement benchmark CLI without holdout leakage**

The CLI reads supplied annotations only from `data/dev/gt/101.json` through `data/dev/gt/200.json` and their matching texts from `data/dev/input/`. It refuses `--holdout` unless `--locked-config` points to an existing JSON configuration created by a completed development run, its SHA-256 matches, and no tuning flag is present. Development mode performs deterministic five-fold splitting of IDs 101–180, evaluates each alpha, and writes every path in `required_metric_paths()` per fold plus mean/std. Entity FP/FN are assigned to the containing clinical section. Relaxed overlap remains under `diagnostic` and is never read by the selector. IDs 1–100 from these directories are pseudo-GT: any diagnostic report over them must be labeled `untrusted`, and their scores must never enter model/configuration selection.

- [ ] **Step 5: Run full tests and establish current trusted baseline**

Run:

```powershell
python -m pytest -v
python -m src.evaluation.benchmark --dev-pool --baseline --output reports/baseline-dev.json
```

Expected: all tests pass; baseline report covers only IDs 101–180.

- [ ] **Step 6: Calibrate alpha and thresholds on development folds**

Run:

```powershell
python -m src.evaluation.benchmark --dev-pool --alphas 0.60 0.70 0.75 0.80 0.90 --output reports/precision-first-cv.json --write-locked-config reports/locked-config.json
```

Only configurations whose mean final score is not below the rerun baseline and whose relevant stage metric improves (exact entity F0.5 for NER, assertion macro F0.5 for assertion rules, or candidate Jaccard for retrieval/selector changes) are eligible. Select the eligible configuration with the highest mean final score; when the difference is at most `0.005`, select higher exact entity precision. If none is eligible, lock the baseline configuration instead of forcing the rewrite. Record selected alpha, NER/assertion/selector thresholds, baseline comparison, fold IDs, and a SHA-256 hash of the locked configuration in `locked-config.json`.

- [ ] **Step 7: Run untouched holdout exactly once**

Run:

```powershell
python -m src.evaluation.benchmark --holdout --locked-config reports/locked-config.json --output reports/final-holdout.json
```

Expected: report covers only IDs 181–200 and contains no parameter search. After a successful run, write `reports/holdout-run-<locked-config-hash>.json`; refuse a second holdout run for the same locked hash unless the user starts a new development cycle with a new lock.

- [ ] **Step 8: Rerun public inference and validate before packaging**

Run:

```powershell
python src/pipeline/main.py --input data/input --output data/output --config reports/locked-config.json
python scripts/package_submission.py --input data/input --output data/output --zip output.zip --db data/kb/metadata.db
```

Expected: 100 predictions validate with zero schema, offset, or code-integrity errors; zip contains `output/1.json` through `output/100.json`.

- [ ] **Step 9: Update README with exact commands and limitations**

Document environment setup, trusted data policy, development benchmark, locked holdout, public inference, validation, packaging, and the fact that neural NER remains future work.

- [ ] **Step 10: Run final verification**

Run:

```powershell
python -m pytest -v
python src/metrics.py test
python scripts/audit_overrides.py --db data/kb/metadata.db --overrides src/resources/verified_overrides.json
python scripts/package_submission.py --input data/input --output data/output --zip output.zip --db data/kb/metadata.db --validate-only
git status --short
```

Expected: all tests pass, audit exit 0, validation exit 0, and only intended report/output artefacts remain ignored or explicitly documented.

- [ ] **Step 11: Commit Task 9**

```powershell
git add src/evaluation/benchmark.py src/evaluate.py tests/test_pipeline_regressions.py README.md
git commit -m "feat: calibrate and verify precision-first pipeline"
```

---

## Plan Self-Review Checklist

- [x] Every design requirement maps to a task.
- [x] Every production behavior has a preceding failing test.
- [x] Trusted IDs are consistent across metrics, benchmark, and docs.
- [x] Retrieval interfaces use the same scored candidate types in Tasks 7–9.
- [x] Candidate selector never returns more than two codes.
- [x] No task tunes on IDs 181–200 before locked configuration.
- [x] No placeholder or undefined interface remains.
- [x] Submission packaging always nests JSON under `output/`.
