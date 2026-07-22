# Metadata-First Curriculum Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Triển khai pipeline dữ liệu, huấn luyện ba giai đoạn, assertion, ICD-10/RxNorm linking và notebook Kaggle đúng theo spec metadata-first curriculum v2.0, có OOF/challenge evaluation và artifact có thể kiểm chứng.

**Architecture:** Pipeline được tách thành các module nhỏ theo trách nhiệm: provenance/audit, record metadata, development split, token features, metrics, sampling, curriculum orchestration, assertion, KB contract, mention recovery, experiment tracking và Kaggle orchestration. Stage 1 được cache theo fingerprint; Stage 2–3 chạy theo OOF fold để chọn cấu hình, sau đó final-fit dùng toàn bộ 100 nhãn BTC và 2.000 synthetic theo chính sách đã khóa.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers 4.41–4.x, Accelerate >=1.1, XLM-R, BM25s, FAISS CPU, sentence-transformers, pytest, JSON/JSONL/GZIP, Jupyter/Kaggle single GPU 16 GB.

## Global Constraints

- Spec nguồn: `docs/superpowers/specs/2026-07-23-metadata-first-curriculum-training-design.md` tại commit `6612c62`.
- Không sửa raw text hoặc offset; mọi `raw_text[start:end]` phải bằng `entity.text`.
- ID 1–100 luôn bị loại khỏi fitting, calibration và model selection.
- ID 101–200 là nhãn organizer đáng tin cậy; 90 tài liệu dùng OOF 5-fold, khoảng 10 tài liệu làm blind challenge; final-fit dùng đủ 100.
- Toàn bộ ID 201–2200 đủ điều kiện Stage 1–2; synthetic holdout chiếm 20% ở development; Stage 3 replay chiếm 15–20%.
- NER chỉ có `DISEASE`, `DRUG`, `SYMPTOM`, `LAB_NAME`, `LAB_RESULT`.
- Assertion chỉ áp dụng cho `DISEASE`, `DRUG`, `SYMPTOM` và chỉ xuất `isNegated`, `isHistorical`, `isFamily`.
- ICD-10 chỉ dành cho diagnosis; RxNorm chỉ dành cho drug; candidate rỗng là abstention hợp lệ.
- Retrieval nội bộ mặc định top-20; output cuối mặc định `candidate_output_k = 1`.
- Generic disease/drug/symptom regex không được bật trong primary detector; structural record/section regex được phép.
- `max_length = 512`, `stride = 128`; mỗi gold entity chỉ đóng góp loss trong một owner window.
- Qwen/vLLM là tùy chọn; training và deterministic inference không được phụ thuộc Qwen.
- Runtime tối thiểu là một GPU Kaggle T4/P100 16 GB; không phụ thuộc multi-GPU.
- Chỉ tuyên bố Kaggle thành công sau một lần `Run All` thực tế và audit artifact tải về.
- Không stage/commit dataset, model weights, scratch artifact hoặc file người dùng không liên quan.

## File Structure

### Files to create

- `pytest.ini` — làm cho test import được `v2/clinical_nlp_lab` khi chạy từ repository root.
- `v2/clinical_nlp_lab/provenance.py` — fingerprint và trạng thái current/stale/archived của report.
- `v2/clinical_nlp_lab/audit.py` — sampling audit phân tầng và kiểm tra hai reviewer độc lập.
- `v2/clinical_nlp_lab/records.py` — record/patient-block span và boundary confidence.
- `v2/clinical_nlp_lab/splitting.py` — blind challenge, grouped OOF và synthetic holdout.
- `v2/clinical_nlp_lab/metrics.py` — entity-level exact/overlap/macro metrics và selection score.
- `v2/clinical_nlp_lab/sampling.py` — source-balanced sampling và synthetic replay selection.
- `v2/clinical_nlp_lab/curriculum.py` — stage/fold/final-fit orchestration và resume manifests.
- `v2/clinical_nlp_lab/assertion_training.py` — assertion dataset, threshold calibration và artifact contract.
- `v2/clinical_nlp_lab/kb_contract.py` — canonical/display IDs và gold KB coverage gate.
- `v2/clinical_nlp_lab/mention_recovery.py` — KB-first proposal filtering và incremental metrics.
- `v2/clinical_nlp_lab/experiments.py` — ablation matrix, OOF aggregation và challenge lock.
- `v2/tools/audit_dataset.py` — CLI tạo/kiểm tra audit packet.
- `v2/scripts/train_curriculum.py` — CLI training chính cho notebook.
- Các test mới dưới `v2/tests/` tương ứng từng module.

### Files to modify

- `v2/clinical_nlp_lab/config.py` và `v2/artifacts/config.json` — cấu hình curriculum/OOF/assertion/linking.
- `v2/clinical_nlp_lab/dataset_quality.py` — manifest fields và source-aware candidate validation.
- `v2/clinical_nlp_lab/training.py` — owner-window features và document-level metric callback.
- `v2/clinical_nlp_lab/schema.py` — `RecordSpan`/metadata types nếu interface chung cần dùng.
- `v2/clinical_nlp_lab/kb.py` — build runtime artifact với canonical/display mapping.
- `v2/clinical_nlp_lab/retrieval.py` — retrieval trả display ID và lexical-only fallback an toàn.
- `v2/clinical_nlp_lab/pipeline.py` — trained assertion predictor và filtered KB-first recovery.
- `v2/clinical_nlp_lab/artifacts.py` — model status và artifact inventory mới.
- `v2/clinical_nlp_lab/evaluation.py` — OOF, candidate abstention và KB-first incremental metrics.
- `v2/tools/build_kaggle_notebook.py` — `full`, `resume`, `inference_only` orchestration.
- `v2/medical_information_extraction_kaggle.ipynb` — regenerated canonical notebook.
- `v2/KAGGLE_RUNBOOK.md`, `v2/README.md` — command và artifact contract.

---

### Task 1: Reproducible test bootstrap and configuration contract

**Files:**
- Create: `pytest.ini`
- Modify: `v2/clinical_nlp_lab/config.py`
- Modify: `v2/artifacts/config.json`
- Create: `v2/tests/test_curriculum_config.py`
- Modify: `v2/README.md`

**Interfaces:**
- Consumes: `clinical_nlp_lab.config.load_config(path=None) -> dict[str, Any]`
- Produces: curriculum keys used by every later task and root-level pytest discovery.

- [ ] **Step 1: Write the failing configuration test**

```python
from clinical_nlp_lab.config import load_config


def test_curriculum_defaults_match_approved_spec():
    config = load_config()
    assert config["cv_folds"] == 5
    assert config["challenge_size"] == 10
    assert config["synthetic_holdout_fraction"] == 0.20
    assert config["stage1_epochs"] == 3
    assert config["stage2_epochs"] == 2
    assert config["stage3_epoch_cap"] == 8
    assert config["organizer_sampling_fraction"] == 0.35
    assert config["stage3_replay_fraction"] == 0.20
    assert config["candidate_top_k"] == 20
    assert config["candidate_output_k"] == 1
    assert config["max_length"] == 512
    assert config["stride"] == 128
```

- [ ] **Step 2: Reproduce the root test-import failure**

Run from repository root:

```powershell
python -m pytest v2/tests/test_curriculum_config.py -q -p no:cacheprovider
```

Expected: FAIL because the new config keys do not exist; if import fails first, record `ModuleNotFoundError: clinical_nlp_lab`.

- [ ] **Step 3: Add root pytest configuration and exact defaults**

```ini
[pytest]
pythonpath = v2
testpaths = v2/tests
```

Add these exact values to `DEFAULT_CONFIG` and mirror them into `v2/artifacts/config.json`:

```python
"run_mode": "full",
"cv_folds": 5,
"challenge_size": 10,
"synthetic_holdout_fraction": 0.20,
"near_duplicate_threshold": 0.92,
"stage1_epochs": 3,
"stage2_epochs": 2,
"stage3_epoch_cap": 8,
"stage1_learning_rate": 3e-5,
"stage2_learning_rate": 2e-5,
"stage3_learning_rate": 8e-6,
"organizer_sampling_fraction": 0.35,
"stage3_replay_fraction": 0.20,
"candidate_top_k": 20,
```

- [ ] **Step 4: Run targeted and full tests**

```powershell
python -m pytest v2/tests/test_curriculum_config.py -q -p no:cacheprovider
python -m pytest v2/tests -q -p no:cacheprovider
```

Expected: targeted PASS; existing suite PASS from repository root.

- [ ] **Step 5: Commit**

```powershell
git add -- pytest.ini v2/clinical_nlp_lab/config.py v2/artifacts/config.json v2/tests/test_curriculum_config.py v2/README.md
git commit -m "test: make curriculum configuration reproducible"
```

### Task 2: Dataset, KB, manifest, and report provenance

**Files:**
- Create: `v2/clinical_nlp_lab/provenance.py`
- Create: `v2/tests/test_provenance.py`
- Modify: `v2/clinical_nlp_lab/artifacts.py`
- Modify: `v2/clinical_nlp_lab/dataset_quality.py`

**Interfaces:**
- Produces: `FingerprintSet`, `fingerprint_dataset()`, `report_envelope()`, `validate_current_reports()`.
- Consumed by: Tasks 3, 5, 8, 13, and 14.

- [ ] **Step 1: Write failing provenance tests**

```python
from clinical_nlp_lab.provenance import FingerprintSet, report_envelope, validate_current_reports


def test_stale_report_is_rejected():
    expected = FingerprintSet(dataset="data-a", manifest="manifest-a", kb="kb-a")
    report = report_envelope({"documents": 2200}, expected)
    report["fingerprints"]["dataset"] = "data-b"
    errors = validate_current_reports([report], expected)
    assert errors == ["report[0] dataset fingerprint mismatch: data-b != data-a"]


def test_two_current_reports_cannot_disagree():
    fp = FingerprintSet(dataset="d", manifest="m", kb="k")
    left = report_envelope({"documents": 2200}, fp)
    right = report_envelope({"documents": 2000}, fp)
    errors = validate_current_reports([left, right], fp)
    assert any("conflicting current report" in item for item in errors)
```

- [ ] **Step 2: Verify tests fail**

```powershell
python -m pytest v2/tests/test_provenance.py -q -p no:cacheprovider
```

Expected: FAIL with missing `clinical_nlp_lab.provenance`.

- [ ] **Step 3: Implement immutable provenance interfaces**

```python
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Iterable


@dataclass(frozen=True, slots=True)
class FingerprintSet:
    dataset: str
    manifest: str
    kb: str


def report_envelope(payload: dict[str, Any], fingerprints: FingerprintSet, status: str = "current") -> dict[str, Any]:
    if status not in {"current", "stale", "archived"}:
        raise ValueError(f"Unsupported report status: {status}")
    return {
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "fingerprints": asdict(fingerprints),
        "payload": payload,
    }


def validate_current_reports(reports: Iterable[dict[str, Any]], expected: FingerprintSet) -> list[str]:
    errors: list[str] = []
    current_payloads: list[dict[str, Any]] = []
    for index, report in enumerate(reports):
        if report.get("status") != "current":
            continue
        fingerprints = report.get("fingerprints", {})
        for field, expected_value in asdict(expected).items():
            actual = fingerprints.get(field)
            if actual != expected_value:
                errors.append(f"report[{index}] {field} fingerprint mismatch: {actual} != {expected_value}")
        current_payloads.append(report.get("payload", {}))
    if len({str(sorted(item.items())) for item in current_payloads}) > 1:
        errors.append("conflicting current report payloads for identical fingerprints")
    return errors
```

Implement `fingerprint_dataset(input_dir, gt_dir)`, `fingerprint_manifest(path)`, and `fingerprint_kb(paths)` with SHA-256 over sorted relative paths and bytes. Do not hash timestamps.

- [ ] **Step 4: Integrate envelopes into dataset/artifact reports and test**

```powershell
python -m pytest v2/tests/test_provenance.py v2/tests/test_dataset_quality.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/provenance.py v2/clinical_nlp_lab/artifacts.py v2/clinical_nlp_lab/dataset_quality.py v2/tests/test_provenance.py
git commit -m "feat: version dataset and audit reports"
```

### Task 3: Independent audit packet and two-reviewer gate

**Files:**
- Create: `v2/clinical_nlp_lab/audit.py`
- Create: `v2/tools/audit_dataset.py`
- Create: `v2/tests/test_audit_gate.py`

**Interfaces:**
- Consumes: `FingerprintSet`, dataset manifest JSONL.
- Produces: `build_audit_sample(records, seed)`, `validate_reviews(reviews, fingerprint)`, `validate_synthetic_profile(stats)` and `audit_manifest.json`.

- [ ] **Step 1: Write failing audit-gate tests**

```python
import pytest
from clinical_nlp_lab.audit import validate_reviews, validate_synthetic_profile


def test_two_distinct_reviewers_are_required():
    reviews = [
        {"reviewer_id": "agent-a", "dataset_fingerprint": "fp", "decisions": []},
        {"reviewer_id": "agent-a", "dataset_fingerprint": "fp", "decisions": []},
    ]
    with pytest.raises(ValueError, match="two distinct reviewers"):
        validate_reviews(reviews, "fp")


def test_review_fingerprint_must_match_current_dataset():
    reviews = [
        {"reviewer_id": "agent-a", "dataset_fingerprint": "old", "decisions": []},
        {"reviewer_id": "agent-b", "dataset_fingerprint": "old", "decisions": []},
    ]
    with pytest.raises(ValueError, match="fingerprint"):
        validate_reviews(reviews, "current")


def test_synthetic_profile_cannot_regress_beyond_five_percent():
    stats = {
        "documents": 2_000,
        "genre_count": 12,
        "mean_words": 413.8,
        "longtail_documents": 380,
        "unique_icd10": 391,
        "unique_rxnorm": 394,
        "fixed_exact_lines": 0,
        "validation_errors": 0,
    }
    assert validate_synthetic_profile(stats) == []

    stats["longtail_documents"] = 379
    assert validate_synthetic_profile(stats) == [
        "longtail_documents 379 < 380"
    ]
```

- [ ] **Step 2: Run tests and confirm failure**

```powershell
python -m pytest v2/tests/test_audit_gate.py -q -p no:cacheprovider
```

Expected: FAIL with missing audit module.

- [ ] **Step 3: Implement stratified packet and gate**

```python
def validate_reviews(reviews: list[dict], dataset_fingerprint: str) -> dict:
    reviewer_ids = {str(item.get("reviewer_id", "")) for item in reviews}
    if len(reviews) < 2 or len(reviewer_ids) < 2:
        raise ValueError("two distinct reviewers are required")
    mismatched = [item.get("reviewer_id") for item in reviews if item.get("dataset_fingerprint") != dataset_fingerprint]
    if mismatched:
        raise ValueError(f"review fingerprint mismatch for: {mismatched}")
    disagreements = []
    by_key: dict[tuple[str, int], list[dict]] = {}
    for review in reviews:
        for decision in review.get("decisions", []):
            key = (str(decision["document_id"]), int(decision["entity_index"]))
            by_key.setdefault(key, []).append(decision)
    for key, decisions in by_key.items():
        verdicts = {item["verdict"] for item in decisions}
        if len(verdicts) > 1:
            disagreements.append({"document_id": key[0], "entity_index": key[1], "decisions": decisions})
    return {"reviewer_ids": sorted(reviewer_ids), "disagreements": disagreements, "passed": not disagreements}


def validate_synthetic_profile(stats: dict[str, int | float]) -> list[str]:
    errors: list[str] = []
    exact_requirements = {"documents": 2_000, "genre_count": 12}
    minimums = {
        "longtail_documents": 380,
        "unique_icd10": 391,
        "unique_rxnorm": 394,
    }
    for key, expected in exact_requirements.items():
        actual = int(stats[key])
        if actual != expected:
            errors.append(f"{key} {actual} != {expected}")
    for key, minimum in minimums.items():
        actual = int(stats[key])
        if actual < minimum:
            errors.append(f"{key} {actual} < {minimum}")

    mean_words = float(stats["mean_words"])
    if not 393.11 <= mean_words <= 434.49:
        errors.append(f"mean_words {mean_words:.2f} outside [393.11, 434.49]")
    if int(stats["fixed_exact_lines"]) != 0:
        errors.append("fixed_exact_lines must be zero")
    if int(stats["validation_errors"]) != 0:
        errors.append("validation_errors must be zero")
    return errors
```

`build_audit_sample` must select at least one example for every genre, long-tail flag, entity type, assertion label, candidate-empty bucket, and rare-candidate bucket using deterministic seed 42.

The CLI must calculate and gate the approved v2 synthetic profile before it
emits the packet: exactly 2,000 documents and 12 genres; at least 380 long-tail
documents, 391 unique ICD-10 codes, and 394 unique RxNorm codes (the 5% lower
bound from the approved baseline, rounded upward); mean length within 5% of
413.8 words; no globally
repeated fixed line; and zero schema, span-offset, raw-KB, or age/sex consistency
errors.

- [ ] **Step 4: Test CLI and create a current audit packet without modifying GT**

```powershell
python v2/tools/audit_dataset.py --dataset data_v2/Training_data/synthetic_train_v2 --output scratch/current_audit_packet.json --seed 42
python -m pytest v2/tests/test_audit_gate.py -q -p no:cacheprovider
```

Expected: packet contains dataset fingerprint and strata coverage; tests PASS. During execution, dispatch two independent review agents against this packet and store their reports under `scratch/`; do not auto-apply label changes.

- [ ] **Step 5: Commit code only**

```powershell
git add -- v2/clinical_nlp_lab/audit.py v2/tools/audit_dataset.py v2/tests/test_audit_gate.py
git commit -m "feat: require independent current-dataset audits"
```

### Task 4: Record and patient-block boundaries

**Files:**
- Create: `v2/clinical_nlp_lab/records.py`
- Create: `v2/tests/test_record_boundaries.py`
- Modify: `v2/clinical_nlp_lab/schema.py`
- Modify: `v2/clinical_nlp_lab/dataset_quality.py`

**Interfaces:**
- Produces: `RecordSpan`, `detect_record_spans(raw_text)`, `validate_record_spans(raw_text, spans, entities)`.
- Consumed by: feature creation, assertion context, inference merge.

- [ ] **Step 1: Write failing boundary tests**

```python
from clinical_nlp_lab.records import detect_record_spans, validate_record_spans
from clinical_nlp_lab.schema import EntityAnnotation


def test_structural_patient_headers_create_non_overlapping_records():
    text = "BỆNH NHÂN 1\nĐau ngực.\nBỆNH NHÂN 2\nSốt cao."
    spans = detect_record_spans(text)
    assert [(item.patient_block_id, item.start, item.end) for item in spans] == [
        ("record-0001", 0, 22),
        ("record-0002", 22, len(text)),
    ]
    assert all(item.confidence == "high" for item in spans)


def test_entity_crossing_record_boundary_is_rejected():
    text = "BỆNH NHÂN 1\nĐau\nBỆNH NHÂN 2\nSốt"
    spans = detect_record_spans(text)
    entity = EntityAnnotation(text=text[15:30], type="SYMPTOM", position=(15, 30))
    assert validate_record_spans(text, spans, [entity]) == ["entity[0] crosses record boundary"]
```

- [ ] **Step 2: Verify failure**

```powershell
python -m pytest v2/tests/test_record_boundaries.py -q -p no:cacheprovider
```

Expected: FAIL with missing records module.

- [ ] **Step 3: Implement structural-only detection**

```python
from dataclasses import dataclass
import re

_RECORD_HEADER = re.compile(r"(?imu)^(?:#{1,3}\s*)?(?:bệnh\s+nhân|hồ\s+sơ|ca\s+bệnh)\s*(?:số|thứ)?\s*\d+\b[^\n]*\n?")


@dataclass(frozen=True, slots=True)
class RecordSpan:
    patient_block_id: str
    start: int
    end: int
    confidence: str


def detect_record_spans(raw_text: str) -> list[RecordSpan]:
    starts = [match.start() for match in _RECORD_HEADER.finditer(raw_text)]
    if not starts:
        return [RecordSpan("record-0001", 0, len(raw_text), "uncertain")]
    ends = starts[1:] + [len(raw_text)]
    return [RecordSpan(f"record-{index:04d}", start, end, "high") for index, (start, end) in enumerate(zip(starts, ends), 1)]
```

Implement validation so spans cover the document without overlap and no entity crosses a high-confidence boundary.

- [ ] **Step 4: Run boundary and dataset tests**

```powershell
python -m pytest v2/tests/test_record_boundaries.py v2/tests/test_dataset_quality.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/records.py v2/clinical_nlp_lab/schema.py v2/clinical_nlp_lab/dataset_quality.py v2/tests/test_record_boundaries.py
git commit -m "feat: preserve patient boundaries across the pipeline"
```

### Task 5: Near-duplicate groups, blind challenge, and grouped OOF

**Files:**
- Create: `v2/clinical_nlp_lab/splitting.py`
- Create: `v2/tests/test_oof_split.py`
- Modify: `v2/clinical_nlp_lab/dataset_quality.py`
- Deprecate only after migration: `v2/clinical_nlp_lab/data.py:264-321`

**Interfaces:**
- Produces: `DevelopmentSplit`, `SyntheticSplit`, `compute_near_duplicate_groups()`, `build_development_split()`, `build_synthetic_split()`.
- Consumed by: curriculum and experiments.

- [ ] **Step 1: Write failing split tests**

```python
from clinical_nlp_lab.splitting import build_development_split, build_synthetic_split


def test_every_oof_document_validates_once_and_challenge_never_trains():
    records = [
        {"document_id": str(101 + index), "hard_group": f"g-{index}", "stratum": f"s-{index % 5}"}
        for index in range(100)
    ]
    split = build_development_split(records, challenge_size=10, n_splits=5, seed=42)
    validation_ids = [item for fold in split.folds for item in fold.validation_ids]
    assert len(split.challenge_ids) == 10
    assert len(validation_ids) == 90
    assert len(set(validation_ids)) == 90
    assert not set(split.challenge_ids) & set(validation_ids)
    assert all(not set(split.challenge_ids) & set(fold.train_ids) for fold in split.folds)


def test_synthetic_development_split_is_grouped_and_exactly_eighty_twenty():
    records = [
        {
            "document_id": str(201 + index),
            "hard_group": f"syn-{index}",
            "stratum": f"genre-{index % 12}|longtail-{index % 5 == 0}",
        }
        for index in range(2_000)
    ]
    split = build_synthetic_split(records, holdout_fraction=0.20, seed=42)
    assert len(split.train_ids) == 1_600
    assert len(split.validation_ids) == 400
    assert not set(split.train_ids) & set(split.validation_ids)
```

- [ ] **Step 2: Run test and confirm failure**

```powershell
python -m pytest v2/tests/test_oof_split.py -q -p no:cacheprovider
```

Expected: FAIL with missing splitting module.

- [ ] **Step 3: Implement deterministic grouped allocation**

```python
from dataclasses import dataclass
import random


@dataclass(frozen=True, slots=True)
class FoldSplit:
    fold: int
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DevelopmentSplit:
    challenge_ids: tuple[str, ...]
    folds: tuple[FoldSplit, ...]


@dataclass(frozen=True, slots=True)
class SyntheticSplit:
    train_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]


def build_development_split(records: list[dict], challenge_size: int, n_splits: int, seed: int) -> DevelopmentSplit:
    rng = random.Random(seed)
    groups: dict[str, list[str]] = {}
    for record in records:
        groups.setdefault(str(record["hard_group"]), []).append(str(record["document_id"]))
    ordered = sorted(groups.items(), key=lambda item: (-len(item[1]), item[0]))
    rng.shuffle(ordered)
    challenge: list[str] = []
    remaining: list[list[str]] = []
    for _, ids in ordered:
        if len(challenge) + len(ids) <= challenge_size:
            challenge.extend(ids)
        else:
            remaining.append(ids)
    buckets: list[list[str]] = [[] for _ in range(n_splits)]
    for ids in sorted(remaining, key=lambda item: -len(item)):
        min(buckets, key=len).extend(ids)
    universe = {item for ids in remaining for item in ids}
    folds = tuple(FoldSplit(index, tuple(sorted(universe - set(bucket))), tuple(sorted(bucket))) for index, bucket in enumerate(buckets))
    if len(folds) != n_splits or any(not fold.validation_ids for fold in folds):
        raise ValueError("cannot create requested non-empty grouped OOF folds")
    return DevelopmentSplit(tuple(sorted(challenge)), folds)
```

Replace the size-only choices in the compact skeleton above with one shared,
deterministic grouped allocator. Aggregate a `Counter` of `stratum` values for
every hard group. For each target bucket, score a candidate group by
`abs(projected_size - target_size) / target_size + sum(abs(projected_share[s] -
global_share[s]))`; break ties by seeded stable rank and then group ID. Refuse a
split when whole hard groups cannot produce the exact requested cardinality.
Use target sizes `(10, 18, 18, 18, 18, 18)` for organizer challenge plus five
OOF validation folds, and `(1_600, 400)` for synthetic development train/holdout.

`compute_near_duplicate_groups` must normalize Unicode/case/whitespace, create
5-token shingles, compare records in the same or adjacent length buckets, and
union pairs whose Jaccard similarity is at least
`near_duplicate_threshold=0.92`. Do not block comparisons by genre: a copied
template with a changed genre still leaks. The split audit must additionally
list cross-source components. Every challenge-near synthetic member is excluded
from all development training; for OOF, synthetic members sharing a component
with that fold's validation IDs are excluded from that fold only. Persist these
exclusions in each fold manifest so final-fit may deliberately re-include all
2,000 synthetic records after model selection is frozen.

- [ ] **Step 4: Verify real organizer split and no hard leakage**

```powershell
python -m pytest v2/tests/test_oof_split.py v2/tests/test_grouped_split.py -q -p no:cacheprovider
python v2/tools/audit_dataset.py --dataset data_v2/Training_data/synthetic_train_v2 --check-splits --output scratch/split_audit.json
```

Expected: 10 challenge IDs, five 18-document validation folds covering the
other 90 organizer IDs exactly once, a 1,600/400 grouped synthetic split, zero
hard-group leakage, and explicit fold-wise cross-source exclusions.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/splitting.py v2/clinical_nlp_lab/dataset_quality.py v2/clinical_nlp_lab/data.py v2/tests/test_oof_split.py v2/tests/test_grouped_split.py
git commit -m "feat: add leakage-safe organizer OOF splits"
```

### Task 6: Record-aware owner-window supervision

**Files:**
- Modify: `v2/clinical_nlp_lab/training.py`
- Modify: `v2/tests/test_chunking_policy.py`
- Create: `v2/tests/test_owner_window.py`

**Interfaces:**
- Produces: `assign_owner_windows(offset_mappings, entities) -> dict[int, int]` and record-aware `prepare_token_classification_features()`.
- Consumed by: every NER training stage.

- [ ] **Step 1: Write the failing duplicate-supervision test**

```python
from clinical_nlp_lab.schema import EntityAnnotation
from clinical_nlp_lab.training import assign_owner_windows, character_spans_to_bio


def test_complete_entity_contributes_loss_in_exactly_one_window():
    entity = EntityAnnotation(text="đau ngực", type="SYMPTOM", position=(20, 29))
    offsets = [[(0, 0), (18, 22), (23, 29), (0, 0)], [(0, 0), (20, 24), (25, 29), (0, 0)]]
    owners = assign_owner_windows(offsets, [entity])
    labels = [
        character_spans_to_bio(window, [entity], {"O": 0, "B-SYMPTOM": 1, "I-SYMPTOM": 2}, owned_entity_indices={0} if owners[0] == index else set())
        for index, window in enumerate(offsets)
    ]
    assert sum(any(value in {1, 2} for value in item) for item in labels) == 1
```

- [ ] **Step 2: Run test to verify failure**

```powershell
python -m pytest v2/tests/test_owner_window.py -q -p no:cacheprovider
```

Expected: FAIL because ownership is not implemented.

- [ ] **Step 3: Implement ownership scoring**

```python
def assign_owner_windows(offset_mappings, entities):
    owners: dict[int, int] = {}
    for entity_index, entity in enumerate(entities):
        candidates: list[tuple[int, int]] = []
        for window_index, offsets in enumerate(offset_mappings):
            visible = [(start, end) for start, end in offsets if end > start]
            if not visible:
                continue
            chunk_start = min(start for start, _ in visible)
            chunk_end = max(end for _, end in visible)
            if chunk_start <= entity.start and entity.end <= chunk_end:
                margin = min(entity.start - chunk_start, chunk_end - entity.end)
                candidates.append((margin, -window_index))
        if not candidates:
            raise ValueError(f"entity[{entity_index}] has no complete token window")
        _, negative_index = max(candidates)
        owners[entity_index] = -negative_index
    return owners
```

Change `character_spans_to_bio` to mask complete non-owner copies with `-100`. Tokenize each high-confidence record independently, then add the record’s raw start offset back to tokenizer offsets.

- [ ] **Step 4: Run all chunking tests**

```powershell
python -m pytest v2/tests/test_chunking_policy.py v2/tests/test_owner_window.py v2/tests/test_record_boundaries.py -q -p no:cacheprovider
```

Expected: PASS; every entity has exactly one owner.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/training.py v2/tests/test_chunking_policy.py v2/tests/test_owner_window.py
git commit -m "feat: assign one loss-owning window per entity"
```

### Task 7: Document-level entity metrics and checkpoint selection

**Files:**
- Create: `v2/clinical_nlp_lab/metrics.py`
- Create: `v2/tests/test_selection_metrics.py`
- Modify: `v2/clinical_nlp_lab/training.py`
- Modify: `v2/scripts/train_ner_subprocess.py`

**Interfaces:**
- Produces: `compute_document_metrics()`, `selection_score()`, `build_document_compute_metrics()`.
- Consumed by: Hugging Face Trainer and experiment reports.

- [ ] **Step 1: Write failing selection tests**

```python
from clinical_nlp_lab.metrics import selection_score


def test_selection_score_uses_approved_weights():
    metrics = {"exact_f1": 0.80, "overlap_f1": 0.90, "macro_type_f1": 0.70}
    assert selection_score(metrics) == 0.81


def test_bad_boundaries_cannot_hide_behind_token_f1():
    metrics = {"exact_f1": 0.20, "overlap_f1": 0.90, "macro_type_f1": 0.40}
    assert selection_score(metrics) == 0.36
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_selection_metrics.py -q -p no:cacheprovider
```

Expected: FAIL with missing metrics module.

- [ ] **Step 3: Implement exact/overlap/macro metric aggregation**

```python
def selection_score(metrics: dict[str, float]) -> float:
    return round(0.70 * metrics["exact_f1"] + 0.20 * metrics["overlap_f1"] + 0.10 * metrics["macro_type_f1"], 12)
```

`compute_document_metrics` must use one-to-one matching per document and type; exact requires identical spans, overlap requires positive intersection, and macro type F1 averages all five official types including types with zero predictions.

Replace Trainer configuration with:

```python
training_kwargs["metric_for_best_model"] = "selection_score"
training_kwargs["greater_is_better"] = True
trainer_kwargs["compute_metrics"] = build_document_compute_metrics(validation_features, validation_documents, id_to_label)
```

- [ ] **Step 4: Run metric and training-helper tests**

```powershell
python -m pytest v2/tests/test_selection_metrics.py v2/tests/test_entity_metrics.py v2/tests/test_training_helpers.py -q -p no:cacheprovider
```

Expected: PASS; token metrics remain diagnostics only.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/metrics.py v2/clinical_nlp_lab/training.py v2/scripts/train_ner_subprocess.py v2/tests/test_selection_metrics.py
git commit -m "feat: select NER checkpoints by entity metrics"
```

### Task 8: Source-aware sampling and synthetic replay

**Files:**
- Create: `v2/clinical_nlp_lab/sampling.py`
- Create: `v2/tests/test_source_sampling.py`
- Modify: `v2/clinical_nlp_lab/training.py`

**Interfaces:**
- Produces: `source_weights()`, `select_replay_documents()`, `SourceAwareTrainer`.
- Consumed by: curriculum stages 2–3.

- [ ] **Step 1: Write failing deterministic-sampling tests**

```python
from clinical_nlp_lab.sampling import select_replay_documents, source_weights


def test_organizer_chunk_weight_targets_35_percent_exposure():
    weights = source_weights(["organizer"] * 10 + ["synthetic"] * 90, organizer_fraction=0.35)
    organizer_mass = sum(weights[:10]) / sum(weights)
    assert round(organizer_mass, 6) == 0.35


def test_replay_prefers_longtail_and_is_deterministic():
    records = [{"document_id": str(i), "long_tail": i < 4, "rarity": 1.0 if i < 4 else 0.0} for i in range(20)]
    first = select_replay_documents(records, count=4, seed=42)
    second = select_replay_documents(records, count=4, seed=42)
    assert first == second
    assert all(int(item) < 4 for item in first)
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_source_sampling.py -q -p no:cacheprovider
```

Expected: FAIL with missing sampling module.

- [ ] **Step 3: Implement weights and replay ranking**

```python
def source_weights(sources: list[str], organizer_fraction: float) -> list[float]:
    organizer_count = sum(item == "organizer" for item in sources)
    synthetic_count = sum(item == "synthetic" for item in sources)
    if not organizer_count or not synthetic_count:
        return [1.0] * len(sources)
    organizer_weight = organizer_fraction / organizer_count
    synthetic_weight = (1.0 - organizer_fraction) / synthetic_count
    return [organizer_weight if item == "organizer" else synthetic_weight for item in sources]


def select_replay_documents(records: list[dict], count: int, seed: int) -> list[str]:
    ordered = sorted(records, key=lambda item: (-int(bool(item.get("long_tail"))), -float(item.get("rarity", 0.0)), str(item["document_id"])))
    return [str(item["document_id"]) for item in ordered[:count]]
```

`SourceAwareTrainer._get_train_sampler` must return `torch.utils.data.WeightedRandomSampler` seeded by stage/fold seed. Log realized organizer/synthetic chunk and document exposure after each epoch.

- [ ] **Step 4: Run sampling tests**

```powershell
python -m pytest v2/tests/test_source_sampling.py v2/tests/test_training_helpers.py -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/sampling.py v2/clinical_nlp_lab/training.py v2/tests/test_source_sampling.py
git commit -m "feat: balance organizer exposure and synthetic replay"
```

### Task 9: Three-stage curriculum orchestration and resume safety

**Files:**
- Create: `v2/clinical_nlp_lab/curriculum.py`
- Create: `v2/scripts/train_curriculum.py`
- Create: `v2/tests/test_curriculum_orchestration.py`
- Modify: `v2/clinical_nlp_lab/artifacts.py`

**Interfaces:**
- Consumes: development split, owner-window features, source sampling, selection metrics.
- Produces: `run_curriculum() -> CurriculumResult`, stage manifests, OOF predictions, and final model.

- [ ] **Step 1: Write failing stage-order and cache tests**

```python
from clinical_nlp_lab.curriculum import StageManifest, next_required_stage


def test_stage_order_and_hash_safe_resume():
    manifests = [StageManifest(name="stage1", status="complete", input_hash="abc", output_dir="stage1")]
    assert next_required_stage(manifests, expected_input_hash="abc") == "stage2"
    assert next_required_stage(manifests, expected_input_hash="changed") == "stage1"
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_curriculum_orchestration.py -q -p no:cacheprovider
```

Expected: FAIL with missing curriculum module.

- [ ] **Step 3: Implement stage contracts**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageManifest:
    name: str
    status: str
    input_hash: str
    output_dir: str


def next_required_stage(manifests: list[StageManifest], expected_input_hash: str) -> str:
    complete = {item.name: item for item in manifests if item.status == "complete" and item.input_hash == expected_input_hash}
    for name in ("stage1", "stage2", "stage3", "assertion", "linking", "final_fit"):
        if name not in complete:
            return name
    return "done"
```

`run_curriculum` must:

1. train/cache Stage 1 on synthetic train;
2. initialize each fold Stage 2 from Stage 1;
3. initialize each fold Stage 3 from its Stage 2 best checkpoint;
4. write fold OOF predictions;
5. choose frozen epochs/LRs from aggregate OOF;
6. run final Stage 1 on all 2.000 synthetic, Stage 2 on all organizer+synthetic, Stage 3 on all organizer+replay;
7. atomically write each manifest only after reload smoke test.

- [ ] **Step 4: Run a dependency-free fake-trainer integration test**

```powershell
python -m pytest v2/tests/test_curriculum_orchestration.py -q -p no:cacheprovider
python v2/scripts/train_curriculum.py --help
```

Expected: tests PASS; CLI exposes `--run-mode`, `--train-source`, `--model-source`, `--output-dir`, `--config-path`, `--fast-dev-run`.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/curriculum.py v2/clinical_nlp_lab/artifacts.py v2/scripts/train_curriculum.py v2/tests/test_curriculum_orchestration.py
git commit -m "feat: orchestrate resumable three-stage NER training"
```

### Task 10: Trainable multi-label assertion component

**Files:**
- Create: `v2/clinical_nlp_lab/assertion_training.py`
- Create: `v2/tests/test_assertion_training.py`
- Modify: `v2/clinical_nlp_lab/assertions.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py`
- Modify: `v2/clinical_nlp_lab/artifacts.py`

**Interfaces:**
- Produces: `build_assertion_labels()`, `calibrate_assertion_thresholds()`, `TrainedAssertionPredictor`.
- Consumed by: curriculum, final inference, experiment evaluation.

- [ ] **Step 1: Write failing label and calibration tests**

```python
from clinical_nlp_lab.assertion_training import build_assertion_labels, calibrate_assertion_thresholds


def test_assertion_labels_are_three_independent_axes():
    assert build_assertion_labels(["isNegated", "isFamily"]) == [1.0, 0.0, 1.0]


def test_lab_entities_are_not_assertion_examples():
    assert build_assertion_labels([], entity_type="LAB_NAME") is None


def test_thresholds_are_calibrated_per_axis():
    probabilities = [[0.9, 0.2, 0.8], [0.1, 0.7, 0.2]]
    labels = [[1, 0, 1], [0, 1, 0]]
    thresholds = calibrate_assertion_thresholds(probabilities, labels)
    assert set(thresholds) == {"isNegated", "isHistorical", "isFamily"}
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_assertion_training.py -q -p no:cacheprovider
```

Expected: FAIL with missing assertion training module.

- [ ] **Step 3: Implement pure label/calibration layer**

```python
ASSERTION_ORDER = ("isNegated", "isHistorical", "isFamily")
ASSERTION_ENTITY_TYPES = {"DISEASE", "DRUG", "SYMPTOM"}


def build_assertion_labels(assertions, entity_type="DISEASE"):
    if entity_type not in ASSERTION_ENTITY_TYPES:
        return None
    values = set(assertions)
    return [float(label in values) for label in ASSERTION_ORDER]
```

Calibrate each label over thresholds generated by `[value / 100 for value in range(5, 100, 5)]`, choose maximum F1, and break ties by the higher threshold. Use `AutoModelForSequenceClassification(num_labels=3, problem_type="multi_label_classification")`; add `<ENT>` and `</ENT>` tokenizer tokens; train fold models for OOF probabilities and a final model with frozen epochs.

- [ ] **Step 4: Integrate predictor with rule fallback and run tests**

```powershell
python -m pytest v2/tests/test_assertion_training.py v2/tests/test_assertion_scope.py v2/tests/test_assertion_vllm_compat.py -q -p no:cacheprovider
```

Expected: trained predictor used when artifact exists; scoped rule predictor remains deterministic fallback; all PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/assertion_training.py v2/clinical_nlp_lab/assertions.py v2/clinical_nlp_lab/pipeline.py v2/clinical_nlp_lab/artifacts.py v2/tests/test_assertion_training.py
git commit -m "feat: train and calibrate clinical assertions"
```

### Task 11: KB candidate contract and 100% organizer-gold coverage

**Files:**
- Create: `v2/clinical_nlp_lab/kb_contract.py`
- Create: `v2/tests/test_kb_contract.py`
- Modify: `v2/clinical_nlp_lab/kb.py`
- Modify: `v2/tools/build_knowledge_bases.py`
- Modify: `v2/clinical_nlp_lab/retrieval.py`
- Modify: `v2/clinical_nlp_lab/candidate_policy.py`

**Interfaces:**
- Produces: `CandidateIdentity`, `audit_gold_candidate_coverage()`, `official_output_id()`.
- Consumed by: retrieval, pipeline, quality gate, notebook.

- [ ] **Step 1: Write failing candidate-contract tests**

```python
from clinical_nlp_lab.kb_contract import CandidateIdentity, audit_gold_candidate_coverage


def test_icd_display_marker_is_preserved_separately():
    identity = CandidateIdentity(canonical_id="K93.1", official_display_id="K93.1*")
    assert identity.canonical_id == "K93.1"
    assert identity.official_display_id == "K93.1*"


def test_empty_gold_candidate_is_valid_abstention():
    report = audit_gold_candidate_coverage([{"document_id": "101", "candidates": []}], {"K93.1"})
    assert report["missing"] == []
    assert report["abstentions"] == 1
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_kb_contract.py -q -p no:cacheprovider
```

Expected: FAIL with missing KB contract module.

- [ ] **Step 3: Implement identity and fail-closed coverage**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    canonical_id: str
    official_display_id: str


def canonical_icd_id(candidate_id: str) -> str:
    return candidate_id.strip().rstrip("*")


def audit_gold_candidate_coverage(entities: list[dict], runtime_ids: set[str]) -> dict:
    missing: list[dict] = []
    abstentions = 0
    for entity in entities:
        candidates = [str(item) for item in entity.get("candidates", [])]
        if not candidates:
            abstentions += 1
            continue
        for candidate in candidates:
            if canonical_icd_id(candidate) not in runtime_ids and candidate not in runtime_ids:
                missing.append({"document_id": str(entity["document_id"]), "candidate": candidate})
    return {"missing": missing, "abstentions": abstentions, "coverage": 1.0 if not missing else 0.0}
```

Extend runtime records with `canonical_id`, `official_display_ids`, raw source and aliases. Fix lexical-only retrieval so it never calls a missing embedding model/FAISS index.

- [ ] **Step 4: Build KB and verify organizer gold coverage**

```powershell
python v2/tools/build_knowledge_bases.py --help
python -m pytest v2/tests/test_kb_contract.py v2/tests/test_candidate_policy.py v2/tests/test_retrieval_resources.py -q -p no:cacheprovider
```

Expected: tests PASS. The real build gate must report zero missing non-empty organizer candidates before Task 14.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/kb_contract.py v2/clinical_nlp_lab/kb.py v2/clinical_nlp_lab/retrieval.py v2/clinical_nlp_lab/candidate_policy.py v2/tools/build_knowledge_bases.py v2/tests/test_kb_contract.py
git commit -m "feat: enforce organizer candidate coverage"
```

### Task 12: Precision-controlled KB-first mention recovery

**Files:**
- Create: `v2/clinical_nlp_lab/mention_recovery.py`
- Create: `v2/tests/test_mention_recovery.py`
- Modify: `v2/clinical_nlp_lab/ner.py`
- Modify: `v2/clinical_nlp_lab/pipeline.py`
- Modify: `v2/clinical_nlp_lab/evaluation.py`

**Interfaces:**
- Produces: `MentionProposal`, `filter_mention_proposals()`, `incremental_recovery_metrics()`.
- Consumed by: hybrid inference and ablation runner.

- [ ] **Step 1: Write failing ambiguity and incremental-metric tests**

```python
from clinical_nlp_lab.mention_recovery import MentionProposal, filter_mention_proposals, incremental_recovery_metrics


def test_ambiguous_single_token_alias_is_rejected_without_context():
    proposal = MentionProposal(start=0, end=2, entity_type="DISEASE", candidate_id="I21", alias="MI", score=0.99, ambiguous=True)
    assert filter_mention_proposals("MI bình thường", [proposal]) == []


def test_incremental_metrics_count_new_true_and_false_mentions():
    metrics = incremental_recovery_metrics(gold={(0, 4, "DRUG")}, ner=set(), hybrid={(0, 4, "DRUG"), (8, 10, "DRUG")})
    assert metrics == {"incremental_tp": 1, "incremental_fp": 1, "incremental_precision": 0.5, "incremental_recall": 1.0}
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_mention_recovery.py -q -p no:cacheprovider
```

Expected: FAIL with missing mention recovery module.

- [ ] **Step 3: Implement proposal contract and hard filters**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class MentionProposal:
    start: int
    end: int
    entity_type: str
    candidate_id: str
    alias: str
    score: float
    ambiguous: bool = False


def filter_mention_proposals(raw_text: str, proposals: list[MentionProposal]) -> list[MentionProposal]:
    accepted = []
    for item in proposals:
        if item.start < 0 or item.end > len(raw_text) or item.start >= item.end:
            continue
        if raw_text[item.start:item.end].casefold() != item.alias.casefold():
            continue
        left_ok = item.start == 0 or not raw_text[item.start - 1].isalnum()
        right_ok = item.end == len(raw_text) or not raw_text[item.end].isalnum()
        if not left_ok or not right_ok or item.ambiguous:
            continue
        accepted.append(item)
    return accepted
```

Merge proposals with NER only after filtering; preserve `patient_block_id`; never merge across records. Keep generic regex disabled.

- [ ] **Step 4: Run detector, pipeline and evaluation tests**

```powershell
python -m pytest v2/tests/test_mention_recovery.py v2/tests/test_ner_policy.py v2/tests/test_candidate_policy.py -q -p no:cacheprovider
```

Expected: PASS; KB-first false positives are visible in diagnostics.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/mention_recovery.py v2/clinical_nlp_lab/ner.py v2/clinical_nlp_lab/pipeline.py v2/clinical_nlp_lab/evaluation.py v2/tests/test_mention_recovery.py
git commit -m "feat: recover KB mentions with precision gates"
```

### Task 13: Baseline, ablation matrix, OOF aggregation, and challenge lock

**Files:**
- Create: `v2/clinical_nlp_lab/experiments.py`
- Create: `v2/tests/test_experiment_manifest.py`
- Modify: `v2/clinical_nlp_lab/evaluation.py`

**Interfaces:**
- Produces: `ExperimentManifest`, `approved_variants()`, `aggregate_oof()`, `lock_challenge_evaluation()`.
- Consumed by: curriculum and notebook.

- [ ] **Step 1: Write failing manifest tests**

```python
import pytest
from clinical_nlp_lab.experiments import approved_variants, lock_challenge_evaluation


def test_six_approved_variants_are_fixed():
    assert [item.name for item in approved_variants()] == [
        "current_ner", "owner_window", "curriculum", "curriculum_assertion", "curriculum_kb_first", "full_pipeline"
    ]


def test_challenge_cannot_run_twice_for_same_experiment():
    manifest = {"experiment_id": "exp-1", "challenge_evaluated": True}
    with pytest.raises(ValueError, match="already evaluated"):
        lock_challenge_evaluation(manifest)
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_experiment_manifest.py -q -p no:cacheprovider
```

Expected: FAIL with missing experiments module.

- [ ] **Step 3: Implement fixed variants and aggregation**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExperimentVariant:
    name: str
    owner_window: bool
    curriculum: bool
    assertion_model: bool
    kb_first: bool
    qwen: bool


def approved_variants():
    return (
        ExperimentVariant("current_ner", False, False, False, False, False),
        ExperimentVariant("owner_window", True, False, False, False, False),
        ExperimentVariant("curriculum", True, True, False, False, False),
        ExperimentVariant("curriculum_assertion", True, True, True, False, False),
        ExperimentVariant("curriculum_kb_first", True, True, False, True, False),
        ExperimentVariant("full_pipeline", True, True, True, True, False),
    )
```

Aggregate OOF mean/std/worst-fold for exact/overlap/macro, assertion F1, candidate accuracy/coverage and incremental recovery. All variants must use the same decoder and candidate policy. Record git commit, dataset/KB fingerprints, seed, fold IDs and runtime.

- [ ] **Step 4: Run tests and serialize a fake manifest**

```powershell
python -m pytest v2/tests/test_experiment_manifest.py v2/tests/test_entity_metrics.py -q -p no:cacheprovider
```

Expected: PASS; JSON serialization contains six variants and challenge state.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/clinical_nlp_lab/experiments.py v2/clinical_nlp_lab/evaluation.py v2/tests/test_experiment_manifest.py
git commit -m "feat: track OOF ablations and blind challenge"
```

### Task 14: Kaggle notebook orchestration and artifact contract

**Files:**
- Modify: `v2/tools/build_kaggle_notebook.py`
- Modify: `v2/medical_information_extraction_kaggle.ipynb`
- Modify: `v2/clinical_nlp_lab/artifacts.py`
- Modify: `v2/tests/test_kaggle_observability.py`
- Create: `v2/tests/test_curriculum_notebook.py`
- Modify: `v2/KAGGLE_RUNBOOK.md`
- Modify: `v2/README.md`

**Interfaces:**
- Consumes: `train_curriculum.py`, experiment/audit/provenance artifacts, final NER/assertion/linking configuration.
- Produces: a single Run All flow with `full`, `resume`, `inference_only` modes and required ZIPs.

- [ ] **Step 1: Write failing notebook-contract tests**

```python
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_training_notebook_exposes_all_run_modes_and_stage_artifacts():
    source = (ROOT / "tools" / "build_kaggle_notebook.py").read_text(encoding="utf-8")
    for token in (
        'RUN_MODE = "full"', '"resume"', '"inference_only"', "train_curriculum.py",
        "stage1_synthetic_checkpoint", "stage2_mixed_checkpoint", "final_ner_model",
        "assertion_model", "candidate_calibration.json", "oof_predictions.jsonl",
        "experiment_manifest.json", "audit_manifest.json", "kb_coverage_report.json",
    ):
        assert token in source
```

- [ ] **Step 2: Confirm failure**

```powershell
python -m pytest v2/tests/test_curriculum_notebook.py -q -p no:cacheprovider
```

Expected: FAIL because the current notebook has one-stage training.

- [ ] **Step 3: Replace one-stage training cells with curriculum CLI orchestration**

The generated notebook must construct the subprocess command exactly from `RUN_MODE`, paths and `FAST_DEV_RUN`, stream logs, validate the returned stage manifest, reload `final_ner_model`, load assertion thresholds and candidate calibration, then run inference. Use `subprocess.Popen` argument lists, not shell strings.

Required mode behavior:

```python
if RUN_MODE not in {"full", "resume", "inference_only"}:
    raise ValueError(f"Unsupported RUN_MODE: {RUN_MODE}")
if RUN_MODE == "inference_only" and not FINAL_MODEL_DIR.is_dir():
    raise FileNotFoundError(f"Final model not found: {FINAL_MODEL_DIR}")
```

Package only reloadable final artifacts and reports; remove nested `checkpoint-*` directories after successful reload and before ZIP creation.

- [ ] **Step 4: Regenerate notebook and run notebook/full suite tests**

```powershell
python v2/tools/build_kaggle_notebook.py --output v2/medical_information_extraction_kaggle.ipynb
python -m pytest v2/tests/test_curriculum_notebook.py v2/tests/test_kaggle_observability.py v2/tests/test_inference_notebook.py -q -p no:cacheprovider
python -m pytest v2/tests -q -p no:cacheprovider
```

Expected: notebook compiles, matches generator output and full suite PASS.

- [ ] **Step 5: Commit**

```powershell
git add -- v2/tools/build_kaggle_notebook.py v2/medical_information_extraction_kaggle.ipynb v2/clinical_nlp_lab/artifacts.py v2/tests/test_curriculum_notebook.py v2/tests/test_kaggle_observability.py v2/KAGGLE_RUNBOOK.md v2/README.md
git commit -m "feat: run the full curriculum pipeline on Kaggle"
```

### Task 15: Final verification, audits, and Kaggle acceptance

**Files:**
- Modify only if verification exposes a defect: files owned by the failing task.
- Produce without committing large binaries: `scratch/`, `/kaggle/working/`, and dataset report artifacts.

**Interfaces:**
- Consumes: all prior tasks.
- Produces: verification evidence, valid `output.zip`, reloadable model archive and audited run manifest.

- [ ] **Step 1: Run all deterministic checks from repository root**

```powershell
python -m pytest v2/tests -q -p no:cacheprovider
python v2/tools/audit_dataset.py --dataset data_v2/Training_data/synthetic_train_v2 --check-splits --check-kb --output scratch/final_data_audit.json
python v2/tools/build_kaggle_notebook.py --output scratch/generated_kaggle_check.ipynb
```

Expected: all tests PASS; zero structural/offset/hard-leakage/runtime-KB errors; generated notebook validates.

- [ ] **Step 2: Complete and verify two independent current-dataset reviews**

Use two independent agents with the same immutable `scratch/current_audit_packet.json`. Each returns JSON containing `reviewer_id`, `dataset_fingerprint`, per-entity verdict/evidence and no file mutations. Run:

```powershell
python v2/tools/audit_dataset.py --dataset data_v2/Training_data/synthetic_train_v2 --review scratch/reviewer_a.json --review scratch/reviewer_b.json --output scratch/audit_manifest.json
```

Expected: two distinct reviewers, matching fingerprint, disagreements explicitly queued. Any confirmed error is repaired through a separate TDD task, then both audit reports are regenerated for the new fingerprint.

- [ ] **Step 3: Run fast development end-to-end smoke**

```powershell
python v2/scripts/train_curriculum.py --run-mode full --train-source data_v2/Training_data/synthetic_train_v2 --model-source artifacts/tiny_xlmr --output-dir scratch/curriculum_smoke --config-path v2/artifacts/config.json --fast-dev-run true
```

Expected: Stage 1→2→3→assertion→linking→final-fit logical flow completes, final model reloads, no stale manifest is accepted, and no model download is required when the tiny local fixture is supplied.

- [ ] **Step 4: Run post-training inference and archive validation**

```powershell
python v2/tools/run_pipeline.py --help
python -m pytest v2/tests/test_curriculum_notebook.py v2/tests/test_inference_notebook.py -q -p no:cacheprovider
```

Expected: every output JSON passes schema/offset validation; each non-empty candidate exists in runtime KB; `output.zip` contains only `output/<document_id>.json`; model ZIP CRC passes and contains no `checkpoint-*` directories.

- [ ] **Step 5: Run one real Kaggle GPU acceptance session**

Upload/import `v2/medical_information_extraction_kaggle.ipynb`, attach the current dataset/KB/model inputs, choose a T4 or P100, set `RUN_MODE="full"`, and select **Run All**. Download and audit:

```text
/kaggle/working/output.zip
/kaggle/working/trained_ner_artifacts.zip
/kaggle/working/run_manifest.json
/kaggle/working/evaluation_report.json
/kaggle/working/experiment_manifest.json
/kaggle/working/diagnostics/run_summary.json
```

Expected: run completes without manual cell intervention; manifests match dataset/KB fingerprints; final checkpoint reload is recorded; ZIP CRC and inventory pass. If no Kaggle session is available, report this exact external blocker and do not mark the implementation complete.

- [ ] **Step 6: Confirm the source tree is clean after verification**

```powershell
git status --short
```

Expected: no uncommitted tracked source changes from Tasks 1–14. If verification exposes a defect, return to the task that owns the failing file, add a failing regression test, implement the minimal fix, rerun that task’s verification, and commit only the explicit paths listed by that task. Do not stage `scratch/`, `data_v2/`, model weights, user documents, unrelated untracked files or Kaggle downloads.

## Completion Gate

The implementation is complete only when:

- every Task 1–14 commit exists and the full root-level test command passes;
- Task 15 deterministic audit, two-agent audit and local fast-dev smoke pass;
- the v2 synthetic-profile gate passes without exceeding the approved 5%
  relative degradation and with zero hard validation errors;
- non-empty organizer gold candidate runtime coverage is 100%;
- OOF/challenge and final-fit manifests match the approved split contract;
- final model and assertion model reload offline;
- Kaggle `Run All` evidence is available and audited.

If the final Kaggle session is unavailable, source implementation may be reported as locally verified, but the overall goal remains externally blocked rather than complete.
