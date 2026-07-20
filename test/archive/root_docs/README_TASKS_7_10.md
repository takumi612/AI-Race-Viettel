# Task 7–10 Precision-first Clinical NLP Implementation Plan

*Kế hoạch bàn giao bằng tiếng Việt cho người triển khai tiếp.*

> **Cập nhật 2026-07-19:** Task 7–10 trong tài liệu này đã được triển khai tại
> commit `d0cae07` trên branch `develop`. Các checkbox bên dưới được giữ như
> lịch sử thiết kế/review, không phải backlog còn mở. Phase A của pipeline
> huấn luyện modular (contracts, validation, split, projections và atomic
> dataset build) được triển khai sau đó; hướng dẫn vận hành nằm tại
> `docs/training/DATA_FOUNDATION.md`.

> **Dành cho agent/người triển khai tiếp:** dùng `superpowers:subagent-driven-development` (khuyến nghị) hoặc `superpowers:executing-plans`; thực hiện từng task bằng TDD, commit riêng và review độc lập trước khi sang task kế tiếp.

**Mục tiêu:** hoàn thiện BM25-first hybrid retrieval, candidate selection ưu tiên precision, benchmark trên nhãn đáng tin cậy, khóa cấu hình, chạy holdout đúng một lần và tạo submission hợp lệ cho 100 file public.

**Kiến trúc:** Task 7 tạo scored retrieval và fusion có tổng trọng số bằng 1. Task 8 lọc ứng viên bằng clinical predicate rồi chọn Top-1/Top-2/reject. Task 9 đo từng stage trên development folds, khóa config trước holdout và tích hợp CLI. Task 10 review toàn nhánh, chạy inference public, validate và đóng gói artefact cuối.

**Tech stack:** Python 3, pytest, SQLite, `bm25s`, FAISS (optional/fallback BM25), NumPy, PyTorch/Transformers hiện có trong project.

## 1. Trạng thái bàn giao

- Worktree: `D:\AI Race Viettel\.worktrees\precision-first-pipeline`
- Branch hiện tại: `develop`
- Baseline lịch sử của Task 7: `ae9f610a9bbfcb44eac51e4719768e9a1f1c6740`
- Task 7–10 đã hoàn tất tại `d0cae07`; baseline đó có **200 passed**.
- Training data foundation Phase A được thiết kế tại `f9b20cb`, lập kế hoạch
  tại `fd79f4f` và triển khai bằng các checkpoint TDD sau đó. Verification
  Phase A: **245 passed**, metric self-test và production path/override audit
  đều đạt.
- Input public: `D:\AI Race Viettel\data\input` (100 file); bản copy ignored đã có trong worktree ở `data/input`.
- Ground truth: ID 1–100 là pseudo-GT tự tạo, **không được dùng để chọn cấu hình**; ID 101–200 là nhãn được cung cấp.
- Development pool đáng tin cậy: ID 101–180.
- Holdout khóa: ID 181–200; không đọc để tune rule/threshold/alpha.
- Hạn chế đã biết, không phải blocker: văn bản NFD canonically equivalent có thể miss lexicon NFC; offset/Unicode boundary với dữ liệu NFC hiện tại đã được test.

Chạy preflight trước khi sửa code:

```powershell
Set-Location 'D:\AI Race Viettel\.worktrees\precision-first-pipeline'
git status --short
git rev-parse HEAD
python -m pytest -q
```

Kỳ vọng hiện tại: branch `develop`, worktree sạch sau checkpoint cuối và full
suite không regression. Không reset về `ae9f610`; đó chỉ là baseline lịch sử.

## Global Constraints

- Metric chính cho entity/assertion là exact-span precision, recall và **F0.5**; khi phải đánh đổi, ưu tiên precision.
- Candidate mapping dùng Candidate Jaccard làm metric chọn; candidate precision, Top-1 hit rate và Recall@20 là diagnostic theo stage.
- Fusion duy nhất:

  ```text
  fusion_score = alpha * normalized_bm25
               + (1 - alpha) * normalized_semantic
  ```

  `alpha` nằm trong `[0, 1]`, mặc định `0.75`; `alpha + (1 - alpha) == 1`. Không thêm exact-match, dose-form hoặc RRF bonus ngoài công thức.
- BM25 và semantic đều lấy internal Top-20; chỉ slice output sau fusion/validation/selection.
- Mọi output entity phải thỏa `document[start:end] == entity["text"]`.
- Không tạo entity tổng hợp, không fixed Top-5; candidate output tối đa 2 và được phép là `[]`.
- Chỉ load `src/resources/verified_overrides.json`; `data/kb/override_dict.json` là legacy bị quarantine.
- Rule/resource mới phải strict-schema, UTF-8, có provenance/source, injectable và immutable sau load.
- LLM reranker mặc định tắt; nếu bật chỉ được trả về tập con của candidate pool, có timeout và deterministic fallback.
- Không dùng API thương mại, không fine-tune model mới, không tối ưu riêng theo từng file public.
- Mỗi task: RED → GREEN → full suite → `git diff --check` → commit → review độc lập. Sửa mọi Critical/Important rồi re-review.

---

## Task 7: Thay rank-only RRF bằng normalized weighted fusion ưu tiên BM25

### Files

- Create: `src/retrieval/types.py`
- Create: `src/retrieval/score_fusion.py`
- Create: `tests/test_score_fusion.py`
- Modify: `src/retrieval/bm25_retriever.py`
- Modify: `src/retrieval/hybrid_retriever.py`

### Interfaces phải tạo

```python
from collections.abc import Sequence, Set
from dataclasses import dataclass


@dataclass(frozen=True)
class ComponentCandidate:
    code: str
    score: float
    rank: int

@dataclass(frozen=True)
class RetrievedCandidate:
    code: str
    fusion_score: float
    bm25_score: float
    semantic_score: float
    bm25_rank: int | None
    semantic_rank: int | None

```

- `minmax_scores(candidates: Sequence[ComponentCandidate]) -> dict[str, float]`
- `fuse_candidates(bm25: Sequence[ComponentCandidate], semantic: Sequence[ComponentCandidate], alpha: float, valid_codes: Set[str] | None = None) -> list[RetrievedCandidate]`

Thêm:

- `BM25Retriever.retrieve_scored(query, top_k) -> list[ComponentCandidate]`
- `HybridRetriever.retrieve_scored(query, top_k=None) -> list[RetrievedCandidate]`
- Giữ `retrieve(query, top_k=5) -> list[str]` làm compatibility wrapper.

### Các bước

- [ ] **7.1 — Viết test RED cho contract và fusion**

Tạo tối thiểu các test sau trong `tests/test_score_fusion.py`:

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


def test_invalid_kb_codes_are_removed_before_normalization():
    bm25 = [ComponentCandidate("VALID", 2.0, 0), ComponentCandidate("MISSING", 100.0, 1)]
    fused = fuse_candidates(bm25, [], alpha=0.75, valid_codes={"VALID"})
    assert [item.code for item in fused] == ["VALID"]
    assert fused[0].bm25_score == pytest.approx(1.0)


def test_ties_are_deterministic():
    values = [ComponentCandidate("B", 1.0, 0), ComponentCandidate("A", 1.0, 1)]
    assert minmax_scores(values) == {"B": 1.0, "A": 1.0}
```

Thêm test cho: alpha là bool/NaN/ngoài khoảng; duplicate code; score non-finite; semantic-only/BM25-only; tie-break `fusion desc → bm25_rank → semantic_rank → code`; dataclass immutable.

- [ ] **7.2 — Chạy RED**

```powershell
python -m pytest tests/test_score_fusion.py -v
```

Kỳ vọng: fail do scored contracts/fusion chưa tồn tại.

- [ ] **7.3 — Implement pure score fusion**

Trong `score_fusion.py`:

1. Validate alpha là finite number, không nhận bool, trong `[0,1]`.
2. Chuẩn hóa/strip code; loại code ngoài `valid_codes` **trước** min-max.
3. Deduplicate từng component theo score tốt nhất, rồi rank tốt nhất, rồi code.
4. Min-max độc lập theo component; nếu mọi score bằng nhau, mọi candidate có normalized score `1.0`.
5. Candidate thiếu ở component nhận `0.0`.
6. Sort deterministic theo rule ở 7.1.

- [ ] **7.4 — Expose raw BM25 score**

`bm25s.BM25.retrieve()` trả documents và scores; chuyển đúng cặp `(code, raw_score)` thành `ComponentCandidate`. `retrieve()` chỉ unwrap code từ `retrieve_scored()`.

- [ ] **7.5 — Expose FAISS score và tích hợp**

- Dùng `PipelineConfig.retrieval.alpha/internal_top_k/hierarchical_expansion` thay cho `w_bm25`, `w_faiss`, `k` hardcode.
- FAISS IP/cosine là higher-is-better; deduplicate code trước fusion.
- Load tập code hợp lệ của đúng table (`icd10` hoặc `rxnorm`) một lần, read-only; không giữ code ngoài KB.
- Khi FAISS/model/index không sẵn sàng, trả BM25-only có score contract đúng; không tải online ngầm trong test/offline path.
- Xóa RRF hiện tại và oral-tablet/capsule bonus.
- Chỉ hierarchical-expand khi `hierarchical_expansion=True`; mặc định false.

- [ ] **7.6 — GREEN, smoke và full suite**

```powershell
python -m pytest tests/test_score_fusion.py -v
python -m pytest -q
python -c "from src.retrieval.hybrid_retriever import HybridRetriever; print(HybridRetriever('icd10').retrieve('tăng huyết áp', 5))"
git diff --check
```

Kỳ vọng: focused/full pass; smoke trả code hợp lệ, không exception khi semantic unavailable.

- [ ] **7.7 — Commit và review**

```powershell
git add src/retrieval/types.py src/retrieval/score_fusion.py src/retrieval/bm25_retriever.py src/retrieval/hybrid_retriever.py tests/test_score_fusion.py
git commit -m "feat: fuse normalized BM25 and semantic scores"
```

Review gate Task 7 phải xác nhận không còn RRF/bonus ngoài công thức, tổng weight bằng 1, BM25 mặc định nặng hơn semantic và compatibility wrapper không đổi kiểu trả về.

---

## Task 8: Candidate selection ưu tiên precision và clinical validation an toàn

### Files

- Create: `src/retrieval/candidate_selector.py`
- Create: `src/resources/clinical_validation_rules.json`
- Create: `tests/test_candidate_selector.py`
- Modify: `src/validation/clinical_validator.py`
- Modify: `src/ranking/llm_reranker.py`
- Modify: `src/pipeline/main.py`

### Interfaces phải tạo

- `CandidateSelector(config).select(entity_type, ranked, is_valid) -> list[str]`
- `ClinicalValidator.is_candidate_valid(entity, code, patient_info) -> bool`
- `dose_form_is_compatible(drug_text, rxcui_name, rules) -> bool`
- `LLMReranker.parse_selected_codes(payload, allowed_codes) -> list[str]`

### Các bước

- [ ] **8.1 — Viết test RED**

```python
import pytest

from src.retrieval.candidate_selector import CandidateSelector
from src.retrieval.types import RetrievedCandidate


def candidate(code: str, score: float, rank: int = 0) -> RetrievedCandidate:
    return RetrievedCandidate(code, score, score, 0.0, rank, None)


def test_selector_returns_top1_for_clear_margin():
    ranked = [candidate("A", 0.90, 0), candidate("B", 0.60, 1)]
    assert CandidateSelector().select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A"]


def test_selector_returns_top2_only_for_close_valid_scores():
    ranked = [candidate("A", 0.80, 0), candidate("B", 0.78, 1), candidate("C", 0.50, 2)]
    assert CandidateSelector().select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A", "B"]


def test_selector_rejects_low_confidence_or_invalid_codes():
    ranked = [candidate("A", 0.40, 0), candidate("B", 0.30, 1)]
    assert CandidateSelector().select("THUỐC", ranked, lambda _: True) == []
    assert CandidateSelector().select("THUỐC", ranked, lambda _: False) == []
```

Thêm regression cho: không quá 2 code; validate trước threshold/margin; stable order; unknown entity type; dual-code không sinh entity; historical RxNorm mặc định không load; injected dose-form rules; LLM trả foreign code bị reject.

- [ ] **8.2 — Chạy RED**

```powershell
python -m pytest tests/test_candidate_selector.py -v
```

- [ ] **8.3 — Implement selector**

Policy chính xác:

1. Lọc candidate không hợp lệ trước.
2. Dùng `icd_min_score` cho `CHẨN_ĐOÁN`, `rxnorm_min_score` cho `THUỐC`.
3. Không còn candidate trên minimum → `[]`.
4. Chỉ một candidate hợp lệ → Top-1.
5. Margin Top-1/Top-2 `>= top1_margin` → Top-1.
6. Cả hai trên minimum và margin `<= top2_margin` → Top-2.
7. Trường hợp còn lại → Top-1.
8. Không bao giờ trả quá 2 code.

- [ ] **8.4 — Refactor clinical validator thành predicate**

- Chuyển age/sex/dose-form rules vào `is_candidate_valid()`.
- `clinical_validation_rules.json` phải có version, source cho mỗi group, strict loader, immutable data và injectable path/object.
- `dose_form_is_compatible()` là pure function; contradiction chỉ được reject, không cộng điểm.
- `check_and_fix_candidates()` giữ compatibility nhưng gọi predicate/selector, không tự append candidate.
- `check_dual_codes()` không tạo entity mới; ban đầu trả entity list không đổi và chỉ log metadata phân tích.
- `load_historical_rxnorm=False` phải tạo **0 query/0 retained mapping**; nếu bật, historical codes chỉ mở rộng internal pool trước validation, không append thẳng output.

- [ ] **8.5 — Khóa LLM subset semantics**

Chỉ instantiate/call LLM khi `config.reranker.enabled=True`; truyền `timeout_seconds`. Parser chấp nhận:

```json
{"selected_codes": ["A"]}
```

hoặc legacy `{"best_code": "A"}`. Mọi code phải thuộc input pool; parse/timeout/foreign code phải fallback nguyên ranked pool để deterministic selector xử lý.

- [ ] **8.6 — Tích hợp pipeline**

Cho entity chẩn đoán/thuốc:

1. Chuẩn hóa query.
2. Lấy `retrieve_scored(expanded_text, top_k=self.config.retrieval.internal_top_k)`.
3. Lọc bằng `is_candidate_valid()`.
4. Nếu bật, LLM chỉ thu hẹp pool.
5. Chạy `CandidateSelector`.
6. Gán 0–2 code.

Xóa SQLite exact-match rerank loop trong `main.py`, fixed Top-5 và ingredient/historical append trực tiếp. Verified override path vẫn chỉ dùng loader đã audit; giữ thứ tự resource, revalidate code integrity và không output quá 2 code.

- [ ] **8.7 — GREEN và regression**

```powershell
python -m pytest tests/test_candidate_selector.py -v
python -m pytest tests/test_score_fusion.py tests/test_submission.py tests/test_override_validator.py -v
python -m pytest -q
git diff --check
```

- [ ] **8.8 — Commit và review**

```powershell
git add src/retrieval/candidate_selector.py src/resources/clinical_validation_rules.json src/validation/clinical_validator.py src/ranking/llm_reranker.py src/pipeline/main.py tests/test_candidate_selector.py
git commit -m "feat: select precision-first candidate subsets"
```

Review gate Task 8 phải probe false positive candidate, memory/query khi historical mapping tắt, LLM foreign-code, synthetic entity và mọi đường output `len(candidates) <= 2`.

---

## Task 9: Benchmark, calibrate trên trusted data và khóa config

### Files

- Create: `src/evaluation/benchmark.py`
- Create: `tests/test_pipeline_regressions.py`
- Modify: `src/evaluate.py`
- Modify: `src/pipeline/main.py`
- Modify: `README.md`

### Interfaces/CLI phải tạo

- `BaselinePipeline.process_text(text: str) -> list[dict]`
- `required_metric_paths() -> frozenset[str]`
- `python -m src.evaluation.benchmark --dev-pool --alphas 0.60 0.70 0.75 0.80 0.90`
- `python -m src.evaluation.benchmark --holdout --locked-config <path>`
- `python src/pipeline/main.py --input <dir> --output <dir> --config <json>`

### Các bước

- [ ] **9.1 — Viết pipeline/benchmark regressions RED**

`required_metric_paths()` phải đúng tập sau:

```python
frozenset({
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
```

Thêm test:

- Medication `amlodipine 10 mg po daily` giữ exact offset, `isHistorical`, candidate tối đa 2.
- Mọi entity ở `data/input/35.txt` là slice thật; không có synthetic prefix.
- `process_text()` không ghi file.
- Unknown config key fail trước inference.
- Benchmark dev chỉ đọc 101–180; pseudo 1–100 chỉ được report với label `untrusted`.
- Holdout từ chối khi chưa có locked config/hash hoặc còn tuning flags.
- Report thiếu bất kỳ required metric path nào phải fail.

- [ ] **9.2 — Chạy RED**

```powershell
python -m pytest tests/test_pipeline_regressions.py -v
```

- [ ] **9.3 — Hoàn thiện pipeline API/CLI**

- Tách logic hiện tại từ `process_file()` sang `process_text()`.
- `process_file()` chỉ đọc UTF-8 rồi delegate.
- Mọi component nhận cùng validated `PipelineConfig`; không dùng default riêng làm lệch alpha/threshold.
- CLI dùng `PipelineConfig.from_mapping()`; unknown/wrong type fail-closed.
- Giữ chunk-once và exact-offset invariants đã khóa ở Task 6.

- [ ] **9.4 — Implement benchmark không leakage**

Development mode:

- Chỉ ID 101–180.
- Five-fold deterministic; lưu fold IDs.
- Đánh giá alpha `0.60 0.70 0.75 0.80 0.90` và threshold được khai báo trong config grid.
- Báo mean/std, per-type, per-section, assertion labels, candidate metrics, Recall@20 và final score.
- Relaxed overlap chỉ nằm dưới `diagnostic`, selector không được đọc metric này.
- Tự tạo parent directory cho `--output`/`--write-locked-config` bằng thao tác an toàn; không phụ thuộc `reports/` có sẵn.

Holdout mode:

- Chỉ ID 181–200.
- Bắt buộc locked config JSON tồn tại và SHA-256 khớp metadata từ dev run.
- Cấm mọi tuning flag.
- Sau lần thành công, ghi marker `reports/holdout-run-<hash>.json`; cùng hash không được chạy lần hai.

- [ ] **9.5 — Chạy baseline development**

```powershell
python -m pytest -q
python -m src.evaluation.benchmark --dev-pool --baseline --output reports/baseline-dev.json
```

Kỳ vọng: report chỉ chứa ID 101–180 và đủ required metric paths.

- [ ] **9.6 — Cross-validation và khóa cấu hình**

```powershell
python -m src.evaluation.benchmark --dev-pool --alphas 0.60 0.70 0.75 0.80 0.90 --output reports/precision-first-cv.json --write-locked-config reports/locked-config.json
```

Rule chọn config:

1. Mean final score không thấp hơn baseline rerun.
2. Stage metric liên quan phải cải thiện: entity exact F0.5, assertion macro F0.5 hoặc candidate Jaccard.
3. Chọn mean final score cao nhất.
4. Nếu chênh lệch `<= 0.005`, chọn exact entity precision cao hơn.
5. Không có cấu hình eligible → khóa baseline, không ép dùng rewrite.

Locked config phải ghi alpha, NER/assertion/selector thresholds, baseline comparison, fold IDs và SHA-256.

- [ ] **9.7 — Chạy holdout đúng một lần**

```powershell
python -m src.evaluation.benchmark --holdout --locked-config reports/locked-config.json --output reports/final-holdout.json
```

Sau lệnh này không sửa rule/threshold dựa trên holdout. Nếu holdout kém, ghi nhận limitation và bắt đầu development cycle mới với lock hash mới; không chạy lại cùng hash.

- [ ] **9.8 — Documentation và commit**

README chính phải có setup, trusted-data policy, benchmark dev, locked holdout, public inference, validation/package và giới hạn neural NER chưa triển khai.

```powershell
git add src/evaluation/benchmark.py src/evaluate.py src/pipeline/main.py tests/test_pipeline_regressions.py README.md
git commit -m "feat: calibrate and verify precision-first pipeline"
```

Review gate Task 9 phải audit trực tiếp file IDs đã đọc và chứng minh pseudo-GT/holdout không tham gia model selection.

---

## Task 10: Whole-branch review, public inference và artefact bàn giao

### Artefacts đầu ra

- `reports/baseline-dev.json`
- `reports/precision-first-cv.json`
- `reports/locked-config.json`
- `reports/final-holdout.json`
- `data/output/1.json` … `data/output/100.json`
- `output.zip` chứa `output/1.json` … `output/100.json`

### Các bước

- [ ] **10.1 — Review độc lập toàn nhánh**

Reviewer đọc spec, README này và diff:

```powershell
git diff --check
git diff --stat 80ed529..HEAD
git log --oneline 80ed529..HEAD
```

Audit bắt buộc:

- Precision metric/F0.5 và trusted split đúng.
- Không hardcode clinical vocabulary ngoài strict resources.
- Không legacy override trong inference.
- Exact offsets và deterministic ordering.
- Fusion chỉ có hai trọng số tổng bằng 1; BM25 mặc định nặng hơn.
- Candidate tối đa 2; clinical validation không sinh entity/candidate mới.
- Reranker disabled path không gọi model/network.
- Holdout không có tuning/read-before-lock.

Mọi Critical/Important phải sửa bằng test regression và re-review; không bỏ qua để chạy submission.

- [ ] **10.2 — Chạy verification đầy đủ**

```powershell
python -m pytest -v
python src/metrics.py test
python scripts/audit_overrides.py --db data/kb/metadata.db --overrides src/resources/verified_overrides.json
python scripts/package_submission.py --input data/input --output data/output --zip output.zip --db data/kb/metadata.db --validate-only
git diff --check
```

Kỳ vọng: tests/metrics/audit/validate đều exit 0.

- [ ] **10.3 — Chạy public inference bằng config đã khóa**

```powershell
python src/pipeline/main.py --input data/input --output data/output --config reports/locked-config.json
```

Yêu cầu: đúng 100 JSON; `errors.jsonl` không có lỗi; mỗi entity exact-slice; assertions/candidates đúng schema; mọi candidate tồn tại trong KB.

- [ ] **10.4 — Validate rồi tạo zip duy nhất**

```powershell
python scripts/package_submission.py --input data/input --output data/output --zip output.zip --db data/kb/metadata.db
python -c "import zipfile; z=zipfile.ZipFile('output.zip'); names=z.namelist(); assert names == [f'output/{i}.json' for i in range(1,101)]; print(len(names))"
```

Kỳ vọng: in `100`; zip không có file thừa và không chứa artefact lỗi.

- [ ] **10.5 — Final state và bàn giao**

```powershell
git status --short
git log -4 --oneline
```

Ghi trong báo cáo bàn giao:

- branch/HEAD;
- số test pass;
- selected config + SHA-256;
- dev mean/std và holdout final score;
- entity precision/recall/F0.5, assertion macro F0.5, candidate Jaccard/precision, Recall@20;
- đường dẫn `output.zip`;
- limitation NFD và neural NER chưa triển khai;
- xác nhận holdout chỉ chạy một lần cho lock hash.

Không commit `output.zip`, public predictions hoặc report chứa dữ liệu nếu `.gitignore`/chính sách project không cho phép. Chỉ commit production code, tests và documentation có chủ đích.

## 3. Definition of Done Task 7–10

- [ ] Full suite pass sau từng task và ở final HEAD.
- [ ] Task 7/8/9 có commit riêng và review độc lập sạch Critical/Important.
- [ ] Fusion weights luôn tổng bằng 1; default `0.75 BM25 + 0.25 semantic`.
- [ ] Không fixed Top-5; output candidate tối đa 2 hoặc `[]`.
- [ ] Không entity synthetic; mọi offset exact.
- [ ] Pseudo-GT 1–100 không tham gia tuning.
- [ ] Dev 101–180 hoàn tất trước khi lock; holdout 181–200 chạy đúng một lần.
- [ ] 100 public outputs validate sạch và `output.zip` có đúng layout.
- [ ] README chính chứa lệnh tái lập và limitation.
- [ ] Whole-branch reviewer kết luận clean.

## 4. Tài liệu tham chiếu

- `docs/superpowers/specs/2026-07-18-precision-first-hybrid-pipeline-design.md`
- `docs/superpowers/plans/2026-07-18-precision-first-hybrid-pipeline.md`
- `.superpowers/sdd/global-constraints.md`
- `.superpowers/sdd/progress.md`
- `.superpowers/sdd/task-7-brief.md`
- `.superpowers/sdd/task-8-brief.md`
- `.superpowers/sdd/task-9-brief.md`
