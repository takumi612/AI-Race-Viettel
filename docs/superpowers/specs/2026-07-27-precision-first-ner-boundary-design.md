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

## 10. Chiến lược kiểm thử

Phần triển khai tuân theo Test-Driven Development (TDD). Với mỗi hành vi cần sửa,
quy trình là: viết một test mô tả kết quả đúng, chạy để xác nhận test đang thất bại
vì code cũ chưa có hành vi đó, viết lượng code nhỏ nhất để test vượt qua, sau đó chạy
lại test liên quan và toàn bộ test suite để kiểm tra hồi quy.

### 10.1 Unit test

Unit test gọi trực tiếp logic thật với dữ liệu nhỏ, cố định và không cần tải model.
Qwen được thay bằng fake engine trả JSON xác định trước; mục tiêu là kiểm tra parser,
validation contract và luồng xử lý của hệ thống, không kiểm tra chất lượng của model
ngôn ngữ trong môi trường unit test.

Các trường hợp cụ thể gồm:

| Trường hợp | Dữ liệu kiểm tra | Kết quả bắt buộc |
|---|---|---|
| Hai window dự đoán giống nhau | Hai proposal cùng type và cùng `[start, end]` | Chỉ còn một entity, giữ đúng boundary gốc và ghi nhận hai nguồn bằng chứng |
| Hai window gần giống nhau | Ví dụ `[10, 20]` và `[10, 21]` | Chọn một boundary đã quan sát; không tạo boundary mới |
| Hai window xung đột | Ví dụ `[10, 40]` và `[30, 60]` | Tuyệt đối không tạo union `[10, 60]`; chọn proposal thắng theo policy |
| Thứ tự độ mạnh bằng chứng | Exact-KB span cạnh tranh với Transformer span | Exact-KB thắng; nếu cùng nguồn thì xét đồng thuận, confidence rồi boundary ngắn hơn |
| Entity KB hợp lệ nhưng rất dài | Exact-KB entity dài hơn 100 ký tự | Entity vẫn được giữ nguyên, chứng minh hệ thống không dùng hard cap độ dài |
| Qwen `keep` | Fake engine trả action `keep` | Entity giữ nguyên text, type, position và offset vẫn hợp lệ |
| Qwen `drop` | Fake engine trả action `drop` | Entity bị loại khỏi kết quả trước bước linking/assertion |
| Qwen `trim` hợp lệ | Span gốc `[100, 160]`, Qwen trả relative range nằm bên trong | Tạo entity mới bằng đúng substring của raw text và offset tuyệt đối được dịch chính xác |
| Phản hồi Qwen không hợp lệ | JSON lỗi, action/type lạ, relative offset ngoài span hoặc thiếu/thừa response | Ném `RequiredQwenError`; Phase 12 không được đóng gói output |
| Metadata sau khi trim/retype | Qwen thay text, type hoặc position | Candidate retrieval và assertion phải chạy lại trên entity mới; metadata của entity cũ không được giữ lại |
| Tắt Qwen | `ENABLE_QWEN_RERANKER=False` | Không gọi fake engine; boundary-consensus merge vẫn chạy và cho kết quả xác định |
| Bộ đếm Phase 12 | Một batch có đủ `keep`, `drop`, `trim` và KB bypass | Summary/diagnostics báo đúng số query, quyết định, type và bucket độ dài trước–sau |

### 10.2 Component và contract test chạy local

Không xây dựng một integration test giả lập toàn bộ Kaggle Phase 12 trong local/CI.
Phase này cần checkpoint Transformer thật, assertion head, KB artifacts, CUDA và
vLLM Qwen; thay toàn bộ các dependency đó bằng mock sẽ chỉ kiểm tra orchestration
chứ không chứng minh pipeline runtime thật hoạt động.

Thay vào đó, test local được chia theo các ranh giới khả thi:

1. Component test của `infer_document()` dùng fake NER, fake Qwen validator, fake KB
   linker và fake assertion predictor nhưng chạy logic orchestration thật. Test đưa
   vào các proposal mô phỏng nhiều window, rồi kiểm tra thứ tự merge → validate →
   relink → assertion và raw offset cuối cùng.
2. Contract test của Phase 12 patch các dependency nặng, tương tự test hiện có trong
   `test_required_qwen_phase.py`. Test chỉ xác nhận toggle, bộ đếm, propagation lỗi,
   cleanup và nguyên tắc không tạo ZIP khi validation thất bại.
3. Packaging contract test tạo một thư mục output tạm với vài JSON đã hợp lệ, chạy
   riêng logic đóng gói và kiểm tra member `output/<id>.json`, thứ tự, CRC và việc
   không có file thừa. Test này không tuyên bố đã kiểm tra model inference.

### 10.3 End-to-end smoke test trên Kaggle

Kaggle Run All với GPU, Internet, dữ liệu attach và checkpoint/model thật là test
end-to-end duy nhất. Bước này được thực hiện sau khi toàn bộ test local vượt qua và
phải kiểm tra:

- cả 13 phase hoàn tất;
- Qwen entity validation có trạng thái `COMPLETED`;
- các bộ đếm `keep`, `drop`, `trim` và KB bypass xuất hiện trong Phase 12;
- đủ 100 JSON, ZIP đúng cấu trúc và CRC;
- không có offset error;
- báo cáo độ dài span trước–sau và validation metrics được sinh ra.

Smoke test Kaggle không chạy tự động trong local/CI vì phụ thuộc GPU T4, CUDA/vLLM,
model tải từ Hugging Face và thời gian huấn luyện dài. Ngoài các test mới, toàn bộ
test hiện có về offset, ZIP, Qwen bắt buộc, candidate, assertion và notebook contract
phải tiếp tục vượt qua.

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
