# Kế hoạch thực thi contract-first, resource-safe trên Kaggle

**Design nguồn:** `docs/superpowers/specs/2026-07-23-contract-first-resource-safe-kaggle-design.md`

**Phương pháp:** khóa contract → implement theo ownership → regression checks
sau implementation → review độc lập → Kaggle acceptance. Không phụ thuộc chu
trình TDD hoặc GPU Kaggle trong lúc viết code.

## Nguyên tắc điều phối

- Một worker chỉ sở hữu một nhóm file; không hoàn nguyên thay đổi của worker khác.
- Mỗi nhóm có report gồm diff, commands, kết quả regression và concern.
- Review kiểm tra spec compliance và code quality; finding quan trọng phải sửa
  trước khi qua nhóm kế tiếp.
- Dataset, model weights, `scratch/` và Kaggle downloads không được commit.
- Không tuyên bố Kaggle success khi chưa có một `Run All` thật.

## Work package 1 — Runtime control-plane

**Ownership:** `runtime_control.py`, regression tests tương ứng.

Deliverables:

- structured JSONL logger với start/end/error và memory snapshot;
- positive-schema context/error logging, redaction an toàn và state machine khóa
  terminal theo `(phase, attempt)` cùng aggregate terminal;
- atomic JSON writer;
- resource profile 16 GB, host/disk admission và đúng một OOM retry 2→1;
- Qwen 3B default-off: T4 `1024`, batch ladder `8→4→1`; P100/unsupported
  capability hoặc kernel probe bị skip trước load;
- strict unique source resolver kèm decision audit accept/reject;
- model inventory nhất quán, có source/hash và fail-closed budget ≤9B.

Evidence: module import không cần torch, focused regression pass, full current
suite không regression.

## Work package 2 — Data/KB preflight fail-closed

**Ownership:** `preflight.py`, CLI `preflight_pipeline.py`, regression fixtures.

Deliverables:

- exact input/GT/manifest pairing;
- aggregate ordered input+GT fingerprint, manifest/config/KB hashes;
- explicit normalization `reconstructed→quarantine`,
  `organizer_gt→organizer`, `synthetic→synthetic`;
- missing manifest/entry/eligibility là error, không default eligible;
- schema, offset, assertion/candidate shape;
- organizer non-empty candidate runtime coverage report;
- current/stale report inventory;
- machine-readable `preflight_report.json` và nonzero exit khi hard gate fail.

Evidence trên dataset thật hiện tại phải chỉ đúng đồng thời các blocker đã audit:
legacy manifest thiếu pair-provenance contract, `candidate_top_k=10` và 258
organizer RxNorm ID còn thiếu; không che thành warning hoặc chỉ dừng ở lỗi đầu.

## Work package 3 — Provenance/config và runtime KB coverage repair

**Ownership:** KB builder/contract, generated small runtime artifacts và tests.

Deliverables:

- trích đúng organizer RXCUI còn thiếu từ raw RxNorm `data/kb`;
- không phát minh ID; lưu source row/evidence;
- canonical/display ICD mapping cho marker `*`/`†`;
- runtime artifacts có chain raw hash → build metadata → artifact hash;
- preflight organizer coverage đạt 100% hoặc dừng với unresolved evidence list.
- nâng manifest bằng tool deterministic: schema version, per-input/per-GT/pair
  hash, full manifest hash và dataset pair fingerprint; không sửa input/GT;
- giữ legacy normalized-text `sha256` dưới tên rõ nghĩa; raw pair hash dùng
  domain-separated length framing, canonical JSONL và detached descriptor để
  không có self-hash;
- đổi `candidate_top_k` về 20, giữ `candidate_output_k=1` và regex primary off;
- mọi report cũ khác fingerprint được đánh dấu stale/archived.

Không tự sửa GT organizer. Artifact chỉ mở rộng từ raw KB có bằng chứng.

## Work package 4 — Development split, chunking và curriculum

**Ownership:** split/record/training/curriculum modules và CLI train.

Deliverables:

- 10 blind challenge, 72 organizer train, 18 organizer validation;
- synthetic 1.600/400 development split;
- hard groups không union chỉ vì cùng clinical surface;
- synthetic near-duplicate exclusions theo fold;
- owner-window: mỗi gold entity đóng góp loss đúng một lần;
- document entity metrics thay token-F1 selection;
- Stage 1/2/3/final adaptation tuần tự, stage manifest atomic;
- Stage-2 checkpoint là blind baseline, Stage-3 checkpoint là candidate; proxy
  metric và decoder hash được khóa trước one-shot blind gate;
- `FAST_DEV_RUN` và `oof_extended` không thể mở 10 blind documents;
- sau Stage 3 shared encoder/tokenizer được freeze; final-fit chỉ cập nhật NER
  token-classification head để assertion artifact không lệch encoder hash;
- stage training chạy subprocess, source-aware sampling và replay 15–20%;
- `fixed_fold` mặc định, `oof_extended` chạy fold độc lập.

Evidence: split/owner invariants, fake trainer stage order/resume, CPU feature
validation và full regression suite.

## Work package 5 — Assertion và inference hardening

**Ownership:** assertion head/adapter, NER inference micro-batch, Qwen guard.

Deliverables:

- assertion ba sigmoid head dùng lại feature `CLS + mention pooling` của frozen
  NER encoder, không resize tokenizer; per-axis thresholds và encoder/tokenizer
  hash được đóng dấu;
- chỉ disease/drug/symptom vào assertion;
- NER overflow chunks infer theo micro-batch và backoff;
- deterministic NER + KB + candidate policy luôn là primary path;
- Qwen mặc định off, load sau khi release NER, P100 skip, T4 conservative
  profile, optional failure không làm mất submission;
- retrieval top-20, output tối đa một candidate, abstention giữ nguyên.

Evidence: model inventory dưới 9B, fallback/error simulation và schema/offset
validation.

## Work package 6 — Kaggle orchestrator và notebook 13 phase

**Ownership:** Kaggle orchestrator, notebook builder, canonical notebook.

Deliverables:

- `RUN_MODE=full|resume|inference_only`;
- strict overrides/auto-discovery, in đầy đủ candidates accept/reject;
- resolver chỉ dùng standard library chạy trước dependency install/preflight;
- dependency preflight chạy trên wheelhouse/model source đã resolve, rồi mới tới
  hardware/model-budget và torch/model load;
- run directory có `run_id`; toàn bộ artifact được fsync rồi atomic rename một
  immutable directory, sau đó atomic cập nhật `LATEST.json` commit pointer;
- subprocess training/OOM retry/resume mismatch;
- fresh-process reload trước package;
- model/output ZIP inventory, SHA-256 và CRC;
- mọi phase có start + terminal event + duration/memory;
- cell chỉ điều phối module, không copy business logic lớn.

Evidence: AST compile mọi code cell, generator/notebook deterministic, simulated
three-mode execution và regression suite.

## Work package 7 — Runbook và tài liệu vận hành

**Ownership:** `KAGGLE_RUNBOOK.md`, `README.md`, `PIPELINE_VI.md`.

Runbook bắt buộc có:

- cấu trúc từng Kaggle Dataset: input, train v2, model, optional wheelhouse;
- online/offline bootstrap;
- giá trị override và strict ambiguity behavior;
- ba run mode, fixed-fold vs extended OOF;
- model budget inventory;
- bảng lỗi `E_*`, nhất là OOM, path ambiguity, stale resume, missing KB;
- cách đọc JSONL event và xác định cell/stage/attempt lỗi;
- cách resume an toàn và khi nào phải chạy lại từ đầu;
- artifact cần tải và lệnh kiểm CRC/inventory;
- checklist trước `Save Version → Run All`.

`PIPELINE_VI.md` phải có Mermaid data flow, giải thích kỹ thuật từng phase và
ELI5 bằng ví dụ bệnh viện.

## Work package 8 — Verification và independent review

Chạy theo thứ tự:

1. compile/import/AST;
2. full CPU regression suite;
3. real dataset preflight;
4. fake curriculum + OOM/resume simulations;
5. notebook build determinism;
6. local checkpoint reload/inference smoke nếu local artifact hợp lệ;
7. hai agent độc lập review diff và runbook;
8. một Kaggle `Run All` thật khi môi trường khả dụng.

Trạng thái trước bước 8 chỉ được ghi “local contract verified; Kaggle acceptance
pending”. Goal chỉ complete khi artifact thật chứng minh `Run All` thành công,
hoặc người dùng thay đổi rõ tiêu chí nghiệm thu.
