# Runbook Kaggle — Clinical NLP contract-first

Notebook canonical: `medical_information_extraction_kaggle.ipynb`.

Tài liệu giải thích cho người mới: `KAGGLE_PIPELINE_ELI5_VI.md`.

Notebook chỉ là client gọi API của `clinical_nlp_lab.orchestration`; không chứa
bản sao business logic train hoặc inference. Local chỉ chạy contract test trên
CPU. Phiên `Save Version → Run All` thật được thực hiện trên Kaggle.

## Code runtime bằng Git clone

Notebook mặc định clone repository GitHub thay vì đóng gói toàn bộ source vào
Kaggle Dataset:

```text
GIT_CLONE_URL=https://github.com/takumi612/AI-Race-Viettel.git
GIT_CLONE_REF=codex/kaggle-end-to-end-pipeline
GIT_CLONE_DIR=/kaggle/working/AI-Race-Viettel
USE_GIT_CLONE=1
```

Vì vậy chỉ cần bật Internet. Nếu muốn dùng code Dataset đã attach, đặt
`USE_GIT_CLONE=0` và `PROJECT_ROOT_OVERRIDE=/kaggle/input/<code-dataset>`.

## Dataset và input runtime

Attach Dataset có cấu trúc sau, hoặc đặt `PROJECT_ROOT_OVERRIDE` tới code Dataset
đã mount:

```text
ai-race-clinical-data/
├── input/<document_id>.txt
└── synthetic_train_v2/
    ├── input/<document_id>.txt
    ├── gt/<document_id>.json
    └── reports/dataset_manifest.jsonl
```

Trước khi chạy, bật GPU Accelerator và Internet để notebook clone source/model
và cài dependency nếu cần. Các biến môi trường được hỗ trợ:

```text
RUN_MODE=full|resume|inference_only
RUN_ID=<stable run id dùng cho resume>
DATASET_ROOT=/kaggle/input/.../synthetic_train_v2
OUTPUT_DIR=/kaggle/working/run_output
PROJECT_ROOT_OVERRIDE=<chỉ cần khi không dùng Git clone>
GIT_CLONE_URL=https://github.com/takumi612/AI-Race-Viettel.git
GIT_CLONE_REF=codex/kaggle-end-to-end-pipeline
GIT_CLONE_DIR=/kaggle/working/AI-Race-Viettel
USE_GIT_CLONE=1
FAST_DEV_RUN=0
INPUT_SOURCE=/kaggle/input/.../input
ARTIFACT_DIR=/kaggle/working/artifacts
ARTIFACT_SOURCE_DIR=/kaggle/input/.../artifacts
MODEL_SOURCE=xlm-roberta-base
EXPECTED_GPU_COUNT=2
USE_DISTRIBUTED=1
INSTALL_RUNTIME_DEPS=1
```

## Ba chế độ chạy

| Chế độ | Hành vi |
|---|---|
| `full` | Chạy đủ 13 phase canonical: training hooks, inference, validation và packaging. |
| `resume` | Đọc `LATEST.json`, kiểm tra config fingerprint rồi chạy tiếp sau phase terminal cuối. |
| `inference_only` | Chạy preflight/source/model checks, inference, validation và packaging; bỏ qua toàn bộ training. |

Notebook tự bind `build_kaggle_phase_runners(config)` vào `RunConfig`. Orchestrator
vẫn fail-closed nếu runner bị thiếu hoặc một runner không publish được artifact.
Hook thiếu sẽ tạo artifact `*.error.json` theo cách atomic và không cập nhật
`LATEST.json`.

Mỗi phase ghi một event `PHASE_START` và đúng một event terminal vào `run.jsonl`.
Artifact phase được ghi vào file tạm, flush/fsync rồi atomic rename. `LATEST.json`
chỉ được publish sau khi artifact hoàn tất, nên phase bị ngắt không được xem là
đã đủ điều kiện resume.

Notebook hiển thị một code cell cho từng phase. Các cell gọi
`run_phase(SESSION, PHASE_NAME)` và in JSON result ngay sau khi phase kết thúc.
Nếu lỗi, cell hiện tại dừng với traceback; `run.jsonl` và artifact
`*.error.json` cho biết chính xác phase/cell bị lỗi.

Trong `full`, phase 07–10 gọi `train_ner_subprocess.py`. Khi
`USE_DISTRIBUTED=1` và có đủ 2 GPU, runner dùng `torch.distributed.run` với
`--nproc_per_node=2`, phù hợp T4×2. Phase 11 reload final checkpoint để fit
assertion head/candidate calibration; phase 12 reload bundle và tạo `output.zip`.

## Artifact kỳ vọng

```text
/kaggle/working/run_output/
├── LATEST.json
└── <run_id>/
    ├── run.jsonl
    ├── run_manifest.json
    ├── artifacts/phase_*.json
    ├── checkpoints/<stage>/ner_model/
    ├── output.zip
    └── trained_artifacts.zip
```

Runner thật trên Kaggle sẽ publish `output.zip`, `trained_artifacts.zip`, model
inventory, checksum và diagnostics. Không được tuyên bố `PASS` cho đến khi một
phiên Kaggle `Run All` thật sự sinh ra các artifact này.

## Khi Kaggle lỗi: gói thông tin bàn giao

Gửi lại các file sau:

1. `run_manifest.json` và `LATEST.json` nếu có;
2. `run.jsonl` trong thư mục run bị lỗi;
3. `resource_plan.json`, tên phase/cell và toàn bộ traceback;
4. `output.zip`/`trained_artifacts.zip` dở dang cùng inventory nếu có.

Session tiếp theo sẽ kiểm tra phase lỗi trước, đối chiếu fingerprint rồi hướng
dẫn `resume` hoặc chạy `full` mới. Không dùng local training thay cho chẩn đoán
Kaggle.

## Cổng kiểm tra CPU trước Kaggle

Các lệnh này không load model weights và không train:

```powershell
python -m pytest tests/test_training_contract.py tests/test_primary_inference_path.py tests/test_orchestration_contract.py -q -p no:cacheprovider
python tools/build_kaggle_notebook.py --output C:\path\to\audit.ipynb
```

Sau khi cổng CPU đạt, upload notebook lên Kaggle, chọn `Save Version → Run All`,
rồi gửi các artifact ở trên nếu cần phân tích lỗi tiếp.
