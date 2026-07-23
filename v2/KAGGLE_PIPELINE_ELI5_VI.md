# Giải thích Pipeline Kaggle — ELI5

Tài liệu này giải thích pipeline bằng ngôn ngữ đơn giản nhất có thể. Giả sử
người đọc chưa biết Python, machine learning, GPU hay Kaggle.

Notebook chính là:

`medical_information_extraction_kaggle.ipynb`

Code xử lý thật nằm trong thư mục `clinical_nlp_lab/`, còn notebook chỉ gọi
những hàm điều phối. Đây là chủ ý thiết kế: notebook dễ theo dõi, còn code có
thể kiểm thử và chạy lại.

---

## 1. Tóm tắt trong một phút

Hệ thống nhận vào:

- các file văn bản bệnh án `.txt`;
- nhãn đúng tương ứng trong các file `.json`;
- từ điển mã bệnh ICD-10 và mã thuốc RxNorm;
- một model ngôn ngữ ban đầu, mặc định là `xlm-roberta-base`.

Ở mức rất cao, có **7 nhóm việc lớn**:

1. kiểm tra dữ liệu có hợp lệ không;
2. chia dữ liệu thành các phần nhỏ để model đọc được;
3. huấn luyện model nhận diện bệnh, thuốc, triệu chứng và xét nghiệm;
4. học thêm các trạng thái như “không mắc”, “tiền sử”, “gia đình”;
5. nối bệnh/thuốc với mã chuẩn trong knowledge base;
6. chạy model trên input thật;
7. kiểm tra kết quả và đóng thành `output.zip`.

Đây là bản tóm tắt để dễ hình dung, không phải danh sách phase kỹ thuật.
Một nhóm lớn có thể chứa nhiều phase; ví dụ “huấn luyện NER” chứa bốn stage.

### Sơ đồ đầy đủ 13 trạm

Hãy hình dung mỗi phase kỹ thuật là một trạm riêng trong dây chuyền bệnh viện:

```text
01 Preflight dữ liệu
        ↓
02 Resolve đường dẫn input/model/KB
        ↓
03 Kiểm kê GPU và model
        ↓
04 Tạo record metadata
        ↓
05 Tạo fixed/OOF splits
        ↓
06 Tạo owner-window training contract
        ↓
07 Train curriculum stage 1
        ↓
08 Train curriculum stage 2
        ↓
09 Train curriculum stage 3
        ↓
10 Final-fit encoder
        ↓
11 Train assertion head + candidate calibration
        ↓
12 Inference + KB-first recovery
        ↓
13 Validate và đóng gói ZIP
```

### 7 nhóm lớn ánh xạ vào 13 phase

| Nhóm tóm tắt | Các phase kỹ thuật | Giải thích |
|---|---|---|
| 1. Kiểm tra dữ liệu | 01, 02, 03 | Kiểm tra dữ liệu, đường dẫn, GPU và model |
| 2. Chia dữ liệu | 04, 05, 06 | Metadata, split và window/tensor contract |
| 3. Train NER | 07, 08, 09, 10 | Bốn lần train: stage 1, 2, 3 và final fit |
| 4. Học trạng thái | 11 | Assertion head học phủ định/tiền sử/gia đình |
| 5. Nối mã chuẩn | 11, 12 | Phase 11 học calibration; phase 12 tra KB khi inference |
| 6. Chạy input thật | 12 | Reload model và tạo prediction |
| 7. Kiểm tra/đóng gói | 13 | Kiểm tra ZIP, CRC và inventory |

Vì vậy không phải 7 phase bị thiếu. Có **7 cách nói ở mức nghiệp vụ** và
**13 phase ở mức implementation**. Phase 11 xuất hiện ở hai nhóm vì nó vừa
chuẩn bị candidate calibration, vừa hoàn thiện phần nối mã; phase 12 vừa chạy
model vừa dùng KB để recovery.

---

## 2. Ba khái niệm nền tảng

### 2.1 Notebook là gì?

Notebook là một tài liệu có các ô gọi là **cell**. Có hai loại cell:

- **Markdown cell**: chỉ để viết tiêu đề và giải thích.
- **Code cell**: cell Python được chạy từ trên xuống dưới.

Notebook này có:

| Loại | Số lượng | Vai trò |
|---|---:|---|
| Markdown | 14 | Mô tả setup và 13 phase |
| Code | 15 | 1 setup + 13 phase + 1 finalization |
| Tổng | 29 | Một notebook hoàn chỉnh |

Mỗi phase có một code cell riêng. Vì vậy khi Kaggle dừng ở phase 08, ta biết
ngay đang lỗi ở trạm nào; không phải đoán trong một cell dài hàng nghìn dòng.

### 2.2 Artifact là gì?

Artifact là “bằng chứng trên giấy” được chương trình lưu ra đĩa:

- phase đã chạy chưa;
- fingerprint/hash của dữ liệu;
- checkpoint model;
- log lỗi;
- file kết quả.

Nếu không có artifact, một dòng chữ `PASS` trên màn hình chưa đủ để tin rằng
phase đã hoàn thành.

### 2.3 Fingerprint/hash là gì?

Hash giống như dấu vân tay của một file hoặc một bộ dữ liệu. Nếu dữ liệu thay
đổi một byte, hash thường cũng thay đổi.

Pipeline dùng hash để trả lời:

> “Checkpoint này có được tạo từ đúng dataset/config hiện tại không?”

Nếu câu trả lời là không, resume sẽ bị chặn thay vì tiếp tục một cách nguy
hiểm.

---

## 3. Cấu trúc thư mục quan trọng

```text
v2/
├── medical_information_extraction_kaggle.ipynb  ← notebook người dùng chạy
├── clinical_nlp_lab/                            ← business logic thật
│   ├── orchestration.py                         ← điều phối session/phase
│   ├── kaggle_phases.py                         ← implementation 13 phase
│   ├── runtime_bundle.py                        ← reload model để inference
│   ├── examples.py                              ← owner-window
│   ├── collation.py                             ← window → tensor
│   ├── curriculum.py                            ← stage và resume hash
│   ├── training.py                              ← training contract
│   ├── assertion_model.py                       ← assertion head
│   ├── candidate_training.py                    ← candidate/calibration
│   ├── inference.py                             ← merge proposal/offset
│   └── pipeline.py                              ← đường inference chính
├── scripts/
│   └── train_ner_subprocess.py                  ← Trainer thật
├── tools/
│   └── build_kaggle_notebook.py                 ← sinh notebook canonical
├── artifacts/                                   ← config + KB + mapping
├── KAGGLE_RUNBOOK.md                            ← hướng dẫn vận hành
└── PIPELINE_VI.md                               ← sơ đồ pipeline tiếng Việt
```

Trên Kaggle, notebook mặc định clone branch:

```text
https://github.com/takumi612/AI-Race-Viettel.git
codex/kaggle-end-to-end-pipeline
```

Sau khi clone, code nằm ở:

```text
/kaggle/working/AI-Race-Viettel/v2/
```

Dataset train vẫn phải được attach riêng vì đó là dữ liệu lớn; Git clone thay
thế việc upload **code Dataset**, không tự tạo ra dữ liệu train.

---

## 4. Notebook làm gì từ đầu đến cuối?

### Cell 00 — Markdown giới thiệu

Cell này không chạy code. Nó nói ba việc:

1. notebook chỉ là client gọi API;
2. business logic nằm trong `clinical_nlp_lab`;
3. Kaggle cần Internet, GPU và lệnh `Save Version → Run All`.

### Cell 01 — Setup runtime

Đây là cell code quan trọng nhất trước phase 01.

#### Bước 1: nhận diện môi trường

```python
IS_KAGGLE = Path("/kaggle/input").is_dir()
```

Nếu thư mục `/kaggle/input` tồn tại, code hiểu rằng nó đang chạy trên Kaggle.
Khi chạy local, thư mục này thường không tồn tại.

#### Bước 2: clone source code

Các biến mặc định:

```python
GIT_CLONE_URL = "https://github.com/takumi612/AI-Race-Viettel.git"
GIT_CLONE_REF = "codex/kaggle-end-to-end-pipeline"
GIT_CLONE_DIR = "/kaggle/working/AI-Race-Viettel"
USE_GIT_CLONE = True trên Kaggle
```

Code tương đương ý nghĩa:

```text
nếu đang ở Kaggle và chưa chỉ định PROJECT_ROOT_OVERRIDE:
    nếu repo chưa tồn tại:
        git clone branch đã chỉ định
    PROJECT_ROOT = repo/v2
```

Nếu thư mục clone đã có đúng `v2/clinical_nlp_lab`, code không clone lại.

Nếu muốn dùng code Dataset thay vì Git:

```text
USE_GIT_CLONE=0
PROJECT_ROOT_OVERRIDE=/kaggle/input/<code-dataset>/v2
```

#### Bước 3: cài dependency nếu thiếu

Code tìm `requirements-kaggle.txt`. Nếu thiếu các package chính như
`transformers`, `accelerate`, `sentencepiece` hoặc `safetensors`, nó chạy pip.

Đây là lý do Internet phải bật nếu các package chưa có sẵn trong Kaggle image.

#### Bước 4: tìm dataset và input

Notebook đọc các biến:

```text
DATASET_ROOT       ← thư mục synthetic_train_v2
INPUT_SOURCE       ← thư mục/file input để inference
ARTIFACT_SOURCE_DIR← config + KB gốc trong repo clone
ARTIFACT_DIR       ← bản writable ở /kaggle/working/artifacts
```

Nếu không đặt biến, notebook tự tìm thư mục `synthetic_train_v2` dưới
`/kaggle/input`.

#### Bước 5: tính fingerprint

Notebook tính hash của `config.json` và đọc fingerprint dataset từ
`reports/dataset_provenance.json` nếu có.

Điều này giúp resume biết config/dataset đang dùng có giống lúc checkpoint
được tạo hay không.

#### Bước 6: tạo `RunConfig`

`RunConfig` là một tờ “phiếu yêu cầu chạy”:

```text
run_mode          = full / resume / inference_only
dataset_root      = dữ liệu train
output_dir        = nơi lưu run
model_source      = xlm-roberta-base hoặc model path
expected_gpu_count= 2
use_distributed   = True
fast_dev_run      = False
```

#### Bước 7: bind 13 runner

```python
config = replace(
    config,
    phase_runners=build_kaggle_phase_runners(config),
)
```

Nói đơn giản: notebook lấy “danh sách tên phase” và nối mỗi tên với một hàm
xử lý thật.

#### Bước 8: mở `RunSession`

- `full`: mở đủ 13 phase.
- `resume`: đọc `LATEST.json`, bỏ qua phase đã PASS.
- `inference_only`: chỉ chạy preflight, source/model check, inference và pack.

Cuối cell, notebook in ra `run_id` và danh sách phase đang active.

### Pattern của 13 phase code cell

Mỗi code cell có cùng hình dạng:

```python
PHASE_NAME = "phase_01_preflight"
PHASE_INDEX = 1

if PHASE_NAME in ACTIVE_PHASES:
    log_step(PHASE_INDEX, "START", ...)
    result = run_phase(SESSION, PHASE_NAME)
    log_step(PHASE_INDEX, "END", ..., result=result)
```

Ý nghĩa:

1. biết mình đang ở phase nào;
2. báo `START` để nhìn thấy cell đã bắt đầu;
3. gọi đúng một runner;
4. chỉ ghi `PASS` sau khi runner trả về thành công;
5. in JSON result để người dùng xem ngay trong notebook.

Notebook không copy code train dài vào cell. Cell chỉ là nút bấm điều phối.

### Cell cuối — Finalization

Cell cuối gọi:

```python
SUMMARY = finish_run(SESSION)
```

Chỉ gọi khi phase cuối của chế độ hiện tại đã hoàn thành. Nó tạo
`run_manifest.json` và in summary cuối cùng.

---

## 5. Orchestrator hoạt động như thế nào?

### 5.1 `RunConfig`

Đây là cấu hình tĩnh của một run. Nó không làm việc; nó chỉ trả lời “chạy với
điều kiện nào?”.

### 5.2 `RunSession`

Đây là cuốn sổ của run đang chạy:

```text
run_id      ← tên duy nhất của run
run_dir     ← thư mục lưu run
phases      ← các phase phải chạy
completed   ← các phase đã PASS trong session hiện tại
context     ← dataset fingerprint, seed, stage info...
```

### 5.3 `start_run`

`start_run` kiểm tra trước rằng mọi phase cần chạy đều có runner. Nếu phase
runner bị thiếu, chương trình dừng ngay; không tạo cảm giác giả rằng pipeline
đã chạy.

### 5.4 `run_phase`

`run_phase` thực hiện đúng một phase:

```text
ghi PHASE_START
    ↓
gọi runner thật
    ↓ thành công
ghi artifacts/phase_xx.json với PASS
ghi PHASE_END
cập nhật LATEST.json
    ↓ lỗi
ghi artifacts/phase_xx.error.json
ghi PHASE_ERROR
không cập nhật LATEST.json
```

### 5.5 Atomic artifact lifecycle

Khi ghi JSON, code ghi vào file tạm trước, flush/fsync, sau đó rename atomically.

Ví dụ:

```text
.phase_04_build_metadata.json.<random>.tmp
        ↓ ghi xong và flush
phase_04_build_metadata.json
```

Nếu Kaggle bị ngắt giữa chừng, file PASS cũ không bị ghi đè bởi một file dở
dang. `LATEST.json` chỉ trỏ tới phase đã publish đầy đủ.

---

## 6. Giải thích từng phase

### Phase 01 — Preflight dữ liệu và provenance

#### Mục tiêu ELI5

Giống lễ tân kiểm tra hồ sơ trước khi cho bệnh nhân vào bệnh viện.

#### Code chính

`clinical_nlp_lab/kaggle_phases.py::_phase_01_preflight`

#### Việc thực hiện

1. tạo writable artifact directory;
2. copy config/KB từ `ARTIFACT_SOURCE_DIR` sang `ARTIFACT_DIR` nếu cần;
3. tạo config mặc định nếu config chưa tồn tại;
4. chạy `build_preflight_report`;
5. yêu cầu report phải có status `PASS`.

#### Kiểm tra những gì?

- TXT và GT có đi thành cặp không;
- JSON có đọc được không;
- offset có hợp lệ không;
- manifest và provenance có khớp không;
- KB ICD-10/RxNorm có đủ không;
- fingerprint có được ghi không.

#### Artifact

```text
<run_dir>/artifacts/preflight_report.json
<run_dir>/artifacts/phase_01_preflight.json
```

#### Nếu lỗi

Không được đi tiếp tới training. Sửa dataset hoặc artifact trước; không nên
“bỏ qua cho chạy thử” vì lỗi offset sẽ làm hỏng output sau này.

### Phase 02 — Resolve nguồn input/model/KB

#### Mục tiêu ELI5

Điều phối viên kiểm tra xem các thùng đồ ở đúng địa chỉ chưa.

#### Code chính

`_phase_02_resolve_sources`

#### Việc thực hiện

- kiểm tra `DATASET_ROOT` là directory;
- tìm `INPUT_SOURCE` dạng file hoặc directory;
- ghi lại model source đang dùng;
- xác nhận pipeline đang chạy online/expected network.

#### Artifact

`phase_02_resolve_sources.json` chứa đường dẫn dataset/input/model đã resolve.

#### Lỗi thường gặp

```text
DATASET_ROOT does not exist
inference input source does not exist
```

Sửa đường dẫn Dataset hoặc `INPUT_SOURCE` trong setup cell.

### Phase 03 — Inventory model và resource budget

#### Mục tiêu ELI5

Kiểm tra bệnh viện có đủ phòng và máy móc trước khi nhận bệnh nhân.

#### Code chính

`_phase_03_inventory_models`

#### Việc thực hiện

1. import PyTorch;
2. đếm GPU bằng `torch.cuda.device_count()`;
3. yêu cầu ít nhất `EXPECTED_GPU_COUNT`, mặc định là 2;
4. ghi tên từng GPU;
5. ghi distributed mode có bật hay không.

#### Vì sao fail-closed?

Nếu bạn yêu cầu T4×2 nhưng Kaggle chỉ cấp một GPU, chạy tiếp sẽ làm thời gian,
batch size và memory estimate sai. Phase này dừng sớm để tránh kết quả không
đáng tin.

#### Artifact

`phase_03_inventory_models.json`.

### Phase 04 — Build record metadata

#### Mục tiêu ELI5

Đánh số và lập mục lục cho từng hồ sơ.

#### Code chính

`_phase_04_build_metadata`

#### Việc thực hiện

- tạo metadata cho từng record/document;
- nhóm các record gần trùng nhau;
- tạo edge near-duplicate để tránh train và validation chứa bản sao của nhau;
- ghi hash và số lượng record.

#### Artifact

```text
artifacts/metadata/metadata_manifest.jsonl
artifacts/metadata/near_duplicate_edges.json
artifacts/metadata/metadata_descriptor.json
```

`metadata_descriptor.json` là bảng tóm tắt: dataset fingerprint, số record,
số document, số near-duplicate edge và hash của các file metadata.

### Phase 05 — Build fixed/OOF splits

#### Mục tiêu ELI5

Chia học sinh thành nhóm học và nhóm kiểm tra, nhưng không để hai bản sao của
cùng một bài nằm ở hai nhóm khác nhau.

#### Code chính

`_phase_05_build_splits`

#### Việc thực hiện

1. tạo fixed-fold split để pipeline chính dùng ổn định;
2. tạo thêm 5 OOF split cho đánh giá mở rộng;
3. giữ lại partition synthetic/organizer;
4. ghi manifest và fingerprint của từng split.

#### Artifact

```text
artifacts/splits/split_fixed_fold.json
artifacts/splits/split_oof_fold_0.json ... split_oof_fold_4.json
artifacts/splits/split_descriptor.json
```

Training stages đọc partition trong `split_descriptor.json`, không tự bốc dữ
liệu ngẫu nhiên một lần nữa.

### Phase 06 — Prepare owner-window training contract

#### Mục tiêu ELI5

Một bệnh án dài hơn một tờ giấy. Ta cắt nó thành các trang 512 token, nhưng
phải quyết định entity thuộc về trang nào để không đếm một entity hai lần.

#### Code chính

`_phase_06_prepare_training_contract`

#### Việc thực hiện

- load tokenizer;
- đọc document và entity;
- tạo BIO label map;
- parse record;
- tạo sample training contract;
- lưu owner-window, entity span và assertion target.

#### Artifact

`artifacts/training_contract.json`.

Phase này là contract mẫu. Các training stage sau đó tạo contract đầy đủ cho
tập train/validation của stage tương ứng.

### Phase 07 — Curriculum Stage 1

#### Mục tiêu ELI5

Cho model học bài dễ trước: tập synthetic train và synthetic validation.

#### Code chính

`_phase_07_stage1` → `_run_training_stage(..., "stage1", ...)`

#### Việc thực hiện

- tạo `stage_inputs/stage1.json`;
- chọn synthetic train/validation IDs;
- chạy `train_ner_subprocess.py` từ model nguồn ban đầu;
- dùng torch distributed nếu có ít nhất 2 GPU và `USE_DISTRIBUTED=1`;
- lưu checkpoint NER.

#### Artifact

```text
checkpoints/stage1/ner_model/
checkpoints/stage1/stage_manifest.json
artifacts/stage1_training.log
```

### Phase 08 — Curriculum Stage 2

#### Mục tiêu ELI5

Cho model học thêm dữ liệu organizer sau khi đã có nền tảng từ stage 1.

#### Việc thực hiện

- train IDs = synthetic + organizer train;
- validation IDs = synthetic + organizer validation;
- load checkpoint stage 1 làm model cha;
- tạo checkpoint stage 2.

#### Artifact

```text
checkpoints/stage2/ner_model/
checkpoints/stage2/stage_manifest.json
artifacts/stage2_training.log
```

### Phase 09 — Curriculum Stage 3

#### Mục tiêu ELI5

Chạy stage tiếp theo với manifest riêng, hash riêng và checkpoint stage 2 làm
đầu vào. Dataset partition hiện tại vẫn dựa trên synthetic + organizer, nhưng
stage 3 được ghi nhận riêng để curriculum state machine có thể áp dụng cấu hình
và resume guard riêng.

#### Artifact

```text
checkpoints/stage3/ner_model/
checkpoints/stage3/stage_manifest.json
artifacts/stage3_training.log
```

### Phase 10 — Final-fit encoder

#### Mục tiêu ELI5

Sau khi đã kiểm tra qua curriculum, cho model học trên toàn bộ train partition
để tạo encoder cuối cùng.

#### Việc thực hiện

- load checkpoint stage 3;
- train trên synthetic + organizer train + validation IDs;
- không dùng validation window riêng ở final fit;
- lưu model cuối vào `checkpoints/final_fit/ner_model`.

#### Artifact

```text
checkpoints/final_fit/ner_model/
checkpoints/final_fit/stage_manifest.json
artifacts/final_fit_training.log
```

Đây là checkpoint được phase 11 và phase 12 reload.

### Phase 11 — Fit assertion và candidate heads

#### Mục tiêu ELI5

Model NER trả lời “đây có phải bệnh/thuốc không?”. Phase này dạy thêm hai nhóm
câu hỏi:

1. entity có bị phủ định, là tiền sử hay thuộc gia đình không?
2. entity nên nối với mã chuẩn nào?

#### Assertion head

- load final encoder;
- giữ encoder ở chế độ eval và đóng băng weight;
- chỉ train một head nhỏ ở phía trên;
- học ba trục: `isNegated`, `isHistorical`, `isFamily`;
- lab name/result bị mask khỏi assertion loss;
- fit threshold cho từng trục;
- lưu encoder hash và tokenizer hash trong binding.

#### Candidate calibration

- load ICD-10 và RxNorm dictionary;
- lấy positive code từ ground truth;
- tạo một số hard negative candidate;
- fit calibration artifact;
- bind artifact vào KB fingerprint và `CandidatePolicy`.

#### Artifact

```text
artifacts/heads/assertion_head.pt
artifacts/heads/assertion_binding.json
artifacts/heads/assertion_thresholds.json
artifacts/heads/assertion_entity_type_map.json
artifacts/heads/candidate_training_artifact.json
artifacts/heads/candidate_calibration.json
artifacts/model_status.json
```

### Phase 12 — Inference raw-offset và KB recovery

#### Mục tiêu ELI5

Đem model đã học ra đọc các file input thật và tạo bài nộp.

#### Việc thực hiện

1. reload final NER checkpoint;
2. reload assertion head/threshold;
3. reload ICD-10/RxNorm KB;
4. chạy NER để tạo span proposal;
5. chạy KB-first recovery cho alias chính xác trong raw text;
6. merge các proposal không chồng sai;
7. áp assertion;
8. áp candidate policy, tối đa một candidate hoặc abstain;
9. kiểm tra lại raw offset và schema;
10. ghi output và diagnostics.

#### Vì sao “raw-offset” quan trọng?

Model có thể được chạy trên text đã normalize hoặc window đã cắt. Nhưng output
phải trỏ đúng về text gốc:

```text
raw_text[start:end] == entity_text
```

Nếu không đúng, downstream không biết entity nằm ở đâu trong bệnh án.

#### Artifact

```text
output/
diagnostics/
output.zip
```

### Phase 13 — Validate và packaging

#### Mục tiêu ELI5

Đóng toàn bộ hồ sơ thành một gói có niêm phong và kiểm tra niêm phong.

#### Việc thực hiện

- lấy toàn bộ file trong run directory;
- tạo `trained_artifacts.zip`;
- chạy CRC test bằng `ZipFile.testzip()`;
- ghi inventory package.

#### Artifact

```text
trained_artifacts.zip
artifacts/package_inventory.json
```

Nếu CRC fail, phase 13 dừng và không coi package là hợp lệ.

---

## 7. Owner-window, collator và curriculum — giải thích thật chậm

### 7.1 Vì sao phải cắt window?

Tokenizer/model có giới hạn độ dài. Một bệnh án có thể dài hàng nghìn token,
trong khi model chỉ đọc được tối đa 512 token một lần.

Pipeline dùng:

```text
max_length = 512
stride     = 128
```

Nghĩa là trang sau chồng lên trang trước 128 token để entity ở biên trang không
bị mất.

### 7.2 Owner window là gì?

Nếu một entity xuất hiện trong nhiều window do overlap, chỉ một window được gọi
là **owner**. Window owner được phép tính loss cho entity; window không owner
bị mask.

Điều này giống một học sinh chỉ được tính điểm một lần cho một câu hỏi, dù câu
hỏi xuất hiện trên nhiều bản photo.

### 7.3 Collator làm gì?

Collator biến danh sách window Python thành batch tensor cho PyTorch:

```text
TokenWindow[]
      ↓
ClinicalTokenCollator
      ↓
input_ids
attention_mask
ner_labels
entity_spans
entity_types
assertion_targets
assertion_mask
```

Các tensor có thể được đưa lên GPU.

### 7.4 Assertion target là gì?

Mỗi entity disease/drug/symptom có ba nhãn nhị phân:

```text
isNegated   = có “không”, “không có” không?
isHistorical = có phải tiền sử không?
isFamily    = có phải bệnh của người nhà không?
```

`LAB_NAME` và `LAB_RESULT` không dùng assertion head nên được mask khỏi loss.

### 7.5 Curriculum và resume hash

Mỗi stage có manifest ghi:

```text
stage name
dataset fingerprint
split fingerprint
config fingerprint
parent checkpoint hash
```

Nếu bạn đổi dataset nhưng cố dùng checkpoint cũ, hash mismatch sẽ chặn resume.
Đó là cơ chế bảo vệ, không phải lỗi vô nghĩa.

---

## 8. Training subprocess và T4×2

File thực hiện training là:

`scripts/train_ner_subprocess.py`

Nó làm các bước:

1. đọc command-line arguments từ phase runner;
2. load config;
3. load annotated documents;
4. validate documents;
5. nếu có stage manifest, chọn đúng train/validation IDs;
6. build training contract;
7. load tokenizer;
8. load `AutoModelForTokenClassification`;
9. tạo Hugging Face `Trainer`;
10. train;
11. lưu model/tokenizer/checkpoint;
12. ghi `training_result.json`.

Khi có hai GPU, phase runner tạo command tương đương:

```text
python -m torch.distributed.run \
  --standalone \
  --nproc_per_node 2 \
  scripts/train_ner_subprocess.py ...
```

`torch.distributed.run` khởi chạy hai process, mỗi process phụ trách một GPU.
Hugging Face Trainer dùng distributed environment đó để đồng bộ gradient.

Không chạy phần này local trong workflow hiện tại. Local chỉ chạy test contract
và fake runner; training thật dành cho Kaggle T4×2.

---

## 9. Ba chế độ chạy

### `RUN_MODE=full`

Chạy phase 01 → 13.

Dùng cho lần chạy train đầu tiên.

### `RUN_MODE=resume`

Đọc:

```text
/kaggle/working/run_output/LATEST.json
```

Ví dụ `LATEST.json` nói phase cuối là `phase_08_stage2`, notebook chạy tiếp từ
phase 09.

Resume chỉ an toàn khi:

- cùng config fingerprint;
- cùng dataset fingerprint;
- cùng run ID nếu bạn đặt `RUN_ID`;
- checkpoint/manifest tương ứng còn tồn tại.

### `RUN_MODE=inference_only`

Không train. Chạy:

```text
phase 01 preflight
phase 02 resolve sources
phase 03 inventory GPU/model
phase 12 inference
phase 13 packaging
```

Dùng khi đã có final checkpoint và chỉ muốn tạo output mới.

---

## 10. Những file cần xem khi Kaggle lỗi

Giả sử run ID là `run-abc123`:

```text
/kaggle/working/run_output/
├── LATEST.json
└── run-abc123/
    ├── run.jsonl
    ├── run_manifest.json              ← có nếu finalization chạy
    ├── artifacts/
    │   ├── phase_01_preflight.json
    │   ├── phase_02_resolve_sources.json
    │   ├── phase_08_stage2.error.json ← file quan trọng khi lỗi
    │   ├── stage1_training.log
    │   └── package_inventory.json
    ├── checkpoints/
    ├── diagnostics/
    ├── output.zip
    └── trained_artifacts.zip
```

### Đọc `run.jsonl`

Mỗi dòng là một JSON event. Các event quan trọng:

```text
PHASE_START  ← phase bắt đầu
PHASE_END    ← phase PASS
PHASE_ERROR  ← phase dừng vì exception
```

Tìm event cuối cùng để biết pipeline dừng ở đâu.

### Đọc `*.error.json`

File này có:

```json
{
  "phase": "phase_08_stage2",
  "status": "ERROR",
  "error_type": "RuntimeError",
  "error": "..."
}
```

Khi gửi log để xử lý, gửi cả file này và traceback trong notebook.

---

## 11. Bảng lỗi thường gặp

| Lỗi | Nghĩa đơn giản | Cách xử lý |
|---|---|---|
| `clinical_nlp_lab is unavailable` | Không clone được hoặc PROJECT_ROOT sai | Bật Internet, kiểm tra branch/URL, hoặc đặt `PROJECT_ROOT_OVERRIDE` |
| `DATASET_ROOT does not exist` | Kaggle không thấy dataset train | Kiểm tra Dataset đã attach và đặt `DATASET_ROOT` |
| `input source does not exist` | Không thấy file/dir inference | Đặt `INPUT_SOURCE` đúng path |
| `expected at least 2 CUDA devices` | Không phải T4×2 | Chọn GPU T4×2, đặt đúng `EXPECTED_GPU_COUNT` |
| preflight status `FAIL` | TXT/GT/manifest/KB có vấn đề | Đọc `preflight_report.json` |
| download model timeout | Không tải được XLM-R | Bật Internet hoặc attach model cache |
| CUDA out of memory | Batch/model quá lớn | Gửi training log; giảm profile theo hướng dẫn, không tự xóa artifact |
| `resume config fingerprint mismatch` | Config hiện tại khác lúc train | Dùng đúng config hoặc chạy `full` mới |
| `phase runner missing` | Notebook không bind implementation | Dùng notebook canonical mới nhất |
| `no positive candidates` | KB không có candidate ground truth | Kiểm tra ICD-10/RxNorm artifact và coverage |
| `CRC validation failed` | ZIP bị hỏng | Gửi package/log; chạy lại phase 13 sau khi nguyên nhân được sửa |

---

## 12. Cách chạy trên Kaggle — checklist ELI5

### Trước Run All

- [ ] Upload notebook mới nhất.
- [ ] Attach Dataset có `synthetic_train_v2` và `input`.
- [ ] Bật Internet.
- [ ] Chọn GPU T4×2.
- [ ] Để `RUN_MODE=full`.
- [ ] Để `FAST_DEV_RUN=0` cho training thật.
- [ ] Để `EXPECTED_GPU_COUNT=2`.
- [ ] Để `USE_DISTRIBUTED=1`.

### Khi bấm Run All

1. Cell setup phải in `[GIT_CLONE]` hoặc xác nhận repo đã tồn tại.
2. Phase 01 phải tạo preflight report.
3. Phase 03 phải in hai GPU.
4. Phase 07–10 sẽ lâu nhất vì đây là training.
5. Phase 11 tạo assertion/candidate heads.
6. Phase 12 tạo output.
7. Phase 13 tạo package.

### Sau Run All

Tải về tối thiểu:

```text
run.jsonl
LATEST.json
run_manifest.json
output.zip
trained_artifacts.zip
```

Nếu lỗi, tải thêm:

```text
artifacts/<phase>.error.json
artifacts/<stage>_training.log
```

---

## 13. Những điều pipeline không làm

- Notebook không chứa bản copy khổng lồ của business logic.
- Local không train model trong bộ kiểm thử hiện tại.
- Phase lỗi không được tự đánh dấu `PASS`.
- `LATEST.json` không được cập nhật nếu artifact phase chưa publish xong.
- Qwen không phải đường chính bắt buộc; đường chính là NER + KB + candidate policy.
- Không tự coi “đã chạy hết cell” là “đã nghiệm thu Kaggle”. Cần artifact thật.

Trạng thái chính xác của dự án là:

```text
Implementation và pre-Kaggle verification: hoàn tất
Kaggle Save Version → Run All thật: người dùng thực hiện
```

---

## 14. Glossary — từ điển mini

| Từ | Nghĩa ELI5 |
|---|---|
| Dataset | Tập dữ liệu để máy học |
| Ground truth/GT | Đáp án đúng do con người cung cấp |
| Token | Mảnh nhỏ của câu sau khi tokenizer cắt |
| Tokenizer | Máy cắt câu thành token |
| Entity | Một đoạn text có ý nghĩa, ví dụ tên bệnh |
| Span | Vị trí bắt đầu/kết thúc của entity trong text |
| NER | Bài toán tìm entity trong câu |
| BIO label | Nhãn Begin/Inside/Outside cho entity |
| Window | Một đoạn tối đa 512 token |
| Owner window | Window duy nhất được tính điểm cho entity |
| Collator | Máy gom nhiều mẫu thành tensor batch |
| Encoder | Model biến token thành vector hiểu ngữ cảnh |
| Head | Lớp nhỏ phía trên encoder để trả lời một câu hỏi |
| Assertion | Trạng thái phủ định/tiền sử/gia đình |
| Candidate | Mã chuẩn có thể ứng viên cho entity |
| KB | Knowledge base, ở đây là từ điển mã chuẩn |
| Calibration | Chuyển score thành quyết định đáng tin hơn |
| Checkpoint | Bản lưu model tại một thời điểm |
| Curriculum | Lộ trình train theo nhiều stage |
| DDP | Chia training cho nhiều GPU/process |
| Resume | Chạy tiếp từ phase/checkpoint đã PASS |
| Fingerprint | Dấu vân tay hash của dữ liệu/config |
| Artifact | File bằng chứng do pipeline tạo |
| CRC | Kiểm tra file ZIP có bị hỏng không |

---

## 15. Đọc code từ đâu nếu muốn học sâu hơn?

Đọc theo thứ tự này sẽ ít bị ngợp:

1. `medical_information_extraction_kaggle.ipynb` — xem cell nào gọi API nào.
2. `clinical_nlp_lab/orchestration.py` — hiểu session, log và atomic lifecycle.
3. `clinical_nlp_lab/kaggle_phases.py` — hiểu phase thật làm gì.
4. `scripts/train_ner_subprocess.py` — hiểu một stage training.
5. `clinical_nlp_lab/training.py` — hiểu contract/window/batch.
6. `clinical_nlp_lab/inference.py` và `runtime_bundle.py` — hiểu inference.
7. `KAGGLE_RUNBOOK.md` — hiểu cách vận hành trên Kaggle.

Nếu chỉ muốn xử lý một lỗi Kaggle, không cần đọc toàn bộ code. Hãy gửi phase,
`run.jsonl`, file `.error.json` và traceback; chỉ cần bắt đầu từ phase lỗi.
