# Thiết kế pipeline huấn luyện modular precision-first trên Colab

Ngày: 2026-07-19

Trạng thái: Đã duyệt định hướng

Nhánh nền: `develop` tại commit `d0cae07`

## 1. Mục tiêu

Xây dựng ba pipeline huấn luyện độc lập nhưng dùng chung hợp đồng dữ liệu:

1. NER y khoa tiếng Việt.
2. Fine-tune embedding BGE-M3 cho ICD-10 và RxNorm.
3. Qwen2.5-7B QLoRA để rerank tập ứng viên đã được retriever sinh ra.

Hệ thống phải:

- Huấn luyện được tuần tự trên Google Colab miễn phí, giả định GPU T4 16 GB.
- Lưu checkpoint lên Google Drive và resume sau khi runtime bị ngắt.
- Chạy inference cuối trên GPU.
- Tối ưu theo hướng precision-first; metric chọn cấu hình chính là exact-match `F0.5`.
- Giữ BM25 là tín hiệu retrieval chính.
- Không làm rò rỉ nhãn holdout hoặc public input vào training.
- Có thể bật, tắt hoặc rollback từng model mà không phá pipeline hiện tại.

## 2. Ngoài phạm vi

- Không huấn luyện đồng thời ba model theo kiểu end-to-end.
- Không full-fine-tune BGE-M3 hoặc Qwen2.5-7B trên Colab miễn phí.
- Không dùng `data/input/` làm dữ liệu huấn luyện.
- Không dùng `data/dev/gt/181.json` đến `200.json` để chọn model, threshold hoặc epoch.
- Không coi metric trên synthetic là bằng chứng về chất lượng trên dữ liệu thật.
- Không thay thế assertion rules hiện tại bằng model học máy trong phiên bản đầu.

## 3. Kiến trúc tổng thể

```text
Validated source records
        |
        v
Canonical dataset + immutable split manifest
        |
        +--> NER BIO dataset ----------> XLM-R base adapter/checkpoint
        |
        +--> Entity-code pairs --------> BGE-M3 adapter --> rebuilt FAISS
        |
        +--> Frozen candidate pools ---> Qwen 7B QLoRA adapter
        |
        v
Trusted out-of-fold calibration
        |
        v
Locked pipeline config --> one-time holdout --> GPU inference
```

Mỗi model có config, checkpoint, metric report và artifact manifest riêng. Pipeline tích hợp chỉ được dùng artifact đã qua validation và có fingerprint khớp.

## 4. Nguồn dữ liệu và mức tin cậy

| Nguồn | Vai trò | Được cập nhật gradient |
|---|---|---:|
| `data/synthetic_train_v1/` | Pretrain và augmentation | Có |
| `data/dev/input`, `data/dev/gt`, ID 1–100 | Pseudo-label, error discovery và regression | Không mặc định |
| `data/dev/input`, `data/dev/gt`, ID 101–180 | Trusted refinement và model selection | Có |
| `data/dev/input`, `data/dev/gt`, ID 181–200 | Final holdout | Không |
| `data/input/` | Public inference và đóng gói bài nộp | Không |

Bộ synthetic được giả định có 2.000 record và chỉ được promote từ `.failed-validation` sang `synthetic_train_v1` khi toàn bộ validation gate đạt.

## 5. Canonical record

Mỗi bệnh án được biểu diễn bằng một record JSONL:

```json
{
  "record_id": "synthetic-0001",
  "source": "synthetic",
  "trust_tier": "synthetic_validated",
  "text": "Bệnh án gốc...",
  "entities": [
    {
      "text": "tăng huyết áp",
      "type": "CHẨN_ĐOÁN",
      "position": [10, 24],
      "assertions": {
        "isNegated": false,
        "isHistorical": false,
        "isFamily": false
      },
      "codes": ["I10"]
    }
  ],
  "sha256": "...",
  "split_group": "..."
}
```

Ràng buộc:

- `position` là Unicode half-open `[start, end)`.
- `text[start:end]` phải bằng chính xác `entity.text`.
- ICD-10 và RxNorm phải tồn tại trong `metadata.db`.
- Mọi sample dẫn xuất từ cùng một record phải nằm trong cùng split.
- Record không được chứa đường dẫn máy cụ thể.
- Artifact dẫn xuất phải ghi source fingerprint, DB fingerprint, seed và Git commit.

## 6. Validation gate

Dataset builder từ chối xử lý synthetic nếu có bất kỳ lỗi nào:

- Không đủ đúng 2.000 input và 2.000 ground-truth.
- Sai schema, type, span, assertion hoặc namespace code.
- Exact duplicate.
- Near-duplicate vượt ngưỡng cấu hình.
- Trùng hoặc gần trùng với `data/dev/input` hay `data/input`.
- Code không tồn tại trong database.
- Thiếu provenance, manifest hoặc checksum.
- Tham chiếu nhãn holdout trong quá trình generation.

Khi validation thất bại, output phải ở thư mục có hậu tố `.failed-validation`. Không trainer nào được phép đọc thư mục này.

## 7. Split và chống leakage

### 7.1 Synthetic

- 1.700 record train.
- 300 record validation.
- Seed cố định và được ghi vào split manifest.
- Stratify theo family, entity type và assertion combination.
- Các template hoặc near-duplicate cluster phải nằm cùng một split.
- Synthetic validation chỉ dùng early stopping và phát hiện regression.

### 7.2 Trusted five-fold

ID 101–180 được chia thành năm fold, mỗi fold 16 record:

1. Khởi tạo từ synthetic checkpoint.
2. Fine-tune trên 64 record.
3. Đánh giá trên 16 record còn lại.
4. Gom out-of-fold predictions của cả năm fold.
5. Chọn threshold, epoch budget và hyperparameter bằng exact `F0.5`, precision là tiebreaker.
6. Reset về synthetic checkpoint.
7. Fine-tune final artifact trên toàn bộ 101–180 bằng cấu hình đã khóa.
8. Chạy 181–200 đúng một lần.

Mọi fold phải group theo record và kiểm tra leakage trước khi train.

## 8. Nền tảng training dùng chung

Package dự kiến:

```text
src/training/
├── artifacts.py
├── data_contracts.py
├── fingerprints.py
├── split_builder.py
├── validation.py
├── ner/
├── embedding/
└── reranker/
```

Config:

```text
configs/training/
├── data.yaml
├── ner_xlmr_base.yaml
├── embedding_bge_m3_lora.yaml
└── reranker_qwen25_7b_qlora.yaml
```

Notebook Colab chỉ mount Drive, cài dependency và gọi CLI. Logic xử lý dữ liệu và huấn luyện phải nằm trong `src/training`, không nằm riêng trong notebook.

Mỗi artifact chứa:

- Model/adapter checkpoint.
- Tokenizer nếu có thay đổi.
- Config đã resolve.
- Dataset và split fingerprint.
- Git commit.
- Seed.
- Metric report.
- Artifact manifest.
- Trạng thái `candidate`, `validated` hoặc `locked`.

Write phải atomic: ghi vào thư mục tạm, validation xong mới rename thành artifact chính thức.

## 9. Pipeline NER

### 9.1 Model

- Backbone mặc định: `xlm-roberta-base`.
- Head: token classification.
- Không dùng CRF ở baseline đầu tiên.
- Dùng constrained BIO decoding để ngăn chuỗi nhãn không hợp lệ.
- CRF chỉ được thêm nếu ablation trên trusted out-of-fold cải thiện `F0.5` mà không giảm precision.

`xlm-roberta-large` không phải mặc định vì chi phí VRAM và thời gian trên T4 không tương xứng với 2.000 synthetic record.

### 9.2 Chuẩn bị nhãn

- Tokenize raw text bằng fast tokenizer và `return_offsets_mapping`.
- Gán `B-*` cho token đầu, `I-*` cho các token tiếp theo giao với span.
- Special token và padding dùng label `-100`.
- Dataset validator từ chối nested hoặc crossing span không được competition schema hỗ trợ.
- Chunk dài theo ranh giới clinical section/câu; offset output luôn quy về văn bản gốc.

### 9.3 Train và metric

- FP16 trên T4.
- Gradient accumulation và gradient checkpointing có thể cấu hình.
- Early stopping trên synthetic validation.
- Trusted refinement bằng five-fold.
- Metric chính: exact entity type + exact span `F0.5`.
- Báo cáo thêm precision, recall, per-type metrics và boundary error.
- Threshold confidence được chọn từ trusted out-of-fold predictions.

### 9.4 Tích hợp runtime

Ba mode:

- `rule`: hành vi hiện tại.
- `model`: chỉ dùng checkpoint NER.
- `hybrid`: model đề xuất span, rule engine xác nhận type/context và giữ các dictionary match có precision đã được xác minh.

Mặc định triển khai đầu tiên là `hybrid`. Nếu checkpoint thiếu hoặc fingerprint sai, pipeline fail closed về `rule`.

Assertion (`isNegated`, `isHistorical`, `isFamily`) tiếp tục dùng analyzer hiện tại. Việc học assertion bằng model là một pha riêng sau khi NER span ổn định.

## 10. Pipeline embedding BGE-M3

### 10.1 Model và chiến lược

- Base model: BGE-M3 hiện tại.
- Fine-tune bằng adapter/LoRA, không full-fine-tune.
- FP16, gradient checkpointing, micro-batch nhỏ và accumulation.
- Checkpoint adapter lưu trên Drive.

### 10.2 Dataset

Positive:

- Mention/context tới ICD-10 đúng.
- Mention/context tới RxNorm đúng.
- Tên Việt, tên Anh và synonym đã xác minh của cùng code.

Hard negative:

- BM25 top results sai nhưng lexical overlap cao.
- Semantic top results sai.
- ICD sibling hoặc code gần trong hierarchy.
- Thuốc gần tên, khác hoạt chất hoặc khác dạng/liều.

Hard negative chỉ được sinh sau split. Label từ validation/holdout không được dùng để tạo negative cho train.

### 10.3 Metric và selection gate

Embedding report gồm:

- Recall@1, Recall@5, Recall@10.
- MRR@10.
- nDCG@10.
- Candidate coverage trước reranker.
- Downstream exact code `F0.5`.

Adapter chỉ được promote nếu:

- Không làm giảm candidate recall so với BGE-M3 baseline ngoài tolerance đã cấu hình.
- Cải thiện trusted out-of-fold downstream `F0.5`.
- Không giảm precision khi giữ cùng candidate budget.

### 10.4 Index và fusion

Fine-tune xong phải sinh lại toàn bộ embedding và FAISS index. Index manifest phải ghi model/adapter fingerprint; runtime từ chối index sinh bởi model khác.

Fusion giữ dạng:

```text
score = alpha * normalized_bm25 + (1 - alpha) * normalized_semantic
```

Ràng buộc:

- `0 <= alpha <= 1`.
- Tổng trọng số bằng `1`.
- BM25-first, `alpha` mặc định `0.75`.
- Alpha cuối được chọn trên trusted out-of-fold predictions và khóa trước holdout.

## 11. Pipeline Qwen2.5-7B QLoRA reranker

### 11.1 Model

- `Qwen2.5-7B-Instruct`.
- Quantization 4-bit NF4.
- LoRA trên attention và feed-forward projection modules.
- FP16 trên T4.
- Gradient checkpointing.
- Sequence length mặc định 1.024 token.
- Micro-batch 1, gradient accumulation cấu hình được.
- Checkpoint định kỳ lên Drive và resume theo manifest.

### 11.2 Dataset

Input:

- Clinical context/chunk.
- Entity text và type.
- Assertion đã phân tích.
- Candidate codes và mô tả từ frozen retriever.

Output duy nhất:

```json
{"selected_codes": ["I10"]}
```

Cho phép:

- Chọn tối đa hai code từ candidate pool.
- Trả `[]` để abstain khi không đủ bằng chứng.

Không cho phép:

- Sinh code ngoài candidate pool.
- Thêm giải thích tự do.
- Dùng candidate pool được tạo bởi retriever khác fingerprint.

Nếu code đúng không nằm trong top-k, record được ghi là retrieval miss và chuyển thành hard-negative evidence cho embedding. Không tự chèn code đúng vào candidate pool rồi dùng làm số liệu retrieval.

### 11.3 Metric và runtime safety

Metric:

- Exact code precision, recall, `F0.5`.
- Invalid JSON rate.
- Out-of-pool rate.
- Abstention rate.
- Performance theo entity type và clinical section.

Out-of-pool rate của artifact hợp lệ phải bằng `0`. Runtime parser luôn kiểm tra subset và fallback về deterministic ranking nếu model lỗi.

Backend:

- `disabled`.
- `http` tương thích API hiện tại.
- `local_transformers` để inference trực tiếp trên GPU.

## 12. Thứ tự huấn luyện trên Colab

Mỗi giai đoạn dùng một runtime riêng:

1. Build và validate canonical dataset.
2. Train NER synthetic checkpoint, trusted folds và final adapter.
3. Train BGE-M3 adapter, rebuild FAISS và benchmark retrieval.
4. Freeze NER + retriever, sinh candidate pool.
5. Train Qwen QLoRA, trusted folds và final adapter.
6. Chạy integrated development evaluation.
7. Khóa config và artifact hashes.
8. Chạy final holdout một lần.
9. Chạy public inference và package submission.

Không giữ đồng thời BGE-M3, XLM-R và Qwen trong VRAM trong quá trình train.

## 13. Precision-first selection

Metric chọn cấu hình:

```text
F0.5 = 1.25 * precision * recall / (0.25 * precision + recall)
```

Thứ tự so sánh:

1. Artifact phải qua schema, leakage và safety gates.
2. Exact `F0.5` cao hơn.
3. Nếu chênh lệch nằm trong tolerance, chọn precision cao hơn.
4. Nếu vẫn bằng nhau, chọn model nhỏ/nhanh hơn.
5. Recall floor là config bắt buộc để tránh model abstain gần như toàn bộ.

Không chọn model dựa trên synthetic metric hoặc relaxed-overlap metric.

## 14. Kiểm thử

### Unit

- Canonical schema và Unicode offsets.
- BIO alignment.
- Constrained decoding.
- Deterministic/group-aware split.
- Leakage detector.
- Hard-negative generation.
- Candidate-pool subset parser.
- Artifact fingerprint và atomic promotion.

### Integration

- Build dataset từ fixture nhỏ.
- Train một batch bằng mocked/tiny backbone.
- Save và resume checkpoint.
- Rebuild index nhỏ và kiểm tra model/index fingerprint.
- Chạy pipeline hybrid với model bật/tắt.
- Fallback khi checkpoint, API hoặc JSON lỗi.

### Colab smoke

- GPU và CUDA được nhận diện.
- Một mini epoch chạy được.
- Checkpoint xuất hiện trên Drive.
- Restart runtime và resume đúng global step.

### Acceptance

- Full test suite hiện tại tiếp tục pass.
- Các CLI có `--help` và validate path/config.
- Không có path Windows hardcode trong training package.
- Không trainer nào đọc `.failed-validation`, pseudo-label hoặc holdout trái phép.
- Mọi promoted artifact có manifest và metric report.

## 15. Bảo vệ dữ liệu

Không được xóa hoặc ghi đè:

- `data/dev/input/`.
- `data/dev/gt/`.
- `data/input/`.
- `data/kb/metadata.db`.
- `data/models/bge-m3/`.
- BGE-M3 FAISS index đã khóa.
- Final artifact và locked config đã dùng cho holdout.

Trên Colab, `data/models`, `data/kb` và `data/input` có thể là symlink tới Drive; xóa nội dung qua symlink đồng nghĩa xóa bản gốc trên Drive.

## 16. Phân rã triển khai

### Pha A: Data foundation

Canonical contract, validation gate, split manifest, fingerprints và ba dataset builder.

### Pha B: NER

XLM-R training, constrained decoding, trusted refinement và hybrid runtime.

### Pha C: Embedding

BGE-M3 adapter training, hard negatives, index rebuild và BM25-first calibration.

### Pha D: Reranker

Frozen candidate dataset, Qwen QLoRA, strict subset parser và GPU backend.

### Pha E: Integrated evaluation

Artifact locking, out-of-fold model selection, one-time holdout và submission packaging.

Mỗi pha có implementation plan, test gate và commit riêng. Pha sau chỉ bắt đầu khi interface và artifact contract của pha trước ổn định.
