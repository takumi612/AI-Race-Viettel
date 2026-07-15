# MASTER PLAN v4 — AI Race 2026, Bài 2 (Ontological Reasoning in Medical Knowledge Retrieval)

## 0. Ghi chú quyết định đã chốt (đọc trước khi làm bất cứ việc gì)

Đây là những điểm từng mơ hồ ở bản v3, nay đã được chốt rõ — mọi thiết kế bên dưới đều dựa trên các quyết định này. Nếu sau này có ai muốn đổi lại, phải sửa lại toàn bộ phần liên quan, không chỉ đổi một dòng.

1. **Phạm vi hiện tại = tối ưu cho Vòng 1 (Sơ loại).** Không xây dựng API endpoint phục vụ realtime cho Vòng 2/3 ở giai đoạn này. Kiến trúc pipeline vẫn nên viết theo dạng module hóa rõ ràng (để sau này bọc thành API không phải viết lại từ đầu), nhưng công sức triển khai serving thực tế sẽ đánh giá lại sau khi có kết quả Vòng 1. **Rủi ro đã chấp nhận:** nếu vào Vòng 2, sẽ có ít thời gian chuẩn bị hơn — chấp nhận đánh đổi này để tập trung 100% cho 15 ngày còn lại của Vòng 1.
2. **Giới hạn 9B tham số áp dụng cho TỪNG model self-host riêng lẻ**, không cộng dồn toàn hệ thống. Nghĩa là NER model, embedding model, và LLM reranker mỗi cái được phép tối đa 9B tham số của riêng nó — không phải tổng 3 model cộng lại phải ≤ 9B. Điều này mở ra dư địa: có thể chọn LLM reranker mạnh hơn (kể cả gần chạm 9B) mà không lo ảnh hưởng tới ngân sách của NER/embedding.
3. **Reproducibility (chạy được trên máy người khác) là yêu cầu hạng nhất ngay từ đầu**, không phải việc dồn vào cuối chỉ khi lọt top 15. Lý do: (a) không biết trước có lọt top 15 hay không nên chuẩn bị sẵn an toàn hơn; (b) việc containerize sớm giúp phát hiện bug môi trường (thiếu dependency, sai đường dẫn tuyệt đối, quên set seed...) ngay khi còn dễ sửa, thay vì dồn cục vào 1-2 ngày cuối khi không còn thời gian xử lý sự cố.
4. **Baseline chạy được end-to-end phải có trong 1-2 ngày đầu tiên**, dù chất lượng thấp, và phải nộp thử lên leaderboard sớm để đối chiếu điểm thật với điểm tự chấm bằng `metrics.py` — tránh việc tối ưu suốt 2 tuần theo một thước đo tự suy diễn rồi mới phát hiện lệch với cách BTC chấm thật.
5. **Chống rò rỉ dữ liệu và Overfit:** Tuyệt đối không sử dụng bất kỳ tệp dữ liệu nào từ thư mục `input` của BTC để làm dữ liệu kiểm thử (dev set) hoặc huấn luyện (train set). Tập dữ liệu kiểm thử cục bộ (`data/dev/`) phải được trích xuất từ dữ liệu tự sinh (Synthetic Data) ở Giai đoạn 1 nhằm đảm bảo an toàn tuyệt đối và tính khách quan cho mô hình.
6. **Quản lý dữ liệu tập trung (`data/`):** Toàn bộ dữ liệu dự án bao gồm CSDL y khoa (`kb/`), dữ liệu tự sinh thô (`raw/`), dữ liệu đã gán nhãn (`processed/`), tập validation (`dev/`), dữ liệu đầu vào/ra của cuộc thi (`input/`, `output/`) phải được quản lý tập trung tại thư mục [data/](../data/) để phục vụ Dockerize và đồng bộ hóa môi trường phát triển.
7. **Cơ chế Fail-safe cho Bộ kiểm duyệt lâm sàng (`ClinicalValidator`):** Bộ kiểm duyệt lâm sàng (lọc mã bệnh theo tuổi, giới tính) phải hoạt động theo nguyên tắc cắm rút an toàn:
   * Chỉ trích xuất thông tin tuổi và giới tính trong 150 ký tự đầu của văn bản.
   * Hủy kết quả trích xuất nếu phát hiện từ khóa liên quan đến người thân (bố, mẹ, vợ...) hoặc nhân viên y tế (bác sĩ...).
   * Trong trường hợp trích xuất thất bại hoặc trả về `None`, validator sẽ tự động giữ nguyên toàn bộ candidates gốc của mô hình NER để bảo vệ an toàn cho điểm số.
8. **Thời gian chờ 600 giây trên hệ thống BTC nhiều khả năng là thời gian cooldown giữa 2 lần nộp (rate limit)** hơn là thời gian giới hạn chạy pipeline. Do hệ thống chạy offline và nộp file tĩnh `output.zip`, rate-limit cooldown này không ảnh hưởng trực tiếp đến thiết kế thời gian chạy của pipeline, nhưng đòi hỏi mỗi lượt nộp phải cực kỳ chất lượng (chỉ có 5 lượt/ngày).
9. **Chiến lược "Ship sớm, Iterate nhanh" (đã chốt — 15/07):** Với 15 ngày còn lại (15/07 → 30/07), mọi quyết định thiết kế phải ưu tiên **có bài nộp chạy được** trước, sau đó mới tối ưu chất lượng. Cụ thể: (a) Baseline LLM few-shot phải có trước khi bắt tay train NER riêng; (b) QLoRA fine-tune LLM reranker là "nice-to-have", zero-shot/few-shot là mức tối thiểu bắt buộc; (c) NLI grounding verifier (mDeBERTa-XNLI) bị loại bỏ hoàn toàn — constrained decoding + logprobs đủ mạnh và đơn giản hơn.
10. **Hạ cấp kỹ thuật có chủ đích để tối ưu thời gian:**
    * **Loại bỏ:** NLI verifier (mDeBERTa-XNLI), CTranslate2 optimization, A/B test backbone NER (chọn thẳng XLM-R).
    * **Hạ cấp:** QLoRA fine-tune LLM → zero-shot/few-shot trước, QLoRA chỉ khi còn ≥ 5 ngày trước deadline.
    * **Thay thế:** Baseline rule-based/regex NER → Baseline LLM few-shot NER (nhanh hơn xây, chất lượng tốt hơn regex đáng kể).
    * **Bổ sung:** SapBERT-XLMR cho medical entity linking (nếu bge-m3 chưa đủ tốt trên domain y khoa).
11. **Cơ chế nhật ký thực nghiệm (Experiment Tracking):** Để tránh mất dấu cấu hình tối ưu khi tune nhiều tham số (backbone, ngưỡng epsilon, trọng số RRF, có QLoRA hay không), bắt buộc duy trì tệp [docs/experiment_log.md](experiment_log.md) để ghi chép có hệ thống: cấu hình → điểm tự chấm `metrics.py` → điểm leaderboard thật sau mỗi lần nộp.

---

## 1. Cấu trúc thư mục dự án (Tái cấu trúc Modular)

Để tối ưu hóa khả năng làm việc song song, quản lý dữ liệu tập trung và đóng gói Docker dễ dàng, dự án được tổ chức lại theo cấu trúc sau:

```text
d:\AI Race Viettel\
├── docs/                      # Tài liệu phân tích và kế hoạch triển khai
│   ├── Kế hoạch triển khai.md
│   ├── Phân tích đề bài.md
│   └── eval_assumptions.md
├── data/                      # Quản lý tập trung dữ liệu y khoa (gitignored tệp lớn)
│   ├── kb/                    # Cơ sở tri thức (ICD10.xlsx, metadata.db, các file context)
│   ├── raw/                   # Chứa dữ liệu tự sinh thô (Synthetic Data) (Người B)
│   ├── processed/             # Dữ liệu BIO đã được gán nhãn để train NER (Người B)
│   ├── dev/                   # Tập dữ liệu validation cục bộ (chứa file .gitkeep, sẽ sinh sau)
│   ├── input/                 # 100 file .txt đầu vào tập test của BTC (Người dùng cung cấp)
│   └── output/                # Kết quả JSON dự đoán đầu ra của pipeline (chứa file .gitkeep)
├── src/                       # Mã nguồn chính của dự án
│   ├── data_generation/       # Sinh dữ liệu huấn luyện (Synthetic / BIO format) (Người B)
│   ├── ner/                   # Mô hình trích xuất thực thể (XLM-R) (Người B)
│   ├── assertion/             # Phân tích thuộc tính (isNegated, isFamily, isHistorical) (Người B + A)
│   ├── retrieval/             # Truy xuất candidates (BM25s + FAISS) (Người A)
│   ├── ranking/               # Định chuẩn & LLM Reranker (Người B)
│   ├── validation/            # Bộ kiểm duyệt lâm sàng (clinical_validator.py) (Người A)
│   ├── pipeline/              # Bộ điều phối (Orchestrator) kết nối các module trên để xuất file JSON đúng schema
│   ├── utils/
│   │   ├── paths.py           # Quản lý đường dẫn tập trung (Single Source of Truth)
│   │   └── setup_db.py        # Thiết lập SQLite y khoa
│   ├── metrics.py             # Engine tự chấm điểm cục bộ
│   └── evaluate.py            # CLI chạy chấm điểm cục bộ
```

---

## 3. Giai đoạn 0: Giải mã đề bài & Evaluation Engine — [Người A]

Quan trọng nhất và dễ bị bỏ qua nhất, vì không tạo ra sản phẩm nhìn thấy được nhưng nếu sai, mọi nỗ lực train ở các giai đoạn sau bị tối ưu lệch hướng.

### 3.1. Vì sao phải làm trước tiên

Công thức chấm có 3 thành phần (`text_score`, `assertions_score`, `candidates_score`), tính theo từng sample — nghĩa là phải biết prediction nào khớp với ground truth nào trước khi tính. Đề bài không nói rõ thuật toán ghép cặp, chỉ có 1 manh mối: đoán đúng text nhưng sai type thì khái niệm bị tính 2 lần, mỗi lần 0 điểm — câu này ngầm xác nhận có một bước matching trước khi chấm.

### 3.2. Thuật toán matching prediction ↔ ground truth

1. Với mỗi cặp (prediction, ground truth) trong cùng sample, tính IoU trên khoảng `[start, end]` của `position`.
2. IoU ≥ ngưỡng (mặc định 0.5) **và** cùng `type` → coi là 1 cặp match.
3. Overlap vị trí nhưng khác `type` → tính thành 2 khái niệm riêng biệt, cả 2 bên 0 điểm cả 3 metric (đúng luật BTC nêu).
4. Prediction không match ai → false positive. Ground truth không được match → false negative.

### 3.3. WER khi 1 sample có nhiều concept

Tính WER riêng cho từng cặp đã match (so `text` ở mức từ), lấy trung bình các cặp trong sample ra WER(i) đại diện. Cặp không match (FP/FN) → WER = 1.

### 3.4. Code hóa 3 công thức và thuật toán matching
* `text_score`, `assertions_score`, `candidates_score`. 
* **Lưu ý quan trọng về cách biểu diễn Jaccard:** Tránh gom nhãn assertions/candidates thành một tập chung không phân biệt thực thể (dễ bị lỗi gán ngược mã vẫn đạt điểm tối đa). Cần biểu diễn các tập hợp dưới dạng các tuple liên kết vị trí thực thể: `(start, end, label_or_code)`.
* **Giới hạn phạm vi tính Jaccard để tránh điểm ảo:** 
  * Tập assertions của mẫu $i$ chỉ chứa tuple của các thực thể loại `CHẨN_ĐOÁN`, `THUỐC`, `TRIỆU_CHỨNG`.
  * Tập candidates của mẫu $i$ chỉ chứa tuple của các thực thể loại `CHẨN_ĐOÁN`, `THUỐC`.
  * Không tính match rỗng cho các thực thể loại khác nhằm tránh tăng điểm Jaccard giả tạo.
* Xử lý trường hợp biên Jaccard đúng theo đề bài: cả hai rỗng $\rightarrow$ J=1; một bên rỗng $\rightarrow$ J=0; còn lại tính $|Giao| / |Hợp|$.

### 3.5. Kiểm chứng bằng ví dụ mẫu

Chạy `metrics.py` với prediction = ground truth y hệt trên đúng ví dụ mẫu trong đề bài → phải ra 1.0 cho cả 3 thành phần. Đây là điều kiện cần nhưng **chưa đủ** — xem 3.6.

### 3.6. (MỚI) Hiệu chỉnh bằng điểm leaderboard thật & Sweep IoU tự động

Unit test ở 3.5 chỉ xác nhận code không có bug logic hiển nhiên, **không xác nhận giả định của bước 3.2/3.3 khớp với cách BTC thực sự chấm**. Bắt buộc: ngay khi có baseline chạy được (Giai đoạn 0.5), nộp thử 1 lần, lấy điểm leaderboard thật, so với điểm `metrics.py` tự chấm trên đúng output đã nộp.

* **Sweep IoU tự động:** Do quota nộp bài giới hạn (5 lần/ngày), để đối chiếu nhanh và tránh chỉnh tay mất thời gian, tích hợp sẵn chế độ sweep ngưỡng IoU (`0.3`, `0.5`, `0.7`...) trong script `evaluate.py`. Chỉ cần chạy 1 lệnh, script sẽ in ra bảng điểm đối sánh với nhiều ngưỡng khác nhau để tìm ngay ngưỡng IoU trùng khớp với điểm leaderboard thật.
* **Hành động khi lệch:** Nếu lệch nhiều → quay lại xét lại giả định ngưỡng IoU hoặc cách gộp WER đa-concept, sửa càng sớm càng đỡ tốn công tối ưu sai hướng ở các giai đoạn sau.

### 3.7. Deliverable — **[ĐÃ HOÀN THÀNH]**

Các file đã tạo thành công:
* **`metrics.py`** — Engine tự chấm điểm hoàn chỉnh với 5 unit test passed (perfect match, WER > 0, type mismatch, swapped candidates, case-insensitive).
* **`evaluate.py`** — CLI runner: `python src/evaluate.py [--verbose] [--file N] [--gt data/dev/] [--pred data/output/]`.
* **`docs/eval_assumptions.md`** — Tài liệu ghi rõ toàn bộ giả định thuật toán (ngưỡng IoU, cách tính WER đa-concept, Jaccard có trọng số, JSON schema động).
* **`data/dev/`** — Tập dữ liệu validation cục bộ (sẽ được trích xuất từ dữ liệu huấn luyện tự sinh ở Giai đoạn 1, tuyệt đối không sử dụng tệp trong input của BTC để tránh overfit/leakage).
* **`data/output/`** — Thư mục chứa các tệp JSON kết quả dự đoán xuất ra của pipeline (có file `.gitkeep`).

Kết quả test 10 mẫu dev (Average Case): `text=0.9009 | assertions=0.8476 | candidates=0.7742 | final=0.8342`.

**Phát hiện quan trọng về `position`:**
* `position = [start, end]` đếm theo **ký tự Unicode codepoint** (không phải UTF-8 byte).
* `end = start + len(entity_text)` theo Python `str.find()` — dấu cách, `\n`, tab đều được tính.
* File input dùng **LF only** (`\n` = 1 ký tự).
* Chiến lược tìm position trong pipeline: sau khi NER trả về text chuỗi, dùng `text.find(entity_text)` trên văn bản gốc để lấy offset chính xác. Nếu entity text bị thay đổi nhẹ (WER), dùng sliding window fuzzy search.

Bước còn lại (chưa thực hiện được): **Bước 3.6** — đối chiếu điểm tự chấm với leaderboard thật sau khi có Baseline (Giai đoạn 0.5).

---

## 4. Giai đoạn 0.5: Baseline chạy end-to-end — [Người A dựng khung, Người B thiết kế prompt]

Mục tiêu: pipeline `.txt` → `.json` đúng schema, **nộp thử leaderboard ít nhất 1 lần**, dù chất lượng thấp. Đây là điều kiện tiên quyết trước khi train bất kỳ model nào.

### Phương án Baseline: LLM Few-shot NER (thay cho rule-based/regex)

Thay vì dùng regex (chất lượng rất thấp, tốn thời gian viết rules), sử dụng trực tiếp **LLM (Qwen — cùng model đã chọn cho Tầng 3)** làm NER bằng few-shot prompting:

- **NER + Assertion bằng LLM few-shot:** Thiết kế prompt template kèm 2-3 ví dụ mẫu từ đề bài, yêu cầu LLM extract entities + assertions trong **1 lần gọi duy nhất**. Output trực tiếp là JSON đúng schema. Ưu điểm: (a) Chất lượng cao hơn regex đáng kể; (b) Chỉ mất vài giờ build; (c) Tận dụng model sẵn có; (d) **Dùng làm Plan B** nếu NER chuyên dụng (XLM-R) chưa kịp train xong.
- **Retrieval tạm:** BM25 thuần (`bm25s`) trên tên gốc ICD-10 trong SQLite (chưa cần FAISS). Với THUỐC: nếu RxNorm chưa có, trả `candidates: []` — chấp nhận mất điểm THUỐC tạm thời.
- **Reranker tạm:** Top-1 candidate từ BM25, không qua LLM reranker riêng.
- **Đóng gói & đường dẫn:** `main.py` nối chuỗi 3 bước, ghi `output/N.json`. `try-except` toàn pipeline. Áp dụng `src/utils/paths.py` (Single Source of Truth) ngay từ bước này.
- **Dockerfile tối giản:** Chỉ cần chạy được baseline trên máy khác, chứng minh reproducibility sớm.

### Lộ trình nâng cấp dần (Iterative Upgrade Path)

| Phiên bản | NER | Retrieval | Reranker |
|:---|:---|:---|:---|
| v0.5 (Baseline) | LLM few-shot | BM25 only | Top-1 BM25 |
| v1.0 | XLM-R fine-tuned | BM25 + normalize | Top-K RRF |
| v2.0 | + FAISS semantic | + RRF fusion | LLM rerank |
| v3.0 (Final) | + error feedback | + abbreviation dict | + constrained decoding |

Mỗi bước nâng cấp phải chạy `metrics.py` và nộp thử leaderboard để đo cải thiện thật.

**Ý nghĩa kỹ thuật:** Tránh waterfall. Baseline LLM few-shot đồng thời đóng vai trò **Plan B an toàn** — nếu các module chuyên dụng không kịp hoàn thành, vẫn có bài nộp chất lượng chấp nhận được.

---

## 5. Giai đoạn 1: Dữ liệu & CSDL chuẩn

### Phần Người A — Hạ tầng CSDL và tra cứu

**1.1. Thu thập, chuẩn bị và đóng gói CSDL ICD-10 và RxNorm — [ĐÃ HOÀN THÀNH PHẦN ICD-10 & LUẬT LÂM SÀNG]** 
* **`ICD10.xlsx` chính thức của Bộ Y tế** được đặt tại [docs/ICD10.xlsx](ICD10.xlsx) đã được parse sạch sẽ.
* **`src/setup_database.py`**: Script setup duy nhất tự động parse và nạp **25,123 mã ICD-10 song ngữ** cùng toàn bộ hệ luật lâm sàng từ các sheet phụ của file Excel vào SQLite [docs/metadata.db](metadata.db) chỉ bằng 1 dòng lệnh.
* **`src/clinical_validator.py`**: Rule Engine y khoa hoạt động trên RAM giúp tự động kiểm tra chéo và lọc lỗi candidate vi phạm giới tính, độ tuổi, định hướng mã kép và kiểm tra bệnh chính.
* **Context Files cho LLM**: Các file text phẳng [icd10_context.txt](icd10_context.txt) và JSON phẳng [icd10_dictionary.json](icd10_dictionary.json) được sinh tự động giúp LLM làm RAG/Prompt không cần kết nối DB.
* *Phần còn lại ở Giai đoạn 1 (Chốt hạn RxNorm)*: Thu thập CSDL RxNorm chính thức (tiếng Anh) từ UMLS — **đang chờ phê duyệt tài khoản UMLS (tính đến 15/07)**. 
  * **Hạn chót trigger date cứng:** Nếu đến **hết ngày 17/07** tài khoản UMLS chưa được duyệt, chuyển ngay lập tức sang phương án fallback: Tải thông tin mã thuốc trực tiếp từ [RxNav REST API](https://rxnav.nlm.nih.gov/REST) (không yêu cầu tài khoản) kết hợp crawl DrugBank/OpenFDA nguồn mở để tự xây dựng SQLite mapping hoạt chất Việt-Anh. Tuyệt đối không để rủi ro này trôi tự do sau ngày 17/07.
  * Khi có dữ liệu, parse bảng RXNCONSO (chứa mapping RXCUI ↔ tên thuốc) vào SQLite, sinh `rxnorm_context.txt` và `rxnorm_dictionary.json` tương tự quy trình ICD-10.

**1.2. Lexical Index (Tra cứu theo từ khóa) — Sử dụng thư viện `bm25s`** (Cài đặt: `pip install "bm25s[core]"`):
* **Mục tiêu:** Tìm kiếm và khớp các mã bệnh/mã thuốc dựa trên sự trùng lặp chính xác từng chữ cái (ví dụ: khớp biệt dược *"Amlodipine 5mg"* và *"Amlodipine 10mg"*).
* **Lý do chọn công nghệ:** Chọn trực tiếp `bm25s` ngay từ đầu (không sử dụng thư viện `rank_bm25` chậm chạp chạy bằng Python thuần). `bm25s` được viết tối ưu bằng ngôn ngữ C thông qua thư viện ma trận thưa Scipy, giúp tăng tốc độ tìm kiếm nhanh hơn gấp hàng trăm lần, đạt hiệu năng ngang ngửa Elasticsearch nhưng hoàn toàn gọn nhẹ, chạy offline mà không cần cài đặt Java hay dựng server cồng kềnh.

**1.3. Semantic Index (Tra cứu theo ý nghĩa) — Sử dụng thư viện `FAISS` với cấu hình `IndexFlatIP`**:
* **Mục tiêu:** Khớp mã bệnh/mã thuốc khi bác sĩ viết tắt, viết lệch từ hoặc viết bằng tiếng Anh (ví dụ: khớp từ *"THA"*, *"áp huyết cao"* về danh mục chuẩn *"Tăng huyết áp"*). Mô hình AI (`bge-m3`) của Người B sẽ chuyển các từ này thành các chuỗi số (vector embedding) đại diện cho ý nghĩa của chúng, sau đó so khớp độ tương đồng ngữ nghĩa.
* **Lý do chọn cấu hình `IndexFlatIP` (Brute-force):** Cấu hình `IndexFlatIP` thực hiện tính toán so sánh trực tiếp vector đầu vào với toàn bộ kho lưu trữ mà không sử dụng thuật toán phân cụm xấp xỉ (như IVF hay HNSW). Vì quy mô dữ liệu của cuộc thi (khoảng 70.000 mã ICD-10 + vài trăm nghìn mã RxNorm) đủ nhỏ để việc so sánh brute-force vẫn diễn ra cực kỳ nhanh (dưới 2 mili-giây/truy vấn), chúng ta sử dụng `IndexFlatIP` để đạt độ chính xác tuyệt đối 100% về mặt toán học mà không phải đánh đổi độ chính xác lấy tốc độ.

**1.4. Đảm bảo hoạt động offline (Air-gapped):** Thiết lập kiểm thử ngắt kết nối mạng hoàn toàn. Đảm bảo toàn bộ thư viện, mô hình và dữ liệu đã được lưu trữ cục bộ, không thực hiện bất kỳ lệnh ngầm nào gọi ra ngoài (bao gồm việc tự động tải model từ HuggingFace Hub khi runtime) — tất cả model weights và index phải được cache hoặc build sẵn vào Docker image.

### Phần Người B — Chất lượng dữ liệu huấn luyện

**1.5. Sinh dữ liệu huấn luyện bằng phương pháp sinh có căn cứ (Grounded Generation):** Chọn trước mã y khoa ICD-10/RxNorm chuẩn từ CSDL, sau đó dùng LLM viết câu bệnh án chứa khái niệm đó với các biến thể tự nhiên (viết tắt, lỗi chính tả nhẹ, từ ngữ địa phương). Phương pháp này giúp nhãn dữ liệu chuẩn xác 100% theo thiết kế mà không gặp lỗi ảo tưởng (hallucination) của LLM.

**1.6. Mô phỏng đa dạng văn phong & cấu trúc dữ liệu.** Để khớp với phổ dữ liệu thật đa dạng của đề bài (EHR, giấy xuất viện, ghi chú bác sĩ có cấu trúc liệt kê):
* Sinh synthetic data theo 3 định dạng cấu trúc:
  1. *Câu văn xuôi tự do:* Sử dụng câu thuần Việt hoặc câu Việt xen block thuốc tiếng Anh.
  2. *Dạng danh sách liệt kê đánh số:* Mô phỏng giấy xuất viện/toa thuốc (ví dụ: "1. amlodipine 5mg po daily; 2. metformin 500mg...").
  3. *Dạng nhãn khóa-giá trị (Key-Value):* Mô phỏng phần tiền sử/khám bệnh của EHR (ví dụ: "Tiền sử: Tăng huyết áp 5 năm, ĐTĐ typ 2. Thuốc đang dùng: Metformin.").
* **Ý nghĩa:** Tránh việc mô hình NER bị overfit vào câu văn liền mạch và trích xuất sót khi gặp cấu trúc liệt kê dạng bảng/dòng riêng biệt trong tập test thật.

**1.7. Sinh embedding cho CSDL.** Chạy `bge-m3` trên toàn bộ tên bệnh/thuốc, giao vector cho Người A build FAISS. **Nâng cấp tùy chọn:** Nếu recall FAISS trên tập chốt (1.9) dưới 80%, thay thế hoặc kết hợp bằng `SapBERT-XLMR` — mô hình pre-trained trên UMLS synonym pairs, chuyên biệt cho biomedical entity linking xuyên ngôn ngữ, đạt SOTA trên benchmark BioNNE-L 2025.

**1.8. Chuyển sang BIO format** kèm nhãn assertion — input trực tiếp để train NER ở Giai đoạn 2.

**1.9. Tạo tập dữ liệu kiểm thử chốt (Tập chốt thủ công):** Biên soạn thủ công 20-30 mẫu bệnh án mô phỏng sát nhất có thể văn phong lâm sàng thực tế của đề bài (không sử dụng dữ liệu sinh bằng LLM). Tập dữ liệu này đóng vai trò như một "thước đo thực tế" để kiểm chứng xem mô hình có bị overfit vào cách hành văn của LLM sinh dữ liệu ở bước 1.5 hay không, dùng làm căn cứ chốt các siêu tham số (ngưỡng lọc ε, trọng số RRF) trước khi chạy thật.

**Luồng phụ thuộc:** CSDL gốc (A) → B lấy mã thật sinh câu synthetic có grounded label → B tính embedding → A build FAISS từ embedding đó → cả 2 sẵn sàng cho Giai đoạn 2/3.

---

## 6. Giai đoạn 2: Tầng 1 — Extractor (NER + Assertion)

Chiếm 60% trọng số điểm (30% text_score + 30% assertions_score) — ảnh hưởng trực tiếp và ngay lập tức, không qua trung gian như tầng candidate.

### Phần Người B — Huấn luyện mô hình

**2.1. Backbone đã chốt: XLM-RoBERTa-large + CRF (không A/B test — tiết kiệm thời gian).** Lý do chọn trực tiếp XLM-R: (a) Dữ liệu thực tế cuộc thi là văn bản lâm sàng song ngữ Việt-Anh trộn lẫn (tên thuốc tiếng Anh xen kẽ ghi chú Việt) → XLM-R xử lý tốt hơn; (b) Không cần word segmentation (loại bỏ dependency vào RDRSegmenter, giảm bug offset-mapping); (c) Nghiên cứu SOTA (như benchmark VietMed-NER công bố tại NAACL 2025 Industry Track) xác nhận các pre-trained multilingual encoder nói chung vượt trội hơn monolingual encoder trên tác vụ NER lâm sàng tiếng Việt. **Ghi chú 9B:** XLM-R large (~560M) rất nhỏ so với trần 9B/model.

**2.2. Fine-tune token classifier** trên BIO data (1.8), 5 nhãn: `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`, `CHẨN_ĐOÁN`, `THUỐC`.

**2.3. Theo dõi confusion matrix CHẨN_ĐOÁN ↔ TRIỆU_CHỨNG.** Đề phạt rất nặng lỗi này (tính 2 lần, 0 điểm cả 3 metric). Tune trọng số loss dựa trên số liệu quan sát thực tế trên tập dev trong suốt quá trình train, không áp hệ số cố định từ đầu.

**2.4. Assertion classifier phụ trợ** cho câu phức không khớp rule (2.8 của Người A) — train trên dữ liệu synthetic có nhãn assertion.

**2.5. Quy trình tối ưu hóa dữ liệu đóng (Feedback Loop):** Sau mỗi lượt đánh giá hiệu năng trên tập dev/tập chốt, tiến hành phân tích chi tiết ma trận nhầm lẫn (Confusion Matrix). Thay vì chỉ ghi nhận lỗi, quy trình bắt buộc phải quay lại bước 1.5 để sinh thêm dữ liệu synthetic nhắm trúng vào loại lỗi đang mắc phải nhiều nhất (ví dụ: nếu mô hình hay nhầm lẫn giữa Triệu chứng và Chẩn đoán trong các câu liệt kê dài, sinh thêm nhiều mẫu dạng liệt kê để huấn luyện lại).

**2.6. Deliverable giao cho Người A.** Model NER export (weights + script inference, hoặc ONNX), input raw text, output list `(vị trí bắt đầu, vị trí kết thúc, type)` theo định dạng thống nhất trước.

### Phần Người A — Kỹ thuật xử lý xung quanh model

**2.7. Đồng bộ vị trí ký tự (Offset-mapping) bằng Fast Tokenizer của XLM-R:**
* **Quyết định chốt:** Sử dụng trực tiếp mảng offset có sẵn từ tokenizer của HuggingFace thông qua thuộc tính `return_offsets_mapping=True` khi gọi model XLM-R.
* **Lý do kỹ thuật tối ưu:** 
  1. *Loại bỏ hoàn toàn Word Segmentation:* PhoBERT đòi hỏi phải chạy bộ tách từ (như RDRSegmenter) trước khi đưa vào tokenizer. Việc này tạo ra một "hộp đen" dịch chuyển chỉ số (index shift) cực kỳ dễ lỗi. Khi văn bản gốc có khoảng trắng kép, xuống dòng `\n`, tab `\t`, hoặc dấu câu dính liền từ, bộ tách từ sẽ thay đổi cấu trúc chuỗi, khiến việc ánh xạ ngược lại vị trí ký tự gốc để tính `position = [start, end]` (Unicode codepoint) rất dễ bị lệch 1-2 ký tự (gây mất điểm IoU).
  2. *Độ chính xác tuyệt đối từ HuggingFace:* XLM-R Tokenizer (cơ chế BPE trên ký tự thô) cho phép map trực tiếp từng subtoken về khoảng index ký tự `(start, end)` trên đúng chuỗi văn bản gốc ban đầu mà không thay đổi bất kỳ ký tự nào, kể cả khoảng trắng hay dấu xuống dòng. Điều này loại bỏ hoàn toàn mã nguồn ánh xạ thủ công phức tạp và các unit test biên liên quan.

**2.8. Phân tích thuộc tính theo quy luật và phạm vi (Rule/Scope-based Assertion):** Áp dụng thuật toán kiểu ConText/NegEx để phát hiện thuộc tính của thực thể:
* Từ khóa kích hoạt (Trigger Terms): "không", "chưa phát hiện", "chưa thấy" $\rightarrow$ `isNegated`; "tiền sử", "trước đây", "tiền sử gia đình có", "người nhà có tiền sử" $\rightarrow$ `isHistorical` hoặc `isFamily` tương ứng; "bố bị", "mẹ bị", "gia đình" $\rightarrow$ `isFamily`.
* Định vị phạm vi (Scope): Phạm vi ảnh hưởng của từ khóa được kéo dài từ vị trí từ khóa đến hết câu, hoặc kết thúc sớm khi gặp dấu câu (dấu phẩy, dấu chấm) hoặc các liên từ đảo hướng (như "nhưng", "tuy nhiên", "mặc dù") chứ không sử dụng khoảng cách số từ cố định.

**2.9. Kết hợp rule + classifier phụ trợ.** Rule chạy trước bắt case rõ ràng, case không khớp đẩy qua classifier của Người B.

**2.10. Đóng gói module Tầng 1** với interface duy nhất: nhận `text` thô, trả `{text, position, type, assertions}` — input trực tiếp cho Tầng 2.

---

## 7. Giai đoạn 3: Tầng 2 — Retrieval — [Người A toàn bộ]

Thuần hạ tầng tìm kiếm. Input: khái niệm `CHẨN_ĐOÁN`/`THUỐC` đã có `text`, `position`, `type` từ Tầng 1. Output: candidate list ICD/RxNorm cho Tầng 3.

**3.1. Text Normalizer, Fuzzy Matcher & Bảng viết tắt y khoa.**
* Bóc phần liều/tần suất khỏi tên thuốc trước khi tra mã (RxNorm định danh theo hoạt chất + hàm lượng, không theo tần suất uống) — regex nhận diện pattern phổ biến (po, bid, qid, daily, prn, q6h, qam, qhs...). Với `CHẨN_ĐOÁN`: chuẩn hóa khoảng trắng, dấu câu thừa.
* **Giải pháp chuẩn hóa từ vựng:** Tích hợp thuật toán so khớp mờ (`rapidfuzz`) để chuẩn hóa nhanh các tên chẩn đoán/tên thuốc bị viết tắt hoặc sai chính tả nhẹ về danh mục chuẩn trong SQLite trước khi tiến hành tính vector embedding hay tra cứu BM25.
* **(MỚI) Bảng viết tắt y khoa Việt Nam:** Xây dựng dictionary ánh xạ viết tắt → tên đầy đủ (VD: THA → Tăng huyết áp, ĐTĐ → Đái tháo đường, TBMN → Tai biến mạch não, COPD → Bệnh phổi tắc nghẽn mạn tính...). Áp dụng query expansion: truy vấn cả tên viết tắt lẫn tên đầy đủ để tăng recall. Dictionary trích xuất từ: (a) danh mục ICD-10 song ngữ đã có; (b) danh mục từ viết tắt lâm sàng chuẩn hóa công bố bởi các bệnh viện lớn (như Bệnh viện Bạch Mai, Bệnh viện Chợ Rẫy); (c) bổ sung thủ công các viết tắt lâm sàng thông dụng Việt Nam. *Không thu thập trực tiếp từ 100 file input của BTC nhằm bảo đảm tính khách quan tuyệt đối.*
* **(MỚI) Rule matching cứng cho các thực thể y khoa tần suất cao từ nguồn độc lập:** Để tăng tối đa `candidates_score` (chiếm 40% điểm) và khắc phục hạn chế của embedding đối với biệt dược/thuốc có alias phức tạp, xây dựng bảng tra cứu cứng (lookup dictionary) cho khoảng 50-100 hoạt chất thuốc/tên bệnh phổ biến nhất tại Việt Nam. Nguồn dữ liệu này được trích xuất hoàn toàn độc lập từ danh mục thuốc của Bộ Y Tế, cơ sở dữ liệu DrugBank Việt Nam, và CSDL OpenFDA công khai.
* **Luật gán đè (Override) hợp lệ:** Nếu thực thể do mô hình NER trích xuất trùng khớp chính xác với key trong bảng tra cứu độc lập này → hệ thống tự động gán mã hóa cứng tương ứng làm candidate top-1, bỏ qua so khớp vector. Cách tiếp cận này hoàn toàn khách quan, có khả năng áp dụng trực tiếp lên **private test set** và vượt qua khâu audit code của BTC một cách an toàn.

**3.2. Tìm kiếm kết hợp (Hybrid Retrieval) bằng thuật toán RRF:** Truy vấn song song bằng `bm25s` và `FAISS` (IndexFlatIP), sau đó hợp nhất kết quả bằng công thức Reciprocal Rank Fusion (RRF):

$$\text{Score}(d) = \frac{W_{BM25}}{\text{Rank}_{BM25}(d) + k} + \frac{W_{FAISS}}{\text{Rank}_{FAISS}(d) + k}$$

Thiết lập trọng số của BM25 (`W_BM25`) lớn hơn đáng kể so với FAISS (`W_FAISS`). Lý do: sai lệch một ký tự hoặc một con số ở tên hoạt chất/hàm lượng thuốc (ví dụ "Amlodipine 5mg" vs "10mg") sẽ dẫn tới sai mã hoàn toàn về mặt lâm sàng, và BM25 bắt chính xác các lỗi này tốt hơn FAISS. FAISS đóng vai trò bổ trợ (fallback) khi phương pháp so khớp từ khóa chính xác BM25 không tìm thấy kết quả. Tỷ lệ trọng số cụ thể được tinh chỉnh bằng thực nghiệm trên tập kiểm thử chốt (1.9).

**3.3. Cơ chế giới hạn candidate động:** Tránh cấu hình cứng số lượng ứng viên (ví dụ lấy cố định Top 5) vì số lượng mã chính xác của mỗi thực thể là khác nhau (có những thực thể chỉ có 1 mã chuẩn duy nhất, nhưng có thực thể cần 2-3 mã song hành). Lập trình thuật toán lọc động: giữ lại mọi candidate có điểm RRF nằm trong khoảng sai số X% so với ứng viên đứng đầu bảng xếp hạng. Ngưỡng X% này sẽ do Người B tinh chỉnh ở Giai đoạn 4 cùng cơ chế tự tin (confidence score) của LLM.

**Deliverable:** module nhận khái niệm đã normalize, trả candidate list kèm điểm RRF đã xếp hạng — input cho Tầng 3.

---

## 8. Giai đoạn 4: Tầng 3 — Reranker (LLM ≤ 9B/model)

Quyết định cuối cùng mã nào vào output — chiếm 40% điểm (`candidates_score`), phần nặng nhất công thức.

### Phần Người B

**4.1. Lựa chọn và lượng hóa mô hình (Quantization):** Tiến hành thực nghiệm so sánh (A/B Test) giữa `Qwen2.5-7B-Instruct` và `Qwen3-8B` trên tập kiểm thử chốt để đánh giá độ chính xác suy luận lâm sàng. Thực hiện lượng hóa mô hình sang định dạng AWQ hoặc GGUF và triển khai trên công cụ vLLM (để tối ưu hóa tốc độ xử lý hàng loạt - batching) hoặc llama.cpp (nếu bị giới hạn phần cứng máy chấm). Thực hiện đo đạc (benchmark) tốc độ xử lý thực tế trước khi lựa chọn công cụ chính thức.
* **Phương án triển khai offline (đã đơn giản hóa):** Ưu tiên vLLM (batching + structured decoding native) hoặc llama.cpp (VRAM hạn chế). Lượng hóa AWQ hoặc GGUF Q4/Q5 đã đủ tốt — **không dùng CTranslate2** (thêm dependency phức tạp, lợi ích không đáng kể so với vLLM/llama.cpp).

**4.2. Ép JSON schema & Constrained Decoding (chốt chặn kỹ thuật chính).**
* **Constrained decoding (bắt buộc):** Cấu hình giải mã ràng buộc ở tầng inference: vLLM hỗ trợ tham số `structured_outputs` nhận JSON schema (sử dụng thư viện XGrammar để JIT-compile grammar) đảm bảo đầu ra luôn tuân thủ đúng định dạng JSON. Quan trọng nhất: cấu hình thuộc tính `candidates` trong schema chỉ được phép chọn các giá trị nằm trong danh sách mã y khoa mà Tầng 2 (Retrieval) trả về (thông qua cơ chế `enum` động). Việc này loại bỏ hoàn toàn khả năng LLM tự sinh (bịa) mã không tồn tại.
* **Overhead Benchmark của Enum động (Mới):** XGrammar JIT-compilation compile schema mỗi khi enum thay đổi. Vì mỗi thực thể y khoa có một danh sách candidate (enum) khác nhau, quá trình JIT compile sẽ diễn ra liên tục trên từng request và không dùng được cache tĩnh. Bắt buộc phải thực hiện đo đạc (benchmark) latency của cơ chế **dynamic enum** này trong giai đoạn tích hợp, nhằm tránh trường hợp nghẽn cổ chai tốc độ xử lý do biên dịch schema liên tục.
* **QLoRA fine-tune (tùy chọn — chỉ khi còn ≥ 5 ngày trước deadline):** Ưu tiên zero-shot/few-shot prompting trước. QLoRA chỉ thực hiện nếu: (a) pipeline đã chạy ổn định end-to-end; (b) có đủ training pairs (entity text → mã chuẩn); (c) benchmark cho thấy cải thiện đáng kể. Nếu không, giữ nguyên few-shot — vẫn hoạt động tốt nhờ constrained decoding.
* **Xác thực Grounding (đã đơn giản hóa):** ~~NLI verifier (mDeBERTa-XNLI)~~ → **Loại bỏ** — thêm 1 model dependency, tăng VRAM, lợi ích marginal khi đã có constrained decoding. Dựa hoàn toàn vào: (a) Constrained decoding ép cứng chọn mã từ enum Tầng 2; (b) Logprobs threshold lọc candidate xác suất thấp. Hai cơ chế kết hợp đã đủ mạnh.

**4.3. Cơ chế đánh giá độ tin cậy (Confidence Score):** Nhằm tránh việc LLM tự đánh giá cảm tính, hệ thống kết hợp hai tín hiệu khách quan:
* **Logprobs từ LLM:** Kích hoạt tham số `logprobs=True` trên vLLM. Lấy trung bình cộng logprob của toàn bộ chuỗi token tạo nên mã bệnh/mã thuốc, tránh sai lệch khi chỉ tính xác suất của token đầu tiên.
* **Điểm RRF từ Tầng 2 (Retrieval):** Đây là tín hiệu tham chiếu ổn định, ít bị ảnh hưởng bởi sự sai lệch hiệu chuẩn (calibration drift) thường gặp ở các LLM cỡ nhỏ sau quá trình huấn luyện căn chỉnh (RLHF/Instruction Tuning).
Độ tự tin cuối cùng để chốt candidate được tính toán bằng tổ hợp tuyến tính có trọng số từ cả hai nguồn tín hiệu trên.

**4.4. Calibrate ngưỡng ε bằng đo lường.** Chạy nhiều giá trị ε trên tập chốt (1.9), đo `candidates_score` bằng `metrics.py`, chọn ε điểm cao nhất thực đo — không suy luận lý thuyết.

**4.5. Đo và báo cáo tài nguyên thực tế.** VRAM tiêu thụ, thời gian inference trung bình/file — giao cho Người A đưa vào README (Giai đoạn 5).

### Phần Người A — Interface và fallback

**4.6. Định dạng dữ liệu vào prompt.** Output Tầng 2 (candidate list kèm điểm RRF) đưa vào prompt Tầng 3 đúng cấu trúc Người B cần — danh sách candidate kèm tên chuẩn + điểm số, sắp theo thứ hạng.

**4.7. Fallback khi lỗi.** Dù đã có structured decoding ở 4.2 (giảm mạnh rủi ro lỗi parse), vẫn giữ `try-except` bọc lời gọi LLM: nếu lỗi (timeout, OOM, exception bất kỳ), trả về đúng schema với `candidates: []`, giữ nguyên `assertions` từ Tầng 1 — không mất điểm `text_score`/`assertions_score` đã có sẵn, chỉ mất phần `candidates_score` của riêng khái niệm đó.

**Deliverable:** module nhận khái niệm (`CHẨN_ĐOÁN`/`THUỐC`) + candidate list từ Tầng 2, trả danh sách mã cuối cùng đúng schema — sẵn sàng cho Giai đoạn 5.

---

## 9. Giai đoạn 5: Tích hợp, Kiểm thử, Đảm bảo tái lập

Đây là giai đoạn đã được nâng thành **yêu cầu hạng nhất từ đầu** (xem mục 0.3), không phải việc dồn cuối. Nhờ đã có khung Docker từ Giai đoạn 0.5 và test liên tục mỗi khi thêm module, khối lượng còn lại ở đây chủ yếu là hoàn thiện, không phải làm từ số 0.

### Phần Người A

**5.1. Đóng gói `main.py` hoàn chỉnh và áp dụng Schema động.** 
* Đọc `.txt` → Tầng 1 (Giai đoạn 2) → Text Normalizer + Hybrid Retrieval (Giai đoạn 3) → LLM Reranker (Giai đoạn 4) → ghi `output/N.json`. Vì mỗi module do 2 người phát triển riêng, kiểm tra kỹ ranh giới giữa module (kiểu số nguyên vs chuỗi, thứ tự field trong dict) — lỗi tích hợp kiểu này rất phổ biến.
* **Ràng buộc định dạng JSON đầu ra theo từng loại thực thể (Bắt buộc):** Trước khi xuất file JSON, module đóng gói phải loại bỏ các trường không hợp lệ đối với từng loại thực thể để tránh lỗi schema của BTC:
  * `TRIỆU_CHỨNG`: Chỉ có 4 trường (`text`, `type`, `position`, `assertions`). *Không chứa key `candidates`*.
  * `TÊN_XÉT_NGHIỆM`, `KẾT_QUẢ_XÉT_NGHIỆM`: Chỉ có 3 trường (`text`, `type`, `position`). *Không chứa key `assertions` và `candidates`*.
  * `CHẨN_ĐOÁN`, `THUỐC`: Đầy đủ cả 5 trường.

**5.2. Tích hợp bộ kiểm duyệt lâm sàng (Clinical Validation Engine) — [MỚI]**
* **Nhiệm vụ:** Đưa file [clinical_validator.py](../src/clinical_validator.py) vào khâu Hậu xử lý (Post-processing) của pipeline trước khi xuất JSON.
* **Thiết kế Modular (Cắm/Rút & Fail-safe):**
  * **Tách biệt module trích xuất:** Tạo class `PatientExtractor` (trong `src/patient_extractor.py` hoặc tích hợp riêng biệt) chịu trách nhiệm trích xuất thông tin hành chính thô.
  * **Chiến lược giảm nhiễu (Anti-noise):**
    * Chỉ quét tìm thông tin Giới tính (Nam/Nữ) và Tuổi (quy đổi sang ngày) trong **150 ký tự đầu tiên** (hoặc dòng đầu tiên) của file `.txt`.
    * Tự động bỏ qua (hủy kết quả) nếu phát hiện các từ khóa nhiễu liên quan đến người thân (bố, mẹ, vợ, chồng, con...) hoặc nhân viên y tế (bác sĩ, điều dưỡng) trong vùng quét.
  * **Cơ chế Fallback an toàn:** Nếu không tìm thấy thông tin hoặc trích xuất thất bại, trả về `None`. Khi đó, validator sẽ giữ nguyên mọi candidates gốc, đảm bảo không lọc nhầm khi thiếu thông tin.
  * **Thiết kế Plug-and-Play ở Pipeline chính:**
    * Sử dụng cờ cấu hình `USE_CLINICAL_VALIDATOR = True/False` ở file cấu hình để bật/tắt tức thì mà không ảnh hưởng logic chính.
    * Toàn bộ khâu Hậu xử lý y khoa được bọc trong khối `try-except` tổng để đảm bảo lỗi ở validator không bao giờ gây crash pipeline chính, tự động fallback về kết quả NER thô khi có sự cố.

**5.3. Container hóa — "hệ thống kép" và cấu hình đường dẫn động.**
* Không hard-code đường dẫn dữ liệu dạng `test/input/`. Do thực tế cấu trúc thư mục giải nén có thể là `input/input/`, code đọc dữ liệu phải tự động quét hoặc nhận tham số đầu vào động.
* `Dockerfile` chuẩn chỉnh — phương án chính, đóng gói pipeline + weights + 3 file index vào 1 image, chỉ cần `docker run`.
* `setup_scripts/install_linux.sh` + `install_windows.bat` + `environment.yml` — phương án dự phòng chạy native, dùng khi máy chấm (có thể Windows không cài NVIDIA Container Toolkit) không pass-through GPU vào Docker được, tránh việc Qwen chạy CPU-only timeout khi xử lý 100 file.

README ghi rõ: nếu chạy Docker không thấy GPU được nhận diện, chuyển sang script native tương ứng hệ điều hành.

**5.4. `check_env.py`** chạy đầu tiên ở cả 2 đường, kiểm tra GPU/CUDA/VRAM khả dụng, báo lỗi rõ ràng ngay từ đầu thay vì để pipeline chạy chậm âm thầm.

**5.5. Đo thời gian xử lý thực tế trên 100 file**, cả 2 kịch bản kiến trúc VRAM (load 1 lần giữ suốt runtime vs load-unload theo tầng) — quyết định kiến trúc cuối dựa trên số đo thật, ưu tiên phương án an toàn cho cấu hình máy yếu hơn nếu không chắc máy BTC dùng để chấm.

**5.6. README hoàn chỉnh:** yêu cầu phần cứng tối thiểu (VRAM thực đo từ 4.5), thời gian xử lý ước tính 100 file, hướng dẫn cài đặt cả 2 đường + khi nào dùng đường nào, giả định chấm điểm đã document ở Giai đoạn 0 (thuật toán matching, cách tính WER đa-concept, kết quả đối chiếu leaderboard ở 3.6) — để hiểu logic hệ thống khi đối chiếu kết quả.

**5.7. Fallback toàn pipeline.** `try-except` quanh toàn bộ, không chỉ Tầng 3. Lỗi ở bất kỳ tầng nào với 1 file → ghi `[]` hợp lệ cho file đó, tiếp tục xử lý các file còn lại, không dừng toàn bộ.

### Phần Người B

**5.8. Cung cấp số liệu tài nguyên chính xác** (đã nêu 4.5) cho Người A tổng hợp README.

**5.9. Kiểm tra chất lượng sau tích hợp toàn pipeline.** Chất lượng test rời từng module không đảm bảo chất lượng khi chạy pipeline thật (quantization ảnh hưởng khác khi chạy liên tục hàng trăm request so với dev thử vài câu; lỗi offset-mapping nhỏ ở Tầng 1 có thể làm Tầng 2 truy vấn sai mà test riêng Tầng 2 không phát hiện). Chạy `metrics.py` trên **output cuối cùng của pipeline đã đóng gói**, không phải bản dev rời rạc, trước khi nộp chính thức.

---

## 10. Bảng tổng hợp phân công

|Giai đoạn|Người A (Backend/Hệ thống)|Người B (LLM/Training)|
|---|---|---|
|0. Eval Engine|Toàn bộ + đối chiếu leaderboard|—|
|0.5. Baseline LLM few-shot|Dựng khung + BM25 + Docker|Thiết kế prompt template NER|
|1. CSDL & Dữ liệu|SQLite + RxNorm, bm25s/FAISS hạ tầng|Synthetic data, embedding (bge-m3/SapBERT), BIO format, tập chốt|
|2. Extractor|Offset-mapping, rule assertion, đóng gói module|Train NER (XLM-R + CRF), assertion classifier, feedback loop|
|3. Retrieval|Toàn bộ (Normalizer, abbreviation dict, Hybrid RRF, bm25s, FAISS)|—|
|4. Reranker LLM|Interface/fallback|Qwen 7-8B, constrained decoding, few-shot (QLoRA tùy chọn)|
|5. Tích hợp & Deploy|Toàn bộ (Docker kép, script, README)|Kiểm tra chất lượng tích hợp|

**Điểm đồng bộ bắt buộc (họp chốt trước khi làm song song):**

1. Sau Giai đoạn 0: thống nhất `metrics.py` + kết quả đối chiếu leaderboard (3.6) là chuẩn chung.
2. Trước Giai đoạn 2: chốt định dạng output model NER.
3. Trước Giai đoạn 4: chốt định dạng candidate list (kèm điểm RRF) từ Tầng 2 sang Tầng 3.
4. Mỗi khi thêm module mới vào `main.py`: build lại Docker + chạy thử trên máy của người còn lại trong vòng 24h (nguyên tắc containerize sớm, mục 1).

---

## 10.5. Lộ trình thực hiện chi tiết (15/07 → 30/07)

### Tuần 3 — Xây nền (15/07 → 21/07)

| Ngày | Người A | Người B |
|:---:|:---|:---|
| 15-16/07 | Baseline LLM few-shot NER + BM25 → nộp thử leaderboard | Thiết kế prompt template NER + bắt đầu sinh synthetic data |
| 17-18/07 | **Hạn chót trigger date:** UMLS duyệt hoặc chuyển ngay sang RxNav API. Xây dựng RxNorm DB + fix `metrics.py` theo leaderboard | Sinh synthetic data BIO (cả 3 văn phong) + bắt đầu train XLM-R NER |
| 19-21/07 | Build FAISS index + Hybrid Retrieval (BM25+FAISS+RRF) | Fine-tune XLM-R NER + sinh embedding bge-m3 |

### Tuần 4 — Tối ưu & Tích hợp (22/07 → 28/07)

| Ngày | Người A | Người B |
|:---:|:---|:---|
| 22-23/07 | Tích hợp XLM-R NER vào pipeline thay LLM baseline | Setup LLM reranker (Qwen) + constrained decoding |
| 24-25/07 | Tích hợp LLM reranker + ClinicalValidator vào pipeline | Tune hyperparams (ngưỡng ε, trọng số RRF) trên tập chốt |
| 26-27/07 | Docker hoàn chỉnh + script native + README | Kiểm tra chất lượng + QLoRA nếu còn thời gian |
| 28/07 | Kiểm thử Docker trên máy sạch | Chạy metrics.py trên output cuối |

### Ngày chốt (29-30/07)

- **29/07:** Nộp thử 2-3 lần, so điểm, fix bug nếu có.
- **30/07 sáng:** Nộp chính thức trước 12:00, giữ ít nhất 1 lượt dự phòng.

---

## 10.6. Plan B — Kịch bản tối thiểu khả thi (nếu hết thời gian)

Nếu đến ngày 26/07 mà NER chuyên dụng hoặc LLM reranker chưa ổn định, chuyển sang Plan B:

| Thành phần | Phương án Plan B | Ước tính điểm |
|:---|:---|:---:|
| NER | LLM few-shot (Qwen) — đã có từ baseline | text_score ~0.6-0.7 |
| Assertion | LLM extract trong cùng prompt NER | assertions_score ~0.5-0.6 |
| Retrieval | BM25 thuần trên ICD-10 + RxNorm | — |
| Reranker | Top-1 BM25 (không LLM reranker) | candidates_score ~0.3-0.4 |
| **Ước tính tổng** | | **~0.4-0.5** |

Plan B đảm bảo có bài nộp chạy được, ước tính lọt top 50% bảng xếp hạng (điểm top 10 hiện tại ~51-53). Mọi module chuyên dụng hoàn thành kịp sẽ được swap vào thay thế từng phần tương ứng.

---

## 10.7. Mẫu nhật ký thực nghiệm (Experiment Tracking Table)

Nhật ký chính thức được lưu tại [docs/experiment_log.md](experiment_log.md). Mỗi lượt nộp bài phải cập nhật vào bảng:

| Version | NER Model | Retrieval | Rerank / Thresh | Điểm tự chấm (`metrics.py`) | Điểm Leaderboard thật | Ghi chú / Cải tiến |
|---|---|---|---|---|---|---|
| v0.5 | Few-shot Qwen | BM25s | Top-1 BM25 | Text: ... Ass: ... Cand: ... | ... | Lượt nộp baseline đầu tiên để test offset, rate limit |
| ... | ... | ... | ... | ... | ... | ... |

---

## 11. Checklist trước khi nộp

- [x] Tái cấu trúc thư mục dự án theo mô hình modular (phân chia rõ ràng data/ và src/).
- [x] Xây dựng tệp quản lý đường dẫn tập trung [paths.py](../src/utils/paths.py) phục vụ Dockerize.
- [x] Thiết lập CSDL SQLite y khoa ICD-10 song ngữ và nạp luật lâm sàng từ Excel ([setup_db.py](../src/utils/setup_db.py)).
- [x] Phát triển bộ luật kiểm duyệt lâm sàng y khoa trên bộ nhớ RAM ([clinical_validator.py](../src/validation/clinical_validator.py)) để lọc lỗi.
- [x] Xây dựng công thức tính điểm tự chấm [metrics.py](../src/metrics.py) vượt qua 5 unit test logic.
- [x] Phát triển công cụ CLI chạy đánh giá và chấm điểm cục bộ [evaluate.py](../src/evaluate.py).
- [x] Tài liệu hóa các giả định tính điểm và logic ghép cặp [eval_assumptions.md](eval_assumptions.md).
- [ ] Baseline khung xương đã nộp thử ít nhất 1 lần, đối chiếu điểm `metrics.py` với điểm leaderboard thật, chênh lệch đã được giải thích/xử lý.
- [ ] Docker image chạy độc lập, không cần mạng, build thành công từ đầu trên máy sạch.
- [ ] Script native (`.sh` và `.bat`) chạy được trên máy khác (không phải máy dev).
- [ ] `check_env.py` báo lỗi rõ ràng khi thiếu GPU/CUDA.
- [ ] Toàn bộ 100 file output đúng schema, kể cả khi cố tình gây lỗi vài file để test fallback.
- [ ] Structured decoding đã xác nhận: không có candidate nào ngoài danh sách Tầng 2 trả về xuất hiện trong output.
- [ ] README đầy đủ: phần cứng yêu cầu, thời gian xử lý, giả định eval, hướng dẫn 2 đường cài đặt.
- [ ] `metrics.py` chạy trên output cuối cùng của pipeline đã đóng gói, không phải bản dev rời rạc.
- [ ] Nộp chính thức trước hạn (30/07) ít nhất nửa ngày, còn dư ít nhất 1 lượt nộp trong ngày để dự phòng lỗi phút chót.