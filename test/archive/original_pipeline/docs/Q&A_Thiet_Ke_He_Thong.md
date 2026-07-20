# 📖 Q&A Toàn Diện: Dự Án AI Race Viettel — Bài 2

> Tài liệu giải thích chi tiết từng thành phần kỹ thuật của hệ thống theo phong cách **ELI5** (Explain Like I'm 5 — giải thích đơn giản như cho người mới). Mỗi câu hỏi đều dựa trên đề bài cuộc thi, kế hoạch triển khai và các nghiên cứu khoa học mới nhất (2024-2026).

---

## PHẦN A: TỔNG QUAN DỰ ÁN

### Q1: Bài toán cuộc thi yêu cầu chúng ta làm gì?

**Ví dụ dễ hiểu:** Hãy tưởng tượng bạn là một thư ký bệnh viện. Bác sĩ viết bệnh án bằng văn bản tự do (có trộn tiếng Việt và tiếng Anh), và nhiệm vụ của bạn là:

1. **Đọc bệnh án** và **gạch chân** tất cả các tên bệnh, tên thuốc, triệu chứng, xét nghiệm.
2. **Phân loại** từng cụm từ gạch chân đó thuộc loại gì: `CHẨN_ĐOÁN`, `THUỐC`, `TRIỆU_CHỨNG`, `TÊN_XÉT_NGHIỆM`, hay `KẾT_QUẢ_XÉT_NGHIỆM`.
3. **Tra mã chuẩn quốc tế** cho mỗi bệnh (mã ICD-10, ví dụ `I10` = tăng huyết áp) và mỗi thuốc (mã RxNorm, ví dụ `308135` = amlodipine 10mg).
4. **Ghi chú thêm** nếu bệnh/thuốc đó bị phủ định ("không bị tiểu đường"), là tiền sử ("trước đây bị sốt rét"), hay là bệnh của người nhà ("bố bị cao huyết áp").
5. **Ghi lại vị trí chính xác** (ký tự thứ bao nhiêu đến bao nhiêu) của mỗi cụm từ trong văn bản gốc.

**Kết quả cuối cùng:** Xuất ra file JSON cho mỗi bệnh án, chứa danh sách các thực thể đã trích xuất kèm đầy đủ thông tin trên.

---

### Q2: Công thức tính điểm cuộc thi hoạt động như thế nào?

Điểm cuối cùng được tính theo công thức:

$$\text{final\_score} = 0.3 \times \text{text\_score} + 0.3 \times \text{assertions\_score} + 0.4 \times \text{candidates\_score}$$

**Giải thích từng thành phần:**

| Thành phần | Trọng số | Ý nghĩa dễ hiểu |
|:---|:---:|:---|
| `text_score` | 30% | Bạn có gạch chân đúng tên bệnh/thuốc không? Đo bằng **WER** (Word Error Rate — tỷ lệ lỗi từ). Càng ít lỗi từ, điểm càng cao. |
| `assertions_score` | 30% | Bạn có ghi chú đúng ngữ cảnh không? (phủ định, tiền sử, gia đình). Đo bằng **Jaccard Similarity** — tỷ lệ trùng khớp giữa nhãn bạn đoán và nhãn đúng. |
| `candidates_score` | 40% | Bạn có tra đúng mã ICD-10/RxNorm không? Cũng đo bằng Jaccard, nhưng có **trọng số theo độ phức tạp** của mỗi thực thể. |

> [!IMPORTANT]
> **Quy tắc phạt nặng:** Nếu bạn gạch chân đúng cụm từ nhưng phân loại sai loại (ví dụ: đoán `CHẨN_ĐOÁN` nhưng thực tế là `TRIỆU_CHỨNG`), bạn bị **phạt gấp đôi** — tạo ra 2 thực thể sai và cả 2 đều nhận 0 điểm cho tất cả 3 metric.

---

### Q3: Trường `position` là gì và tại sao nó lại quan trọng đến vậy?

**Ví dụ dễ hiểu:** Hãy tưởng tượng văn bản bệnh án là một sợi dây dài, mỗi ký tự là một hạt cườm trên dây. `position = [189, 202]` nghĩa là cụm từ bắt đầu từ hạt cườm thứ 189 và kết thúc ở hạt thứ 202.

**Tại sao quan trọng:**
- Hệ thống chấm điểm dùng **IoU** (Intersection over Union — tỷ lệ giao/hợp) trên vị trí ký tự để ghép cặp dự đoán với nhãn chuẩn. Nếu IoU < 0.5 (tức vị trí lệch quá nhiều), dự đoán sẽ bị coi là **sai hoàn toàn**.
- Vị trí tính theo **Unicode codepoint** (mỗi ký tự tiếng Việt có dấu vẫn chỉ tính là 1 đơn vị), không phải byte.

---

### Q4: Có những ràng buộc và giới hạn nào từ Ban Tổ Chức?

| Ràng buộc | Chi tiết |
|:---|:---|
| **Giới hạn model** | Mỗi model tối đa **9 tỷ tham số (9B params)**. Không cộng dồn giữa các model. |
| **Không dùng API ngoài** | Phải tự chạy model trên máy riêng (**self-host**). Không được gọi ChatGPT, Gemini, Claude... |
| **Thời gian chạy** | Tối đa **600 giây** (10 phút) để xử lý toàn bộ 100 file bệnh án. |
| **Nộp bài** | Tối đa **5 lần/ngày**. File nộp là `output.zip` chứa 100 file JSON. |
| **Source code** | Top ~15 đội phải gửi source code để BTC dựng lại và kiểm tra trên dữ liệu private. |

---

## PHẦN B: KIẾN TRÚC HỆ THỐNG (PIPELINE)

### Q5: Tại sao hệ thống lại được thiết kế thành 5 tầng nối tiếp nhau?

**Ví dụ dễ hiểu:** Hãy nghĩ hệ thống như một dây chuyền sản xuất trong nhà máy:

```
Nguyên liệu thô (văn bản) → [Máy cắt] → [Máy phân loại] → [Máy tra mã] → [Kiểm định chất lượng] → [Đóng gói] → Sản phẩm (JSON)
```

Mỗi "máy" (tầng) chỉ làm đúng **một việc duy nhất** và làm tốt nhất việc đó:

| Tầng | "Máy" | Công việc |
|:---:|:---|:---|
| 1 | **NER Extractor** | Cắt ra các cụm từ y khoa + xác định vị trí chính xác |
| 1 | **Assertion Analyzer** | Phân loại ngữ cảnh (phủ định? tiền sử? gia đình?) |
| 2 | **Normalizer** | Làm sạch, chuẩn hóa tên (bỏ liều lượng, mở rộng viết tắt) |
| 3 | **Hybrid Retrieval** | Tra cứu mã ICD-10/RxNorm từ cơ sở dữ liệu |
| 4 | **ClinicalValidator** | Loại bỏ mã phi lý (mã bệnh phụ khoa cho bệnh nhân nam) |
| 5 | **LLM Reranker** | Đọc ngữ cảnh bệnh án, chọn mã chính xác nhất |

**Lợi ích:** Dễ sửa lỗi (chỉ cần sửa 1 tầng), dễ thay thế (đổi model NER mà không ảnh hưởng tầng khác), dễ kiểm tra (đo hiệu năng từng tầng riêng biệt).

---

### Q6: Nếu Tầng 1 (NER) sai, toàn bộ hệ thống có bị sai theo không?

**Trả lời ngắn:** Đúng vậy. Đây gọi là hiện tượng **"Lan truyền lỗi" (Error Propagation)**.

**Ví dụ dễ hiểu:** Giống như xây nhà — nếu móng (Tầng 1) bị lệch, tường (Tầng 3) sẽ nghiêng, và mái (Tầng 5) sẽ đổ. Cụ thể:
- NER trích xuất sai cụm từ `"1 tuần nay"` thành `CHẨN_ĐOÁN`
- → Retrieval đi tìm mã ICD-10 cho `"1 tuần nay"` → ra danh sách mã rác
- → LLM bị ép chọn mã từ danh sách rác → kết quả sai hoàn toàn

**Cơ chế phòng ngự:**
1. **ClinicalValidator:** Lọc bỏ mã phi lý trước khi đến LLM (ví dụ: mã ung thư cổ tử cung cho bệnh nhân nam).
2. **Ngưỡng tự tin LLM:** Nếu tất cả mã ứng viên đều có điểm quá thấp, hệ thống trả về `[]` (rỗng) thay vì đoán bừa.
3. **Fallback toàn pipeline:** Nếu bất kỳ file nào gặp lỗi crash, hệ thống ghi `[]` và tiếp tục file tiếp theo.

---

## PHẦN C: TẦNG 1 — NER & ASSERTION

### Q7: NER là gì? Tại sao chọn XLM-RoBERTa mà không dùng model khác?

**NER (Named Entity Recognition)** — dịch nôm na là "Nhận diện Thực thể có Tên". Nó giống như việc bạn dùng bút highlight để **tô màu** các cụm từ quan trọng trong sách giáo khoa, mỗi màu ứng với một loại (bệnh = đỏ, thuốc = xanh, triệu chứng = vàng...).

**Tại sao chọn XLM-RoBERTa-large:**

| Tiêu chí | XLM-RoBERTa ✅ | PhoBERT ❌ | LLM (Qwen/GPT) ❌ |
|:---|:---|:---|:---|
| Xử lý song ngữ Việt-Anh | Xuất sắc (được huấn luyện trên 100 ngôn ngữ) | Chỉ tiếng Việt | Tốt nhưng chậm |
| Tách từ (Word Segmentation) | Không cần | Bắt buộc dùng RDRSegmenter (dễ lệch vị trí) | Không cần |
| Tốc độ xử lý | ~5ms/file | ~5ms/file | ~3-5 giây/file |
| Độ chính xác vị trí ký tự | 100% nhờ `return_offsets_mapping` | Dễ bị lệch offset do tách từ | Không thể đếm ký tự chính xác |
| Kích thước | ~560M params (rất nhỏ) | ~135M params | ~7B params (quá lớn) |

> **Nghiên cứu:** Benchmark VietMed-NER (NAACL 2025 Industry Track) xác nhận rằng mô hình multilingual encoder (như XLM-R) **vượt trội** mô hình đơn ngữ (monolingual) trên bài toán NER lâm sàng tiếng Việt.

---

### Q8: `return_offsets_mapping` là gì? Tại sao nó giúp xác định vị trí chính xác 100%?

**Ví dụ dễ hiểu:** Khi model đọc từ `"paracetamol"`, nó không đọc nguyên từ mà chia nhỏ thành các mảnh (subtoken): `["para", "##ceta", "##mol"]`. Vấn đề là: mỗi mảnh nằm ở vị trí nào trong văn bản gốc?

`return_offsets_mapping` giải quyết việc này bằng cách trả về một bản đồ chính xác:
```
"para"     → (0, 4)    # từ ký tự 0 đến 4
"##ceta"   → (4, 8)    # từ ký tự 4 đến 8
"##mol"    → (8, 11)   # từ ký tự 8 đến 11
```

→ Ghép lại: `"paracetamol"` nằm từ ký tự 0 đến 11. **Chính xác tuyệt đối, không cần đoán!**

---

### Q9: Assertion là gì? NegEx là gì?

**Assertion** là nhãn mô tả **ngữ cảnh lâm sàng** của một thực thể. Có 3 loại:

| Assertion | Ý nghĩa | Ví dụ trong bệnh án |
|:---|:---|:---|
| `isNegated` | Bệnh/triệu chứng bị **phủ định** | *"**Không** phát hiện tiểu đường"* |
| `isHistorical` | Là **tiền sử** (đã xảy ra trong quá khứ) | *"Tiền sử bị **sốt rét** năm 2020"* |
| `isFamily` | Là bệnh của **người nhà**, không phải bệnh nhân | *"**Bố** bị cao huyết áp"* |

**NegEx** là một thuật toán dạng **rule-based** (dựa trên luật) — giống như dùng bộ lọc từ khóa:
- Nếu trước thực thể có từ `"không"`, `"chưa"`, `"loại trừ"` → gán `isNegated`
- Nếu có từ `"tiền sử"`, `"trước đây"` → gán `isHistorical`
- Nếu có từ `"bố"`, `"mẹ"`, `"anh chị em"` → gán `isFamily`

**Tại sao quan trọng:** Nếu không phát hiện `isNegated`, hệ thống sẽ gán mã bệnh cho một bệnh mà bệnh nhân **không hề mắc** → sai hoàn toàn.

---

### Q10: Làm sao để đảm bảo mô hình NER hoạt động tốt nhất?

1. **Dữ liệu huấn luyện chất lượng (Synthetic Data):** Dùng LLM (Qwen) sinh bệnh án tiếng Việt giả lập theo 3 văn phong khác nhau (văn xuôi, liệt kê toa thuốc, EHR key-value), rồi gán nhãn BIO tự động.
2. **Xử lý mất cân bằng nhãn:** Trong bệnh án, 90% từ là từ thường (`O`), chỉ 10% là thực thể. Dùng **Focal Loss** hoặc **Weighted Cross-Entropy** để model không bị "lười" chỉ đoán `O` cho mọi từ.
3. **Feedback Loop (Vòng lặp cải tiến):** Sau mỗi lần chạy thử, phân tích **confusion matrix** (bảng nhầm lẫn) xem NER hay nhầm loại nào nhất (ví dụ: hay lẫn `CHẨN_ĐOÁN` với `TRIỆU_CHỨNG`), rồi sinh thêm dữ liệu huấn luyện nhắm trúng loại lỗi đó.
4. **Regex Post-processing:** Sau khi NER trích xuất xong, dùng biểu thức chính quy để chuẩn hóa ranh giới (ví dụ: đảm bảo liều lượng thuốc như `500mg` luôn nằm trong thực thể thuốc).

---

## PHẦN D: TẦNG 2 — TIỀN XỬ LÝ & CHUẨN HÓA

### Q11: Tại sao phải "làm sạch" tên thực thể trước khi tra mã?

**Ví dụ dễ hiểu:** Nếu NER trích xuất được `"amlodipine 10 mg po daily"`, chúng ta không thể đem nguyên cả chuỗi này đi tìm trong CSDL thuốc vì:
- `"po"` (per os = uống) là **đường dùng** — RxNorm không phân loại theo đường dùng
- `"daily"` (mỗi ngày) là **tần suất** — không liên quan đến mã thuốc
- `"10 mg"` là **hàm lượng** — quan trọng nhưng cần tách riêng

**Quy trình chuẩn hóa (Normalizer):**
1. **Unicode NFC:** Đưa tất cả ký tự về dạng chuẩn (chữ `ở` có 1 cách viết duy nhất)
2. **Chữ thường:** `"Paracetamol"` → `"paracetamol"`
3. **Bóc liều/tần suất thuốc:** `"amlodipine 10 mg po daily"` → `"amlodipine 10 mg"`
4. **Mở rộng viết tắt:** `"THA"` → `"tăng huyết áp"`, `"ĐTĐ"` → `"đái tháo đường"`

---

### Q12: Override Dictionary là gì? Tại sao cần nó?

**Ví dụ dễ hiểu:** Giống như bạn có một cuốn sổ ghi nhớ cá nhân: nếu thấy tên bệnh `"tăng huyết áp"`, bạn ngay lập tức biết mã là `I10` mà không cần tra Google. Nhanh, chính xác, không lãng phí thời gian.

Override Dictionary chứa 50-100 bệnh/thuốc phổ biến nhất (nguồn từ Bộ Y Tế, DrugBank VN). Nếu tên thực thể đã chuẩn hóa **khớp chính xác** với danh sách này → gán thẳng mã chuẩn, **bỏ qua** Retrieval và LLM → tiết kiệm thời gian xử lý.

---

## PHẦN E: TẦNG 3 — HYBRID RETRIEVAL

### Q13: BM25s là gì? Giải thích đơn giản.

**Ví dụ dễ hiểu:** BM25 giống như tìm kiếm của Google nhưng **chỉ dựa trên từ khóa**. Nó đếm xem từ bạn tìm xuất hiện bao nhiêu lần trong mỗi tài liệu, và tài liệu nào chứa nhiều từ khớp nhất sẽ được xếp hạng cao nhất.

**Tại sao BM25 mạnh trong y khoa:**
- Trong y khoa, sai **1 con số** ở hàm lượng thuốc = sai mã hoàn toàn. Ví dụ: `"amlodipine 5mg"` (mã `197361`) ≠ `"amlodipine 10mg"` (mã `308135`). BM25 bắt chính xác sự khác biệt này.
- `BM25s` là phiên bản tối ưu viết bằng C qua thư viện Scipy, nhanh hơn gấp trăm lần so với phiên bản Python thuần.

---

### Q14: FAISS là gì? Embedding là gì?

**Embedding** — dịch nôm na là "chuyển chữ thành số". Mỗi cụm từ được biến thành một dãy số dài (ví dụ: 1024 con số). Các cụm từ có ý nghĩa giống nhau sẽ có dãy số **gần nhau** trong không gian toán học.

```
"tăng huyết áp" → [0.12, -0.34, 0.56, ...]  ← gần nhau
"cao huyết áp"  → [0.13, -0.33, 0.55, ...]  ← gần nhau
"đau bụng"      → [-0.87, 0.22, -0.11, ...] ← xa hẳn
```

**FAISS** (Facebook AI Similarity Search) là thư viện do Meta phát triển, chuyên **tìm kiếm nhanh** các dãy số gần nhau nhất. Nó giống như một "GPS" trong không gian số: cho nó một tọa độ, nó sẽ tìm 5 tọa độ gần nhất trong vài mili-giây.

---

### Q15: Tại sao cần kết hợp BM25 và FAISS (Hybrid Retrieval)? Dùng 1 cái thôi không được à?

| Tình huống | BM25 (từ khóa) | FAISS (ngữ nghĩa) |
|:---|:---|:---|
| Tìm `"amlodipine 5mg"` | ✅ Tìm chính xác con số `5mg` | ❌ Có thể nhầm với `10mg` vì ngữ nghĩa gần |
| Tìm `"THA"` (viết tắt) | ❌ Không hiểu `THA` là gì | ✅ Hiểu `THA` ≈ `tăng huyết áp` |
| Tìm `"đau ngực"` → `"angina pectoris"` | ❌ Không khớp từ nào | ✅ Hiểu 2 cụm có nghĩa giống nhau |
| Tìm `"paracetamol 325mg"` | ✅ Tìm chính xác `325mg` | ❌ Có thể nhầm liều lượng |

> **Nghiên cứu (2024-2025):** Cộng đồng nghiên cứu AI đã đạt được sự đồng thuận rằng các hệ thống tìm kiếm y khoa hiệu suất cao nhất đều sử dụng kiến trúc **Hybrid (kết hợp)**. BM25 đảm bảo độ chính xác ký tự/con số, FAISS đảm bảo bao phủ ngữ nghĩa và đồng nghĩa.

---

### Q16: RRF (Reciprocal Rank Fusion) là gì? Tại sao dùng nó?

**Ví dụ dễ hiểu:** Hãy tưởng tượng bạn nhờ 2 người bạn (BM25 và FAISS) cùng xếp hạng 10 bộ phim hay nhất. Mỗi người có tiêu chí riêng (BM25 thích phim có tên trùng khớp, FAISS thích phim có nội dung tương đồng). Vấn đề là điểm số của họ không so sánh được (BM25 cho điểm 12.4, FAISS cho điểm 0.85).

**RRF giải quyết bằng cách bỏ qua điểm số, chỉ dùng thứ hạng:**

$$\text{Score}(phim) = \frac{W_{BM25}}{Rank_{BM25} + 60} + \frac{W_{FAISS}}{Rank_{FAISS} + 60}$$

- Phim nào được **cả 2 người** xếp hạng cao → điểm RRF cao nhất
- Phim chỉ 1 người thích → điểm thấp hơn
- Số `60` là hằng số phạt rank tiêu chuẩn, giúp giảm ảnh hưởng của các phim xếp hạng thấp

**Ưu điểm chính:** Không cần chuẩn hóa điểm số (normalization-free), đơn giản nhưng hiệu quả cao, được sử dụng trong Elasticsearch, Azure AI Search và các hệ thống production lớn.

---

### Q17: Recall@5 là gì? Tại sao lại đo chỉ số này?

**Recall@5** đo khả năng của hệ thống tìm kiếm: *"Trong 5 mã ứng viên trả về, có chứa mã đúng hay không?"*

**Ví dụ dễ hiểu:** Bạn đang chơi trò chơi đoán số. Bạn được đoán 5 lần. Nếu trong 5 lần đó có 1 lần đúng → Recall = 100%. Nếu cả 5 lần đều sai → Recall = 0%.

**Tại sao quan trọng:**
- Recall@5 là **"trần" (ceiling)** của điểm số cuối cùng. Nếu Retrieval trả về Top 5 mà không chứa mã đúng, thì dù LLM có thông minh đến đâu cũng không thể chọn đúng (vì LLM bị ép chỉ chọn từ danh sách Top 5).
- **Kết quả thực tế trên dự án:** BGE-M3 đạt Recall@5 = **61.13%** (285/427 chẩn đoán + 61/139 thuốc), dưới ngưỡng yêu cầu 80% → bắt buộc phải kích hoạt SapBERT làm phương án dự phòng.

---

### Q18: SapBERT là gì? Tại sao nó được kỳ vọng tốt hơn BGE-M3 cho y khoa?

**BGE-M3** là mô hình embedding đa ngôn ngữ, đa nhiệm (biết nhiều thứ nhưng không chuyên sâu y tế).

**SapBERT** (Self-Alignment Pretraining for BERT) là mô hình embedding được huấn luyện **chuyên biệt trên UMLS** — cơ sở tri thức y khoa lớn nhất thế giới chứa hàng triệu khái niệm y tế và các mối quan hệ đồng nghĩa giữa chúng.

**Ví dụ dễ hiểu:**
- BGE-M3 giống như một sinh viên đa ngành — biết rộng nhưng không sâu về y khoa
- SapBERT giống như một bác sĩ chuyên khoa — chỉ biết y khoa nhưng biết rất sâu

> **Nghiên cứu (BioNNE-L 2025, BioASQ Workshop tại CLEF):** Các hệ thống đạt giải cao nhất đều sử dụng SapBERT làm thành phần cốt lõi cho giai đoạn truy vấn ứng viên (Candidate Retrieval), kết hợp với bộ xếp hạng lại (Re-ranker) để đạt độ chính xác tối ưu.

---

## PHẦN F: TẦNG 4 — HẬU KIỂM Y KHOA (CLINICAL VALIDATION)

### Q19: ClinicalValidator làm gì? Cho ví dụ cụ thể.

ClinicalValidator là bộ lọc hậu kiểm dựa trên **luật y khoa** để loại bỏ các mã ứng viên phi lý:

| Luật | Ví dụ bị loại |
|:---|:---|
| **Luật giới tính** | Mã `N76.0` (Viêm âm đạo cấp) cho bệnh nhân **nam** |
| **Luật độ tuổi** | Mã `P07.3` (Sinh non) cho bệnh nhân **65 tuổi** |
| **Luật dạng bào chế** | Mã thuốc dạng **tiêm** nhưng bệnh án ghi **uống** |
| **Mã kép Dagger/Asterisk** | Tự động bổ sung mã phụ nếu phát hiện mã chính |

**Cơ chế fail-safe:** Nếu ClinicalValidator không thể trích xuất tuổi/giới tính từ bệnh án, nó sẽ **không lọc gì cả** (trả về nguyên danh sách) thay vì lọc sai.

---

## PHẦN G: TẦNG 5 — LLM RERANKER

### Q20: Tại sao cần LLM ở cuối pipeline? Retrieval đã tìm được mã rồi mà?

**Ví dụ dễ hiểu:** Retrieval giống như Google — cho bạn 5 kết quả liên quan nhất. Nhưng Google không đọc hiểu ngữ cảnh câu hỏi của bạn. LLM giống như một chuyên gia y khoa ngồi đọc cả bệnh án rồi nói: *"Trong 5 mã này, mã số 3 là phù hợp nhất vì bệnh nhân có triệu chứng X kết hợp Y."*

**Ví dụ cụ thể:**
> Bệnh án: *"Bệnh nhân vào viện vì ho sốt. Tiền sử bố bị **tăng huyết áp**. Đã loại trừ **hẹp van tim**."*

- Retrieval trả về mã `I10` (tăng huyết áp) và `I35.0` (hẹp van tim) vì chúng có mặt trong văn bản
- LLM đọc ngữ cảnh và nhận ra:
  - `I10` → là bệnh của **bố**, không phải bệnh nhân → **loại**
  - `I35.0` → đã bị **loại trừ** → **loại**

---

### Q21: XGrammar / Constrained Decoding là gì? Tại sao bắt buộc phải dùng?

**Vấn đề:** LLM sinh văn bản tự do (free-form text). Nếu hỏi LLM "chọn mã nào?", nó có thể trả lời `"Tôi nghĩ mã I10 phù hợp"` hoặc thậm chí bịa ra mã không tồn tại `"I99.999"`. Điều này gọi là **Hallucination** (ảo giác).

**XGrammar giải quyết:** Nó giống như đặt LLM vào một phòng thi trắc nghiệm — LLM chỉ được chọn A, B, C, D, E (tương ứng 5 mã ứng viên từ Retrieval). Không thể tự viết đáp án mới.

**Cách hoạt động kỹ thuật:** Tại mỗi bước sinh token, XGrammar kiểm tra token nào hợp lệ theo JSON schema và **đặt xác suất = 0** cho tất cả token vi phạm. Kết quả: output luôn là JSON hợp lệ 100%, mã luôn nằm trong danh sách cho phép.

> **Nghiên cứu (2025-2026):** XGrammar được tích hợp mặc định trong vLLM, sử dụng pushdown automaton (PDA) để theo dõi trạng thái ngữ pháp. Phiên bản XGrammar-2 mới nhất hỗ trợ cả cấu trúc tool calling và reasoning channels cho các hệ thống AI Agent phức tạp.

---

### Q22: Quantization (lượng tử hóa) là gì? Tại sao model 7B vẫn chạy được trên laptop?

**Ví dụ dễ hiểu:** Hãy tưởng tượng mỗi "neuron" trong model là một bình nước. Model gốc dùng bình 16 lít (FP16) để chứa nước. Quantization đổi sang bình 4 lít (INT4) — bình nhỏ hơn 4 lần, chứa ít nước hơn một chút nhưng vẫn đủ dùng.

**Kết quả thực tế:**
- Model Qwen2.5-7B gốc: cần **~14GB** RAM/VRAM
- Sau quantize GGUF Q4: chỉ cần **~5GB** RAM/VRAM
- Chất lượng suy luận giảm không đáng kể (~1-2%)

**Hai định dạng phổ biến:**
- **GGUF** (dùng với llama.cpp): Chạy được trên CPU, phù hợp máy không có GPU mạnh
- **AWQ** (dùng với vLLM): Tối ưu cho GPU, tốc độ inference nhanh hơn

---

## PHẦN H: TỐI ƯU HÓA PHẦN CỨNG

### Q23: Làm sao 1 máy chạy được 3 model cùng lúc mà không bị tràn bộ nhớ?

**Bí quyết:** Không phải 3 model chạy cùng lúc! Thiết kế thông minh giúp giảm tải:

| Model | Khi nào chạy | RAM/VRAM cần |
|:---|:---|:---|
| **Embedding (BGE-M3/SapBERT)** | Chỉ chạy **1 lần duy nhất** trước cuộc thi để sinh file index FAISS tĩnh | 0 GB lúc chạy pipeline (chỉ nạp file index ~1.5GB vào RAM) |
| **NER (XLM-R)** | Chạy đầu pipeline cho mỗi file | ~1-2 GB |
| **LLM (Qwen2.5-7B GGUF)** | Chạy cuối pipeline cho mỗi file | ~5-6 GB |
| **Tổng lúc chạy thực tế** | | **~8-9 GB** |

---

## PHẦN I: CÂU HỎI VỀ LLM & NER

### Q24: LLM có thể phát hiện thực thể mà NER bỏ sót không?

**Có thể**, nhưng phải thực hiện rất cẩn thận. LLM đọc hiểu ngữ cảnh tốt hơn NER nên có thể nhận ra các thực thể ẩn (ví dụ: *"bệnh nhân được kê thuốc hạ áp"* → LLM hiểu đây là `THUỐC` dù không có tên cụ thể).

**Giải pháp an toàn (String Matching Fallback):**
1. LLM trả về **tên chữ** của thực thể phát hiện thêm (ví dụ: `"đái tháo đường"`)
2. Code Python tự động tìm vị trí chính xác: `text.find("đái tháo đường")` → `position = [start, end]`
3. Kiểm tra kết quả: nếu tìm thấy → bổ sung vào JSON, nếu không → bỏ qua

**Tại sao không để LLM tự đoán vị trí:**
LLM hoạt động trên cơ chế subtoken (BPE/SentencePiece), không thể đếm ký tự Unicode chính xác. Sai lệch 1-2 ký tự → IoU < 0.5 → bị phạt điểm.

---

### Q25: Tại sao không để LLM làm tất cả (cả NER, cả tra mã, cả ghi chú)?

Về lý thuyết, LLM đủ thông minh để làm mọi thứ. Nhưng trong cuộc thi này, có 3 rào cản cực lớn:

1. **Tốc độ:** LLM mất 3-5 giây/file. 100 file = 300-500 giây. Cộng thêm thời gian khởi động model → **dễ bị timeout** (giới hạn 600 giây). Trong khi NER chỉ mất 0.5 giây cho 100 file.
2. **Vị trí ký tự:** LLM không thể đếm ký tự Unicode chính xác → sai `position` → mất điểm `text_score`.
3. **Hallucination mã:** LLM có thể bịa ra mã ICD-10/RxNorm không tồn tại. Dù có XGrammar ép JSON, nếu không có Retrieval cung cấp danh sách hợp lệ → không có gì để ép.

> **Nghiên cứu "Is Information Extraction Solved by ChatGPT?" (Li et al., EMNLP 2023):** Các mô hình nhỏ được fine-tune vẫn **vượt trội hoàn toàn** LLM về mặt định vị chính xác vị trí và nhãn thực thể trong tài liệu chuyên ngành y tế.

---

## PHẦN J: CƠ SỞ DỮ LIỆU Y KHOA

### Q26: ICD-10 là gì?

**ICD-10** (International Classification of Diseases, 10th Revision) là hệ thống mã hóa bệnh tật quốc tế do WHO (Tổ chức Y tế Thế giới) ban hành. Mỗi bệnh được gán một mã duy nhất.

Ví dụ:
- `I10` = Tăng huyết áp nguyên phát
- `E11.9` = Đái tháo đường type 2 không biến chứng
- `J18.9` = Viêm phổi không xác định

**Trong dự án:** CSDL chứa **25,123 mã ICD-10** kèm tên tiếng Việt và tiếng Anh, lưu trong SQLite.

---

### Q27: RxNorm là gì?

**RxNorm** là hệ thống mã hóa thuốc do NLM (Thư viện Y khoa Quốc gia Mỹ) quản lý. Mỗi thuốc (hoạt chất + hàm lượng + dạng bào chế) có một mã `rxcui` duy nhất.

Ví dụ:
- `308135` = amlodipine 10 MG Oral Tablet
- `197527` = clonazepam 0.5 MG Oral Tablet

**Trong dự án:** CSDL chứa **362,401 bản ghi thuốc** + **372,592 bản ghi ánh xạ lịch sử** (old_cui → new_cui), lưu trong SQLite.

> [!WARNING]
> **Quyết định kỹ thuật then chốt:** Bắt buộc dùng bản **RxNorm Full** (thư mục `rrf/`) thay vì bản rút gọn (`prescribe/`), vì nhiều thuốc trong đề bài đã bị **Obsolete** (ngưng sử dụng) và không tồn tại trong bản rút gọn.

---

## PHẦN K: XU HƯỚNG AI MỚI NHẤT

### Q28: Xu hướng "Retriever-Reranker" trong AI y khoa 2025 là gì?

Đây chính xác là kiến trúc mà dự án của chúng ta đang sử dụng! Xu hướng này đã trở thành **tiêu chuẩn vàng** trong lĩnh vực Biomedical Entity Linking (Liên kết thực thể y sinh):

1. **Retriever (Bộ truy vấn):** Dùng SapBERT/BGE-M3 để nhanh chóng thu hẹp không gian tìm kiếm từ hàng trăm nghìn mã xuống chỉ còn Top 5-10.
2. **Reranker (Bộ xếp hạng lại):** Dùng Cross-Encoder hoặc LLM để phân tích sâu ngữ cảnh và chọn ra mã chính xác nhất.

> **Cuộc thi BioNNE-L 2025 (BioASQ Workshop tại CLEF):** Tất cả các đội top đều sử dụng kiến trúc Retriever-Reranker. Đội thắng giải kết hợp SapBERT embedding + Jaccard lexical similarity + Context-aware reranker.

---

### Q29: Data-Centric AI là gì? Liên quan gì đến dự án?

**Data-Centric AI** là triết lý: *"Thay vì tốn công cải tiến model, hãy tập trung cải tiến dữ liệu huấn luyện."*

**Trong dự án:** Chúng ta dùng LLM (Qwen) để **sinh dữ liệu huấn luyện chất lượng cao** cho model NER nhỏ (XLM-R). LLM đóng vai trò "thầy giáo" tạo bài tập, model NER nhỏ đóng vai "học sinh" luyện tập. Đây là cách sử dụng LLM hiệu quả nhất — dùng sức mạnh suy luận của LLM để nâng cao chất lượng training data thay vì chạy LLM trực tiếp lúc inference (tốn thời gian).

---

### Q30: "Self-host model" nghĩa là gì trong cuộc thi?

**Self-host** = Tự chạy model trên máy của mình, không gọi API bên ngoài.

Trong cuộc thi, điều này có nghĩa:
- ❌ KHÔNG được gọi OpenAI API, Google Gemini API, Anthropic API
- ✅ Phải tải model weights về máy và chạy inference cục bộ
- ✅ Có thể dùng các thư viện như `vLLM`, `llama-cpp-python` để chạy model
- ✅ Model tối đa **9B tham số** mỗi model

**Ví dụ cấu hình chạy trong Docker:**
```
Docker container khởi động
→ Nạp model NER (~560M) vào RAM
→ Nạp model LLM (~7B GGUF) vào GPU
→ Nạp FAISS index vào RAM
→ Xử lý 100 file lần lượt
→ Xuất 100 file JSON
```
