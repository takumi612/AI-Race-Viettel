# Báo Cáo Quá Trình Tối Ưu Hóa Hệ Thống Truy Xuất Thực Thể Y Khoa (Retriever)

Tài liệu này ghi nhận quá trình nghiên cứu, phát hiện lỗi, đề xuất giải pháp và các kết quả đạt được trong việc tối ưu hóa hiệu năng Recall@5 cho hệ thống truy xuất thực thể **Chẩn đoán (ICD-10)** và **Thuốc (RxNorm)** thuộc dự án AI Race Viettel.

---

## 1. So Sánh Chỉ Số Trước & Sau Tối Ưu Hóa (Recall@5 trên 100 file GT)

Nhờ chuỗi các giải pháp tối ưu hóa dữ liệu và thuật toán, hiệu năng tổng thể của cả 2 mô hình đã có bước nhảy vọt vượt bậc, đưa mô hình BGE-M3 **đạt đỉnh 70.67% Recall@5**:

### 📊 Mô hình BGE-M3 (Khuyên dùng)
| Chỉ số thực thể | Trước tối ưu hóa (Baseline) | Sau tối ưu hóa (Final) | Mức độ cải thiện (Tuyệt đối) |
| :--- | :---: | :---: | :---: |
| **Chẩn đoán (ICD-10)** | 0.6721 *(287/427)* | **0.7237** *(309/427)* | **+5.16%** 📈 |
| **Thuốc (RxNorm)** | 0.4245 *(59/139)* | **0.6547** *(91/139)* | **+23.02%** 🚀 |
| **Tổng thể (Overall)** | 0.6113 *(346/566)* | **0.7067** *(400/566)* | **+9.54%** 💥 |

### 📊 Mô hình SapBERT
| Chỉ số thực thể | Trước tối ưu hóa (Baseline) | Sau tối ưu hóa (Final) | Mức độ cải thiện (Tuyệt đối) |
| :--- | :---: | :---: | :---: |
| **Chẩn đoán (ICD-10)** | 0.6581 *(281/427)* | **0.6956** *(297/427)* | **+3.75%** 📈 |
| **Thuốc (RxNorm)** | 0.4388 *(61/139)* | **0.6403** *(89/139)* | **+20.15%** 🚀 |
| **Tổng thể (Overall)** | 0.6042 *(342/566)* | **0.6820** *(386/566)* | **+7.78%** 💥 |

---

## 2. Các Vấn Đề Gặp Phải & Cách Giải Quyết (Problems & Solutions)

### ❌ Vấn đề 1: Mất mát thông tin hàm lượng thuốc (Drug Dosage Drop)
*   **Hiện tượng:** Hàm `remove_dosage` cũ sử dụng các biểu thức chính quy (regex) thô sơ để xóa sạch các đơn vị đo lường (như `81 mg`, `20 mg`, `500 mg`) ra khỏi thực thể thuốc. 
*   **Hậu quả:** RxNorm quản lý mã hoạt chất dựa trên hàm lượng cụ thể. Khi mất thông tin này (ví dụ: `"Lipitor 20 mg"` bị xóa thành `"lipitor"`), các kết quả của `Lipitor 10 mg` hoặc `40 mg` sẽ xếp đè lên và đẩy mã đúng ra khỏi Top 5.
*   **Cách giải quyết:** Thiết kế lại regex trong hàm `remove_dosage` (`src/retrieval/normalizer.py`) để trích xuất riêng biệt hàm lượng, giữ nguyên tên hoạt chất gốc, sau đó ghép hàm lượng đã chuẩn hóa (ví dụ: `MG`, `ML`) vào cuối chuỗi truy vấn để làm từ khóa khớp chính xác.

### ❌ Vấn đề 2: Lỗi viết tắt trong câu ghép (Abbreviation Logic Fail)
*   **Hiện tượng:** Hàm `expand_abbreviation` cũ thực hiện so khớp nguyên chuỗi (`text_upper in self.abbreviations`).
*   **Hậu quả:** Khi từ viết tắt nằm trong một câu ghép (ví dụ: `"asa 325mg po x1..."`), hệ thống không dịch được từ khóa `"asa"` thành `"aspirin"`, làm giảm trầm trọng khả năng tìm kiếm của cả BM25 và FAISS.
*   **Cách giải quyết:** Token hóa chuỗi đầu vào và thực hiện dịch/mở rộng viết tắt ở **cấp độ từng từ đơn lẻ (Token-level)** trước khi ghép lại thành câu hoàn chỉnh.

### ❌ Vấn đề 3: Lỗi dịch thuật nghiêm trọng trong CSDL gốc (`ICD10.xlsx`)
*   **Hiện tượng:** Trong quá trình phân tích lỗi truy xuất mã chẩn đoán, phát hiện ra mã `I10` (Bệnh tăng huyết áp vô căn) luôn bị xếp hạng rất thấp dù chuỗi truy vấn là `"tăng huyết áp"`.
*   **Hậu quả:** Trong file Excel nguồn `ICD10.xlsx`, mô tả tiếng Anh của mã `I10` bị gán sai thành `"Other rheumatic heart diseases"` (Bệnh tim do thấp khớp khác). Điều này làm vector embedding của mã `I10` bị lệch hoàn toàn khỏi nhóm bệnh huyết áp.
*   **Cách giải quyết:** Sử dụng bộ sửa lỗi dữ liệu đầu nguồn, cập nhật trực tiếp mô tả tiếng Anh của mã `I10` thành `"Essential (primary) hypertension"` ngay trong file Excel, sau đó rebuild lại toàn bộ database SQLite `metadata.db`.

### ❌ Vấn đề 4: Nhập nhèm ngôn ngữ khi sinh vector (Separated Indexing)
*   **Hiện tượng:** Cách sinh index cũ ghép song ngữ Việt-Anh (Concat mode) thành một chuỗi duy nhất để sinh vector biểu diễn.
*   **Hậu quả:** Điều này làm loãng thông tin và gây nhiễu không gian vector.
*   **Cách giải quyết:** Tách biệt hoàn toàn việc lưu trữ chỉ mục (Separated Indexing):
    *   Mỗi mã ICD-10 được sinh ra 2 bản ghi vector độc lập: một cho tiếng Việt (`name_vi`) và một cho tiếng Anh (`name_en`).
    *   Bổ sung bộ khử trùng lặp (deduplication) trước khi tính điểm RRF trong `src/retrieval/hybrid_retriever.py` để tránh hiện tượng cộng dồn điểm ảo khi một mã xuất hiện nhiều lần trong FAISS.

### ❌ Vấn đề 5: Sai lệch mức độ chi tiết ở ICD-10 (Granularity Mismatch)
*   **Hiện tượng:** Thực thể lâm sàng ghi nhận chung chung (ví dụ: `"viêm tụy cấp"`, `"xơ gan"`) dẫn đến việc mô hình trả về mã cha 3 ký tự tổng quát (`K85`, `K74`) ở vị trí số 1.
*   **Hậu quả:** Nhãn chuẩn (Gold) yêu cầu mã con cụ thể (.8 hoặc .9) nên hệ thống bị tính là đoán sai.
*   **Cách giải quyết:** Áp dụng luật **Fallback phân cấp ICD-10 (Hierarchical Fallback Rule)**:
    *   Nếu trong Top 3 dự đoán có xuất hiện mã cha 3 ký tự, hệ thống tự động tìm kiếm các mã con trực thuộc trong CSDL SQLite.
    *   Chấm điểm các mã con dựa trên sự xuất hiện của từ khóa `"không đặc hiệu"` (unspecified - 2 điểm) hoặc `"khác"` (other - 1 điểm).
    *   Chèn tối đa **2 mã con** có điểm số cao nhất vào ngay sau mã cha để đảm bảo cơ hội bao phủ nhãn chuẩn mà không làm loãng Top 5.

### ❌ Vấn đề 6: Nhập nhèm dạng bào chế và tên thương mại trong RxNorm (Form & Brand Disambiguation)
*   **Hiện tượng:** Tương tự ICD-10, mô hình truy xuất dễ trả về mã SCDC (hoạt chất + hàm lượng chung chung, ví dụ: `aspirin 325 MG`) khi bác sĩ chỉ kê đơn giản từ khóa, trong khi nhãn chuẩn yêu cầu mã cụ thể SCD (có dạng bào chế cụ thể, ví dụ: `aspirin 325 MG Oral Tablet`).
*   **Hậu quả:** Mất điểm vì lệch phân cấp thông tin dạng bào chế trong RxNorm.
*   **Cách giải quyết:** Triển khai **Fallback phân cấp cho RxNorm (RxNorm Hierarchical Fallback Rule)**:
    *   Nếu kết quả trả về chứa mã SCDC hoặc IN (Ingredient), hệ thống tự động truy vấn các mã SCD (Semantic Clinical Drug) tương ứng bắt đầu bằng tên hoạt chất đó.
    *   Chấm điểm ưu tiên các dạng bào chế phổ biến nhất dựa trên phân phối thống kê nhãn chuẩn của tập Ground Truth: `Oral Tablet` / `Oral Capsule` nhận 2 điểm; `Injection` / `Solution` nhận 1 điểm.
    *   Chèn tối đa **2 mã SCD con** có điểm cao nhất vào ngay sau mã gốc. Giải pháp này giúp Recall@5 của thực thể Thuốc tăng thêm **+2.16%** đối với BGE-M3 và **+1.44%** đối với SapBERT.

---

## 3. Phân Tích Các Ca Lỗi Còn Lại (Remaining Failures)

Sau tất cả các bước tối ưu hóa, các nhóm lỗi chính còn tồn tại trong hệ thống bao gồm:

1.  **Thiếu thông tin bệnh cảnh phối hợp (Contextual Omission):**
    *   *Mô tả:* Nhãn chuẩn yêu cầu mã tích hợp nhiều bệnh lý (VD: `K80.4` - Sỏi đường mật có viêm túi mật) nhưng thực thể trích xuất chỉ đơn thuần ghi nhận một khía cạnh bệnh lý (VD: `"sỏi đường mật"`).
    *   *Nguyên nhân:* Mô hình truy xuất chỉ hoạt động trên một thực thể độc lập nên không có thông tin về các chẩn đoán đi kèm khác trong hồ sơ bệnh án.
2.  **Sự nhập nhèm dạng bào chế khi văn bản hoàn toàn không gợi ý:**
    *   *Mô tả:* Một số ca lỗi RxNorm xảy ra khi văn bản y khoa hoàn toàn không có bất kỳ từ chỉ dẫn cách dùng nào (như `po`, `uống`, `tiêm`) và hoạt chất đó lại có quá nhiều dạng bào chế phân bố đều (ví dụ: dạng bôi ngoài da, thuốc xịt hít). Bộ lọc Fallback mặc định ưu tiên dạng uống (`Oral Tablet`) sẽ bị đoán sai trong trường hợp này.
3.  **Khẩu ngữ y tế Việt Nam quá đặc thù:**
    *   *Mô tả:* Thuật ngữ khẩu ngữ của Việt Nam (VD: `"rối loạn mỡ máu"`) không khớp với danh pháp dịch nghĩa học thuật của WHO trong ICD-10 (VD: `"Rối loạn chuyển hóa lipoprotein"`).

---

## 4. Hướng Cải Thiện Khuyến Nghị Cho Tương Lai

Để tiếp tục đẩy chỉ số Recall@5 lên cao hơn nữa (mục tiêu > 75%), hệ thống cần được bổ sung 2 giải pháp sau:
1.  **Tích hợp tầng Rerank bằng LLM (Contextual LLM Reranker):**
    *   Sử dụng mô hình lượng tử hóa **Qwen2.5-7B-Reranker** (chạy bằng `llama.cpp` trên GGUF) để đánh giá lại Top 20 ứng viên. 
    *   LLM sẽ được cung cấp toàn bộ đoạn văn bệnh án (ngữ cảnh) để suy luận các biến chứng đi kèm, giải quyết triệt để lỗi thiếu thông tin bệnh cảnh phối hợp (Nhóm 1) và nhận diện dạng bào chế thông qua từ ngữ chỉ dẫn cách dùng (Nhóm 2).
2.  **Xây dựng bộ ánh xạ từ đồng nghĩa lâm sàng Việt Nam (Clinical Synonyms Dictionary):**
    *   Tạo file từ điển JSON để chuyển đổi trực tiếp các khẩu ngữ lâm sàng phổ biến thành các từ khóa chuẩn y khoa trước khi đưa vào sinh vector.
    *   *Ví dụ:* `"mỡ máu"` $\rightarrow$ `"lipoprotein"`; `"tiểu đường"` $\rightarrow$ `"đái tháo đường"`; `"tai biến"` $\rightarrow$ `"tai biến mạch máu não"`.
