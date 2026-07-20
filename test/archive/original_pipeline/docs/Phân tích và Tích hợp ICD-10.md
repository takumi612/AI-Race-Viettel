# Phân tích & Tích hợp Bộ dữ liệu ICD-10 (Bệnh / Chẩn đoán)

Tài liệu này trình bày chi tiết kết quả phân tích nghiệp vụ tập dữ liệu danh mục ICD-10 của Bộ Y tế Việt Nam và giải pháp kỹ thuật tích hợp vào CSDL SQLite của hệ thống AI Race Viettel.

---

## 1. Phân tích Dữ liệu Nguồn (ICD10.xlsx)

* **Nguồn gốc**: File Excel danh mục ICD-10 chính thức do Bộ Y tế Việt Nam ban hành phục vụ quản lý khám chữa bệnh và thanh toán bảo hiểm y tế.
* **Quy mô dữ liệu**: Chứa **25,123 mã ICD-10** độc nhất sau khi làm sạch và tích hợp song ngữ.
* **Cấu trúc chi tiết**:
  * **Master sheet (`ICD10`)**:
    * Cột 17 (`Ma_Benh`): Mã ký tự của bệnh lý theo tiêu chuẩn WHO (ví dụ: `I10`, `E11.9`).
    * Cột 19 (`Ten_Tieng_Anh`): Tên bệnh tiếng Anh chuẩn.
    * Cột 20 (`Ten_Tieng_Viet`): Tên bệnh tiếng Việt tương ứng.
  * **Các sheet luật lâm sàng (Clinical Constraints)**:
    * **Sheet `A1` (Mã kép Dagger/Asterisk - $†$ và $*$)**: Quy định các cặp mã luôn đi kèm nhau. Mã căn nguyên (Dagger $†$) phải đi trước mã biểu hiện (Asterisk $*$). Ví dụ: bệnh võng mạc do đái tháo đường cần cặp mã `E11.3†` và `H36.0*`.
    * **Sheet `A2` (Không làm bệnh chính)**: Chứa các mã ICD-10 chỉ được dùng làm chẩn đoán phụ (không được đặt làm chẩn đoán chính/principal diagnosis).
    * **Sheet `A3.1` đến `A3.10` (Giới hạn nhóm tuổi)**: Quy định tuổi tối thiểu và tối đa được phép áp dụng cho từng mã bệnh (ví dụ: các mã sơ sinh `P00-P96` chỉ dùng dưới 28 ngày tuổi).
    * **Sheet `A4.1` (Chỉ dùng cho Nữ) và `A4.2` (Chỉ dùng cho Nam)**: Ràng buộc giới tính sinh học đối với mã bệnh (ví dụ: các mã u xơ tử cung chỉ cho Nữ, phì đại tuyến tiền liệt chỉ cho Nam).

---

## 2. Thiết kế Cơ sở Dữ liệu SQLite (`metadata.db`)

Dữ liệu ICD-10 được lưu trữ trong 5 bảng quan hệ chuyên biệt để phục vụ truy vấn nhanh và hậu kiểm luật:

### Bảng `icd10` (Từ điển chẩn đoán chuẩn)
```sql
CREATE TABLE IF NOT EXISTS icd10 (
    code TEXT PRIMARY KEY,
    name_vi TEXT,
    name_en TEXT
);
CREATE INDEX IF NOT EXISTS idx_icd10_code ON icd10(code);
```

### Bảng `icd10_rules_sex` (Luật giới tính sinh học)
```sql
CREATE TABLE IF NOT EXISTS icd10_rules_sex (
    code TEXT PRIMARY KEY,
    allowed_sex TEXT -- 'M' cho Nam, 'F' cho Nữ
);
```

### Bảng `icd10_rules_age` (Luật độ tuổi áp dụng)
```sql
CREATE TABLE IF NOT EXISTS icd10_rules_age (
    code TEXT PRIMARY KEY,
    min_days INTEGER, -- Tuổi tối thiểu quy đổi sang ngày
    max_days INTEGER, -- Tuổi tối đa quy đổi sang ngày
    description TEXT
);
```

### Bảng `icd10_rules_dual` (Luật kết hợp mã kép Dagger-Asterisk)
```sql
CREATE TABLE IF NOT EXISTS icd10_rules_dual (
    dagger_code TEXT,
    asterisk_code TEXT,
    PRIMARY KEY (dagger_code, asterisk_code)
);
```

### Bảng `icd10_rules_not_primary` (Luật chẩn đoán chính)
```sql
CREATE TABLE IF NOT EXISTS icd10_rules_not_primary (
    code TEXT PRIMARY KEY
);
```

---

## 3. Quy trình Tích hợp và Xử lý trong Pipeline

Quy trình xử lý thực thể `CHẨN_ĐOÁN` trong pipeline end-to-end diễn ra qua các bước sau:

1. **NER Extractor**: Nhận diện thực thể bệnh lý (ví dụ: *"đái tháo đường"*).
2. **Text Normalization**: Chuyển chữ thường, xóa khoảng trắng thừa.
3. **Retrieval**: BM25s tìm kiếm trên song ngữ (`name_vi` và `name_en` của bảng `icd10`) để tìm ra Top 5 ứng viên mã ICD-10 có điểm tương đồng cao nhất.
4. **Clinical Validation (Hậu xử lý)**:
   * **Trích xuất thông tin hành chính**: Quét tìm thông tin giới tính và tuổi trong 150 ký tự đầu của bệnh án (qua `PatientExtractor`).
   * **Lọc theo Luật Giới tính & Độ tuổi**: Lọc bỏ các ứng viên ICD-10 vi phạm luật so với thông tin bệnh nhân.
   * **Kiểm tra mã kép**: Nếu chẩn đoán có chứa mã Dagger ($†$), kiểm tra và tự động bổ sung mã Asterisk ($*$) tương ứng để bảo toàn cấu trúc dữ liệu.
