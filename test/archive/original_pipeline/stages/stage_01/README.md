# Giai đoạn 1 - Contract và Data Audit

## Trạng thái

Hoàn thành ngày 2026-07-19. Không huấn luyện hoặc fit trên `input.zip`.

## Đã hoàn thành

- Chuyển prompt nguồn thành `SPEC.md` và critical invariants.
- Audit strict UTF-8, CRC, filename và độ dài của 100 tài liệu.
- Audit workbook ICD-10 và quét streaming RXNCONSO/RXNREL.
- Xác định thiếu train/validation annotations và official label mappings.
- Chốt kiến trúc mặc định, leakage policy và các organizer confirmations còn thiếu.

## Bằng chứng

- `reports/PHASE1_DATA_AUDIT.md`
- `reports/phase1_audit.json`
- `tools/phase1_data_audit.py`
- 100/100 input hợp lệ; 0 annotation files; 12.219 ICD-10 rows; 1.202.603 RXNCONSO rows; 7.423.180 RXNREL rows.

## Quyết định

Không suy diễn nhãn từ private input. Các module supervised chỉ chạy khi annotated loader tìm thấy ground truth hợp lệ.

