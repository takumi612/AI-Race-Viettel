# Giai đoạn 9 - Final File Generation

## Trạng thái

Hoàn thành.

## Đã tạo

- `medical_information_extraction_lab.ipynb`: notebook duy nhất gồm section 0-30 theo dependency order.
- `README.md`: hướng dẫn cài đặt, từng stage, FAST_DEV_RUN, inference, rebuild KB và limitations.
- `requirements.txt`: baseline và optional supervised dependencies.
- README riêng cho stage 1-9.
- `tools/build_notebook.py` và `tools/execute_notebook.py`.
- `COLAB_RUNBOOK.md`: cấu trúc Drive, annotation layouts, GPU setup và đường dẫn `output.zip`.

## Bằng chứng

- Notebook JSON `nbformat=4`, không có code cell rỗng.
- Executor chạy tuần tự toàn bộ code cells thành công.
- Production data gate tự tìm dữ liệu thật, từ chối smoke input mặc định và lưu ZIP về Drive trên Colab.
- Full unit test suite pass.
- `ARTIFACT_MANIFEST.json` liệt kê file, size, SHA-256, producer/consumer và rebuild instruction.
- `PROJECT_STATE.md` và `DECISIONS.md` có stage checkpoints/decision evidence.
- `reports/final_verification.json` ghi exact counts, test result, notebook execution, ZIP validation, reload equivalence và private-test fitting flag.

## Giới hạn được ghi rõ

Không có train/validation annotation trong dữ liệu cung cấp, nên notebook không bịa supervised score. Official entity/assertion mapping đã được xác nhận bằng validator trong repository; khi có annotation thật, notebook sẽ train theo layout trong `train/README.md` rồi reload checkpoint trước inference.
