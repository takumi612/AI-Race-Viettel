# Thiết kế bootstrap dữ liệu training từ Google Drive

## Mục tiêu

Repo GitHub không chứa dataset lớn. Mọi hướng dẫn training phải bắt đầu bằng
việc tải thư mục `data` canonical từ Google Drive, sau đó mới kiểm tra dữ liệu,
build dataset trung gian và chạy từng trainer.

Nguồn canonical:

- URL: `https://drive.google.com/drive/folders/1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe?usp=drive_link`
- Folder ID: `1WdqC1BHvbcm0xDw2KjJ4uKOqxZsPMiQe`
- Đích trong repo: `data/`

## Phạm vi thay đổi

Đồng bộ ba tài liệu:

1. `README.md`: đặt bước tải `data` trước cài đặt/build/training và thay link
   Drive cũ.
2. `docs/training/DATA_FOUNDATION.md`: mô tả nguồn Drive canonical và cách
   bootstrap dữ liệu trên Colab.
3. `docs/training/MODEL_TRAINING_COLAB.md`: thay luồng mount/symlink phụ thuộc
   cấu trúc MyDrive bằng luồng tải trực tiếp, có kiểm tra trước khi train.

Không thay đổi code training, config model hoặc dataset local. Không đưa nội
dung `data` vào Git.

## Luồng Colab đã chọn

1. Clone repo và chuyển vào thư mục gốc.
2. Cài `gdown>=6.0.0`; phiên bản 6 hỗ trợ folder có hơn 50 file.
3. Tải toàn bộ Drive folder vào đúng `data/`, không tạo `data/data`.
4. Kiểm tra các đường dẫn bắt buộc:
   `data/synthetic_train_v1`, `data/dev`, `data/kb/metadata.db` và
   `data/models`.
5. Kiểm tra synthetic có đúng 2.000 file input, 2.000 file GT, QA đạt và
   SHA-256 của `manifest.jsonl` bằng
   `66bd0e58ae1adc72ae2b00ed36df42b6b1012a4ec4e8367c43ef5c0d2a54292a`.
6. Cài dependency training, chạy test và build `data/training`.
7. Chỉ chạy trainer sau khi build manifest hợp lệ.

Lệnh download dùng URL folder đầy đủ và output directory tuyệt đối trên
Colab. Trên fresh clone, placeholder `.gitkeep` không được coi là dữ liệu.
Hướng dẫn phải báo lỗi rõ nếu thư mục Drive chưa public hoặc download thiếu.

## An toàn và khả năng khôi phục

- Không dùng lệnh xóa đệ quy trong notebook.
- Nếu `data` đã có dữ liệu, người dùng phải đổi tên/backup trước khi tải lại;
  không tự ghi đè một dataset không rõ nguồn gốc.
- Không tạo symlink `data/models/models`.
- `data/input` chỉ dùng inference/nộp bài, không tham gia training.
- Artifact training và checkpoint phải được copy sang Drive nếu cần giữ qua
  lần reset runtime; dữ liệu nguồn vẫn giữ fingerprint canonical.

## Tiêu chí hoàn thành

- Ba tài liệu dùng cùng một Drive URL và cùng cấu trúc `data`.
- README thể hiện rõ download là bước đầu tiên sau clone.
- Các command Colab chạy từ `/content/AI-Race-Viettel` và không tạo folder
  lồng sai.
- Có cell kiểm tra count, QA, manifest hash và kích thước `metadata.db`.
- Không còn link Drive dữ liệu cũ trong ba tài liệu training.
- `git diff --check` và kiểm tra link/path tĩnh đều đạt.
