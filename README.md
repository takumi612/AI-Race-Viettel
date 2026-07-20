# Clinical NLP Colab Pipeline

Repository tối giản với hai notebook:

- `medical_information_extraction_lab.ipynb`: Colab Run all.
- `medical_information_extraction_kaggle.ipynb`: train XLM-R bằng GPU Kaggle,
  dùng checkpoint vừa train để tạo `output.zip`.

## File runtime được giữ lại

```text
AI-Race-Viettel/
├── artifacts/                         # config, mapping và KB cache
├── clinical_nlp_lab/                  # package inference/training
├── tools/                             # build notebook/KB và CLI inference
├── test/                              # archive, không dùng khi Run all
├── medical_information_extraction_lab.ipynb
├── medical_information_extraction_kaggle.ipynb
├── requirements-colab.txt
├── requirements-kaggle.txt
├── COLAB_RUNBOOK.md
├── KAGGLE_RUNBOOK.md
└── README.md
```

Notebook không phụ thuộc vào nội dung trong `test/`.

## Chạy nhanh trên Colab

Chuẩn bị một trong hai nguồn input trên Google Drive:

```text
MyDrive/AI-Race-Viettel/data/input/<id>.txt
```

hoặc:

```text
MyDrive/AI-Race-Viettel/data/input.zip
```

Sau đó:

1. Mở notebook trên Colab.
2. Chọn GPU tại **Runtime → Change runtime type**.
3. Chọn **Runtime → Run all**.
4. Chấp nhận mount Google Drive.

Notebook tự clone nhánh `Pipeline_colab`, cài dependency, tìm dữ liệu thật và
lưu kết quả tại:

```text
MyDrive/AI-Race-Viettel/output/output.zip
```

Hướng dẫn đầy đủ nằm trong `COLAB_RUNBOOK.md`.

## Training trên Kaggle

Import `medical_information_extraction_kaggle.ipynb`, attach Dataset có input
và annotation, bật GPU rồi Run All. Kết quả được lưu tại
`/kaggle/working/output.zip` và `/kaggle/working/trained_ner_artifacts.zip`.
Xem `KAGGLE_RUNBOOK.md` để setup Internet-on hoặc offline bằng attached Dataset.

## Training tùy chọn

Để train trước inference, dùng một trong hai layout:

```text
data/train/001.txt
data/train/001.json
```

hoặc:

```text
data/synthetic_train_v1/input/001.txt
data/synthetic_train_v1/gt/001.json
```

Nếu không có annotation, notebook bỏ qua supervised training và chạy baseline
artifact/rule-dictionary; không tạo metric giả.

## Thành phần trong `test/`

`test/archive/` chứa source pipeline cũ, reports, stage README, test suite và
công cụ audit đã dùng để phát triển/kiểm chứng. Chúng được giữ để tra cứu nhưng
không tham gia Colab `Run all`.
