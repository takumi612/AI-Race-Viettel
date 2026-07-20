# Training annotations

Đặt dữ liệu có nhãn trong thư mục này trước khi chạy notebook trên Colab.

## Dạng 1: cặp TXT và JSON

```text
train/
├── 001.txt
└── 001.json
```

`001.json` là danh sách entity:

```json
[
  {
    "text": "đái tháo đường",
    "type": "DISEASE",
    "candidates": ["E11.9"],
    "assertions": ["AFFIRMED", "CURRENT"],
    "position": [25, 39]
  }
]
```

`position` dùng `[start, end]` với `end` exclusive và bắt buộc thỏa `raw_text[start:end] == text`.

## Dạng 2: thư mục input/gt

```text
synthetic_train_v1/
├── input/
│   └── 001.txt
└── gt/
    └── 001.json
```

JSON trong `gt/` dùng cùng entity schema như dạng 1. Đây là layout được notebook tự nhận khi `TRAIN_DIR_OVERRIDE` trỏ tới thư mục `synthetic_train_v1`.

## Dạng 3: JSON record tự chứa text

```json
{
  "document_id": "001",
  "raw_text": "...",
  "entities": [
    {
      "text": "...",
      "type": "DISEASE",
      "candidates": [],
      "assertions": [],
      "position": [0, 3]
    }
  ],
  "relations": []
}
```

Không đặt private/test input vào thư mục này. Notebook split train/validation theo document trước khi tokenization và chunking.
