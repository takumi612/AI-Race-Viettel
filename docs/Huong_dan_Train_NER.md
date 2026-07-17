# 📖 Hướng Dẫn Chi Tiết Huấn Luyện Mô Hình NER (XLM-RoBERTa-large)

Tài liệu này hướng dẫn chi tiết cách chuẩn bị dữ liệu, cấu hình mã nguồn và thực hiện huấn luyện (fine-tune) mô hình **XLM-RoBERTa-large** kết hợp lớp phân loại **CRF** (Conditional Random Fields) cho nhiệm vụ Nhận diện Thực thể Y khoa (NER).

---

## 1. Chuẩn Bị Dữ Liệu Huấn Luyện (BIO Format)

Mô hình NER yêu cầu dữ liệu đầu vào được gán nhãn theo định dạng **BIO** ở cấp độ từ (word-level) hoặc subtoken-level:
* **B- {Entity_Type}** (Beginning): Bắt đầu của thực thể.
* **I- {Entity_Type}** (Inside): Các từ tiếp theo nằm trong thực thể.
* **O** (Outside): Các từ thường không thuộc thực thể nào.

### Ví dụ dữ liệu đầu vào:
```text
Bệnh_nhân    O
uống         O
Amlodipine   B-THUỐC
10           I-THUỐC
mg           I-THUỐC
hàng_ngày    O
.            O
```

### 5 loại nhãn thực thể y khoa cần nhận diện:
1. `CHẨN_ĐOÁN`
2. `THUỐC`
3. `TRIỆU_CHỨNG`
4. `TÊN_XÉT_NGHIỆM`
5. `KẾT_QUẢ_XÉT_NGHIỆM`

---

## 2. Thiết Kế Mô Hình XLM-R + CRF

Sử dụng mạng backbone **XLM-RoBERTa-large** (trích xuất đặc trưng ngữ cảnh) kết hợp lớp **CRF** ở cuối để tối ưu hóa việc dự đoán chuỗi nhãn liên tục (tránh trường hợp nhãn `I-THUỐC` đứng ngay sau nhãn `B-CHẨN_ĐOÁN` vốn phi lý ngữ pháp).

### Cấu trúc mô hình trong PyTorch:
```python
import torch
import torch.nn as nn
from transformers import XLMRobertaModel
from torchcrf import CRF  # thư viện pytorch-crf

class XLMRobertaCRFForNER(nn.Module):
    def __init__(self, model_name="xlm-roberta-large", num_labels=11):
        super().__init__()
        self.xlm_roberta = XLMRobertaModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.xlm_roberta.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.xlm_roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs[0]  # Shape: [batch_size, seq_len, hidden_size]
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # Shape: [batch_size, seq_len, num_labels]
        
        # Tạo mask kiểu ByteTensor cho CRF
        crf_mask = attention_mask.to(torch.uint8)
        
        if labels is not None:
            # Tính toán Loss trong quá trình Train
            log_likelihood = self.crf(emissions, labels, mask=crf_mask, reduction='mean')
            return -log_likelihood  # Loss là âm log-likelihood
        else:
            # Giải mã nhãn tối ưu trong quá trình Inference
            return self.crf.decode(emissions, mask=crf_mask)
```

---

## 3. Quy Trình Căn Chỉnh Vị Trí Ký Tự (Offset Mapping)

Để đảm bảo không bị lệch tọa độ tuyệt đối `position: [start, end]` trên văn bản gốc:
1. Sử dụng `XLMRobertaTokenizerFast` với thuộc tính `return_offsets_mapping=True`.
2. Ánh xạ từng nhãn BIO từ danh sách từ khóa y khoa về đúng subtoken tương ứng.

```python
from transformers import XLMRobertaTokenizerFast

tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")

def tokenize_and_align_labels(examples, label_to_id):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding="max_length",
        max_length=256
    )
    # Logic gán ID nhãn cho từng subtoken dựa trên offset_mapping...
    return tokenized_inputs
```

---

## 4. Tham Số Huấn Luyện Khuyến Nghị (Hyperparameters)

| Tham số | Giá trị đề xuất | Ý nghĩa |
|:---|:---:|:---|
| **Learning Rate** | `2e-5` cho XLM-R, `1e-3` cho CRF | Tránh làm hỏng các weights đã học trước của mạng lớn |
| **Batch Size** | `16` (hoặc `8` nếu thiếu VRAM) | Đảm bảo tính ổn định của gradient |
| **Epochs** | `5 - 8` | Đủ để hội tụ mà không bị Overfitting |
| **Weight Decay** | `0.01` | Chống Overfitting |
| **Loss Function** | CRF Loss | Đảm bảo tính tối ưu của toàn chuỗi nhãn |
| **Lr Scheduler** | Linear warmup to decay | Tăng đều ở bước đầu, giảm dần ở các bước sau |

---

## 5. Đánh Giá Mô Hình (Evaluation Metrics)

Sử dụng thư viện `seqeval` để đánh giá hiệu năng mô hình NER dựa trên các thực thể hoàn chỉnh (chỉ tính đúng khi khớp cả cụm từ, không tính điểm đơn lẻ trên từng từ):

* **Precision (Độ chính xác):** Tỷ lệ thực thể dự đoán đúng trên tổng số thực thể dự đoán ra.
* **Recall (Độ bao phủ):** Tỷ lệ thực thể dự đoán đúng trên tổng số thực thể thực tế (Ground Truth).
* **F1-Score:** Trung bình điều hòa của Precision và Recall.

### Lệnh chạy đánh giá mẫu:
```python
from seqeval.metrics import classification_report

# y_true và y_pred là list các list nhãn dạng chữ (ví dụ: ["O", "B-THUỐC", "I-THUỐC"])
print(classification_report(y_true, y_pred))
```
