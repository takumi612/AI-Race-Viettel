# 📖 Hướng Dẫn Chi Tiết Huấn Luyện Mô Hình Embedding (SapBERT)

Tài liệu này hướng dẫn cách huấn luyện (fine-tune) mô hình embedding chuyên sâu y học **SapBERT** sử dụng học tương phản (Contrastive Learning) trên tập dữ liệu từ vựng y tế lâm sàng để tối ưu hóa khả năng tìm kiếm ứng viên mã ICD-10 và RxNorm.

---

## 1. Bản Chất Của Học Tương Phản (Contrastive Learning) Trong Y Khoa

**Ý tưởng cốt lõi:** Làm sao để mô hình hiểu được các thuật ngữ viết khác nhau nhưng cùng mô tả một loại bệnh lý/dược chất thì phải có khoảng cách vector rất gần nhau.
* **Positive Pairs (Cặp tích cực):** (`"tăng huyết áp"`, `"cao huyết áp"`), (`"đái tháo đường type 2"`, `"tiểu đường tuýp 2"`). Mô hình cần kéo vector của chúng lại gần nhau.
* **Negative Pairs (Cặp tiêu cực):** (`"tăng huyết áp"`, `"huyết áp thấp"`), (`"đái tháo đường"`, `"sốt xuất huyết"`). Mô hình cần đẩy vector của chúng ra xa nhau.

---

## 2. Chuẩn Bị Dữ Liệu Huấn Luyện (Triplets / Pairs)

Dữ liệu huấn luyện tốt nhất thu thập từ UMLS hoặc CSDL Bộ Y tế dưới dạng các cặp đồng nghĩa hoặc bộ ba (Anchor, Positive, Negative):

```json
[
  {
    "anchor": "ĐTĐ tuýp 2",
    "positive": "đái tháo đường không phụ thuộc insulin",
    "negative": "đái tháo đường type 1"
  },
  {
    "anchor": "amlodipine 5 mg",
    "positive": "amlodipin 5mg oral tablet",
    "negative": "amlodipine 10 mg"
  }
]
```

---

## 3. Thiết Kế Loss Function (Multi-Similarity Loss / InfoNCE)

Để tối ưu hóa cấu trúc vector trong không gian đa chiều, sử dụng hàm **Multi-Similarity Loss** (đề xuất gốc của SapBERT) hoặc **OnlineContrastiveLoss** từ thư viện `sentence-transformers`.

### Code Huấn luyện mẫu bằng `sentence-transformers`:
```python
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader

# 1. Load model nền SapBERT (hoặc XLM-RoBERTa-base làm khởi đầu)
model = SentenceTransformer("cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR")

# 2. Chuẩn bị tập Train dưới dạng InputExample
train_examples = [
    InputExample(texts=["ĐTĐ tuýp 2", "Đái tháo đường type 2"]),
    InputExample(texts=["tăng huyết áp", "cao huyết áp"]),
    InputExample(texts=["Amlodipine 5mg", "Amlodipin 5 mg oral tablet"])
]

# 3. Sử dụng DataLoader để quản lý batch
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=32)

# 4. Sử dụng MultipleNegativesRankingLoss (InfoNCE Loss) 
# Tự động coi các sample khác trong cùng batch làm Negative samples
train_loss = losses.MultipleNegativesRankingLoss(model=model)

# 5. Thực hiện huấn luyện
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,
    warmup_steps=100,
    output_path="data/models/sapbert_fine_tuned"
)
```

---

## 4. Tham Số Huấn Luyện Khuyến Nghị (Hyperparameters)

| Tham số | Giá trị đề xuất | Ý nghĩa |
|:---|:---:|:---|
| **Learning Rate** | `2e-5` đến `5e-6` | SapBERT rất nhạy cảm, học quá nhanh sẽ phá hủy cấu trúc vector UMLS gốc |
| **Batch Size** | `64` hoặc `128` | Batch size lớn giúp hàm loss InfoNCE có nhiều negative samples để so sánh |
| **Pooling Mode** | `CLS` token | SapBERT được thiết kế để biểu diễn ngữ nghĩa cô đọng tại CLS token |
| **Max Sequence Length**| `64` | Tên thực thể y khoa thường ngắn, độ dài 64 giúp tiết kiệm bộ nhớ |
| **Optimizer** | AdamW | Bộ tối ưu hóa tiêu chuẩn |

---

## 5. Đánh Giá Mô Hình (Recall@K Evaluation)

Sử dụng tập dữ liệu dev gồm các từ lâm sàng thực tế và mã ICD-10/RxNorm tương ứng:
1. Sinh vector cho toàn bộ CSDL mã chuẩn.
2. Với mỗi từ truy vấn trong tập dev, tìm Top K (K=1, K=5) mã có khoảng cách cosine gần nhất.
3. Đo chỉ số **Recall@1** và **Recall@5** (tỷ lệ tìm trúng mã chuẩn trong danh sách Top K).
4. Nếu Recall@5 tăng đáng kể so với model gốc $\rightarrow$ Mô hình fine-tune hoạt động tốt.
