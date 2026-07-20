# 📖 Hướng Dẫn Chi Tiết Huấn Luyện Mô Hình LLM Reranker (Qwen2.5-7B)

Tài liệu này hướng dẫn chi tiết cách chuẩn bị dữ liệu huấn luyện, cấu hình phương pháp **QLoRA** (Quantized Low-Rank Adaptation) và chạy lệnh huấn luyện cho mô hình **Qwen2.5-7B-Instruct** để tối ưu hóa khả năng suy luận ngữ cảnh y khoa lâm sàng và lựa chọn mã chuẩn chính xác.

---

## 1. Chuẩn Bị Dữ Liệu Huấn Luyện (SFT Format)

Huấn luyện LLM Reranker sử dụng phương pháp **Supervised Fine-Tuning (SFT)** với định dạng Chat (System, User, Assistant). Dữ liệu huấn luyện được thiết kế dưới dạng đưa cho LLM ngữ cảnh bệnh án + danh sách ứng viên, bắt LLM sinh ra đúng cấu trúc JSON chứa mã được chọn.

### Định dạng dữ liệu mẫu trong file `sft_train_data.json`:
```json
[
  {
    "instruction": "Bạn là chuyên gia mã hóa y tế lâm sàng. Hãy đọc bệnh án gốc, thực thể cần gán mã và danh sách các mã ứng viên y tế tương ứng để chọn ra duy nhất 1 mã chính xác nhất.",
    "input": "Bệnh án: Bệnh nhân nam 70 tuổi nhập viện vì ho sốt. Đã làm siêu âm tim và loại trừ hẹp van động mạch chủ.\nThực thể cần mã hóa: 'hẹp van động mạch chủ'\nLoại thực thể: CHẨN_ĐOÁN\nThuộc tính: isNegated (Đã bị loại trừ)\nCác mã ứng viên y tế:\n- I35.0 (Hẹp van động mạch chủ)\n- I35.2 (Hẹp van động mạch chủ kèm hở van)\n- Q23.0 (Hẹp van động mạch chủ bẩm sinh)",
    "output": "{\n  \"candidates\": [\"I35.0\"]\n}"
  }
]
```

---

## 2. Huấn Luyện Tiết Kiệm Tài Nguyên Với QLoRA

Do mô hình Qwen2.5-7B khá nặng, việc huấn luyện full-weights sẽ làm tràn VRAM GPU thông thường. Chúng ta sử dụng kỹ thuật **QLoRA**:
* **Lượng tử hóa 4-bit (NF4):** Đóng băng mô hình nền ở định dạng 4-bit siêu nhẹ.
* **Low-Rank Adapters (LoRA):** Chỉ huấn luyện các ma trận bổ trợ (rank $r=16$, $alpha=32$) được nhúng vào các lớp Attention và FFN của mô hình.

### Các Target Modules cần nhúng LoRA adapters:
* `q_proj`, `k_proj`, `v_proj`, `o_proj` (Attention)
* `gate_proj`, `up_proj`, `down_proj` (Feed-Forward Network)

---

## 3. Cấu Hình Huấn Luyện Bằng LLaMA-Factory (Khuyên Dùng)

**LLaMA-Factory** là framework mã nguồn mở tối ưu nhất hiện nay để huấn luyện LLM cục bộ một cách nhanh chóng và ổn định.

### Bước 1: Tạo file cấu hình `qwen_lora_sft.yaml`
```yaml
### Cấu hình mô hình nền
model_name_or_path: data/models/Qwen2.5-7B-Instruct

### Phương pháp huấn luyện
stage: sft
do_train: true
finetuning_type: lora
lora_target: q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj
lora_rank: 16
lora_alpha: 32

### Dữ liệu
dataset: sft_train_data
dataset_dir: data/processed
template: qwen
cutoff_len: 1024
max_samples: 5000
val_size: 0.1

### Tham số tối ưu
quantization_bit: 4
learning_rate: 2e-4
num_train_epochs: 3.0
plot_loss: true
output_dir: data/models/qwen_reranker_lora

### Tăng tốc phần cứng
fp16: true
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

### Bước 2: Khởi chạy lệnh huấn luyện
```bash
llamafactory-cli train qwen_lora_sft.yaml
```

---

## 4. Tham Số Huấn Luyện Khuyến Nghị (Hyperparameters)

| Tham số | Giá trị đề xuất | Ý nghĩa |
|:---|:---:|:---|
| **Learning Rate** | `2e-4` | Mức học chuẩn cho adapter LoRA để hội tụ nhanh |
| **LoRA Rank (r)** | `16` (hoặc `32`) | Rank 16 đủ để học các cấu trúc ngữ nghĩa y khoa phức tạp |
| **LoRA Alpha** | `32` (hoặc `64`) | Thường thiết lập bằng $2 \times r$ để cân bằng trọng số |
| **Accumulation Steps** | `8` | Tạo batch size ảo lớn hơn giúp gradient ổn định hơn trên GPU ít VRAM |
| **Cutoff Length** | `1024` hoặc `2048` | Độ dài tối đa của prompt y khoa (bao gồm danh sách ứng viên) |

---

## 5. Xuất File Model Lượng Tử Hóa (AWQ / GGUF)

Sau khi huấn luyện LoRA thành công, chúng ta cần **Merge** adapter vào mô hình gốc và đóng gói sang định dạng lượng tử hóa **GGUF** để chạy nhẹ nhàng trên CPU/GPU máy chấm:

```bash
# 1. Merge weights
llamafactory-cli export \
    --model_name_or_path data/models/Qwen2.5-7B-Instruct \
    --adapter_name_or_path data/models/qwen_reranker_lora \
    --template qwen \
    --export_dir data/models/Qwen2.5-7B-Reranker-Merged \
    --export_size 2 \
    --export_device cpu

# 2. Chuyển đổi sang GGUF Q4_K_M bằng llama.cpp
python llama.cpp/convert_hf_to_gguf.py data/models/Qwen2.5-7B-Reranker-Merged \
    --outtype q4_k_m \
    --outfile data/models/qwen_reranker_q4.gguf
```
