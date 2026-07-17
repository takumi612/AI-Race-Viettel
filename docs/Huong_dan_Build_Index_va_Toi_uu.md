# Hướng dẫn Dựng Chỉ mục (Build Index) và Chiến lược Tối ưu hóa Hệ thống CPU-only

Tài liệu này chi tiết hóa các tệp tin được sinh ra sau khi chạy các tập lệnh sinh chỉ mục, hướng dẫn từng bước vận hành cụ thể và phân tích kỹ thuật chuyên sâu về các giải pháp tối ưu hóa hiệu năng trên phần cứng CPU của bạn.

---

## 1. Kết quả Đầu ra: Các tệp tin được sinh ra (Generated Files)

Sau khi Người A chạy hoàn tất các bước ở mục vận hành, hệ thống sẽ sinh ra các tệp tin và cấu trúc thư mục sau trong thư mục cục bộ `data/kb/`:

### 1.1. Các thư mục Chỉ mục FAISS hoàn chỉnh (Sử dụng lúc chạy Pipeline chính)
Đây là các thư mục đích chứa chỉ mục vector tĩnh đã được build sẵn, được lớp `HybridRetriever` trực tiếp nạp lúc runtime:

1. **Chỉ mục ICD-10 (Chẩn đoán) - Mô hình BGE-M3 (Chính):**
   - Đường dẫn thư mục: `data/kb/icd10_bge-m3_index/`
   - Tệp tin bên trong:
     - `index.faiss`: Tệp nhị phân chứa cấu trúc cây tìm kiếm không gian vector IndexFlatIP (dung lượng khoảng 100 MB).
     - `codes.txt`: Danh sách mã ICD-10 tương ứng với thứ tự vector trong chỉ mục.
2. **Chỉ mục RxNorm (Thuốc) - Mô hình BGE-M3 (Chính):**
   - Đường dẫn thư mục: `data/kb/rxnorm_bge-m3_index/`
   - Tệp tin bên trong:
     - `index.faiss`: Chỉ mục FAISS chứa vector của 362k loại thuốc (dung lượng khoảng 1.4 GB).
     - `codes.txt`: Danh sách mã RxCUI tương ứng.
3. **Chỉ mục ICD-10 (Chẩn đoán) - Mô hình SapBERT (Dự phòng):**
   - Đường dẫn thư mục: `data/kb/icd10_sapbert_index/`
   - Tệp tin bên trong:
     - `index.faiss`: Chỉ mục vector 768 chiều (dung lượng khoảng 75 MB).
     - `codes.txt`: Danh sách mã ICD-10 tương ứng.
4. **Chỉ mục RxNorm (Thuốc) - Mô hình SapBERT (Dự phòng):**
   - Đường dẫn thư mục: `data/kb/rxnorm_sapbert_index/`
   - Tệp tin bên trong:
     - `index.faiss`: Chỉ mục vector 768 chiều của 362k loại thuốc (dung lượng khoảng 1.1 GB).
     - `codes.txt`: Danh sách mã RxCUI tương ứng.

### 1.2. Các tệp trung gian (Intermediate Files - Có thể xóa sau khi build xong)
Đây là các tệp vector thô dạng numpy và tệp mã code tạm thời được xuất ra bởi `generate_embeddings.py` để làm đầu vào cho `build_faiss_index.py`:
* `data/kb/icd10_bge-m3_embeddings.npy` & `icd10_bge-m3_codes.txt`
* `data/kb/rxnorm_bge-m3_embeddings.npy` & `rxnorm_bge-m3_codes.txt`
* `data/kb/icd10_sapbert_embeddings.npy` & `icd10_sapbert_codes.txt`
* `data/kb/rxnorm_sapbert_embeddings.npy` & `rxnorm_sapbert_codes.txt`

> [!TIP]
> Sau khi Người A chạy xong lệnh `build_faiss_index.py` và kiểm tra thấy các folder index có đầy đủ tệp `index.faiss`, bạn hoàn toàn có thể xóa các tệp `.npy` và `.txt` trung gian này để giải phóng khoảng **3 - 4 GB** không gian ổ đĩa.

---

## 2. Chi tiết các Bước Thực thi (Execution Steps)

Để dựng lại chỉ mục từ đầu trên toàn bộ cơ sở dữ liệu, Người A thực hiện theo các bước sau:

### Bước 0: Tải và Nạp Mô hình Lâm sàng Offline từ HuggingFace (Cần kết nối mạng)
Trước khi có thể sinh vector và build chỉ mục offline, chúng ta bắt buộc phải tải đầy đủ file cấu hình và trọng số (weights) của 2 mô hình từ HuggingFace Hub về máy cục bộ.

1. **Yêu cầu cài đặt thư viện hỗ trợ (Chạy một lần duy nhất):**
   Mô hình SapBERT sử dụng bộ mã hóa XLM-RoBERTa, yêu cầu các thư viện sau để nạp bộ tách từ:
   ```bash
   pip install sentencepiece protobuf tiktoken
   ```
2. **Chạy script tải tự động:**
   Chạy tập lệnh `download_models.py` để tải và lưu trữ mô hình trực tiếp vào thư mục `data/models/`:
   ```bash
   python src/utils/download_models.py
   ```
   *Lưu ý:*
   - Lệnh này sẽ tải mô hình **BGE-M3** (khoảng 2.27 GB) và lưu tại `data/models/bge-m3/`.
   - Tải mô hình **SapBERT** (khoảng 1.11 GB) và lưu tại `data/models/sapbert/`.
   - Đường truyền tải sử dụng giao thức đa luồng trực tiếp từ máy chủ Amazon CloudFront CDN của HuggingFace gốc, tự động ngắt kết nối IPv6 nếu chập chờn và chuyển hướng sang IPv4 ổn định.

### Bước 1: Sinh chỉ mục cho ICD-10 (BGE-M3)
1. Mở terminal tại thư mục gốc của dự án `d:\AI Race Viettel\`.
2. Chạy lệnh sinh vector:
   ```bash
   python src/retrieval/generate_embeddings.py --model BGE-M3 --table icd10
   ```
3. Chạy lệnh dựng index FAISS từ vector thô:
   ```bash
   python src/retrieval/build_faiss_index.py --model BGE-M3 --table icd10
   ```

### Bước 2: Sinh chỉ mục cho RxNorm (BGE-M3)
1. Chạy lệnh sinh vector cho 362k bản ghi thuốc:
   ```bash
   python src/retrieval/generate_embeddings.py --model BGE-M3 --table rxnorm
   ```
2. Chạy lệnh dựng index FAISS:
   ```bash
   python src/retrieval/build_faiss_index.py --model BGE-M3 --table rxnorm
   ```

### Bước 3: Dựng chỉ mục dự phòng cho SapBERT (Chỉ chạy khi Recall@5 của BGE-M3 < 80%)
```bash
# Cho ICD-10
python src/retrieval/generate_embeddings.py --model SAPBERT --table icd10
python src/retrieval/build_faiss_index.py --model SAPBERT --table icd10

# Cho RxNorm
python src/retrieval/generate_embeddings.py --model SAPBERT --table rxnorm
python src/retrieval/build_faiss_index.py --model SAPBERT --table rxnorm
```

---

## 3. Phân tích Hệ thống & Chiến lược Tối ưu hóa CPU-only

### 3.1. Phương pháp Phân tích Phần cứng Hệ thống (System Profiling)
Để xác định phương án tối ưu hóa tính toán phù hợp nhất trên từng hệ thống mới, chúng ta sử dụng các câu lệnh PowerShell sau để kiểm tra cấu hình phần cứng:

1. **Kiểm tra thông tin bộ vi xử lý (CPU Cores & Threads):**
   Sử dụng lệnh truy vấn đối tượng CPU qua lớp WMI để xem số lõi vật lý thực (Cores) và luồng ảo (Logical Processors):
   ```powershell
   Get-CimInstance Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors
   ```
   *Mục đích:* Số lõi thực (`NumberOfCores`) chính là con số tối ưu để thiết lập cho `torch.set_num_threads()` (trên máy bạn là **8**).
2. **Kiểm tra card đồ họa (GPU Display Adapter):**
   Dùng lệnh kiểm tra thiết bị hiển thị của hệ thống để xác định có GPU NVIDIA hỗ trợ CUDA hay không:
   ```powershell
   Get-CimInstance Win32_VideoController | Select-Object Name
   ```
   *Mục đích:* Nếu kết quả trả về không chứa chữ `"NVIDIA"` (trên máy bạn là `AMD Radeon(TM) 860M`), PyTorch sẽ mặc định chạy trên CPU.
3. **Kiểm tra tài nguyên tiến trình tải và tính toán (Giám sát RAM/CPU thực tế):**
   Trong lúc chạy tác vụ nền, ta có thể theo dõi xem Python đang ngốn bao nhiêu RAM thực tế (Working Set - WS) và lượng tải CPU:
   ```powershell
   Get-Process -Name "python" | Select-Object Id, CPU, WS, Path
   ```
4. **Giám sát kết nối mạng của luồng tải (TCP Connections):**
   Để kiểm tra xem tiến trình Python có đang bị đứng kết nối mạng hay không, dùng lệnh tra cứu cổng mạng của PID Python tương ứng:
   ```powershell
   Get-NetTCPConnection -OwningProcess <Id_cua_Python> | Select-Object LocalAddress, LocalPort, RemoteAddress, RemotePort, State
   ```
   *Mục đích:* Giúp phát hiện nhanh các kết nối bị kẹt ở trạng thái `TimeWait` hoặc `Established` chậm của máy chủ CDN.

Do không có GPU hỗ trợ CUDA, chúng ta bắt buộc phải chạy các mô hình ngôn ngữ lớn (LLM/Embedding) trên CPU. Nếu chạy bằng cấu hình mặc định (vốn được tối ưu cho GPU), thời gian sinh embedding cho 362k RxNorm sẽ mất khoảng **19 - 24 giờ**. Bằng việc phân tích sâu cấu trúc CPU, chúng ta đã tối ưu hóa thời gian này xuống còn khoảng **7 - 10 giờ** (nhanh hơn gấp hơn 2 lần) nhờ 4 kỹ thuật cốt lõi dưới đây:

---

### 3.2. Bốn kỹ thuật tối ưu hóa CPU chuyên sâu

#### Kỹ thuật 1: Kích hoạt CPU BFloat16 Mixed Precision (Độ chính xác hỗn hợp)
* **Tại sao tối ưu:** Mặc định, PyTorch chạy trên CPU bằng kiểu dữ liệu số thực 32-bit (`Float32`). Bộ vi xử lý AMD Ryzen thế hệ mới (Zen 5) hỗ trợ tập lệnh **AVX-512 BF16** trực tiếp ở cấp độ phần cứng. Ép kiểu dữ liệu sang `BFloat16` giảm kích thước bộ nhớ của các ma trận trọng số đi một nửa (từ 4 bytes xuống 2 bytes) và cho phép CPU xử lý vector toán học song song tốc độ cao.
* **Cách thực hiện:** Sử dụng khối lệnh `torch.amp.autocast('cpu', dtype=torch.bfloat16)` bao bọc xung quanh hàm Forward/Encode của mô hình.
* **Hiệu quả thực tế:** Đo đạc thực tế trên máy của bạn cho thấy thời gian chạy giảm từ **3.29 giây** xuống **2.14 giây** (tốc độ xử lý tăng **35%**).

#### Kỹ thuật 2: Điều chỉnh số lượng CPU Threads tối ưu (8 Threads thay vì 16 Threads)
* **Tại sao tối ưu:** CPU của bạn có 8 nhân vật lý và 16 luồng ảo (Hyper-Threading). Trong tính toán khoa học và Deep Learning trên CPU, hai luồng ảo chung một nhân vật lý sẽ tranh chấp các khối thực thi toán học số thực (FPU - Floating Point Unit). Nếu đặt 16 threads, hiện tượng nghẽn luồng và tranh chấp tài nguyên sẽ xảy ra khiến CPU bị quá nhiệt và giảm xung nhịp.
* **Cách thực hiện:** Thiết lập `torch.set_num_threads(8)` (đúng bằng số nhân vật lý của chip).
* **Hiệu quả thực tế:** Tốc độ chạy 8 threads đạt **3.29s/batch**, nhanh hơn khi chạy full 16 threads (**3.47s/batch**) và giúp CPU mát hơn đáng kể.

#### Kỹ thuật 3: Tối ưu bộ nhớ đệm CPU Cache bằng cách giảm kích thước Batch (Batch Size = 64)
* **Tại sao tối ưu:** Trên GPU, batch_size càng lớn càng tốt (256, 512) để tận dụng hàng ngàn nhân CUDA song song. Tuy nhiên, trên CPU, việc đặt batch_size lớn (ví dụ 256) sẽ tạo ra các ma trận kích thước lớn vượt quá dung lượng bộ nhớ đệm L2/L3 của CPU (CPU Cache). Điều này dẫn đến hiện tượng **Cache Thrashing** (CPU liên tục phải xóa bộ nhớ đệm để đọc dữ liệu mới từ RAM hệ thống, vốn có băng thông rất hẹp so với cache). Giảm batch_size xuống 64 hoặc 32 giúp toàn bộ các tensor trung gian nằm trọn vẹn trong bộ nhớ đệm L3 của AMD Ryzen, giúp CPU xử lý với băng thông tối đa.
* **Cách thực hiện:** Đặt `batch_size = 64` nếu chạy trên CPU thay vì 256 của GPU.

#### Kỹ thuật 4: Giới hạn độ dài chuỗi tối đa (`max_seq_length = 128`)
* **Tại sao tối ưu:** Mô hình BGE-M3 có thiết lập mặc định hỗ trợ độ dài chuỗi cực đại lên đến 8192 tokens. Khi chạy SentenceTransformers, thư viện sẽ tự động padding các chuỗi trong batch về độ dài lớn nhất. Tên thực thể chẩn đoán ICD-10 và tên thuốc RxNorm thực tế rất ngắn (thường dưới 30 tokens). Nếu không giới hạn, mô hình sẽ tính toán attention vô ích trên các tokens đệm trống (padding tokens), lãng phí tài nguyên CPU theo hàm số mũ.
* **Cách thực hiện:** Gán thuộc tính `model.max_seq_length = 128` cho lớp SentenceTransformer trước khi encode.

---

## 4. Các Vấn đề và Lỗi phát sinh trong quá trình tải mô hình (Troubleshooting)

Trong quá trình xây dựng hệ thống tải mô hình y khoa offline này, chúng tôi đã gặp phải một số lỗi kỹ thuật liên quan đến mạng và thư viện. Dưới đây là cách chẩn đoán và xử lý chi tiết để bạn phòng tránh hoặc khắc phục khi triển khai trên hệ thống mới:

### Lỗi 1: Đường truyền bị nghẽn vô hạn do sử dụng Endpoint Mirror (`hf-mirror.com`)
- **Mô tả hiện tượng:** Ban đầu, để tối ưu tốc độ mạng tại Việt Nam, chúng tôi cấu hình endpoint chuyển tiếp `HF_ENDPOINT="https://hf-mirror.com"`. Tuy nhiên, kết nối này bị treo vô hạn ở trạng thái chờ (Timeout) và không thể bắt đầu tải.
- **Cách chẩn đoán:** Sử dụng lệnh kiểm tra phản hồi HTTP header:
  ```powershell
  curl -I https://hf-mirror.com
  ```
  Nếu lệnh bị treo hoặc không phản hồi dữ liệu trong vòng 5 giây, tức là mirror này đang bị nghẽn mạng nghiêm trọng.
- **Giải pháp:** Gỡ bỏ cấu hình biến môi trường mirror trong script, quay lại sử dụng trực tiếp máy chủ gốc `huggingface.co`. Thư viện HuggingFace sẽ tự động tối ưu hóa định tuyến tải qua mạng CDN toàn cầu.

### Lỗi 2: Lỗi hỏng cache (Cache Corruption) khi dừng tác vụ giữa chừng
- **Mô tả hiện tượng:** Nếu tiến trình tải mô hình đang diễn ra mà bị ngắt kết nối (hoặc do người dùng ép dừng tiến trình - `kill task`), các tệp tin lưu tạm dạng `.incomplete` trong cache của HuggingFace sẽ bị lỗi cấu trúc. Lần chạy tiếp theo, Python nạp mô hình từ cache này sẽ báo lỗi:
  `Pooling.__init__() missing 1 required positional argument: 'embedding_dimension'` hoặc lỗi thiếu tệp cấu hình `config.json`.
- **Cách chẩn đoán:** Kiểm tra thư mục cache mặc định của HuggingFace tại máy:
  `C:\Users\PC\.cache\huggingface\hub\models--BAAI--bge-m3`
  Nếu xuất hiện nhiều tệp `.incomplete` trong thư mục `blobs` nhưng chương trình Python báo lỗi không khởi tạo được mô hình, cache đã bị lỗi.
- **Giải pháp:** Xóa sạch thư mục cache lỗi của mô hình đó để bắt đầu lại:
  ```powershell
  Remove-Item -Recurse -Force "C:\Users\PC\.cache\huggingface\hub\models--BAAI--bge-m3"
  ```

### Lỗi 3: Thiếu thư viện giải mã Tokenizer (`sentencepiece`, `protobuf`, `tiktoken`)
- **Mô tả hiện tượng:** Sau khi tải thành công trọng số SapBERT, Python báo lỗi không thể nạp mô hình do trình tách từ của XLM-RoBERTa (nền tảng của SapBERT) sử dụng định dạng nén SentencePiece của Google.
- **Cách chẩn đoán:** Chương trình báo lỗi:
  `SentencePieceExtractor requires the SentencePiece library but it was not found` hoặc `SentencePieceExtractor requires the protobuf library`.
- **Giải pháp:** Cài đặt các thư viện hỗ trợ xử lý giao thức nhị phân của Google và bộ tách từ:
  ```bash
  pip install sentencepiece protobuf tiktoken
  ```
