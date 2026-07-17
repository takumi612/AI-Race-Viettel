import os
import sys
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Thêm project root vào sys.path để hỗ trợ import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import KB_DIR

# Định nghĩa đường dẫn lưu model cục bộ
MODELS_DIR = os.path.join(os.path.dirname(KB_DIR), "models")
BGE_M3_LOCAL_PATH = os.path.join(MODELS_DIR, "bge-m3")
SAPBERT_LOCAL_PATH = os.path.join(MODELS_DIR, "sapbert")

def download_and_save_models():
    # Sử dụng chuỗi không dấu để tránh hoàn toàn UnicodeEncodeError trên terminal Windows
    print("=== BAT DAU TAI VA LUU MO HINH OFFLINE ===")
    
    # 1. Tải và lưu mô hình BGE-M3
    print("\n1. Dang tai mo hinh BGE-M3 (BAAI/bge-m3) tu HuggingFace...")
    try:
        if not os.path.exists(BGE_M3_LOCAL_PATH):
            os.makedirs(BGE_M3_LOCAL_PATH)
        # Tải model về RAM
        model = SentenceTransformer("BAAI/bge-m3")
        # Lưu ra ổ đĩa local
        print(f"   Dang luu mo hinh BGE-M3 ve: {BGE_M3_LOCAL_PATH}...")
        model.save(BGE_M3_LOCAL_PATH)
        print("   -> Tai va luu BGE-M3 thanh cong!")
    except Exception as e:
        print(f"   [ERROR] Khong the tai BGE-M3: {e}")
        
    # 2. Tải và lưu mô hình SapBERT-XLMR bản base (768 chiều)
    sapbert_repo = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
    print(f"\n2. Dang tai mo hinh SapBERT ({sapbert_repo}) tu HuggingFace...")
    try:
        if not os.path.exists(SAPBERT_LOCAL_PATH):
            os.makedirs(SAPBERT_LOCAL_PATH)
        
        # Tải model và tokenizer bằng AutoModel/AutoTokenizer để lấy chính xác CLS token sau này
        print("   Dang tai model...")
        model_sap = AutoModel.from_pretrained(sapbert_repo)
        print("   Dang tai tokenizer...")
        tokenizer_sap = AutoTokenizer.from_pretrained(sapbert_repo)
        
        # Lưu ra ổ đĩa local
        print(f"   Dang luu mo hinh SapBERT ve: {SAPBERT_LOCAL_PATH}...")
        model_sap.save_pretrained(SAPBERT_LOCAL_PATH)
        tokenizer_sap.save_pretrained(SAPBERT_LOCAL_PATH)
        print("   -> Tai va luu SapBERT thanh cong!")
    except Exception as e:
        print(f"   [ERROR] Khong the tai SapBERT: {e}")

    print("\n=== HOAN THANH TAI MO HINH OFFLINE ===")

if __name__ == "__main__":
    download_and_save_models()
