import os
import sys
import platform
import sqlite3

def check_environment():
    print("=================== ENVIRONMENT CHECK ===================")
    
    # 1. OS & Python
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python Version: {sys.version}")
    
    # 2. Check Libraries
    libs = {
        "numpy": "Numpy (Matrix operations)",
        "openpyxl": "Openpyxl (Reading ICD-10 Excel)",
        "sqlite3": "Sqlite3 (Embedded Database)",
        "bm25s": "BM25S (Fast Lexical Retrieval)",
        "faiss": "FAISS (Vector Similarity Search)",
        "torch": "PyTorch (Inference/Model backbone)"
    }
    
    print("\n[Library Check]")
    for lib_name, desc in libs.items():
        try:
            __import__(lib_name)
            print(f"  - {lib_name:<10}: INSTALLED ({desc})")
        except ImportError:
            print(f"  - {lib_name:<10}: NOT INSTALLED !!! ({desc})")
            
    # 3. Hardware & GPU check
    print("\n[Hardware & GPU Check]")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"  - CUDA Available : {cuda_avail}")
        if cuda_avail:
            print(f"  - CUDA Device Qty: {torch.cuda.device_count()}")
            print(f"  - Current GPU    : {torch.cuda.get_device_name(0)}")
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"  - Total VRAM     : {vram_gb:.2f} GB")
        else:
            print("  - [WARNING] CUDA is not available. System will run on CPU fallback mode.")
    except Exception as e:
        print(f"  - GPU check skipped or failed: {e}")
        
    # 4. Database check
    print("\n[Database Check]")
    from src.utils.paths import DB_PATH
    print(f"  - DB Path: {DB_PATH}")
    if os.path.exists(DB_PATH):
        print("  - File Status: EXISTS")
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            
            # Count ICD-10
            cursor.execute("SELECT COUNT(*) FROM icd10;")
            icd_cnt = cursor.fetchone()[0]
            print(f"  - ICD-10 Records  : {icd_cnt}")
            
            # Count RxNorm
            cursor.execute("SELECT COUNT(*) FROM rxnorm;")
            rx_cnt = cursor.fetchone()[0]
            print(f"  - RxNorm Records  : {rx_cnt}")
            
            # Count mapping
            cursor.execute("SELECT COUNT(*) FROM rxnorm_mapping;")
            map_cnt = cursor.fetchone()[0]
            print(f"  - Historical Maps : {map_cnt}")
            
            conn.close()
        except Exception as e:
            print(f"  - Error reading database: {e}")
    else:
        print("  - [ERROR] Database file does not exist! Please run setup scripts first.")
        
    print("\n=========================================================")

if __name__ == "__main__":
    # Add project root to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    check_environment()
