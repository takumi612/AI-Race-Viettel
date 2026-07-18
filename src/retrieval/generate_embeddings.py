import os
import sys
import sqlite3
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

# Thêm project root vào sys.path để hỗ trợ import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import DB_PATH, KB_DIR

# Thiet lap ma hoa tieng Viet khong dau tren console de tranh UnicodeEncodeError
import io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Dinh nghia cac duong dan model va data
MODELS_DIR = os.path.join(os.path.dirname(KB_DIR), "models")
BGE_M3_LOCAL_PATH = os.path.join(MODELS_DIR, "bge-m3")
SAPBERT_LOCAL_PATH = os.path.join(MODELS_DIR, "sapbert")

def load_data(table_name, limit=None, db_path=DB_PATH):
    """Doc du lieu tu CSDL SQLite va tra ve list cac chuoi chu lam sach kem ma code tuong ung."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"CSDL khong ton tai tai: {db_path}")
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    codes = []
    texts = []
    
    limit_clause = f" LIMIT {limit}" if limit is not None else ""
    
    if table_name == "icd10":
        cursor.execute(f"SELECT code, name_vi, name_en FROM icd10{limit_clause};")
        rows = cursor.fetchall()
        for code, name_vi, name_en in rows:
            c = code.strip()
            vi = (name_vi or "").strip()
            en = (name_en or "").strip()
            if vi:
                codes.append(c)
                texts.append(vi)
            if en:
                codes.append(c)
                texts.append(en)
    elif table_name == "rxnorm":
        cursor.execute(f"SELECT rxcui, name FROM rxnorm{limit_clause};")
        rows = cursor.fetchall()
        for rxcui, name in rows:
            codes.append(rxcui.strip())
            texts.append(name.strip())
    else:
        conn.close()
        raise ValueError(f"Khong ho tro bang: {table_name}")
        
    conn.close()
    return codes, texts

def generate_embeddings(
    model_type,
    table_name,
    limit=None,
    model_path=None,
    output_dir=None,
    db_path=DB_PATH,
    batch_size=None,
    cpu_threads=None,
):
    """Sinh vector embedding va luu ra file numpy (.npy) + file ma (.txt)."""
    print(f"\n=== BAT DAU SINH EMBEDDING: Model={model_type}, Table={table_name}, Limit={limit} ===")
    
    # 1. Load du lieu
    codes, texts = load_data(table_name, limit=limit, db_path=db_path)
    print(f"Loaded {len(texts)} records from table {table_name}.")
    
    if not texts:
        print("[WARNING] Khong co du lieu de sinh embedding.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device dang su dung: {device}")
    if batch_size is not None and batch_size < 1:
        raise ValueError("batch_size must be positive")
    if cpu_threads is not None and cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if device == "cpu" and cpu_threads is not None:
        torch.set_num_threads(cpu_threads)
        print(f"   Set PyTorch threads = {cpu_threads}")
        
    # 2. Logic sinh vector
    embeddings = None
    
    if model_type == "BGE-M3":
        resolved_model_path = model_path or BGE_M3_LOCAL_PATH
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f"Model BGE-M3 khong tim thay tai: {resolved_model_path}")
            
        print("Loading BGE-M3 local model...")
        model = SentenceTransformer(
            resolved_model_path,
            device=device,
            local_files_only=True,
        )
        # Gioi han max_seq_length vi ten ICD-10 va RxNorm rat ngan (tranh padding attention thua)
        model.max_seq_length = 128
        
        # CPU cache optimization
        resolved_batch_size = batch_size or (16 if device == "cpu" else 32)
        print(
            "Generating embeddings using SentenceTransformer "
            f"(batch_size={resolved_batch_size})..."
        )
        
        if device == "cpu":
            print("   Using CPU BFloat16 Mixed Precision...")
            with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                embeddings = model.encode(
                    texts,
                    batch_size=resolved_batch_size,
                    show_progress_bar=True,
                )
        else:
            embeddings = model.encode(
                texts,
                batch_size=resolved_batch_size,
                show_progress_bar=True,
            )
        
    elif model_type == "SAPBERT":
        resolved_model_path = model_path or SAPBERT_LOCAL_PATH
        if not os.path.exists(resolved_model_path):
            raise FileNotFoundError(f"Model SapBERT khong tim thay tai: {resolved_model_path}")
            
        print("Loading SapBERT local model and tokenizer...")
        model = AutoModel.from_pretrained(
            resolved_model_path,
            local_files_only=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            local_files_only=True,
        )
        
        model.to(device)
        model.eval()
        
        resolved_batch_size = batch_size or (16 if device == "cpu" else 32)
        print(
            "Generating embeddings using Transformers "
            f"(CLS pooling, batch_size={resolved_batch_size})..."
        )
        embeddings_list = []
        
        # Batch inference cho SapBERT dung CLS pooling
        for i in range(0, len(texts), resolved_batch_size):
            batch_texts = texts[i : i + resolved_batch_size]
            # Tokenize batch
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            )
            # Day inputs vao GPU/CPU
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                if device == "cpu":
                    with torch.amp.autocast('cpu', dtype=torch.bfloat16):
                        outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                # Lay vector tu CLS token (token dau tien) theo khuyen nghi cua tac gia SapBERT
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings_list.append(cls_embeddings)
                
            if (i // resolved_batch_size) % 20 == 0:
                total_batches = len(texts) // resolved_batch_size + (
                    1 if len(texts) % resolved_batch_size != 0 else 0
                )
                print(
                    "   Processed batch "
                    f"{i // resolved_batch_size + 1}/{total_batches} - "
                    f"{min(i + resolved_batch_size, len(texts))}/{len(texts)} "
                    "records...",
                    flush=True,
                )
                
        embeddings = np.vstack(embeddings_list)
    else:
        raise ValueError(f"Khong ho tro loai model: {model_type}")
        
    # 3. Luu ket qua trung gian
    npy_filename = f"{table_name}_{model_type.lower()}_embeddings.npy"
    txt_filename = f"{table_name}_{model_type.lower()}_codes.txt"
    
    resolved_output_dir = output_dir or KB_DIR
    os.makedirs(resolved_output_dir, exist_ok=True)
    npy_path = os.path.join(resolved_output_dir, npy_filename)
    txt_path = os.path.join(resolved_output_dir, txt_filename)
    
    print(f"Luu vector matrix (shape={embeddings.shape}) vao: {npy_path}")
    np.save(npy_path, embeddings)
    
    print(f"Luu codes text list vao: {txt_path}")
    with open(txt_path, "w", encoding="utf-8") as f:
        for code in codes:
            f.write(f"{code}\n")
            
    print(f"=== HOAN THANH SINH EMBEDDING {model_type} CHO BANG {table_name} ===")
    return {
        "embeddings": npy_path,
        "codes": txt_path,
        "count": len(codes),
        "dimension": int(embeddings.shape[1]),
        "model_path": resolved_model_path,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sinh Embedding y khoa offline")
    parser.add_argument("--model", type=str, default="BGE-M3", choices=["BGE-M3", "SAPBERT"], help="Loai model su dung")
    parser.add_argument("--table", type=str, default="icd10", choices=["icd10", "rxnorm"], help="Ten bang du lieu can sinh")
    parser.add_argument("--limit", type=int, default=None, help="Gioi han so ban ghi chay thu (de kiem tra nhanh)")
    parser.add_argument("--model-path", type=str, default=None, help="Model/adaptor SentenceTransformer da train")
    parser.add_argument("--output-dir", type=str, default=None, help="Thu muc ghi embeddings/codes")
    parser.add_argument("--db", type=str, default=DB_PATH, help="metadata.db")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--cpu-threads", type=int, default=None)
    args = parser.parse_args()
    
    generate_embeddings(
        model_type=args.model,
        table_name=args.table,
        limit=args.limit,
        model_path=args.model_path,
        output_dir=args.output_dir,
        db_path=args.db,
        batch_size=args.batch_size,
        cpu_threads=args.cpu_threads,
    )
