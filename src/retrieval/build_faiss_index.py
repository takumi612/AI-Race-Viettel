import os
import sys
import numpy as np
import faiss

# Thêm project root vào sys.path để hỗ trợ import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import KB_DIR

# Thiet lap ma hoa tieng Viet khong dau de tranh UnicodeEncodeError tren Windows
import io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def build_faiss_index(model_type, table_name):
    """Xay dung faiss index FlatIP tu ma tran vector embedding da sinh."""
    print(f"\n=== BAT DAU XAY DUNG FAISS INDEX: Model={model_type}, Table={table_name} ===")
    
    # 1. Xac dinh file nguon
    npy_filename = f"{table_name}_{model_type.lower()}_embeddings.npy"
    txt_filename = f"{table_name}_{model_type.lower()}_codes.txt"
    
    npy_path = os.path.join(KB_DIR, npy_filename)
    txt_path = os.path.join(KB_DIR, txt_filename)
    
    if not os.path.exists(npy_path) or not os.path.exists(txt_path):
        raise FileNotFoundError(
            f"Khong tim thay file embedding nguon. Can chay generate_embeddings.py truoc.\n"
            f"   Npy: {npy_path}\n   Txt: {txt_path}"
        )
        
    # 2. Load du lieu vector va ma code
    print("Loading embeddings...")
    embeddings = np.load(npy_path).astype("float32")
    
    print("Loading codes list...")
    with open(txt_path, "r", encoding="utf-8") as f:
        codes = [line.strip() for line in f if line.strip()]
        
    assert len(embeddings) == len(codes), (
        f"Mismatch giua so luong vector ({len(embeddings)}) va so luong ma code ({len(codes)})!"
    )
    
    # 3. Chuon hoa L2 (L2 Normalization)
    print("Normalizing vectors (L2)...")
    faiss.normalize_L2(embeddings)
    
    # 4. Xay dung index FlatIP
    dimension = embeddings.shape[1]
    print(f"Initializing FAISS IndexFlatIP. Dimension={dimension}, Vectors={len(embeddings)}")
    index = faiss.IndexFlatIP(dimension)
    
    print("Adding vectors to index...")
    index.add(embeddings)
    
    # 5. Luu index va codes vao thu muc dich tuong ung
    # Vi du: data/kb/icd10_bge-m3_index hoac data/kb/icd10_sapbert_index
    output_dir_name = f"{table_name}_{model_type.lower()}_index"
    output_dir = os.path.join(KB_DIR, output_dir_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    index_output_path = os.path.join(output_dir, "index.faiss")
    codes_output_path = os.path.join(output_dir, "codes.txt")
    
    print(f"Writing FAISS index to: {index_output_path}...")
    faiss.write_index(index, index_output_path)
    
    print(f"Writing codes map to: {codes_output_path}...")
    with open(codes_output_path, "w", encoding="utf-8") as f:
        for code in codes:
            f.write(f"{code}\n")
            
    print(f"=== XAY DUNG FAISS INDEX HOAN THANH VA LUU TAI: {output_dir} ===")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xay dung FAISS index tu offline embeddings")
    parser.add_argument("--model", type=str, default="BGE-M3", choices=["BGE-M3", "SAPBERT"], help="Loai model embedding")
    parser.add_argument("--table", type=str, default="icd10", choices=["icd10", "rxnorm"], help="Ten bang du lieu")
    args = parser.parse_args()
    
    build_faiss_index(model_type=args.model, table_name=args.table)
