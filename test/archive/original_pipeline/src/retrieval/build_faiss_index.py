import os
import shutil
import sys

import faiss
import numpy as np

# Thêm project root vào sys.path để hỗ trợ import paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import KB_DIR
from src.training.embedding.index_manifest import write_index_manifest

# Thiet lap ma hoa tieng Viet khong dau de tranh UnicodeEncodeError tren Windows
import io
if sys.platform.startswith("win"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def build_faiss_index(
    model_type,
    table_name,
    embedding_dir=None,
    output_dir=None,
    adapter_dir=None,
    base_model=None,
    db_path=None,
):
    """Xay dung faiss index FlatIP tu ma tran vector embedding da sinh."""
    print(f"\n=== BAT DAU XAY DUNG FAISS INDEX: Model={model_type}, Table={table_name} ===")
    
    # 1. Xac dinh file nguon
    npy_filename = f"{table_name}_{model_type.lower()}_embeddings.npy"
    txt_filename = f"{table_name}_{model_type.lower()}_codes.txt"
    
    resolved_embedding_dir = embedding_dir or KB_DIR
    npy_path = os.path.join(resolved_embedding_dir, npy_filename)
    txt_path = os.path.join(resolved_embedding_dir, txt_filename)
    
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
        
    if len(embeddings) != len(codes):
        raise ValueError(
            "Mismatch giua so luong vector "
            f"({len(embeddings)}) va so luong ma code ({len(codes)})!"
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
    output_dir = output_dir or os.path.join(KB_DIR, output_dir_name)
    
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

    embeddings_output_path = os.path.join(output_dir, "embeddings.npy")
    if os.path.abspath(npy_path) != os.path.abspath(embeddings_output_path):
        shutil.copy2(npy_path, embeddings_output_path)

    manifest_path = None
    if adapter_dir is not None:
        if not base_model or not db_path:
            raise ValueError(
                "base_model and db_path are required when adapter_dir is supplied"
            )
        manifest_path = write_index_manifest(
            output_dir,
            base_model=base_model,
            adapter_dir=adapter_dir,
            database=db_path,
            embeddings=embeddings_output_path,
            index=index_output_path,
            codes=codes_output_path,
            count=len(codes),
            dimension=dimension,
        )
            
    print(f"=== XAY DUNG FAISS INDEX HOAN THANH VA LUU TAI: {output_dir} ===")
    return {
        "index_dir": output_dir,
        "index": index_output_path,
        "codes": codes_output_path,
        "embeddings": embeddings_output_path,
        "manifest": str(manifest_path) if manifest_path else None,
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Xay dung FAISS index tu offline embeddings")
    parser.add_argument("--model", type=str, default="BGE-M3", choices=["BGE-M3", "SAPBERT"], help="Loai model embedding")
    parser.add_argument("--table", type=str, default="icd10", choices=["icd10", "rxnorm"], help="Ten bang du lieu")
    parser.add_argument("--embedding-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--adapter-dir", type=str, default=None)
    parser.add_argument("--base-model", type=str, default=None)
    parser.add_argument("--db", type=str, default=None)
    args = parser.parse_args()
    
    build_faiss_index(
        model_type=args.model,
        table_name=args.table,
        embedding_dir=args.embedding_dir,
        output_dir=args.output_dir,
        adapter_dir=args.adapter_dir,
        base_model=args.base_model,
        db_path=args.db,
    )
