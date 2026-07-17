import os
import sys
import numpy as np
import torch

# Thêm project root vào sys.path để hỗ trợ import chéo
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.retrieval.bm25_retriever import BM25Retriever

# Import faiss tùy chọn để chạy offline air-gapped
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

class HybridRetriever:
    # Class variables de cache model dung chung giua cac instance (tranh tich luy VRAM/RAM)
    _cached_bge_model = None
    _cached_sapbert_model = None
    _cached_sapbert_tokenizer = None

    def __init__(self, table_name="icd10", db_path=None, index_dir=None, w_bm25=1.0, w_faiss=0.5, k=60, embedding_model_type=None):
        self.table_name = table_name
        self.w_bm25 = w_bm25
        self.w_faiss = w_faiss
        self.k = k  # RRF constant k
        
        # Doc loai model tu tham so hoac bien moi truong
        if embedding_model_type is None:
            self.embedding_model_type = os.environ.get("EMBEDDING_MODEL_TYPE", "BGE-M3").upper()
        else:
            self.embedding_model_type = embedding_model_type.upper()
            
        # 1. Khoi tao BM25 Retriever
        bm25_kwargs = {}
        if db_path is not None:
            bm25_kwargs["db_path"] = db_path
        bm25_kwargs["table_name"] = table_name
        self.bm25_retriever = BM25Retriever(**bm25_kwargs)
        
        # 2. Khoi tao FAISS Retriever
        self.faiss_available = False
        self.faiss_index = None
        self.faiss_codes = []
        
        if FAISS_AVAILABLE:
            if index_dir is None:
                from src.utils.paths import KB_DIR
                # Tro vao thu muc index duoc xay dung rieng cho model nay va bang nay
                index_dir = os.path.join(KB_DIR, f"{table_name}_{self.embedding_model_type.lower()}_index")
                
            index_file = os.path.join(index_dir, "index.faiss")
            codes_file = os.path.join(index_dir, "codes.txt")
            
            if os.path.exists(index_file) and os.path.exists(codes_file):
                try:
                    self.faiss_index = faiss.read_index(index_file)
                    with open(codes_file, "r", encoding="utf-8") as f:
                        self.faiss_codes = [line.strip() for line in f if line.strip()]
                    self.faiss_available = True
                    print(f"HybridRetriever [{table_name}] ({self.embedding_model_type}): Loaded FAISS index from {index_dir}.")
                except Exception as e:
                    print(f"[WARNING] Failed to load FAISS index: {e}")
            else:
                # Tuong thich nguoc voi format dat ten cu "icd10_faiss" neu ton tai
                from src.utils.paths import KB_DIR
                fallback_dir = os.path.join(KB_DIR, f"{table_name}_faiss")
                fallback_index = os.path.join(fallback_dir, "index.faiss")
                fallback_codes = os.path.join(fallback_dir, "codes.txt")
                if os.path.exists(fallback_index) and os.path.exists(fallback_codes):
                    try:
                        self.faiss_index = faiss.read_index(fallback_index)
                        with open(fallback_codes, "r", encoding="utf-8") as f:
                            self.faiss_codes = [line.strip() for line in f if line.strip()]
                        self.faiss_available = True
                        print(f"HybridRetriever [{table_name}] (Fallback): Loaded FAISS index from {fallback_dir}.")
                    except Exception:
                        pass
                
                if not self.faiss_available:
                    print(f"HybridRetriever [{table_name}] ({self.embedding_model_type}): FAISS index files not found in {index_dir}. Fallback to BM25 only.")
        else:
            print(f"HybridRetriever [{table_name}]: 'faiss' library not installed. Fallback to BM25 only.")

    def _load_model(self):
        """Nap va cache model embedding de tiet kiem RAM/VRAM."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.embedding_model_type == "BGE-M3":
            if HybridRetriever._cached_bge_model is None:
                from sentence_transformers import SentenceTransformer
                from src.utils.paths import KB_DIR
                model_path = os.path.join(os.path.dirname(KB_DIR), "models", "bge-m3")
                if not os.path.exists(model_path):
                    print("[WARNING] Local BGE-M3 not found. Fallback online...")
                    model_path = "BAAI/bge-m3"
                HybridRetriever._cached_bge_model = SentenceTransformer(model_path, device=device)
            return HybridRetriever._cached_bge_model
            
        elif self.embedding_model_type == "SAPBERT":
            if HybridRetriever._cached_sapbert_model is None:
                from transformers import AutoModel, AutoTokenizer
                from src.utils.paths import KB_DIR
                model_path = os.path.join(os.path.dirname(KB_DIR), "models", "sapbert")
                if not os.path.exists(model_path):
                    print("[WARNING] Local SapBERT not found. Fallback online...")
                    model_path = "cambridgeltl/SapBERT-UMLS-2020AB-all-lang-from-XLMR"
                HybridRetriever._cached_sapbert_model = AutoModel.from_pretrained(model_path).to(device)
                HybridRetriever._cached_sapbert_tokenizer = AutoTokenizer.from_pretrained(model_path)
            return HybridRetriever._cached_sapbert_model, HybridRetriever._cached_sapbert_tokenizer
        else:
            raise ValueError(f"Khong ho tro model type: {self.embedding_model_type}")

    def retrieve(self, query, top_k=5, embedding_model=None):
        """
        Truy xuat ket qua lai (Hybrid Search) bang cach ket hop BM25 va FAISS qua RRF.
        Neu FAISS khong kha dung, tu dong fallback chi dung BM25.
        """
        if not query:
            return []
            
        # 1. Thuc hien BM25 Search
        bm25_results = self.bm25_retriever.retrieve(query, top_k=top_k * 3)
        
        # Neu FAISS khong san sang, dung luon ket qua BM25
        if not self.faiss_available:
            return bm25_results[:top_k]
            
        # 2. Thuc hien FAISS Search
        try:
            query_vector = None
            
            if self.embedding_model_type == "BGE-M3":
                if embedding_model is None:
                    model = self._load_model()
                else:
                    model = embedding_model
                # Sinh vector embedding cho query (BGE-M3)
                query_vector = model.encode([query])[0]
                
            elif self.embedding_model_type == "SAPBERT":
                if embedding_model is None:
                    model, tokenizer = self._load_model()
                else:
                    model, tokenizer = embedding_model
                
                inputs = tokenizer([query], padding=True, truncation=True, max_length=128, return_tensors="pt")
                inputs = {k: v.to(model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    # CLS pooling cho SapBERT
                    query_vector = outputs.last_hidden_state[0, 0, :].cpu().numpy()
                    
            if query_vector is not None:
                query_vector = np.array([query_vector]).astype("float32")
                # Chuon hoa L2 de dung FlatIP giong luc build index
                faiss.normalize_L2(query_vector)
                
                distances, indices = self.faiss_index.search(query_vector, top_k * 3)
                
                faiss_results = []
                seen_codes = set()
                for idx in indices[0]:
                    if 0 <= idx < len(self.faiss_codes):
                        code = self.faiss_codes[idx]
                        if code not in seen_codes:
                            seen_codes.add(code)
                            faiss_results.append(code)
            else:
                faiss_results = []
        except Exception as e:
            print(f"[ERROR] Failed to perform FAISS search: {e}")
            faiss_results = []
            
        # 3. Ket hop bang Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        
        # Rank BM25
        for rank, code in enumerate(bm25_results):
            if code not in rrf_scores:
                rrf_scores[code] = 0.0
            rrf_scores[code] += self.w_bm25 / (rank + self.k)
            
        # Rank FAISS
        for rank, code in enumerate(faiss_results):
            if code not in rrf_scores:
                rrf_scores[code] = 0.0
            rrf_scores[code] += self.w_faiss / (rank + self.k)
            
        # Sap xep va lay Top K
        sorted_codes = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Ap dung Hierarchical Fallback Rule cho ICD-10
        if self.table_name == "icd10":
            final_codes = []
            seen_in_final = set()
            
            # Duyet qua cac code trong sorted_codes de chèn các mã con thich hop
            for code in sorted_codes:
                if len(final_codes) >= top_k:
                    break
                if code not in seen_in_final:
                    final_codes.append(code)
                    seen_in_final.add(code)
                
                # Neu la ma cha 3 ky tu (vi du: K85, K74, H10)
                if len(code) == 3:
                    try:
                        import sqlite3
                        from src.utils.paths import DB_PATH
                        conn = sqlite3.connect(DB_PATH)
                        cursor = conn.cursor()
                        # Tim cac ma con co cham (vi du: K85.%)
                        cursor.execute(
                            "SELECT code, name_vi, name_en FROM icd10 WHERE code LIKE ?",
                            (f"{code}.%",)
                        )
                        children = cursor.fetchall()
                        conn.close()
                        
                        if children:
                            scored_children = []
                            for c_code, c_vi, c_en in children:
                                c_vi_lower = c_vi.lower() if c_vi else ""
                                c_en_lower = c_en.lower() if c_en else ""
                                
                                score = 0
                                if "không đặc hiệu" in c_vi_lower or "unspecified" in c_en_lower:
                                    score = 2
                                elif "khác" in c_vi_lower or "other" in c_en_lower:
                                    score = 1
                                    
                                scored_children.append((c_code, score))
                            
                            # Sap xep ma con theo diem giam dan
                            scored_children.sort(key=lambda x: x[1], reverse=True)
                            
                            # Chi chen toi da 2 ma con co diem > 0 (khong dac hieu hoac khac)
                            inserted_count = 0
                            for c_code, score in scored_children:
                                if score > 0:
                                    if len(final_codes) >= top_k:
                                        break
                                    if c_code not in seen_in_final:
                                        final_codes.append(c_code)
                                        seen_in_final.add(c_code)
                                        inserted_count += 1
                                        if inserted_count >= 2:
                                            break
                    except Exception as ex:
                        print(f"[WARNING] Failed to query children for fallback of {code}: {ex}")
            
            return final_codes[:top_k]
        elif self.table_name == "rxnorm":
            final_codes = []
            seen_in_final = set()
            
            for code in sorted_codes:
                if len(final_codes) >= top_k:
                    break
                if code not in seen_in_final:
                    final_codes.append(code)
                    seen_in_final.add(code)
                
                # Neu la ma SCDC hoac IN, ta tim cac SCD tuong ung
                try:
                    import sqlite3
                    from src.utils.paths import DB_PATH
                    conn = sqlite3.connect(DB_PATH)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name, tty FROM rxnorm WHERE rxcui = ? LIMIT 1", (code,))
                    row = cursor.fetchone()
                    
                    if row and row[1] in ("SCDC", "IN"):
                        name_val, tty_val = row[0], row[1]
                        # Tim cac ma SCD bat dau bang ten nay
                        cursor.execute(
                            "SELECT rxcui, name FROM rxnorm WHERE name LIKE ? AND tty = 'SCD'",
                            (f"{name_val} %",)
                        )
                        children = cursor.fetchall()
                        conn.close()
                        
                        if children:
                            scored_children = []
                            for c_cui, c_name in children:
                                c_name_lower = c_name.lower()
                                score = 0
                                if "oral tablet" in c_name_lower or "oral capsule" in c_name_lower:
                                    score = 2
                                elif "injection" in c_name_lower or "solution" in c_name_lower:
                                    score = 1
                                    
                                scored_children.append((c_cui, score))
                            
                            scored_children.sort(key=lambda x: x[1], reverse=True)
                            
                            inserted_count = 0
                            for c_cui, score in scored_children:
                                if score > 0:
                                    if len(final_codes) >= top_k:
                                        break
                                    if c_cui not in seen_in_final:
                                        final_codes.append(c_cui)
                                        seen_in_final.add(c_cui)
                                        inserted_count += 1
                                        if inserted_count >= 2:
                                            break
                    else:
                        conn.close()
                except Exception as ex:
                    print(f"[WARNING] Failed to query children for RxNorm fallback of {code}: {ex}")
            
            return final_codes[:top_k]
        else:
            return sorted_codes[:top_k]

if __name__ == "__main__":
    # Test thu ca 2 che do model type de bao dam khong co loi
    for model_name in ["BGE-M3", "SAPBERT"]:
        print(f"\n--- TESTING HYBRID RETRIEVER IN {model_name} MODE ---")
        retriever = HybridRetriever(table_name="icd10", embedding_model_type=model_name)
        res = retriever.retrieve("tăng huyết áp", top_k=5)
        print(f"HybridRetriever Test ({model_name}):", res)
        assert len(res) > 0
        print(f"HybridRetriever Test ({model_name}): Passed")
