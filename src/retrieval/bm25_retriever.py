import sqlite3
import os
import sys
import re
import bm25s

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import DB_PATH

class BM25Retriever:
    def __init__(self, db_path=DB_PATH, table_name="icd10"):
        self.db_path = db_path
        self.table_name = table_name
        self.load_corpus()
        
    def tokenize(self, text):
        """Hàm tokenize đơn giản cho tiếng Việt."""
        if not text:
            return []
        text = text.lower()
        # Xóa các ký tự đặc biệt
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def load_corpus(self):
        """Tải danh sách từ SQLite và huấn luyện BM25."""
        if not os.path.exists(self.db_path):
            print(f"[WARNING] Database not found at {self.db_path}. BM25 cannot search.")
            self.codes = []
            self.descriptions = []
            self.retriever = None
            return

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        self.codes = []
        self.descriptions = []
        corpus_tokens = []

        if self.table_name == "icd10":
            cursor.execute("SELECT code, name_vi, name_en FROM icd10;")
            rows = cursor.fetchall()
            for code, name_vi, name_en in rows:
                desc = f"{name_vi} {name_en}".strip()
                self.codes.append(code)
                self.descriptions.append(desc)
                corpus_tokens.append(self.tokenize(desc))
            print(f"Loaded {len(self.codes)} ICD-10 codes for BM25.")
        elif self.table_name == "rxnorm":
            cursor.execute("SELECT rxcui, name FROM rxnorm;")
            rows = cursor.fetchall()
            for rxcui, name in rows:
                desc = name.strip()
                self.codes.append(rxcui)
                self.descriptions.append(desc)
                corpus_tokens.append(self.tokenize(desc))
            print(f"Loaded {len(self.codes)} RxNorm codes for BM25.")
        else:
            print(f"[ERROR] Unsupported table name: {self.table_name}")
            conn.close()
            self.retriever = None
            return

        conn.close()
        
        # Huấn luyện BM25S
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def retrieve(self, query, top_k=5):
        """Truy vấn tìm kiếm Top-K mã ICD-10."""
        if not self.retriever or not query:
            return []
            
        query_tokens = self.tokenize(query)
        # Thực hiện tìm kiếm (nhận đầu vào là danh sách các truy vấn, ta bọc trong list)
        results, scores = self.retriever.retrieve([query_tokens], k=top_k)
        
        candidates = []
        # results là numpy array, lấy dòng đầu tiên của kết quả query
        for idx in results[0]:
            code = self.codes[idx]
            candidates.append(code)
            
        return candidates

if __name__ == "__main__":
    retriever = BM25Retriever()
    print("Query 'tăng huyết áp':", retriever.retrieve("tăng huyết áp", 3))
