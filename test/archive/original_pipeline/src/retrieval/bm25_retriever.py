import os
import re
import sqlite3
import sys

import bm25s


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.retrieval.types import ComponentCandidate
from src.utils.paths import DB_PATH


class BM25Retriever:
    def __init__(self, db_path=DB_PATH, table_name="icd10"):
        self.db_path = db_path
        self.table_name = table_name
        self.load_corpus()

    def tokenize(self, text):
        """Tokenize Vietnamese and English text for BM25."""
        if not text:
            return []
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return text.split()

    def load_corpus(self):
        """Load the configured knowledge-base table and build its BM25 index."""
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
                description = f"{name_vi} {name_en}".strip()
                self.codes.append(code)
                self.descriptions.append(description)
                corpus_tokens.append(self.tokenize(description))
            print(f"Loaded {len(self.codes)} ICD-10 codes for BM25.")
        elif self.table_name == "rxnorm":
            cursor.execute("SELECT rxcui, name FROM rxnorm;")
            rows = cursor.fetchall()
            for rxcui, name in rows:
                description = name.strip()
                self.codes.append(rxcui)
                self.descriptions.append(description)
                corpus_tokens.append(self.tokenize(description))
            print(f"Loaded {len(self.codes)} RxNorm codes for BM25.")
        else:
            print(f"[ERROR] Unsupported table name: {self.table_name}")
            conn.close()
            self.retriever = None
            return

        conn.close()
        self.retriever = bm25s.BM25()
        self.retriever.index(corpus_tokens)

    def retrieve_scored(self, query, top_k=5):
        """Return BM25 candidates with their raw retrieval scores and ranks."""
        if not self.retriever or not query or top_k < 1:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        results, scores = self.retriever.retrieve([query_tokens], k=top_k)
        candidates = []
        for rank, (index, score) in enumerate(zip(results[0], scores[0])):
            index = int(index)
            if 0 <= index < len(self.codes):
                candidates.append(
                    ComponentCandidate(
                        code=str(self.codes[index]), score=float(score), rank=rank
                    )
                )
        return candidates

    def retrieve(self, query, top_k=5):
        """Return BM25 candidate codes for existing callers."""
        return [item.code for item in self.retrieve_scored(query, top_k=top_k)]


if __name__ == "__main__":
    retriever = BM25Retriever()
    print("Query 'tăng huyết áp':", retriever.retrieve("tăng huyết áp", 3))
