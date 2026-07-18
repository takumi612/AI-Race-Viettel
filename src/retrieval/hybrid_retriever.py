import os
import sqlite3
import sys

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import RetrievalConfig
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.score_fusion import fuse_candidates
from src.retrieval.types import ComponentCandidate, RetrievedCandidate


try:
    import faiss

    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class HybridRetriever:
    """Fuse normalized BM25 and local semantic retrieval scores."""

    _cached_bge_model = None
    _cached_sapbert_model = None
    _cached_sapbert_tokenizer = None

    def __init__(
        self,
        table_name="icd10",
        db_path=None,
        index_dir=None,
        alpha=0.75,
        internal_top_k=20,
        hierarchical_expansion=False,
        embedding_model_type=None,
    ):
        self.table_name = table_name
        config = RetrievalConfig(
            alpha=alpha,
            internal_top_k=internal_top_k,
            hierarchical_expansion=hierarchical_expansion,
        )
        self.alpha = config.alpha
        self.internal_top_k = config.internal_top_k
        self.hierarchical_expansion = config.hierarchical_expansion
        self.embedding_model_type = (
            embedding_model_type or os.environ.get("EMBEDDING_MODEL_TYPE", "BGE-M3")
        ).upper()

        bm25_kwargs = {"table_name": table_name}
        if db_path is not None:
            bm25_kwargs["db_path"] = db_path
        self.bm25_retriever = BM25Retriever(**bm25_kwargs)

        self.faiss_available = False
        self.faiss_index = None
        self.faiss_codes = []
        self._load_faiss_index(index_dir)

    def _load_faiss_index(self, index_dir):
        if not FAISS_AVAILABLE:
            print(
                f"HybridRetriever [{self.table_name}]: faiss is unavailable; "
                "using BM25 only."
            )
            return

        from src.utils.paths import KB_DIR

        primary_dir = index_dir or os.path.join(
            KB_DIR, f"{self.table_name}_{self.embedding_model_type.lower()}_index"
        )
        fallback_dir = os.path.join(KB_DIR, f"{self.table_name}_faiss")
        index_locations = [primary_dir]
        if fallback_dir != primary_dir:
            index_locations.append(fallback_dir)

        for location in index_locations:
            index_file = os.path.join(location, "index.faiss")
            codes_file = os.path.join(location, "codes.txt")
            if not (os.path.exists(index_file) and os.path.exists(codes_file)):
                continue
            try:
                self.faiss_index = faiss.read_index(index_file)
                with open(codes_file, "r", encoding="utf-8") as file:
                    self.faiss_codes = [line.strip() for line in file if line.strip()]
                self.faiss_available = True
                print(
                    f"HybridRetriever [{self.table_name}] ({self.embedding_model_type}): "
                    f"loaded FAISS index from {location}."
                )
                return
            except Exception as error:
                print(f"[WARNING] Failed to load FAISS index at {location}: {error}")

        print(
            f"HybridRetriever [{self.table_name}] ({self.embedding_model_type}): "
            "FAISS index files not found; using BM25 only."
        )

    def _load_model(self):
        """Load a local embedding model only; never attempt a network download."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        from src.utils.paths import KB_DIR

        models_dir = os.path.join(os.path.dirname(KB_DIR), "models")
        if self.embedding_model_type == "BGE-M3":
            if HybridRetriever._cached_bge_model is None:
                model_path = os.path.join(models_dir, "bge-m3")
                if not os.path.isdir(model_path):
                    raise FileNotFoundError(f"Local BGE-M3 model not found at {model_path}")
                from sentence_transformers import SentenceTransformer

                HybridRetriever._cached_bge_model = SentenceTransformer(
                    model_path, device=device, local_files_only=True
                )
            return HybridRetriever._cached_bge_model

        if self.embedding_model_type == "SAPBERT":
            if HybridRetriever._cached_sapbert_model is None:
                model_path = os.path.join(models_dir, "sapbert")
                if not os.path.isdir(model_path):
                    raise FileNotFoundError(f"Local SapBERT model not found at {model_path}")
                from transformers import AutoModel, AutoTokenizer

                HybridRetriever._cached_sapbert_model = AutoModel.from_pretrained(
                    model_path, local_files_only=True
                ).to(device)
                HybridRetriever._cached_sapbert_tokenizer = AutoTokenizer.from_pretrained(
                    model_path, local_files_only=True
                )
            return HybridRetriever._cached_sapbert_model, HybridRetriever._cached_sapbert_tokenizer

        raise ValueError(f"Unsupported embedding model type: {self.embedding_model_type}")

    def _semantic_candidates(self, query, top_k, embedding_model=None):
        if not self.faiss_available:
            return []

        try:
            if self.embedding_model_type == "BGE-M3":
                model = embedding_model or self._load_model()
                query_vector = model.encode([query])[0]
            elif self.embedding_model_type == "SAPBERT":
                model, tokenizer = embedding_model or self._load_model()
                inputs = tokenizer(
                    [query],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt",
                )
                inputs = {key: value.to(model.device) for key, value in inputs.items()}
                with torch.no_grad():
                    query_vector = model(**inputs).last_hidden_state[0, 0, :].cpu().numpy()
            else:
                return []

            query_vector = np.array([query_vector], dtype="float32")
            faiss.normalize_L2(query_vector)
            distances, indices = self.faiss_index.search(query_vector, top_k)
        except Exception as error:
            print(f"[WARNING] Semantic retrieval unavailable; using BM25 only: {error}")
            return []

        candidates = []
        seen_codes = set()
        for score, index in zip(distances[0], indices[0]):
            index = int(index)
            if not 0 <= index < len(self.faiss_codes):
                continue
            code = str(self.faiss_codes[index]).strip().upper()
            if not code or code in seen_codes:
                continue
            seen_codes.add(code)
            candidates.append(
                ComponentCandidate(code=code, score=float(score), rank=len(candidates))
            )
        return candidates

    def retrieve_scored(self, query, top_k=None, embedding_model=None):
        """Return normalized weighted candidates, slicing only after fusion."""
        if not query or (top_k is not None and top_k < 1):
            return []

        bm25_candidates = self.bm25_retriever.retrieve_scored(
            query, top_k=self.internal_top_k
        )
        semantic_candidates = self._semantic_candidates(
            query, top_k=self.internal_top_k, embedding_model=embedding_model
        )
        candidates = fuse_candidates(
            bm25_candidates,
            semantic_candidates,
            alpha=self.alpha,
            valid_codes=self.bm25_retriever.codes,
        )
        candidates = self._apply_hierarchical_expansion(candidates)
        return candidates if top_k is None else candidates[:top_k]

    def _hierarchical_children(self, code):
        """Return deterministic, explicitly eligible children for one parent code."""
        if not getattr(self, "hierarchical_expansion", False):
            return []

        table_name = getattr(self, "table_name", "icd10")
        database_path = getattr(self.bm25_retriever, "db_path", None)
        if not database_path:
            return []

        normalized_code = str(code).strip().upper()
        if not normalized_code:
            return []

        try:
            with sqlite3.connect(database_path) as connection:
                if table_name == "icd10" and len(normalized_code) == 3:
                    rows = connection.execute(
                        "SELECT code, name_vi, name_en "
                        "FROM icd10 WHERE code LIKE ?",
                        (f"{normalized_code}.%",),
                    ).fetchall()
                    ranked = []
                    for child_code, name_vi, name_en in rows:
                        child = str(child_code or "").strip().upper()
                        if not child:
                            continue
                        text = f"{name_vi or ''} {name_en or ''}".casefold()
                        priority = 0
                        if "không đặc hiệu" in text or "unspecified" in text:
                            priority = 2
                        elif "khác" in text or "other" in text:
                            priority = 1
                        ranked.append((priority, child))
                    preferred = [item for item in ranked if item[0] > 0]
                    selected = preferred or ranked
                    return [child for _, child in sorted(selected, key=lambda item: (-item[0], item[1]))[:2]]

                if table_name == "rxnorm":
                    parent = connection.execute(
                        "SELECT name, tty FROM rxnorm WHERE rxcui = ? LIMIT 1",
                        (normalized_code,),
                    ).fetchone()
                    if not parent or parent[1] not in {"SCDC", "IN"}:
                        return []
                    rows = connection.execute(
                        "SELECT rxcui FROM rxnorm "
                        "WHERE tty = 'SCD' AND name LIKE ? ORDER BY rxcui ASC",
                        (f"{parent[0]} %",),
                    ).fetchall()
                    return sorted({str(child[0]).strip().upper() for child in rows if child[0]})[:2]
        except (OSError, sqlite3.Error) as error:
            print(f"[WARNING] Hierarchy expansion unavailable: {error}")
        return []

    def _apply_hierarchical_expansion(self, candidates):
        """Insert opt-in hierarchy children after their fused parent candidates."""
        if not getattr(self, "hierarchical_expansion", False):
            return candidates

        expanded = []
        seen_codes = set()
        for candidate in candidates:
            code = candidate.code
            if code in seen_codes:
                continue
            expanded.append(candidate)
            seen_codes.add(code)
            for child_code in self._hierarchical_children(code):
                if child_code in seen_codes:
                    continue
                expanded.append(
                    RetrievedCandidate(
                        code=child_code,
                        fusion_score=0.0,
                        bm25_score=0.0,
                        semantic_score=0.0,
                        bm25_rank=None,
                        semantic_rank=None,
                    )
                )
                seen_codes.add(child_code)
        return expanded

    def retrieve(self, query, top_k=5):
        """Compatibility wrapper that returns codes instead of scored candidates."""
        return [candidate.code for candidate in self.retrieve_scored(query, top_k=top_k)]


if __name__ == "__main__":
    retriever = HybridRetriever(table_name="icd10")
    print(retriever.retrieve("tăng huyết áp", top_k=5))
