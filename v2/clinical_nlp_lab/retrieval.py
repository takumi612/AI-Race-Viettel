from __future__ import annotations

import logging
import re
from typing import Any, Iterable

import numpy as np

# Các thư viện này cần được cài đặt trong môi trường Kaggle
try:
    import bm25s
except ImportError:
    bm25s = None

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .text import normalize_alias
from .linking import parse_medication_attributes
from .kb_contract import KBContractError, candidate_identity, official_output_id


def _ranked_candidate_payload(
    record: dict[str, Any],
    *,
    score: float,
    raw_score: float,
    method: str,
    requested_display_id: str | None = None,
    canonical_id: str | None = None,
) -> dict[str, Any]:
    identity_record = record
    if not record.get("canonical_id") and not record.get("candidate_id") and canonical_id:
        identity_record = {**record, "candidate_id": canonical_id, "canonical_id": canonical_id}
    identity = official_output_id(identity_record, requested_display_id=requested_display_id)
    return {
        "candidate_id": identity.official_display_id,
        "canonical_id": identity.canonical_id,
        "official_display_id": identity.official_display_id,
        "score": score,
        "raw_score": raw_score,
        "method": method,
        "name": record.get("canonical_name") or record.get("name_vi") or record.get("name_en") or "",
    }


def create_embedding_model(model_name: str):
    if SentenceTransformer is None:
        logging.warning("sentence-transformers is unavailable; using lexical-only retrieval")
        return None
    try:
        return SentenceTransformer(model_name, device="cpu")
    except Exception as exc:
        logging.warning("Could not load embedding model %s (%s); using lexical-only retrieval", model_name, exc)
        return None


class HybridCandidateIndex:
    """
    Kết hợp BM25s (từ vựng) và FAISS (ngữ nghĩa) sử dụng RRF (Reciprocal Rank Fusion).
    """

    def __init__(
        self,
        records: Iterable[dict[str, Any]],
        name: str,
        embedding_model_name: str = "BAAI/bge-m3",
        embedding_model: Any | None = None,
    ) -> None:
        self.name = name
        self.records: dict[str, dict[str, Any]] = {}
        self.corpus_texts: list[str] = []
        self.corpus_ids: list[str] = []

        # Parse records
        for record in records:
            candidate_id = str(record["candidate_id"])
            self.records[candidate_id] = record
            aliases = list(record.get("aliases") or [])
            if not aliases:
                fallback = record.get("canonical_name") or record.get("name_vi") or record.get("name_en")
                aliases = [fallback] if fallback else []
            for alias in aliases:
                normalized = normalize_alias(str(alias))
                if normalized:
                    self.corpus_texts.append(normalized)
                    self.corpus_ids.append(candidate_id)

        self.is_built = False
        self.bm25_retriever = None
        self.faiss_index = None
        self.embedding_model = embedding_model
        self.embedding_model_name = embedding_model_name
        self.lexical_only = False

    def build_indexes(self) -> None:
        """Xây dựng index cho BM25s và FAISS"""
        if not self.corpus_texts:
            logging.warning(f"[{self.name}] No data to build index.")
            return

        logging.info(f"[{self.name}] Building BM25s index with {len(self.corpus_texts)} records...")
        if bm25s is not None and hasattr(bm25s, "tokenize"):
            corpus_tokens = bm25s.tokenize(self.corpus_texts)
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(corpus_tokens)

        logging.info(f"[{self.name}] Building FAISS index using model {self.embedding_model_name}...")
        if faiss is None or self.embedding_model is None:
            self.lexical_only = True
            self.is_built = True
            logging.info("[%s] Semantic model unavailable; lexical-only index is active", self.name)
            return

        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device="cpu")
        # Encode with batch processing
        embeddings = self.embedding_model.encode(
            self.corpus_texts, 
            batch_size=128, 
            show_progress_bar=False, 
            normalize_embeddings=True
        )
        
        # IndexFlatIP cho cosine similarity (vì vector đã được normalize)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))

        self.is_built = True
        logging.info(f"[{self.name}] Indexes built successfully.")

    def release(self) -> None:
        """Release index and encoder references before loading the LLM."""
        self.bm25_retriever = None
        self.faiss_index = None
        self.embedding_model = None
        self.is_built = False

    def retrieve(self, query: str, top_k: int = 10, k_rrf: int = 60, w_bm25: float = 0.6, w_faiss: float = 0.4) -> list[dict[str, Any]]:
        if not self.is_built:
            raise RuntimeError("You must call build_indexes() before retrieve().")
            
        normalized_query = normalize_alias(query)
        if not normalized_query:
            return []

        # 1. TÌM KIẾM BM25s (Lexical)
        query_tokens = bm25s.tokenize([normalized_query])
        # Lấy top_k * 5 để fusion
        results_bm25 = [[]]
        if any(len(tokens) for tokens in query_tokens):
            results_bm25, _scores_bm25 = self.bm25_retriever.retrieve(
                query_tokens, k=min(top_k * 5, len(self.corpus_texts))
            )
        
        bm25_rank = {}
        # results_bm25[0] là array các index của corpus
        for rank, corpus_idx in enumerate(results_bm25[0]):
            candidate_id = self.corpus_ids[corpus_idx]
            # Giữ rank tốt nhất nếu 1 candidate có nhiều alias
            if candidate_id not in bm25_rank:
                bm25_rank[candidate_id] = rank + 1

        # 2. TÌM KIẾM FAISS (Semantic)
        query_embedding = self.embedding_model.encode([normalized_query], normalize_embeddings=True)
        scores_faiss, results_faiss = self.faiss_index.search(np.array(query_embedding, dtype=np.float32), min(top_k * 5, len(self.corpus_texts)))
        
        faiss_rank = {}
        for rank, corpus_idx in enumerate(results_faiss[0]):
            if corpus_idx != -1:
                candidate_id = self.corpus_ids[corpus_idx]
                if candidate_id not in faiss_rank:
                    faiss_rank[candidate_id] = rank + 1

        # 3. RRF FUSION
        all_candidates = set(bm25_rank.keys()) | set(faiss_rank.keys())
        scored: dict[str, float] = {}
        
        for candidate_id in all_candidates:
            score = 0.0
            if candidate_id in bm25_rank:
                score += w_bm25 / (bm25_rank[candidate_id] + k_rrf)
            if candidate_id in faiss_rank:
                score += w_faiss / (faiss_rank[candidate_id] + k_rrf)
            scored[candidate_id] = score

        # Xếp hạng
        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)[:top_k]
        max_score = ranked[0][1] if ranked else 0.0

        return [
            _ranked_candidate_payload(
                self.records[candidate_id],
                score=round(score / max_score, 6) if max_score else 0.0,
                raw_score=round(score, 6),
                method="hybrid_rrf",
                canonical_id=candidate_id,
            )
            for candidate_id, score in ranked
        ]


class _LegacyHybridEntityLinker:
    """
    Kết nối Entity tới ICD-10 và RxNorm dùng Hybrid Retrieval.
    """
    def __init__(self, icd10_index: HybridCandidateIndex, rxnorm_index: HybridCandidateIndex, top_k: int = 10):
        self.icd10_index = icd10_index
        self.rxnorm_index = rxnorm_index
        self.top_k = top_k

    def retrieve(self, entity_type: str, text: str) -> tuple[list[str], list[dict[str, Any]]]:
        if entity_type == "DISEASE":
            index = self.icd10_index
            query = text
        elif entity_type == "DRUG":
            index = self.rxnorm_index
            # Cơ bản: loại bỏ strength, route để tìm tên hoạt chất nếu cần (có thể tái sử dụng parse_medication_attributes của linking.py)
            query = text
        else:
            return [], []

        ranked = index.retrieve(query, top_k=self.top_k)
        for item in ranked:
            item.setdefault("canonical_id", str(item.get("candidate_id", "")))
            item.setdefault("official_display_id", str(item.get("candidate_id", "")))
        candidate_ids = [item["candidate_id"] for item in ranked]
        
        return candidate_ids, ranked


# The public linker is intentionally defined again at the end of this legacy module so
# older serialized imports remain compatible while the runtime uses the stricter API.
class HybridEntityLinker:
    """Link disease/drug mentions and retain a retrieval pool for reranking."""

    def __init__(self, icd10_index: HybridCandidateIndex, rxnorm_index: HybridCandidateIndex, top_k: int = 10):
        self.icd10_index = icd10_index
        self.rxnorm_index = rxnorm_index
        self.top_k = top_k

    def retrieve(
        self,
        entity_type: str,
        text: str,
        mention_head: str | None = None,
        existing_candidates: Iterable[str] | None = None,
    ) -> tuple[list[str], list[dict[str, Any]]]:
        if entity_type == "DISEASE":
            index = self.icd10_index
            query = mention_head or text
        elif entity_type == "DRUG":
            index = self.rxnorm_index
            parsed = parse_medication_attributes(text)
            query = mention_head or str(parsed["drug_name"])
        else:
            return [], []

        ranked = index.retrieve(query, top_k=self.top_k)
        for item in ranked:
            item.setdefault("canonical_id", str(item.get("candidate_id", "")))
            item.setdefault("official_display_id", str(item.get("candidate_id", "")))
        exact_items: list[dict[str, Any]] = []
        ontology = "icd10" if entity_type == "DISEASE" else "rxnorm"
        for requested_id in existing_candidates or ():
            try:
                identity = candidate_identity(ontology, str(requested_id))
            except KBContractError:
                continue
            if identity.canonical_id not in index.records:
                continue
            record = index.records[identity.canonical_id]
            try:
                exact_items.append(
                    _ranked_candidate_payload(
                        record,
                        score=1.0,
                        raw_score=1.0,
                        method="detector_exact_phrase",
                        requested_display_id=identity.official_display_id,
                    )
                )
            except KBContractError:
                continue
        existing_ids = {item["canonical_id"] for item in exact_items}
        ranked = (exact_items + [item for item in ranked if item["canonical_id"] not in existing_ids])[: self.top_k]
        return [item["candidate_id"] for item in ranked], ranked


# Runtime-safe index wrapper.  Semantic embeddings improve recall when
# available, but they are not allowed to prevent a valid lexical submission
# when Hugging Face/network access is unavailable.
_LegacyHybridCandidateIndex = HybridCandidateIndex


class HybridCandidateIndex(_LegacyHybridCandidateIndex):
    def build_indexes(self) -> None:
        if self.embedding_model is not None:
            try:
                return super().build_indexes()
            except Exception as exc:
                logging.warning("Falling back to lexical retrieval for %s: %s", self.name, exc)
                self.faiss_index = None
        if bm25s is not None and hasattr(bm25s, "tokenize"):
            corpus_tokens = bm25s.tokenize(self.corpus_texts)
            self.bm25_retriever = bm25s.BM25()
            self.bm25_retriever.index(corpus_tokens)
        self.lexical_only = True
        self.is_built = True

    def retrieve(self, query: str, top_k: int = 10, k_rrf: int = 60, w_bm25: float = 0.6, w_faiss: float = 0.4):
        if not getattr(self, "lexical_only", False):
            return super().retrieve(query, top_k, k_rrf, w_bm25, w_faiss)
        normalized_query = normalize_alias(query)
        if not normalized_query:
            return []
        if self.bm25_retriever is not None and bm25s is not None and hasattr(bm25s, "tokenize"):
            query_tokens = bm25s.tokenize([normalized_query])
            results, _ = self.bm25_retriever.retrieve(query_tokens, k=min(top_k * 5, len(self.corpus_texts)))
            indices = list(results[0])
        else:
            terms = set(normalized_query.split())
            scored = [(len(terms & set(text.split())), idx) for idx, text in enumerate(self.corpus_texts)]
            indices = [idx for score, idx in sorted(scored, reverse=True) if score][:top_k * 5]
        seen = set()
        output = []
        for rank, idx in enumerate(indices):
            candidate_id = self.corpus_ids[idx]
            if candidate_id in seen:
                continue
            seen.add(candidate_id)
            output.append(
                _ranked_candidate_payload(
                    self.records[candidate_id],
                    score=round(1.0 / (rank + 1), 6),
                    raw_score=round(1.0 / (rank + 1), 6),
                    method="lexical_fallback",
                    canonical_id=candidate_id,
                )
            )
            if len(output) >= top_k:
                break
        return output
