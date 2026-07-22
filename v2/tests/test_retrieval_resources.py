import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_retrieval():
    package = types.ModuleType("clinical_nlp_lab")
    package.__path__ = [str(ROOT / "clinical_nlp_lab")]
    sys.modules["clinical_nlp_lab"] = package
    for module_name in ("bm25s", "faiss", "sentence_transformers"):
        sys.modules.setdefault(module_name, types.ModuleType(module_name))
    fake_st = sys.modules["sentence_transformers"]
    fake_st.SentenceTransformer = object
    text_spec = importlib.util.spec_from_file_location("clinical_nlp_lab.text", ROOT / "clinical_nlp_lab" / "text.py")
    text_module = importlib.util.module_from_spec(text_spec)
    sys.modules["clinical_nlp_lab.text"] = text_module
    text_spec.loader.exec_module(text_module)
    spec = importlib.util.spec_from_file_location("clinical_nlp_lab.retrieval", ROOT / "clinical_nlp_lab" / "retrieval.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["clinical_nlp_lab.retrieval"] = module
    spec.loader.exec_module(module)
    return module


def test_candidate_index_accepts_shared_encoder_and_releases_resources():
    retrieval = _load_retrieval()
    encoder = object()
    index = retrieval.HybridCandidateIndex(
        [{"candidate_id": "I10", "aliases": ["tăng huyết áp"]}],
        "ICD-10",
        embedding_model=encoder,
    )
    index.bm25_retriever = object()
    index.faiss_index = object()

    assert index.embedding_model is encoder
    index.release()
    assert index.embedding_model is None
    assert index.bm25_retriever is None
    assert index.faiss_index is None


def test_hybrid_index_skips_bm25_when_tokenizer_returns_no_terms(monkeypatch):
    retrieval = _load_retrieval()

    class FakeBM25:
        def retrieve(self, *_args, **_kwargs):
            raise AssertionError("BM25 must not receive an empty token query")

    class FakeEncoder:
        def encode(self, _queries, **_kwargs):
            return [[1.0, 0.0]]

    class FakeFaiss:
        def search(self, _embeddings, _limit):
            return [[0.9]], [[0]]

    monkeypatch.setattr(retrieval.bm25s, "tokenize", lambda _queries: [[]], raising=False)
    index = retrieval.HybridCandidateIndex.__new__(retrieval.HybridCandidateIndex)
    index.is_built = True
    index.bm25_retriever = FakeBM25()
    index.embedding_model = FakeEncoder()
    index.faiss_index = FakeFaiss()
    index.corpus_texts = ["placeholder"]
    index.corpus_ids = ["I10"]
    index.records = {"I10": {"canonical_name": "Tăng huyết áp"}}

    ranked = index.retrieve("đ")

    assert ranked[0]["candidate_id"] == "I10"


def test_hybrid_index_has_lexical_fallback_without_embedding_model():
    retrieval = _load_retrieval()
    index = retrieval.HybridCandidateIndex(
        [{"candidate_id": "I10", "canonical_name": "tang huyet ap", "aliases": ["tang huyet ap"]}],
        "ICD-10",
        embedding_model=None,
    )
    index.build_indexes()
    ranked = index.retrieve("tang huyet ap")
    assert ranked and ranked[0]["candidate_id"] == "I10"
    assert ranked[0]["method"] in {"lexical_fallback", "hybrid_rrf"}
