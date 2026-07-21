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

