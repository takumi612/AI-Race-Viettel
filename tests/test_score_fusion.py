import sqlite3

import pytest

from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.score_fusion import fuse_candidates, minmax_scores
from src.retrieval.types import ComponentCandidate


def test_fusion_weights_sum_to_one_and_favor_bm25():
    bm25 = [ComponentCandidate("A", 10.0, 0), ComponentCandidate("B", 5.0, 1)]
    semantic = [
        ComponentCandidate("B", 0.95, 0),
        ComponentCandidate("C", 0.90, 1),
    ]

    fused = fuse_candidates(bm25, semantic, alpha=0.75)

    by_code = {item.code: item for item in fused}
    assert by_code["A"].fusion_score == pytest.approx(0.75)
    assert by_code["C"].fusion_score <= 0.25


def test_alpha_boundaries_are_exact_component_modes():
    bm25 = [ComponentCandidate("A", 2.0, 0)]
    semantic = [ComponentCandidate("B", 0.9, 0)]

    assert fuse_candidates(bm25, semantic, alpha=1.0)[0].code == "A"
    assert fuse_candidates(bm25, semantic, alpha=0.0)[0].code == "B"


def test_equal_component_scores_are_deterministic():
    values = [ComponentCandidate("B", 1.0, 0), ComponentCandidate("A", 1.0, 1)]

    assert minmax_scores(values) == {"B": 1.0, "A": 1.0}


def test_invalid_kb_codes_are_removed_before_fusion():
    bm25 = [
        ComponentCandidate("VALID", 2.0, 0),
        ComponentCandidate("MISSING", 1.0, 1),
    ]

    fused = fuse_candidates(bm25, [], alpha=0.75, valid_codes={"VALID"})

    assert [candidate.code for candidate in fused] == ["VALID"]


def test_fusion_normalizes_codes_and_filters_before_minmax():
    bm25 = [
        ComponentCandidate(" valid ", 2.0, 0),
        ComponentCandidate("missing", 10.0, 1),
    ]

    fused = fuse_candidates(bm25, [], alpha=0.75, valid_codes={"VALID"})

    assert fused[0].code == "VALID"
    assert fused[0].bm25_score == pytest.approx(1.0)
    assert fused[0].fusion_score == pytest.approx(0.75)


def test_fusion_ties_break_by_component_rank_then_code():
    bm25 = [ComponentCandidate("B", 1.0, 1), ComponentCandidate("A", 1.0, 0)]
    semantic = [ComponentCandidate("B", 1.0, 0), ComponentCandidate("A", 1.0, 1)]

    fused = fuse_candidates(bm25, semantic, alpha=0.5)

    assert [candidate.code for candidate in fused] == ["A", "B"]


@pytest.mark.parametrize("alpha", [-0.01, 1.01])
def test_alpha_outside_unit_interval_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        fuse_candidates([], [], alpha=alpha)


def test_bm25_retrieve_scored_exposes_raw_scores_and_keeps_code_wrapper():
    class FakeBM25:
        def retrieve(self, queries, k):
            assert queries == [["tăng", "huyết", "áp"]]
            assert k == 2
            return [[1, 0]], [[0.8, 0.6]]

    retriever = BM25Retriever.__new__(BM25Retriever)
    retriever.retriever = FakeBM25()
    retriever.codes = ["I10", "I11"]

    scored = retriever.retrieve_scored("tăng huyết áp", top_k=2)

    assert scored == [
        ComponentCandidate("I11", 0.8, 0),
        ComponentCandidate("I10", 0.6, 1),
    ]
    assert retriever.retrieve("tăng huyết áp", top_k=2) == ["I11", "I10"]


def test_hybrid_scored_retrieval_uses_top_twenty_and_falls_back_to_bm25():
    class StubBM25:
        codes = ["I10", "I11"]

        def __init__(self):
            self.requested_top_k = None

        def retrieve_scored(self, query, top_k):
            self.requested_top_k = top_k
            return [
                ComponentCandidate("I10", 2.0, 0),
                ComponentCandidate("I11", 1.0, 1),
            ]

    hybrid = HybridRetriever.__new__(HybridRetriever)
    hybrid.alpha = 0.75
    hybrid.internal_top_k = 20
    hybrid.bm25_retriever = StubBM25()
    hybrid.faiss_available = False

    scored = hybrid.retrieve_scored("tăng huyết áp", top_k=1)

    assert hybrid.bm25_retriever.requested_top_k == 20
    assert [candidate.code for candidate in scored] == ["I10"]
    assert hybrid.retrieve("tăng huyết áp", top_k=1) == ["I10"]


def test_hybrid_model_failure_uses_deterministic_bm25_only_fusion():
    class StubBM25:
        codes = ["I10", "I11"]

        def retrieve_scored(self, query, top_k):
            return [
                ComponentCandidate("I10", 2.0, 0),
                ComponentCandidate("I11", 1.0, 1),
            ]

    hybrid = HybridRetriever.__new__(HybridRetriever)
    hybrid.alpha = 0.75
    hybrid.internal_top_k = 20
    hybrid.bm25_retriever = StubBM25()
    hybrid.faiss_available = True
    hybrid.embedding_model_type = "BGE-M3"
    hybrid._load_model = lambda: (_ for _ in ()).throw(FileNotFoundError("offline"))

    scored = hybrid.retrieve_scored("tăng huyết áp", top_k=2)

    assert [candidate.code for candidate in scored] == ["I10", "I11"]
    assert [candidate.semantic_score for candidate in scored] == [0.0, 0.0]
    assert [candidate.fusion_score for candidate in scored] == [0.75, 0.0]


def test_hierarchical_expansion_is_opt_in_and_deterministic(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE icd10 (code TEXT, name_vi TEXT, name_en TEXT)")
        connection.executemany(
            "INSERT INTO icd10 VALUES (?, ?, ?)",
            [
                ("A00", "Cholera", "Cholera"),
                ("A00.1", "Cholera khác", "Other cholera"),
                ("A00.2", "Cholera không đặc hiệu", "Unspecified cholera"),
                ("A00.3", "Cholera cụ thể", "Specific cholera"),
            ],
        )

    class StubBM25:
        codes = ["A00"]

        def __init__(self):
            self.db_path = str(db_path)

        def retrieve_scored(self, query, top_k):
            return [ComponentCandidate("A00", 1.0, 0)]

    def build(expand):
        hybrid = HybridRetriever.__new__(HybridRetriever)
        hybrid.alpha = 0.75
        hybrid.internal_top_k = 20
        hybrid.hierarchical_expansion = expand
        hybrid.bm25_retriever = StubBM25()
        hybrid.faiss_available = False
        return hybrid

    assert [item.code for item in build(False).retrieve_scored("cholera", 3)] == ["A00"]
    assert [item.code for item in build(True).retrieve_scored("cholera", 3)] == [
        "A00",
        "A00.2",
        "A00.1",
    ]


def test_icd_hierarchy_expansion_is_opt_in_and_deterministic(tmp_path):
    database_path = tmp_path / "codes.db"
    connection = sqlite3.connect(database_path)
    connection.execute("CREATE TABLE icd10 (code TEXT, name_vi TEXT, name_en TEXT)")
    connection.executemany(
        "INSERT INTO icd10 VALUES (?, ?, ?)",
        [
            ("I10", "Tăng huyết áp", "Hypertension"),
            ("I10.1", "Tăng huyết áp thứ phát", "Secondary hypertension"),
            ("I10.0", "Tăng huyết áp nguyên phát", "Primary hypertension"),
            ("Z99", "Phụ thuộc thiết bị", "Dependence on enabling machines"),
        ],
    )
    connection.commit()
    connection.close()

    class StubBM25:
        codes = ["I10", "I10.0", "I10.1", "Z99"]
        db_path = str(database_path)

        def retrieve_scored(self, query, top_k):
            return [
                ComponentCandidate("I10", 2.0, 0),
                ComponentCandidate("Z99", 1.0, 1),
            ]

    def hybrid(hierarchical_expansion):
        retriever = HybridRetriever.__new__(HybridRetriever)
        retriever.alpha = 0.75
        retriever.internal_top_k = 20
        retriever.table_name = "icd10"
        retriever.hierarchical_expansion = hierarchical_expansion
        retriever.bm25_retriever = StubBM25()
        retriever.faiss_available = False
        return retriever

    assert hybrid(False).retrieve("tăng huyết áp", top_k=3) == ["I10", "Z99"]
    assert hybrid(True).retrieve("tăng huyết áp", top_k=3) == [
        "I10",
        "I10.0",
        "I10.1",
    ]
