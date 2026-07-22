from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
package = types.ModuleType("clinical_nlp_lab")
package.__path__ = [str(ROOT / "clinical_nlp_lab")]
sys.modules.setdefault("clinical_nlp_lab", package)
for module_name in ("bm25s", "faiss", "sentence_transformers"):
    sys.modules.setdefault(module_name, types.ModuleType(module_name))
sys.modules["sentence_transformers"].SentenceTransformer = object

from clinical_nlp_lab.candidate_policy import CandidatePolicy, apply_candidate_policy
from clinical_nlp_lab.retrieval import HybridEntityLinker


def test_candidate_output_k_is_enforced_without_qwen():
    policy = CandidatePolicy(min_score=0.0, min_margin=0.0, output_k=1)
    ranked = [
        {"candidate_id": "A", "score": 0.9},
        {"candidate_id": "B", "score": 0.8},
    ]

    assert apply_candidate_policy(ranked, policy) == ["A"]


def test_candidate_policy_abstains_below_threshold():
    policy = CandidatePolicy(min_score=0.5, min_margin=0.0, output_k=1)

    assert apply_candidate_policy([{"candidate_id": "A", "score": 0.2}], policy) == []


def test_candidate_policy_abstains_when_top_margin_is_ambiguous():
    policy = CandidatePolicy(min_score=0.5, min_margin=0.1, output_k=1)
    ranked = [
        {"candidate_id": "A", "score": 0.72},
        {"candidate_id": "B", "score": 0.68},
    ]

    assert apply_candidate_policy(ranked, policy) == []


def test_drug_retrieval_uses_mention_head_without_dose_attributes():
    class FakeIndex:
        def __init__(self):
            self.last_query = None

        def retrieve(self, query, top_k):
            self.last_query = query
            return [{"candidate_id": "6809", "score": 1.0, "name": "metformin"}]

    icd = FakeIndex()
    rx = FakeIndex()
    linker = HybridEntityLinker(icd, rx, top_k=20)

    candidate_ids, _ranked = linker.retrieve(
        "DRUG",
        "metformin 500 mg uống ngày 2 lần",
        mention_head="metformin",
    )

    assert candidate_ids == ["6809"]
    assert rx.last_query == "metformin"
