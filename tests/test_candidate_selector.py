import json
import sqlite3

import pytest

from src.ranking.llm_reranker import LLMReranker
from src.retrieval.candidate_selector import CandidateSelector
from src.retrieval.types import RetrievedCandidate
from src.validation.clinical_validator import ClinicalValidator, dose_form_is_compatible


def candidate(code: str, score: float, rank: int = 0) -> RetrievedCandidate:
    return RetrievedCandidate(code, score, score, 0.0, rank, None)


def test_selector_returns_top1_for_clear_margin():
    ranked = [candidate("A", 0.90), candidate("B", 0.60, 1)]
    assert CandidateSelector().select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A"]


def test_selector_returns_top2_only_for_close_valid_scores():
    ranked = [candidate("A", 0.80), candidate("B", 0.78, 1), candidate("C", 0.50, 2)]
    assert CandidateSelector().select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A", "B"]


def test_selector_rejects_low_or_unknown_and_validates_before_threshold():
    calls = []
    ranked = [candidate("LOW", 0.10), candidate("GOOD", 0.80, 1)]
    result = CandidateSelector().select("THUỐC", ranked, lambda code: calls.append(code) or code == "GOOD")
    assert result == ["GOOD"]
    assert calls == ["LOW", "GOOD"]
    assert CandidateSelector().select("TRIỆU_CHỨNG", ranked, lambda _: True) == []


def test_selector_accepts_mapping_config_and_caps_output():
    selector = CandidateSelector({"selection": {"icd_min_score": 0.1, "top1_margin": 1.0, "top2_margin": 0.0}})
    ranked = [candidate("A", 0.8), candidate("B", 0.79, 1), candidate("C", 0.78, 2)]
    assert selector.select("CHẨN_ĐOÁN", ranked, lambda _: True) == ["A"]


def test_dual_code_check_never_creates_new_entity(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE icd10_rules_sex (code TEXT, allowed_sex TEXT)")
        connection.execute("CREATE TABLE icd10_rules_age (code TEXT, min_days INT, max_days INT, description TEXT)")
        connection.execute("CREATE TABLE icd10_rules_dual (dagger_code TEXT, asterisk_code TEXT)")
        connection.execute("CREATE TABLE icd10_rules_not_primary (code TEXT)")
    validator = ClinicalValidator(db_path=str(db_path))
    entities = [{"text": "gout", "type": "CHẨN_ĐOÁN", "position": [0, 4], "assertions": [], "candidates": ["M10.9"]}]
    assert validator.check_dual_codes(entities) is entities
    assert len(entities) == 1


def test_historical_rxnorm_mapping_is_opt_in(tmp_path):
    db_path = tmp_path / "metadata.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE icd10_rules_sex (code TEXT, allowed_sex TEXT)")
        connection.execute("CREATE TABLE icd10_rules_age (code TEXT, min_days INT, max_days INT, description TEXT)")
        connection.execute("CREATE TABLE icd10_rules_dual (dagger_code TEXT, asterisk_code TEXT)")
        connection.execute("CREATE TABLE icd10_rules_not_primary (code TEXT)")
        connection.execute("CREATE TABLE rxnorm_mapping (old_cui TEXT, new_cui TEXT)")
        connection.execute("INSERT INTO rxnorm_mapping VALUES ('old', 'new')")
    assert ClinicalValidator(db_path=str(db_path), load_historical_rxnorm=False).rxnorm_mapping == {}
    assert ClinicalValidator(db_path=str(db_path), load_historical_rxnorm=True).rxnorm_mapping == {"new": ["old"]}


def test_dose_form_validation_uses_supplied_rules():
    rules = {"route_groups": [{"name": "custom", "mention_terms": ["đường zeta"], "rxnorm_terms": ["zeta form"]}]}
    assert dose_form_is_compatible("thuốc đường zeta", "ingredient zeta form", rules)
    assert not dose_form_is_compatible("thuốc đường zeta", "ingredient oral tablet", rules)


def test_llm_reranker_accepts_only_a_subset_of_input_candidates():
    assert LLMReranker.parse_selected_codes('{"selected_codes": ["A"]}', ["A", "B"]) == ["A"]
    assert LLMReranker.parse_selected_codes('{"best_code": "B"}', ["A", "B"]) == ["B"]
    with pytest.raises(ValueError, match="candidate pool"):
        LLMReranker.parse_selected_codes('{"selected_codes": ["FOREIGN"]}', ["A", "B"])
    with pytest.raises(ValueError, match="at most two"):
        LLMReranker.parse_selected_codes(
            '{"selected_codes": ["A", "B", "C"]}',
            ["A", "B", "C"],
        )


def test_local_llm_reranker_is_lazy_and_falls_back_deterministically():
    class LocalBackend:
        def __init__(self, output=None, error=None):
            self.output = output
            self.error = error

        def generate(self, example):
            if self.error is not None:
                raise self.error
            return self.output

    reranker = LLMReranker(
        use_llm=True,
        backend="local_transformers",
        model_artifact="artifacts/reranker",
    )
    reranker._local_backend = LocalBackend('{"selected_codes":["B"]}')
    assert reranker.rerank(
        "context", "entity", "CHẨN_ĐOÁN", ["A", "B"]
    ) == ["B"]

    reranker._local_backend = LocalBackend(error=RuntimeError("GPU OOM"))
    assert reranker.rerank(
        "context", "entity", "CHẨN_ĐOÁN", ["A", "B"]
    ) == ["A", "B"]
