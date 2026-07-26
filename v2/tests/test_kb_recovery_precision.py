from __future__ import annotations

import pytest

from clinical_nlp_lab.reranker import ClinicalLLMReranker
from clinical_nlp_lab.runtime_bundle import KBFirstRecovery


@pytest.mark.parametrize("alias", ["và", "tin", "phẫu", "bệnh", "không"])
def test_generic_or_short_single_token_alias_never_creates_entity(alias):
    recovery = KBFirstRecovery(
        [{"candidate_id": "I00", "aliases": [alias]}],
        [],
    )

    assert recovery.scan_raw_text(f"Người bệnh có {alias} trong câu.") == []


def test_unique_clinical_alias_preserves_raw_offset_and_candidate_pool():
    recovery = KBFirstRecovery(
        [{"candidate_id": "I10", "aliases": ["tăng huyết áp"]}],
        [],
    )
    raw_text = "Theo dõi Tăng huyết áp nguyên phát."

    proposals = recovery.scan_raw_text(raw_text)

    assert len(proposals) == 1
    assert proposals[0].text == "Tăng huyết áp"
    assert (proposals[0].start, proposals[0].end) == (
        raw_text.index("Tăng"),
        raw_text.index("Tăng") + len("Tăng huyết áp"),
    )
    assert [item["candidate_id"] for item in proposals[0].ranked_candidates] == [
        "I10"
    ]


@pytest.mark.parametrize("alias", ["glucose", "creatinine", "protein", "prothrombin"])
def test_lab_analyte_alias_does_not_create_drug_entity(alias):
    recovery = KBFirstRecovery(
        [],
        [{"candidate_id": "RX-LAB", "aliases": [alias]}],
    )

    assert recovery.scan_raw_text(f"Xét nghiệm {alias} trong máu.") == []


def test_ambiguous_alias_produces_one_ranked_pool_instead_of_competing_spans():
    recovery = KBFirstRecovery(
        [
            {"candidate_id": "I10", "aliases": ["tăng huyết áp"]},
            {"candidate_id": "I15", "aliases": ["tăng huyết áp"]},
        ],
        [],
    )

    proposals = recovery.scan_raw_text("Chẩn đoán tăng huyết áp.")

    assert len(proposals) == 1
    assert [item["candidate_id"] for item in proposals[0].ranked_candidates] == [
        "I10",
        "I15",
    ]


def test_kb_candidate_pool_is_directly_renderable_by_qwen_prompt():
    recovery = KBFirstRecovery(
        [{"candidate_id": "I10", "canonical_name": "Hypertension", "aliases": ["tăng huyết áp"]}],
        [],
    )
    proposal = recovery.scan_raw_text("Chẩn đoán tăng huyết áp.")[0]

    prompt = ClinicalLLMReranker.__new__(ClinicalLLMReranker)._build_prompt(
        "Chẩn đoán tăng huyết áp.",
        proposal.text,
        proposal.entity_type,
        list(proposal.ranked_candidates),
    )

    assert "ID: I10 | Name: Hypertension" in prompt


def test_ner_mentions_can_retrieve_ranked_candidates_without_creating_new_span():
    recovery = KBFirstRecovery(
        [{"candidate_id": "E11", "aliases": ["diabetes", "diabetes mellitus"]}],
        [{"candidate_id": "RX1", "aliases": ["metformin"]}],
    )

    ranked = recovery.rank_candidates("DISEASE", "Diabetes", limit=20)

    assert ranked[0]["candidate_id"] == "E11"
    assert ranked[0]["score"] == 1.0
    assert all(item["candidate_id"] != "RX1" for item in ranked)
