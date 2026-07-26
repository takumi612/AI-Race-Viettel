from __future__ import annotations

from clinical_nlp_lab.inference import FinalModelBundle, InferenceConfig, SpanProposal, infer_document
from clinical_nlp_lab.runtime_bundle import KBFirstRecovery


class FakeNER:
    def propose(self, raw_text: str, config: InferenceConfig):
        start = raw_text.index("fever")
        return [SpanProposal("fever", "SYMPTOM", start, start + 5, 0.91, "ner")]


class FakeKB:
    def scan_raw_text(self, raw_text: str):
        start = raw_text.index("diabetes")
        return [
            SpanProposal(
                "diabetes",
                "DISEASE",
                start,
                start + 8,
                0.96,
                "kb_first",
                ranked_candidates=(
                    {"candidate_id": "E11", "score": 0.94},
                    {"candidate_id": "E13", "score": 0.20},
                ),
            )
        ]


class RankingKB:
    def scan_raw_text(self, raw_text: str):
        return []

    def rank_candidates(self, entity_type: str, text: str, limit: int = 20):
        assert entity_type == "DISEASE"
        assert text == "diabetes"
        assert limit == 20
        return [
            {"candidate_id": "E11", "score": 0.96},
            {"candidate_id": "E13", "score": 0.72},
        ]


class FakeAssertion:
    def predict(self, raw_text, entities):
        return {(entity.start, entity.end, entity.type): ["isNegated"] for entity in entities}


class FakePolicy:
    def apply(self, ranked):
        return [ranked[0]["candidate_id"]] if ranked else []


def test_inference_composes_ner_kb_assertion_and_candidate_policy():
    raw_text = "HS-0300: patient has diabetes and fever"
    bundle = FinalModelBundle(
        ner_model=FakeNER(),
        tokenizer=object(),
        assertion_model=FakeAssertion(),
        candidate_policy=FakePolicy(),
        kb_linker=FakeKB(),
    )

    document = infer_document("300", raw_text, bundle, InferenceConfig())

    assert [(entity.text, entity.type) for entity in document.entities] == [
        ("diabetes", "DISEASE"),
        ("fever", "SYMPTOM"),
    ]
    disease, symptom = document.entities
    assert disease.candidates == ["E11"]
    assert disease.assertions == ["isNegated"]
    assert symptom.assertions == ["isNegated"]


def test_ner_disease_receives_ranked_candidates_before_policy_and_qwen():
    class DiseaseNER:
        def propose(self, raw_text: str, config: InferenceConfig):
            start = raw_text.index("diabetes")
            return [
                SpanProposal(
                    "diabetes", "DISEASE", start, start + 8, 0.95, "ner"
                )
            ]

    document = infer_document(
        "302",
        "Patient has diabetes.",
        FinalModelBundle(
            ner_model=DiseaseNER(),
            tokenizer=object(),
            candidate_policy=FakePolicy(),
            kb_linker=RankingKB(),
        ),
        InferenceConfig(enable_kb_recovery=False),
    )

    assert document.entities[0].candidates == ["E11"]
    assert [
        item["candidate_id"] for item in document.entities[0].ranked_candidates
    ] == ["E11", "E13"]


def test_invalid_proposal_is_rejected_before_assertion_or_candidate_steps():
    class InvalidNER:
        def propose(self, raw_text: str, config: InferenceConfig):
            return [SpanProposal("wrong", "DISEASE", 0, 5, 0.9, "ner")]

    document = infer_document(
        "301",
        "HS-0301: patient has fever",
        FinalModelBundle(ner_model=InvalidNER(), tokenizer=object()),
        InferenceConfig(),
    )
    assert document.entities == []


def test_inference_low_numeric_id_without_organizer_headers_is_one_record():
    raw_text = "THIẾU MEN G6PD là gì?\n\n1. Thiếu men G6PD là bệnh gì?"

    document = infer_document(
        "1",
        raw_text,
        FinalModelBundle(ner_model=None, tokenizer=None),
        InferenceConfig(enable_kb_recovery=False),
    )

    assert document.document_id == "1"
    assert document.raw_text == raw_text
    assert document.entities == []


def test_kb_first_recovery_preserves_raw_offsets_for_aliases():
    recovery = KBFirstRecovery(
        [{"candidate_id": "E11", "aliases": ["diabetes"]}],
        [{"candidate_id": "RX1", "aliases": ["metformin"]}],
    )
    proposals = recovery.scan_raw_text("HS-0300: Diabetes and metformin")
    assert [(item.text, item.start, item.end, item.entity_type) for item in sorted(proposals, key=lambda item: item.start)] == [
        ("Diabetes", 9, 17, "DISEASE"),
        ("metformin", 22, 31, "DRUG"),
    ]
