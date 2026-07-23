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
