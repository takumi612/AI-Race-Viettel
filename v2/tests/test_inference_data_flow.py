from __future__ import annotations

from dataclasses import replace

from clinical_nlp_lab.records import ClinicalRecord
from clinical_nlp_lab.inference import (
    FinalModelBundle,
    InferenceConfig,
    SpanProposal,
    infer_document,
    merge_raw_span_proposals,
)
from clinical_nlp_lab.schema import EntityAnnotation


def test_kb_first_recovery_when_ner_misses():
    raw_text = "HS-0300: Bệnh nhân bị sốt xuất huyết nặng."
    records = [
        ClinicalRecord("300", "300:record-0001", 0, len(raw_text), (0,))
    ]

    target = "sốt xuất huyết"
    start = raw_text.index(target)
    end = start + len(target)

    kb_proposal = SpanProposal(target, "DISEASE", start, end, 0.95, "kb_first")
    merged = merge_raw_span_proposals([kb_proposal], records, raw_text=raw_text)

    assert len(merged) == 1
    assert merged[0].text == "sốt xuất huyết"
    assert merged[0].type == "DISEASE"


def test_merge_retains_ranked_candidates_without_changing_submission_payload():
    raw_text = "HS-0301: sốt cao."
    records = [ClinicalRecord("301", "301:record-0001", 0, len(raw_text), (0,))]
    target = "sốt cao"
    start = raw_text.index(target)
    ranked_candidates = (
        {"candidate_id": "A00", "score": 0.91},
        {"candidate_id": "B00", "score": 0.42},
    )

    merged = merge_raw_span_proposals(
        [
            SpanProposal(
                target,
                "DISEASE",
                start,
                start + len(target),
                0.95,
                "kb_first",
                ranked_candidates=ranked_candidates,
                candidate_ids=("A00",),
            )
        ],
        records,
        raw_text=raw_text,
    )

    assert merged[0].ranked_candidates == list(ranked_candidates)
    assert merged[0].candidates == ["A00"]
    submission = merged[0].to_submission("CHẨN_ĐOÁN", [])
    assert set(submission) == {"text", "type", "position", "assertions", "candidates"}
    assert "ranked_candidates" not in submission


def test_invalid_round_trip_proposal_filtered():
    raw_text = "Bệnh nhân bị ho kéo dài."
    records = [
        ClinicalRecord("301", "301:record-0001", 0, len(raw_text), (0,))
    ]

    # Substring ở (0, 4) là "Bệnh", nhưng proposal bảo text = "ho" -> hỏng round-trip
    bad_proposal = SpanProposal("ho", "SYMPTOM", 0, 4, 0.90, "ner")
    merged = merge_raw_span_proposals([bad_proposal], records, raw_text=raw_text)

    assert len(merged) == 0


def test_no_merge_across_record_boundary():
    raw_text = "HS-0302: Bệnh nhân 1. HS-0303: Bệnh nhân 2."
    records = [
        ClinicalRecord("302", "302:record-0001", 0, 21, (0,)),
        ClinicalRecord("303", "303:record-0001", 22, len(raw_text), (0,)),
    ]

    # Proposal đè qua 2 record (10 đến 25)
    cross_proposal = SpanProposal("1. HS-0303:", "DISEASE", 19, 31, 0.80, "ner")
    merged = merge_raw_span_proposals([cross_proposal], records, raw_text=raw_text)

    # Phải bị loại do vượt biên record
    assert len(merged) == 0


def test_qwen_failure_fallback_deterministic():
    raw_text = "HS-0304: Sốt cao."
    bundle = FinalModelBundle(ner_model=None, tokenizer=None)
    config = InferenceConfig(enable_qwen=True)

    doc = infer_document("304", raw_text, bundle, config)
    assert doc.document_id == "304"


def test_trimmed_entity_is_relinked_and_asserted_using_final_boundary():
    """A Qwen trim must occur before candidate lookup and assertions."""
    raw_text = "risk of pneumonia due to immobility."
    calls = []

    class Ner:
        def detect(self, _raw_text):
            return [
                EntityAnnotation(
                    text="risk of pneumonia due to immobility",
                    type="DISEASE",
                    position=(0, 35),
                    confidence=0.98,
                    evidence=["transformer_window:0"],
                )
            ]

    class Refiner:
        def validate_entities(self, entities, _raw_text):
            calls.append("validate")
            return (
                replace(
                    entities[0],
                    text="pneumonia",
                    position=(8, 17),
                    candidates=[],
                    assertions=[],
                    ranked_candidates=[],
                ),
            )

        def refine(self, entities, _raw_text):
            calls.append("metadata")
            return entities

    class Linker:
        def rank_candidates(self, entity_type, mention, limit=20):
            calls.append(("link", entity_type, mention))
            return [{"candidate_id": "J18.9", "name": "Pneumonia", "score": 1.0}]

    class Assertion:
        def predict(self, _raw_text, entities):
            calls.append(("assert", entities[0].position))
            return {(8, 17, "DISEASE"): ["isHistorical"]}

    class AcceptTopPolicy:
        def apply(self, ranked):
            return [ranked[0]["candidate_id"]]

    document = infer_document(
        "1",
        raw_text,
        FinalModelBundle(Ner(), None, Assertion(), AcceptTopPolicy(), Linker(), Refiner()),
        InferenceConfig(enable_qwen=True, enable_kb_recovery=False),
    )

    assert document.entities[0].text == "pneumonia"
    assert document.entities[0].candidates == ["J18.9"]
    assert document.entities[0].assertions == ["isHistorical"]
    assert calls.index("validate") < calls.index(("link", "DISEASE", "pneumonia"))
    assert calls.index(("assert", (8, 17))) < calls.index("metadata")


def test_exact_kb_span_over_160_characters_is_not_filtered():
    raw_text = "disease " * 40
    text = raw_text.rstrip()
    proposal = SpanProposal(
        text,
        "DISEASE",
        0,
        len(text),
        0.99,
        "kb_first",
        ranked_candidates=({"candidate_id": "Z99", "name": "Long ICD", "score": 1.0},),
    )
    records = [ClinicalRecord("1", "1:record-0001", 0, len(raw_text), (0,))]

    merged = merge_raw_span_proposals([proposal], records, raw_text=raw_text)

    assert len(merged) == 1
    assert len(merged[0].text) > 160


def test_exact_kb_proposal_beats_overlapping_transformer_proposal():
    raw_text = "severe heart failure today"
    transformer = SpanProposal("severe heart failure", "DISEASE", 0, 20, 0.99, "ner")
    exact_kb = SpanProposal(
        "heart failure",
        "DISEASE",
        7,
        20,
        0.50,
        "kb_first",
        ranked_candidates=({"candidate_id": "I50.9", "score": 1.0},),
    )

    merged = merge_raw_span_proposals(
        [transformer, exact_kb],
        [ClinicalRecord("1", "1:record-0001", 0, len(raw_text), (0,))],
        raw_text=raw_text,
    )

    assert [(entity.text, entity.position) for entity in merged] == [
        ("heart failure", (7, 20))
    ]
