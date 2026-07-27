from __future__ import annotations

from clinical_nlp_lab.inference import SpanProposal, merge_raw_span_proposals
from clinical_nlp_lab.records import ClinicalRecord


def _record(raw_text: str) -> list[ClinicalRecord]:
    return [ClinicalRecord("doc", "record", 0, len(raw_text), ())]


def test_inference_rejects_punctuation_midword_and_multiline_spans():
    raw_text = "alpha, beta\ngamma " + ("x" * 180)
    proposals = [
        SpanProposal(",", "DISEASE", 5, 6, 0.99, "ner"),
        SpanProposal("pha", "DISEASE", 2, 5, 0.99, "ner"),
        SpanProposal("beta\ngamma", "LAB_NAME", 7, 17, 0.99, "ner"),
    ]

    assert merge_raw_span_proposals(proposals, _record(raw_text), raw_text) == ()


def test_higher_confidence_precise_span_wins_over_long_overlap():
    raw_text = "severe heart failure today"
    long = SpanProposal("severe heart failure", "DISEASE", 0, 20, 0.60, "ner")
    precise = SpanProposal("heart failure", "DISEASE", 7, 20, 0.95, "kb_first")

    merged = merge_raw_span_proposals([long, precise], _record(raw_text), raw_text)

    assert [(item.text, item.position) for item in merged] == [
        ("heart failure", (7, 20))
    ]
