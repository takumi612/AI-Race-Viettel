from __future__ import annotations

import pytest
from clinical_nlp_lab.records import ClinicalRecord
from clinical_nlp_lab.inference import (
    FinalModelBundle,
    InferenceConfig,
    SpanProposal,
    infer_document,
    merge_raw_span_proposals,
)


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
