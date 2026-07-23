from __future__ import annotations

from pathlib import Path

import pytest

from clinical_nlp_lab.records import (
    OFFSET_SPACE,
    RecordContractError,
    build_record_metadata,
    detect_record_spans,
)


ROOT = Path(__file__).parents[1]
HEADER = "- Chi ti\u1ebft h\u1ed3 s\u01a1 b\u1ec7nh nh\u00e2n th\u1ee9 {ordinal}:"


def _entity(text: str, value: str) -> dict[str, object]:
    start = text.index(value)
    return {"position": [start, start + len(value)]}


def test_organizer_records_cover_preamble_and_sequential_headers():
    text = "Preamble\n\n" + HEADER.format(ordinal=1) + "\nalpha\n\n" + HEADER.format(
        ordinal=2
    ) + "\nbeta"
    spans = detect_record_spans("101", text, [_entity(text, "alpha"), _entity(text, "beta")])

    assert len(spans) == 2
    assert spans[0].start == 0
    assert spans[0].end == text.index(HEADER.format(ordinal=2))
    assert spans[-1].end == len(text)
    assert [span.patient_block_id for span in spans] == [
        "101:record-0001",
        "101:record-0002",
    ]


def test_ambiguous_ordinal_and_crossing_entity_fail_closed():
    text = HEADER.format(ordinal=1) + "\nalpha\n" + HEADER.format(ordinal=3) + "\nbeta"
    with pytest.raises(RecordContractError, match="ordinals"):
        detect_record_spans("101", text, [])

    valid = HEADER.format(ordinal=1) + "\nalpha\n" + HEADER.format(ordinal=2) + "\nbeta"
    boundary = valid.index(HEADER.format(ordinal=2))
    with pytest.raises(RecordContractError, match="crosses"):
        detect_record_spans("101", valid, [{"position": [boundary - 1, boundary + 1]}])


def test_synthetic_record_requires_one_matching_hospital_id():
    text = "Mã hồ sơ HS-0201. alpha"
    spans = detect_record_spans("201", text, [_entity(text, "alpha")])
    assert len(spans) == 1
    assert spans[0].start == 0 and spans[0].end == len(text)

    with pytest.raises(RecordContractError, match="matching"):
        detect_record_spans("201", "HS-9999 alpha", [])


def test_real_metadata_has_exact_audited_record_counts_and_bindings():
    dataset = ROOT.parent / "data_v2" / "Training_data" / "synthetic_train_v2"
    if not dataset.is_dir():
        pytest.skip("real workspace dataset is not attached")

    snapshot = build_record_metadata(dataset)
    first_200 = sum(len(row.record_spans) for row in snapshot.rows if int(row.document_id) <= 200)
    synthetic = sum(len(row.record_spans) for row in snapshot.rows if int(row.document_id) > 200)

    assert len(snapshot.rows) == 2_200
    assert snapshot.record_count == 3_204
    assert first_200 == 1_204
    assert synthetic == 2_000
    assert OFFSET_SPACE == "utf8-universal-newline-codepoint/v1"
    assert len(snapshot.metadata_sha256) == 64
    assert all(span.confidence == "high" for row in snapshot.rows for span in row.record_spans)
