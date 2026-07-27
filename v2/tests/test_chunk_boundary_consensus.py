from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from clinical_nlp_lab.ner import merge_chunk_predictions
from clinical_nlp_lab.schema import EntityAnnotation


def entity(raw: str, start: int, end: int, confidence: float, window: int) -> EntityAnnotation:
    return EntityAnnotation(
        text=raw[start:end],
        type="DISEASE",
        position=(start, end),
        confidence=confidence,
        evidence=[f"transformer_window:{window}"],
    )


def test_conflicting_overlap_never_creates_union_boundary():
    raw = "0123456789" * 8

    merged = merge_chunk_predictions(
        [entity(raw, 10, 40, 0.97, 0), entity(raw, 30, 60, 0.96, 1)], raw
    )

    assert [(item.start, item.end) for item in merged] in ([(10, 40)], [(30, 60)])
    assert (10, 60) not in {(item.start, item.end) for item in merged}


def test_identical_boundaries_accumulate_window_evidence():
    raw = "Bệnh nhân được chẩn đoán viêm phổi cộng đồng."
    start = raw.index("viêm phổi cộng đồng")
    end = start + len("viêm phổi cộng đồng")

    merged = merge_chunk_predictions(
        [entity(raw, start, end, 0.91, 0), entity(raw, start, end, 0.95, 1)], raw
    )

    assert len(merged) == 1
    assert merged[0].position == (start, end)
    assert merged[0].confidence == 0.95
    assert merged[0].evidence == [
        "transformer_window:0",
        "transformer_window:1",
    ]


def test_conflicting_overlap_prefers_more_independent_window_support():
    raw = "0123456789" * 8
    lower_confidence_with_two_windows = entity(raw, 10, 40, 0.90, 0)
    lower_confidence_with_two_windows.evidence.append("transformer_window:1")

    merged = merge_chunk_predictions(
        [lower_confidence_with_two_windows, entity(raw, 30, 60, 0.99, 2)], raw
    )

    assert [(item.start, item.end) for item in merged] == [(10, 40)]


def test_conflicting_equal_rank_prefers_shorter_observed_boundary():
    raw = "0123456789" * 8

    merged = merge_chunk_predictions(
        [entity(raw, 10, 40, 0.95, 0), entity(raw, 10, 35, 0.95, 1)], raw
    )

    assert [(item.start, item.end) for item in merged] == [(10, 35)]


def test_near_boundaries_choose_one_observed_boundary_without_crossing_clause_mark():
    raw = "Tiền sử viêm phổi; hiện không khó thở."
    start = raw.index("viêm phổi")
    exact_end = start + len("viêm phổi")

    merged = merge_chunk_predictions(
        [entity(raw, start, exact_end, 0.94, 0), entity(raw, start, exact_end + 1, 0.93, 1)], raw
    )

    assert len(merged) == 1
    assert merged[0].position in {(start, exact_end), (start, exact_end + 1)}
    assert ";" not in merged[0].text


def test_near_boundaries_aggregate_window_evidence_without_delimiter():
    raw = "Tiền sử viêm phổi cấp hiện ổn định."
    start = raw.index("viêm phổi")
    exact_end = start + len("viêm phổi")

    merged = merge_chunk_predictions(
        [entity(raw, start, exact_end, 0.94, 0), entity(raw, start, exact_end + 1, 0.93, 1)], raw
    )

    assert [(item.start, item.end) for item in merged] == [(start, exact_end)]
    assert merged[0].evidence == ["transformer_window:0", "transformer_window:1"]


def test_long_observed_boundary_is_not_removed_by_length():
    raw = "x " + ("chẩn đoán hợp lệ " * 8).strip() + "."
    start, end = 2, len(raw) - 1

    merged = merge_chunk_predictions([entity(raw, start, end, 0.99, 0)], raw)

    assert len(merged) == 1
    assert len(merged[0].text) > 100


def test_identical_boundary_tie_uses_deterministic_metadata():
    raw = "viêm phổi"
    first = entity(raw, 0, len(raw), 0.95, 0)
    first.mention_head = "zulu"
    second = entity(raw, 0, len(raw), 0.95, 1)
    second.mention_head = "alpha"

    forward = merge_chunk_predictions([first, second], raw)
    reverse = merge_chunk_predictions([second, first], raw)

    assert [item.mention_head for item in forward] == ["alpha"]
    assert [item.mention_head for item in reverse] == ["alpha"]
