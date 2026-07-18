import json

import pytest

from src.chunking.clinical_chunker import ClinicalChunker
from src.config import ChunkingConfig


def test_chunk_offsets_always_slice_original_text():
    text = (
        "1. Tiền sử bệnh\n"
        "Thuốc trước khi nhập viện\n"
        "- metoprolol 25mg po bid\n\n"
        "2. Kết quả xét nghiệm\n"
        "creatinine: 1.2 mg/dL"
    )

    chunks = ClinicalChunker().chunk(text)

    assert chunks
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


def test_pre_admission_medication_section_is_preserved():
    text = "Thuốc trước khi nhập viện\n- metoprolol 25mg po bid"

    chunk = ClinicalChunker().chunk(text)[0]

    assert chunk.section_type == "pre_admission_medications"
    assert chunk.header_text == "Thuốc trước khi nhập viện"


def test_long_section_uses_overlap_without_losing_offsets():
    text = "Triệu chứng hiện tại\n" + "ho khó thở " * 1000
    chunker = ClinicalChunker(ChunkingConfig(max_tokens=50, overlap_tokens=10))

    chunks = chunker.chunk(text)

    assert len(chunks) > 1
    assert chunks[1].start < chunks[0].end
    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)


def test_lab_context_does_not_cross_into_medication_section():
    text = (
        "Thuốc hiện tại\n"
        "- metoprolol 25mg\n"
        "Kết quả xét nghiệm\n"
        "creatinine: 1.2 mg/dL"
    )
    chunker = ClinicalChunker()
    chunks = chunker.chunk(text)
    start = text.index("creatinine")
    lab_chunk = next(chunk for chunk in chunks if chunk.start <= start < chunk.end)

    context = chunker.context_for_span(
        text, lab_chunk, start, start + len("creatinine"), "TÊN_XÉT_NGHIỆM"
    )

    assert "creatinine: 1.2 mg/dL" in context
    assert "metoprolol" not in context


def test_generic_heading_does_not_classify_a_clinical_content_line():
    text = "xét nghiệm creatinine tăng cao\nĐánh giá bệnh nhân ổn định"

    chunks = ClinicalChunker().chunk(text)

    assert {chunk.section_type for chunk in chunks} == {"unknown"}


def test_overlapping_heading_prefers_the_longest_pattern():
    text = "Tiền sử bệnh hiện tại\nHo kéo dài"

    chunk = ClinicalChunker().chunk(text)[0]

    assert chunk.section_type == "current_symptoms"
    assert chunk.header_text == "Tiền sử bệnh hiện tại"


def test_empty_document_has_no_chunks():
    assert ClinicalChunker().chunk("") == []


def test_unknown_preamble_is_retained_before_the_first_heading():
    text = "Bệnh nhân nhập viện vì ho.\nTiền sử bệnh\nTăng huyết áp"

    chunks = ClinicalChunker().chunk(text)

    assert chunks[0].section_type == "unknown"
    assert chunks[0].text == "Bệnh nhân nhập viện vì ho."


def test_crlf_and_unicode_offsets_are_exact_slices():
    text = "Tiền sử bệnh\r\nĐau đầu 😊\r\nKết quả xét nghiệm\r\nNa⁺: 140 mmol/L"

    chunks = ClinicalChunker().chunk(text)

    assert all(text[chunk.start : chunk.end] == chunk.text for chunk in chunks)
    assert any("😊" in chunk.text for chunk in chunks)


def test_medication_context_returns_header_and_its_line_only():
    text = "Thuốc hiện tại\n- metoprolol 25mg\n- aspirin 81mg"
    chunker = ClinicalChunker()
    chunk = chunker.chunk(text)[0]
    start = text.index("metoprolol")

    context = chunker.context_for_span(
        text, chunk, start, start + len("metoprolol"), "THUỐC"
    )

    assert "Thuốc hiện tại" in context
    assert "metoprolol 25mg" in context
    assert "aspirin" not in context


@pytest.mark.parametrize("entity_type", ["CHẨN_ĐOÁN", "TRIỆU_CHỨNG"])
def test_diagnosis_or_symptom_context_includes_neighbor_sentence(entity_type):
    text = "Triệu chứng hiện tại\nBệnh nhân ho kéo dài. Khó thở khi gắng sức. Không sốt."
    chunker = ClinicalChunker()
    chunk = chunker.chunk(text)[0]
    start = text.index("ho kéo dài")

    context = chunker.context_for_span(
        text, chunk, start, start + len("ho kéo dài"), entity_type
    )

    assert "Bệnh nhân ho kéo dài." in context
    assert "Khó thở khi gắng sức." in context
    assert "Không sốt." not in context


@pytest.mark.parametrize("entity_type", ["TÊN_XÉT_NGHIỆM", "KẾT_QUẢ_XÉT_NGHIỆM"])
def test_lab_context_returns_only_its_line(entity_type):
    text = "Kết quả xét nghiệm\ncreatinine: 1.2 mg/dL\nurea: 5.0 mmol/L"
    chunker = ClinicalChunker()
    chunk = chunker.chunk(text)[0]
    start = text.index("creatinine")

    context = chunker.context_for_span(
        text, chunk, start, start + len("creatinine"), entity_type
    )

    assert context == "creatinine: 1.2 mg/dL"


@pytest.mark.parametrize("start,end", [(-1, 1), (0, 0), (1, 0), (0, 999)])
def test_context_rejects_invalid_or_outside_spans(start, end):
    text = "Kết quả xét nghiệm\ncreatinine: 1.2 mg/dL"
    chunk = ClinicalChunker().chunk(text)[0]

    with pytest.raises(ValueError, match="span"):
        ClinicalChunker().context_for_span(text, chunk, start, end, "TÊN_XÉT_NGHIỆM")


def test_pattern_schema_error_is_actionable(tmp_path):
    path = tmp_path / "invalid_patterns.json"
    path.write_text(json.dumps({"version": 1, "sections": [{"section_type": "lab"}]}), encoding="utf-8")

    with pytest.raises(ValueError, match="patterns"):
        ClinicalChunker(pattern_path=path)


def test_injected_pattern_path_and_tokenizer_control_behavior(tmp_path):
    path = tmp_path / "patterns.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "sections": [
                    {"section_type": "custom", "patterns": ["Custom heading"], "source": "test"}
                ],
            }
        ),
        encoding="utf-8",
    )
    tokenizer = lambda text: [(match.start(), match.end()) for match in __import__("re").finditer(r"\S+", text)]
    chunker = ClinicalChunker(
        ChunkingConfig(max_tokens=2, overlap_tokens=1), pattern_path=path, tokenizer=tokenizer
    )

    chunks = chunker.chunk("Custom heading\na b c d")

    assert chunks[0].section_type == "custom"
    assert len(chunks) > 1
    assert all("Custom heading" not in chunk.text or chunk.start == 0 for chunk in chunks)


def test_pathological_long_unit_has_bounded_deduplicated_progressing_windows():
    text = "Triệu chứng hiện tại\n" + "x " * 101
    config = ChunkingConfig(max_tokens=10, overlap_tokens=9)

    chunks = ClinicalChunker(config).chunk(text)

    assert all(len(chunk.text.split()) <= config.max_tokens for chunk in chunks)
    assert [(chunk.start, chunk.end) for chunk in chunks] == list(
        dict.fromkeys((chunk.start, chunk.end) for chunk in chunks)
    )
    assert all(later.start > earlier.start for earlier, later in zip(chunks[1:], chunks[2:]))
