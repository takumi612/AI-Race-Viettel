from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .schema import SectionSpan


TOKEN_PATTERN = re.compile(r"(?u)\b[\w]+(?:[./-][\w]+)*\b")


SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("HISTORY", re.compile(r"^(?:\d+[.)]\s*)?(?:tiền sử|lịch sử bệnh)\b", re.IGNORECASE)),
    ("PRESENT_ILLNESS", re.compile(r"^(?:\d+[.)]\s*)?(?:bệnh sử|tiền sử bệnh hiện tại|lịch sử bệnh hiện tại)\b", re.IGNORECASE)),
    ("MEDICATIONS", re.compile(r"^(?:thuốc|thuốc trước khi nhập viện|đơn thuốc)\b", re.IGNORECASE)),
    ("FAMILY_HISTORY", re.compile(r"^(?:tiền sử gia đình|gia đình)\b", re.IGNORECASE)),
    ("ASSESSMENT", re.compile(r"^(?:\d+[.)]\s*)?(?:đánh giá|đánh giá tại bệnh viện|chẩn đoán)\b", re.IGNORECASE)),
    ("LABS", re.compile(r"^(?:kết quả xét nghiệm|xét nghiệm)\b", re.IGNORECASE)),
    ("PLAN", re.compile(r"^(?:kế hoạch|điều trị|kế hoạch điều trị)\b", re.IGNORECASE)),
    ("PHYSICAL_EXAM", re.compile(r"^(?:khám lâm sàng|kết quả khám lâm sàng)\b", re.IGNORECASE)),
]


@dataclass(slots=True)
class NormalizedText:
    raw_text: str
    model_text: str
    model_to_raw: list[int]

    def raw_span(self, model_start: int, model_end: int) -> tuple[int, int]:
        if not (0 <= model_start <= model_end <= len(self.model_text)):
            raise ValueError("Normalized span is out of range")
        if model_start == model_end:
            raw_index = self.model_to_raw[model_start] if model_start < len(self.model_to_raw) else len(self.raw_text)
            return raw_index, raw_index
        raw_start = self.model_to_raw[model_start]
        raw_end = self.model_to_raw[model_end - 1] + 1
        return raw_start, raw_end


def normalize_with_mapping(raw_text: str) -> NormalizedText:
    output: list[str] = []
    mapping: list[int] = []
    previous_space = False
    for raw_index, character in enumerate(raw_text):
        normalized = unicodedata.normalize("NFC", character).casefold()
        for normalized_character in normalized:
            if normalized_character.isspace():
                if previous_space:
                    continue
                output.append(" ")
                mapping.append(raw_index)
                previous_space = True
            else:
                output.append(normalized_character)
                mapping.append(raw_index)
                previous_space = False
    return NormalizedText(raw_text=raw_text, model_text="".join(output), model_to_raw=mapping)


def tokenize_with_offsets(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def normalize_alias(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    normalized = re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()


def classify_heading(line: str) -> str | None:
    stripped = line.strip(" \t:-")
    for section_name, pattern in SECTION_PATTERNS:
        if pattern.search(stripped):
            return section_name
    return None


def detect_sections(raw_text: str) -> list[SectionSpan]:
    line_ranges: list[tuple[int, int, str]] = []
    offset = 0
    for line in raw_text.splitlines(keepends=True):
        line_ranges.append((offset, offset + len(line), line))
        offset += len(line)
    if offset < len(raw_text):
        line_ranges.append((offset, len(raw_text), raw_text[offset:]))

    headings: list[tuple[int, str]] = []
    for start, _end, line in line_ranges:
        section_name = classify_heading(line)
        if section_name is not None:
            headings.append((start, section_name))

    if not headings:
        section = SectionSpan("UNKNOWN", 0, len(raw_text), raw_text)
        section.validate(raw_text)
        return [section]

    if headings[0][0] > 0:
        headings.insert(0, (0, "PREAMBLE"))

    sections: list[SectionSpan] = []
    for index, (start, name) in enumerate(headings):
        end = headings[index + 1][0] if index + 1 < len(headings) else len(raw_text)
        section = SectionSpan(name, start, end, raw_text[start:end])
        section.validate(raw_text)
        sections.append(section)
    return sections


def split_sentences(raw_text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for line_match in re.finditer(r"[^\n]+(?:\n|$)", raw_text):
        line = line_match.group(0)
        line_start = line_match.start()
        for match in re.finditer(r"[^.!?;\n]+(?:[.!?;]+|$)", line):
            text = match.group(0)
            if not text.strip():
                continue
            start = line_start + match.start()
            end = line_start + match.end()
            while start < end and raw_text[start].isspace():
                start += 1
            while end > start and raw_text[end - 1].isspace():
                end -= 1
            spans.append((start, end, raw_text[start:end]))
    return spans


def containing_section(position: tuple[int, int], sections: Iterable[SectionSpan]) -> str:
    start, end = position
    for section in sections:
        if section.start <= start and end <= section.end:
            return section.section_name
    return "UNKNOWN"
