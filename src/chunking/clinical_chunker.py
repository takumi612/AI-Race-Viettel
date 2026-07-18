"""Offset-safe, section-aware chunking for clinical documents."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Callable, Iterable, Sequence
import unicodedata

from src.config import ChunkingConfig


TokenSpan = tuple[int, int]
Tokenizer = Callable[[str], Sequence[TokenSpan]]


@dataclass(frozen=True)
class ClinicalChunk:
    """A document slice with absolute offsets and section provenance."""

    text: str
    start: int
    end: int
    section_type: str
    header_text: str


@dataclass(frozen=True)
class _SectionPattern:
    section_type: str
    normalized: str
    source: str
    order: int


@dataclass(frozen=True)
class _SectionSpan:
    start: int
    end: int
    section_type: str
    header_start: int | None
    header_end: int | None
    header_text: str


_LINE_NUMBER_PREFIX = re.compile(r"^[ \t]*(?:(?:\d+|[ivxlcdm]+)[.)][ \t]*)?", re.IGNORECASE)
_SENTENCE = re.compile(r"[^\r\n.!?]+[.!?]?")


def _normalize(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _trimmed_span(text: str, start: int, end: int) -> TokenSpan | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def _whitespace_token_spans(text: str) -> list[TokenSpan]:
    return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


class ClinicalChunker:
    """Chunk headings and clinical lines without changing document offsets."""

    def __init__(
        self,
        config: ChunkingConfig | None = None,
        *,
        pattern_path: str | Path | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self.config = config or ChunkingConfig()
        self._patterns = self._load_patterns(
            Path(pattern_path)
            if pattern_path is not None
            else Path(__file__).resolve().parents[1] / "resources" / "section_patterns.json"
        )
        self._tokenizer = tokenizer or _whitespace_token_spans

    @staticmethod
    def _load_patterns(path: Path) -> tuple[_SectionPattern, ...]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"section pattern resource does not exist: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"section pattern resource is invalid JSON: {path}: {error.msg}") from error

        if not isinstance(payload, dict):
            raise ValueError("section pattern resource must be a JSON object")
        if payload.get("version") != 1:
            raise ValueError("section pattern resource version must be 1")
        sections = payload.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ValueError("section pattern resource sections must be a non-empty list")

        patterns: list[_SectionPattern] = []
        for section_index, section in enumerate(sections):
            location = f"sections[{section_index}]"
            if not isinstance(section, dict):
                raise ValueError(f"section pattern resource {location} must be an object")
            section_type = section.get("section_type")
            source = section.get("source")
            raw_patterns = section.get("patterns")
            if not isinstance(section_type, str) or not section_type.strip():
                raise ValueError(f"section pattern resource {location}.section_type must be a non-empty string")
            if not isinstance(raw_patterns, list) or not raw_patterns:
                raise ValueError(f"section pattern resource {location}.patterns must be a non-empty list")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"section pattern resource {location}.source must be a non-empty string")
            for pattern_index, pattern in enumerate(raw_patterns):
                if not isinstance(pattern, str) or not pattern.strip():
                    raise ValueError(
                        f"section pattern resource {location}.patterns[{pattern_index}] must be a non-empty string"
                    )
                patterns.append(
                    _SectionPattern(section_type, _normalize(pattern.strip()), source, section_index)
                )

        return tuple(sorted(patterns, key=lambda item: (-len(item.normalized), item.order, item.normalized)))

    @staticmethod
    def _lines(document: str) -> Iterable[TokenSpan]:
        start = 0
        for line in document.splitlines(keepends=True):
            end = start + len(line)
            content_end = end
            while content_end > start and document[content_end - 1] in "\r\n":
                content_end -= 1
            yield start, content_end
            start = end
        if start < len(document):
            yield start, len(document)

    def _heading_for_line(self, document: str, start: int, end: int) -> tuple[_SectionPattern, int, int] | None:
        line = document[start:end]
        prefix = _LINE_NUMBER_PREFIX.match(line)
        assert prefix is not None
        heading_start = start + prefix.end()
        candidate_end = end
        while candidate_end > heading_start and document[candidate_end - 1].isspace():
            candidate_end -= 1
        if candidate_end > heading_start and document[candidate_end - 1] == ":":
            candidate_end -= 1
            while candidate_end > heading_start and document[candidate_end - 1].isspace():
                candidate_end -= 1
        if heading_start >= candidate_end:
            return None
        normalized = _normalize(document[heading_start:candidate_end])
        for pattern in self._patterns:
            if normalized == pattern.normalized:
                return pattern, heading_start, candidate_end
        return None

    def _section_spans(self, document: str) -> list[_SectionSpan]:
        headings: list[tuple[int, int, _SectionPattern, int, int]] = []
        for line_start, line_end in self._lines(document):
            match = self._heading_for_line(document, line_start, line_end)
            if match is not None:
                pattern, header_start, header_end = match
                headings.append((line_start, line_end, pattern, header_start, header_end))

        if not headings:
            return [_SectionSpan(0, len(document), "unknown", None, None, "")] if document else []

        spans: list[_SectionSpan] = []
        if headings[0][0] > 0:
            spans.append(_SectionSpan(0, headings[0][0], "unknown", None, None, ""))
        for index, (line_start, _, pattern, header_start, header_end) in enumerate(headings):
            next_start = headings[index + 1][0] if index + 1 < len(headings) else len(document)
            spans.append(
                _SectionSpan(
                    line_start,
                    next_start,
                    pattern.section_type,
                    header_start,
                    header_end,
                    document[header_start:header_end],
                )
            )
        return spans

    def _token_spans(self, text: str) -> list[TokenSpan]:
        spans = list(self._tokenizer(text))
        previous_end = 0
        for span in spans:
            if (
                not isinstance(span, tuple)
                or len(span) != 2
                or not all(isinstance(value, int) for value in span)
                or span[0] < previous_end
                or span[0] >= span[1]
                or span[1] > len(text)
            ):
                raise ValueError("tokenizer must return ordered (start, end) spans inside its input")
            previous_end = span[1]
        return spans

    def _units(self, document: str, section: _SectionSpan) -> list[TokenSpan]:
        units: list[TokenSpan] = []
        for line_start, line_end in self._lines(document[section.start : section.end]):
            span = _trimmed_span(document, section.start + line_start, section.start + line_end)
            if span is not None:
                units.append(span)
        return units

    def _chunks_for_section(self, document: str, section: _SectionSpan) -> list[ClinicalChunk]:
        chunks: list[ClinicalChunk] = []
        buffered_start: int | None = None
        buffered_end: int | None = None
        buffered_tokens = 0

        def emit(start: int, end: int) -> None:
            chunks.append(ClinicalChunk(document[start:end], start, end, section.section_type, section.header_text))

        def flush() -> None:
            nonlocal buffered_start, buffered_end, buffered_tokens
            if buffered_start is not None and buffered_end is not None:
                emit(buffered_start, buffered_end)
            buffered_start = None
            buffered_end = None
            buffered_tokens = 0

        for unit_start, unit_end in self._units(document, section):
            unit_tokens = self._token_spans(document[unit_start:unit_end])
            if len(unit_tokens) > self.config.max_tokens:
                token_index = 0
                if buffered_start is not None:
                    remaining = self.config.max_tokens - buffered_tokens
                    if remaining > 0:
                        first_window_end = min(remaining, len(unit_tokens))
                        buffered_end = unit_start + unit_tokens[first_window_end - 1][1]
                        flush()
                        if first_window_end == len(unit_tokens):
                            continue
                        token_index = max(0, first_window_end - self.config.overlap_tokens)
                    else:
                        flush()
                stride = self.config.max_tokens - self.config.overlap_tokens
                while token_index < len(unit_tokens):
                    window_end = min(token_index + self.config.max_tokens, len(unit_tokens))
                    emit(unit_start + unit_tokens[token_index][0], unit_start + unit_tokens[window_end - 1][1])
                    if window_end == len(unit_tokens):
                        break
                    token_index += stride
                continue
            if buffered_start is not None and buffered_tokens + len(unit_tokens) > self.config.max_tokens:
                flush()
            if buffered_start is None:
                buffered_start = unit_start
            buffered_end = unit_end
            buffered_tokens += len(unit_tokens)
        flush()
        return chunks

    def chunk(self, document: str) -> list[ClinicalChunk]:
        if not isinstance(document, str):
            raise TypeError("document must be a string")
        chunks: list[ClinicalChunk] = []
        seen: set[tuple[int, int, str]] = set()
        for section in self._section_spans(document):
            for chunk in self._chunks_for_section(document, section):
                key = (chunk.start, chunk.end, chunk.section_type)
                if key not in seen:
                    seen.add(key)
                    chunks.append(chunk)
        return chunks

    @staticmethod
    def _line_containing(document: str, position: int, lower: int, upper: int) -> TokenSpan:
        start = document.rfind("\n", lower, position) + 1
        start = max(start, lower)
        line_end = document.find("\n", position, upper)
        end = upper if line_end == -1 else line_end
        if end > start and document[end - 1] == "\r":
            end -= 1
        span = _trimmed_span(document, start, end)
        if span is None:
            raise ValueError("span is not inside a clinical line")
        return span

    @staticmethod
    def _sentence_spans(document: str, line_start: int, line_end: int) -> list[TokenSpan]:
        spans: list[TokenSpan] = []
        for match in _SENTENCE.finditer(document[line_start:line_end]):
            span = _trimmed_span(document, line_start + match.start(), line_start + match.end())
            if span is not None:
                spans.append(span)
        return spans

    def context_for_span(
        self,
        document: str,
        chunk: ClinicalChunk,
        start: int,
        end: int,
        entity_type: str,
    ) -> str:
        if not isinstance(document, str):
            raise TypeError("document must be a string")
        if not isinstance(chunk, ClinicalChunk):
            raise TypeError("chunk must be a ClinicalChunk")
        if not (0 <= chunk.start <= chunk.end <= len(document)) or document[chunk.start:chunk.end] != chunk.text:
            raise ValueError("chunk bounds or text do not match the document")
        if not (0 <= start < end <= len(document)):
            raise ValueError("span bounds are invalid")
        if start < chunk.start or end > chunk.end:
            raise ValueError("span is outside the chunk")

        section = next(
            (item for item in self._section_spans(document) if item.start <= start and end <= item.end),
            None,
        )
        if section is None:
            raise ValueError("span is outside a document section")
        lower = max(chunk.start, section.start)
        upper = min(chunk.end, section.end)
        if start < lower or end > upper:
            raise ValueError("span crosses a section or chunk boundary")

        line_start, line_end = self._line_containing(document, start, lower, upper)
        normalized_type = _normalize(entity_type)
        line_text = document[line_start:line_end]
        if normalized_type in {"thuốc", "medication", "drug"}:
            if (
                section.header_start is not None
                and section.header_end is not None
                and lower <= section.header_start
                and section.header_end <= upper
                and (section.header_start, section.header_end) != (line_start, line_end)
            ):
                return f"{document[section.header_start:section.header_end]}\n{line_text}"
            return line_text
        if normalized_type in {"tên_xét_nghiệm", "kết_quả_xét_nghiệm", "lab_test", "lab_result"}:
            return line_text
        if normalized_type in {"chẩn_đoán", "triệu_chứng", "diagnosis", "symptom"}:
            sentences = self._sentence_spans(document, line_start, line_end)
            containing = next(
                (index for index, span in enumerate(sentences) if span[0] <= start and end <= span[1]),
                None,
            )
            if containing is None:
                return line_text
            selected = [sentences[containing]]
            if containing + 1 < len(sentences):
                selected.append(sentences[containing + 1])
            elif containing > 0:
                selected.insert(0, sentences[containing - 1])
            return document[selected[0][0] : selected[-1][1]]
        return line_text
