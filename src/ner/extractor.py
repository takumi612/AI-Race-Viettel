"""Dictionary mention detection followed by contextual clinical type resolution."""

from __future__ import annotations

import os
import sys

if __package__ in {None, ""}:
    _MODULE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
    if sys.path and os.path.normcase(sys.path[0]) == os.path.normcase(_MODULE_DIRECTORY):
        sys.path.pop(0)
    sys.path.insert(0, os.path.dirname(os.path.dirname(_MODULE_DIRECTORY)))

from collections.abc import Iterable
from pathlib import Path
import re
import sqlite3
import unicodedata

from src.chunking.clinical_chunker import ClinicalChunk, ClinicalChunker
from src.config import NERConfig
from src.ner.lexicon_loader import ClinicalLexicon, ClinicalTerm, normalize_term
from src.ner.type_resolver import ContextualTypeResolver, TypeRules
from src.ner.types import MentionCandidate
from src.utils.paths import DB_PATH


_LAB_TEST = "TÊN_XÉT_NGHIỆM"
_LAB_RESULT = "KẾT_QUẢ_XÉT_NGHIỆM"
_MEDICATION = "THUỐC"


def _is_word_character(value: str) -> bool:
    return value == "_" or value.isalnum() or unicodedata.category(value).startswith("M")


def _fold_with_offsets(text: str) -> tuple[str, list[int], list[int]]:
    folded: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, character in enumerate(text):
        replacement = character.casefold()
        folded.extend(replacement)
        starts.extend([index] * len(replacement))
        ends.extend([index + 1] * len(replacement))
    return "".join(folded), starts, ends


class TrieMatcher:
    """Detect exact mentions without deciding their final clinical type."""

    def __init__(self) -> None:
        self.root: dict[object, object] = {}

    def insert(self, word: str, type_name: str, source: str = "dictionary") -> None:
        if not isinstance(word, str) or not word.strip():
            raise ValueError("trie word must be a non-empty string")
        if not isinstance(type_name, str) or not type_name:
            raise ValueError("trie type must be a non-empty string")
        if not isinstance(source, str) or not source:
            raise ValueError("trie source must be a non-empty string")
        normalized = normalize_term(word)
        node = self.root
        for character in normalized:
            node = node.setdefault(character, {})
        terminal = node.setdefault(None, {})
        terminal.setdefault(type_name, set()).add(source)

    def search_in_text(self, text: str, *, offset: int = 0) -> list[MentionCandidate]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError("offset must be a non-negative integer")
        folded, starts, ends = _fold_with_offsets(text)
        spans: dict[tuple[int, int], tuple[set[str], set[str]]] = {}
        for folded_start in range(len(folded)):
            if folded_start > 0 and starts[folded_start] == starts[folded_start - 1]:
                continue
            original_start = starts[folded_start]
            if original_start > 0 and _is_word_character(text[original_start - 1]):
                continue
            node = self.root
            cursor = folded_start
            while cursor < len(folded) and folded[cursor] in node:
                node = node[folded[cursor]]
                cursor += 1
                terminal = node.get(None)
                if terminal is None:
                    continue
                original_end = ends[cursor - 1]
                if original_end < len(text) and _is_word_character(text[original_end]):
                    continue
                key = (original_start, original_end)
                candidate_types, sources = spans.setdefault(key, (set(), set()))
                for entity_type, terminal_sources in terminal.items():
                    candidate_types.add(entity_type)
                    sources.update(terminal_sources)
        return [
            MentionCandidate(
                text[start:end],
                offset + start,
                offset + end,
                frozenset(candidate_types),
                frozenset(sources),
                True,
            )
            for (start, end), (candidate_types, sources) in sorted(spans.items())
        ]

    def sources_for(self, mention: MentionCandidate, entity_type: str) -> frozenset[str]:
        node = self.root
        for character in normalize_term(mention.text):
            child = node.get(character)
            if child is None:
                return frozenset()
            node = child
        terminal = node.get(None, {})
        return frozenset(terminal.get(entity_type, ()))


class BaselineExtractor:
    def __init__(
        self,
        config: NERConfig | None = None,
        *,
        load_database: bool = True,
        clinical_lexicon_path: str | Path | None = None,
        type_rules_path: str | Path | None = None,
        clinical_terms: Iterable[ClinicalTerm] | None = None,
        type_rules: TypeRules | None = None,
    ) -> None:
        if not isinstance(load_database, bool):
            raise ValueError("load_database must be a boolean")
        if clinical_terms is not None and clinical_lexicon_path is not None:
            raise ValueError("provide clinical_terms or clinical_lexicon_path, not both")
        self.matcher = TrieMatcher()
        self._chunker = ClinicalChunker()
        default_lexicon = Path(__file__).resolve().parents[1] / "resources" / "clinical_lexicon.json"
        terms = tuple(clinical_terms) if clinical_terms is not None else ClinicalLexicon.load(
            clinical_lexicon_path or default_lexicon
        )
        if any(not isinstance(term, ClinicalTerm) for term in terms):
            raise ValueError("clinical_terms must contain ClinicalTerm values")
        for term in terms:
            self.matcher.insert(term.term, term.entity_type, term.source)

        source_statuses: dict[str, str] = {}
        for term in terms:
            current = source_statuses.get(term.source)
            if current is None or term.status == "unverified":
                source_statuses[term.source] = term.status
        if load_database:
            self._load_database_terms()
        self._resolver = ContextualTypeResolver(
            config,
            rules_path=type_rules_path,
            rules=type_rules,
            source_statuses=source_statuses,
            source_lookup=self.matcher.sources_for,
            chunker=self._chunker,
        )

    def _load_database_terms(self) -> None:
        try:
            connection = sqlite3.connect(DB_PATH)
            try:
                cursor = connection.cursor()
                cursor.execute("SELECT code, name_vi, name_en FROM icd10;")
                for code, name_vi, name_en in cursor.fetchall():
                    if code and code.upper().startswith("R"):
                        continue
                    if name_vi:
                        self.matcher.insert(name_vi, "CHẨN_ĐOÁN", "icd10")
                    if name_en:
                        self.matcher.insert(name_en, "CHẨN_ĐOÁN", "icd10")
                cursor.execute(
                    "SELECT name FROM rxnorm "
                    "WHERE tty IN ('BN', 'IN', 'PIN', 'SBD', 'SCD', 'SCDC');"
                )
                for (name,) in cursor.fetchall():
                    if name:
                        self.matcher.insert(name, _MEDICATION, "rxnorm")
            finally:
                connection.close()
        except (OSError, sqlite3.Error):
            # The resource lexicon remains usable when the optional local database is absent.
            return

    @staticmethod
    def _lab_result_candidate(
        document: str, mention: MentionCandidate, chunk: ClinicalChunk
    ) -> MentionCandidate | None:
        if _LAB_TEST not in mention.candidate_types:
            return None
        line_end = document.find("\n", mention.end, chunk.end)
        line_end = chunk.end if line_end == -1 else line_end
        tail = document[mention.end:line_end]
        match = re.match(r"^[ \t]*(?:\([^\r\n)]*\)[ \t]*)?(?:là|:|=)?[ \t]*(\d+(?:[.,]\d+)?)", tail, re.IGNORECASE)
        if match is None:
            return None
        start = mention.end + match.start(1)
        end = mention.end + match.end(1)
        return MentionCandidate(
            document[start:end], start, end, frozenset({_LAB_RESULT}), mention.sources, True
        )

    def _dose_expansion(self, document: str, start: int, end: int) -> int:
        line_end = document.find("\n", end)
        line_end = len(document) if line_end == -1 else line_end
        tail = document[end:line_end]
        route_patterns = [
            r"[ \t]+".join(re.escape(token) for token in value.split()) + r"(?!\w)"
            for value in sorted(
                self._resolver.rules.route_terms, key=lambda value: (-len(value), value)
            )
        ]
        frequency_patterns = []
        for frequency in self._resolver.rules.frequency_patterns:
            tokens = [
                r"\d+" if token == "{number}" else re.escape(token)
                for token in frequency.tokens
            ]
            frequency_patterns.append(r"[ \t]+".join(tokens) + r"(?!\w)")
        instruction = "|".join(frequency_patterns + route_patterns)
        dosage_unit = "|".join(
            re.escape(value)
            for value in sorted(
                self._resolver.rules.dosage_units, key=lambda value: (-len(value), value)
            )
        )
        pattern = (
            r"^[ \t]*\d+(?:[.,]\d+)?"
            rf"(?:[ \t]*(?:{dosage_unit})(?!\w))?"
            rf"(?:[ \t]+(?:{instruction}))*"
        )
        match = re.match(pattern, tail, re.IGNORECASE | re.UNICODE)
        return end + match.end() if match is not None else end

    @staticmethod
    def _validate_chunks(document: str, chunks: Iterable[ClinicalChunk]) -> tuple[ClinicalChunk, ...]:
        materialized = tuple(chunks)
        for chunk in materialized:
            if not isinstance(chunk, ClinicalChunk):
                raise ValueError("chunks must contain ClinicalChunk values")
            if not (0 <= chunk.start <= chunk.end <= len(document)):
                raise ValueError("chunk bounds are invalid")
            if document[chunk.start : chunk.end] != chunk.text:
                raise ValueError("chunk text does not match the document slice")
        return materialized

    def extract_entities(
        self, text: str, chunks: Iterable[ClinicalChunk] | None = None
    ) -> list[dict]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        if not text:
            return []
        document_chunks = self._validate_chunks(
            text, self._chunker.chunk(text) if chunks is None else chunks
        )
        resolved: dict[tuple[int, int, str], tuple[float, int, int, str]] = {}
        for chunk in document_chunks:
            mentions = self.matcher.search_in_text(chunk.text, offset=chunk.start)
            result_mentions = [
                result
                for mention in mentions
                if (result := self._lab_result_candidate(text, mention, chunk)) is not None
            ]
            for mention in mentions + result_mentions:
                decision = self._resolver.resolve(mention, text, chunk)
                if decision.entity_type is None:
                    continue
                start, end = mention.start, mention.end
                if decision.entity_type == _MEDICATION:
                    end = self._dose_expansion(text, start, end)
                key = (start, end, decision.entity_type)
                current = resolved.get(key)
                candidate = (decision.confidence, start, end, decision.entity_type)
                if current is None or candidate[0] > current[0]:
                    resolved[key] = candidate

        ranked = sorted(
            resolved.values(),
            key=lambda item: (-item[0], -(item[2] - item[1]), item[1], item[3]),
        )
        accepted: list[tuple[float, int, int, str]] = []
        for candidate in ranked:
            _, start, end, _ = candidate
            if any(not (end <= other_start or start >= other_end) for _, other_start, other_end, _ in accepted):
                continue
            accepted.append(candidate)
        accepted.sort(key=lambda item: (item[1], item[2], item[3]))
        return [
            {"text": text[start:end], "type": entity_type, "position": [start, end]}
            for _, start, end, entity_type in accepted
        ]


if __name__ == "__main__":
    extractor = BaselineExtractor()
    sample_text = (
        "Bệnh nhân nam 70 tuổi, ho đờm xanh, tức ngực.\n"
        "Thuốc hiện tại\nChlorpheniramine 0.4 mg po\n"
        "Kết quả xét nghiệm\nWBC: 14,43 mg/dL"
    )
    for entity in extractor.extract_entities(sample_text):
        start, end = entity["position"]
        assert sample_text[start:end] == entity["text"]
        summary = f"[{entity['type']}] {entity['text']} at {entity['position']}"
        print(summary.encode("ascii", "backslashreplace").decode("ascii"))
