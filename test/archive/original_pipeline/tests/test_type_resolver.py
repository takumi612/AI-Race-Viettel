import json
from pathlib import Path
import subprocess
import sys
from types import MappingProxyType

import pytest

from src.chunking.clinical_chunker import ClinicalChunk, ClinicalChunker
from src.config import ChunkingConfig, NERConfig
from src.ner.extractor import BaselineExtractor, TrieMatcher
from src.ner.lexicon_loader import ClinicalLexicon, ClinicalTerm
from src.ner.type_resolver import ContextualTypeResolver, FrequencyPattern, TypeRules
from src.ner.types import MentionCandidate, TypeDecision


def _write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def _entry(term="ho", entity_type="TRIỆU_CHỨNG", source="test", status="verified"):
    return {"term": term, "type": entity_type, "source": source, "status": status}


def _rules_payload():
    return {
        "schema_version": 1,
        "section_priors": {
            "laboratory": {"TÊN_XÉT_NGHIỆM": 1.0, "THUỐC": -1.0},
            "treatment_current_medications": {"THUỐC": 1.0},
        },
        "medication_signals": ["thuốc", "điều trị"],
        "laboratory_units": ["mg/dl", "mmol/l"],
        "dosage_units": ["mg", "mg/ml"],
        "route_terms": ["po", "uống"],
        "frequency_patterns": [
            {"pattern": "ngày {number} lần", "source": "verified-test"}
        ],
        "generic_terms": ["yếu", "loét"],
        "source_confidence": {"verified": 1.5, "unverified": 0.3, "database": 1.0},
        "weights": {
            "exact": 0.5,
            "medication_signal": 0.8,
            "laboratory_signal": 1.0,
            "route_signal": 0.8,
            "generic_penalty": -2.0,
        },
    }


def test_type_contracts_are_deeply_immutable():
    mention = MentionCandidate("ho", 0, 2, {"TRIỆU_CHỨNG"}, {"test"}, True)
    original_scores = {"TRIỆU_CHỨNG": 0.8}
    decision = TypeDecision("TRIỆU_CHỨNG", 0.8, original_scores, "accepted")
    original_scores["TRIỆU_CHỨNG"] = 0.1

    assert isinstance(mention.candidate_types, frozenset)
    assert isinstance(mention.sources, frozenset)
    assert isinstance(decision.scores, MappingProxyType)
    assert decision.scores["TRIỆU_CHỨNG"] == 0.8
    with pytest.raises(TypeError):
        decision.scores["TRIỆU_CHỨNG"] = 0.2


def test_clinical_term_is_immutable_and_normalized(tmp_path):
    path = _write_json(
        tmp_path / "lexicon.json",
        {"schema_version": 1, "entries": [_entry(term="  Ho  ")]},
    )

    term = ClinicalLexicon.load(path)[0]

    assert term == ClinicalTerm("Ho", "ho", "TRIỆU_CHỨNG", "test", "verified")
    with pytest.raises((AttributeError, TypeError)):
        term.term = "changed"


def test_creatinine_in_lab_section_is_not_drug():
    text = "Kết quả xét nghiệm\ncreatinine: 1.2 mg/dL"
    chunk = ClinicalChunker().chunk(text)[0]
    start = text.index("creatinine")
    mention = MentionCandidate(
        "creatinine",
        start,
        start + len("creatinine"),
        frozenset({"THUỐC", "TÊN_XÉT_NGHIỆM"}),
        frozenset({"database", "verified"}),
        True,
    )

    decision = ContextualTypeResolver().resolve(mention, text, chunk)

    assert decision.entity_type == "TÊN_XÉT_NGHIỆM"
    assert all(0.0 <= score <= 1.0 for score in decision.scores.values())


def test_generic_word_is_not_extracted_inside_longer_word():
    extractor = BaselineExtractor(load_database=False)

    assert not any(
        entity["text"].casefold() == "yếu"
        for entity in extractor.extract_entities("Các yếu tố nguy cơ")
    )


def test_ambiguous_low_confidence_span_is_rejected():
    mention = MentionCandidate(
        "loét",
        0,
        4,
        frozenset({"CHẨN_ĐOÁN", "TRIỆU_CHỨNG"}),
        frozenset({"legacy-clinical-lexicon"}),
        True,
    )
    chunk = ClinicalChunk("loét", 0, 4, "unknown", "")

    decision = ContextualTypeResolver().resolve(mention, "loét", chunk)

    assert decision.entity_type is None
    assert "threshold" in decision.reason or "ambiguous" in decision.reason


def test_resolver_uses_only_configured_thresholds_and_margin(tmp_path):
    rules_path = _write_json(tmp_path / "rules.json", _rules_payload())
    mention = MentionCandidate("ho", 0, 2, {"TRIỆU_CHỨNG"}, {"verified"}, True)
    chunk = ClinicalChunk("ho", 0, 2, "unknown", "")
    permissive = ContextualTypeResolver(
        NERConfig(default_threshold=0.1, per_type_thresholds={}, ambiguity_margin=0.0),
        rules_path=rules_path,
    )
    strict = ContextualTypeResolver(
        NERConfig(default_threshold=0.99, per_type_thresholds={}, ambiguity_margin=0.0),
        rules_path=rules_path,
    )

    assert permissive.resolve(mention, "ho", chunk).entity_type == "TRIỆU_CHỨNG"
    assert strict.resolve(mention, "ho", chunk).entity_type is None


def test_context_signal_is_recognized_next_to_punctuation(tmp_path):
    rules_path = _write_json(tmp_path / "rules.json", _rules_payload())
    text = "Thuốc: aspirin"
    start = text.index("aspirin")
    mention = MentionCandidate("aspirin", start, len(text), {"THUỐC"}, {"unverified"}, True)
    resolver = ContextualTypeResolver(
        NERConfig(default_threshold=0.75, per_type_thresholds={}, ambiguity_margin=0.0),
        rules_path=rules_path,
    )

    decision = resolver.resolve(
        mention, text, ClinicalChunk(text, 0, len(text), "unknown", "")
    )

    assert decision.entity_type == "THUỐC"


def test_extractor_uses_injected_clinical_lexicon(tmp_path):
    path = _write_json(
        tmp_path / "clinical_lexicon.json",
        {"schema_version": 1, "entries": [_entry("xét nghiệm zeta", "TÊN_XÉT_NGHIỆM", "verified-test")]},
    )
    extractor = BaselineExtractor(load_database=False, clinical_lexicon_path=path)

    entities = extractor.extract_entities("Kết quả xét nghiệm\nxét nghiệm zeta: 1.2")

    assert any(entity["text"] == "xét nghiệm zeta" for entity in entities)


def test_load_database_false_is_lightweight_and_injectable(tmp_path, monkeypatch):
    path = _write_json(
        tmp_path / "clinical_lexicon.json",
        {"schema_version": 1, "entries": [_entry("zeta medicine", "THUỐC", "verified-test")]},
    )
    monkeypatch.setattr(
        "src.ner.extractor.sqlite3.connect",
        lambda *_args, **_kwargs: pytest.fail("database connection attempted"),
    )

    extractor = BaselineExtractor(load_database=False, clinical_lexicon_path=path)

    assert extractor.extract_entities("Thuốc hiện tại\nzeta medicine 25mg po")


def test_lexicon_requires_source_and_status(tmp_path):
    path = _write_json(
        tmp_path / "lexicon.json",
        {"schema_version": 1, "entries": [{"term": "ho", "type": "TRIỆU_CHỨNG"}]},
    )

    with pytest.raises(ValueError, match="source|status"):
        ClinicalLexicon.load(path)


def test_lexicon_deduplicates_normalized_term_and_type_using_strongest_provenance(tmp_path):
    path = _write_json(
        tmp_path / "lexicon.json",
        {
            "schema_version": 1,
            "entries": [
                _entry(" Ho ", source="legacy", status="unverified"),
                _entry("ho", source="curated", status="verified"),
                _entry("HO", source="other", status="unverified"),
            ],
        },
    )

    terms = ClinicalLexicon.load(path)

    assert len(terms) == 1
    assert (terms[0].source, terms[0].status) == ("curated", "verified")


@pytest.mark.parametrize(
    "payload,match",
    [
        ({"schema_version": 1, "entries": [], "extra": 1}, "unknown"),
        ({"schema_version": True, "entries": []}, "schema_version"),
        ({"schema_version": 1, "entries": [_entry() | {"extra": 1}]}, "unknown"),
        ({"schema_version": 1, "entries": [_entry(term=True)]}, "term"),
        ({"schema_version": 1, "entries": [_entry(term=" ")]}, "term"),
        ({"schema_version": 1, "entries": [_entry(source=" ")]}, "source"),
        ({"schema_version": 1, "entries": [_entry(status="trusted")]}, "status"),
        ({"schema_version": 1, "entries": [_entry(entity_type="DRUG")]}, "type"),
    ],
)
def test_lexicon_rejects_unknown_keys_wrong_types_and_invalid_values(tmp_path, payload, match):
    path = _write_json(tmp_path / "lexicon.json", payload)

    with pytest.raises(ValueError, match=match):
        ClinicalLexicon.load(path)


def test_lexicon_rejects_malformed_json(tmp_path):
    path = tmp_path / "lexicon.json"
    path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="JSON"):
        ClinicalLexicon.load(path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("extra", 1), "unknown"),
        (("schema_version", True), "schema_version"),
        (("medication_signals", [True]), "medication_signals"),
        (("laboratory_units", []), "laboratory_units"),
        (("frequency_patterns", []), "frequency_patterns"),
        (("frequency_patterns", [{"pattern": True, "source": "test"}]), "pattern"),
        (("frequency_patterns", [{"pattern": "ngày {count} lần", "source": "test"}]), "placeholder"),
        (("frequency_patterns", [{"pattern": "ngày {number} lần", "source": "test", "extra": 1}]), "unknown"),
        (("source_confidence", {"verified": True}), "source_confidence"),
        (("weights", {"exact": 0.5}), "weights"),
    ],
)
def test_type_rules_reject_unknown_keys_wrong_types_and_incomplete_weights(tmp_path, mutation, match):
    payload = _rules_payload()
    payload[mutation[0]] = mutation[1]
    path = _write_json(tmp_path / "rules.json", payload)

    with pytest.raises(ValueError, match=match):
        TypeRules.load(path)


def test_type_rules_are_deeply_immutable(tmp_path):
    path = _write_json(tmp_path / "rules.json", _rules_payload())

    rules = TypeRules.load(path)

    assert isinstance(rules.section_priors, MappingProxyType)
    assert isinstance(rules.section_priors["laboratory"], MappingProxyType)
    with pytest.raises(TypeError):
        rules.section_priors["laboratory"]["THUỐC"] = 0.0


def test_directly_injected_type_rules_are_validated_and_deeply_immutable():
    sections = {"laboratory": {"TÊN_XÉT_NGHIỆM": 1.0}}
    source_confidence = {"verified": 1.5}
    weights = _rules_payload()["weights"]

    rules = TypeRules(
        sections,
        ["thuốc"],
        ["mg/dl"],
        ["mg"],
        ["po"],
        [FrequencyPattern(("ngày", "{number}", "lần"), "verified-test")],
        ["yếu"],
        source_confidence,
        weights,
    )
    sections["laboratory"]["TÊN_XÉT_NGHIỆM"] = -9.0
    source_confidence["verified"] = -9.0

    assert rules.section_priors["laboratory"]["TÊN_XÉT_NGHIỆM"] == 1.0
    assert rules.source_confidence["verified"] == 1.5
    assert isinstance(rules.medication_signals, tuple)
    with pytest.raises(ValueError, match="source_confidence"):
        TypeRules(
            sections,
            ["thuốc"],
            ["mg/dl"],
            ["mg"],
            ["po"],
            [FrequencyPattern(("ngày", "{number}", "lần"), "verified-test")],
            ["yếu"],
            {"verified": True},
            weights,
        )


def test_trie_aggregates_candidate_types_and_sources_for_same_span():
    matcher = TrieMatcher()
    matcher.insert("creatinine", "THUỐC", "rxnorm")
    matcher.insert("creatinine", "TÊN_XÉT_NGHIỆM", "lab")

    mentions = matcher.search_in_text("creatinine")

    assert mentions == [
        MentionCandidate(
            "creatinine",
            0,
            len("creatinine"),
            frozenset({"THUỐC", "TÊN_XÉT_NGHIỆM"}),
            frozenset({"rxnorm", "lab"}),
            True,
        )
    ]


def test_resolver_uses_only_sources_supporting_each_candidate_type(tmp_path):
    matcher = TrieMatcher()
    matcher.insert("creatinine", "THUỐC", "unverified-drug")
    matcher.insert("creatinine", "TÊN_XÉT_NGHIỆM", "verified-lab")
    mention = matcher.search_in_text("creatinine")[0]
    rules_path = _write_json(tmp_path / "rules.json", _rules_payload())
    resolver = ContextualTypeResolver(
        NERConfig(default_threshold=0.1, per_type_thresholds={}, ambiguity_margin=0.0),
        rules_path=rules_path,
        source_statuses={"unverified-drug": "unverified", "verified-lab": "verified"},
        source_lookup=matcher.sources_for,
    )

    decision = resolver.resolve(
        mention, "creatinine", ClinicalChunk("creatinine", 0, 10, "unknown", "")
    )

    assert decision.scores["TÊN_XÉT_NGHIỆM"] > decision.scores["THUỐC"]
    assert decision.entity_type == "TÊN_XÉT_NGHIỆM"


@pytest.mark.parametrize("text", ["siêuyếu", "yếu_tố", "precreatinine", "creatinine2"])
def test_trie_enforces_unicode_aware_word_boundaries(text):
    matcher = TrieMatcher()
    matcher.insert("yếu", "TRIỆU_CHỨNG", "test")
    matcher.insert("creatinine", "TÊN_XÉT_NGHIỆM", "test")

    assert matcher.search_in_text(text) == []


def test_overlapping_chunks_do_not_duplicate_entities(tmp_path):
    path = _write_json(
        tmp_path / "lexicon.json",
        {"schema_version": 1, "entries": [_entry("ho kéo dài", source="verified-test")]},
    )
    text = "Triệu chứng hiện tại\n" + "mệt " * 20 + "ho kéo dài " + "mệt " * 20
    chunks = ClinicalChunker(ChunkingConfig(max_tokens=20, overlap_tokens=10)).chunk(text)
    extractor = BaselineExtractor(load_database=False, clinical_lexicon_path=path)

    entities = extractor.extract_entities(text, chunks=chunks)

    matches = [entity for entity in entities if entity["text"] == "ho kéo dài"]
    assert len(matches) == 1
    assert text[slice(*matches[0]["position"])] == matches[0]["text"]


def test_drug_dose_expansion_preserves_exact_document_slice_and_offsets(tmp_path):
    path = _write_json(
        tmp_path / "lexicon.json",
        {"schema_version": 1, "entries": [_entry("metoprolol", "THUỐC", "verified-test")]},
    )
    text = "Thuốc hiện tại\n- metoprolol 25mg po bid\n"
    extractor = BaselineExtractor(load_database=False, clinical_lexicon_path=path)

    entity = next(item for item in extractor.extract_entities(text) if item["type"] == "THUỐC")

    start, end = entity["position"]
    assert entity["text"] == text[start:end]
    assert entity["text"] == "metoprolol 25mg po bid"


def test_drug_dose_expansion_does_not_treat_prose_as_a_unit(tmp_path):
    path = _write_json(
        tmp_path / "lexicon.json",
        {"schema_version": 1, "entries": [_entry("aspirin", "THUỐC", "verified-test")]},
    )
    text = "Thuốc hiện tại\naspirin 25 bệnh nhân ổn định"
    extractor = BaselineExtractor(load_database=False, clinical_lexicon_path=path)

    entity = next(item for item in extractor.extract_entities(text) if item["type"] == "THUỐC")

    assert entity["text"] == "aspirin 25"


def test_default_rules_expand_compositional_frequency_to_exact_full_span():
    text = "Thuốc hiện tại\naspirin 25 mg ngày 2 lần"
    extractor = BaselineExtractor(load_database=False)

    entity = next(item for item in extractor.extract_entities(text) if item["type"] == "THUỐC")

    assert entity["text"] == "aspirin 25 mg ngày 2 lần"
    assert entity["text"] == text[slice(*entity["position"])]


def test_dose_instruction_does_not_match_route_prefix_inside_word():
    text = "Thuốc hiện tại\naspirin 25 pokemon"
    extractor = BaselineExtractor(load_database=False)

    entity = next(item for item in extractor.extract_entities(text) if item["type"] == "THUỐC")

    assert entity["text"] == "aspirin 25"


def test_injected_route_vocabulary_has_no_hidden_instruction_fallback(tmp_path):
    lexicon_path = _write_json(
        tmp_path / "lexicon.json",
        {"schema_version": 1, "entries": [_entry("aspirin", "THUỐC", "verified-test")]},
    )
    payload = _rules_payload()
    payload["route_terms"] = ["po"]
    rules_path = _write_json(tmp_path / "rules.json", payload)
    extractor = BaselineExtractor(
        load_database=False,
        clinical_lexicon_path=lexicon_path,
        type_rules_path=rules_path,
    )

    entity = next(
        item
        for item in extractor.extract_entities("Thuốc hiện tại\naspirin 25 sáng")
        if item["type"] == "THUỐC"
    )

    assert entity["text"] == "aspirin 25"


def test_frequency_grammar_requires_source_and_is_deeply_immutable(tmp_path):
    payload = _rules_payload()
    path = _write_json(tmp_path / "rules.json", payload)

    rules = TypeRules.load(path)

    assert rules.frequency_patterns[0].tokens == ("ngày", "{number}", "lần")
    assert rules.frequency_patterns[0].source == "verified-test"
    with pytest.raises((AttributeError, TypeError)):
        rules.frequency_patterns[0].tokens = ("changed",)

    payload["frequency_patterns"] = [{"pattern": "ngày {number} lần"}]
    _write_json(path, payload)
    with pytest.raises(ValueError, match="source"):
        TypeRules.load(path)


def test_extractor_self_check_runs_as_a_direct_script():
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [sys.executable, "src/ner/extractor.py"],
        cwd=project_root,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
