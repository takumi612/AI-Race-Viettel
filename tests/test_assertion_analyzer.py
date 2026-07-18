import json

import pytest

from src.assertion.rule_based import AssertionAnalyzer
from src.chunking.clinical_chunker import ClinicalChunker
from src.config import AssertionConfig, ChunkingConfig, PipelineConfig
from src.ner.extractor import BaselineExtractor
from src.ner.lexicon_loader import ClinicalTerm
from src.pipeline.main import BaselinePipeline


ASSERTION_ORDER = ["isNegated", "isHistorical", "isFamily"]
RULE_GROUPS = {
    "negation_cues",
    "compound_negation_cues",
    "negation_exclusions",
    "historical_cues",
    "family_cues",
    "post_patient_family_cues",
    "assertion_terminators",
    "scope_boundaries",
    "section_priors",
    "patient_return_cues",
    "confidence_weights",
}


@pytest.mark.parametrize(
    "text,entity",
    [
        ("Bệnh nhân không có ho", "ho"),
        ("Chưa phát hiện viêm phổi", "viêm phổi"),
        ("Bệnh nhân phủ nhận có đau ngực", "đau ngực"),
    ],
)
def test_negation_phrase_keeps_compound_cue(text, entity):
    start = text.index(entity)

    result = AssertionAnalyzer().analyze(text, start, start + len(entity))

    assert "isNegated" in result


@pytest.mark.parametrize(
    "text,entity",
    [
        ("Không ho.\n- đau ngực", "đau ngực"),
        ("Không ho\n• đau ngực", "đau ngực"),
    ],
)
def test_negation_does_not_cross_sentence_or_bullet_scope(text, entity):
    start = text.index(entity)

    result = AssertionAnalyzer().analyze(text, start, start + len(entity))

    assert "isNegated" not in result


@pytest.mark.parametrize(
    "section_type",
    ["pre_admission_medications", "past_medical_history"],
)
def test_historical_section_prior_is_applied(section_type):
    text = "Thuốc trước khi nhập viện\n- metoprolol 25mg po bid"
    start = text.index("metoprolol")

    result = AssertionAnalyzer().analyze(
        text,
        start,
        start + len("metoprolol 25mg po bid"),
        section_type=section_type,
        header_text="Thuốc trước khi nhập viện",
    )

    assert "isHistorical" in result


def test_family_section_prior_applies_before_patient_return():
    text = "Tiền sử gia đình\n- mẹ tăng huyết áp"
    start = text.index("tăng huyết áp")

    result = AssertionAnalyzer().analyze(
        text,
        start,
        start + len("tăng huyết áp"),
        section_type="family_history",
        header_text="Tiền sử gia đình",
    )

    assert "isFamily" in result


def test_family_prior_stops_when_context_returns_to_patient():
    text = "Tiền sử gia đình: mẹ tăng huyết áp. Bệnh nhân hiện đau ngực."
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text,
        start,
        start + len("đau ngực"),
        section_type="family_history",
        header_text="Tiền sử gia đình",
    )

    assert "isFamily" not in result


def test_patient_return_suppresses_generic_family_phrase_after_return():
    text = (
        "Tiền sử gia đình: mẹ tăng huyết áp.\n"
        "Bệnh nhân hiện sống cùng gia đình và đau ngực."
    )
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text,
        start,
        start + len("đau ngực"),
        section_type="family_history",
        header_text="Tiền sử gia đình",
    )

    assert "isFamily" not in result


def test_patient_return_keeps_a_genuine_later_kinship_cue():
    text = "Bệnh nhân kể mẹ tăng huyết áp"
    start = text.index("tăng huyết áp")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("tăng huyết áp")
    )

    assert "isFamily" in result


@pytest.mark.parametrize("ambiguous_subject", ["anh", "chị", "em"])
def test_ambiguous_post_patient_subject_is_not_a_family_cue(ambiguous_subject):
    text = f"Bệnh nhân cho biết {ambiguous_subject} đang đau ngực"
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("đau ngực")
    )

    assert "isFamily" not in result


@pytest.mark.parametrize(
    "kinship_phrase",
    [
        "bố",
        "cha",
        "mẹ",
        "anh trai",
        "chị gái",
        "em trai",
        "em gái",
        "người nhà",
    ],
)
def test_unambiguous_post_patient_kinship_phrase_is_a_family_cue(kinship_phrase):
    text = f"Bệnh nhân cho biết {kinship_phrase}, bị tăng huyết áp"
    start = text.index("tăng huyết áp")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("tăng huyết áp")
    )

    assert "isFamily" in result


def test_generic_family_context_after_patient_return_is_not_a_family_cue():
    text = "Bệnh nhân sống cùng gia đình và đau ngực"
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("đau ngực")
    )

    assert "isFamily" not in result


def test_patient_return_in_an_earlier_section_does_not_suppress_family_prior():
    text = "Bệnh nhân ổn định.\nTiền sử gia đình\n- mẹ tăng huyết áp"
    start = text.index("tăng huyết áp")

    result = AssertionAnalyzer().analyze(
        text,
        start,
        start + len("tăng huyết áp"),
        section_type="family_history",
        header_text="Tiền sử gia đình",
    )

    assert "isFamily" in result


def test_assertion_analyzer_uses_injected_rule_resource(project_root, tmp_path):
    default_path = project_root / "src/resources/assertion_rules.json"
    rules = json.loads(default_path.read_text(encoding="utf-8"))
    rules["negation_cues"].append("tuyệt đối vắng")
    custom_path = tmp_path / "assertion_rules.json"
    custom_path.write_text(
        json.dumps(rules, ensure_ascii=False), encoding="utf-8"
    )
    text = "Bệnh nhân tuyệt đối vắng đau ngực"
    start = text.index("đau ngực")

    result = AssertionAnalyzer(rules_path=custom_path).analyze(
        text, start, start + len("đau ngực")
    )

    assert "isNegated" in result


def test_default_rule_resource_is_strict_versioned_and_correct_utf8(project_root):
    path = project_root / "src/resources/assertion_rules.json"
    rules = json.loads(path.read_text(encoding="utf-8"))

    assert rules["version"] == 1
    assert set(rules) == {"version", "provenance", *RULE_GROUPS}
    assert set(rules["provenance"]) == RULE_GROUPS
    assert all(
        set(provenance) == {"source"}
        and isinstance(provenance["source"], str)
        and provenance["source"].strip()
        for provenance in rules["provenance"].values()
    )
    assert {"không có", "chưa phát hiện", "phủ nhận có"} <= set(
        rules["compound_negation_cues"]
    )
    serialized = json.dumps(rules, ensure_ascii=False)
    assert "khÃ´ng" not in serialized
    assert "\ufffd" not in serialized


def test_post_patient_family_resource_uses_unambiguous_kinship_phrases(project_root):
    path = project_root / "src/resources/assertion_rules.json"
    rules = json.loads(path.read_text(encoding="utf-8"))
    cues = set(rules["post_patient_family_cues"])

    assert not {"anh", "chị", "em"} & cues
    assert {
        "bố",
        "cha",
        "mẹ",
        "anh trai",
        "chị gái",
        "em trai",
        "em gái",
        "người nhà",
    } <= cues


@pytest.mark.parametrize("invalid_version", [1.0, True, "1"])
def test_rule_resource_rejects_non_integer_version(
    project_root, tmp_path, invalid_version
):
    default_path = project_root / "src/resources/assertion_rules.json"
    rules = json.loads(default_path.read_text(encoding="utf-8"))
    rules["version"] = invalid_version
    custom_path = tmp_path / "invalid_version_assertion_rules.json"
    custom_path.write_text(
        json.dumps(rules, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="version"):
        AssertionAnalyzer(rules_path=custom_path)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda rules: rules.update({"unexpected": []}), "unknown keys"),
        (
            lambda rules: rules["provenance"].pop("negation_cues"),
            "provenance",
        ),
        (
            lambda rules: rules["confidence_weights"].update(
                {"negation_cue": 1.1}
            ),
            "confidence",
        ),
    ],
)
def test_rule_resource_schema_errors_are_actionable(
    project_root, tmp_path, mutation, match
):
    default_path = project_root / "src/resources/assertion_rules.json"
    rules = json.loads(default_path.read_text(encoding="utf-8"))
    mutation(rules)
    custom_path = tmp_path / "invalid_assertion_rules.json"
    custom_path.write_text(
        json.dumps(rules, ensure_ascii=False), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=match):
        AssertionAnalyzer(rules_path=custom_path)


def test_rule_resource_is_loaded_once_and_exposed_immutably(project_root, tmp_path):
    source_path = project_root / "src/resources/assertion_rules.json"
    custom_path = tmp_path / "cached_assertion_rules.json"
    custom_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
    first = AssertionAnalyzer(rules_path=custom_path)

    assert isinstance(first.rules["negation_cues"], tuple)
    with pytest.raises(TypeError):
        first.rules["negation_cues"] = ()

    custom_path.write_text("{}", encoding="utf-8")
    second = AssertionAnalyzer(rules_path=custom_path)

    assert second.rules is first.rules


def test_assertion_analyzer_exposes_deterministic_calibratable_scores():
    text = "Tiền sử gia đình: mẹ trước đây không ho"
    start = text.index("ho")
    analyzer = AssertionAnalyzer()

    first = analyzer.score(text, start, start + len("ho"))
    second = analyzer.score(text, start, start + len("ho"))

    assert list(first) == ASSERTION_ORDER
    assert first == second
    assert all(0.0 <= score <= 1.0 for score in first.values())
    assert first["isNegated"] >= first["isHistorical"]
    assert analyzer.analyze(text, start, start + len("ho")) == ASSERTION_ORDER


def test_analyze_uses_pipeline_assertion_thresholds():
    config = PipelineConfig(
        assertion=AssertionConfig(
            negated_threshold=0.95,
            historical_threshold=0.95,
            family_threshold=0.95,
        )
    )
    text = "Bệnh nhân không ho"
    start = text.index("ho")

    result = AssertionAnalyzer(config=config).analyze(
        text, start, start + len("ho")
    )

    assert result == []


@pytest.mark.parametrize(
    "full_text,start,end,error_type",
    [
        (None, 0, 1, TypeError),
        ("ho", -1, 1, ValueError),
        ("ho", 0, 0, ValueError),
        ("ho", 1, 0, ValueError),
        ("ho", 0, 3, ValueError),
        ("ho", True, 1, ValueError),
    ],
)
def test_score_rejects_invalid_document_span_contract(
    full_text, start, end, error_type
):
    with pytest.raises(error_type):
        AssertionAnalyzer().score(full_text, start, end)


def test_pipeline_reuses_chunks_and_propagates_section_context_without_family_leakage(
    tmp_path,
):
    text = (
        "Tiền sử gia đình\n"
        "- mẹ tăng huyết áp.\n"
        "- Bệnh nhân hiện đau ngực."
    )
    file_path = tmp_path / "record.txt"
    file_path.write_text(text, encoding="utf-8")
    start = text.index("đau ngực")

    class CountingChunker:
        def __init__(self):
            self.delegate = ClinicalChunker()
            self.calls = 0
            self.last_chunks = None

        def chunk(self, document):
            self.calls += 1
            self.last_chunks = self.delegate.chunk(document)
            return self.last_chunks

    class StubPatientExtractor:
        @staticmethod
        def extract(document):
            return {}

    class RecordingExtractor:
        def __init__(self):
            self.chunks = None

        def extract_entities(self, document, chunks=None):
            self.chunks = chunks
            return [
                {
                    "text": document[start : start + len("đau ngực")],
                    "type": "TRIỆU_CHỨNG",
                    "position": [start, start + len("đau ngực")],
                }
            ]

    class RecordingAssertionAnalyzer:
        def __init__(self):
            self.delegate = AssertionAnalyzer()
            self.calls = []

        def analyze(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.delegate.analyze(*args, **kwargs)

    class StubValidator:
        @staticmethod
        def check_dual_codes(entities):
            return entities

    pipeline = BaselinePipeline.__new__(BaselinePipeline)
    pipeline.patient_extractor = StubPatientExtractor()
    pipeline.clinical_chunker = CountingChunker()
    pipeline.ner_extractor = RecordingExtractor()
    pipeline.assertion_analyzer = RecordingAssertionAnalyzer()
    pipeline.clinical_validator = StubValidator()

    result = pipeline.process_file(file_path)

    assert pipeline.clinical_chunker.calls == 1
    assert pipeline.ner_extractor.chunks is pipeline.clinical_chunker.last_chunks
    assert result[0]["assertions"] == []
    _, context = pipeline.assertion_analyzer.calls[0]
    assert context == {
        "section_type": "family_history",
        "header_text": "Tiền sử gia đình",
    }


def test_pipeline_maps_dose_expansion_by_entity_start_without_rechunking(tmp_path):
    text = "Thuốc hiện tại\nalpha beta gamma aspirin 25 mg po bid"
    file_path = tmp_path / "dose-expansion.txt"
    file_path.write_text(text, encoding="utf-8")

    class CountingChunker:
        def __init__(self):
            self.delegate = ClinicalChunker(
                ChunkingConfig(max_tokens=5, overlap_tokens=1)
            )
            self.calls = 0
            self.last_chunks = None

        def chunk(self, document):
            self.calls += 1
            self.last_chunks = self.delegate.chunk(document)
            return self.last_chunks

    class StubPatientExtractor:
        @staticmethod
        def extract(document):
            return {}

    class RecordingExtractor:
        def __init__(self):
            self.delegate = BaselineExtractor(
                load_database=False,
                clinical_terms=[
                    ClinicalTerm(
                        "aspirin", "aspirin", "THUỐC", "review", "verified"
                    )
                ],
            )
            self.chunks = None

        def extract_entities(self, document, chunks=None):
            self.chunks = chunks
            return self.delegate.extract_entities(document, chunks=chunks)

    class RecordingAssertionAnalyzer:
        def __init__(self):
            self.calls = []

        def analyze(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return []

    class StubNormalizer:
        @staticmethod
        def clean_text(value):
            return value

        @staticmethod
        def remove_dosage(value):
            return "aspirin"

        @staticmethod
        def expand_abbreviation(value):
            return value

    class StubRetriever:
        @staticmethod
        def retrieve(value, top_k):
            return []

    class StubValidator:
        @staticmethod
        def check_and_fix_candidates(entity, patient_info):
            return entity

        @staticmethod
        def check_dual_codes(entities):
            return entities

    pipeline = BaselinePipeline.__new__(BaselinePipeline)
    pipeline.patient_extractor = StubPatientExtractor()
    pipeline.clinical_chunker = CountingChunker()
    pipeline.ner_extractor = RecordingExtractor()
    pipeline.assertion_analyzer = RecordingAssertionAnalyzer()
    pipeline.normalizer = StubNormalizer()
    pipeline.rxnorm_retriever = StubRetriever()
    pipeline.clinical_validator = StubValidator()
    pipeline.override_dict = {}
    pipeline.llm_reranker = None

    result = pipeline.process_file(file_path)

    assert pipeline.clinical_chunker.calls == 1
    assert pipeline.ner_extractor.chunks is pipeline.clinical_chunker.last_chunks
    assert result[0]["text"] == "aspirin 25 mg po bid"
    args, context = pipeline.assertion_analyzer.calls[0]
    assert args[1:3] == tuple(result[0]["position"])
    assert context == {
        "section_type": "treatment_current_medications",
        "header_text": "Thuốc hiện tại",
    }


@pytest.mark.parametrize("marker", ["•", "-", "*"])
def test_negation_stops_at_nearest_inline_bullet(marker):
    text = f"Không ho {marker} đau ngực"
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("đau ngực")
    )

    assert "isNegated" not in result


def test_inline_bullet_keeps_negation_inside_its_own_item():
    text = "• không có đau ngực"
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("đau ngực")
    )

    assert "isNegated" in result


def test_hyphenated_word_is_not_treated_as_a_bullet_boundary():
    text = "Không triệu-chứng đau ngực"
    start = text.index("đau ngực")

    result = AssertionAnalyzer().analyze(
        text, start, start + len("đau ngực")
    )

    assert "isNegated" in result
