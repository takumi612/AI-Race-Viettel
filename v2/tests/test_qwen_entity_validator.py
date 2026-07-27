from __future__ import annotations

import sys
import types

import pytest

from clinical_nlp_lab.qwen_entity_validator import EntityValidationCounters, QwenEntityValidator
from clinical_nlp_lab.qwen_refiner import RequiredQwenError, RequiredQwenRefiner
from clinical_nlp_lab.schema import EntityAnnotation


class FakeEngine:
    def __init__(self, responses: list[list[str]]):
        self.responses = list(responses)
        self.generate_calls = 0

    def generate(self, prompts, sampling_params, use_tqdm=False):
        self.generate_calls += 1
        batch = self.responses.pop(0)
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]) for text in batch]


@pytest.fixture
def fake_engine(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=FakeSamplingParams))
    return lambda responses: FakeEngine([responses])


RAW_TEXT = "Patient has risk of pneumonia because of prolonged bed rest."


def ner_entity() -> EntityAnnotation:
    text = "risk of pneumonia because of prolonged bed rest"
    start = RAW_TEXT.index(text)
    return EntityAnnotation(
        text=text,
        type="DISEASE",
        position=(start, start + len(text)),
        candidates=["J18.9"],
        assertions=["isHistorical"],
        evidence=["proposal_ner"],
    )


def kb_entity(score: float = 1.0) -> EntityAnnotation:
    text = "pneumonia"
    start = RAW_TEXT.index(text)
    return EntityAnnotation(
        text=text,
        type="DISEASE",
        position=(start, start + len(text)),
        evidence=["proposal_kb_first"],
        ranked_candidates=[{"candidate_id": "J18.9", "name": text, "score": score}],
    )


def test_trim_translates_relative_offsets_and_clears_metadata(fake_engine):
    original = ner_entity()
    start = original.text.index("pneumonia")
    engine = fake_engine([
        '{"action":"trim","relative_start":%d,"relative_end":%d,"entity_type":"DISEASE"}'
        % (start, start + len("pneumonia"))
    ])

    result = QwenEntityValidator(engine).validate((original,), RAW_TEXT)

    assert result.entities[0].text == "pneumonia"
    assert result.entities[0].position == (20, 29)
    assert result.entities[0].candidates == []
    assert result.entities[0].assertions == []
    assert result.entities[0].evidence == ["proposal_ner"]


def test_keep_preserves_the_original_entity_and_raw_offsets(fake_engine):
    original = ner_entity()

    result = QwenEntityValidator(fake_engine(['{"action":"keep"}'])).validate((original,), RAW_TEXT)

    assert result.entities == (original,)
    assert result.entities[0].position == (12, 59)
    assert RAW_TEXT[slice(*result.entities[0].position)] == result.entities[0].text
    assert result.counters.keep == 1


def test_drop_removes_the_entity(fake_engine):
    result = QwenEntityValidator(fake_engine(['{"action":"drop"}'])).validate((ner_entity(),), RAW_TEXT)

    assert result.entities == ()
    assert result.counters.drop == 1


@pytest.mark.parametrize(
    "response",
    [
        "not json",
        '{"action":"extend","entity_type":"DISEASE"}',
        '{"action":"trim","relative_start":-1,"relative_end":4,"entity_type":"DISEASE"}',
        '{"action":"trim","relative_start":0,"relative_end":999,"entity_type":"DISEASE"}',
        '{"action":"trim","relative_start":0,"relative_end":4,"entity_type":"UNKNOWN"}',
    ],
)
def test_invalid_entity_decision_is_rejected(fake_engine, response):
    with pytest.raises(ValueError):
        QwenEntityValidator(fake_engine([response])).validate((ner_entity(),), RAW_TEXT)


def test_whitespace_only_trim_is_rejected(fake_engine):
    original = EntityAnnotation(text="x   y", type="DISEASE", position=(0, 5))
    raw_text = "x   y"

    with pytest.raises(ValueError, match="whitespace"):
        QwenEntityValidator(fake_engine(['{"action":"trim","relative_start":1,"relative_end":4,"entity_type":"DISEASE"}'])).validate(
            (original,), raw_text
        )


def test_response_count_mismatch_is_rejected(fake_engine):
    with pytest.raises(ValueError, match="returned 0 outputs"):
        QwenEntityValidator(fake_engine([])).validate((ner_entity(),), RAW_TEXT)


def test_exact_kb_entity_bypasses_qwen(fake_engine):
    entity = kb_entity()
    engine = fake_engine([])

    result = QwenEntityValidator(engine).validate((entity,), RAW_TEXT)

    assert result.entities == (entity,)
    assert engine.generate_calls == 0
    assert result.counters.kb_bypass == 1


def test_non_exact_kb_entity_still_queries_qwen(fake_engine):
    entity = kb_entity(score=0.99)
    engine = fake_engine(['{"action":"keep"}'])

    result = QwenEntityValidator(engine).validate((entity,), RAW_TEXT)

    assert result.entities == (entity,)
    assert engine.generate_calls == 1
    assert result.counters.kb_bypass == 0


def test_counters_add_and_delta_are_deterministic():
    before = EntityValidationCounters(query_count=2, keep=1, drop=1)
    increment = EntityValidationCounters(query_count=3, keep=1, trim=2, kb_bypass=4)
    after = before.copy().add(increment)

    assert after.to_dict()["query_count"] == 5
    assert after.delta(before).to_dict()["query_count"] == 3
    assert after.delta(before).to_dict()["trim"] == 2


def test_counter_delta_does_not_leak_a_prior_document_maximum():
    cumulative = EntityValidationCounters()
    cumulative.add(EntityValidationCounters(max_before_length=150, max_after_length=150))
    before_short_document = cumulative.copy()
    cumulative.add(EntityValidationCounters(max_before_length=9, max_after_length=9))

    document_delta = cumulative.delta(before_short_document)

    assert document_delta.max_before_length == 9
    assert document_delta.max_after_length == 9


def test_required_refiner_exposes_validation_and_accumulates_counters(fake_engine):
    refiner = RequiredQwenRefiner(fake_engine(['{"action":"drop"}']))

    assert refiner.validate_entities((ner_entity(),), RAW_TEXT) == ()

    counters = refiner.validation_counters()
    assert counters["query_count"] == 1
    assert counters["drop"] == 1


def test_required_refiner_wraps_invalid_entity_validation(fake_engine):
    refiner = RequiredQwenRefiner(fake_engine(["not JSON"]))

    with pytest.raises(RequiredQwenError, match="malformed JSON"):
        refiner.validate_entities((ner_entity(),), RAW_TEXT)
