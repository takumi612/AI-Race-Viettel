from __future__ import annotations

import json
import re
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

from clinical_nlp_lab.qwen_entity_validator import (
    EntityValidationCounters,
    QwenEntityValidator,
    _parse_decision,
    _validation_schema,
)
from clinical_nlp_lab.qwen_refiner import RequiredQwenError, RequiredQwenRefiner
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation, OFFICIAL_SCHEMA_KEYS


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


@pytest.mark.parametrize(
    ("relative_start", "relative_end"),
    [
        (1, len(ner_entity().text)),
        (0, len(ner_entity().text) - 1),
    ],
)
def test_word_boundary_invalid_trim_preserves_original_entity(
    fake_engine, relative_start, relative_end
):
    original = ner_entity()
    response = json.dumps(
        {
            "action": "trim",
            "relative_start": relative_start,
            "relative_end": relative_end,
            "entity_type": "DISEASE",
        }
    )

    result = QwenEntityValidator(fake_engine([response])).validate(
        (original,), RAW_TEXT
    )

    assert result.entities == (original,)
    assert result.entities[0].position == original.position
    assert result.entities[0].candidates == original.candidates
    assert result.entities[0].assertions == original.assertions


def test_prompt_json_examples_are_valid_under_the_runtime_parser():
    entity = ner_entity()
    prompt = QwenEntityValidator._build_prompt(RAW_TEXT, entity)
    examples = re.findall(r"\{[^{}\r\n]+\}", prompt)

    decisions = [_parse_decision(example, entity) for example in examples]

    assert [decision.action for decision in decisions] == ["keep", "drop", "trim"]
    trim = decisions[-1]
    assert trim.relative_start == 0
    assert trim.relative_end < len(entity.text)
    assert not (
        entity.text[trim.relative_end - 1].isalnum()
        and entity.text[trim.relative_end].isalnum()
    )


def test_guided_schema_requires_fields_for_each_action():
    schema = _validation_schema(ner_entity())
    variants = {
        variant["properties"]["action"]["enum"][0]: variant
        for variant in schema["oneOf"]
    }

    assert set(variants) == {"keep", "drop", "trim"}
    assert variants["keep"]["required"] == ["action"]
    assert variants["drop"]["required"] == ["action"]
    assert set(variants["trim"]["required"]) == {
        "action",
        "relative_start",
        "relative_end",
        "entity_type",
    }
    assert all(variant["additionalProperties"] is False for variant in variants.values())
    assert variants["trim"]["properties"]["relative_start"]["minimum"] == 0
    assert variants["trim"]["properties"]["relative_end"]["maximum"] == len(ner_entity().text)


@pytest.mark.parametrize(
    ("relative_start", "relative_end"),
    [
        (0, len("risk of pneumonia")),
        (ner_entity().text.index("pneumonia"), len(ner_entity().text)),
    ],
)
def test_parse_decision_allows_one_sided_trim(relative_start, relative_end):
    entity = ner_entity()
    decision = _parse_decision(
        json.dumps(
            {
                "action": "trim",
                "relative_start": relative_start,
                "relative_end": relative_end,
                "entity_type": "DISEASE",
            }
        ),
        entity,
    )

    assert decision.relative_start == relative_start
    assert decision.relative_end == relative_end


def test_parse_decision_rejects_no_op_trim():
    entity = ner_entity()
    response = json.dumps(
        {
            "action": "trim",
            "relative_start": 0,
            "relative_end": len(entity.text),
            "entity_type": "DISEASE",
        }
    )

    with pytest.raises(ValueError, match="shorten"):
        _parse_decision(response, entity)


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

    result = QwenEntityValidator(
        fake_engine(
            ['{"action":"trim","relative_start":1,"relative_end":4,"entity_type":"DISEASE"}']
        )
    ).validate((original,), raw_text)

    assert result.entities == (original,)
    assert result.counters.to_dict()["trim_fallback_reasons"] == {
        "whitespace_only": 1
    }


def test_punctuation_only_trim_preserves_original_and_records_reason(fake_engine):
    original = EntityAnnotation(text="pain - fever", type="SYMPTOM", position=(0, 12))
    engine = fake_engine(
        ['{"action":"trim","relative_start":5,"relative_end":6,"entity_type":"SYMPTOM"}']
    )

    result = QwenEntityValidator(engine).validate((original,), "pain - fever")

    assert result.entities == (original,)
    assert result.counters.to_dict()["trim_fallback_reasons"] == {
        "punctuation_only": 1
    }


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

    assert after.delta(before).to_dict() == increment.to_dict()


def test_validation_counters_use_document_length_buckets(fake_engine):
    raw_text = "a" * 50 + " " + "b" * 100 + " " + "c" * 101
    entities = (
        EntityAnnotation(text="a" * 50, type="DISEASE", position=(0, 50)),
        EntityAnnotation(text="b" * 100, type="DISEASE", position=(51, 151)),
        EntityAnnotation(text="c" * 101, type="DISEASE", position=(152, 253)),
    )

    result = QwenEntityValidator(
        fake_engine(['{"action":"keep"}', '{"action":"keep"}', '{"action":"keep"}'])
    ).validate(entities, raw_text)

    expected = {"1-50": 1, "51-100": 1, "101+": 1}
    assert result.counters.before_length_buckets == expected
    assert result.counters.after_length_buckets == expected
    assert result.counters.max_before_length == 101
    assert result.counters.max_after_length == 101


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

    counters["before_type_counts"]["MUTATED"] = 999
    assert "MUTATED" not in refiner.validation_counters()["before_type_counts"]


class _CounterSnapshotRefiner:
    def __init__(self):
        self._counters = EntityValidationCounters()

    def validation_counters(self):
        return deepcopy(self._counters.to_dict())

    def validation_counter_snapshot(self):
        return self._counters.copy()

    def validate_one_document(self):
        self._counters.add(
            EntityValidationCounters(
                query_count=1,
                keep=1,
                before_type_counts={"SYMPTOM": 1},
                after_type_counts={"SYMPTOM": 1},
                before_length_buckets={"1-50": 1},
                after_length_buckets={"1-50": 1},
                max_before_length=5,
                max_after_length=5,
            )
        )


def test_pipeline_writes_per_document_qwen_validation_counter_delta(tmp_path: Path, monkeypatch):
    from clinical_nlp_lab.pipeline import run_inference_with_bundle

    raw_text = "fever"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "001.txt").write_text(raw_text, encoding="utf-8")
    refiner = _CounterSnapshotRefiner()

    def fake_infer(_document_id, _raw_text, _bundle, _config):
        refiner.validate_one_document()
        return ClinicalDocument(
            "001",
            raw_text,
            entities=[EntityAnnotation(text="fever", type="SYMPTOM", position=(0, 5))],
        )

    monkeypatch.setattr("clinical_nlp_lab.inference.infer_document", fake_infer)
    run_inference_with_bundle(
        input_dir,
        tmp_path / "output",
        bundle=types.SimpleNamespace(qwen_reranker=refiner, ner_model=None),
        entity_mapping={
            "internal_to_official": {"SYMPTOM": next(key for key in OFFICIAL_SCHEMA_KEYS if "TRI" in key)},
            "drop_unmapped": True,
        },
        create_zip=False,
    )

    diagnostic = json.loads(
        (tmp_path / "diagnostics" / "001.json").read_text(encoding="utf-8")
    )
    assert diagnostic["qwen_entity_validation"] == {
        "query_count": 1,
        "keep": 1,
        "drop": 0,
        "trim": 0,
        "kb_bypass": 0,
        "before_type_counts": {"SYMPTOM": 1},
        "after_type_counts": {"SYMPTOM": 1},
        "before_length_buckets": {"1-50": 1},
        "after_length_buckets": {"1-50": 1},
        "max_before_length": 5,
        "max_after_length": 5,
        "trim_fallback_reasons": {},
    }


def test_required_refiner_wraps_invalid_entity_validation(fake_engine):
    refiner = RequiredQwenRefiner(fake_engine(["not JSON"]))

    with pytest.raises(RequiredQwenError, match="malformed JSON"):
        refiner.validate_entities((ner_entity(),), RAW_TEXT)
