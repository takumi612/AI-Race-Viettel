from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from clinical_nlp_lab.data import ClinicalDocument
from clinical_nlp_lab.pipeline import run_inference_with_bundle
from clinical_nlp_lab.qwen_refiner import RequiredQwenError, RequiredQwenRefiner
from clinical_nlp_lab.schema import EntityAnnotation, validate_submission_payload


ROOT = Path(__file__).parents[1]


class FakeEngine:
    def __init__(self, response_batches: list[list[str]], *, destroy_error: Exception | None = None):
        self.response_batches = list(response_batches)
        self.destroy_error = destroy_error
        self.destroyed = False

    def generate(self, prompts, sampling_params, use_tqdm=False):
        response_texts = self.response_batches.pop(0)
        return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text=text)]) for text in response_texts]

    def destroy(self):
        self.destroyed = True
        if self.destroy_error is not None:
            raise self.destroy_error


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch):
    class FakeSamplingParams:
        def __init__(self, *, temperature, max_tokens):
            self.temperature = temperature
            self.max_tokens = max_tokens

    monkeypatch.setitem(sys.modules, "vllm", types.SimpleNamespace(SamplingParams=FakeSamplingParams))


def _entity(*, candidates: list[str] | None = None) -> EntityAnnotation:
    return EntityAnnotation(
        text="fever",
        type="DISEASE",
        position=(8, 13),
        candidates=candidates or ["A01"],
        ranked_candidates=[
            {"candidate_id": "A01", "name": "first"},
            {"candidate_id": "B02", "name": "second"},
        ],
    )


def _assertion_response() -> str:
    return '{"polarity":"NEGATED","temporality":"CURRENT","certainty":"CONFIRMED","experiencer":"PATIENT"}'


def test_refiner_selects_only_an_id_from_the_preserved_pool():
    engine = FakeEngine([['{"selected_id":"B02"}'], [_assertion_response()]])

    refined = RequiredQwenRefiner(engine).refine((_entity(),), "Patient fever today.")

    assert refined[0].candidates == ["B02"]
    assert refined[0].assertions == ["isNegated"]


def test_refiner_assertions_survive_real_mapping_and_submission_validation(tmp_path: Path, monkeypatch):
    raw_text = "Patient fever today."
    engine = FakeEngine([
        ['{"selected_id":"B02"}'],
        ['{"polarity":"NEGATED","temporality":"HISTORICAL","certainty":"POSSIBLE","experiencer":"FAMILY"}'],
    ])
    refined = RequiredQwenRefiner(engine).refine((_entity(),), raw_text)
    document = ClinicalDocument("001", raw_text, entities=list(refined))
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "001.txt").write_text(raw_text, encoding="utf-8")
    monkeypatch.setattr("clinical_nlp_lab.inference.infer_document", lambda *_args: document)

    run_inference_with_bundle(
        input_dir,
        tmp_path / "output",
        bundle=object(),
        entity_mapping=json.loads((ROOT / "artifacts" / "entity_type_mapping.json").read_text(encoding="utf-8")),
        assertion_mapping=json.loads((ROOT / "artifacts" / "assertion_mapping.json").read_text(encoding="utf-8")),
        create_zip=False,
    )

    payload = json.loads((tmp_path / "output" / "001.json").read_text(encoding="utf-8"))
    assert payload[0]["assertions"] == ["isNegated", "isHistorical", "isFamily"]
    assert validate_submission_payload(payload, raw_text) == []


def test_refiner_accepts_null_as_an_abstention():
    entity = _entity(candidates=["A01"])
    engine = FakeEngine([['{"selected_id":null}'], [_assertion_response()]])

    refined = RequiredQwenRefiner(engine).refine((entity,), "Patient fever today.")

    assert refined[0].candidates == []


def test_refiner_rejects_an_unknown_candidate_id():
    engine = FakeEngine([['{"selected_id":"NOT-IN-POOL"}']])

    with pytest.raises(RequiredQwenError, match="unknown selected_id"):
        RequiredQwenRefiner(engine).refine((_entity(),), "Patient fever today.")


def test_refiner_rejects_a_malformed_qwen_response():
    engine = FakeEngine([["not JSON"]])

    with pytest.raises(RequiredQwenError, match="malformed JSON"):
        RequiredQwenRefiner(engine).refine((_entity(),), "Patient fever today.")


def test_refiner_rejects_an_invalid_assertion_enum():
    engine = FakeEngine([
        ['{"selected_id":"A01"}'],
        ['{"polarity":"MAYBE","temporality":"CURRENT","certainty":"CONFIRMED","experiencer":"PATIENT"}'],
    ])

    with pytest.raises(RequiredQwenError, match="invalid polarity"):
        RequiredQwenRefiner(engine).refine((_entity(),), "Patient fever today.")


def test_refiner_rejects_malformed_assertion_json():
    engine = FakeEngine([['{"selected_id":"A01"}'], ["not JSON"]])

    with pytest.raises(RequiredQwenError, match="malformed JSON assertion response"):
        RequiredQwenRefiner(engine).refine((_entity(),), "Patient fever today.")


def test_refiner_rejects_a_response_count_mismatch():
    engine = FakeEngine([[]])

    with pytest.raises(RequiredQwenError, match="returned 0 outputs"):
        RequiredQwenRefiner(engine).refine((_entity(),), "Patient fever today.")


def test_refiner_preserves_text_type_and_raw_offsets():
    entity = _entity()
    engine = FakeEngine([['{"selected_id":"B02"}'], [_assertion_response()]])

    refined = RequiredQwenRefiner(engine).refine((entity,), "Patient fever today.")

    assert (refined[0].text, refined[0].type, refined[0].position) == ("fever", "DISEASE", (8, 13))
    assert (entity.text, entity.type, entity.position, entity.candidates) == ("fever", "DISEASE", (8, 13), ["A01"])


def test_refiner_destroy_releases_the_engine_and_propagates_errors():
    cleanup_error = RuntimeError("cleanup failed")
    engine = FakeEngine([], destroy_error=cleanup_error)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        RequiredQwenRefiner(engine).destroy()

    assert engine.destroyed is True
