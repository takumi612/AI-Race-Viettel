import importlib.util
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_assertions():
    package = types.ModuleType("clinical_nlp_lab")
    package.__path__ = [str(ROOT / "clinical_nlp_lab")]
    sys.modules["clinical_nlp_lab"] = package
    _load_module("clinical_nlp_lab.schema", ROOT / "clinical_nlp_lab" / "schema.py")
    _load_module("clinical_nlp_lab.text", ROOT / "clinical_nlp_lab" / "text.py")
    return _load_module("clinical_nlp_lab.assertions", ROOT / "clinical_nlp_lab" / "assertions.py")


def test_assertion_predictor_does_not_use_removed_guided_json(monkeypatch):
    assertions = _load_assertions()

    class FakeSamplingParams:
        def __init__(self, *, temperature, max_tokens, structured_outputs=None):
            self.structured_outputs = structured_outputs

    class FakeStructuredOutputsParams:
        def __init__(self, *, json):
            self.json = json

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_sampling = types.ModuleType("vllm.sampling_params")
    fake_sampling.StructuredOutputsParams = FakeStructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", fake_sampling)

    output = types.SimpleNamespace(outputs=[types.SimpleNamespace(text='```json\n{"polarity":"NEGATED","temporality":"CURRENT","certainty":"CONFIRMED","experiencer":"PATIENT"}\n```')])

    class FakeLLM:
        def generate(self, prompts, sampling_params, use_tqdm=False):
            return [output for _ in prompts]

    predictor = assertions.ClinicalLLMAssertionPredictor(FakeLLM())
    result = predictor.predict_batch([{"context": "không sốt", "entity_text": "sốt"}])

    assert result[0].polarity == "NEGATED"

