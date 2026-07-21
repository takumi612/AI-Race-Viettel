import json
import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).parents[1]
package = types.ModuleType("clinical_nlp_lab")
package.__path__ = [str(ROOT / "clinical_nlp_lab")]
sys.modules["clinical_nlp_lab"] = package
compat_spec = importlib.util.spec_from_file_location("clinical_nlp_lab.vllm_compat", ROOT / "clinical_nlp_lab" / "vllm_compat.py")
compat_module = importlib.util.module_from_spec(compat_spec)
sys.modules["clinical_nlp_lab.vllm_compat"] = compat_module
assert compat_spec.loader is not None
compat_spec.loader.exec_module(compat_module)
_SPEC = importlib.util.spec_from_file_location("clinical_nlp_lab.reranker", ROOT / "clinical_nlp_lab" / "reranker.py")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules["clinical_nlp_lab.reranker"] = _MODULE
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
_build_sampling_kwargs = _MODULE._build_sampling_kwargs
_parse_selected_id = _MODULE._parse_selected_id
_selection_warning_reason = _MODULE._selection_warning_reason
ClinicalLLMReranker = _MODULE.ClinicalLLMReranker


def test_parse_selected_id_accepts_fenced_json_and_candidate_id():
    response = "```json\n{\"selected_id\": \"K59.0\"}\n```"

    assert _parse_selected_id(response, [{"candidate_id": "K59.0"}]) == "K59.0"


def test_parse_selected_id_rejects_explanatory_or_unknown_id():
    response = "The best choice is K59.0."

    assert _parse_selected_id(response, [{"candidate_id": "I10"}]) is None


def test_selection_warning_ignores_valid_null_selection():
    assert _selection_warning_reason('{"selected_id": null}', [{"candidate_id": "I10"}]) is None


def test_selection_warning_reports_unknown_candidate_id():
    reason = _selection_warning_reason('{"selected_id": "K59.0"}', [{"candidate_id": "I10"}])

    assert reason == "unknown selected_id"


def test_build_sampling_kwargs_uses_supported_structured_outputs_keyword():
    class FakeSamplingParams:
        def __init__(self, *, temperature, max_tokens, structured_outputs=None):
            self.temperature = temperature
            self.max_tokens = max_tokens
            self.structured_outputs = structured_outputs

    kwargs = _build_sampling_kwargs(
        FakeSamplingParams,
        ["K59.0"],
        structured_outputs_factory=lambda **payload: payload,
    )

    assert kwargs["temperature"] == 0.0
    assert kwargs["max_tokens"] == 100
    assert "structured_outputs" in kwargs
    assert "guided_json" not in kwargs
    assert json.loads(kwargs["structured_outputs"]["json"])["properties"]["selected_id"]["enum"] == ["K59.0", None]


def test_destroy_tolerates_vllm_without_is_initialized(monkeypatch):
    parallel_state = types.ModuleType("vllm.distributed.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None
    distributed = types.ModuleType("vllm.distributed")
    distributed.__path__ = []
    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    monkeypatch.setitem(sys.modules, "vllm.distributed.parallel_state", parallel_state)

    reranker = ClinicalLLMReranker.__new__(ClinicalLLMReranker)
    reranker.llm = object()

    reranker.destroy()
    assert reranker.llm is None


def test_reranker_forwards_configured_gpu_memory_limit(monkeypatch):
    captured = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    ClinicalLLMReranker(
        model_name="Qwen/test-awq",
        max_model_len=1024,
        batch_size=8,
        gpu_memory_utilization=0.2,
    )

    assert captured["model"] == "Qwen/test-awq"
    assert captured["max_model_len"] == 1024
    assert captured["gpu_memory_utilization"] == 0.2
