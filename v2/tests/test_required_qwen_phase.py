from __future__ import annotations

import json
from pathlib import Path

import pytest

from clinical_nlp_lab import kaggle_phases
from clinical_nlp_lab.inference import FinalModelBundle, InferenceConfig, infer_document
from clinical_nlp_lab.orchestration import PHASES, RunConfig, _execute_phases
from clinical_nlp_lab.qwen_refiner import RequiredQwenError
from clinical_nlp_lab.schema import EntityAnnotation


class _FakeReranker:
    instances: list["_FakeReranker"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.llm = object()
        self.destroyed = False
        type(self).instances.append(self)

    def destroy(self):
        self.destroyed = True


class _FailingReranker(_FakeReranker):
    def __init__(self, **kwargs):
        raise RequiredQwenError("Qwen initialization failed")


class _CleanupFailingReranker(_FakeReranker):
    def destroy(self):
        super().destroy()
        raise RuntimeError("Qwen cleanup failed")


class _ExplodingRefiner:
    def refine(self, entities, raw_text):
        raise RequiredQwenError("Qwen generation failed")


def _phase_paths(tmp_path: Path, config_payload: dict[str, object] | None = None) -> tuple[RunConfig, dict[str, str]]:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "final_fit" / "ner_model").mkdir(parents=True)
    head_dir = run_dir / "artifacts" / "heads"
    head_dir.mkdir(parents=True)
    (head_dir / "candidate_calibration.json").write_text(
        '{"confidence_threshold": 0.5, "objective": "precision_first"}', encoding="utf-8"
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text(json.dumps(config_payload or {}), encoding="utf-8")
    (artifact_dir / "entity_type_mapping.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "assertion_mapping.json").write_text("{}", encoding="utf-8")
    input_source = tmp_path / "input.zip"
    input_source.write_bytes(b"placeholder")
    return (
        RunConfig(artifact_dir=artifact_dir, config_path=artifact_dir / "config.json", input_source=input_source),
        {"run_dir": str(run_dir)},
    )


def _patch_phase_dependencies(monkeypatch, *, pipeline):
    monkeypatch.setattr(kaggle_phases, "load_candidate_dictionary", lambda _path: [])
    monkeypatch.setattr("clinical_nlp_lab.runtime_bundle.load_final_model_bundle", lambda *args, **kwargs: FinalModelBundle(None, None, qwen_reranker=kwargs.get("qwen_reranker")))
    monkeypatch.setattr("clinical_nlp_lab.pipeline.run_inference_with_bundle", pipeline)
    monkeypatch.setattr("clinical_nlp_lab.output_quality.audit_submission_directory", lambda *_args: {})
    monkeypatch.setattr("clinical_nlp_lab.output_quality.enforce_output_quality", lambda _report: None)


def test_phase_12_disabled_never_initializes_qwen_and_reports_disabled(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path)
    seen: list[dict] = []

    def pipeline(**kwargs):
        seen.append(kwargs)
        return {"document_count": 0}

    _patch_phase_dependencies(monkeypatch, pipeline=pipeline)
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _FailingReranker)

    result = kaggle_phases._phase_12_inference(config, "phase_12_inference", context)

    assert seen[0]["config"].enable_qwen is False
    assert result["qwen_requested"] is False
    assert result["qwen_initialized"] is False
    assert result["qwen_model_name"] is None
    assert result["qwen_rerank_query_count"] == 0
    assert result["qwen_assertion_query_count"] == 0
    assert result["qwen_abstention_count"] == 0
    assert result["qwen_status"] == "DISABLED"


def test_phase_12_uses_run_config_toggle_despite_both_config_file_keys(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path, {"enable_qwen": False, "enable_qwen_reranker": False})
    config = RunConfig(**{**config.__dict__, "enable_qwen_reranker": True})
    _FakeReranker.instances.clear()
    seen: list[dict] = []

    def passthrough_refine(_self, entities, _raw_text):
        entities[0].candidates = []
        return entities

    def pipeline(**kwargs):
        seen.append(kwargs)
        kwargs["bundle"].qwen_reranker.refine(
            (
                EntityAnnotation(
                    text="fever",
                    type="DISEASE",
                    position=(0, 5),
                    candidates=["A01"],
                    ranked_candidates=[{"candidate_id": "A01", "name": "first"}],
                ),
            ),
            "fever",
        )
        return {"document_count": 0}

    _patch_phase_dependencies(monkeypatch, pipeline=pipeline)
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _FakeReranker)
    monkeypatch.setattr("clinical_nlp_lab.qwen_refiner.RequiredQwenRefiner.refine", passthrough_refine)

    result = kaggle_phases._phase_12_inference(config, "phase_12_inference", context)

    assert _FakeReranker.instances[0].kwargs == {
        "model_name": "Qwen/Qwen2.5-7B-Instruct-AWQ",
        "gpu_memory_utilization": 0.50,
        "max_model_len": 4096,
        "batch_size": 64,
    }
    assert seen[0]["config"].enable_qwen is True
    assert seen[0]["bundle"].qwen_reranker is not None
    assert _FakeReranker.instances[0].destroyed is True
    assert result["qwen_requested"] is True
    assert result["qwen_initialized"] is True
    assert result["qwen_model_name"] == "Qwen/Qwen2.5-7B-Instruct-AWQ"
    assert result["qwen_rerank_query_count"] == 1
    assert result["qwen_assertion_query_count"] == 1
    assert result["qwen_abstention_count"] == 1
    assert result["qwen_status"] == "COMPLETED"


def test_phase_12_propagates_qwen_initialization_failure_and_records_failed_status(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path)
    config = RunConfig(**{**config.__dict__, "enable_qwen_reranker": True})
    _patch_phase_dependencies(monkeypatch, pipeline=lambda **_kwargs: pytest.fail("inference must not run"))
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _FailingReranker)

    with pytest.raises(RequiredQwenError, match="initialization failed"):
        kaggle_phases._phase_12_inference(config, "phase_12_inference", context)

    qwen_summary = json.loads((Path(context["run_dir"]) / "diagnostics" / "qwen_summary.json").read_text(encoding="utf-8"))
    assert qwen_summary["qwen_status"] == "FAILED"


def test_phase_12_cleans_qwen_up_after_inference_failure(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path)
    config = RunConfig(**{**config.__dict__, "enable_qwen_reranker": True})
    _FakeReranker.instances.clear()
    _patch_phase_dependencies(monkeypatch, pipeline=lambda **_kwargs: (_ for _ in ()).throw(RequiredQwenError("Qwen generation failed")))
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _FakeReranker)

    with pytest.raises(RequiredQwenError, match="generation failed"):
        kaggle_phases._phase_12_inference(config, "phase_12_inference", context)

    assert _FakeReranker.instances[0].destroyed is True
    assert not (Path(context["run_dir"]) / "output.zip").exists()


def test_phase_12_propagates_qwen_cleanup_failures_before_creating_output_zip(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path)
    config = RunConfig(**{**config.__dict__, "enable_qwen_reranker": True})
    _CleanupFailingReranker.instances.clear()
    _patch_phase_dependencies(monkeypatch, pipeline=lambda **_kwargs: {"document_count": 0})
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _CleanupFailingReranker)

    with pytest.raises(RuntimeError, match="cleanup failed"):
        kaggle_phases._phase_12_inference(config, "phase_12_inference", context)

    assert _CleanupFailingReranker.instances[0].destroyed is True
    assert not (Path(context["run_dir"]) / "output.zip").exists()


def test_phase_12_surfaces_cleanup_error_when_inference_also_fails(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path)
    config = RunConfig(**{**config.__dict__, "enable_qwen_reranker": True})
    _CleanupFailingReranker.instances.clear()
    _patch_phase_dependencies(
        monkeypatch,
        pipeline=lambda **_kwargs: (_ for _ in ()).throw(RequiredQwenError("Qwen generation failed")),
    )
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _CleanupFailingReranker)

    with pytest.raises(RuntimeError, match="cleanup failed") as error:
        kaggle_phases._phase_12_inference(config, "phase_12_inference", context)

    assert isinstance(error.value.__cause__, RequiredQwenError)
    assert "generation failed" in str(error.value.__cause__)


def test_required_qwen_failure_stops_before_phase_13_and_pass_manifest(tmp_path: Path, monkeypatch):
    config, context = _phase_paths(tmp_path)
    _patch_phase_dependencies(monkeypatch, pipeline=lambda **_kwargs: pytest.fail("inference must not run"))
    monkeypatch.setattr("clinical_nlp_lab.reranker.ClinicalLLMReranker", _FailingReranker)
    phase_13_called = False

    def phase_13(_config, _phase, _context):
        nonlocal phase_13_called
        phase_13_called = True
        return {}

    config = RunConfig(
        **{
            **config.__dict__,
            "enable_qwen_reranker": True,
            "output_dir": Path(context["run_dir"]).parent,
            "run_id": Path(context["run_dir"]).name,
            "phase_runners": {PHASES[11]: kaggle_phases._phase_12_inference, PHASES[12]: phase_13},
        }
    )

    with pytest.raises(RequiredQwenError, match="initialization failed"):
        _execute_phases(config, (PHASES[11], PHASES[12]), 0, config.run_id)

    assert phase_13_called is False
    assert not (Path(context["run_dir"]) / "run_manifest.json").exists()


def test_infer_document_propagates_required_qwen_errors_when_enabled():
    with pytest.raises(RequiredQwenError, match="generation failed"):
        infer_document(
            "1",
            "fever",
            FinalModelBundle(ner_model=None, tokenizer=None, qwen_reranker=_ExplodingRefiner()),
            InferenceConfig(enable_qwen=True),
        )
