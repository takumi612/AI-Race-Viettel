from __future__ import annotations

import json
from pathlib import Path

from clinical_nlp_lab import kaggle_phases
from clinical_nlp_lab.orchestration import RunConfig


ROOT = Path(__file__).parents[1]


def test_qwen_reranker_is_disabled_by_default():
    assert RunConfig().enable_qwen_reranker is False


def test_config_json_cannot_override_the_run_config_qwen_toggle(tmp_path: Path, monkeypatch):
    """A saved artifact toggle must not silently change the explicit run contract."""
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints" / "final_fit" / "ner_model").mkdir(parents=True)
    head_dir = run_dir / "artifacts" / "heads"
    head_dir.mkdir(parents=True)
    (head_dir / "candidate_calibration.json").write_text(
        '{"confidence_threshold": 0.5, "objective": "precision_first"}', encoding="utf-8"
    )
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "config.json").write_text('{"enable_qwen": true}', encoding="utf-8")
    (artifact_dir / "entity_type_mapping.json").write_text("{}", encoding="utf-8")
    (artifact_dir / "assertion_mapping.json").write_text("{}", encoding="utf-8")
    input_source = tmp_path / "input.zip"
    input_source.write_bytes(b"placeholder")
    seen = []

    monkeypatch.setattr(kaggle_phases, "load_candidate_dictionary", lambda _path: [])
    monkeypatch.setattr("clinical_nlp_lab.runtime_bundle.load_final_model_bundle", lambda *args: object())
    monkeypatch.setattr("clinical_nlp_lab.pipeline.run_inference_with_bundle", lambda **kwargs: seen.append(kwargs["config"]) or {})
    monkeypatch.setattr("clinical_nlp_lab.output_quality.audit_submission_directory", lambda *_args: {})
    monkeypatch.setattr("clinical_nlp_lab.output_quality.enforce_output_quality", lambda _report: None)

    kaggle_phases._phase_12_inference(
        RunConfig(artifact_dir=artifact_dir, config_path=artifact_dir / "config.json", input_source=input_source),
        "phase_12_inference",
        {"run_dir": str(run_dir)},
    )

    assert seen[0].enable_qwen is False
