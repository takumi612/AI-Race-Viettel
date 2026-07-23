from __future__ import annotations

import json
from pathlib import Path

from clinical_nlp_lab.inference import FinalModelBundle, InferenceConfig, SpanProposal
from clinical_nlp_lab.pipeline import run_inference_with_bundle
from clinical_nlp_lab.schema import OFFICIAL_SCHEMA_KEYS


class FakeNER:
    def propose(self, raw_text: str, config: InferenceConfig):
        start = raw_text.index("fever")
        return [SpanProposal("fever", "SYMPTOM", start, start + 5, 0.91, "ner")]


def test_bundle_inference_is_a_valid_primary_submission_path(tmp_path: Path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "300.txt").write_text("HS-0300: patient has fever", encoding="utf-8")
    symptom_type = next(key for key in OFFICIAL_SCHEMA_KEYS if "TRI" in key)

    summary = run_inference_with_bundle(
        input_source=input_dir,
        output_dir=tmp_path / "output",
        bundle=FinalModelBundle(ner_model=FakeNER(), tokenizer=object()),
        entity_mapping={"internal_to_official": {"SYMPTOM": symptom_type}, "drop_unmapped": True},
        assertion_mapping={"internal_to_official": {}},
        config=InferenceConfig(),
    )

    payload = json.loads((tmp_path / "output" / "300.json").read_text(encoding="utf-8"))
    assert payload == [{"text": "fever", "type": symptom_type, "position": [21, 26], "assertions": []}]
    assert summary["training_or_fitting_on_input"] is False
    assert summary["primary_path"] == "final_model_bundle"
    assert (tmp_path / "output.zip").is_file()
