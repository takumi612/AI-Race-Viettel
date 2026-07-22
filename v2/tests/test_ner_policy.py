from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
package = types.ModuleType("clinical_nlp_lab")
package.__path__ = [str(ROOT / "clinical_nlp_lab")]
sys.modules.setdefault("clinical_nlp_lab", package)

from clinical_nlp_lab.ner import DictionaryRuleEntityDetector


def test_generic_symptom_and_patient_regex_are_not_primary_detectors():
    detector = DictionaryRuleEntityDetector([], [])

    assert detector.detect("Nam 45 tuổi, sốt và ho") == []


def test_regex_fallback_can_be_enabled_for_degraded_runtime_only():
    detector = DictionaryRuleEntityDetector([], [], enable_generic_regex=True)

    entities = detector.detect("Nam 45 tuổi, sốt và glucose 8 mmol/L")

    assert {entity.type for entity in entities} == {"SYMPTOM", "LAB_RESULT", "PATIENT_INFO"}
