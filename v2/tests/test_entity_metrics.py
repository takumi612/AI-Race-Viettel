from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
package = types.ModuleType("clinical_nlp_lab")
package.__path__ = [str(ROOT / "clinical_nlp_lab")]
sys.modules.setdefault("clinical_nlp_lab", package)

from clinical_nlp_lab.schema import EntityAnnotation
from clinical_nlp_lab.training import compute_entity_metrics


def test_entity_metrics_separate_exact_and_overlap_boundary_quality():
    expected = {
        "1": [EntityAnnotation("tăng huyết áp", "DISEASE", (0, 13))],
    }
    predicted = {
        "1": [EntityAnnotation("tăng huyết", "DISEASE", (0, 9))],
    }

    metrics = compute_entity_metrics(predicted, expected)

    assert metrics["exact_f1"] == 0.0
    assert metrics["overlap_f1"] == 1.0
