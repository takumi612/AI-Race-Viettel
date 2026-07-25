from __future__ import annotations

import warnings

import numpy as np

from clinical_nlp_lab.assertion_model import (
    fit_assertion_thresholds,
    split_assertion_documents,
)
from clinical_nlp_lab.schema import ClinicalDocument


def test_threshold_fitting_is_numerically_stable_for_extreme_logits():
    logits = np.array([[10000.0, -10000.0, 0.0], [-10000.0, 10000.0, 1.0]])
    targets = np.array([[1, 0, 0], [0, 1, 1]])
    mask = np.ones_like(targets, dtype=bool)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        artifact = fit_assertion_thresholds(logits, targets, mask)

    assert all(0.0 < value < 1.0 for value in artifact.thresholds)


def test_assertion_calibration_documents_are_held_out_by_id():
    documents = [ClinicalDocument(str(index), f"text-{index}") for index in range(1, 6)]

    train, validation = split_assertion_documents(documents, {"2", "5"})

    assert [item.document_id for item in train] == ["1", "3", "4"]
    assert [item.document_id for item in validation] == ["2", "5"]
    assert {item.document_id for item in train}.isdisjoint(
        item.document_id for item in validation
    )
