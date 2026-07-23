from __future__ import annotations

import numpy as np
import pytest
import torch
from clinical_nlp_lab.assertion_model import (
    AssertionHead,
    fit_assertion_thresholds,
    pool_mention_features,
)


def test_pool_mention_features_different_mention_counts():
    B, L, D = 2, 10, 16
    hidden_states = torch.randn(B, L, D)

    # Batch 0 có 2 mentions, Batch 1 có 1 mention -> tổng 3 mentions
    entity_spans = torch.tensor(
        [
            [0, 1, 3],
            [0, 4, 6],
            [1, 2, 5],
        ],
        dtype=torch.long,
    )
    entity_types = torch.tensor([0, 1, 2], dtype=torch.long)

    pooled = pool_mention_features(hidden_states, entity_spans, entity_types, num_types=5)
    assert pooled.shape == (3, D * 4 + 5)


def test_lab_entity_assertion_masking():
    D = 16
    head = AssertionHead(hidden_dim=D, num_types=5, num_targets=3)

    mention_features = torch.randn(2, D * 4 + 5)
    entity_types = torch.tensor([0, 3], dtype=torch.long)  # Type 3 là LAB_NAME

    logits = head(mention_features, entity_types=entity_types)
    probs = torch.sigmoid(logits)

    # Lab entity (hàng thứ 2) phải có logits rất âm -> probs gần 0
    assert probs[1, 0].item() < 1e-3
    assert probs[1, 1].item() < 1e-3
    assert probs[1, 2].item() < 1e-3


def test_three_axis_independent_sigmoid():
    D = 16
    head = AssertionHead(hidden_dim=D, num_types=5, num_targets=3)
    mention_features = torch.randn(1, D * 4 + 5)

    logits = head(mention_features)
    probs = torch.sigmoid(logits)

    assert probs.shape == (1, 3)
    # Tổng xác suất 3 trục không bắt buộc bằng 1 (sigmoid độc lập, không phải softmax)
    assert not torch.allclose(probs.sum(), torch.tensor(1.0))


def test_fit_assertion_thresholds_tie_breaking():
    logits = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    targets = np.array([[0, 0, 0], [1, 1, 1]])
    mask = np.array([[True, True, True], [True, True, True]])

    # Khi hòa F1 giữa 0.4 và 0.6, chọn threshold cao hơn (0.6)
    grid = [0.4, 0.6]
    artifact = fit_assertion_thresholds(
        logits, targets, mask, grid=grid, encoder_hash="enc123", tokenizer_hash="tok123"
    )

    assert artifact.thresholds == (0.6, 0.6, 0.6)
    assert artifact.encoder_hash == "enc123"
