from __future__ import annotations

import pytest
import torch
from torch import nn

from clinical_nlp_lab.assertion_model import (
    build_frozen_assertion_adapter,
    validate_assertion_binding,
)


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.projection = nn.Linear(3, hidden_size)

    def forward(self, input_ids, attention_mask=None):
        values = torch.nn.functional.one_hot(input_ids % 3, num_classes=3).float()
        return type("Output", (), {"last_hidden_state": self.projection(values)})()


def test_frozen_shared_encoder_adapter_trains_only_assertion_head():
    adapter = build_frozen_assertion_adapter(
        FakeEncoder(),
        hidden_dim=4,
        encoder_hash="e" * 64,
        tokenizer_hash="t" * 64,
    )

    assert all(not parameter.requires_grad for parameter in adapter.encoder.parameters())
    assert any(parameter.requires_grad for parameter in adapter.head.parameters())
    adapter.train()
    assert adapter.encoder.training is False
    logits = adapter(
        input_ids=torch.tensor([[1, 2, 0]]),
        attention_mask=torch.ones((1, 3), dtype=torch.long),
        entity_spans=torch.tensor([[0, 1, 2], [0, 2, 3]]),
        entity_types=torch.tensor([0, 3]),
    )
    assert logits.shape == (2, 3)
    assert torch.all(logits[1] < -1000)


def test_assertion_binding_rejects_encoder_or_tokenizer_hash_drift():
    adapter = build_frozen_assertion_adapter(
        FakeEncoder(),
        hidden_dim=4,
        encoder_hash="e" * 64,
        tokenizer_hash="t" * 64,
    )
    validate_assertion_binding(adapter, encoder_hash="e" * 64, tokenizer_hash="t" * 64)
    with pytest.raises(ValueError, match="encoder hash"):
        validate_assertion_binding(adapter, encoder_hash="x" * 64, tokenizer_hash="t" * 64)
    with pytest.raises(ValueError, match="tokenizer hash"):
        validate_assertion_binding(adapter, encoder_hash="e" * 64, tokenizer_hash="x" * 64)
