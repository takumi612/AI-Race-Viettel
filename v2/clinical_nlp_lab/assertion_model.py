from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
import torch
from torch import nn, Tensor


@dataclass(frozen=True)
class AssertionThresholdArtifact:
    schema_id: str
    schema_version: int
    thresholds: tuple[float, float, float]  # is_negated, is_historical, is_family
    encoder_hash: str
    tokenizer_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "thresholds": list(self.thresholds),
            "encoder_hash": self.encoder_hash,
            "tokenizer_hash": self.tokenizer_hash,
        }


@dataclass(frozen=True)
class AssertionBindingArtifact:
    schema_id: str
    schema_version: int
    encoder_hash: str
    tokenizer_hash: str
    encoder_frozen: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "encoder_hash": self.encoder_hash,
            "tokenizer_hash": self.tokenizer_hash,
            "encoder_frozen": self.encoder_frozen,
        }


def pool_mention_features(
    hidden_states: Tensor,  # [B, L, D]
    entity_spans: Tensor,   # [N, 3] = (b_idx, start_tok, end_tok)
    entity_types: Tensor,   # [N] entity_type_id (0..4)
    num_types: int = 5,
) -> Tensor:
    if entity_spans.numel() == 0:
        D = hidden_states.shape[-1]
        return torch.empty((0, D * 4 + num_types), dtype=hidden_states.dtype, device=hidden_states.device)

    features: list[Tensor] = []
    num_mentions = entity_spans.shape[0]

    for i in range(num_mentions):
        b_idx = int(entity_spans[i, 0].item())
        start_tok = int(entity_spans[i, 1].item())
        end_tok = max(start_tok + 1, int(entity_spans[i, 2].item()))

        cls_emb = hidden_states[b_idx, 0]  # [D]
        span_tokens = hidden_states[b_idx, start_tok:end_tok]  # [K, D]
        mean_emb = span_tokens.mean(dim=0)  # [D]
        first_emb = span_tokens[0]  # [D]
        last_emb = span_tokens[-1]  # [D]

        t_type = int(entity_types[i].item())
        type_onehot = torch.zeros(num_types, dtype=hidden_states.dtype, device=hidden_states.device)
        if 0 <= t_type < num_types:
            type_onehot[t_type] = 1.0

        mention_vec = torch.cat([cls_emb, mean_emb, first_emb, last_emb, type_onehot], dim=-1)
        features.append(mention_vec)

    return torch.stack(features, dim=0)


class AssertionHead(nn.Module):
    def __init__(self, hidden_dim: int, num_types: int = 5, num_targets: int = 3) -> None:
        super().__init__()
        input_dim = hidden_dim * 4 + num_types
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, num_targets),
        )

    def forward(self, mention_features: Tensor, entity_types: Tensor | None = None) -> Tensor:
        if mention_features.numel() == 0:
            return torch.empty((0, 3), dtype=mention_features.dtype, device=mention_features.device)

        logits = self.mlp(mention_features)  # [N, 3]

        if entity_types is not None and entity_types.numel() > 0:
            # Mask lab entities (types 3: LAB_NAME, 4: LAB_RESULT) to zero logits
            is_lab = (entity_types == 3) | (entity_types == 4)
            if is_lab.any():
                logits = logits.masked_fill(is_lab.unsqueeze(-1), -10000.0)

        return logits


class FrozenAssertionAdapter(nn.Module):
    """Reuse the final NER encoder while allowing gradients only in the head."""

    def __init__(
        self,
        encoder: nn.Module,
        head: AssertionHead,
        binding: AssertionBindingArtifact,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = head
        self.binding = binding
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True) -> "FrozenAssertionAdapter":
        """Keep the shared encoder in eval mode even when the head trains."""
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        entity_spans: Tensor,
        entity_types: Tensor,
    ) -> Tensor:
        with torch.no_grad():
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            hidden_states = outputs.last_hidden_state
        mention_features = pool_mention_features(hidden_states, entity_spans, entity_types)
        return self.head(mention_features, entity_types=entity_types)


def build_frozen_assertion_adapter(
    encoder: nn.Module,
    hidden_dim: int,
    encoder_hash: str,
    tokenizer_hash: str,
) -> FrozenAssertionAdapter:
    if len(encoder_hash) != 64 or len(tokenizer_hash) != 64:
        raise ValueError("encoder_hash and tokenizer_hash must be SHA-256 values")
    binding = AssertionBindingArtifact(
        schema_id="clinical_nlp.assertion_binding",
        schema_version=1,
        encoder_hash=encoder_hash,
        tokenizer_hash=tokenizer_hash,
        encoder_frozen=True,
    )
    return FrozenAssertionAdapter(encoder, AssertionHead(hidden_dim=hidden_dim), binding)


def validate_assertion_binding(
    adapter: FrozenAssertionAdapter,
    encoder_hash: str,
    tokenizer_hash: str,
) -> None:
    if adapter.binding.encoder_hash != encoder_hash:
        raise ValueError("encoder hash mismatch")
    if adapter.binding.tokenizer_hash != tokenizer_hash:
        raise ValueError("tokenizer hash mismatch")
    if not adapter.binding.encoder_frozen or any(parameter.requires_grad for parameter in adapter.encoder.parameters()):
        raise ValueError("assertion encoder must remain frozen")


def fit_assertion_thresholds(
    logits: np.ndarray,
    targets: np.ndarray,
    mask: np.ndarray,
    grid: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8),
    encoder_hash: str = "",
    tokenizer_hash: str = "",
) -> AssertionThresholdArtifact:
    probs = 1.0 / (1.0 + np.exp(-logits))  # Sigmoid
    selected_thresholds: list[float] = []

    for axis in range(3):
        axis_probs = probs[:, axis]
        axis_targets = targets[:, axis]
        axis_mask = mask[:, axis]

        valid_p = axis_probs[axis_mask]
        valid_t = axis_targets[axis_mask]

        best_f1 = -1.0
        best_th = 0.5

        if valid_t.size > 0 and valid_t.sum() > 0:
            for th in sorted(grid):
                preds = (valid_p >= th).astype(int)
                tp = int(((preds == 1) & (valid_t == 1)).sum())
                fp = int(((preds == 1) & (valid_t == 0)).sum())
                fn = int(((preds == 0) & (valid_t == 1)).sum())

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

                # Hòa F1 chọn threshold cao hơn
                if f1 >= best_f1:
                    best_f1 = f1
                    best_th = float(th)

        selected_thresholds.append(best_th)

    return AssertionThresholdArtifact(
        schema_id="clinical_nlp.assertion_thresholds",
        schema_version=1,
        thresholds=(selected_thresholds[0], selected_thresholds[1], selected_thresholds[2]),
        encoder_hash=encoder_hash,
        tokenizer_hash=tokenizer_hash,
    )
