from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Literal

from .examples import TokenWindow


@dataclass(frozen=True)
class ReplayItem:
    example_id: str
    document_id: str
    priority_score: float
    reason: str


@dataclass(frozen=True)
class ReplayManifest:
    schema_id: str
    schema_version: int
    seed: int
    fraction: float
    items: tuple[ReplayItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "fraction": self.fraction,
            "items": [
                {
                    "example_id": item.example_id,
                    "document_id": item.document_id,
                    "priority_score": item.priority_score,
                    "reason": item.reason,
                }
                for item in self.items
            ],
        }


def build_source_aware_epoch(
    examples: Sequence[TokenWindow],
    organizer_fraction: float,
    seed: int,
) -> tuple[int, ...]:
    if not examples:
        return ()

    organizer_indices: list[int] = []
    synthetic_indices: list[int] = []

    for idx, ex in enumerate(examples):
        try:
            doc_num = int(ex.document_id)
            if doc_num <= 200:
                organizer_indices.append(idx)
            else:
                synthetic_indices.append(idx)
        except ValueError:
            synthetic_indices.append(idx)

    if not organizer_indices or not synthetic_indices:
        rng = random.Random(seed)
        all_indices = list(range(len(examples)))
        rng.shuffle(all_indices)
        return tuple(all_indices)

    rng = random.Random(seed)
    rng.shuffle(organizer_indices)
    rng.shuffle(synthetic_indices)

    total_needed = len(examples)
    org_target = int(round(total_needed * organizer_fraction))
    syn_target = total_needed - org_target

    sampled_org: list[int] = []
    while len(sampled_org) < org_target:
        sampled_org.extend(organizer_indices)
    sampled_org = sampled_org[:org_target]

    sampled_syn: list[int] = []
    while len(sampled_syn) < syn_target:
        sampled_syn.extend(synthetic_indices)
    sampled_syn = sampled_syn[:syn_target]

    combined = sampled_org + sampled_syn
    rng.shuffle(combined)
    return tuple(combined)


def compute_replay_priority(
    example: TokenWindow,
    metadata: Mapping[str, Any],
) -> tuple[float, str]:
    score = 1.0
    reasons: list[str] = []

    doc_meta = metadata.get(example.document_id, {})
    genre = doc_meta.get("genre", "")
    if genre in {"rare", "long_tail", "specialized"}:
        score += 2.0
        reasons.append("rare_genre")

    if doc_meta.get("has_assertion", False) or doc_meta.get("is_negated", False):
        score += 1.5
        reasons.append("assertion")

    if doc_meta.get("has_unseen_surface", False):
        score += 1.5
        reasons.append("unseen_surface")

    if doc_meta.get("complex_boundary", False):
        score += 1.0
        reasons.append("complex_boundary")

    reason_str = "+".join(reasons) if reasons else "baseline"
    return score, reason_str


def select_replay_examples(
    examples: Sequence[TokenWindow],
    metadata: Mapping[str, Any],
    fraction: float,
    seed: int,
) -> ReplayManifest:
    if fraction <= 0.0 or not examples:
        return ReplayManifest(
            schema_id="clinical_nlp.replay_manifest",
            schema_version=1,
            seed=seed,
            fraction=fraction,
            items=(),
        )

    synthetic_examples: list[tuple[float, TokenWindow, str]] = []
    for ex in examples:
        try:
            doc_num = int(ex.document_id)
            is_synthetic = doc_num > 200
        except ValueError:
            is_synthetic = True

        if is_synthetic:
            score, reason = compute_replay_priority(ex, metadata)
            synthetic_examples.append((score, ex, reason))

    if not synthetic_examples:
        return ReplayManifest(
            schema_id="clinical_nlp.replay_manifest",
            schema_version=1,
            seed=seed,
            fraction=fraction,
            items=(),
        )

    rng = random.Random(seed)
    synthetic_examples.sort(key=lambda item: (-item[0], item[1].window_id))

    target_count = max(1, int(round(len(synthetic_examples) * fraction)))
    selected = synthetic_examples[:target_count]

    items = [
        ReplayItem(
            example_id=ex.window_id,
            document_id=ex.document_id,
            priority_score=score,
            reason=reason,
        )
        for score, ex, reason in selected
    ]

    return ReplayManifest(
        schema_id="clinical_nlp.replay_manifest",
        schema_version=1,
        seed=seed,
        fraction=fraction,
        items=tuple(items),
    )
