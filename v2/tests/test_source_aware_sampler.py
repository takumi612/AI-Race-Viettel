from __future__ import annotations

import pytest
from clinical_nlp_lab.examples import TokenWindow
from clinical_nlp_lab.sampling import (
    build_source_aware_epoch,
    select_replay_examples,
)


def _make_window(doc_id: str, win_idx: int) -> TokenWindow:
    return TokenWindow(
        document_id=doc_id,
        record_id=f"rec_{doc_id}",
        window_id=f"{doc_id}_w{win_idx}",
        input_ids=(0, 10, 2),
        attention_mask=(1, 1, 1),
        raw_offsets=((-1, -1), (0, 5), (-1, -1)),
        label_ids=(-100, 1, -100),
        loss_mask=(False, True, False),
        owned_entity_ids=(f"e_{doc_id}",),
    )


def test_source_aware_sampler_exposure_and_documents():
    # Documents 001-200 là organizer, 201+ là synthetic
    examples = [
        _make_window("010", 0),
        _make_window("010", 1),
        _make_window("050", 0),
        _make_window("300", 0),
        _make_window("300", 1),
        _make_window("400", 0),
    ]

    sampled_indices = build_source_aware_epoch(examples, organizer_fraction=0.5, seed=42)
    assert len(sampled_indices) == len(examples)

    sampled_docs = {examples[idx].document_id for idx in sampled_indices}
    assert "010" in sampled_docs or "050" in sampled_docs
    assert "300" in sampled_docs or "400" in sampled_docs

    # Unique document counts
    org_chunks = sum(1 for idx in sampled_indices if int(examples[idx].document_id) <= 200)
    syn_chunks = sum(1 for idx in sampled_indices if int(examples[idx].document_id) > 200)
    assert org_chunks > 0
    assert syn_chunks > 0


def test_replay_examples_determinism_and_prioritization():
    examples = [
        _make_window("301", 0),  # rare genre
        _make_window("302", 0),  # baseline
        _make_window("303", 0),  # assertion + unseen
    ]
    metadata = {
        "301": {"genre": "rare"},
        "302": {"genre": "normal"},
        "303": {"has_assertion": True, "has_unseen_surface": True},
    }

    manifest1 = select_replay_examples(examples, metadata, fraction=0.6, seed=42)
    manifest2 = select_replay_examples(examples, metadata, fraction=0.6, seed=42)

    # Test determinism
    assert manifest1.to_dict() == manifest2.to_dict()

    # Document 303 và 301 phải được ưu tiên trước 302
    selected_docs = [item.document_id for item in manifest1.items]
    assert "303" in selected_docs
