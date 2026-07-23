from __future__ import annotations

import pytest

from clinical_nlp_lab.curriculum import (
    build_stage_manifest,
    plan_curriculum,
    validate_resume_manifest,
)
from clinical_nlp_lab.records import ClinicalRecord
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation
from clinical_nlp_lab.training import build_bio_label_map, build_training_contract


class ContractTokenizer:
    def __call__(self, text: str, **_: object) -> dict[str, list[object]]:
        offsets: list[tuple[int, int]] = [(0, 0)]
        ids: list[int] = [0]
        cursor = 0
        for token_index, token in enumerate(text.split()):
            start = text.index(token, cursor)
            end = start + len(token)
            offsets.append((start, end))
            ids.append(token_index + 10)
            cursor = end
        offsets.append((0, 0))
        ids.append(2)
        return {"input_ids": ids, "offset_mapping": offsets}


def _document() -> tuple[ClinicalDocument, tuple[ClinicalRecord, ...]]:
    raw_text = "patient has fever"
    entity_start = raw_text.index("fever")
    document = ClinicalDocument(
        document_id="doc-1",
        raw_text=raw_text,
        entities=[
            EntityAnnotation(
                text="fever",
                type="SYMPTOM",
                position=(entity_start, entity_start + len("fever")),
            )
        ],
    )
    records = (ClinicalRecord("doc-1", "record-1", 0, len(raw_text), (0,)),)
    return document, records


def test_build_training_contract_uses_owner_windows_and_collator():
    document, records = _document()
    label_to_id, _ = build_bio_label_map(["DISEASE", "DRUG", "SYMPTOM", "LAB_NAME", "LAB_RESULT"])

    contract = build_training_contract(
        documents=(document,),
        records_by_document={document.document_id: records},
        tokenizer=ContractTokenizer(),
        label_to_id=label_to_id,
        dataset_fingerprint="dataset-1",
        split_fingerprint="split-1",
        batch_size=1,
    )

    assert contract.window_count == 1
    assert contract.owned_entity_count == 1
    assert contract.fingerprints == {
        "dataset": "dataset-1",
        "split": "split-1",
        "label_map": contract.fingerprints["label_map"],
    }
    assert len(contract.batches) == 1
    assert contract.batches[0]["input_ids"].shape[0] == 1
    assert contract.batches[0]["entity_spans"].shape == (1, 3)


def test_stage_manifest_resume_rejects_stale_fingerprint():
    stage = plan_curriculum("full")[0]
    manifest = build_stage_manifest(
        stage,
        fingerprints={"dataset": "dataset-1", "split": "split-1"},
        checkpoint_sha256="a" * 64,
    )

    validate_resume_manifest(
        manifest,
        stage,
        current_fingerprints={"dataset": "dataset-1", "split": "split-1"},
    )
    with pytest.raises(ValueError, match="fingerprint"):
        validate_resume_manifest(
            manifest,
            stage,
            current_fingerprints={"dataset": "dataset-changed", "split": "split-1"},
        )


def test_training_contract_carries_curriculum_manifest_for_resume():
    document, records = _document()
    label_to_id, _ = build_bio_label_map(["SYMPTOM"])
    stage = plan_curriculum("full")[0]
    manifest = build_stage_manifest(
        stage,
        fingerprints={"dataset": "dataset-1", "split": "split-1"},
        checkpoint_sha256="b" * 64,
    )
    contract = build_training_contract(
        documents=(document,),
        records_by_document={document.document_id: records},
        tokenizer=ContractTokenizer(),
        label_to_id=label_to_id,
        dataset_fingerprint="dataset-1",
        split_fingerprint="split-1",
        curriculum_manifest=manifest,
    )
    assert contract.curriculum_manifest["stage_name"] == "stage1"
    assert contract.to_dict()["curriculum_manifest"]["checkpoint_sha256"] == "b" * 64
