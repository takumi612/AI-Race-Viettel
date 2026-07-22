from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
package = types.ModuleType("clinical_nlp_lab")
package.__path__ = [str(ROOT / "clinical_nlp_lab")]
sys.modules.setdefault("clinical_nlp_lab", package)

from clinical_nlp_lab.dataset_quality import (
    DatasetRecord,
    build_dataset_manifest,
    validate_dataset_contract,
)
from clinical_nlp_lab.schema import ClinicalDocument, EntityAnnotation


def _document(entity: EntityAnnotation, document_id: str = "1") -> ClinicalDocument:
    return ClinicalDocument(document_id=document_id, raw_text=entity.text, entities=[entity])


def test_contract_rejects_unknown_disease_candidate():
    entity = EntityAnnotation(
        text="tăng huyết áp",
        type="DISEASE",
        position=(0, len("tăng huyết áp")),
        candidates=["UNKNOWN"],
    )

    report = validate_dataset_contract([_document(entity)], {"I10"}, set())

    assert report["is_valid"] is False
    assert report["errors"][0]["code"] == "unknown_icd_candidate"


def test_contract_rejects_missing_drug_candidate_and_lab_assertion():
    text = "metformin glucose 8 mmol/L"
    document = ClinicalDocument(
        document_id="2",
        raw_text=text,
        entities=[
            EntityAnnotation("metformin", "DRUG", (0, 9), candidates=[]),
            EntityAnnotation(
                "glucose 8 mmol/L",
                "LAB_RESULT",
                (10, len(text)),
                assertions=["isNegated"],
            ),
        ],
    )

    report = validate_dataset_contract([document], set(), {"6809"})

    assert {error["code"] for error in report["errors"]} == {
        "missing_rxnorm_candidate",
        "assertion_not_allowed",
    }


def test_manifest_records_origin_hash_and_supplied_generation_metadata():
    entity = EntityAnnotation(
        text="viêm phổi",
        type="DISEASE",
        position=(0, len("viêm phổi")),
        candidates=["J18.9"],
    )
    document = _document(entity, document_id="201")

    records = build_dataset_manifest(
        [document],
        metadata_by_id={
            "201": {
                "genre": "cap_cuu",
                "template_group": "cap_cuu:respiratory",
                "long_tail": False,
            }
        },
    )

    assert records == [
        DatasetRecord(
            document_id="201",
            source_bucket="synthetic",
            template_group="cap_cuu:respiratory",
            genre="cap_cuu",
            long_tail=False,
            primary_surfaces=("viem phoi",),
            sha256=hashlib.sha256("viêm phổi".encode("utf-8")).hexdigest(),
        )
    ]


def test_manifest_distinguishes_reconstructed_and_organizer_ground_truth():
    documents = [ClinicalDocument(str(item), "") for item in (1, 100, 101, 200, 201)]

    records = build_dataset_manifest(documents)

    assert [record.source_bucket for record in records] == [
        "reconstructed",
        "reconstructed",
        "organizer_gt",
        "organizer_gt",
        "synthetic",
    ]
