from __future__ import annotations

import sys
import types
from pathlib import Path


ROOT = Path(__file__).parents[1]
package = types.ModuleType("clinical_nlp_lab")
package.__path__ = [str(ROOT / "clinical_nlp_lab")]
sys.modules.setdefault("clinical_nlp_lab", package)

from clinical_nlp_lab.data import audit_split_leakage, grouped_train_validation_split
from clinical_nlp_lab.dataset_quality import DatasetRecord
from clinical_nlp_lab.schema import ClinicalDocument


def _record(document_id: str, template: str, surface: str) -> DatasetRecord:
    return DatasetRecord(
        document_id=document_id,
        source_bucket="synthetic",
        template_group=template,
        genre="test",
        long_tail=False,
        primary_surfaces=(surface,),
        sha256=document_id,
    )


def test_grouped_split_never_crosses_template_or_surface_components():
    documents = [ClinicalDocument(str(index), f"document {index}") for index in range(1, 9)]
    records = [
        _record("1", "a", "alpha"),
        _record("2", "a", "beta"),
        _record("3", "b", "beta"),
        _record("4", "c", "gamma"),
        _record("5", "d", "delta"),
        _record("6", "e", "epsilon"),
        _record("7", "f", "zeta"),
        _record("8", "g", "eta"),
    ]

    train, validation, manifest = grouped_train_validation_split(
        documents, records, validation_fraction=0.25, seed=42
    )
    audit = audit_split_leakage(train, validation, records)

    assert train
    assert validation
    assert audit == {"document_ids": [], "template_groups": [], "surface_groups": []}
    assert manifest["seed"] == 42
    assert set(manifest["train_ids"]) == {document.document_id for document in train}
    assert set(manifest["validation_ids"]) == {document.document_id for document in validation}


def test_grouped_split_is_deterministic():
    documents = [ClinicalDocument(str(index), f"document {index}") for index in range(1, 7)]
    records = [_record(str(index), f"template-{index}", f"surface-{index}") for index in range(1, 7)]

    first = grouped_train_validation_split(documents, records, 0.33, 7)[2]
    second = grouped_train_validation_split(documents, records, 0.33, 7)[2]

    assert first == second
