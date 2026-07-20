from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.config import load_config
from clinical_nlp_lab.data import document_train_validation_split, load_annotated_documents
from clinical_nlp_lab.training import (
    build_bio_label_map,
    chunk_token_indices,
    train_transformer_ner,
    transformer_training_availability,
)
from clinical_nlp_lab.schema import write_json


def main() -> None:
    config = load_config(PROJECT_ROOT / "artifacts/config.json")
    annotated = load_annotated_documents(PROJECT_ROOT / config["train_dir"])
    train_documents, validation_documents = document_train_validation_split(
        annotated, config["validation_fraction"], int(config["seed"])
    )
    entity_types = {entity.type for document in annotated for entity in document.entities}
    label_to_id, id_to_label = build_bio_label_map(entity_types)
    availability = transformer_training_availability()
    training_result = train_transformer_ner(
        train_documents,
        validation_documents,
        PROJECT_ROOT / "artifacts/ner_model",
        model_name=config["ner_model_name"],
        max_length=int(config["max_length"]),
        stride=int(config["stride"]),
        learning_rate=float(config["learning_rate"]),
        epochs=int(config["ner_epochs"]),
        batch_size=int(config["batch_size"]),
        seed=int(config["seed"]),
    )
    report = {
        "stage": 4,
        "annotated_documents": len(annotated),
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "document_split_leakage": bool(
            {item.document_id for item in train_documents} & {item.document_id for item in validation_documents}
        ),
        "bio_label_count": len(label_to_id),
        "bio_labels": label_to_id,
        "sliding_window_example": chunk_token_indices(1200, int(config["max_length"]), int(config["stride"])),
        "offset_contract": "return_offsets_mapping=True + return_overflowing_tokens=True + raw slice reconstruction",
        "transformer_availability": {
            "available": availability.available,
            "missing_packages": availability.missing_packages,
            "reason": availability.reason,
        },
        "training_result": training_result,
        "evaluation": {
            "status": "not_scored" if not annotated else "requires_execution_on_annotation",
            "reason": "No annotation was supplied in the workspace; no NER score was invented." if not annotated else "Run validation after supervised training.",
        },
    }
    write_json(PROJECT_ROOT / "reports/stage_04_entity_extraction.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
