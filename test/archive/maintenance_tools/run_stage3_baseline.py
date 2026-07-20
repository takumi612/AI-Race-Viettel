from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.config import load_config
from clinical_nlp_lab.data import describe_documents, load_annotated_documents, load_input_documents, validate_documents
from clinical_nlp_lab.kb import load_candidate_dictionary
from clinical_nlp_lab.ner import DictionaryRuleEntityDetector
from clinical_nlp_lab.schema import write_json
from clinical_nlp_lab.text import detect_sections


def main() -> None:
    root = PROJECT_ROOT
    config = load_config(root / "artifacts/config.json")
    documents = load_input_documents(root / config["input_zip"])
    annotated = load_annotated_documents(root / config["train_dir"])
    input_validation = validate_documents(documents)
    annotated_validation = validate_documents(annotated)
    detector = DictionaryRuleEntityDetector(
        load_candidate_dictionary(root / "artifacts/icd10/icd10_dictionary.jsonl.gz"),
        load_candidate_dictionary(root / "artifacts/rxnorm/rxnorm_dictionary.jsonl.gz"),
    )
    section_counts = Counter()
    baseline_type_counts = Counter()
    baseline_entity_count = 0
    offset_checks = 0
    for document in documents:
        section_counts.update(section.section_name for section in detect_sections(document.raw_text))
        entities = detector.detect(document.raw_text)
        baseline_entity_count += len(entities)
        baseline_type_counts.update(entity.type for entity in entities)
        for entity in entities:
            entity.validate_offset(document.raw_text)
            offset_checks += 1

    report = {
        "stage": 3,
        "data_summary": describe_documents(documents),
        "input_validator": input_validation,
        "annotated_train_summary": describe_documents(annotated),
        "annotated_train_validator": annotated_validation,
        "annotated_train_documents_found": len(annotated),
        "section_counts": dict(section_counts.most_common()),
        "baseline": {
            "method": "ICD-10/RxNorm dictionary phrase matching plus generic clinical rules",
            "entity_count": baseline_entity_count,
            "type_counts": dict(baseline_type_counts.most_common()),
            "offset_checks": offset_checks,
            "offset_errors": 0,
            "fit_on_input": False,
        },
        "evaluation": {
            "status": "not_scored",
            "reason": "No train/validation ground truth is present; no score was invented.",
        },
    }
    write_json(root / "reports/stage_03_eda.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
