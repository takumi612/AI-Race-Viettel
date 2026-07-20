from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.assertions import HybridAssertionPredictor
from clinical_nlp_lab.config import load_config
from clinical_nlp_lab.data import load_input_documents
from clinical_nlp_lab.kb import load_candidate_dictionary
from clinical_nlp_lab.ner import DictionaryRuleEntityDetector
from clinical_nlp_lab.schema import write_json


def main() -> None:
    config = load_config(PROJECT_ROOT / "artifacts/config.json")
    documents = load_input_documents(PROJECT_ROOT / config["input_zip"])
    detector = DictionaryRuleEntityDetector(
        load_candidate_dictionary(PROJECT_ROOT / "artifacts/icd10/icd10_dictionary.jsonl.gz"),
        load_candidate_dictionary(PROJECT_ROOT / "artifacts/rxnorm/rxnorm_dictionary.jsonl.gz"),
    )
    predictor = HybridAssertionPredictor()
    axis_counts = {axis: Counter() for axis in ("polarity", "temporality", "certainty", "experiencer")}
    internal_label_counts = Counter()
    entity_count = 0
    offset_errors = 0
    for document in documents:
        entities = detector.detect(document.raw_text)
        axes = predictor.predict(document.raw_text, entities)
        for entity in entities:
            entity_count += 1
            try:
                entity.validate_offset(document.raw_text)
            except ValueError:
                offset_errors += 1
            values = axes[(entity.start, entity.end, entity.type)]
            for axis in axis_counts:
                axis_counts[axis][getattr(values, axis)] += 1
            internal_label_counts.update(values.labels())

    report = {
        "stage": 5,
        "method": "hybrid clinical cues + section feature; optional multi-task XLM-R heads when annotations exist",
        "document_count": len(documents),
        "entity_count": entity_count,
        "offset_errors": offset_errors,
        "axis_counts": {axis: dict(counter.most_common()) for axis, counter in axis_counts.items()},
        "internal_label_counts": dict(internal_label_counts.most_common()),
        "official_mapping_status": "CONFIRMED_FROM_REPOSITORY_VALIDATOR",
        "official_assertions": ["isNegated", "isHistorical", "isFamily"],
        "threshold_tuning": {
            "status": "not_run",
            "reason": "Validation assertion labels are absent; default deterministic rules retained.",
        },
        "evaluation": {
            "status": "not_scored",
            "reason": "No assertion ground truth was provided; no Jaccard or macro-F1 was invented.",
        },
    }
    write_json(PROJECT_ROOT / "reports/stage_05_clinical_context.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
