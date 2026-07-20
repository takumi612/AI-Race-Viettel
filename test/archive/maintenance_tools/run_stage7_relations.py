from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from clinical_nlp_lab.config import load_config
from clinical_nlp_lab.data import load_input_documents
from clinical_nlp_lab.kb import load_candidate_dictionary
from clinical_nlp_lab.ner import DictionaryRuleEntityDetector
from clinical_nlp_lab.relations import RuleRelationExtractor
from clinical_nlp_lab.schema import write_json


def main() -> None:
    config = load_config(PROJECT_ROOT / "artifacts/config.json")
    documents = load_input_documents(PROJECT_ROOT / config["input_zip"])
    detector = DictionaryRuleEntityDetector(
        load_candidate_dictionary(PROJECT_ROOT / "artifacts/icd10/icd10_dictionary.jsonl.gz"),
        load_candidate_dictionary(PROJECT_ROOT / "artifacts/rxnorm/rxnorm_dictionary.jsonl.gz"),
    )
    extractor = RuleRelationExtractor(int(config["relation_max_distance"]))
    relation_counts = Counter()
    relation_examples: list[dict] = []
    entity_pair_count = 0
    for document in documents:
        entities = detector.detect(document.raw_text)
        relation_predictions = extractor.extract(document.raw_text, entities)
        entity_pair_count += len(entities) * max(0, len(entities) - 1)
        for relation in relation_predictions:
            relation_counts[relation.relation] += 1
            if len(relation_examples) < 20:
                relation_examples.append({"document_id": document.document_id, **relation.to_dict()})

    report = {
        "stage": 7,
        "method": "same-sentence rule baseline + entity distance + type compatibility",
        "document_count": len(documents),
        "candidate_entity_pair_count": entity_pair_count,
        "relation_counts": dict(relation_counts.most_common()),
        "relation_prediction_count": sum(relation_counts.values()),
        "negative_sampling": {
            "status": "interface_ready",
            "reason": "No relation labels; supervised negative sampling was not run.",
        },
        "submission_policy": {
            "relation_in_submission": False,
            "reason": "Official submission schema is unconfirmed and the contract forbids extra keys.",
        },
        "evaluation": {
            "status": "not_scored",
            "reason": "No relation ground truth was provided; no precision/recall/F1 was invented.",
        },
        "examples": relation_examples,
    }
    write_json(PROJECT_ROOT / "reports/stage_07_relations.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
