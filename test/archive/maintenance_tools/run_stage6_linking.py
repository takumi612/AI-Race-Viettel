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
from clinical_nlp_lab.linking import EntityLinker, LexicalCandidateIndex, parse_medication_attributes
from clinical_nlp_lab.ner import DictionaryRuleEntityDetector
from clinical_nlp_lab.schema import write_json
from clinical_nlp_lab.text import normalize_alias


def main() -> None:
    config = load_config(PROJECT_ROOT / "artifacts/config.json")
    icd10_records = load_candidate_dictionary(PROJECT_ROOT / "artifacts/icd10/icd10_dictionary.jsonl.gz")
    rxnorm_records = load_candidate_dictionary(PROJECT_ROOT / "artifacts/rxnorm/rxnorm_dictionary.jsonl.gz")
    detector = DictionaryRuleEntityDetector(icd10_records, rxnorm_records)
    linker = EntityLinker(
        LexicalCandidateIndex(icd10_records, "ICD-10"),
        LexicalCandidateIndex(rxnorm_records, "RxNorm"),
        top_k=config["candidate_top_k"],
        output_k=config["candidate_output_k"],
        minimum_score=config["thresholds"]["candidate_min_score"],
    )
    documents = load_input_documents(PROJECT_ROOT / config["input_zip"])
    type_counts = Counter()
    linked_counts = Counter()
    drug_attribute_counts = Counter()
    offset_errors = 0
    score_counts = Counter()
    examples: list[dict] = []
    for document in documents:
        for entity in detector.detect(document.raw_text):
            type_counts[entity.type] += 1
            candidate_ids, ranked = linker.retrieve(entity)
            if candidate_ids:
                linked_counts[entity.type] += 1
            if entity.type == "DRUG":
                parsed = parse_medication_attributes(entity.text)
                drug_attribute_counts.update(key for key, value in parsed.items() if value)
            if ranked:
                score_counts["score_ge_0_5"] += ranked[0]["score"] >= 0.5
                score_counts["score_ge_0_8"] += ranked[0]["score"] >= 0.8
            try:
                entity.validate_offset(document.raw_text)
            except ValueError:
                offset_errors += 1
            if len(examples) < 12 and candidate_ids:
                examples.append(
                    {
                        "document_id": document.document_id,
                        "text": entity.text,
                        "type": entity.type,
                        "candidates": candidate_ids,
                        "ranked": ranked[:3],
                    }
                )

    sanity = []
    for source_name, source_records, index in (
        ("icd10", icd10_records[:500], linker.icd10_index),
        ("rxnorm", rxnorm_records[:500], linker.rxnorm_index),
    ):
      for record in source_records:
        alias = (record.get("detection_aliases") or record.get("aliases") or [""])[0]
        top_ids = index.alias_to_ids.get(normalize_alias(alias), [])
        sanity.append(bool(top_ids and record["candidate_id"] in top_ids))

    report = {
        "stage": 6,
        "method": "type-routed exact lookup + fuzzy SequenceMatcher + character n-gram retrieval + weighted lexical reranking",
        "ontology_sizes": {"icd10": len(icd10_records), "rxnorm": len(rxnorm_records)},
        "input_documents": len(documents),
        "detected_type_counts": dict(type_counts.most_common()),
        "linked_type_counts": dict(linked_counts.most_common()),
        "drug_attribute_counts": dict(drug_attribute_counts.most_common()),
        "offset_errors": offset_errors,
        "score_threshold_counts": dict(score_counts),
        "ontology_self_lookup_sanity": {
            "sample_size": len(sanity),
            "top1_hits": sum(sanity),
            "top1_recall": round(sum(sanity) / len(sanity), 6) if sanity else 0.0,
            "note": "Sanity check on ontology aliases only; not a competition validation score.",
        },
        "candidate_recall_at_k": {
            "status": "not_scored",
            "reason": "No annotated ground truth; private input was not used to fit or tune retrieval.",
        },
        "semantic_embedding": {
            "status": "optional_not_active",
            "reason": "No sentence-transformers/FAISS dependency is required for the verified offline baseline.",
        },
        "examples": examples,
    }
    write_json(PROJECT_ROOT / "reports/stage_06_entity_linking.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
