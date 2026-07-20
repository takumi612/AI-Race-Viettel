from __future__ import annotations

import json
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from .assertions import HybridAssertionPredictor
from .config import load_config
from .data import load_input_documents, natural_document_key
from .kb import load_candidate_dictionary
from .linking import EntityLinker, LexicalCandidateIndex, parse_medication_attributes
from .ner import DictionaryRuleEntityDetector, TransformerNERDetector, refine_boundaries
from .relations import RuleRelationExtractor
from .schema import ClinicalDocument, EntityAnnotation, validate_submission_payload, write_json
from .text import containing_section, detect_sections


class ClinicalNLPPipeline:
    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        ner_model_dir: str | Path | None = None,
    ) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config = load_config(self.artifact_dir / "config.json")
        self.entity_mapping = self._read_json(self.artifact_dir / "entity_type_mapping.json")
        self.assertion_mapping = self._read_json(self.artifact_dir / "assertion_mapping.json")
        self.relation_mapping = self._read_json(self.artifact_dir / "relation_mapping.json")

        icd10_path = self.artifact_dir / "icd10" / "icd10_dictionary.jsonl.gz"
        rxnorm_path = self.artifact_dir / "rxnorm" / "rxnorm_dictionary.jsonl.gz"
        if not icd10_path.exists() or not rxnorm_path.exists():
            raise FileNotFoundError("Knowledge-base artifacts are missing; run tools/build_knowledge_bases.py first")
        self.icd10_records = load_candidate_dictionary(icd10_path)
        self.rxnorm_records = load_candidate_dictionary(rxnorm_path)
        if ner_model_dir is not None and Path(ner_model_dir).is_dir():
            self.detector = TransformerNERDetector(
                ner_model_dir,
                max_length=int(self.config["max_length"]),
                stride=int(self.config["stride"]),
            )
            self.active_ner = "transformer_checkpoint"
        else:
            self.detector = DictionaryRuleEntityDetector(
                self.icd10_records,
                self.rxnorm_records,
                phrase_confidence=float(self.config["thresholds"]["dictionary_phrase"]),
                regex_confidence=float(self.config["thresholds"]["regex_rule"]),
            )
            self.active_ner = "ontology_dictionary_plus_generic_rules"
        self.linker = EntityLinker(
            LexicalCandidateIndex(self.icd10_records, "ICD-10"),
            LexicalCandidateIndex(self.rxnorm_records, "RxNorm"),
            top_k=int(self.config["candidate_top_k"]),
            output_k=int(self.config["candidate_output_k"]),
            minimum_score=float(self.config["thresholds"]["candidate_min_score"]),
        )
        self.assertion_predictor = HybridAssertionPredictor()
        self.relation_extractor = RuleRelationExtractor(int(self.config["relation_max_distance"]))

    @staticmethod
    def _read_json(path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)

    def _official_assertions(self, internal_labels: list[str]) -> list[str]:
        mapping = self.assertion_mapping.get("internal_to_official", {})
        return [mapping[label] for label in internal_labels if mapping.get(label)]

    def process_document(self, document: ClinicalDocument) -> dict[str, Any]:
        raw_text = document.raw_text
        sections = detect_sections(raw_text)
        entities = refine_boundaries(self.detector.detect(raw_text), raw_text)
        axes_by_entity = self.assertion_predictor.predict(raw_text, entities)

        retrieval_diagnostics: list[dict[str, Any]] = []
        for entity in entities:
            candidate_ids, ranked = self.linker.retrieve(entity)
            entity.candidates = candidate_ids
            axes = axes_by_entity[(entity.start, entity.end, entity.type)]
            entity.assertions = axes.labels()
            retrieval_diagnostics.append(
                {
                    "position": [entity.start, entity.end],
                    "internal_type": entity.type,
                    "query": entity.mention_head or entity.text,
                    "top_candidates": ranked[:5],
                    "medication_attributes": parse_medication_attributes(entity.text) if entity.type == "DRUG" else None,
                }
            )
            entity.validate_offset(raw_text)

        relations = self.relation_extractor.extract(raw_text, entities)
        official_type_mapping = self.entity_mapping.get("internal_to_official", {})
        drop_unmapped = bool(self.entity_mapping.get("drop_unmapped", self.config["drop_unmapped_entity_types"]))
        submission_entities: list[dict[str, Any]] = []
        dropped_unmapped = Counter()
        for entity in entities:
            official_type = official_type_mapping.get(entity.type)
            if not official_type:
                dropped_unmapped[entity.type] += 1
                if drop_unmapped:
                    continue
                official_type = entity.type
            payload = entity.to_submission(official_type, self._official_assertions(entity.assertions))
            submission_entities.append(payload)

        validation_errors = validate_submission_payload(submission_entities, raw_text)
        if validation_errors:
            raise ValueError(f"Submission validation failed for {document.document_id}: {validation_errors}")
        return {
            "document_id": document.document_id,
            "submission": submission_entities,
            "diagnostics": {
                "raw_text_length": len(raw_text),
                "sections": [
                    {"section_name": item.section_name, "start": item.start, "end": item.end}
                    for item in sections
                ],
                "internal_entities": [entity.to_diagnostic() for entity in entities],
                "relations": [relation.to_dict() for relation in relations],
                "retrieval": retrieval_diagnostics,
                "dropped_unmapped_types": dict(dropped_unmapped),
                "submission_entity_count": len(submission_entities),
                "offset_validation_passed": True,
            },
        }


def run_inference(
    input_source: str | Path,
    output_dir: str | Path,
    artifact_dir: str | Path,
    create_zip: bool = True,
    diagnostics_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    ner_model_dir: str | Path | None = None,
) -> dict[str, Any]:
    documents = load_input_documents(input_source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics_path = Path(diagnostics_dir) if diagnostics_dir else output_path.parent / "diagnostics"
    diagnostics_path.mkdir(parents=True, exist_ok=True)

    for directory in (output_path, diagnostics_path):
        for existing in directory.glob("*.json"):
            existing.unlink()

    pipeline = ClinicalNLPPipeline(artifact_dir, ner_model_dir=ner_model_dir)
    type_counts = Counter()
    candidate_linked = 0
    relation_count = 0
    submission_entity_count = 0
    unmapped_type_counts: Counter[str] = Counter()
    offset_errors = 0
    results_by_id: dict[str, dict[str, Any]] = {}

    for document in documents:
        result = pipeline.process_document(document)
        results_by_id[document.document_id] = result
        write_json(output_path / f"{document.document_id}.json", result["submission"])
        write_json(diagnostics_path / f"{document.document_id}.json", result["diagnostics"])
        internal_entities = result["diagnostics"]["internal_entities"]
        type_counts.update(item["type"] for item in internal_entities)
        candidate_linked += sum(bool(item["candidates"]) for item in internal_entities)
        relation_count += len(result["diagnostics"]["relations"])
        submission_entity_count += len(result["submission"])
        unmapped_type_counts.update(result["diagnostics"]["dropped_unmapped_types"])
        offset_errors += sum(
            document.raw_text[item["position"][0]:item["position"][1]] != item["text"]
            for item in internal_entities
        )

    actual_files = sorted(output_path.glob("*.json"), key=lambda item: natural_document_key(item.stem))
    if len(actual_files) != len(documents):
        raise ValueError(f"Expected {len(documents)} output files, found {len(actual_files)}")
    for document, path in zip(documents, actual_files):
        if path.stem != document.document_id:
            raise ValueError(f"Output filename mismatch: expected {document.document_id}.json, found {path.name}")
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
        errors = validate_submission_payload(payload, document.raw_text)
        if errors:
            raise ValueError(f"Invalid output {path.name}: {errors}")

    final_zip: Path | None = None
    if create_zip:
        final_zip = Path(zip_path) if zip_path else output_path.parent / "output.zip"
        final_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(final_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in actual_files:
                archive.write(path, arcname=f"output/{path.name}")
        with zipfile.ZipFile(final_zip) as archive:
            names = archive.namelist()
            expected = [f"output/{path.name}" for path in actual_files]
            if names != expected:
                raise ValueError(f"Invalid output.zip structure: {names[:5]}")
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f"CRC failure in output.zip: {bad_member}")

    summary = {
        "document_count": len(documents),
        "output_json_count": len(actual_files),
        "internal_entity_count": sum(type_counts.values()),
        "internal_type_counts": dict(type_counts.most_common()),
        "candidate_linked_entity_count": candidate_linked,
        "diagnostic_relation_count": relation_count,
        "submission_entity_count": submission_entity_count,
        "unmapped_entity_count": sum(unmapped_type_counts.values()),
        "unmapped_type_counts": dict(unmapped_type_counts),
        "offset_error_count": offset_errors,
        "official_mapping_status": pipeline.entity_mapping.get("status"),
        "active_ner": pipeline.active_ner,
        "unmapped_entities_dropped": bool(unmapped_type_counts),
        "zip_path": str(final_zip) if final_zip else None,
        "zip_structure_valid": bool(final_zip),
        "training_or_fitting_on_input": False,
    }
    write_json(diagnostics_path / "run_summary.json", summary)
    return summary


def reload_equivalence_check(
    input_source: str | Path,
    artifact_dir: str | Path,
    sample_index: int = 0,
    ner_model_dir: str | Path | None = None,
) -> dict[str, Any]:
    documents = load_input_documents(input_source)
    if not documents:
        raise ValueError("No documents available for reload check")
    document = documents[sample_index]
    before = ClinicalNLPPipeline(artifact_dir, ner_model_dir=ner_model_dir).process_document(document)
    after = ClinicalNLPPipeline(artifact_dir, ner_model_dir=ner_model_dir).process_document(document)
    equivalent = before == after
    if not equivalent:
        raise AssertionError("Pipeline output changed after artifact reload")
    return {
        "document_id": document.document_id,
        "equivalent": equivalent,
        "internal_entity_count": len(before["diagnostics"]["internal_entities"]),
        "submission_entity_count": len(before["submission"]),
    }
