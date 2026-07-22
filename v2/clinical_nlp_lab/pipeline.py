from __future__ import annotations

import json
import logging
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

from .assertions import HybridAssertionPredictor
from .candidate_policy import CandidatePolicy, apply_candidate_policy
from .config import load_config
from .data import load_input_documents, natural_document_key
from .kb import load_candidate_dictionary
from .linking import parse_medication_attributes
from .retrieval import HybridCandidateIndex, HybridEntityLinker, create_embedding_model
from .ner import DictionaryRuleEntityDetector, TransformerNERDetector, refine_boundaries, resolve_overlaps
from .relations import RuleRelationExtractor
from .schema import ClinicalDocument, validate_submission_payload, write_json
from .text import detect_sections


def enrich_records_from_train_documents(
    icd10_records: list[dict[str, Any]],
    rxnorm_records: list[dict[str, Any]],
    train_documents: Iterable[ClinicalDocument],
) -> tuple[int, int]:
    icd10_map = {str(r["candidate_id"]): r for r in icd10_records}
    rxnorm_map = {str(r["candidate_id"]): r for r in rxnorm_records}
    icd10_added = 0
    rxnorm_added = 0

    for doc in train_documents:
        for entity in doc.entities:
            text = entity.text.strip()
            if not text or not entity.candidates:
                continue
            for code in entity.candidates:
                code_str = str(code).strip()
                if code_str in icd10_map:
                    aliases = icd10_map[code_str].setdefault("aliases", [])
                    if text not in aliases:
                        aliases.append(text)
                        icd10_added += 1
                elif code_str in rxnorm_map:
                    aliases = rxnorm_map[code_str].setdefault("aliases", [])
                    if text not in aliases:
                        aliases.append(text)
                        rxnorm_added += 1

    return icd10_added, rxnorm_added


class ClinicalNLPPipeline:
    def __init__(
        self,
        artifact_dir: str | Path = "artifacts",
        ner_model_dir: str | Path | None = None,
        train_documents: Iterable[ClinicalDocument] | None = None,
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

        if train_documents:
            added_icd, added_rx = enrich_records_from_train_documents(
                self.icd10_records, self.rxnorm_records, train_documents
            )
            print(f"[RE-INDEX] Enriched BM25+FAISS candidate records from train data: +{added_icd} ICD-10 aliases, +{added_rx} RxNorm aliases.")

        self.dict_detector = DictionaryRuleEntityDetector(
            self.icd10_records,
            self.rxnorm_records,
            phrase_confidence=float(self.config["thresholds"]["dictionary_phrase"]),
            regex_confidence=float(self.config["thresholds"]["regex_rule"]),
            enable_generic_regex=bool(self.config.get("enable_regex_fallback", False)),
        )
        if ner_model_dir is not None and Path(ner_model_dir).is_dir():
            self.trans_detector = TransformerNERDetector(
                ner_model_dir,
                max_length=int(self.config["max_length"]),
                stride=int(self.config["stride"]),
            )
            self.active_ner = "hybrid_transformer_and_dictionary"
        else:
            self.trans_detector = None
            self.active_ner = "ontology_dictionary_only"
            
        embedding_model_name = str(self.config["embedding_model_name"])
        shared_embedding_model = create_embedding_model(embedding_model_name)
        icd10_index = HybridCandidateIndex(
            self.icd10_records,
            "ICD-10",
            embedding_model_name=embedding_model_name,
            embedding_model=shared_embedding_model,
        )
        icd10_index.build_indexes()
        rxnorm_index = HybridCandidateIndex(
            self.rxnorm_records,
            "RxNorm",
            embedding_model_name=embedding_model_name,
            embedding_model=shared_embedding_model,
        )
        rxnorm_index.build_indexes()
        self.icd10_index = icd10_index
        self.rxnorm_index = rxnorm_index
        self.linker = HybridEntityLinker(
            icd10_index,
            rxnorm_index,
            top_k=int(self.config["candidate_top_k"])
        )
        self.candidate_policy = CandidatePolicy(
            min_score=float(self.config["thresholds"].get("candidate_min_score", 0.5)),
            min_margin=float(self.config["thresholds"].get("candidate_min_margin", 0.05)),
            output_k=int(self.config.get("candidate_output_k", 1)),
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

    def release_for_llm(self) -> None:
        """Release NER and retrieval resources before vLLM claims GPU/RAM."""
        if self.trans_detector is not None:
            self.trans_detector.release()
            self.trans_detector = None
        self.icd10_index.release()
        self.rxnorm_index.release()
        self.linker = None
        self.dict_detector = None
        self.icd10_records = []
        self.rxnorm_records = []
        self.assertion_predictor = None
        self.relation_extractor = None

    def process_document(self, document: ClinicalDocument) -> dict[str, Any]:
        raw_text = document.raw_text
        sections = detect_sections(raw_text)
        entities = self.dict_detector.detect(raw_text)
        if self.trans_detector:
            trans_entities = self.trans_detector.detect(raw_text)
            
            official_to_internal = {
                v: k for k, v in self.entity_mapping.get("internal_to_official", {}).items() if v
            }
            for e in trans_entities:
                if e.type in official_to_internal:
                    e.type = official_to_internal[e.type]
                    
            entities = resolve_overlaps(entities + trans_entities, raw_text)
        entities = refine_boundaries(entities, raw_text)
        axes_by_entity = self.assertion_predictor.predict(raw_text, entities)

        retrieval_diagnostics: list[dict[str, Any]] = []
        for entity in entities:
            existing_candidates = list(entity.candidates)
            _, ranked = self.linker.retrieve(
                entity.type,
                entity.text,
                mention_head=entity.mention_head,
                existing_candidates=existing_candidates,
            )
            entity.candidates = apply_candidate_policy(ranked, self.candidate_policy)
            axes = axes_by_entity[(entity.start, entity.end, entity.type)]
            entity.assertions = axes.labels()
            retrieval_diagnostics.append(
                {
                    "position": [entity.start, entity.end],
                    "internal_type": entity.type,
                    "query": entity.mention_head or entity.text,
                    "top_candidates": ranked[: int(self.config["candidate_top_k"])],
                    "selected_candidates": list(entity.candidates),
                    "candidate_abstained": not bool(entity.candidates),
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
            "raw_entities": entities,
        }


def run_inference(
    input_source: str | Path,
    output_dir: str | Path,
    artifact_dir: str | Path,
    create_zip: bool = True,
    diagnostics_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    ner_model_dir: str | Path | None = None,
    enable_qwen_reranker: bool = False,
    qwen_model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    qwen_gpu_memory_utilization: float = 0.5,
    qwen_max_model_len: int = 4096,
    qwen_batch_size: int = 64,
    train_source: str | Path | None = None,
    train_documents: Sequence[ClinicalDocument] | None = None,
) -> dict[str, Any]:
    def gpu_memory_snapshot(stage: str) -> dict[str, str | float | int | None]:
        try:
            import torch
            if not torch.cuda.is_available():
                return {
                    "stage": stage,
                    "device": None,
                    "free_gib": None,
                    "total_gib": None,
                    "allocated_gib": None,
                    "reserved_gib": None,
                    "peak_allocated_gib": None,
                }
            free, total = torch.cuda.mem_get_info()
            gib = 1024 ** 3
            return {
                "stage": stage,
                "device": torch.cuda.get_device_name(0),
                "free_gib": round(free / gib, 2),
                "total_gib": round(total / gib, 2),
                "allocated_gib": round(torch.cuda.memory_allocated() / gib, 2),
                "reserved_gib": round(torch.cuda.memory_reserved() / gib, 2),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / gib, 2),
            }
        except Exception:
            return {
                "stage": stage,
                "device": None,
                "free_gib": None,
                "total_gib": None,
                "allocated_gib": None,
                "reserved_gib": None,
                "peak_allocated_gib": None,
            }

    def log_gpu_state(stage: str) -> dict[str, str | float | int | None]:
        snapshot = gpu_memory_snapshot(stage)
        print(f"[GPU] {json.dumps(snapshot, ensure_ascii=False)}")
        return snapshot

    gpu_memory_start = log_gpu_state("pipeline_start")
    documents = load_input_documents(input_source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    diagnostics_path = Path(diagnostics_dir) if diagnostics_dir else output_path.parent / "diagnostics"
    diagnostics_path.mkdir(parents=True, exist_ok=True)

    for directory in (output_path, diagnostics_path):
        for existing in directory.glob("*.json"):
            existing.unlink()

    if train_documents is None and train_source is not None:
        try:
            from .data import load_ner_training_documents
            train_documents = load_ner_training_documents(train_source)
        except Exception:
            train_documents = None

    pipeline = ClinicalNLPPipeline(artifact_dir, ner_model_dir=ner_model_dir, train_documents=train_documents)
    gpu_memory_after_ner_load = log_gpu_state("after_ner_load")
    
    # PASS 1: NER, Assertion (Hybrid fallback), Hybrid Retrieval, Relations
    intermediate_results = {}
    for document in documents:
        intermediate_results[document.document_id] = pipeline.process_document(document)

    gpu_memory_after_ner_pass = log_gpu_state("after_ner_pass")
    gpu_memory_before_release = log_gpu_state("before_ner_release")
    pipeline.release_for_llm()
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gpu_memory_before_qwen = log_gpu_state("before_qwen")

    # PASS 2: LLM Assertion & Reranking (Batched)
    llm_reranker_enabled = False
    llm_assertion_enabled = False
    llm_fallback_reason = None
    reranker = None
    if enable_qwen_reranker:
        try:
            from .reranker import ClinicalLLMReranker
            from .assertions import ClinicalLLMAssertionPredictor
            
            reranker = ClinicalLLMReranker(
                model_name=qwen_model_name,
                gpu_memory_utilization=qwen_gpu_memory_utilization,
                max_model_len=qwen_max_model_len,
                batch_size=qwen_batch_size,
            )
            llm_assertion = ClinicalLLMAssertionPredictor(reranker.llm)
            
            # Prepare queries
            rerank_queries = []
            assertion_queries = []
            entity_refs = []
            
            for document in documents:
                result = intermediate_results[document.document_id]
                raw_text = document.raw_text
                entities = result["raw_entities"]
                
                retrieval_diags = result["diagnostics"]["retrieval"]
                
                for entity, rdiag in zip(entities, retrieval_diags):
                    cands = rdiag["top_candidates"]
                    if cands:
                        start_idx = max(0, entity.start - 50)
                        end_idx = min(len(raw_text), entity.end + 50)
                        context = raw_text[start_idx:end_idx]
                        rerank_queries.append({
                            "context_text": context,
                            "entity_text": entity.text,
                            "entity_type": entity.type,
                            "candidates": cands
                        })
                        entity_refs.append(entity)
                        
                    # Assertion queries
                    start_idx_a = max(0, entity.start - 120)
                    end_idx_a = min(len(raw_text), entity.end + 120)
                    context_a = raw_text[start_idx_a:end_idx_a]
                    assertion_queries.append({
                        "context": context_a,
                        "entity_text": entity.text
                    })
                    
            # Run Rerank
            if rerank_queries:
                rerank_results = reranker.rerank_batch(rerank_queries)
                llm_reranker_enabled = True
                for entity, selected_id in zip(entity_refs, rerank_results):
                    if selected_id:
                        entity.candidates = [selected_id]
                    else:
                        entity.candidates = entity.candidates[:1]
                        
            # Run Assertion
            if assertion_queries:
                assertion_results = llm_assertion.predict_batch(assertion_queries)
                llm_assertion_enabled = True
                flat_entities = [ent for doc in documents for ent in intermediate_results[doc.document_id]["raw_entities"]]
                for entity, axes in zip(flat_entities, assertion_results):
                    entity.assertions = axes.labels()
                    
        except Exception as exc:
            if reranker is not None:
                reranker.destroy()
                reranker = None
            # Deterministic NER, assertion and KB linking are already complete;
            # Qwen is an optional refinement stage and must not invalidate a
            # submission when unavailable or when GPU memory is insufficient.
            llm_fallback_reason = f"{type(exc).__name__}: {exc}"
            llm_reranker_enabled = False
            llm_assertion_enabled = False
        finally:
            if reranker is not None:
                reranker.destroy()
            gpu_memory_after_qwen = log_gpu_state("after_qwen")
    else:
        llm_fallback_reason = "Qwen reranker disabled by configuration"
        gpu_memory_after_qwen = log_gpu_state("after_qwen_disabled")
        
    type_counts = Counter()
    candidate_linked = 0
    relation_count = 0
    submission_entity_count = 0
    unmapped_type_counts: Counter[str] = Counter()
    offset_errors = 0
    
    for document in documents:
        result = intermediate_results[document.document_id]
        entities = result["raw_entities"]
        
        # Re-build submission and diagnostics with updated entities
        submission_entities = []
        official_type_mapping = pipeline.entity_mapping.get("internal_to_official", {})
        drop_unmapped = bool(pipeline.entity_mapping.get("drop_unmapped", pipeline.config.get("drop_unmapped_entity_types", True)))
        
        for entity in entities:
            official_type = official_type_mapping.get(entity.type)
            if not official_type:
                if drop_unmapped:
                    continue
                official_type = entity.type
            payload = entity.to_submission(official_type, pipeline._official_assertions(entity.assertions))
            submission_entities.append(payload)
            
        result["submission"] = submission_entities
        result["diagnostics"]["internal_entities"] = [entity.to_diagnostic() for entity in entities]
        result["diagnostics"]["submission_entity_count"] = len(submission_entities)
        
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
        "llm_reranker_enabled": llm_reranker_enabled,
        "llm_assertion_enabled": llm_assertion_enabled,
        "llm_fallback_reason": llm_fallback_reason,
        "gpu_memory_start": gpu_memory_start,
        "gpu_memory_after_ner_load": gpu_memory_after_ner_load,
        "gpu_memory_after_ner_pass": gpu_memory_after_ner_pass,
        "gpu_memory_before_release": gpu_memory_before_release,
        "gpu_memory_before_qwen": gpu_memory_before_qwen,
        "gpu_memory_after_qwen": gpu_memory_after_qwen,
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
