import os
import sys
import json
import glob

# Thêm project root vào sys.path để hỗ trợ import chéo
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.config import PROJECT_ROOT, PipelineConfig
from src.chunking.clinical_chunker import ClinicalChunker
from src.utils.paths import INPUT_DIR, OUTPUT_DIR
from src.validation.override_validator import load_verified_overrides, normalize_override_term
from src.validation.submission import write_failure_output
from src.validation.patient_extractor import PatientExtractor
from src.ner.extractor import BaselineExtractor
from src.ner.model_extractor import ModelNERExtractor, merge_hybrid_entities
from src.retrieval.normalizer import TextNormalizer
from src.assertion.rule_based import AssertionAnalyzer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.candidate_selector import CandidateSelector
from src.retrieval.types import RetrievedCandidate
from src.validation.clinical_validator import ClinicalValidator
from src.ranking.llm_reranker import LLMReranker

VERIFIED_OVERRIDES_PATH = PROJECT_ROOT / "src" / "resources" / "verified_overrides.json"


class BaselinePipeline:
    def __init__(self, config: PipelineConfig | None = None):
        print("Initializing Baseline Pipeline...")
        self.config = config or PipelineConfig()
        self.clinical_chunker = ClinicalChunker(self.config.chunking)
        self.patient_extractor = PatientExtractor()
        self.ner_extractor = BaselineExtractor()
        self.model_ner_extractor = None
        if self.config.ner.mode != "rule":
            artifact_path = PROJECT_ROOT / str(self.config.ner.model_artifact)
            try:
                self.model_ner_extractor = ModelNERExtractor(
                    artifact_path,
                    threshold=self.config.ner.model_threshold,
                )
            except (OSError, ValueError) as exc:
                if self.config.ner.mode == "model":
                    raise
                print(f"  - NER hybrid fallback to rule mode: {exc}")
        self.normalizer = TextNormalizer()
        self.assertion_analyzer = AssertionAnalyzer(config=self.config)
        retrieval = self.config.retrieval
        embedding_model_path = (
            str(PROJECT_ROOT / retrieval.embedding_model_artifact)
            if retrieval.embedding_model_artifact is not None
            else None
        )
        self.retriever = HybridRetriever(
            table_name="icd10",
            alpha=retrieval.alpha,
            internal_top_k=retrieval.internal_top_k,
            hierarchical_expansion=retrieval.hierarchical_expansion,
            embedding_model_type=retrieval.embedding_model_type,
            embedding_model_path=embedding_model_path,
        )
        self.rxnorm_retriever = HybridRetriever(
            table_name="rxnorm",
            alpha=retrieval.alpha,
            internal_top_k=retrieval.internal_top_k,
            hierarchical_expansion=retrieval.hierarchical_expansion,
            embedding_model_type=retrieval.embedding_model_type,
            embedding_model_path=embedding_model_path,
        )
        self.clinical_validator = ClinicalValidator(
            load_historical_rxnorm=self.config.selection.load_historical_rxnorm
        )
        self.candidate_selector = CandidateSelector(self.config.selection)
        self.llm_reranker = (
            LLMReranker(
                use_llm=True,
                backend=self.config.reranker.backend,
                model_artifact=self.config.reranker.model_artifact,
                project_root=PROJECT_ROOT,
                max_new_tokens=self.config.reranker.max_new_tokens,
                timeout_seconds=self.config.reranker.timeout_seconds,
            )
            if self.config.reranker.enabled
            else None
        )
        
        self.override_dict = {}
        for entry in load_verified_overrides(VERIFIED_OVERRIDES_PATH):
            entries_by_term = self.override_dict.setdefault(entry["type"], {})
            entries_by_term[normalize_override_term(entry["term"])] = entry["codes"]
        print(
            "  - Loaded verified overrides: "
            f"{len(self.override_dict.get('CHẨN_ĐOÁN', {}))} diagnoses, "
            f"{len(self.override_dict.get('THUỐC', {}))} drugs."
        )
            
        print("Pipeline initialized successfully.")

    def _select_ranked(self, entity, ranked, patient_info):
        """Apply clinical predicates, optional LLM subset, then deterministic gates."""
        if not hasattr(self, "candidate_selector") or not hasattr(self.clinical_validator, "is_candidate_valid"):
            return [candidate.code for candidate in ranked]
        entity_type = entity["type"]

        def is_valid(code):
            return self.clinical_validator.is_candidate_valid(entity, code, patient_info)

        valid_ranked = [candidate for candidate in ranked if is_valid(candidate.code)]
        llm_reranker = getattr(self, "llm_reranker", None)
        if llm_reranker and valid_ranked:
            selected_codes = llm_reranker.rerank(
                entity.get("context", ""),
                entity["text"],
                entity_type,
                [candidate.code for candidate in valid_ranked],
            )
            selected_set = set(selected_codes)
            valid_ranked = [candidate for candidate in valid_ranked if candidate.code in selected_set]
        return self.candidate_selector.select(entity_type, valid_ranked, is_valid)

    def _select_override(self, entity, codes, patient_info):
        ranked = [
            RetrievedCandidate(
                code=str(code).strip().upper(),
                fusion_score=1.0,
                bm25_score=1.0,
                semantic_score=0.0,
                bm25_rank=index,
                semantic_rank=None,
            )
            for index, code in enumerate(codes)
        ]
        return self._select_ranked(entity, ranked, patient_info)

    def process_text(self, text: str):
        """Xử lý một file văn bản lâm sàng đầu vào."""
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        # 1. Trích xuất giới tính/độ tuổi bệnh nhân
        patient_info = self.patient_extractor.extract(text)
        print(f"  - Patient Demographics: {patient_info}")
        
        # 2. Nhận dạng thực thể thô (Baseline)
        chunks = self.clinical_chunker.chunk(text)
        rule_entities = self.ner_extractor.extract_entities(text, chunks=chunks)
        model_ner_extractor = getattr(self, "model_ner_extractor", None)
        if model_ner_extractor is None:
            raw_entities = rule_entities
        elif self.config.ner.mode == "model":
            raw_entities = model_ner_extractor.extract_entities(text)
        else:
            model_entities = model_ner_extractor.extract_entities(text)
            raw_entities = merge_hybrid_entities(
                text,
                rule_entities,
                model_entities,
                default_threshold=self.config.ner.default_threshold,
                per_type_thresholds=self.config.ner.per_type_thresholds,
            )
        print(f"  - Extracted {len(raw_entities)} raw entities.")
        
        processed_entities = []
        for ent in raw_entities:
            ent_type = ent['type']
            ent_text = ent['text']
            start_idx, end_idx = ent['position']
            if (
                isinstance(start_idx, bool)
                or isinstance(end_idx, bool)
                or not isinstance(start_idx, int)
                or not isinstance(end_idx, int)
                or not 0 <= start_idx < end_idx <= len(text)
                or text[start_idx:end_idx] != ent_text
            ):
                raise ValueError("entity position must exactly slice its text from the document")
            
            # Cấu trúc đối tượng JSON cho thực thể
            processed_ent = {
                "text": ent_text,
                "type": ent_type,
                "position": ent['position']
            }
            
            # 3. Phân tích thuộc tính ngữ cảnh (chỉ áp dụng cho chẩn đoán, thuốc, triệu chứng)
            if ent_type in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                containing_chunk = next(
                    (
                        chunk
                        for chunk in chunks
                        if chunk.start <= start_idx < chunk.end
                    ),
                    None,
                )
                if containing_chunk is None:
                    raise ValueError("entity start is not contained in a ClinicalChunk")
                assertions = self.assertion_analyzer.analyze(
                    text,
                    start_idx,
                    end_idx,
                    section_type=containing_chunk.section_type,
                    header_text=containing_chunk.header_text,
                )
                processed_ent["assertions"] = assertions
                
            # 4. Tìm kiếm ứng viên (chỉ áp dụng cho chẩn đoán, thuốc)
            if ent_type == "CHẨN_ĐOÁN":
                # Chuẩn hóa văn bản chẩn đoán
                clean_text = self.normalizer.clean_text(ent_text)
                expanded_text = self.normalizer.expand_abbreviation(clean_text)
                
                # 4a. Kiểm tra bảng ánh xạ cứng (Override) trước
                override_key = normalize_override_term(expanded_text)
                if override_key in self.override_dict.get("CHẨN_ĐOÁN", {}):
                    override_codes = self.override_dict["CHẨN_ĐOÁN"][override_key]
                    print(f"  [OVERRIDE MATCH] Mapped diagnosis to {override_codes}")
                    candidates = self._select_override(processed_ent, override_codes, patient_info)
                else:
                    # Tra cứu Hybrid (BM25s + FAISS) tìm mã ICD-10
                    if hasattr(self.retriever, "retrieve_scored") and hasattr(self, "candidate_selector"):
                        ranked = self.retriever.retrieve_scored(
                            expanded_text,
                            top_k=self.config.retrieval.internal_top_k,
                        )
                    # Xếp hạng lại bằng LLM
                        candidates = self._select_ranked(processed_ent, ranked, patient_info)
                    else:
                        candidates = self.retriever.retrieve(expanded_text, top_k=5)
                        processed_ent["candidates"] = candidates
                        processed_ent = self.clinical_validator.check_and_fix_candidates(
                            processed_ent, patient_info
                        )
                
                processed_ent["candidates"] = candidates[:2]
                
                # Áp dụng bộ lọc luật lâm sàng
                
            elif ent_type == "THUỐC":
                # Chuẩn hóa văn bản thuốc
                clean_text = self.normalizer.clean_text(ent_text)
                
                # Bóc tách cách dùng và ký hiệu viết tắt Latin nhưng GIỮ LẠI liều lượng (mg, ml...)
                clean_drug_name = self.normalizer.remove_dosage(clean_text)
                expanded_text = self.normalizer.expand_abbreviation(clean_drug_name)
                
                # 4a. Kiểm tra bảng ánh xạ cứng (Override) trước cho tên đầy đủ
                override_key = normalize_override_term(expanded_text)
                if override_key in self.override_dict.get("THUỐC", {}):
                    candidates = self.override_dict["THUỐC"][override_key]
                    print(f"  [OVERRIDE MATCH] Mapped drug to {candidates}")
                else:
                    # Tra cứu Hybrid (BM25s + FAISS) tìm mã RxNorm bằng tên thuốc sạch có liều lượng
                    if hasattr(self.rxnorm_retriever, "retrieve_scored") and hasattr(self, "candidate_selector"):
                        ranked = self.rxnorm_retriever.retrieve_scored(
                            expanded_text,
                            top_k=self.config.retrieval.internal_top_k,
                        )
                    # Xếp hạng lại bằng LLM
                        candidates = self._select_ranked(processed_ent, ranked, patient_info)
                    else:
                        candidates = self.rxnorm_retriever.retrieve(expanded_text, top_k=5)
                        processed_ent["candidates"] = candidates
                        processed_ent = self.clinical_validator.check_and_fix_candidates(
                            processed_ent, patient_info
                        )
                
                processed_ent["candidates"] = candidates[:2]
                
                # Áp dụng bộ lọc luật lâm sàng (bao gồm cả kiểm tra dạng bào chế)
                
            processed_entities.append(processed_ent)
            
        # 5. Kiểm tra tính toàn vẹn của mã kép chẩn đoán
        processed_entities = self.clinical_validator.check_dual_codes(processed_entities)
        
        return processed_entities

    def process_file(self, file_path):
        """Read UTF-8 text and delegate to the side-effect-free text API."""
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        with open(file_path, "r", encoding="utf-8") as handle:
            return self.process_text(handle.read())

    def run(self, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
        """Chạy pipeline trên toàn bộ file txt trong input_dir."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        print(f"Found {len(txt_files)} text files in input directory: {input_dir}")
        
        for file_path in txt_files:
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.json")
            error_log_path = os.path.join(output_dir, "errors.jsonl")
            try:
                results = self.process_file(file_path)
                
                # Ghi kết quả JSON
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
                print(f"  - Saved prediction to {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")
                write_failure_output(output_path, error_log_path, base_name, e)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Race Viettel Pipeline Runner")
    parser.add_argument("--input", type=str, default=INPUT_DIR, help="Input directory containing text files")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory to save JSON results")
    parser.add_argument("--config", type=str, default=None, help="Validated PipelineConfig JSON")
    args = parser.parse_args()

    loaded_config = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as handle:
            config_payload = json.load(handle)
        if isinstance(config_payload, dict) and isinstance(config_payload.get("config"), dict):
            config_payload = config_payload["config"]
        loaded_config = PipelineConfig.from_mapping(config_payload)
    pipeline = BaselinePipeline(config=loaded_config)
    pipeline.run(input_dir=args.input, output_dir=args.output)
