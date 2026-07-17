import os
import sys
import json
import glob
import re

# Thêm project root vào sys.path để hỗ trợ import chéo
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.paths import INPUT_DIR, OUTPUT_DIR, KB_DIR
from src.validation.patient_extractor import PatientExtractor
from src.ner.extractor import BaselineExtractor
from src.retrieval.normalizer import TextNormalizer
from src.assertion.rule_based import AssertionAnalyzer
from src.retrieval.hybrid_retriever import HybridRetriever
from src.validation.clinical_validator import ClinicalValidator
from src.ranking.llm_reranker import LLMReranker

class BaselinePipeline:
    def __init__(self):
        print("Initializing Baseline Pipeline...")
        self.patient_extractor = PatientExtractor()
        self.ner_extractor = BaselineExtractor()
        self.normalizer = TextNormalizer()
        self.assertion_analyzer = AssertionAnalyzer()
        self.retriever = HybridRetriever(table_name="icd10")
        self.rxnorm_retriever = HybridRetriever(table_name="rxnorm")
        self.clinical_validator = ClinicalValidator()
        self.llm_reranker = LLMReranker(use_llm=os.getenv("USE_LLM_RERANKER", "False").lower() == "true")
        
        # Load override dictionary nếu tồn tại
        self.override_dict = {}
        override_path = os.path.join(KB_DIR, "override_dict.json")
        if os.path.exists(override_path):
            with open(override_path, "r", encoding="utf-8") as f:
                self.override_dict = json.load(f)
            print(f"  - Loaded static overrides: {len(self.override_dict.get('CHẨN_ĐOÁN', {}))} diagnoses, {len(self.override_dict.get('THUỐC', {}))} drugs.")
        else:
            print("  - [WARNING] Static override dictionary not found.")
            
        print("Pipeline initialized successfully.")

    def process_file(self, file_path):
        """Xử lý một file văn bản lâm sàng đầu vào."""
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 1. Trích xuất giới tính/độ tuổi bệnh nhân
        patient_info = self.patient_extractor.extract(text)
        print(f"  - Patient Demographics: {patient_info}")
        
        # 2. Nhận dạng thực thể thô (Baseline)
        raw_entities = self.ner_extractor.extract_entities(text)
        print(f"  - Extracted {len(raw_entities)} raw entities.")
        
        processed_entities = []
        for ent in raw_entities:
            ent_type = ent['type']
            ent_text = ent['text']
            start_idx, end_idx = ent['position']
            
            # Cấu trúc đối tượng JSON cho thực thể
            processed_ent = {
                "text": ent_text,
                "type": ent_type,
                "position": ent['position']
            }
            
            # 3. Phân tích thuộc tính ngữ cảnh (chỉ áp dụng cho chẩn đoán, thuốc, triệu chứng)
            if ent_type in ("CHẨN_ĐOÁN", "THUỐC", "TRIỆU_CHỨNG"):
                assertions = self.assertion_analyzer.analyze(text, start_idx, end_idx)
                processed_ent["assertions"] = assertions
                
            # 4. Tìm kiếm ứng viên (chỉ áp dụng cho chẩn đoán, thuốc)
            if ent_type == "CHẨN_ĐOÁN":
                # Chuẩn hóa văn bản chẩn đoán
                clean_text = self.normalizer.clean_text(ent_text)
                expanded_text = self.normalizer.expand_abbreviation(clean_text)
                
                # 4a. Kiểm tra bảng ánh xạ cứng (Override) trước
                override_key = expanded_text.lower()
                if override_key in self.override_dict.get("CHẨN_ĐOÁN", {}):
                    candidates = self.override_dict["CHẨN_ĐOÁN"][override_key]
                    print(f"  [OVERRIDE MATCH] Mapped diagnosis to {candidates}")
                else:
                    # Tra cứu Hybrid (BM25s + FAISS) tìm mã ICD-10
                    candidates = self.retriever.retrieve(expanded_text, top_k=5)
                    # Xếp hạng lại bằng LLM
                    if self.llm_reranker:
                        candidates = self.llm_reranker.rerank(text, ent_text, ent_type, candidates)
                
                    # Ưu tiên các mã ICD-10 trùng khớp hoàn toàn tên gọi lên đầu
                    exact_matches = []
                    other_candidates = []
                    for cand in candidates:
                        try:
                            conn = sqlite3.connect(self.clinical_validator.db_path)
                            cursor = conn.cursor()
                            cursor.execute("SELECT name_vi, name_en FROM icd10 WHERE code = ? LIMIT 1;", (cand,))
                            row = cursor.fetchone()
                            conn.close()
                            if row:
                                name_vi, name_en = row
                                name_vi_clean = name_vi.lower().strip() if name_vi else ""
                                name_en_clean = name_en.lower().strip() if name_en else ""
                                clean_ent_text = ent_text.lower().strip()
                                if name_vi_clean == clean_ent_text or name_en_clean == clean_ent_text:
                                    exact_matches.append(cand)
                                    continue
                        except Exception:
                            pass
                        other_candidates.append(cand)
                    candidates = exact_matches + other_candidates
                
                processed_ent["candidates"] = candidates[:5]
                
                # Áp dụng bộ lọc luật lâm sàng
                processed_ent = self.clinical_validator.check_and_fix_candidates(processed_ent, patient_info)
                
            elif ent_type == "THUỐC":
                # Chuẩn hóa văn bản thuốc
                clean_text = self.normalizer.clean_text(ent_text)
                
                # Bóc tách cách dùng và ký hiệu viết tắt Latin nhưng GIỮ LẠI liều lượng (mg, ml...)
                clean_drug_name = self.normalizer.remove_dosage(clean_text)
                expanded_text = self.normalizer.expand_abbreviation(clean_drug_name)
                
                # Tìm tên thuốc gốc (không có hàm lượng) để tra cứu hoạt chất chính
                drug_name_only = re.sub(r'\b\d+(?:[\.,]\d+)?\s*(?:mg/ml|mg|ml|g|mcg|ui|iu|MG|ML|G|MCG|UI|IU)\b', '', clean_drug_name).strip()
                drug_name_only = self.normalizer.clean_text(drug_name_only).lower()
                
                # 4a. Kiểm tra bảng ánh xạ cứng (Override) trước cho tên đầy đủ
                override_key = expanded_text.lower()
                if override_key in self.override_dict.get("THUỐC", {}):
                    candidates = self.override_dict["THUỐC"][override_key]
                    print(f"  [OVERRIDE MATCH] Mapped drug to {candidates}")
                else:
                    # Tra cứu Hybrid (BM25s + FAISS) tìm mã RxNorm bằng tên thuốc sạch có liều lượng
                    candidates = self.rxnorm_retriever.retrieve(expanded_text, top_k=5)
                    # Xếp hạng lại bằng LLM
                    if self.llm_reranker:
                        candidates = self.llm_reranker.rerank(text, ent_text, ent_type, candidates)
                
                    # Ánh xạ hoạt chất chính (IN/PIN) và mã lịch sử ở phía sau để tối đa hóa Recall
                    resolved_candidates = []
                    for cand in candidates:
                        resolved_candidates.append(cand)
                        
                        rxcui_name = self.clinical_validator.get_rxnorm_name(cand)
                        # Tìm hoạt chất của candidate
                        ingreds = self.clinical_validator.get_ingredients(rxcui_name)
                        for ing in ingreds:
                            resolved_candidates.append(ing)
                            
                    # Bổ sung hoạt chất chính từ tên thuốc không hàm lượng qua override dict
                    if drug_name_only in self.override_dict.get("THUỐC", {}):
                        for ing in self.override_dict["THUỐC"][drug_name_only]:
                            resolved_candidates.append(ing)
                            
                    # Loại bỏ trùng lặp và giữ thứ tự
                    seen = set()
                    unique_resolved = []
                    for c in resolved_candidates:
                        if c not in seen:
                            seen.add(c)
                            unique_resolved.append(c)
                            
                    candidates = unique_resolved
                        
                processed_ent["candidates"] = candidates[:5]
                
                # Áp dụng bộ lọc luật lâm sàng (bao gồm cả kiểm tra dạng bào chế)
                processed_ent = self.clinical_validator.check_and_fix_candidates(processed_ent, patient_info)
                
            processed_entities.append(processed_ent)
            
        # 5. Kiểm tra tính toàn vẹn của mã kép chẩn đoán
        processed_entities = self.clinical_validator.check_dual_codes(processed_entities)
        
        return processed_entities

    def run(self, input_dir=INPUT_DIR, output_dir=OUTPUT_DIR):
        """Chạy pipeline trên toàn bộ file txt trong input_dir."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        txt_files = glob.glob(os.path.join(input_dir, "*.txt"))
        print(f"Found {len(txt_files)} text files in input directory: {input_dir}")
        
        for file_path in txt_files:
            try:
                results = self.process_file(file_path)
                
                # Ghi kết quả JSON
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{base_name}.json")
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                    
                print(f"  - Saved prediction to {output_path}")
            except Exception as e:
                print(f"[ERROR] Failed to process {file_path}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AI Race Viettel Pipeline Runner")
    parser.add_argument("--input", type=str, default=INPUT_DIR, help="Input directory containing text files")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR, help="Output directory to save JSON results")
    args = parser.parse_args()
    
    pipeline = BaselinePipeline()
    pipeline.run(input_dir=args.input, output_dir=args.output)
