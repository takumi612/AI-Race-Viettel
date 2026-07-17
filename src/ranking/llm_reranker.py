import os
import json
import logging
import sqlite3
from src.utils.paths import KB_DIR

# Cấu hình log
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LLMReranker")

class LLMReranker:
    def __init__(self, use_llm: bool = False, api_url: str = "http://localhost:8000/v1", api_key: str = "token-viettel-race"):
        """
        Khởi tạo bộ Reranker sử dụng LLM.
        
        Args:
            use_llm: Bật/tắt sử dụng LLM. Nếu False, sẽ chạy ở chế độ Fallback (trả về ứng viên Top-1).
            api_url: URL máy chủ API tương thích OpenAI (vLLM, llama.cpp server, hoặc Ollama).
            api_key: Khóa API xác thực nếu có.
        """
        self.use_llm = use_llm
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        
        # Kết nối tới CSDL tri thức để tra cứu mô tả của code
        self.db_path = os.path.join(KB_DIR, "metadata.db")
        
        if self.use_llm:
            logger.info(f"LLM Reranker is ENABLED. Target API: {self.api_url}")
        else:
            logger.info("LLM Reranker is DISABLED. Operating in Fallback/Pass-through mode.")

    def _get_candidate_descriptions(self, table_name: str, codes: list) -> dict:
        """Tra cứu tên mô tả của danh sách mã ứng viên từ database."""
        descriptions = {}
        if not codes:
            return descriptions
            
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            if table_name.lower() == "icd10":
                # ICD-10 sử dụng bảng icd10, các trường code, name_vi, name_en
                placeholders = ",".join(["?"] * len(codes))
                query = f"SELECT code, name_vi, name_en FROM icd10 WHERE code IN ({placeholders})"
                cursor.execute(query, codes)
                for row in cursor.fetchall():
                    code, name_vi, name_en = row
                    descriptions[code] = f"{name_vi} ({name_en})" if name_en else name_vi
            elif table_name.lower() == "rxnorm":
                # RxNorm sử dụng bảng rxnorm, các trường rxcui, name
                placeholders = ",".join(["?"] * len(codes))
                query = f"SELECT rxcui, name FROM rxnorm WHERE rxcui IN ({placeholders})"
                cursor.execute(query, codes)
                for row in cursor.fetchall():
                    rxcui, name = row
                    descriptions[rxcui] = name
                    
            conn.close()
        except Exception as e:
            logger.error(f"Lỗi tra cứu mô tả ứng viên từ DB: {e}")
            
        # Đảm bảo mọi code đầu vào đều có mô tả tối thiểu
        for code in codes:
            if code not in descriptions:
                descriptions[code] = "Không có mô tả chi tiết trong cơ sở dữ liệu."
                
        return descriptions

    def rerank(self, text_context: str, entity_text: str, entity_type: str, candidates: list) -> list:
        """
        Sử dụng LLM để sắp xếp và chọn mã tốt nhất từ danh sách ứng viên dựa trên ngữ cảnh y khoa.
        
        Args:
            text_context: Toàn bộ văn bản bệnh án lâm sàng chứa thực thể.
            entity_text: Tên của thực thể y khoa trích xuất được (ví dụ: 'tăng huyết áp').
            entity_type: Loại thực thể ('CHẨN_ĐOÁN' hoặc 'THUỐC').
            candidates: Danh sách các mã ứng viên thô do Retrieval tìm ra (ví dụ: ['I10', 'I11']).
            
        Returns:
            list: Danh sách ứng viên đã được sắp xếp lại hoặc lọc, mã tốt nhất đứng đầu.
        """
        if not candidates:
            return []
            
        # Chế độ Fallback / Tắt LLM: Giữ nguyên thứ tự của Retrieval
        if not self.use_llm:
            return candidates

        # Lấy mô tả chi tiết của từng mã ứng viên để làm ngữ cảnh cho LLM
        table_name = "icd10" if entity_type == "CHẨN_ĐOÁN" else "rxnorm"
        descriptions = self._get_candidate_descriptions(table_name, candidates)
        
        # Tạo danh sách mô tả định dạng đẹp cho Prompt
        candidates_context = ""
        for idx, code in enumerate(candidates):
            candidates_context += f"{idx + 1}. Mã: `{code}` - Mô tả: {descriptions.get(code, '')}\n"

        # Thiết lập Prompt y khoa tối ưu
        system_prompt = (
            "Bạn là một chuyên gia mã hóa dữ liệu lâm sàng y tế chuyên nghiệp. "
            "Nhiệm vụ của bạn là phân tích ngữ cảnh bệnh án lâm sàng và lựa chọn mã định danh y khoa chính xác nhất "
            "cho thực thể được yêu cầu từ một danh sách các ứng viên cho sẵn."
        )
        
        user_prompt = (
            f"=== NGỮ CẢNH BỆNH ÁN ===\n"
            f"\"\"\"\n{text_context.strip()}\n\"\"\"\n\n"
            f"=== THỰC THỂ CẦN MÃ HÓA ===\n"
            f"- Từ trích xuất: \"{entity_text}\"\n"
            f"- Loại thực thể: {entity_type} (Mã chuẩn tương ứng là {table_name.upper()})\n\n"
            f"=== DANH SÁCH ỨNG VIÊN TIỀN NĂNG ===\n"
            f"{candidates_context}\n"
            f"Dựa trên thông tin bệnh án và danh sách ứng viên, hãy chọn ra mã chính xác nhất.\n"
            f"YÊU CẦU ĐẦU RA:\n"
            f"- Bạn bắt buộc chỉ được chọn duy nhất 1 mã nằm trong danh sách ứng viên trên.\n"
            f"- Trả về kết quả dưới định dạng JSON duy nhất như sau, tuyệt đối không giải thích gì thêm:\n"
            f"{{\n"
            f"  \"best_code\": \"<mã_đã_chọn>\"\n"
            f"}}"
        )

        try:
            # Gửi yêu cầu HTTP đến máy chủ tương thích OpenAI
            import requests
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            payload = {
                "model": "Qwen2.5-7B-Instruct",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,  # Nhiệt độ thấp để đảm bảo tính ổn định và deterministic
                "max_tokens": 50,
                # Ép kiểu output JSON nếu API server hỗ trợ (như vLLM/Ollama)
                "response_format": {"type": "json_object"}
            }
            
            response = requests.post(f"{self.api_url}/chat/completions", headers=headers, json=payload, timeout=5)
            
            if response.status_code == 200:
                result_json = response.json()
                content = result_json["choices"][0]["message"]["content"].strip()
                data = json.loads(content)
                best_code = data.get("best_code")
                
                # Xác thực: Mã được chọn phải nằm trong danh sách candidates ban đầu
                if best_code in candidates:
                    logger.info(f"LLM đã chọn mã thành công: {best_code} cho thực thể '{entity_text}'")
                    # Đưa mã tốt nhất lên đầu danh sách candidates
                    new_candidates = [best_code] + [c for c in candidates if c != best_code]
                    return new_candidates
                else:
                    logger.warning(f"LLM chọn mã '{best_code}' không thuộc danh sách ứng viên. Kịch hoạt Fallback.")
            else:
                logger.warning(f"Máy chủ API trả về mã lỗi HTTP {response.status_code}. Kịch hoạt Fallback.")
                
        except Exception as e:
            logger.warning(f"Gặp lỗi khi gọi máy chủ LLM ({e}). Kịch hoạt chế độ phòng vệ Fallback.")
            
        # Chế độ phòng vệ Fallback: Nếu có bất kỳ lỗi nào, trả về danh sách candidates ban đầu (giữ nguyên Top-1 của Retrieval)
        return candidates
