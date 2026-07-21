from __future__ import annotations

import json
import logging
from typing import Any, Iterable

# Import vLLM nếu có
try:
    from vllm import LLM, SamplingParams
except ImportError:
    LLM = None
    SamplingParams = None


class ClinicalLLMReranker:
    """
    Sử dụng vLLM (Qwen2.5-7B) và Constrained Decoding để chọn mã duy nhất từ danh sách ứng viên (Rerank).
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len: int = 4096):
        self.model_name = model_name
        self.llm = None
        
        logging.info(f"Khởi tạo LLM Reranker với model: {model_name}")
        if LLM is None:
            raise ImportError("Vui lòng cài đặt thư viện `vllm` để sử dụng LLM Reranker.")
            
        # Khởi tạo vLLM engine
        self.llm = LLM(
            model=self.model_name,
            trust_remote_code=True,
            quantization="awq" if "AWQ" in model_name else None,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.5, # Giảm xuống 50% để nhường VRAM còn thừa của PyTorch
            enforce_eager=True # Giảm VRAM overhead
        )

    def _build_prompt(self, context_text: str, entity_text: str, entity_type: str, candidates: list[dict[str, Any]]) -> str:
        """
        Xây dựng prompt đưa vào model.
        """
        # Tạo danh sách enum options cho LLM
        options_text = ""
        for i, cand in enumerate(candidates):
            options_text += f"- ID: {cand['candidate_id']} | Tên: {cand['name']}\n"
            
        system_prompt = (
            "Bạn là một chuyên gia mã hóa y khoa (Medical Coder) chuyên nghiệp. "
            "Nhiệm vụ của bạn là xem xét ngữ cảnh của bệnh án và chọn ra mã chẩn đoán (ICD-10) hoặc mã thuốc (RxNorm) chính xác nhất cho thực thể được yêu cầu."
        )
        
        user_prompt = (
            f"Ngữ cảnh bệnh án:\n\"\"\"{context_text}\"\"\"\n\n"
            f"Thực thể cần mã hóa: [{entity_text}] (Loại: {entity_type})\n\n"
            f"Danh sách các mã ứng viên:\n{options_text}\n"
            "Hãy phân tích ngữ cảnh và trả về JSON chứa trường `selected_id` khớp với ID chính xác nhất trong danh sách ứng viên. "
            "Nếu không có mã nào phù hợp, trả về `selected_id`: null."
        )
        
        # Format theo ChatML (Qwen)
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"

    def _build_json_schema(self, candidates: list[dict[str, Any]]) -> str:
        """
        Sinh ra JSON Schema động giới hạn kết quả trả về chỉ nằm trong các candidate ID (Constrained Decoding).
        """
        valid_ids = [c["candidate_id"] for c in candidates]
        valid_ids.append(None) # Cho phép trả về null nếu không khớp
        
        schema = {
            "type": "object",
            "properties": {
                "selected_id": {
                    "enum": valid_ids
                }
            },
            "required": ["selected_id"]
        }
        return json.dumps(schema)

    def rerank_batch(self, entity_queries: list[dict[str, Any]]) -> list[str | None]:
        """
        Chạy batch rerank trên danh sách các truy vấn.
        Mỗi query trong `entity_queries` cần chứa:
        - `context_text`: Đoạn văn bản chứa thực thể
        - `entity_text`: Tên thực thể
        - `entity_type`: Loại (DISEASE/DRUG)
        - `candidates`: Danh sách trả về từ Hybrid Retrieval
        """
        if not self.llm:
            raise RuntimeError("LLM chưa được khởi tạo.")
            
        prompts = []
        sampling_params_list = []
        
        for query in entity_queries:
            prompt = self._build_prompt(
                query["context_text"], 
                query["entity_text"], 
                query["entity_type"], 
                query["candidates"]
            )
            prompts.append(prompt)
            
            # vLLM hỗ trợ truyền guided_json schema động (sử dụng outlines backend)
            schema_str = self._build_json_schema(query["candidates"])
            sp = SamplingParams(
                temperature=0.0, # Greedy search để ổn định
                max_tokens=100,
                guided_json=schema_str
            )
            sampling_params_list.append(sp)

        # Chạy suy luận batch (rất nhanh)
        logging.info(f"Đang chạy vLLM Rerank cho {len(prompts)} thực thể...")
        outputs = self.llm.generate(prompts, sampling_params=sampling_params_list, use_tqdm=True)
        
        results = []
        for output in outputs:
            generated_text = output.outputs[0].text
            try:
                data = json.loads(generated_text)
                results.append(data.get("selected_id"))
            except json.JSONDecodeError:
                logging.error(f"Lỗi parse JSON từ LLM: {generated_text}")
                results.append(None)
                
        return results

    def destroy(self):
        """
        Hủy engine vLLM để giải phóng VRAM.
        """
        import gc
        import torch
        import vllm.distributed.parallel_state as parallel_state
        
        if parallel_state.is_initialized():
            parallel_state.destroy_model_parallel()
            
        del self.llm
        self.llm = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Đã dọn dẹp xong VRAM của LLM.")
