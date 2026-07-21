from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schema import EntityAnnotation
from .text import containing_section, detect_sections
from .vllm_compat import build_sampling_kwargs, iter_batches, parse_json_object


NEGATION_CUES = re.compile(r"(?iu)\b(?:không|chưa|không\s+có|không\s+ghi\s+nhận|âm\s+tính|phủ\s+nhận)\b")
HISTORICAL_CUES = re.compile(r"(?iu)\b(?:tiền\s+sử|trước\s+đây|đã\s+từng|mãn\s+tính)\b")
PLANNED_CUES = re.compile(r"(?iu)\b(?:sẽ|dự\s+kiến|lên\s+lịch|kế\s+hoạch|chỉ\s+định)\b")
RESOLVED_CUES = re.compile(r"(?iu)\b(?:đã\s+hết|không\s+còn|ổn\s+định|cải\s+thiện)\b")
UNCERTAINTY_CUES = re.compile(r"(?iu)\b(?:nghi|nghi\s+ngờ|có\s+thể|khả\s+năng|theo\s+dõi)\b")
FAMILY_CUES = re.compile(r"(?iu)\b(?:mẹ|cha|bố|anh|chị|em|ông|bà|gia\s+đình|người\s+nhà)\b")


@dataclass(slots=True)
class AssertionAxes:
    polarity: str = "AFFIRMED"
    temporality: str = "CURRENT"
    certainty: str = "CONFIRMED"
    experiencer: str = "PATIENT"

    def labels(self) -> list[str]:
        return [
            f"polarity:{self.polarity}",
            f"temporality:{self.temporality}",
            f"certainty:{self.certainty}",
            f"experiencer:{self.experiencer}",
        ]


class HybridAssertionPredictor:
    def __init__(self, context_window: int = 120) -> None:
        self.context_window = context_window

    def predict_axes(self, raw_text: str, entity: EntityAnnotation) -> AssertionAxes:
        sections = detect_sections(raw_text)
        section_name = containing_section(entity.position, sections)
        left = max(0, entity.start - self.context_window)
        right = min(len(raw_text), entity.end + self.context_window)
        context = raw_text[left:right]
        local_prefix = raw_text[max(0, entity.start - 55):entity.start]

        polarity = "NEGATED" if NEGATION_CUES.search(local_prefix) else "AFFIRMED"
        if PLANNED_CUES.search(context):
            temporality = "PLANNED"
        elif RESOLVED_CUES.search(context):
            temporality = "RESOLVED"
        elif section_name in {"HISTORY", "FAMILY_HISTORY"} or HISTORICAL_CUES.search(context):
            temporality = "HISTORICAL"
        else:
            temporality = "CURRENT"
        certainty = "POSSIBLE" if UNCERTAINTY_CUES.search(context) else "CONFIRMED"
        experiencer = "FAMILY" if section_name == "FAMILY_HISTORY" or FAMILY_CUES.search(local_prefix) else "PATIENT"
        return AssertionAxes(polarity, temporality, certainty, experiencer)

    def predict(self, raw_text: str, entities: Iterable[EntityAnnotation]) -> dict[tuple[int, int, str], AssertionAxes]:
        return {
            (entity.start, entity.end, entity.type): self.predict_axes(raw_text, entity)
            for entity in entities
        }

class ClinicalLLMAssertionPredictor:
    """Sử dụng LLM để dự đoán Assertion (Polarity, Temporality, Certainty, Experiencer)."""
    
    def __init__(self, llm_engine):
        self.llm = llm_engine
        
    def _build_prompt(self, context_text: str, entity_text: str) -> str:
        system_prompt = (
            "Bạn là chuyên gia phân tích hồ sơ bệnh án. "
            "Nhiệm vụ của bạn là trích xuất 4 thuộc tính (Assertion) cho thực thể được chỉ định dựa trên ngữ cảnh bệnh án.\n"
            "- Polarity: AFFIRMED (khẳng định) hoặc NEGATED (phủ định)\n"
            "- Temporality: CURRENT (hiện tại), HISTORICAL (tiền sử), PLANNED (dự kiến), RESOLVED (đã khỏi)\n"
            "- Certainty: CONFIRMED (chắc chắn), POSSIBLE (nghi ngờ/có thể)\n"
            "- Experiencer: PATIENT (bệnh nhân), FAMILY (người nhà)"
        )
        user_prompt = (
            f"Ngữ cảnh:\n\"\"\"{context_text}\"\"\"\n\n"
            f"Thực thể: [{entity_text}]\n\n"
            "Hãy trả về kết quả dưới định dạng JSON với 4 trường: polarity, temporality, certainty, experiencer."
        )
        return f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_prompt}<|im_end|>\n<|im_start|>assistant\n"
        
    def predict_batch(self, queries: list[dict], batch_size: int = 64) -> list[AssertionAxes]:
        if not self.llm or not queries:
            return []

        from vllm import SamplingParams

        schema = {
            "type": "object",
            "properties": {
                "polarity": {"enum": ["AFFIRMED", "NEGATED"]},
                "temporality": {"enum": ["CURRENT", "HISTORICAL", "PLANNED", "RESOLVED"]},
                "certainty": {"enum": ["CONFIRMED", "POSSIBLE"]},
                "experiencer": {"enum": ["PATIENT", "FAMILY"]},
            },
            "required": ["polarity", "temporality", "certainty", "experiencer"],
        }
        allowed = {
            "polarity": {"AFFIRMED", "NEGATED"},
            "temporality": {"CURRENT", "HISTORICAL", "PLANNED", "RESOLVED"},
            "certainty": {"CONFIRMED", "POSSIBLE"},
            "experiencer": {"PATIENT", "FAMILY"},
        }
        defaults = AssertionAxes()
        results: list[AssertionAxes] = []
        for query_batch in iter_batches(queries, batch_size):
            prompts = [self._build_prompt(query["context"], query["entity_text"]) for query in query_batch]
            sampling_params = SamplingParams(
                **build_sampling_kwargs(
                    SamplingParams,
                    schema,
                    temperature=0.0,
                    max_tokens=64,
                )
            )
            outputs = self.llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
            if len(outputs) != len(query_batch):
                raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(query_batch)} assertion prompts")
            for output in outputs:
                data = parse_json_object(output.outputs[0].text) or {}
                values = {
                    field: data.get(field) if data.get(field) in choices else getattr(defaults, field)
                    for field, choices in allowed.items()
                }
                results.append(AssertionAxes(**values))
        return results
