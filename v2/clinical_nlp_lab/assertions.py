from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .schema import EntityAnnotation
from .text import containing_section, detect_sections


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
