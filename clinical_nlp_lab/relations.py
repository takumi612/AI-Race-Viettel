from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Iterable

from .schema import EntityAnnotation
from .text import split_sentences


COMPATIBLE_RELATIONS: dict[tuple[str, str], str] = {
    ("DRUG", "DISEASE"): "DRUG_TREATS_CONDITION",
    ("DRUG", "SYMPTOM"): "DRUG_TREATS_CONDITION",
    ("DISEASE", "SYMPTOM"): "CONDITION_HAS_SYMPTOM",
    ("LAB_RESULT", "DISEASE"): "LAB_ASSOCIATED_WITH_CONDITION",
    ("LAB_RESULT", "SYMPTOM"): "LAB_ASSOCIATED_WITH_SYMPTOM",
}


@dataclass(slots=True)
class RelationPrediction:
    subject: tuple[int, int, str]
    object: tuple[int, int, str]
    relation: str
    confidence: float
    evidence: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["subject"] = list(self.subject)
        payload["object"] = list(self.object)
        return payload


class RuleRelationExtractor:
    def __init__(self, max_distance: int = 256) -> None:
        self.max_distance = max_distance

    @staticmethod
    def _sentence_index(raw_text: str, entity: EntityAnnotation) -> int | None:
        for index, (start, end, _text) in enumerate(split_sentences(raw_text)):
            if start <= entity.start and entity.end <= end:
                return index
        return None

    def extract(self, raw_text: str, entities: Iterable[EntityAnnotation]) -> list[RelationPrediction]:
        entity_list = list(entities)
        sentence_indices = {
            (entity.start, entity.end, entity.type): self._sentence_index(raw_text, entity)
            for entity in entity_list
        }
        relations: list[RelationPrediction] = []
        for subject in entity_list:
            for obj in entity_list:
                if subject is obj:
                    continue
                relation = COMPATIBLE_RELATIONS.get((subject.type, obj.type))
                if relation is None:
                    continue
                distance = max(0, obj.start - subject.end, subject.start - obj.end)
                if distance > self.max_distance:
                    continue
                subject_key = (subject.start, subject.end, subject.type)
                object_key = (obj.start, obj.end, obj.type)
                if sentence_indices[subject_key] is None or sentence_indices[subject_key] != sentence_indices[object_key]:
                    continue
                confidence = max(0.50, 0.88 - distance / max(self.max_distance, 1) * 0.25)
                relations.append(
                    RelationPrediction(
                        subject=subject_key,
                        object=object_key,
                        relation=relation,
                        confidence=round(confidence, 6),
                        evidence="same_sentence+type_compatibility+distance",
                    )
                )
        relations.sort(key=lambda item: (item.subject, item.object, item.relation))
        return relations

