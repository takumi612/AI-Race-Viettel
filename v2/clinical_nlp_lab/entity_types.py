"""Canonical entity-type IDs shared by training artifacts and runtime."""

from __future__ import annotations


ENTITY_TYPE_TO_ID: dict[str, int] = {
    "DISEASE": 0,
    "DRUG": 1,
    "SYMPTOM": 2,
    "LAB_NAME": 3,
    "LAB_RESULT": 4,
}
ID_TO_ENTITY_TYPE: dict[int, str] = {
    value: key for key, value in ENTITY_TYPE_TO_ID.items()
}
ASSERTION_ENTITY_TYPES = frozenset({"DISEASE", "DRUG", "SYMPTOM"})
LAB_ENTITY_TYPES = frozenset({"LAB_NAME", "LAB_RESULT"})


def entity_type_id(entity_type: str) -> int:
    try:
        return ENTITY_TYPE_TO_ID[entity_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported entity type: {entity_type!r}") from exc
