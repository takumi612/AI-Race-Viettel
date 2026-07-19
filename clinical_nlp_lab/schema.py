from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


SUBMISSION_KEYS = {"text", "type", "candidates", "assertions", "position"}


@dataclass(slots=True)
class ClinicalDocument:
    document_id: str
    raw_text: str
    entities: list["EntityAnnotation"] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class EntityAnnotation:
    text: str
    type: str
    position: tuple[int, int]
    candidates: list[str] = field(default_factory=list)
    assertions: list[str] = field(default_factory=list)
    confidence: float = 1.0
    mention_head: str | None = None
    evidence: list[str] = field(default_factory=list)

    @property
    def start(self) -> int:
        return self.position[0]

    @property
    def end(self) -> int:
        return self.position[1]

    def validate_offset(self, raw_text: str) -> None:
        start, end = self.position
        if not (0 <= start <= end <= len(raw_text)):
            raise ValueError(f"Invalid position {self.position} for document length {len(raw_text)}")
        actual = raw_text[start:end]
        if actual != self.text:
            raise ValueError(
                f"Offset mismatch at {self.position}: expected {self.text!r}, actual {actual!r}"
            )

    def to_submission(self, official_type: str, official_assertions: Iterable[str]) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": official_type,
            "candidates": list(dict.fromkeys(str(item) for item in self.candidates)),
            "assertions": list(dict.fromkeys(str(item) for item in official_assertions)),
            "position": [int(self.start), int(self.end)],
        }

    def to_diagnostic(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["position"] = [self.start, self.end]
        return payload


@dataclass(slots=True)
class SectionSpan:
    section_name: str
    start: int
    end: int
    text: str

    def validate(self, raw_text: str) -> None:
        if raw_text[self.start:self.end] != self.text:
            raise ValueError(f"Section offset mismatch: {self.section_name}")


def parse_entity(payload: dict[str, Any], raw_text: str) -> EntityAnnotation:
    required = {"text", "type", "position"}
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Entity is missing required keys: {sorted(missing)}")
    position = payload["position"]
    if not isinstance(position, list | tuple) or len(position) != 2:
        raise ValueError("position must be [start, end]")
    entity = EntityAnnotation(
        text=str(payload["text"]),
        type=str(payload["type"]),
        candidates=[str(value) for value in payload.get("candidates", [])],
        assertions=[str(value) for value in payload.get("assertions", [])],
        position=(int(position[0]), int(position[1])),
    )
    entity.validate_offset(raw_text)
    return entity


def validate_submission_payload(payload: Any, raw_text: str) -> list[str]:
    errors: list[str] = []
    if not isinstance(payload, list):
        return ["Top-level submission JSON must be an array"]
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            errors.append(f"Entity {index} is not an object")
            continue
        keys = set(item)
        if keys != SUBMISSION_KEYS:
            errors.append(
                f"Entity {index} has invalid keys: missing={sorted(SUBMISSION_KEYS - keys)}, extra={sorted(keys - SUBMISSION_KEYS)}"
            )
        try:
            entity = parse_entity(item, raw_text)
            if not isinstance(item.get("candidates"), list):
                errors.append(f"Entity {index} candidates must be a list")
            if not isinstance(item.get("assertions"), list):
                errors.append(f"Entity {index} assertions must be a list")
            entity.validate_offset(raw_text)
        except (TypeError, ValueError) as exc:
            errors.append(f"Entity {index}: {exc}")
    return errors


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return destination

