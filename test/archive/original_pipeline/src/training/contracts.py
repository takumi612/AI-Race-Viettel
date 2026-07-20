"""Immutable canonical records shared by all training-data projections."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Iterable, Mapping


ALLOWED_ENTITY_TYPES = frozenset(
    {
        "CHẨN_ĐOÁN",
        "THUỐC",
        "TRIỆU_CHỨNG",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
    }
)
ALLOWED_ASSERTIONS = frozenset({"isNegated", "isHistorical", "isFamily"})


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, slots=True)
class CanonicalEntity:
    """One entity whose offsets are half-open Unicode string indices."""

    text: str
    entity_type: str
    start: int
    end: int
    assertions: tuple[str, ...] = ()
    codes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, Any],
        source_text: str,
    ) -> "CanonicalEntity":
        if not isinstance(mapping, Mapping):
            raise ValueError("entity must be a mapping")
        _required_text(source_text, "source_text")

        text = _required_text(mapping.get("text"), "entity text")
        entity_type = _required_text(mapping.get("type"), "entity type")
        if entity_type not in ALLOWED_ENTITY_TYPES:
            raise ValueError(f"unsupported entity type: {entity_type}")

        position = mapping.get("position")
        if (
            not isinstance(position, (list, tuple))
            or len(position) != 2
            or any(not isinstance(offset, int) or isinstance(offset, bool) for offset in position)
        ):
            raise ValueError("position must contain exactly two integer offsets")
        start, end = position
        if start < 0 or end <= start or end > len(source_text):
            raise ValueError("position is outside source text")
        if source_text[start:end] != text:
            raise ValueError("span text mismatch")

        raw_assertions = mapping.get("assertions", ())
        if not isinstance(raw_assertions, (list, tuple)):
            raise ValueError("invalid assertions: expected a list")
        invalid_assertions = {
            assertion
            for assertion in raw_assertions
            if not isinstance(assertion, str) or assertion not in ALLOWED_ASSERTIONS
        }
        if invalid_assertions:
            raise ValueError(f"invalid assertions: {sorted(map(str, invalid_assertions))}")
        assertions = tuple(sorted(set(raw_assertions)))

        raw_codes = mapping.get("candidates", ())
        if not isinstance(raw_codes, (list, tuple)):
            raise ValueError("invalid candidates: expected a list")
        normalized_codes: list[str] = []
        seen_codes: set[str] = set()
        for code in raw_codes:
            if isinstance(code, bool) or not isinstance(code, (str, int)):
                raise ValueError(f"invalid candidates: {code!r}")
            normalized = str(code).strip()
            if not normalized:
                raise ValueError("invalid candidates: blank code")
            if normalized not in seen_codes:
                seen_codes.add(normalized)
                normalized_codes.append(normalized)

        return cls(
            text=text,
            entity_type=entity_type,
            start=start,
            end=end,
            assertions=assertions,
            codes=tuple(normalized_codes),
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "type": self.entity_type,
            "position": [self.start, self.end],
            "assertions": list(self.assertions),
            "candidates": list(self.codes),
        }


@dataclass(frozen=True, slots=True)
class CanonicalRecord:
    """Validated source record with immutable metadata and content fingerprint."""

    record_id: str
    source: str
    trust_tier: str
    text: str
    entities: tuple[CanonicalEntity, ...]
    sha256: str
    split_group: str
    metadata: Mapping[str, Any]

    @classmethod
    def create(
        cls,
        *,
        record_id: str,
        source: str,
        trust_tier: str,
        text: str,
        entity_mappings: Iterable[Mapping[str, Any]],
        split_group: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "CanonicalRecord":
        record_id = _required_text(record_id, "record_id")
        source = _required_text(source, "source")
        trust_tier = _required_text(trust_tier, "trust_tier")
        text = _required_text(text, "text")
        if entity_mappings is None:
            raise ValueError("entity_mappings must be iterable")

        entities = tuple(
            CanonicalEntity.from_mapping(entity_mapping, text)
            for entity_mapping in entity_mappings
        )
        normalized_group = record_id if split_group is None else _required_text(
            split_group, "split_group"
        )
        normalized_metadata = MappingProxyType(dict(metadata or {}))

        return cls(
            record_id=record_id,
            source=source,
            trust_tier=trust_tier,
            text=text,
            entities=entities,
            sha256=sha256(text.encode("utf-8")).hexdigest(),
            split_group=normalized_group,
            metadata=normalized_metadata,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "source": self.source,
            "trust_tier": self.trust_tier,
            "text": self.text,
            "entities": [entity.to_mapping() for entity in self.entities],
            "sha256": self.sha256,
            "split_group": self.split_group,
            "metadata": dict(self.metadata),
        }
