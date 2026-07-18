"""Strict loader for provenance-bearing clinical lexicon resources."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import unicodedata


ENTITY_TYPES = frozenset(
    {
        "CHẨN_ĐOÁN",
        "TRIỆU_CHỨNG",
        "THUỐC",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
    }
)
STATUSES = frozenset({"verified", "unverified"})


def normalize_term(value: str) -> str:
    return unicodedata.normalize("NFC", " ".join(value.split())).casefold()


@dataclass(frozen=True)
class ClinicalTerm:
    term: str
    normalized: str
    entity_type: str
    source: str
    status: str


class ClinicalLexicon:
    """Load and validate immutable clinical terms."""

    @staticmethod
    def load(path: str | Path) -> tuple[ClinicalTerm, ...]:
        resource_path = Path(path)
        try:
            payload = json.loads(resource_path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise ValueError(f"clinical lexicon does not exist: {resource_path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"clinical lexicon is invalid JSON: {resource_path}: {error.msg}"
            ) from error
        if not isinstance(payload, dict):
            raise ValueError("clinical lexicon must be a JSON object")
        unknown = set(payload) - {"schema_version", "entries"}
        if unknown:
            raise ValueError(f"clinical lexicon has unknown keys: {', '.join(sorted(unknown))}")
        version = payload.get("schema_version")
        if isinstance(version, bool) or version != 1:
            raise ValueError("clinical lexicon schema_version must be integer 1")
        entries = payload.get("entries")
        if not isinstance(entries, list):
            raise ValueError("clinical lexicon entries must be a list")

        selected: dict[tuple[str, str], ClinicalTerm] = {}
        for index, raw in enumerate(entries):
            location = f"entries[{index}]"
            if not isinstance(raw, dict):
                raise ValueError(f"clinical lexicon {location} must be an object")
            unknown = set(raw) - {"term", "type", "source", "status"}
            if unknown:
                raise ValueError(
                    f"clinical lexicon {location} has unknown keys: {', '.join(sorted(unknown))}"
                )
            for field in ("term", "type", "source", "status"):
                if field not in raw:
                    raise ValueError(f"clinical lexicon {location}.{field} is required")
            term = raw["term"]
            entity_type = raw["type"]
            source = raw["source"]
            status = raw["status"]
            if not isinstance(term, str) or not term.strip():
                raise ValueError(f"clinical lexicon {location}.term must be a non-empty string")
            if not isinstance(entity_type, str) or entity_type not in ENTITY_TYPES:
                raise ValueError(f"clinical lexicon {location}.type is invalid")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"clinical lexicon {location}.source must be a non-empty string")
            if not isinstance(status, str) or status not in STATUSES:
                raise ValueError(f"clinical lexicon {location}.status is invalid")
            display_term = " ".join(term.split())
            candidate = ClinicalTerm(
                display_term,
                normalize_term(display_term),
                entity_type,
                source.strip(),
                status,
            )
            key = (candidate.normalized, candidate.entity_type)
            current = selected.get(key)
            rank = (candidate.status == "verified", candidate.source, candidate.term)
            current_rank = (
                (current.status == "verified", current.source, current.term)
                if current is not None
                else None
            )
            if current is None or rank > current_rank:
                selected[key] = candidate
        return tuple(selected[key] for key in sorted(selected))
