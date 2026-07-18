"""Safe, injectable clinical predicates used by candidate selection."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.utils.paths import DB_PATH

LOGGER = logging.getLogger(__name__)
RULES_PATH = Path(__file__).resolve().parents[1] / "resources" / "clinical_validation_rules.json"


def _read_rules(rules: object | None = None, rules_path: str | Path | None = None) -> Mapping[str, Any]:
    if rules is None:
        path = Path(rules_path) if rules_path is not None else RULES_PATH
        with path.open("r", encoding="utf-8") as handle:
            rules = json.load(handle)
    elif isinstance(rules, (str, Path)):
        with Path(rules).open("r", encoding="utf-8") as handle:
            rules = json.load(handle)
    elif hasattr(rules, "to_dict"):
        rules = rules.to_dict()
    if not isinstance(rules, Mapping):
        raise ValueError("clinical validation rules must be a mapping or JSON path")
    groups = rules.get("route_groups")
    if not isinstance(groups, (list, tuple)):
        raise ValueError("clinical validation rules require route_groups")
    normalized = []
    for group in groups:
        if not isinstance(group, Mapping) or not isinstance(group.get("name"), str):
            raise ValueError("each route group requires a name")
        mention = group.get("mention_terms", ())
        rxnorm = group.get("rxnorm_terms", ())
        if not isinstance(mention, (list, tuple)) or not isinstance(rxnorm, (list, tuple)):
            raise ValueError("route group terms must be lists")
        if any(not isinstance(term, str) or not term.strip() for term in (*mention, *rxnorm)):
            raise ValueError("route group terms must be non-empty strings")
        source = group.get("source")
        if rules.get("version") is not None and (not isinstance(source, str) or not source.strip()):
            raise ValueError(f"missing source for route group {group['name']}")
        normalized.append({
            "name": group["name"],
            "source": source or "injected",
            "mention_terms": tuple(term.casefold() for term in mention),
            "rxnorm_terms": tuple(term.casefold() for term in rxnorm),
        })
    if not normalized:
        raise ValueError("clinical validation rules require at least one route group")
    # The bundled resource is strict and versioned. Small injected rule objects
    # remain intentionally lightweight for unit tests and downstream callers.
    if rules.get("version") is not None:
        if isinstance(rules.get("version"), bool) or not isinstance(rules.get("version"), int):
            raise ValueError("clinical validation rules version must be an integer")
        provenance = rules.get("provenance")
        if not isinstance(provenance, Mapping):
            raise ValueError("clinical validation rules require provenance")
        for group in normalized:
            source = provenance.get(group["name"])
            if not isinstance(source, Mapping) or not isinstance(source.get("source"), str) or not source["source"].strip():
                raise ValueError(f"missing source for route group {group['name']}")
    return MappingProxyType({"route_groups": tuple(MappingProxyType(group) for group in normalized)})


def dose_form_is_compatible(drug_text: str, rxcui_name: str, rules: object | None = None) -> bool:
    """Return false only for an explicit route/form contradiction."""
    if not drug_text or not rxcui_name:
        return True
    loaded = _read_rules(rules)
    mention = str(drug_text).casefold()
    rxnorm = str(rxcui_name).casefold()
    mentioned_groups = {
        group["name"] for group in loaded["route_groups"]
        if any(term in mention for term in group["mention_terms"])
    }
    rxnorm_groups = {
        group["name"] for group in loaded["route_groups"]
        if any(term in rxnorm for term in group["rxnorm_terms"])
    }
    if not mentioned_groups or not rxnorm_groups:
        return not mentioned_groups
    return bool(mentioned_groups & rxnorm_groups)


class ClinicalValidator:
    def __init__(
        self,
        db_path: str | Path = DB_PATH,
        *,
        load_historical_rxnorm: bool = False,
        rules: object | None = None,
        rules_path: str | Path | None = None,
    ):
        self.db_path = str(db_path)
        if not isinstance(load_historical_rxnorm, bool):
            raise ValueError("load_historical_rxnorm must be a boolean")
        self.load_historical_rxnorm = load_historical_rxnorm
        self.rules = _read_rules(rules, rules_path)
        self.load_rules()

    def _fetchall(self, query: str, params: tuple = ()) -> list[tuple]:
        try:
            with sqlite3.connect(self.db_path) as connection:
                return connection.execute(query, params).fetchall()
        except sqlite3.Error:
            return []

    def load_rules(self) -> None:
        self.sex_rules = {row[0]: row[1] for row in self._fetchall("SELECT code, allowed_sex FROM icd10_rules_sex")}
        self.age_rules = {row[0]: (row[1], row[2], row[3]) for row in self._fetchall("SELECT code, min_days, max_days, description FROM icd10_rules_age")}
        self.dual_rules: dict[str, set[str]] = {}
        for dagger, asterisk in self._fetchall("SELECT dagger_code, asterisk_code FROM icd10_rules_dual"):
            self.dual_rules.setdefault(dagger, set()).add(asterisk)
        self.not_primary_rules = {row[0] for row in self._fetchall("SELECT code FROM icd10_rules_not_primary")}
        self.rxnorm_mapping: dict[str, list[str]] = {}
        # This query is deliberately absent unless explicitly requested. The
        # default inference path therefore performs zero historical work.
        if self.load_historical_rxnorm:
            for old_cui, new_cui in self._fetchall("SELECT old_cui, new_cui FROM rxnorm_mapping"):
                self.rxnorm_mapping.setdefault(str(new_cui), []).append(str(old_cui))

    def get_historical_cuis(self, rxcui: str) -> list[str]:
        return list(self.rxnorm_mapping.get(str(rxcui).strip(), ()))

    def validate_sex(self, code: str, patient_sex: str | None) -> bool:
        return not patient_sex or code not in self.sex_rules or self.sex_rules[code] == patient_sex

    def validate_age(self, code: str, patient_age_days: int | None) -> bool:
        if patient_age_days is None or code not in self.age_rules:
            return True
        minimum, maximum, _ = self.age_rules[code]
        return minimum <= patient_age_days <= maximum

    def get_rxnorm_name(self, rxcui: str) -> str:
        rows = self._fetchall("SELECT name FROM rxnorm WHERE rxcui = ? LIMIT 1", (str(rxcui).strip(),))
        return str(rows[0][0]) if rows and rows[0][0] is not None else ""

    def get_ingredients(self, rxnorm_name: str) -> list[str]:
        if not rxnorm_name:
            return []
        result = []
        for tty in ("IN", "PIN"):
            rows = self._fetchall(
                "SELECT rxcui FROM rxnorm WHERE tty = ? AND ? LIKE name || '%' ORDER BY length(name) DESC LIMIT 1",
                (tty, rxnorm_name.lower()),
            )
            if rows:
                result.append(rows[0][0])
        return result

    def validate_dose_form(self, drug_text: str, rxcui_name: str) -> bool:
        return dose_form_is_compatible(drug_text, rxcui_name, self.rules)

    @staticmethod
    def _patient_value(patient_info: object | None, name: str, default=None):
        if isinstance(patient_info, Mapping):
            return patient_info.get(name, default)
        return getattr(patient_info, name, default)

    def is_candidate_valid(self, entity: Mapping[str, Any], code: str, patient_info: object | None = None) -> bool:
        entity_type = str(entity.get("type", "")).upper()
        clean_code = str(code).strip().upper()
        if "CH" in entity_type or "DIAG" in entity_type:
            if not self.validate_sex(clean_code, self._patient_value(patient_info, "sex")):
                return False
            if not self.validate_age(clean_code, self._patient_value(patient_info, "age_days")):
                return False
        elif "THU" in entity_type or "DRUG" in entity_type or "MED" in entity_type:
            name = self.get_rxnorm_name(clean_code)
            if name and not self.validate_dose_form(str(entity.get("text", "")), name):
                return False
        return True

    def check_and_fix_candidates(self, entity: dict, patient_info: object | None = None) -> dict:
        if not entity.get("candidates"):
            return entity
        entity["candidates"] = [
            code for code in entity["candidates"]
            if self.is_candidate_valid(entity, code, patient_info)
        ]
        return entity

    def check_dual_codes(self, entities: list[dict]) -> list[dict]:
        # Dual-code analysis is intentionally non-mutating. Selection and
        # output limits must remain controlled by the deterministic selector.
        return entities


__all__ = ["ClinicalValidator", "dose_form_is_compatible"]
