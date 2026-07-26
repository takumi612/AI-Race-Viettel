from __future__ import annotations

from clinical_nlp_lab.runtime_bundle import KBFirstRecovery


def test_short_kb_alias_is_too_ambiguous_for_entity_recovery():
    recovery = KBFirstRecovery(
        [{"candidate_id": "R05", "aliases": ["ho"]}],
        [],
    )
    raw_text = "Bệnh nhân ho, không phải khoa học hoặc hóa chất."

    proposals = recovery.scan_raw_text(raw_text)

    assert proposals == []
