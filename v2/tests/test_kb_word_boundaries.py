from __future__ import annotations

from clinical_nlp_lab.runtime_bundle import KBFirstRecovery


def test_short_kb_alias_matches_only_as_a_standalone_token():
    recovery = KBFirstRecovery(
        [{"candidate_id": "R05", "aliases": ["ho"]}],
        [],
    )
    raw_text = "Bệnh nhân ho, không phải khoa học hoặc hóa chất."

    proposals = recovery.scan_raw_text(raw_text)

    assert [(item.text, item.start, item.end) for item in proposals] == [
        ("ho", raw_text.index("ho"), raw_text.index("ho") + 2)
    ]
