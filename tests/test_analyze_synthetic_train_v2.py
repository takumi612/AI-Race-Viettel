from __future__ import annotations

from pathlib import Path

from scripts.analyze_synthetic_train_v2 import analyze_dataset, console_json
from scripts.generate_synthetic_train_v2 import generate_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_analyzer_reports_diversity_and_longtail_safety(tmp_path):
    generate_dataset(
        ROOT / "data" / "kb" / "metadata.db",
        tmp_path,
        count=60,
        seed=20260722,
        start_id=201,
    )

    report = analyze_dataset(tmp_path)

    assert report["documents"] == 60
    assert 330 <= report["words_per_document"]["mean"] <= 490
    assert report["entities_per_document"]["min"] >= 8
    assert report["entities_per_document"]["max"] >= 12
    assert report["fixed_line_character_share"] < 0.25
    assert report["duplicate_positive_reason_symptoms"] == 0
    assert report["adult_male_maternity_pediatric_documents"] == 0
    assert report["longtail_neutral_context_documents"] == report["longtail_documents"]
    assert report["quality_gate_passed"] is True


def test_console_json_is_safe_on_legacy_windows_code_pages():
    rendered = console_json({"entity_type": "TRIỆU_CHỨNG"})

    assert rendered.encode("cp1252")
