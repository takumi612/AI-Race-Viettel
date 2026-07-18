from types import MappingProxyType

import pytest

from src.validation.override_validator import (
    load_verified_overrides,
    normalize_override_term,
    validate_override_entries,
)


def test_known_wrong_rxnorm_mapping_is_rejected(metadata_db):
    entries = [
        {
            "term": "ketorolac",
            "type": "THUỐC",
            "codes": ["6809"],
            "source": "legacy",
            "note": "known bad mapping",
        }
    ]
    errors = validate_override_entries(entries, str(metadata_db))
    assert any("6809" in error and "metformin" in error for error in errors)


def test_override_requires_provenance(metadata_db):
    entries = [{"term": "tăng huyết áp", "type": "CHẨN_ĐOÁN", "codes": ["I10"]}]
    errors = validate_override_entries(entries, str(metadata_db))
    assert any("source" in error for error in errors)


def test_runtime_loader_rejects_quarantined_legacy_schema(project_root):
    legacy_path = project_root / "data" / "kb" / "override_dict.json"
    with pytest.raises(ValueError, match="verified override schema"):
        load_verified_overrides(legacy_path)


def test_verified_loader_returns_immutable_entries(tmp_path):
    path = tmp_path / "verified.json"
    path.write_text(
        """{
          "schema_version": 1,
          "entries": [{
            "term": "metformin",
            "type": "THUỐC",
            "codes": ["6809"],
            "source": "RxNorm",
            "note": "direct name match"
          }]
        }""",
        encoding="utf-8",
    )

    entries = load_verified_overrides(path)

    assert isinstance(entries[0], MappingProxyType)
    assert entries[0]["codes"] == ("6809",)
    with pytest.raises(TypeError):
        entries[0]["term"] = "changed"


def test_repository_verified_override_resource_is_clean(project_root, metadata_db):
    path = project_root / "src" / "resources" / "verified_overrides.json"
    entries = load_verified_overrides(path)
    assert validate_override_entries([dict(entry) for entry in entries], str(metadata_db)) == []


def test_pipeline_loads_only_the_verified_override_resource(monkeypatch, project_root):
    from src.pipeline import main as pipeline_main

    for dependency in (
        "PatientExtractor",
        "BaselineExtractor",
        "TextNormalizer",
        "AssertionAnalyzer",
        "HybridRetriever",
        "ClinicalValidator",
        "LLMReranker",
    ):
        monkeypatch.setattr(pipeline_main, dependency, lambda *args, **kwargs: object())

    loaded_paths = []

    def fake_loader(path):
        loaded_paths.append(path)
        return (
            MappingProxyType(
                {
                    "term": "  me\u0301tformin  ",
                    "type": "THUỐC",
                    "codes": ("6809",),
                    "source": "test",
                    "note": "direct name match",
                }
            ),
        )

    monkeypatch.setattr(pipeline_main, "load_verified_overrides", fake_loader)

    pipeline = pipeline_main.BaselinePipeline()

    assert loaded_paths == [
        project_root / "src" / "resources" / "verified_overrides.json"
    ]
    assert pipeline.override_dict == {"THUỐC": {"métformin": ("6809",)}}
    assert normalize_override_term(" MÉTFORMIN ") == "métformin"
