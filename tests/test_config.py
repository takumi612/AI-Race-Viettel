import json

import pytest

from src.config import NERConfig, PipelineConfig, RerankerConfig, RetrievalConfig
from src.validation.override_validator import find_machine_specific_paths


def test_retrieval_weights_sum_to_one():
    config = RetrievalConfig(alpha=0.75)
    assert config.bm25_weight == 0.75
    assert config.semantic_weight == 0.25
    assert config.bm25_weight + config.semantic_weight == pytest.approx(1.0)


def test_alpha_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        RetrievalConfig(alpha=1.1)


def test_default_pipeline_is_precision_first():
    config = PipelineConfig()
    assert config.retrieval.alpha == 0.75
    assert config.ner.beta == 0.5
    assert set(config.ner.per_type_thresholds) == {
        "CHẨN_ĐOÁN",
        "TRIỆU_CHỨNG",
        "THUỐC",
        "TÊN_XÉT_NGHIỆM",
        "KẾT_QUẢ_XÉT_NGHIỆM",
    }
    assert config.retrieval.internal_top_k == 20
    assert config.retrieval.embedding_model_type == "BGE-M3"
    assert config.retrieval.embedding_model_artifact is None
    assert config.retrieval.icd_index_artifact is None
    assert config.retrieval.rxnorm_index_artifact is None
    assert config.assertion.negated_threshold == 0.70
    assert config.selection.load_historical_rxnorm is False
    assert config.reranker.enabled is False
    assert config.reranker.backend == "http"
    assert config.reranker.model_artifact is None
    assert config.reranker.max_new_tokens == 64
    assert config.reranker.timeout_seconds == 30.0


def test_non_positive_reranker_timeout_is_rejected():
    with pytest.raises(ValueError):
        RerankerConfig(timeout_seconds=0)


def test_local_reranker_requires_project_relative_artifact():
    with pytest.raises(ValueError, match="model_artifact"):
        RerankerConfig(enabled=True, backend="local_transformers")
    with pytest.raises(ValueError, match="project-relative"):
        RerankerConfig(
            enabled=True,
            backend="local_transformers",
            model_artifact="/content/model",
        )
    config = RerankerConfig(
        enabled=True,
        backend="local_transformers",
        model_artifact="artifacts/training/reranker/final",
    )
    assert config.model_artifact == "artifacts/training/reranker/final"


def test_threshold_outside_unit_interval_is_rejected():
    with pytest.raises(ValueError):
        NERConfig(default_threshold=1.01)


def test_config_mapping_rejects_unknown_keys_and_preserves_weight_invariant():
    config = PipelineConfig.from_mapping(
        {
            "retrieval": {
                "alpha": 0.80,
                "embedding_model_type": "BGE-M3",
                "embedding_model_artifact": "artifacts/training/embedding/final",
                "icd_index_artifact": "artifacts/indexes/icd10",
                "rxnorm_index_artifact": "artifacts/indexes/rxnorm",
            }
        }
    )
    assert config.retrieval.bm25_weight + config.retrieval.semantic_weight == pytest.approx(1.0)
    assert (
        config.retrieval.embedding_model_artifact
        == "artifacts/training/embedding/final"
    )
    assert config.retrieval.icd_index_artifact == "artifacts/indexes/icd10"
    assert config.retrieval.rxnorm_index_artifact == "artifacts/indexes/rxnorm"
    with pytest.raises(ValueError, match="unknown"):
        PipelineConfig.from_mapping({"retrieval": {"bonus": 0.20}})


@pytest.mark.parametrize(
    "artifact",
    [r"D:\models\bge-m3", "/content/models/bge-m3", "../models/bge-m3"],
)
def test_retrieval_model_artifact_must_be_project_relative(artifact):
    with pytest.raises(ValueError, match="project-relative"):
        RetrievalConfig(embedding_model_artifact=artifact)


@pytest.mark.parametrize(
    "values",
    [
        {"unexpected": True},
        {"chunking": {"unexpected": True}},
        {"ner": {"unexpected": True}},
        {"assertion": {"unexpected": True}},
        {"selection": {"unexpected": True}},
        {"reranker": {"unexpected": True}},
    ],
)
def test_config_mapping_rejects_unknown_keys_at_every_level(values):
    with pytest.raises(ValueError, match="unknown"):
        PipelineConfig.from_mapping(values)


def test_to_dict_returns_json_compatible_data_detached_from_config():
    config = PipelineConfig.from_mapping(
        {"ner": {"per_type_thresholds": {"THUỐC": 0.81}}}
    )

    result = config.to_dict()

    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    assert result["ner"]["per_type_thresholds"] == {"THUỐC": 0.81}
    result["ner"]["per_type_thresholds"]["THUỐC"] = 0.1
    assert config.ner.per_type_thresholds["THUỐC"] == 0.81


def test_absolute_path_audit_reports_file_line_and_value(tmp_path):
    source_path = tmp_path / "bad_runtime.py"
    source_path.write_text('DATA_DIR = r"D:\\\\private-data"\n', encoding="utf-8")
    findings = find_machine_specific_paths([source_path])
    assert [(item.path, item.line_number, item.value) for item in findings] == [
        (source_path, 1, r"D:\\private-data")
    ]


@pytest.mark.parametrize(
    "values",
    [
        {"retrieval": {"alpha": True}},
        {"retrieval": {"internal_top_k": 1.5}},
        {"retrieval": {"hierarchical_expansion": "false"}},
        {"selection": {"load_historical_rxnorm": "false"}},
        {"reranker": {"enabled": "false"}},
        {"ner": {"per_type_thresholds": [["THUỐC", 0.8]]}},
        {"ner": {"beta": float("inf")}},
        {"reranker": {"timeout_seconds": float("inf")}},
    ],
)
def test_config_mapping_rejects_invalid_json_types(values):
    with pytest.raises(ValueError):
        PipelineConfig.from_mapping(values)


def test_path_audit_fails_closed_for_missing_targets(tmp_path):
    missing = tmp_path / "missing.py"
    with pytest.raises(ValueError, match="missing.py"):
        find_machine_specific_paths([missing])


def test_recall_file_selection_excludes_pseudo_and_holdout_labels(tmp_path):
    from src.retrieval.eval_recall import select_trusted_ground_truth_files

    for file_id in (1, 100, 101, 102, 180, 181, 200):
        (tmp_path / f"{file_id}.json").write_text("[]", encoding="utf-8")

    selected = select_trusted_ground_truth_files(tmp_path, limit=3)

    assert [path.stem for path in selected] == ["101", "102", "180"]


def test_aggregation_excludes_pseudo_and_holdout_labels(tmp_path):
    from scripts.aggregate_data import aggregate_data

    input_dir = tmp_path / "input"
    gt_dir = tmp_path / "gt"
    input_dir.mkdir()
    gt_dir.mkdir()
    for file_id in (1, 101, 181):
        (input_dir / f"{file_id}.txt").write_text(str(file_id), encoding="utf-8")
        (gt_dir / f"{file_id}.json").write_text("[]", encoding="utf-8")
    out_txt = tmp_path / "combined.txt"
    out_json = tmp_path / "combined.json"

    aggregate_data(input_dir, gt_dir, out_txt, out_json)

    assert set(json.loads(out_json.read_text(encoding="utf-8"))) == {"101"}
    combined_text = out_txt.read_text(encoding="utf-8")
    assert "File 1.txt" not in combined_text
    assert "File 101.txt" in combined_text
    assert "File 181.txt" not in combined_text


@pytest.mark.parametrize("value", ["/data/models", "/workspace/run"])
def test_path_audit_reports_generic_posix_absolute_paths(tmp_path, value):
    source_path = tmp_path / "bad_posix_runtime.py"
    source_path.write_text(f"MODEL_PATH = {value!r}\n", encoding="utf-8")

    findings = find_machine_specific_paths([source_path])

    assert [(item.path, item.line_number, item.value) for item in findings] == [
        (source_path, 1, value)
    ]


@pytest.mark.parametrize(
    "value",
    [
        "data/models",
        "./workspace/run",
        "https://example.test/model",
        "/",
        "/chat/completions",
    ],
)
def test_path_audit_ignores_relative_paths_and_urls(tmp_path, value):
    source_path = tmp_path / "clean_runtime.py"
    source_path.write_text(f"MODEL_PATH = {value!r}\n", encoding="utf-8")

    assert find_machine_specific_paths([source_path]) == []
