import json
import sqlite3

import pytest

import src.retrieval.hybrid_retriever as hybrid_module
from src.retrieval.hybrid_retriever import HybridRetriever
from src.training.embedding.config import EmbeddingTrainingConfig
from src.training.embedding.data import (
    CodeDescriptionStore,
    mine_hard_negative_examples,
    retrieval_ranking_metrics,
    select_embedding_seeds,
)
from src.training.embedding.index_manifest import (
    validate_index_manifest,
    write_index_manifest,
)
from src.training.embedding.train import main


def _database(tmp_path):
    path = tmp_path / "metadata.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE icd10 (code TEXT, name_vi TEXT, name_en TEXT)"
        )
        connection.execute("CREATE TABLE rxnorm (rxcui TEXT, name TEXT)")
        connection.executemany(
            "INSERT INTO icd10 VALUES (?, ?, ?)",
            [
                ("I10", "Tăng huyết áp", "Hypertension"),
                ("I11", "Bệnh tim do tăng huyết áp", "Hypertensive heart disease"),
            ],
        )
        connection.executemany(
            "INSERT INTO rxnorm VALUES (?, ?)",
            [("6809", "metformin"), ("6810", "metformin extended release")],
        )
    return path


def _seed(
    code="I10",
    *,
    entity_type="CHẨN_ĐOÁN",
    split="synthetic_train",
    fold=None,
):
    value = {
        "example_id": f"r1:{code}",
        "record_id": "r1",
        "query": "tăng huyết áp",
        "context": "Bệnh nhân tăng huyết áp.",
        "entity_type": entity_type,
        "positive_codes": [code],
        "split": split,
    }
    if fold is not None:
        value["fold"] = fold
    return value


def _config_mapping():
    return {
        "schema_version": 1,
        "base_model": "data/models/bge-m3",
        "dataset_dir": "data/training",
        "database": "data/kb/metadata.db",
        "output_dir": "artifacts/training/embedding",
        "seed": 20260719,
        "num_train_epochs": 1,
        "learning_rate": 0.0001,
        "train_batch_size": 4,
        "eval_batch_size": 8,
        "gradient_accumulation_steps": 4,
        "warmup_ratio": 0.1,
        "fp16": True,
        "gradient_checkpointing": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "hard_negatives": 3,
        "retrieval_top_k": 20,
        "bm25_alpha": 0.75,
        "recall_floor": 0.80,
        "local_files_only": True,
    }


def test_description_store_uses_correct_ontology_namespace(tmp_path):
    store = CodeDescriptionStore(_database(tmp_path))

    assert "Tăng huyết áp" in store.get("CHẨN_ĐOÁN", "I10")
    assert store.get("THUỐC", "6809") == "metformin"
    with pytest.raises(KeyError):
        store.get("THUỐC", "I10")


def test_hard_negative_mining_never_injects_or_labels_gold_as_negative(tmp_path):
    store = CodeDescriptionStore(_database(tmp_path))

    result = mine_hard_negative_examples(
        (_seed(),),
        store,
        retrieve=lambda entity_type, query, top_k: ["I10", "I11"],
        negatives_per_example=1,
        retrieval_top_k=20,
    )

    assert result.retrieval_misses == ()
    assert result.examples[0]["positive_code"] == "I10"
    assert result.examples[0]["negative_code"] == "I11"
    assert result.examples[0]["negative_code"] not in result.examples[0]["positive_codes"]


def test_embedding_stage_selection_excludes_holdout_and_validation_fold():
    seeds = (
        _seed(split="trusted_fold", fold=0),
        {**_seed(code="I11", split="trusted_fold", fold=1), "record_id": "r2"},
        _seed(split="holdout"),
    )

    train = select_embedding_seeds(
        seeds,
        stage="trusted-fold",
        role="train",
        fold=1,
    )
    evaluation = select_embedding_seeds(
        seeds,
        stage="trusted-fold",
        role="eval",
        fold=1,
    )

    assert [item["fold"] for item in train] == [0]
    assert [item["fold"] for item in evaluation] == [1]
    assert all(item["split"] != "holdout" for item in (*train, *evaluation))


def test_retrieval_metrics_report_recall_mrr_and_ndcg():
    metrics = retrieval_ranking_metrics(
        [
            ({"A"}, ["A", "B", "C"]),
            ({"D"}, ["B", "D", "E"]),
        ]
    )

    assert metrics["recall_at_1"] == pytest.approx(0.5)
    assert metrics["recall_at_5"] == 1.0
    assert metrics["mrr_at_10"] == pytest.approx(0.75)
    assert 0 < metrics["ndcg_at_10"] <= 1
    assert metrics["downstream_precision"] == pytest.approx(0.5)
    assert metrics["downstream_recall"] == 1.0
    assert metrics["downstream_f0_5"] == pytest.approx(5 / 9)


def test_embedding_config_locks_bm25_first_weight_sum():
    config = EmbeddingTrainingConfig.from_mapping(_config_mapping())

    assert config.bm25_weight == 0.75
    assert config.semantic_weight == 0.25
    assert config.bm25_weight + config.semantic_weight == 1.0

    invalid = _config_mapping()
    invalid["bm25_alpha"] = 0.49
    with pytest.raises(ValueError, match="BM25-first"):
        EmbeddingTrainingConfig.from_mapping(invalid)

    invalid_seed = _config_mapping()
    invalid_seed["seed"] = -1
    with pytest.raises(ValueError, match="integer hyperparameter"):
        EmbeddingTrainingConfig.from_mapping(invalid_seed)


def test_index_manifest_detects_adapter_or_index_mismatch(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"adapter")
    index = index_dir / "index.faiss"
    codes = index_dir / "codes.txt"
    embeddings = index_dir / "embeddings.npy"
    index.write_bytes(b"index")
    codes.write_text("I10\n", encoding="utf-8")
    embeddings.write_bytes(b"vectors")
    db = _database(tmp_path)

    write_index_manifest(
        index_dir,
        base_model="data/models/bge-m3",
        adapter_dir=adapter,
        database=db,
        embeddings=embeddings,
        index=index,
        codes=codes,
        count=1,
        dimension=1024,
    )
    manifest = validate_index_manifest(index_dir)
    assert manifest.count == 1

    index.write_bytes(b"changed")
    with pytest.raises(ValueError, match="index_sha256"):
        validate_index_manifest(index_dir)


def test_index_manifest_detects_database_and_adapter_mismatch(tmp_path):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.safetensors").write_bytes(b"adapter")
    index = index_dir / "index.faiss"
    codes = index_dir / "codes.txt"
    embeddings = index_dir / "embeddings.npy"
    index.write_bytes(b"index")
    codes.write_text("I10\n", encoding="utf-8")
    embeddings.write_bytes(b"vectors")
    db = _database(tmp_path)
    write_index_manifest(
        index_dir,
        base_model="data/models/bge-m3",
        adapter_dir=adapter,
        database=db,
        embeddings=embeddings,
        index=index,
        codes=codes,
        count=1,
        dimension=1024,
    )

    validate_index_manifest(
        index_dir,
        expected_database=db,
        expected_adapter_dir=adapter,
    )

    with sqlite3.connect(db) as connection:
        connection.execute(
            "INSERT INTO icd10 VALUES (?, ?, ?)",
            ("I12", "Bệnh thận tăng huyết áp", "Hypertensive kidney disease"),
        )
    with pytest.raises(ValueError, match="database_sha256"):
        validate_index_manifest(index_dir, expected_database=db)

    other_adapter = tmp_path / "other-adapter"
    other_adapter.mkdir()
    (other_adapter / "adapter.safetensors").write_bytes(b"different")
    with pytest.raises(ValueError, match="adapter_sha256"):
        validate_index_manifest(index_dir, expected_adapter_dir=other_adapter)


def test_hybrid_retriever_rejects_index_built_with_another_adapter(
    tmp_path, monkeypatch
):
    index_dir = tmp_path / "index"
    index_dir.mkdir()
    expected_adapter = tmp_path / "expected-adapter"
    expected_adapter.mkdir()
    (expected_adapter / "adapter.safetensors").write_bytes(b"expected")
    runtime_adapter = tmp_path / "runtime-adapter"
    runtime_adapter.mkdir()
    (runtime_adapter / "adapter.safetensors").write_bytes(b"runtime")
    index = index_dir / "index.faiss"
    codes = index_dir / "codes.txt"
    embeddings = index_dir / "embeddings.npy"
    index.write_bytes(b"index")
    codes.write_text("I10\n", encoding="utf-8")
    embeddings.write_bytes(b"vectors")
    db = _database(tmp_path)
    write_index_manifest(
        index_dir,
        base_model="data/models/bge-m3",
        adapter_dir=expected_adapter,
        database=db,
        embeddings=embeddings,
        index=index,
        codes=codes,
        count=1,
        dimension=1024,
    )
    monkeypatch.setattr(hybrid_module, "FAISS_AVAILABLE", True)

    retriever = HybridRetriever.__new__(HybridRetriever)
    retriever.table_name = "icd10"
    retriever.embedding_model_type = "BGE-M3"
    retriever.embedding_model_path = str(runtime_adapter)
    retriever.bm25_retriever = type("BM25", (), {"db_path": str(db)})()

    retriever._load_faiss_index(str(index_dir))

    assert retriever.faiss_available is False


def test_embedding_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--stage" in output
    assert "--initial-checkpoint" in output
    assert "--dry-run" in output
