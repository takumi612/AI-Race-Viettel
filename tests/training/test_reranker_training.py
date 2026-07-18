import json

import pytest

from src.training.reranker.config import RerankerTrainingConfig
from src.training.reranker.data import (
    FrozenCandidateResult,
    build_reranker_prompt,
    completion_only_features,
    freeze_candidate_examples,
    load_frozen_candidate_dataset,
    reranker_generation_metrics,
    select_reranker_examples,
    target_json,
    write_frozen_candidate_dataset,
)
from src.training.reranker.train import main


RETRIEVER_FINGERPRINT = "a" * 64


def _seed(
    *,
    example_id="r1:0",
    code="I10",
    split="synthetic_train",
    fold=None,
):
    value = {
        "example_id": example_id,
        "record_id": example_id.split(":")[0],
        "context": "Bệnh nhân tăng huyết áp.",
        "entity_text": "tăng huyết áp",
        "entity_type": "CHẨN_ĐOÁN",
        "assertions": [],
        "ground_truth_codes": [code],
        "split": split,
    }
    if fold is not None:
        value["fold"] = fold
    return value


def _config_mapping():
    return {
        "schema_version": 1,
        "base_model": "data/models/Qwen2.5-7B-Instruct",
        "dataset_dir": "data/training",
        "candidate_dataset": "artifacts/training/reranker/frozen",
        "database": "data/kb/metadata.db",
        "output_dir": "artifacts/training/reranker",
        "seed": 20260719,
        "num_train_epochs": 1,
        "learning_rate": 0.0001,
        "train_batch_size": 1,
        "eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1024,
        "max_new_tokens": 64,
        "warmup_ratio": 0.1,
        "fp16": True,
        "gradient_checkpointing": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "load_in_4bit": True,
        "bnb_quant_type": "nf4",
        "bnb_use_double_quant": True,
        "local_files_only": True,
    }


def test_freeze_candidates_never_injects_gold_and_records_retrieval_miss():
    seeds = (
        _seed(),
        _seed(example_id="r2:0", code="I11"),
    )

    result = freeze_candidate_examples(
        seeds,
        retrieve=lambda entity_type, query, top_k: (
            ["I10", "I12"] if query == "tăng huyết áp" else ["I12", "I13"]
        ),
        describe=lambda entity_type, code: f"description {code}",
        retriever_fingerprint=RETRIEVER_FINGERPRINT,
        top_k=2,
    )

    assert len(result.examples) == 1
    assert result.examples[0]["selected_codes"] == ["I10"]
    assert [item["code"] for item in result.examples[0]["candidates"]] == [
        "I10",
        "I12",
    ]
    assert result.retrieval_misses == ("r2:0",)
    assert all(
        candidate["code"] != "I11"
        for example in result.examples
        for candidate in example["candidates"]
    )


def test_freeze_candidates_requires_sha256_retriever_identity():
    with pytest.raises(ValueError, match="retriever fingerprint"):
        freeze_candidate_examples(
            (_seed(),),
            retrieve=lambda entity_type, query, top_k: ["I10"],
            describe=lambda entity_type, code: code,
            retriever_fingerprint="not-a-sha",
            top_k=5,
        )


def test_frozen_candidate_artifact_detects_tampering(tmp_path):
    result = FrozenCandidateResult(
        examples=(
            {
                **_seed(),
                "candidates": [{"code": "I10", "description": "Hypertension"}],
                "selected_codes": ["I10"],
                "retriever_fingerprint": RETRIEVER_FINGERPRINT,
            },
        ),
        retrieval_misses=(),
        retriever_fingerprint=RETRIEVER_FINGERPRINT,
    )
    output = tmp_path / "frozen"
    write_frozen_candidate_dataset(
        output,
        result,
        dataset_build_id="build-1",
        dataset_manifest_sha256="b" * 64,
    )

    manifest, examples = load_frozen_candidate_dataset(output)
    assert manifest["retriever_fingerprint"] == RETRIEVER_FINGERPRINT
    assert len(examples) == 1

    (output / "examples.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_frozen_candidate_dataset(output)


def test_reranker_stage_selection_excludes_holdout_and_validation_fold():
    examples = (
        {**_seed(split="trusted_fold", fold=0), "selected_codes": ["I10"]},
        {
            **_seed(example_id="r2:0", split="trusted_fold", fold=1),
            "selected_codes": ["I10"],
        },
        {
            **_seed(example_id="r3:0", split="holdout"),
            "selected_codes": ["I10"],
        },
    )

    train = select_reranker_examples(
        examples, stage="trusted-fold", role="train", fold=1
    )
    evaluation = select_reranker_examples(
        examples, stage="trusted-fold", role="eval", fold=1
    )

    assert [item["fold"] for item in train] == [0]
    assert [item["fold"] for item in evaluation] == [1]
    assert all(item["split"] != "holdout" for item in (*train, *evaluation))


def test_prompt_target_is_canonical_json_subset_with_at_most_two_codes():
    example = {
        **_seed(),
        "candidates": [
            {"code": "I10", "description": "Hypertension"},
            {"code": "I11", "description": "Hypertensive heart disease"},
        ],
        "selected_codes": ["I10"],
    }

    prompt = build_reranker_prompt(example)

    assert "I10" in prompt and "I11" in prompt
    assert target_json(example) == '{"selected_codes":["I10"]}'

    invalid = {**example, "selected_codes": ["I10", "I11", "I12"]}
    with pytest.raises(ValueError, match="zero to two"):
        target_json(invalid)
    outside = {**example, "selected_codes": ["I12"]}
    with pytest.raises(ValueError, match="candidate pool"):
        target_json(outside)


def test_completion_only_features_mask_every_prompt_token():
    class CharacterTokenizer:
        eos_token_id = 0

        def encode(self, text, add_special_tokens=False):
            return [ord(character) for character in text]

    features = completion_only_features(
        CharacterTokenizer(),
        "PROMPT",
        '{"selected_codes":[]}',
        max_length=64,
    )

    assert features["labels"][:6] == [-100] * 6
    assert features["labels"][6:] == features["input_ids"][6:]
    assert features["input_ids"][-1] == 0


def test_generation_metrics_report_invalid_and_out_of_pool_separately():
    examples = [
        {"candidates": [{"code": "A"}, {"code": "B"}], "selected_codes": ["A"]},
        {"candidates": [{"code": "C"}], "selected_codes": ["C"]},
        {"candidates": [{"code": "D"}], "selected_codes": []},
    ]

    metrics = reranker_generation_metrics(
        [
            '{"selected_codes":["A"]}',
            '{"selected_codes":["FOREIGN"]}',
            "not-json",
        ],
        examples,
    )

    assert metrics["invalid_json_rate"] == pytest.approx(1 / 3)
    assert metrics["out_of_pool_rate"] == pytest.approx(1 / 3)
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert 0 < metrics["f0_5"] < 1


def test_qwen_config_enforces_qlora_and_micro_batch_one():
    config = RerankerTrainingConfig.from_mapping(_config_mapping())

    assert config.train_batch_size == 1
    assert config.load_in_4bit is True
    assert config.bnb_quant_type == "nf4"

    invalid = _config_mapping()
    invalid["train_batch_size"] = 2
    with pytest.raises(ValueError, match="micro-batch"):
        RerankerTrainingConfig.from_mapping(invalid)

    invalid = _config_mapping()
    invalid["load_in_4bit"] = False
    with pytest.raises(ValueError, match="4-bit"):
        RerankerTrainingConfig.from_mapping(invalid)


def test_reranker_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert "--stage" in output
    assert "--initial-checkpoint" in output
    assert "--dry-run" in output
