"""Freeze retrieval pools and train the Qwen2.5-7B subset reranker."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.retrieval.hybrid_retriever import HybridRetriever
from src.training.artifacts import start_or_resume_run, update_run_state
from src.training.embedding.data import CodeDescriptionStore
from src.training.embedding.index_manifest import validate_index_manifest
from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)
from src.training.reranker.config import RerankerTrainingConfig
from src.training.reranker.data import (
    build_reranker_prompt,
    completion_only_features,
    freeze_candidate_examples,
    load_frozen_candidate_dataset,
    reranker_generation_metrics,
    select_reranker_examples,
    target_json,
    write_frozen_candidate_dataset,
)


def _load_config(path: Path) -> RerankerTrainingConfig:
    if not path.is_file():
        raise ValueError(f"reranker config is missing: {path}")
    return RerankerTrainingConfig.from_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def _load_jsonl(path: Path) -> tuple[dict[str, Any], ...]:
    if not path.is_file():
        raise ValueError(f"reranker seeds are missing: {path}")
    values: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"blank reranker seed line: {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"reranker seed {line_number} is not an object")
            values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid reranker seeds JSONL") from exc
    return tuple(values)


def _dataset_manifest(root: Path, config: RerankerTrainingConfig):
    path = (root / config.dataset_dir / "manifests" / "build.json").resolve()
    if not path.is_file():
        raise ValueError(f"dataset manifest is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid training dataset manifest") from exc
    if not isinstance(value, dict) or not isinstance(value.get("build_id"), str):
        raise ValueError("training dataset manifest has no build_id")
    return path, value


def _checkpoint_identity(
    project_root: Path,
    initial_checkpoint: str | Path | None,
    *,
    stage: str,
) -> tuple[Path | None, str | None]:
    if stage.startswith("trusted") and initial_checkpoint is None:
        raise ValueError("trusted reranker stages require --initial-checkpoint")
    if stage == "synthetic" and initial_checkpoint is not None:
        raise ValueError("synthetic reranker stage starts from base_model")
    if initial_checkpoint is None:
        return None, None
    checkpoint = Path(initial_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    checkpoint = checkpoint.resolve()
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not checkpoint.is_dir() or not files:
        raise ValueError(f"initial reranker checkpoint is invalid: {checkpoint}")
    return checkpoint, fingerprint_files(files, checkpoint)


def freeze_candidates(
    config: RerankerTrainingConfig,
    *,
    project_root: str | Path,
    icd_index_dir: str | Path,
    rxnorm_index_dir: str | Path,
    embedding_model_artifact: str | Path,
    alpha: float,
    internal_top_k: int,
    candidate_top_k: int,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if not 0.5 <= alpha <= 1:
        raise ValueError("BM25-first candidate freezing requires alpha in [0.5, 1]")
    if (
        isinstance(internal_top_k, bool)
        or not isinstance(internal_top_k, int)
        or internal_top_k < 1
        or isinstance(candidate_top_k, bool)
        or not isinstance(candidate_top_k, int)
        or candidate_top_k < 1
        or candidate_top_k > internal_top_k
    ):
        raise ValueError("invalid candidate retrieval limits")
    database = (root / config.database).resolve()
    adapter = Path(embedding_model_artifact)
    if not adapter.is_absolute():
        adapter = root / adapter
    adapter = adapter.resolve()
    icd_index = Path(icd_index_dir)
    if not icd_index.is_absolute():
        icd_index = root / icd_index
    rxnorm_index = Path(rxnorm_index_dir)
    if not rxnorm_index.is_absolute():
        rxnorm_index = root / rxnorm_index
    icd_manifest = validate_index_manifest(
        icd_index,
        expected_database=database,
        expected_adapter_dir=adapter,
    )
    rxnorm_manifest = validate_index_manifest(
        rxnorm_index,
        expected_database=database,
        expected_adapter_dir=adapter,
    )
    retriever_fingerprint = stable_json_sha256(
        {
            "schema_version": 1,
            "alpha": alpha,
            "semantic_weight": 1.0 - alpha,
            "internal_top_k": internal_top_k,
            "candidate_top_k": candidate_top_k,
            "icd_index": asdict(icd_manifest),
            "rxnorm_index": asdict(rxnorm_manifest),
        }
    )
    retrievers = {
        "CHẨN_ĐOÁN": HybridRetriever(
            table_name="icd10",
            db_path=str(database),
            index_dir=str(icd_index.resolve()),
            alpha=alpha,
            internal_top_k=internal_top_k,
            embedding_model_type="BGE-M3",
            embedding_model_path=str(adapter),
        ),
        "THUỐC": HybridRetriever(
            table_name="rxnorm",
            db_path=str(database),
            index_dir=str(rxnorm_index.resolve()),
            alpha=alpha,
            internal_top_k=internal_top_k,
            embedding_model_type="BGE-M3",
            embedding_model_path=str(adapter),
        ),
    }

    def retrieve(entity_type: str, query: str, top_k: int):
        return retrievers[entity_type].retrieve_scored(query, top_k=top_k)

    descriptions = CodeDescriptionStore(database)
    seeds = _load_jsonl(
        (root / config.dataset_dir / "reranker" / "seeds.jsonl").resolve()
    )
    result = freeze_candidate_examples(
        seeds,
        retrieve=retrieve,
        describe=descriptions.get,
        retriever_fingerprint=retriever_fingerprint,
        top_k=candidate_top_k,
    )
    manifest_path, source_manifest = _dataset_manifest(root, config)
    output = (root / config.candidate_dataset).resolve()
    write_frozen_candidate_dataset(
        output,
        result,
        dataset_build_id=source_manifest["build_id"],
        dataset_manifest_sha256=sha256_file(manifest_path),
    )
    return {
        "candidate_dataset": str(output),
        "retriever_fingerprint": retriever_fingerprint,
        "example_count": len(result.examples),
        "retrieval_miss_count": len(result.retrieval_misses),
        "bm25_weight": alpha,
        "semantic_weight": 1.0 - alpha,
    }


def build_reranker_run_plan(
    config: RerankerTrainingConfig,
    *,
    project_root: str | Path,
    stage: str,
    fold: int | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    source_manifest_path, source_manifest = _dataset_manifest(root, config)
    frozen_manifest, examples = load_frozen_candidate_dataset(
        root / config.candidate_dataset
    )
    if frozen_manifest["dataset_build_id"] != source_manifest["build_id"]:
        raise ValueError("frozen candidates use another dataset build")
    if (
        frozen_manifest["dataset_manifest_sha256"]
        != sha256_file(source_manifest_path)
    ):
        raise ValueError("frozen candidate source manifest mismatch")
    train_examples = select_reranker_examples(
        examples, stage=stage, role="train", fold=fold
    )
    eval_examples = select_reranker_examples(
        examples, stage=stage, role="eval", fold=fold
    )
    if not train_examples:
        raise ValueError(f"reranker stage {stage} has no training examples")
    if stage != "trusted-final" and not eval_examples:
        raise ValueError(f"reranker stage {stage} has no evaluation examples")
    if config.local_files_only:
        model_path = (root / config.base_model).resolve()
        if not model_path.is_dir():
            raise ValueError(f"local Qwen model is missing: {model_path}")
    return {
        "task": "reranker",
        "stage": stage,
        "fold": fold,
        "base_model": config.base_model,
        "config_sha256": stable_json_sha256(config.to_mapping()),
        "dataset_build_id": source_manifest["build_id"],
        "dataset_manifest_sha256": sha256_file(source_manifest_path),
        "retriever_fingerprint": frozen_manifest["retriever_fingerprint"],
        "frozen_examples_sha256": frozen_manifest["examples_sha256"],
        "train_example_count": len(train_examples),
        "eval_example_count": len(eval_examples),
        "train_record_ids": sorted(
            {str(example["record_id"]) for example in train_examples}
        ),
        "eval_record_ids": sorted(
            {str(example["record_id"]) for example in eval_examples}
        ),
    }


def _generate_outputs(
    model: Any,
    tokenizer: Any,
    examples: Sequence[dict[str, Any]],
    *,
    max_seq_length: int,
    max_new_tokens: int,
) -> list[str]:
    outputs: list[str] = []
    model.eval()
    for example in examples:
        prompt = build_reranker_prompt(example)
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length - max_new_tokens,
        )
        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        completion = generated[0, inputs["input_ids"].shape[1] :]
        outputs.append(
            tokenizer.decode(completion, skip_special_tokens=True).strip()
        )
    return outputs


def train_reranker(
    config: RerankerTrainingConfig,
    *,
    project_root: str | Path,
    stage: str,
    fold: int | None,
    run_dir: str | Path | None,
    resume: bool,
    initial_checkpoint: str | Path | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    initial_path, initial_sha256 = _checkpoint_identity(
        root,
        initial_checkpoint,
        stage=stage,
    )
    import torch
    from datasets import Dataset
    from peft import (
        LoraConfig,
        PeftModel,
        TaskType,
        get_peft_model,
        prepare_model_for_kbit_training,
    )
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        raise ValueError("Qwen 4-bit QLoRA requires a CUDA GPU runtime")
    plan = build_reranker_run_plan(
        config,
        project_root=root,
        stage=stage,
        fold=fold,
    )
    source_manifest_path, _ = _dataset_manifest(root, config)
    frozen_manifest, examples = load_frozen_candidate_dataset(
        root / config.candidate_dataset
    )
    train_examples = select_reranker_examples(
        examples, stage=stage, role="train", fold=fold
    )
    eval_examples = select_reranker_examples(
        examples, stage=stage, role="eval", fold=fold
    )
    suffix = stage if fold is None else f"{stage}-{fold}"
    resolved_run_dir = (
        Path(run_dir).resolve()
        if run_dir is not None
        else (root / config.output_dir / f"{suffix}-candidate").resolve()
    )
    run_config = {
        **config.to_mapping(),
        "stage": stage,
        "fold": fold,
        "retriever_fingerprint": frozen_manifest["retriever_fingerprint"],
        "frozen_examples_sha256": frozen_manifest["examples_sha256"],
        "initial_checkpoint_sha256": initial_sha256,
    }
    state = start_or_resume_run(
        resolved_run_dir,
        task="reranker",
        base_model=config.base_model,
        config=run_config,
        dataset_manifest=source_manifest_path,
        seed=config.seed,
        resume=resume,
    )
    model_source = (
        str((root / config.base_model).resolve())
        if config.local_files_only
        else config.base_model
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        local_files_only=config.local_files_only,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=config.bnb_use_double_quant,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_source,
        local_files_only=config.local_files_only,
        quantization_config=quantization,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=config.gradient_checkpointing,
    )
    if initial_path is None:
        model = get_peft_model(
            model,
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules="all-linear",
                task_type=TaskType.CAUSAL_LM,
            ),
        )
    else:
        model = PeftModel.from_pretrained(
            model,
            initial_path,
            is_trainable=True,
            local_files_only=True,
        )

    def encode(example: Mapping[str, Any]):
        return completion_only_features(
            tokenizer,
            build_reranker_prompt(example),
            target_json(example),
            max_length=config.max_seq_length,
        )

    train_dataset = Dataset.from_list(
        [encode(example) for example in train_examples]
    )
    arguments = TrainingArguments(
        output_dir=str(resolved_run_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        logging_steps=10,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
        ),
    )
    resume_checkpoint = (
        str(resolved_run_dir / state.checkpoint)
        if resume and state.checkpoint
        else (True if resume else None)
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = resolved_run_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    (final_dir / "runtime.json").write_text(
        json.dumps(
            {"schema_version": 1, "base_model": config.base_model},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    metrics: dict[str, Any] = dict(train_result.metrics)
    if eval_examples:
        generated_outputs = _generate_outputs(
            model,
            tokenizer,
            eval_examples,
            max_seq_length=config.max_seq_length,
            max_new_tokens=config.max_new_tokens,
        )
        generation_metrics = reranker_generation_metrics(
            generated_outputs,
            eval_examples,
        )
        metrics.update(generation_metrics)
    (resolved_run_dir / "training_metrics.json").write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item(),
        )
        + "\n",
        encoding="utf-8",
    )
    if eval_examples:
        if generation_metrics["out_of_pool_rate"] != 0:
            raise ValueError(
                "reranker generated out-of-pool codes; artifact remains candidate"
            )
    state = update_run_state(
        resolved_run_dir,
        state,
        global_step=int(trainer.state.global_step),
        checkpoint=final_dir,
    )
    plan.update(
        {
            "run_dir": str(resolved_run_dir),
            "global_step": state.global_step,
            "initial_checkpoint_sha256": initial_sha256,
            "metrics": metrics,
        }
    )
    return plan


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze candidates or train Qwen2.5-7B QLoRA reranker."
    )
    parser.add_argument(
        "--config",
        default="configs/training/reranker_qwen25_7b_qlora.yaml",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage",
        choices=["synthetic", "trusted-fold", "trusted-final"],
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--freeze-candidates", action="store_true")
    parser.add_argument("--icd-index-dir")
    parser.add_argument("--rxnorm-index-dir")
    parser.add_argument("--embedding-model-artifact")
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--internal-top-k", type=int, default=20)
    parser.add_argument("--candidate-top-k", type=int, default=10)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    try:
        config = _load_config(config_path)
        if args.freeze_candidates:
            missing = [
                name
                for name, value in (
                    ("--icd-index-dir", args.icd_index_dir),
                    ("--rxnorm-index-dir", args.rxnorm_index_dir),
                    (
                        "--embedding-model-artifact",
                        args.embedding_model_artifact,
                    ),
                )
                if value is None
            ]
            if missing:
                parser.error(
                    "candidate freezing requires " + ", ".join(missing)
                )
            result = freeze_candidates(
                config,
                project_root=root,
                icd_index_dir=args.icd_index_dir,
                rxnorm_index_dir=args.rxnorm_index_dir,
                embedding_model_artifact=args.embedding_model_artifact,
                alpha=args.alpha,
                internal_top_k=args.internal_top_k,
                candidate_top_k=args.candidate_top_k,
            )
        else:
            if args.stage is None:
                parser.error("--stage is required for reranker training")
            if args.dry_run:
                result = build_reranker_run_plan(
                    config,
                    project_root=root,
                    stage=args.stage,
                    fold=args.fold,
                )
            else:
                result = train_reranker(
                    config,
                    project_root=root,
                    stage=args.stage,
                    fold=args.fold,
                    run_dir=args.run_dir,
                    resume=args.resume,
                    initial_checkpoint=args.initial_checkpoint,
                )
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
