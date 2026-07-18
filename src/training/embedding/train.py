"""BGE-M3 PEFT training CLI with split-safe BM25 hard negatives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from src.retrieval.bm25_retriever import BM25Retriever
from src.training.artifacts import start_or_resume_run, update_run_state
from src.training.embedding.config import EmbeddingTrainingConfig
from src.training.embedding.data import (
    CodeDescriptionStore,
    load_embedding_seeds,
    mine_hard_negative_examples,
    select_embedding_seeds,
)
from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)


def _checkpoint_identity(
    project_root: Path,
    initial_checkpoint: str | Path | None,
    *,
    stage: str,
) -> tuple[Path | None, str | None]:
    if stage.startswith("trusted") and initial_checkpoint is None:
        raise ValueError("trusted embedding stages require --initial-checkpoint")
    if stage == "synthetic" and initial_checkpoint is not None:
        raise ValueError("synthetic embedding stage starts from base_model")
    if initial_checkpoint is None:
        return None, None
    checkpoint = Path(initial_checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = project_root / checkpoint
    checkpoint = checkpoint.resolve()
    files = sorted(path for path in checkpoint.rglob("*") if path.is_file())
    if not checkpoint.is_dir() or not files:
        raise ValueError(f"initial embedding checkpoint is invalid: {checkpoint}")
    return checkpoint, fingerprint_files(files, checkpoint)


def build_embedding_run_plan(
    config: EmbeddingTrainingConfig,
    *,
    project_root: str | Path,
    stage: str,
    fold: int | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_dir = (root / config.dataset_dir).resolve()
    database = (root / config.database).resolve()
    manifest_path = dataset_dir / "manifests" / "build.json"
    seeds_path = dataset_dir / "embedding" / "seeds.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("build_id"), str
    ):
        raise ValueError("dataset manifest has no build_id")
    CodeDescriptionStore(database)
    seeds = load_embedding_seeds(seeds_path)
    train = select_embedding_seeds(seeds, stage=stage, role="train", fold=fold)
    evaluation = select_embedding_seeds(seeds, stage=stage, role="eval", fold=fold)
    if not train:
        raise ValueError(f"embedding stage {stage} has no training seeds")
    if stage != "trusted-final" and not evaluation:
        raise ValueError(f"embedding stage {stage} has no evaluation seeds")
    if config.local_files_only:
        model_path = (root / config.base_model).resolve()
        if not model_path.is_dir():
            raise ValueError(f"local BGE-M3 model is missing: {model_path}")
    return {
        "task": "embedding",
        "stage": stage,
        "fold": fold,
        "base_model": config.base_model,
        "bm25_weight": config.bm25_weight,
        "semantic_weight": config.semantic_weight,
        "config_sha256": stable_json_sha256(config.to_mapping()),
        "dataset_build_id": manifest["build_id"],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_seed_count": len(train),
        "eval_seed_count": len(evaluation),
        "train_record_ids": sorted({str(seed["record_id"]) for seed in train}),
        "eval_record_ids": sorted(
            {str(seed["record_id"]) for seed in evaluation}
        ),
    }


def _retrieval_function(database: Path):
    retrievers = {
        "CHẨN_ĐOÁN": BM25Retriever(db_path=str(database), table_name="icd10"),
        "THUỐC": BM25Retriever(db_path=str(database), table_name="rxnorm"),
    }

    def retrieve(entity_type: str, query: str, top_k: int):
        return retrievers[entity_type].retrieve_scored(query, top_k=top_k)

    return retrieve


def train_embedding(
    config: EmbeddingTrainingConfig,
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
    from peft import LoraConfig, TaskType
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from sentence_transformers.sentence_transformer.losses import (
        MultipleNegativesRankingLoss,
    )
    from sentence_transformers.sentence_transformer.training_args import (
        BatchSamplers,
    )

    plan = build_embedding_run_plan(
        config,
        project_root=root,
        stage=stage,
        fold=fold,
    )
    dataset_dir = (root / config.dataset_dir).resolve()
    database = (root / config.database).resolve()
    manifest_path = dataset_dir / "manifests" / "build.json"
    seeds = load_embedding_seeds(dataset_dir / "embedding" / "seeds.jsonl")
    train_seeds = select_embedding_seeds(
        seeds, stage=stage, role="train", fold=fold
    )
    eval_seeds = select_embedding_seeds(
        seeds, stage=stage, role="eval", fold=fold
    )
    descriptions = CodeDescriptionStore(database)
    retrieve = _retrieval_function(database)
    train_mined = mine_hard_negative_examples(
        train_seeds,
        descriptions,
        retrieve=retrieve,
        negatives_per_example=config.hard_negatives,
        retrieval_top_k=config.retrieval_top_k,
    )
    eval_mined = mine_hard_negative_examples(
        eval_seeds,
        descriptions,
        retrieve=retrieve,
        negatives_per_example=config.hard_negatives,
        retrieval_top_k=config.retrieval_top_k,
    )
    if not train_mined.examples:
        raise ValueError("hard-negative mining produced no embedding train examples")

    suffix = stage if fold is None else f"{stage}-{fold}"
    resolved_run_dir = (
        Path(run_dir).resolve()
        if run_dir is not None
        else (root / config.output_dir / f"{suffix}-candidate").resolve()
    )
    state = start_or_resume_run(
        resolved_run_dir,
        task="embedding",
        base_model=config.base_model,
        config={
            **config.to_mapping(),
            "stage": stage,
            "fold": fold,
            "initial_checkpoint_sha256": initial_sha256,
        },
        dataset_manifest=manifest_path,
        seed=config.seed,
        resume=resume,
    )
    model_source = (
        str((root / config.base_model).resolve())
        if config.local_files_only
        else config.base_model
    )
    model = SentenceTransformer(
        model_source,
        local_files_only=config.local_files_only,
    )
    if initial_path is None:
        model.add_adapter(
            LoraConfig(
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                bias="none",
                target_modules="all-linear",
                task_type=TaskType.FEATURE_EXTRACTION,
            )
        )
    else:
        model.load_adapter(
            str(initial_path),
            is_trainable=True,
            local_files_only=True,
        )
    transformer_model = getattr(model, "transformers_model", None)
    if transformer_model is not None:
        transformer_model.enable_input_require_grads()
    if config.gradient_checkpointing:
        first_module = model[0]
        auto_model = getattr(first_module, "auto_model", None)
        if auto_model is not None:
            auto_model.gradient_checkpointing_enable()

    def trainer_rows(examples):
        return [
            {
                "anchor": example["anchor"],
                "positive": example["positive"],
                "negative": example["negative"],
            }
            for example in examples
        ]

    train_dataset = Dataset.from_list(trainer_rows(train_mined.examples))
    eval_dataset = (
        Dataset.from_list(trainer_rows(eval_mined.examples))
        if eval_mined.examples
        else None
    )
    arguments = SentenceTransformerTrainingArguments(
        output_dir=str(resolved_run_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16 and torch.cuda.is_available(),
        bf16=False,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        save_steps=100,
        eval_steps=100 if eval_dataset is not None else None,
        save_total_limit=2,
        logging_steps=20,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
    )
    trainer = SentenceTransformerTrainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=MultipleNegativesRankingLoss(model),
    )
    resume_checkpoint = (
        str(resolved_run_dir / state.checkpoint)
        if resume and state.checkpoint
        else (True if resume else None)
    )
    train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = resolved_run_dir / "final"
    model.save_pretrained(final_dir)
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
            "train_example_count": len(train_mined.examples),
            "eval_example_count": len(eval_mined.examples),
            "train_retrieval_miss_count": len(train_mined.retrieval_misses),
            "eval_retrieval_miss_count": len(eval_mined.retrieval_misses),
            "metrics": dict(train_result.metrics),
        }
    )
    (resolved_run_dir / "training_metrics.json").write_text(
        json.dumps(
            plan["metrics"],
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=lambda value: value.item(),
        )
        + "\n",
        encoding="utf-8",
    )
    return plan


def _load_config(path: Path) -> EmbeddingTrainingConfig:
    if not path.is_file():
        raise ValueError(f"embedding config is missing: {path}")
    return EmbeddingTrainingConfig.from_mapping(
        yaml.safe_load(path.read_text(encoding="utf-8"))
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train BGE-M3 retrieval LoRA.")
    parser.add_argument(
        "--config",
        default="configs/training/embedding_bge_m3_lora.yaml",
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["synthetic", "trusted-fold", "trusted-final"],
    )
    parser.add_argument("--fold", type=int)
    parser.add_argument("--run-dir")
    parser.add_argument("--initial-checkpoint")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
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
        if args.dry_run:
            result = build_embedding_run_plan(
                config,
                project_root=root,
                stage=args.stage,
                fold=args.fold,
            )
        else:
            result = train_embedding(
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
