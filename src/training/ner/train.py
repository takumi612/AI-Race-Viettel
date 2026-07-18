"""XLM-R NER training CLI with fingerprint-locked resume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.training.artifacts import start_or_resume_run, update_run_state
from src.training.fingerprints import (
    fingerprint_files,
    sha256_file,
    stable_json_sha256,
)
from src.training.metrics import exact_fbeta
from src.training.ner.bio import (
    ID2LABEL,
    LABEL2ID,
    decode_bio_entities,
    merge_decoded_entities,
)
from src.training.ner.config import NERTrainingConfig
from src.training.ner.data import (
    load_ner_jsonl,
    select_ner_records,
    tokenize_ner_records,
)


def build_ner_run_plan(
    config: NERTrainingConfig,
    *,
    project_root: str | Path,
    stage: str,
    fold: int | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    dataset_dir = (root / config.dataset_dir).resolve()
    manifest_path = dataset_dir / "manifests" / "build.json"
    records_path = dataset_dir / "ner" / "records.jsonl"
    if not manifest_path.is_file():
        raise ValueError(f"dataset manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("build_id"), str
    ):
        raise ValueError("dataset manifest has no build_id")
    records = load_ner_jsonl(records_path)
    train_records = select_ner_records(
        records,
        stage=stage,
        role="train",
        fold=fold,
    )
    eval_records = select_ner_records(
        records,
        stage=stage,
        role="eval",
        fold=fold,
    )
    if not train_records:
        raise ValueError(f"NER stage {stage} has no training records")
    if stage != "trusted-final" and not eval_records:
        raise ValueError(f"NER stage {stage} has no evaluation records")
    if config.local_files_only:
        model_path = (root / config.base_model).resolve()
        if not model_path.is_dir():
            raise ValueError(f"local XLM-R model is missing: {model_path}")
    return {
        "task": "ner",
        "stage": stage,
        "fold": fold,
        "base_model": config.base_model,
        "config_sha256": stable_json_sha256(config.to_mapping()),
        "dataset_build_id": manifest["build_id"],
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "train_record_count": len(train_records),
        "eval_record_count": len(eval_records),
        "train_record_ids": [record["record_id"] for record in train_records],
        "eval_record_ids": [record["record_id"] for record in eval_records],
        "dataset_dir": config.dataset_dir.as_posix(),
    }


def build_compute_metrics(eval_features: Sequence[Mapping[str, Any]]):
    def compute(eval_prediction):
        import numpy as np

        if hasattr(eval_prediction, "predictions"):
            logits = eval_prediction.predictions
            label_ids = eval_prediction.label_ids
        else:
            logits, label_ids = eval_prediction
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.asarray(logits).argmax(axis=-1)
        labels = np.asarray(label_ids)
        if len(predictions) != len(eval_features):
            raise ValueError("prediction count does not match NER eval features")

        predicted_entities: list[Mapping[str, Any]] = []
        gold_entities: list[Mapping[str, Any]] = []
        for feature, predicted, gold in zip(eval_features, predictions, labels):
            predicted_entities.extend(
                decode_bio_entities(
                    feature["text"],
                    feature["absolute_offsets"],
                    predicted.tolist(),
                    record_id=feature["record_id"],
                )
            )
            gold_entities.extend(
                decode_bio_entities(
                    feature["text"],
                    feature["absolute_offsets"],
                    gold.tolist(),
                    record_id=feature["record_id"],
                )
            )
        predicted_merged = merge_decoded_entities(predicted_entities)
        gold_merged = merge_decoded_entities(gold_entities)

        def keys(values):
            return {
                (
                    value.get("record_id"),
                    value["type"],
                    value["position"][0],
                    value["position"][1],
                )
                for value in values
            }

        metrics = exact_fbeta(keys(gold_merged), keys(predicted_merged), beta=0.5)
        return {
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f0_5": metrics.f_beta,
        }

    return compute


class _FeatureDataset:
    def __init__(self, features: Sequence[Mapping[str, Any]]):
        self.features = tuple(features)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        feature = self.features[index]
        return {
            "input_ids": feature["input_ids"],
            "attention_mask": feature["attention_mask"],
            "labels": feature["labels"],
        }


def train_ner(
    config: NERTrainingConfig,
    *,
    project_root: str | Path,
    stage: str,
    fold: int | None,
    run_dir: str | Path | None,
    resume: bool,
    initial_checkpoint: str | Path | None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    if stage.startswith("trusted") and initial_checkpoint is None:
        raise ValueError("trusted NER stages require --initial-checkpoint")
    if stage == "synthetic" and initial_checkpoint is not None:
        raise ValueError("synthetic NER stage starts from base_model")
    initial_path = None
    initial_sha256 = None
    if initial_checkpoint is not None:
        initial_path = Path(initial_checkpoint)
        if not initial_path.is_absolute():
            initial_path = root / initial_path
        initial_path = initial_path.resolve()
        initial_files = sorted(
            path for path in initial_path.rglob("*") if path.is_file()
        )
        if not initial_path.is_dir() or not initial_files:
            raise ValueError(
                f"initial NER checkpoint is invalid: {initial_path}"
            )
        initial_sha256 = fingerprint_files(initial_files, initial_path)
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )
    import torch

    plan = build_ner_run_plan(
        config,
        project_root=root,
        stage=stage,
        fold=fold,
    )
    dataset_dir = (root / config.dataset_dir).resolve()
    manifest_path = dataset_dir / "manifests" / "build.json"
    records = load_ner_jsonl(dataset_dir / "ner" / "records.jsonl")
    train_records = select_ner_records(records, stage=stage, role="train", fold=fold)
    eval_records = select_ner_records(records, stage=stage, role="eval", fold=fold)

    base_model_source = (
        (root / config.base_model).resolve()
        if config.local_files_only
        else config.base_model
    )
    model_source = str(initial_path or base_model_source)
    tokenizer = AutoTokenizer.from_pretrained(
        model_source,
        use_fast=True,
        local_files_only=config.local_files_only,
    )
    train_features = tokenize_ner_records(
        train_records,
        tokenizer,
        max_length=config.max_length,
        stride=config.stride,
    )
    eval_features = tokenize_ner_records(
        eval_records,
        tokenizer,
        max_length=config.max_length,
        stride=config.stride,
    )

    suffix = stage if fold is None else f"{stage}-{fold}"
    resolved_run_dir = (
        Path(run_dir).resolve()
        if run_dir is not None
        else (root / config.output_dir / f"{suffix}-candidate").resolve()
    )
    state = start_or_resume_run(
        resolved_run_dir,
        task="ner",
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
    (resolved_run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                **config.to_mapping(),
                "stage": stage,
                "fold": fold,
                "initial_checkpoint_sha256": initial_sha256,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model = AutoModelForTokenClassification.from_pretrained(
        model_source,
        num_labels=len(LABEL2ID),
        label2id=LABEL2ID,
        id2label=ID2LABEL,
        local_files_only=config.local_files_only,
    )
    has_eval = bool(eval_features)
    arguments = TrainingArguments(
        output_dir=str(resolved_run_dir),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        fp16=config.fp16 and torch.cuda.is_available(),
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="steps" if has_eval else "no",
        save_strategy="steps",
        eval_steps=config.save_steps if has_eval else None,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=has_eval,
        metric_for_best_model="f0_5" if has_eval else None,
        greater_is_better=True,
        report_to=[],
        seed=config.seed,
        data_seed=config.seed,
    )
    callbacks = (
        [EarlyStoppingCallback(config.early_stopping_patience)] if has_eval else []
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=_FeatureDataset(train_features),
        eval_dataset=_FeatureDataset(eval_features) if has_eval else None,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(eval_features) if has_eval else None,
        callbacks=callbacks,
    )
    resume_checkpoint = (
        str(resolved_run_dir / state.checkpoint)
        if resume and state.checkpoint
        else (True if resume else None)
    )
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    final_dir = resolved_run_dir / "final"
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    metrics = dict(result.metrics)
    if has_eval:
        metrics.update(trainer.evaluate())
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
    checkpoint = final_dir
    state = update_run_state(
        resolved_run_dir,
        state,
        global_step=int(trainer.state.global_step),
        checkpoint=checkpoint,
    )
    plan["run_dir"] = str(resolved_run_dir)
    plan["global_step"] = state.global_step
    plan["initial_checkpoint_sha256"] = initial_sha256
    plan["metrics"] = metrics
    return plan


def _load_config(path: Path) -> NERTrainingConfig:
    if not path.is_file():
        raise ValueError(f"NER config is missing: {path}")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    return NERTrainingConfig.from_mapping(value)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train XLM-R clinical NER.")
    parser.add_argument(
        "--config",
        default="configs/training/ner_xlmr_base.yaml",
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
            result = build_ner_run_plan(
                config,
                project_root=root,
                stage=args.stage,
                fold=args.fold,
            )
        else:
            result = train_ner(
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
