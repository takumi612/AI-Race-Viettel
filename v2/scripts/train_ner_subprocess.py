import argparse
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from clinical_nlp_lab.config import load_config, set_reproducible_seed
from clinical_nlp_lab.data import grouped_train_validation_split, load_ner_training_documents, validate_documents
from clinical_nlp_lab.dataset_quality import DatasetRecord
from clinical_nlp_lab.schema import write_json
from clinical_nlp_lab.training import (
    build_bio_label_map,
    build_training_contract,
    compute_non_o_metrics,
    remove_nested_checkpoints,
)

class FeatureDataset(Dataset):
    def __init__(self, windows): self.windows = tuple(windows)
    def __len__(self): return len(self.windows)
    def __getitem__(self, index):
        return {"window": self.windows[index]}


def load_stage_selection(path: str | Path) -> dict[str, object]:
    """Load and canonicalize a curriculum stage's train/validation IDs."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("stage manifest must be an object")
    stage_name = payload.get("stage_name")
    train_ids = payload.get("train_ids")
    validation_ids = payload.get("validation_ids")
    if not isinstance(stage_name, str) or not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("stage manifest requires stage_name, train_ids and validation_ids")
    return {
        "stage_name": stage_name,
        "train_ids": tuple(sorted({str(value) for value in train_ids}, key=lambda value: int(value))),
        "validation_ids": tuple(sorted({str(value) for value in validation_ids}, key=lambda value: int(value))),
        "dataset_fingerprint": str(payload.get("dataset_fingerprint", "")),
        "split_fingerprint": str(payload.get("split_fingerprint", "")),
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--model-source", type=str, required=True)
    parser.add_argument("--stage-manifest", type=str, default=None)
    parser.add_argument("--stage-name", type=str, default=None)
    parser.add_argument("--fast-dev-run", type=str, default="False")
    args = parser.parse_args()

    fast_dev_run = args.fast_dev_run.lower() in ("true", "1", "yes")
    print(
        f"[TRAINING_START] train_source={args.train_source} output_dir={args.output_dir} "
        f"model_source={args.model_source} fast_dev_run={fast_dev_run}",
        flush=True,
    )

    config_path = Path(args.config_path)
    config = load_config(config_path)
    set_reproducible_seed(int(config["seed"]))
    
    output_dir = Path(args.output_dir)
    train_source = Path(args.train_source)
    
    print(f"[Subprocess] Loading annotated documents from {train_source}")
    annotated_documents = load_ner_training_documents(train_source)
    report = validate_documents(annotated_documents)
    if not report["is_valid"]:
        raise ValueError(f"Validation failed: {report['errors'][:10]}")
    
    manifest_path = train_source / "reports" / "dataset_manifest.jsonl"
    metadata = {}
    if manifest_path.exists():
        import json
        metadata = {str(item["document_id"]): item for item in (json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    records = [DatasetRecord(document_id=doc.document_id,
                              source_bucket=str(metadata.get(doc.document_id, {}).get("source_bucket", "unknown")),
                              template_group=str(metadata.get(doc.document_id, {}).get("template_group", doc.document_id)),
                              genre=str(metadata.get(doc.document_id, {}).get("genre", "unknown")),
                              long_tail=bool(metadata.get(doc.document_id, {}).get("long_tail", False)),
                              primary_surfaces=tuple(metadata.get(doc.document_id, {}).get("primary_surfaces", [])),
                              sha256=str(metadata.get(doc.document_id, {}).get("sha256", ""))) for doc in annotated_documents]
    stage_contract_manifest = None
    train_docs, val_docs, split_info = grouped_train_validation_split(annotated_documents, records, float(config["validation_fraction"]), int(config["seed"]))
    split_info["grouped"] = True
    
    if args.stage_manifest:
        selection = load_stage_selection(args.stage_manifest)
        if args.stage_name and selection["stage_name"] != args.stage_name:
            raise ValueError("stage-name does not match stage manifest")
        by_id = {document.document_id: document for document in annotated_documents}
        train_docs = [by_id[item] for item in selection["train_ids"] if item in by_id]
        val_docs = [by_id[item] for item in selection["validation_ids"] if item in by_id]
        split_info = {
            "stage_name": selection["stage_name"],
            "train_ids": list(selection["train_ids"]),
            "validation_ids": list(selection["validation_ids"]),
            "dataset_fingerprint": selection["dataset_fingerprint"],
            "split_fingerprint": selection["split_fingerprint"],
        }
        stage_contract_manifest = selection
    write_json(output_dir / "split_manifest.json", split_info)

    if fast_dev_run:
        train_docs = train_docs[: min(16, len(train_docs))]
        val_docs = val_docs[: min(4, len(val_docs))]
        ner_epochs = 1
        train_batch_size = 2
    else:
        ner_epochs = int(config["ner_epochs"])
        train_batch_size = int(config["batch_size"])
        
    learning_rate = float(config["learning_rate"])
    max_length = int(config.get("max_length", 512))
    stride = int(config.get("stride", 128))
    
    entity_types = {entity.type for document in train_docs for entity in document.entities}
    label_to_id, id_to_label = build_bio_label_map(entity_types)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_source, use_fast=True)
    from clinical_nlp_lab.records import parse_document_records

    train_records = {
        document.document_id: parse_document_records(document.document_id, document.raw_text, document.entities)
        for document in train_docs
    }
    validation_records = {
        document.document_id: parse_document_records(document.document_id, document.raw_text, document.entities)
        for document in val_docs
    }
    dataset_fingerprint = hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else "missing-manifest"
    train_contract = build_training_contract(
        train_docs,
        train_records,
        tokenizer,
        label_to_id,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=hashlib.sha256(json.dumps(split_info, sort_keys=True).encode("utf-8")).hexdigest(),
        max_length=max_length,
        stride=stride,
        batch_size=train_batch_size,
        curriculum_manifest=stage_contract_manifest,
    )
    validation_contract = build_training_contract(
        val_docs,
        validation_records,
        tokenizer,
        label_to_id,
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=hashlib.sha256(json.dumps(split_info, sort_keys=True).encode("utf-8")).hexdigest(),
        max_length=max_length,
        stride=stride,
        batch_size=train_batch_size,
        curriculum_manifest=stage_contract_manifest,
    )
    write_json(
        output_dir / "training_contract.json",
        {"train": train_contract.to_dict(), "validation": validation_contract.to_dict()},
    )
    from clinical_nlp_lab.collation import ClinicalTokenCollator
    owner_collator = ClinicalTokenCollator(pad_token_id=int(getattr(tokenizer, "pad_token_id", 1) or 1))

    def collate_owner_windows(examples):
        batch = owner_collator([example["window"] for example in examples])
        return {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"], "labels": batch["ner_labels"]}
    
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_source, num_labels=len(label_to_id), label2id=label_to_id, id2label=id_to_label
    )
    
    output_path = output_dir / "ner_model"
    output_path.mkdir(parents=True, exist_ok=True)
    
    grad_accum_steps = int(config.get("gradient_accumulation_steps", 4))
    
    training_kwargs = {
        "output_dir": str(output_path),
        "learning_rate": learning_rate,
        "num_train_epochs": ner_epochs,
        "per_device_train_batch_size": train_batch_size,
        "per_device_eval_batch_size": train_batch_size,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch" if validation_contract.windows else "no",
        "save_total_limit": 1,
        "load_best_model_at_end": bool(validation_contract.windows),
        "seed": int(config["seed"]),
        "report_to": [],
        "fp16": torch.cuda.is_available(),
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_checkpointing": True,
        "remove_unused_columns": False,
    }
    if validation_contract.windows:
        training_kwargs.update({
            "metric_for_best_model": "f1",
            "greater_is_better": True,
        })
    
    argument_parameters = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in argument_parameters else "evaluation_strategy"
    training_kwargs[evaluation_key] = "epoch" if validation_contract.windows else "no"
    arguments = TrainingArguments(**training_kwargs)
    
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=FeatureDataset(train_contract.windows),
        eval_dataset=FeatureDataset(validation_contract.windows) if validation_contract.windows else None,
        data_collator=collate_owner_windows,
        compute_metrics=compute_non_o_metrics if validation_contract.windows else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if validation_contract.windows else None,
    )
    trainer.processing_class = tokenizer
    
    print(f"[Subprocess] Starting Trainer.train() (epochs={ner_epochs}, batch={train_batch_size}, grad_accum={grad_accum_steps})")
    train_result = trainer.train()
    trainer.save_model(str(output_path))
    is_main_process = not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    if is_main_process:
        tokenizer.save_pretrained(str(output_path))
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
    removed_checkpoints = remove_nested_checkpoints(output_path) if is_main_process else []
    
    metrics = train_result.metrics
    ner_training_result = {
        "trained": True,
        "train_chunks": train_contract.window_count,
        "validation_chunks": validation_contract.window_count,
        "training_loss": metrics.get("train_loss", 0.0),
        "best_metric": trainer.state.best_metric,
        "removed_checkpoints": removed_checkpoints,
        "output_dir": str(output_path),
    }
    
    if is_main_process:
        write_json(output_dir / "training_result.json", ner_training_result)
    print(
        f"[TRAINING_END] trained=True epochs={ner_epochs} train_chunks={train_contract.window_count} "
        f"validation_chunks={validation_contract.window_count} best_metric={trainer.state.best_metric}",
        flush=True,
    )
    print("[Subprocess] Training finished. Script terminating to free VRAM natively.")

if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"[TRAINING_ERROR] {type(exc).__name__}: {exc}", flush=True)
        traceback.print_exc()
        raise
