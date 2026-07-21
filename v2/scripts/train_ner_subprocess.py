import argparse
import inspect
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from clinical_nlp_lab.config import load_config, set_reproducible_seed
from clinical_nlp_lab.data import document_train_validation_split, load_annotated_documents, validate_documents
from clinical_nlp_lab.schema import write_json
from clinical_nlp_lab.training import build_bio_label_map, prepare_token_classification_features

class FeatureDataset(Dataset):
    def __init__(self, features): self.features = features
    def __len__(self): return len(self.features)
    def __getitem__(self, index):
        return {k: v for k, v in self.features[index].items() if k not in {"offset_mapping", "document_id"}}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-source", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--config-path", type=str, required=True)
    parser.add_argument("--model-source", type=str, required=True)
    parser.add_argument("--fast-dev-run", type=str, default="False")
    args = parser.parse_args()

    fast_dev_run = args.fast_dev_run.lower() in ("true", "1", "yes")

    config_path = Path(args.config_path)
    config = load_config(config_path)
    set_reproducible_seed(int(config["seed"]))
    
    output_dir = Path(args.output_dir)
    train_source = Path(args.train_source)
    
    print(f"[Subprocess] Loading annotated documents from {train_source}")
    annotated_documents = load_annotated_documents(train_source)
    report = validate_documents(annotated_documents)
    if not report["is_valid"]:
        raise ValueError(f"Validation failed: {report['errors'][:10]}")
    
    train_docs, val_docs = document_train_validation_split(
        annotated_documents, float(config["validation_fraction"]), int(config["seed"])
    )
    
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
    train_features = prepare_token_classification_features(train_docs, tokenizer, label_to_id, max_length, stride)
    validation_features = prepare_token_classification_features(val_docs, tokenizer, label_to_id, max_length, stride)
    
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
        "save_strategy": "epoch" if validation_features else "no",
        "save_total_limit": 1,
        "load_best_model_at_end": bool(validation_features),
        "seed": int(config["seed"]),
        "report_to": [],
        "fp16": torch.cuda.is_available(),
        "gradient_accumulation_steps": grad_accum_steps,
        "gradient_checkpointing": True,
    }
    
    argument_parameters = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in argument_parameters else "evaluation_strategy"
    training_kwargs[evaluation_key] = "epoch" if validation_features else "no"
    arguments = TrainingArguments(**training_kwargs)
    
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=FeatureDataset(train_features),
        eval_dataset=FeatureDataset(validation_features) if validation_features else None,
        data_collator=DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8),
    )
    trainer.processing_class = tokenizer
    
    print(f"[Subprocess] Starting Trainer.train() (epochs={ner_epochs}, batch={train_batch_size}, grad_accum={grad_accum_steps})")
    train_result = trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    
    metrics = train_result.metrics
    ner_training_result = {
        "trained": True,
        "train_chunks": len(train_features),
        "validation_chunks": len(validation_features),
        "training_loss": metrics.get("train_loss", 0.0),
        "output_dir": str(output_path),
    }
    
    write_json(output_dir / "training_result.json", ner_training_result)
    print("[Subprocess] Training finished. Script terminating to free VRAM natively.")

if __name__ == "__main__":
    main()
