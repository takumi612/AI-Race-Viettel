from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import Any, Iterable, Sequence

from .schema import ClinicalDocument, EntityAnnotation


def compute_non_o_metrics(eval_prediction: Any) -> dict[str, float]:
    """Compute token metrics while ignoring padding and treating label 0 as O."""
    import numpy as np

    logits, labels = eval_prediction
    predictions = np.asarray(logits).argmax(axis=-1)
    labels = np.asarray(labels)
    valid = labels != -100
    y_true = labels[valid]
    y_pred = predictions[valid]
    if y_true.size == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}
    true_positive = int(((y_pred == y_true) & (y_true != 0)).sum())
    predicted_positive = int((y_pred != 0).sum())
    actual_positive = int((y_true != 0).sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = float((y_pred == y_true).mean())
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def compute_entity_metrics(expected_documents, predicted_documents):
    """Return exact-span and type-matched overlap micro metrics."""
    def spans(docs):
        return [(doc_id, e.start, e.end, e.type) for doc_id, entities in docs.items() for e in entities]
    gold = spans(expected_documents)
    pred = spans(predicted_documents)
    exact_tp = len(set(gold) & set(pred))
    overlap_tp = 0
    used: set[int] = set()
    for doc_id, ps, pe, pt in pred:
        for idx, (gd, gs, ge, gt) in enumerate(gold):
            if idx in used or doc_id != gd or pt != gt:
                continue
            if max(ps, gs) < min(pe, ge):
                overlap_tp += 1
                used.add(idx)
                break
    def score(tp: int, p: int, g: int):
        precision = tp / p if p else 0.0
        recall = tp / g if g else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return precision, recall, f1
    ep, er, ef = score(exact_tp, len(pred), len(gold))
    op, ors, of = score(overlap_tp, len(pred), len(gold))
    return {"exact_precision": ep, "exact_recall": er, "exact_f1": ef,
            "overlap_precision": op, "overlap_recall": ors, "overlap_f1": of,
            "gold_entities": len(gold), "predicted_entities": len(pred)}


def remove_nested_checkpoints(model_dir: str | Path) -> list[str]:
    """Remove duplicate Trainer checkpoint directories after the final model is saved."""
    root = Path(model_dir).resolve()
    removed: list[str] = []
    for checkpoint in sorted(root.glob("checkpoint-*")):
        if checkpoint.is_dir() and checkpoint.parent.resolve() == root:
            shutil.rmtree(checkpoint)
            removed.append(checkpoint.name)
    return removed


@dataclass(slots=True)
class TrainingAvailability:
    available: bool
    missing_packages: list[str]
    reason: str


def chunk_token_indices(token_count: int, max_length: int = 512, stride: int = 128) -> list[tuple[int, int]]:
    """Return overlapping token windows; split is always called per document."""
    if token_count < 0 or max_length <= 0 or stride < 0 or stride >= max_length:
        raise ValueError("Require token_count >= 0, max_length > stride >= 0")
    if token_count == 0:
        return []
    windows: list[tuple[int, int]] = []
    start = 0
    while start < token_count:
        end = min(token_count, start + max_length)
        windows.append((start, end))
        if end >= token_count:
            break
        start = end - stride
    return windows


def transformer_training_availability() -> TrainingAvailability:
    missing: list[str] = []
    for package in ("torch", "transformers", "accelerate"):
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        return TrainingAvailability(False, missing, f"Missing optional packages: {', '.join(missing)}")
    return TrainingAvailability(True, [], "Transformer training dependencies are available")


def build_bio_label_map(entity_types: Iterable[str]) -> tuple[dict[str, int], dict[int, str]]:
    labels = ["O"]
    for entity_type in sorted(set(entity_types)):
        labels.extend([f"B-{entity_type}", f"I-{entity_type}"])
    label_to_id = {label: index for index, label in enumerate(labels)}
    return label_to_id, {index: label for label, index in label_to_id.items()}


def character_spans_to_bio(
    offsets: Sequence[tuple[int, int]],
    entities: Iterable[EntityAnnotation],
    label_to_id: dict[str, int],
    special_token_label: int = -100,
) -> list[int]:
    entity_list = sorted(entities, key=lambda item: (item.start, item.end))
    labels: list[int] = []
    previous_entity: EntityAnnotation | None = None
    for token_start, token_end in offsets:
        if token_start == token_end:
            labels.append(special_token_label)
            continue
        matching = [
            entity
            for entity in entity_list
            if token_start < entity.end and entity.start < token_end
        ]
        if not matching:
            labels.append(label_to_id["O"])
            previous_entity = None
            continue
        entity = matching[0]
        prefix = "I" if previous_entity is entity else "B"
        label = f"{prefix}-{entity.type}"
        if label not in label_to_id:
            raise ValueError(f"Missing BIO label: {label}")
        labels.append(label_to_id[label])
        previous_entity = entity
    return labels


def bio_predictions_to_spans(
    label_ids: Sequence[int],
    offsets: Sequence[tuple[int, int]],
    id_to_label: dict[int, str],
    raw_text: str,
    confidences: Sequence[float] | None = None,
) -> list[EntityAnnotation]:
    spans: list[EntityAnnotation] = []
    current_type: str | None = None
    current_start: int | None = None
    current_end: int | None = None
    current_confidences: list[float] = []

    def flush() -> None:
        nonlocal current_type, current_start, current_end, current_confidences
        if current_type is not None and current_start is not None and current_end is not None:
            entity = EntityAnnotation(
                text=raw_text[current_start:current_end],
                type=current_type,
                position=(current_start, current_end),
                confidence=sum(current_confidences) / len(current_confidences) if current_confidences else 1.0,
                evidence=["transformer_bio"],
            )
            entity.validate_offset(raw_text)
            spans.append(entity)
        current_type = None
        current_start = None
        current_end = None
        current_confidences = []

    for index, (label_id, (start, end)) in enumerate(zip(label_ids, offsets)):
        if start == end or label_id == -100:
            continue
        label = id_to_label.get(int(label_id), "O")
        if label == "O":
            flush()
            continue
        prefix, entity_type = label.split("-", 1)
        if prefix == "B" or entity_type != current_type:
            flush()
            current_type = entity_type
            current_start = start
            current_end = end
        else:
            current_end = max(current_end or end, end)
        if confidences is not None:
            current_confidences.append(float(confidences[index]))
    flush()
    return spans


def prepare_token_classification_features(
    documents: Sequence[ClinicalDocument],
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_length: int,
    stride: int,
) -> list[dict[str, Any]]:
    features: list[dict[str, Any]] = []
    for document in documents:
        encoded = tokenizer(
            document.raw_text,
            truncation=True,
            max_length=max_length,
            stride=stride,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
        )
        for chunk_index, offsets in enumerate(encoded["offset_mapping"]):
            feature = {
                key: encoded[key][chunk_index]
                for key in encoded
                if key not in {"offset_mapping", "overflow_to_sample_mapping"}
            }
            feature["labels"] = character_spans_to_bio(offsets, document.entities, label_to_id)
            feature["offset_mapping"] = offsets
            feature["document_id"] = document.document_id
            features.append(feature)
    return features


def train_transformer_ner(
    train_documents: Sequence[ClinicalDocument],
    validation_documents: Sequence[ClinicalDocument],
    output_dir: str | Path,
    model_name: str = "xlm-roberta-base",
    max_length: int = 512,
    stride: int = 128,
    learning_rate: float = 2e-5,
    epochs: int = 3,
    batch_size: int = 8,
    gradient_accumulation_steps: int = 1,
    seed: int = 42,
) -> dict[str, Any]:
    if not train_documents:
        return {"trained": False, "reason": "No annotated training documents were provided"}
    availability = transformer_training_availability()
    if not availability.available:
        return {"trained": False, "reason": availability.reason, "missing_packages": availability.missing_packages}

    import inspect
    import torch
    from torch.utils.data import Dataset
    from transformers import (
        AutoModelForTokenClassification,
        AutoTokenizer,
        DataCollatorForTokenClassification,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    entity_types = {entity.type for document in train_documents for entity in document.entities}
    label_to_id, id_to_label = build_bio_label_map(entity_types)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    train_features = prepare_token_classification_features(
        train_documents, tokenizer, label_to_id, max_length, stride
    )
    validation_features = prepare_token_classification_features(
        validation_documents, tokenizer, label_to_id, max_length, stride
    )

    class FeatureDataset(Dataset):
        def __init__(self, features: list[dict[str, Any]]) -> None:
            self.features = features

        def __len__(self) -> int:
            return len(self.features)

        def __getitem__(self, index: int) -> dict[str, Any]:
            return {
                key: value
                for key, value in self.features[index].items()
                if key not in {"offset_mapping", "document_id"}
            }

    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        label2id=label_to_id,
        id2label=id_to_label,
    )
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    training_kwargs = {
        "output_dir": str(output_path),
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "save_strategy": "epoch" if validation_features else "no",
        "save_total_limit": 1,
        "load_best_model_at_end": bool(validation_features),
        "seed": seed,
        "report_to": [],
        "fp16": torch.cuda.is_available(),
        "gradient_accumulation_steps": max(1, int(gradient_accumulation_steps)),
        "gradient_checkpointing": bool(torch.cuda.is_available()),
        "metric_for_best_model": "f1" if validation_features else None,
        "greater_is_better": True if validation_features else None,
    }
    if not validation_features:
        training_kwargs.pop("metric_for_best_model", None)
        training_kwargs.pop("greater_is_better", None)
    # Transformers renamed evaluation_strategy to eval_strategy in newer releases.
    argument_parameters = inspect.signature(TrainingArguments.__init__).parameters
    evaluation_key = "eval_strategy" if "eval_strategy" in argument_parameters else "evaluation_strategy"
    training_kwargs[evaluation_key] = "epoch" if validation_features else "no"
    arguments = TrainingArguments(**training_kwargs)
    trainer_kwargs = {
        "model": model,
        "args": arguments,
        "train_dataset": FeatureDataset(train_features),
        "eval_dataset": FeatureDataset(validation_features) if validation_features else None,
        "data_collator": DataCollatorForTokenClassification(tokenizer),
        "compute_metrics": compute_non_o_metrics if validation_features else None,
    }
    # `processing_class` replaced the older `tokenizer` Trainer argument.
    trainer_parameters = inspect.signature(Trainer.__init__).parameters
    trainer_kwargs["processing_class" if "processing_class" in trainer_parameters else "tokenizer"] = tokenizer
    if validation_features:
        trainer_kwargs["callbacks"] = [EarlyStoppingCallback(early_stopping_patience=2)]
    trainer = Trainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    removed_checkpoints = remove_nested_checkpoints(output_path)
    evaluation = trainer.evaluate() if validation_features else {}
    return {
        "trained": True,
        "train_documents": len(train_documents),
        "validation_documents": len(validation_documents),
        "train_chunks": len(train_features),
        "validation_chunks": len(validation_features),
        "label_to_id": label_to_id,
        "training_loss": float(train_result.training_loss),
        "evaluation": {key: float(value) for key, value in evaluation.items() if isinstance(value, (int, float))},
        "best_metric": trainer.state.best_metric,
        "removed_checkpoints": removed_checkpoints,
        "output_dir": str(output_path),
    }


def build_multitask_assertion_model(model_name: str, head_sizes: dict[str, int]) -> Any:
    availability = transformer_training_availability()
    if not availability.available:
        raise RuntimeError(availability.reason)
    import torch
    from torch import nn
    from transformers import AutoModel

    class MultiTaskAssertionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = AutoModel.from_pretrained(model_name)
            hidden_size = int(self.encoder.config.hidden_size)
            self.dropout = nn.Dropout(0.1)
            self.heads = nn.ModuleDict(
                {name: nn.Linear(hidden_size, size) for name, size in head_sizes.items()}
            )

        def forward(self, input_ids: Any, attention_mask: Any = None) -> dict[str, Any]:
            outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
            pooled = outputs.last_hidden_state[:, 0]
            pooled = self.dropout(pooled)
            return {name: head(pooled) for name, head in self.heads.items()}

    return MultiTaskAssertionModel()


def build_assertion_examples(documents: Iterable[ClinicalDocument], context_window: int = 160) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for document in documents:
        for entity in document.entities:
            left = max(0, entity.start - context_window)
            right = min(len(document.raw_text), entity.end + context_window)
            local_start = entity.start - left
            local_end = entity.end - left
            context = document.raw_text[left:right]
            marked = context[:local_start] + "<ENT>" + context[local_start:local_end] + "</ENT>" + context[local_end:]
            examples.append(
                {
                    "document_id": document.document_id,
                    "entity_type": entity.type,
                    "text": marked,
                    "assertions": list(entity.assertions),
                }
            )
    return examples


def build_relation_examples(documents: Iterable[ClinicalDocument], max_distance: int = 256) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for document in documents:
        relation_lookup = {
            (tuple(item.get("subject", [])), tuple(item.get("object", []))): item.get("type")
            for item in document.relations
        }
        for subject in document.entities:
            for obj in document.entities:
                if subject is obj:
                    continue
                distance = max(0, obj.start - subject.end, subject.start - obj.end)
                if distance > max_distance:
                    continue
                key = ((subject.start, subject.end, subject.type), (obj.start, obj.end, obj.type))
                examples.append(
                    {
                        "document_id": document.document_id,
                        "subject": list(key[0]),
                        "object": list(key[1]),
                        "label": relation_lookup.get(key, "NO_RELATION"),
                        "distance": distance,
                    }
                )
    return examples
