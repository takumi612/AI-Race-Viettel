"""Optional local Transformers NER with precision-gated hybrid merging."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.training.fingerprints import fingerprint_files
from src.training.ner.bio import decode_bio_entities, merge_decoded_entities


class _TransformersPredictor:
    def __init__(self, artifact_dir: Path):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        model_dir = artifact_dir / "final"
        if not model_dir.is_dir():
            model_dir = artifact_dir
        config_path = artifact_dir / "resolved_config.json"
        config = (
            json.loads(config_path.read_text(encoding="utf-8"))
            if config_path.is_file()
            else {}
        )
        self.max_length = int(config.get("max_length", 384))
        self.stride = int(config.get("stride", 64))
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_dir,
            use_fast=True,
            local_files_only=True,
        )
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()

    def predict(self, text: str) -> list[dict[str, Any]]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            padding=True,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            return_tensors="pt",
        )
        offsets = encoded.pop("offset_mapping").tolist()
        encoded.pop("overflow_to_sample_mapping", None)
        model_inputs = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.inference_mode():
            logits = self.model(**model_inputs).logits
            probabilities = logits.softmax(dim=-1)
            confidences, label_ids = probabilities.max(dim=-1)
        return [
            {
                "offset_mapping": window_offsets,
                "label_ids": window_labels,
                "confidences": window_confidences,
            }
            for window_offsets, window_labels, window_confidences in zip(
                offsets,
                label_ids.cpu().tolist(),
                confidences.cpu().tolist(),
            )
        ]


def _validate_artifact(path: Path) -> None:
    manifest_path = path / "artifact_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"NER artifact manifest is missing: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("status") not in {"validated", "locked"}
        or not isinstance(manifest.get("run"), dict)
        or manifest["run"].get("task") != "ner"
    ):
        raise ValueError("NER artifact is not validated for task=ner")
    files = sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.name != "artifact_manifest.json"
    )
    if fingerprint_files(files, path) != manifest.get("artifact_sha256"):
        raise ValueError("NER artifact fingerprint mismatch")


class ModelNERExtractor:
    def __init__(
        self,
        artifact_dir: str | Path | None = None,
        *,
        predictor: Any | None = None,
        threshold: float = 0.70,
    ):
        if (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 <= threshold <= 1
        ):
            raise ValueError("NER model threshold must be in [0, 1]")
        if predictor is None:
            if artifact_dir is None:
                raise ValueError("artifact_dir is required without an injected predictor")
            path = Path(artifact_dir).resolve()
            _validate_artifact(path)
            predictor = _TransformersPredictor(path)
        if not callable(getattr(predictor, "predict", None)):
            raise ValueError("NER predictor must provide predict(text)")
        self.predictor = predictor
        self.threshold = float(threshold)

    def extract_entities(self, text: str, chunks=None) -> tuple[dict[str, Any], ...]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        decoded: list[Mapping[str, Any]] = []
        for window in self.predictor.predict(text):
            decoded.extend(
                decode_bio_entities(
                    text,
                    window["offset_mapping"],
                    window["label_ids"],
                    confidences=window.get("confidences"),
                )
            )
        return tuple(
            entity
            for entity in merge_decoded_entities(decoded)
            if float(entity.get("confidence", 0.0)) >= self.threshold
        )


def merge_hybrid_entities(
    text: str,
    rule_entities: Sequence[Mapping[str, Any]],
    model_entities: Sequence[Mapping[str, Any]],
    *,
    default_threshold: float,
    per_type_thresholds: Mapping[str, float],
) -> tuple[dict[str, Any], ...]:
    merged: dict[tuple[str, int, int], dict[str, Any]] = {}
    for entity in rule_entities:
        start, end = entity["position"]
        if text[start:end] != entity["text"]:
            raise ValueError("rule entity is not an exact document slice")
        merged[(entity["type"], start, end)] = dict(entity)
    for entity in model_entities:
        start, end = entity["position"]
        if text[start:end] != entity["text"]:
            raise ValueError("model entity is not an exact document slice")
        threshold = per_type_thresholds.get(entity["type"], default_threshold)
        if float(entity.get("confidence", 0.0)) < threshold:
            continue
        key = (entity["type"], start, end)
        merged.setdefault(key, dict(entity))
    return tuple(
        merged[key]
        for key in sorted(merged, key=lambda value: (value[1], value[2], value[0]))
    )
