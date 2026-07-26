"""Concrete Kaggle phase runners.

The notebook only presents these runners.  All data/model work lives here so a
phase can be tested with a fake runner locally and executed with real weights on
Kaggle.  No function in this module is called by the local test suite with a
training backend.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping
import zipfile

from .artifacts import initialize_runtime_artifacts, inventory_files, sha256_file, write_json
from .config import load_config
from .curriculum import build_stage_manifest, plan_curriculum
from .data import load_ner_training_documents
from .dataset_quality import DatasetRecord
from .orchestration import PHASES, PhaseRunner, RunConfig
from .preflight import build_preflight_report
from .records import build_record_metadata, parse_document_records
from .splitting import (
    build_near_duplicate_groups,
    build_split_plan,
    metadata_artifact_payloads,
    near_duplicate_edges_payload,
)
from .training import build_bio_label_map, build_training_contract
from .candidate_training import ScoredCandidate, build_candidate_training_artifact
from .candidate_policy import CandidatePolicy
from .assertion_model import build_frozen_assertion_adapter, fit_assertion_thresholds
from .kb import load_candidate_dictionary


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _run_dir(context: Mapping[str, Any]) -> Path:
    return Path(str(context["run_dir"]))


def _artifact_dir(config: RunConfig) -> Path:
    return Path(config.artifact_dir)


def _hash_tree(root: Path) -> str:
    digest = hashlib.sha256()
    if not root.exists():
        return digest.hexdigest()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _hash_named_files(root: Path, names: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    found = False
    for name in names:
        path = root / name
        if path.is_file():
            found = True
            digest.update(name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest() if found else _hash_tree(root)


def _phase_01_preflight(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    if config.run_mode == "full" and config.fast_dev_run:
        raise ValueError("Cannot execute full production run when fast_dev_run is True. Set fast_dev_run=False in RunConfig.")
    artifact_dir = _artifact_dir(config)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    if config.artifact_source_dir is not None:
        source_dir = Path(config.artifact_source_dir)
        required_kb = artifact_dir / "icd10" / "icd10_dictionary.jsonl.gz"
        source_kb = source_dir / "icd10" / "icd10_dictionary.jsonl.gz"
        if not required_kb.is_file() and source_kb.is_file():
            shutil.copytree(source_dir, artifact_dir, dirs_exist_ok=True)
    runtime_config = Path(config.config_path) if config.config_path else artifact_dir / "config.json"
    if not runtime_config.exists():
        initialize_runtime_artifacts(load_config(None), artifact_dir)
    report_path = _run_dir(context) / "artifacts" / "preflight_report.json"
    report = build_preflight_report(config.dataset_root, artifact_dir, runtime_config, report_path)
    if report["status"] != "PASS":
        raise RuntimeError(f"dataset preflight failed with {len(report['errors'])} errors")
    return {
        "phase": phase,
        "status": report["status"],
        "dataset_fingerprint": report["dataset_pair_fingerprint"],
        "config_fingerprint": report["config_sha256"],
        "kb_hashes": report["kb_hashes"],
        "report": str(report_path),
    }


def _phase_02_resolve_sources(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    dataset_root = Path(config.dataset_root).resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"DATASET_ROOT does not exist: {dataset_root}")
    input_source = Path(config.input_source)
    if not input_source.is_absolute():
        candidates = (Path.cwd() / input_source, dataset_root.parent / input_source)
        input_source = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    if not input_source.exists():
        raise FileNotFoundError(f"inference input source does not exist: {input_source}")
    return {
        "phase": phase,
        "dataset_root": str(dataset_root),
        "input_source": str(input_source.resolve()),
        "model_source": str(config.model_source),
        "network_expected": True,
    }


def _phase_03_inventory_models(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("Kaggle runtime requires torch") from exc
    gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    if gpu_count < int(config.expected_gpu_count):
        raise RuntimeError(
            f"expected at least {config.expected_gpu_count} CUDA devices, found {gpu_count}"
        )
    return {
        "phase": phase,
        "gpu_count": gpu_count,
        "devices": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "distributed": bool(config.use_distributed and gpu_count > 1),
        "model_source": str(config.model_source),
    }


def _phase_04_build_metadata(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    records = build_record_metadata(config.dataset_root)
    near = build_near_duplicate_groups(config.dataset_root)
    manifest_bytes, edges_bytes = metadata_artifact_payloads(records, near)
    out = _run_dir(context) / "artifacts" / "metadata"
    _atomic_bytes(out / "metadata_manifest.jsonl", manifest_bytes)
    _atomic_bytes(out / "near_duplicate_edges.json", edges_bytes)
    descriptor = {
        "dataset_fingerprint": records.dataset_fingerprint,
        "manifest_sha256": records.manifest_sha256,
        "metadata_sha256": records.metadata_sha256,
        "record_count": records.record_count,
        "document_count": len(records.rows),
        "near_duplicate_edge_count": len(near.edges),
    }
    _atomic_bytes(out / "metadata_descriptor.json", _json_bytes(descriptor))
    return {"phase": phase, "metadata_dir": str(out), **descriptor}


def _phase_05_build_splits(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    records = build_record_metadata(config.dataset_root)
    near = build_near_duplicate_groups(config.dataset_root)
    out = _run_dir(context) / "artifacts" / "splits"
    out.mkdir(parents=True, exist_ok=True)
    fixed = build_split_plan(config.dataset_root, seed=config.seed, eval_profile="fixed_fold", record_metadata=records, near_duplicates=near)
    _atomic_bytes(out / "split_fixed_fold.json", fixed.manifest_bytes)
    oof_hashes: list[str] = []
    for fold in range(5):
        plan = build_split_plan(config.dataset_root, seed=config.seed, eval_profile="oof_extended", fold_index=fold, record_metadata=records, near_duplicates=near)
        _atomic_bytes(out / f"split_oof_fold_{fold}.json", plan.manifest_bytes)
        oof_hashes.append(plan.manifest_sha256)
    descriptor = {
        "phase": phase,
        "split_dir": str(out),
        "dataset_fingerprint": records.dataset_fingerprint,
        "fixed_split_sha256": fixed.manifest_sha256,
        "oof_split_sha256": oof_hashes,
        "fixed_partitions": fixed.manifest["partitions"],
    }
    _atomic_bytes(out / "split_descriptor.json", _json_bytes(descriptor))
    return descriptor


def _phase_06_prepare_training_contract(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Kaggle runtime requires transformers") from exc
    documents = load_ner_training_documents(config.dataset_root)
    label_to_id, _ = build_bio_label_map({entity.type for document in documents for entity in document.entities})
    tokenizer = AutoTokenizer.from_pretrained(str(config.model_source), use_fast=True)
    sample = documents[: min(32, len(documents))]
    records_by_document = {
        document.document_id: parse_document_records(document.document_id, document.raw_text, document.entities)
        for document in sample
    }
    contract = build_training_contract(
        sample,
        records_by_document,
        tokenizer,
        label_to_id,
        dataset_fingerprint=context.get("dataset_fingerprint", "unbound"),
        split_fingerprint="prepared-contract",
        max_length=512,
        stride=128,
        batch_size=8,
    )
    path = _run_dir(context) / "artifacts" / "training_contract.json"
    _atomic_bytes(path, _json_bytes(contract.to_dict()))
    return {
        "phase": phase,
        "contract": str(path),
        "sample_document_count": len(sample),
        "sample_window_count": contract.window_count,
        "label_map": label_to_id,
    }


def _stage_manifest_path(config: RunConfig, context: Mapping[str, Any], stage_name: str) -> Path:
    return _run_dir(context) / "artifacts" / "stage_inputs" / f"{stage_name}.json"


def _write_stage_input(config: RunConfig, context: Mapping[str, Any], stage_name: str) -> Path:
    from .sampling import select_stage_document_ids

    split_payload = json.loads((_run_dir(context) / "artifacts" / "splits" / "split_descriptor.json").read_text(encoding="utf-8"))
    partitions = split_payload["fixed_partitions"]
    synthetic_train = list(partitions["synthetic_train_ids"])
    synthetic_validation = list(partitions["synthetic_validation_ids"])
    organizer_train = list(partitions["organizer_train_ids"])
    organizer_validation = list(partitions["organizer_validation_ids"])
    stage_spec = next(spec for spec in plan_curriculum("full") if spec.name == stage_name)
    if stage_name == "stage1":
        train_ids = list(
            select_stage_document_ids(
                synthetic_train, organizer_train, stage_spec, seed=42
            )
        )
        validation_ids = synthetic_validation
    elif stage_name in {"stage2", "stage3"}:
        train_ids = list(
            select_stage_document_ids(
                synthetic_train, organizer_train, stage_spec, seed=42
            )
        )
        validation_ids = synthetic_validation + organizer_validation
    else:
        train_ids = list(
            select_stage_document_ids(
                synthetic_train,
                organizer_train,
                stage_spec,
                seed=42,
            )
        )
        validation_ids = synthetic_validation + organizer_validation
    payload = {
        "schema_id": "clinical_nlp.kaggle_stage_input",
        "schema_version": 1,
        "stage_name": stage_name,
        "dataset_root": str(Path(config.dataset_root).resolve()),
        "train_ids": train_ids,
        "validation_ids": sorted(set(validation_ids), key=lambda value: int(value)),
        "stage_spec": stage_spec.to_dict(),
        "dataset_fingerprint": split_payload["dataset_fingerprint"],
        "split_fingerprint": split_payload["fixed_split_sha256"],
    }
    path = _stage_manifest_path(config, context, stage_name)
    _atomic_bytes(path, _json_bytes(payload))
    return path


def _build_training_command(
    config: RunConfig,
    script: Path,
    args: list[str],
    *,
    gpu_count: int,
) -> list[str]:
    if config.use_distributed and gpu_count > 1:
        return [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node",
            str(gpu_count),
            str(script),
            *args,
        ]
    return [sys.executable, str(script), *args]


def read_training_stage_result(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / "training_result.json"
    if not path.is_file():
        raise FileNotFoundError(f"training stage result missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "trained",
        "train_chunks",
        "validation_chunks",
        "configured_epochs",
        "learning_rate",
        "batch_size",
        "training_loss",
        "best_metric",
        "best_checkpoint",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"training stage result missing keys: {missing}")
    return payload


def _run_training_stage(config: RunConfig, phase: str, context: Mapping[str, Any], stage_name: str, parent_checkpoint: str | None) -> Mapping[str, Any]:
    stage_input = _write_stage_input(config, context, stage_name)
    run_dir = _run_dir(context)
    output_dir = run_dir / "checkpoints" / stage_name
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parents[1] / "scripts" / "train_ner_subprocess.py"
    model_source = parent_checkpoint or str(config.model_source)
    args = [
        "--train-source", str(config.dataset_root),
        "--output-dir", str(output_dir),
        "--config-path", str(config.config_path or (_artifact_dir(config) / "config.json")),
        "--model-source", model_source,
        "--stage-manifest", str(stage_input),
        "--stage-name", stage_name,
        "--fast-dev-run", str(config.fast_dev_run),
    ]
    try:
        import torch
        gpu_count = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except ImportError:
        gpu_count = 0
    command = _build_training_command(config, script, args, gpu_count=gpu_count)
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    log_path = run_dir / "artifacts" / f"{stage_name}_training.log"
    _atomic_bytes(log_path, (completed.stdout + "\n" + completed.stderr).encode("utf-8", errors="replace"))
    if completed.returncode != 0:
        raise RuntimeError(f"{stage_name} training failed with exit code {completed.returncode}; see {log_path}")
    checkpoint_dir = output_dir / "ner_model"
    checkpoint_hash = _hash_tree(checkpoint_dir)
    if not checkpoint_hash or not (checkpoint_dir / "config.json").is_file():
        raise RuntimeError(f"{stage_name} did not publish a valid checkpoint")
    stage_spec = next(spec for spec in plan_curriculum("full") if spec.name == stage_name)
    manifest = build_stage_manifest(stage_spec, {"dataset": json.loads(stage_input.read_text(encoding="utf-8"))["dataset_fingerprint"], "split": json.loads(stage_input.read_text(encoding="utf-8"))["split_fingerprint"]}, checkpoint_hash)
    manifest_path = output_dir / "stage_manifest.json"
    write_json(manifest_path, manifest.to_dict())
    training_result = read_training_stage_result(output_dir)
    return {
        "phase": phase,
        "stage": stage_name,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_sha256": checkpoint_hash,
        "log": str(log_path),
        "stage_manifest": str(manifest_path),
        "training": training_result,
    }


def _phase_07_stage1(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    return _run_training_stage(config, phase, context, "stage1", None)


def _phase_08_stage2(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = _run_dir(context) / "checkpoints" / "stage1" / "ner_model"
    return _run_training_stage(config, phase, context, "stage2", str(parent))


def _phase_09_stage3(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = _run_dir(context) / "checkpoints" / "stage2" / "ner_model"
    return _run_training_stage(config, phase, context, "stage3", str(parent))


def _phase_10_final_fit(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    parent = _run_dir(context) / "checkpoints" / "stage3" / "ner_model"
    return _run_training_stage(config, phase, context, "final_fit", str(parent))


def _phase_11_fit_heads(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn.functional as F
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("phase_11_fit_heads requires torch, transformers and numpy") from exc

    run_dir = _run_dir(context)
    checkpoint = run_dir / "checkpoints" / "final_fit" / "ner_model"
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"final-fit checkpoint missing: {checkpoint}")
    from .assertion_model import split_assertion_documents
    from .entity_types import ENTITY_TYPE_TO_ID

    documents = load_ner_training_documents(config.dataset_root)
    split_payload = json.loads(
        (_run_dir(context) / "artifacts" / "splits" / "split_descriptor.json").read_text(
            encoding="utf-8"
        )
    )
    fixed_partitions = split_payload["fixed_partitions"]
    validation_ids = set(fixed_partitions["synthetic_validation_ids"]) | set(
        fixed_partitions["organizer_validation_ids"]
    )
    train_documents, validation_documents = split_assertion_documents(
        documents, validation_ids
    )
    if config.fast_dev_run:
        train_documents = train_documents[:32]
        validation_documents = validation_documents[:8]
    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint), use_fast=True)
    label_to_id, _ = build_bio_label_map({entity.type for document in documents for entity in document.entities})
    train_records = {
        document.document_id: parse_document_records(document.document_id, document.raw_text, document.entities)
        for document in train_documents
    }
    validation_records = {
        document.document_id: parse_document_records(document.document_id, document.raw_text, document.entities)
        for document in validation_documents
    }
    encoder = AutoModel.from_pretrained(str(checkpoint))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder_hash = _hash_tree(checkpoint)
    tokenizer_hash = _hash_named_files(
        checkpoint,
        ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "sentencepiece.bpe.model"),
    )
    adapter = build_frozen_assertion_adapter(
        encoder,
        hidden_dim=int(encoder.config.hidden_size),
        encoder_hash=encoder_hash,
        tokenizer_hash=tokenizer_hash,
    ).to(device)
    train_contract = build_training_contract(
        train_documents,
        train_records,
        tokenizer,
        label_to_id,
        dataset_fingerprint=context.get("dataset_fingerprint", "unbound"),
        split_fingerprint="final-fit",
        max_length=512,
        stride=128,
        batch_size=8,
    )
    validation_contract = build_training_contract(
        validation_documents,
        validation_records,
        tokenizer,
        label_to_id,
        dataset_fingerprint=context.get("dataset_fingerprint", "unbound"),
        split_fingerprint="assertion-validation",
        max_length=512,
        stride=128,
        batch_size=8,
    )
    optimizer = torch.optim.AdamW(adapter.head.parameters(), lr=1e-3)
    epochs = 1 if config.fast_dev_run else 3
    for _epoch in range(epochs):
        adapter.train()
        for batch in train_contract.batches:
            spans = batch["entity_spans"].to(device)
            types = batch["entity_types"].to(device)
            if spans.numel() == 0:
                continue
            logits = adapter(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                spans,
                types,
            )
            targets = batch["assertion_targets"].to(device)
            mask = batch["assertion_mask"].to(device)
            loss_values = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            loss = (loss_values * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    collected_logits: list[np.ndarray] = []
    collected_targets: list[np.ndarray] = []
    collected_masks: list[np.ndarray] = []
    adapter.eval()
    with torch.inference_mode():
        for batch in validation_contract.batches:
            spans = batch["entity_spans"].to(device)
            if spans.numel() == 0:
                continue
            logits = adapter(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                spans,
                batch["entity_types"].to(device),
            )
            collected_logits.append(logits.cpu().numpy())
            collected_targets.append(batch["assertion_targets"].numpy())
            collected_masks.append(batch["assertion_mask"].numpy())
    if not collected_logits:
        raise RuntimeError("assertion validation produced no calibration mentions")
    logits_array = np.concatenate(collected_logits)
    targets_array = np.concatenate(collected_targets)
    masks_array = np.concatenate(collected_masks)
    head_dir = run_dir / "artifacts" / "heads"
    head_dir.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.head.state_dict(), head_dir / "assertion_head.pt")
    write_json(head_dir / "assertion_binding.json", adapter.binding.to_dict())
    thresholds = fit_assertion_thresholds(
        logits_array,
        targets_array,
        masks_array,
        encoder_hash=encoder_hash,
        tokenizer_hash=tokenizer_hash,
    )
    write_json(head_dir / "assertion_thresholds.json", thresholds.to_dict())
    write_json(
        head_dir / "assertion_entity_type_map.json",
        ENTITY_TYPE_TO_ID,
    )

    kb_candidates: dict[str, list[dict[str, Any]]] = {}
    kb_hash_parts: list[str] = []
    for system, relative in (
        ("ICD10", Path("icd10") / "icd10_dictionary.jsonl.gz"),
        ("RXNORM", Path("rxnorm") / "rxnorm_dictionary.jsonl.gz"),
    ):
        path = _artifact_dir(config) / relative
        if not path.is_file():
            raise FileNotFoundError(f"runtime KB missing: {path}")
        kb_hash_parts.append(sha256_file(path))
        for record in load_candidate_dictionary(path):
            kb_candidates.setdefault(system, []).append(record)
    kb_hash = hashlib.sha256("".join(kb_hash_parts).encode("ascii")).hexdigest()
    positives: list[dict[str, Any]] = []
    retrieved: list[list[dict[str, Any]]] = []
    scored: list[ScoredCandidate] = []
    for document in documents:
        for entity in document.entities:
            if not entity.candidates or entity.type not in {"DISEASE", "DRUG"}:
                continue
            system = "ICD10" if entity.type == "DISEASE" else "RXNORM"
            positive_code = str(entity.candidates[0])
            records = kb_candidates[system]
            positive_record = next((record for record in records if str(record.get("candidate_id")) == positive_code), None)
            if positive_record is None:
                continue
            alternatives = [record for record in records if str(record.get("candidate_id")) != positive_code][:5]
            positive = {"code": positive_code, "system": system, "aliases": positive_record.get("aliases", [])}
            candidates = [positive, *({"code": str(record.get("candidate_id")), "system": system, "aliases": record.get("aliases", [])} for record in alternatives)]
            positives.append(positive)
            retrieved.append(candidates)
            scored.append(ScoredCandidate(entity.text, entity.type, positive_code, system, 1.0, True))
            scored.extend(ScoredCandidate(entity.text, entity.type, str(record.get("candidate_id")), system, 0.2, False) for record in alternatives)
    if positives:
        candidate_artifact = build_candidate_training_artifact(positives, retrieved, scored, kb_hash=kb_hash)
        write_json(head_dir / "candidate_training_artifact.json", candidate_artifact.to_dict())
        write_json(head_dir / "candidate_calibration.json", candidate_artifact.calibration.to_dict())
    else:
        raise RuntimeError("no positive candidates were available to fit candidate calibration")
    model_status_path = _artifact_dir(config) / "model_status.json"
    model_status = json.loads(model_status_path.read_text(encoding="utf-8")) if model_status_path.is_file() else {}
    model_status.update(
        {
            "ner_model": {"trained": True, "checkpoint": str(run_dir / "checkpoints" / "final_fit" / "ner_model")},
            "assertion_model": {"trained": True, "head": str(head_dir / "assertion_head.pt")},
            "candidate_model": {"trained": True, "artifact": str(head_dir / "candidate_training_artifact.json")},
            "active_ner": "transformer_owner_window",
            "active_assertion": "frozen_shared_encoder_head",
        }
    )
    write_json(model_status_path, model_status)
    return {
        "phase": phase,
        "heads_dir": str(head_dir),
        "assertion_mentions": int(len(logits_array)),
        "candidate_examples": len(scored),
        "candidate_artifact": str(head_dir / "candidate_training_artifact.json"),
    }


def _phase_12_inference(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    from .candidate_policy import CandidatePolicy
    from .inference import InferenceConfig
    from .pipeline import run_inference_with_bundle
    from .runtime_bundle import load_final_model_bundle

    qwen_requested = bool(config.enable_qwen_reranker)
    qwen_summary: dict[str, Any] = {
        "qwen_requested": qwen_requested,
        "qwen_initialized": False,
        "qwen_model_name": None,
        "qwen_rerank_query_count": 0,
        "qwen_assertion_query_count": 0,
        "qwen_abstention_count": 0,
        "qwen_status": "DISABLED" if not qwen_requested else "FAILED",
    }
    qwen_reranker = None
    qwen_runtime = None
    phase_error: Exception | None = None
    run_dir = _run_dir(context)
    diagnostics_dir = run_dir / "diagnostics"

    try:
        if qwen_requested:
            from .entity_types import ASSERTION_ENTITY_TYPES
            from .qwen_refiner import RequiredQwenRefiner
            from .reranker import ClinicalLLMReranker

            class ObservedRequiredQwenRefiner(RequiredQwenRefiner):
                def refine(self, entities, raw_text):
                    rerank_indices = [
                        index
                        for index, entity in enumerate(entities)
                        if entity.type in {"DISEASE", "DRUG"} and entity.ranked_candidates
                    ]
                    assertion_count = sum(entity.type in ASSERTION_ENTITY_TYPES for entity in entities)
                    qwen_summary["qwen_rerank_query_count"] += len(rerank_indices)
                    qwen_summary["qwen_assertion_query_count"] += assertion_count
                    refined = super().refine(entities, raw_text)
                    qwen_summary["qwen_abstention_count"] += sum(
                        not refined[index].candidates for index in rerank_indices
                    )
                    return refined

            qwen_model_name = "Qwen/Qwen2.5-7B-Instruct-AWQ"
            qwen_runtime = ClinicalLLMReranker(
                model_name=qwen_model_name,
                gpu_memory_utilization=0.50,
                max_model_len=4096,
                batch_size=64,
            )
            qwen_reranker = ObservedRequiredQwenRefiner(qwen_runtime.llm, batch_size=64)
            qwen_summary["qwen_initialized"] = True
            qwen_summary["qwen_model_name"] = qwen_model_name

        checkpoint = run_dir / "checkpoints" / "final_fit" / "ner_model"
        head_dir = run_dir / "artifacts" / "heads"
        if not checkpoint.is_dir() or not head_dir.is_dir():
            raise FileNotFoundError("final checkpoint or fitted heads are missing")
        icd_path = _artifact_dir(config) / "icd10" / "icd10_dictionary.jsonl.gz"
        rx_path = _artifact_dir(config) / "rxnorm" / "rxnorm_dictionary.jsonl.gz"
        icd_records = load_candidate_dictionary(icd_path)
        rx_records = load_candidate_dictionary(rx_path)
        calibration_payload = json.loads((head_dir / "candidate_calibration.json").read_text(encoding="utf-8"))
        policy = CandidatePolicy.from_calibration(calibration_payload)
        bundle_args = (checkpoint, head_dir, icd_records, rx_records, policy)
        bundle = (
            load_final_model_bundle(*bundle_args, qwen_reranker=qwen_reranker)
            if qwen_reranker is not None
            else load_final_model_bundle(*bundle_args)
        )
        entity_mapping = json.loads((_artifact_dir(config) / "entity_type_mapping.json").read_text(encoding="utf-8"))
        assertion_mapping = json.loads((_artifact_dir(config) / "assertion_mapping.json").read_text(encoding="utf-8"))
        input_source = Path(config.input_source)
        if not input_source.is_absolute():
            input_source = Path.cwd() / input_source
        output_dir = run_dir / "output"
        summary = run_inference_with_bundle(
            input_source=input_source,
            output_dir=output_dir,
            bundle=bundle,
            entity_mapping=entity_mapping,
            assertion_mapping=assertion_mapping,
            config=InferenceConfig(enable_kb_recovery=True, enable_qwen=qwen_requested),
            create_zip=False,
            diagnostics_dir=diagnostics_dir,
        )
        from .output_quality import audit_submission_directory, enforce_output_quality

        quality_report = audit_submission_directory(input_source, output_dir)
        quality_path = diagnostics_dir / "output_quality.json"
        write_json(quality_path, quality_report)
        enforce_output_quality(quality_report)
        if qwen_requested:
            qwen_summary["qwen_status"] = "COMPLETED"
    except Exception as exc:
        phase_error = exc
        if qwen_requested:
            qwen_summary["qwen_status"] = "FAILED"
        raise
    finally:
        try:
            if qwen_runtime is not None:
                qwen_runtime.destroy()
        except Exception as cleanup_error:
            if qwen_requested:
                qwen_summary["qwen_status"] = "FAILED"
            if phase_error is not None:
                raise cleanup_error from phase_error
            raise cleanup_error
        finally:
            if qwen_requested and qwen_summary["qwen_status"] != "FAILED":
                qwen_summary["qwen_status"] = "COMPLETED"
            write_json(diagnostics_dir / "qwen_summary.json", qwen_summary)

    output_zip = run_dir / "output.zip"
    output_files = sorted(output_dir.glob("*.json"), key=lambda item: (int(item.stem), item.stem) if item.stem.isdigit() else (10**12, item.stem))
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for output_file in output_files:
            archive.write(output_file, arcname=f"output/{output_file.name}")
    with zipfile.ZipFile(output_zip) as archive:
        expected = [f"output/{path.name}" for path in output_files]
        if archive.namelist() != expected or archive.testzip() is not None:
            raise ValueError("Invalid output.zip structure or CRC")
    summary.update({"zip_path": str(output_zip), "zip_structure_valid": True})
    return {
        "phase": phase,
        **summary,
        **qwen_summary,
        "output_dir": str(output_dir),
        "quality_report": str(quality_path),
        "quality": quality_report,
    }


def _phase_13_packaging(config: RunConfig, phase: str, context: Mapping[str, Any]) -> Mapping[str, Any]:
    run_dir = _run_dir(context)
    package_path = run_dir / "trained_artifacts.zip"
    members = [path for path in run_dir.rglob("*") if path.is_file() and path != package_path]
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(members):
            archive.write(path, arcname=path.relative_to(run_dir).as_posix())
    with zipfile.ZipFile(package_path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("trained_artifacts.zip CRC validation failed")
    inventory = inventory_files([package_path])
    inventory_path = run_dir / "artifacts" / "package_inventory.json"
    write_json(inventory_path, {"members": len(members), "inventory": inventory})
    return {"phase": phase, "package": str(package_path), "inventory": str(inventory_path), "member_count": len(members)}


def build_kaggle_phase_runners(config: RunConfig) -> dict[str, PhaseRunner]:
    """Return the concrete dispatcher used by the Kaggle notebook."""
    return {
        PHASES[0]: _phase_01_preflight,
        PHASES[1]: _phase_02_resolve_sources,
        PHASES[2]: _phase_03_inventory_models,
        PHASES[3]: _phase_04_build_metadata,
        PHASES[4]: _phase_05_build_splits,
        PHASES[5]: _phase_06_prepare_training_contract,
        PHASES[6]: _phase_07_stage1,
        PHASES[7]: _phase_08_stage2,
        PHASES[8]: _phase_09_stage3,
        PHASES[9]: _phase_10_final_fit,
        PHASES[10]: _phase_11_fit_heads,
        PHASES[11]: _phase_12_inference,
        PHASES[12]: _phase_13_packaging,
    }


__all__ = ["build_kaggle_phase_runners"]
