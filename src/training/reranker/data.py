"""Frozen candidate contracts and completion-only reranker examples."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from uuid import uuid4

from src.training.fingerprints import sha256_file
from src.training.metrics import exact_fbeta


_CODED_ENTITY_TYPES = frozenset({"CHẨN_ĐOÁN", "THUỐC"})


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _candidate_code(candidate: Any) -> str:
    if isinstance(candidate, Mapping):
        value = candidate.get("code")
    else:
        value = getattr(candidate, "code", candidate)
    return str(value).strip()


@dataclass(frozen=True, slots=True)
class FrozenCandidateResult:
    examples: tuple[dict[str, Any], ...]
    retrieval_misses: tuple[str, ...]
    retriever_fingerprint: str


def write_frozen_candidate_dataset(
    output_dir: str | Path,
    result: FrozenCandidateResult,
    *,
    dataset_build_id: str,
    dataset_manifest_sha256: str,
) -> Path:
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"frozen candidate dataset already exists: {output}")
    if not result.examples:
        raise ValueError("cannot write an empty frozen candidate dataset")
    if not dataset_build_id or not _is_sha256(dataset_manifest_sha256):
        raise ValueError("invalid source dataset identity")
    temporary = output.parent / f".{output.name}.{uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        examples_path = temporary / "examples.jsonl"
        examples_path.write_text(
            "".join(
                json.dumps(example, ensure_ascii=False, sort_keys=True) + "\n"
                for example in result.examples
            ),
            encoding="utf-8",
        )
        manifest = {
            "schema_version": 1,
            "dataset_build_id": dataset_build_id,
            "dataset_manifest_sha256": dataset_manifest_sha256.casefold(),
            "retriever_fingerprint": result.retriever_fingerprint,
            "examples_sha256": sha256_file(examples_path),
            "example_count": len(result.examples),
            "retrieval_miss_count": len(result.retrieval_misses),
            "retrieval_misses": list(result.retrieval_misses),
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(output)
    finally:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
    return output


def load_frozen_candidate_dataset(
    directory: str | Path,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    root = Path(directory).resolve()
    manifest_path = root / "manifest.json"
    examples_path = root / "examples.jsonl"
    if not manifest_path.is_file() or not examples_path.is_file():
        raise ValueError(f"frozen candidate dataset is incomplete: {root}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen candidate manifest") from exc
    expected_keys = {
        "schema_version",
        "dataset_build_id",
        "dataset_manifest_sha256",
        "retriever_fingerprint",
        "examples_sha256",
        "example_count",
        "retrieval_miss_count",
        "retrieval_misses",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("frozen candidate manifest keys do not match schema")
    if (
        manifest["schema_version"] != 1
        or not _is_sha256(manifest["dataset_manifest_sha256"])
        or not _is_sha256(manifest["retriever_fingerprint"])
        or not _is_sha256(manifest["examples_sha256"])
        or sha256_file(examples_path) != manifest["examples_sha256"]
    ):
        raise ValueError("frozen candidate dataset fingerprint mismatch")
    for field in ("example_count", "retrieval_miss_count"):
        value = manifest[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"invalid frozen candidate {field}")
    if not isinstance(manifest["retrieval_misses"], list) or len(
        manifest["retrieval_misses"]
    ) != manifest["retrieval_miss_count"]:
        raise ValueError("invalid frozen candidate retrieval misses")
    examples: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            examples_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"blank frozen candidate line: {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"frozen candidate line {line_number} is not an object")
            if value.get("retriever_fingerprint") != manifest["retriever_fingerprint"]:
                raise ValueError("example retriever fingerprint mismatch")
            target_json(value)
            examples.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid frozen candidate JSONL") from exc
    if len(examples) != manifest["example_count"]:
        raise ValueError("frozen candidate example count mismatch")
    return manifest, tuple(examples)


def freeze_candidate_examples(
    seeds: Sequence[Mapping[str, Any]],
    *,
    retrieve: Callable[[str, str, int], Sequence[Any]],
    describe: Callable[[str, str], str],
    retriever_fingerprint: str,
    top_k: int,
) -> FrozenCandidateResult:
    """Freeze only retrieved pools; gold outside the pool is never inserted."""
    if not _is_sha256(retriever_fingerprint):
        raise ValueError("retriever fingerprint must be a SHA-256 digest")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("candidate top_k must be positive")

    examples: list[dict[str, Any]] = []
    misses: list[str] = []
    for seed in seeds:
        example_id = str(seed.get("example_id", "")).strip()
        entity_type = seed.get("entity_type")
        query = seed.get("entity_text")
        gold_values = seed.get("ground_truth_codes")
        if (
            not example_id
            or entity_type not in _CODED_ENTITY_TYPES
            or not isinstance(query, str)
            or not query.strip()
            or not isinstance(gold_values, list)
            or not gold_values
        ):
            raise ValueError(f"invalid reranker seed: {example_id or '<unknown>'}")
        gold = tuple(
            dict.fromkeys(str(code).strip() for code in gold_values if str(code).strip())
        )
        candidate_rows: list[dict[str, Any]] = []
        seen_codes: set[str] = set()
        for rank, candidate in enumerate(
            retrieve(str(entity_type), query, top_k),
            start=1,
        ):
            code = _candidate_code(candidate)
            if not code or code in seen_codes:
                continue
            try:
                description = describe(str(entity_type), code)
            except KeyError:
                continue
            seen_codes.add(code)
            row: dict[str, Any] = {
                "code": code,
                "description": str(description),
                "rank": rank,
            }
            for source, target in (
                ("fusion_score", "fusion_score"),
                ("bm25_score", "bm25_score"),
                ("semantic_score", "semantic_score"),
            ):
                value = (
                    candidate.get(source)
                    if isinstance(candidate, Mapping)
                    else getattr(candidate, source, None)
                )
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    row[target] = float(value)
            candidate_rows.append(row)
            if len(candidate_rows) >= top_k:
                break

        selected = [code for code in gold if code in seen_codes][:2]
        if not selected:
            misses.append(example_id)
            continue
        example = dict(seed)
        example["ground_truth_codes"] = list(gold)
        example["candidates"] = candidate_rows
        example["selected_codes"] = selected
        example["retriever_fingerprint"] = retriever_fingerprint.casefold()
        examples.append(example)

    examples.sort(key=lambda value: str(value["example_id"]))
    return FrozenCandidateResult(
        examples=tuple(examples),
        retrieval_misses=tuple(sorted(set(misses))),
        retriever_fingerprint=retriever_fingerprint.casefold(),
    )


def select_reranker_examples(
    examples: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    role: str,
    fold: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if stage not in {"synthetic", "trusted-fold", "trusted-final"}:
        raise ValueError(f"unsupported reranker stage: {stage}")
    if role not in {"train", "eval"}:
        raise ValueError("reranker role must be train or eval")
    if stage == "trusted-fold":
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 5:
            raise ValueError("trusted-fold requires fold in [0, 4]")
    elif fold is not None:
        raise ValueError(f"{stage} does not accept a fold")

    selected: list[dict[str, Any]] = []
    for example in examples:
        split = example.get("split")
        if isinstance(split, str) and "pseudo" in split.casefold():
            raise ValueError("pseudo-label reranker examples are forbidden")
        include = False
        if stage == "synthetic":
            include = split == (
                "synthetic_train" if role == "train" else "synthetic_validation"
            )
        elif stage == "trusted-fold" and split == "trusted_fold":
            include = (
                example.get("fold") != fold
                if role == "train"
                else example.get("fold") == fold
            )
        elif stage == "trusted-final":
            include = role == "train" and split == "trusted_fold"
        if include:
            selected.append(dict(example))
    return tuple(
        sorted(selected, key=lambda value: str(value.get("example_id", "")))
    )


def _candidate_codes(example: Mapping[str, Any]) -> list[str]:
    candidates = example.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("reranker example has no frozen candidate pool")
    codes = [_candidate_code(candidate) for candidate in candidates]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("reranker candidate codes must be unique and non-empty")
    return codes


def build_reranker_prompt(example: Mapping[str, Any]) -> str:
    codes = _candidate_codes(example)
    context = example.get("context")
    entity_text = example.get("entity_text")
    entity_type = example.get("entity_type")
    assertions = example.get("assertions", [])
    if (
        not isinstance(context, str)
        or not isinstance(entity_text, str)
        or entity_type not in _CODED_ENTITY_TYPES
        or not isinstance(assertions, list)
    ):
        raise ValueError("invalid reranker prompt fields")
    candidate_lines = []
    for index, (candidate, code) in enumerate(
        zip(example["candidates"], codes),
        start=1,
    ):
        description = (
            candidate.get("description", "")
            if isinstance(candidate, Mapping)
            else ""
        )
        candidate_lines.append(
            f"{index}. code={code}; description={description}"
        )
    return (
        "Select zero, one, or two clinically supported codes only from the "
        "candidate pool. Return JSON only with schema "
        '{"selected_codes":["CODE"]}. Do not invent a code.\n'
        f"Context: {context}\n"
        f"Entity: {entity_text}\n"
        f"Entity type: {entity_type}\n"
        f"Assertions: {json.dumps(assertions, ensure_ascii=False)}\n"
        "Candidates:\n"
        + "\n".join(candidate_lines)
        + "\nJSON:"
    )


def target_json(example: Mapping[str, Any]) -> str:
    pool = set(_candidate_codes(example))
    selected = example.get("selected_codes")
    if (
        not isinstance(selected, list)
        or len(selected) > 2
        or any(not isinstance(code, str) or not code for code in selected)
    ):
        raise ValueError("selected_codes must contain zero to two strings")
    normalized = list(dict.fromkeys(selected))
    if len(normalized) != len(selected):
        raise ValueError("selected_codes must be unique")
    if any(code not in pool for code in normalized):
        raise ValueError("selected code is outside the candidate pool")
    return json.dumps(
        {"selected_codes": normalized},
        ensure_ascii=False,
        separators=(",", ":"),
    )


def completion_only_features(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 2:
        raise ValueError("max_length must be at least two")
    prompt_ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    target_ids = list(tokenizer.encode(target, add_special_tokens=False))
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer requires eos_token_id")
    completion = target_ids + [eos_token_id]
    if len(completion) >= max_length:
        raise ValueError("reranker target exceeds max_seq_length")
    prompt_ids = prompt_ids[-(max_length - len(completion)) :]
    input_ids = prompt_ids + completion
    labels = [-100] * len(prompt_ids) + completion
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": labels,
    }


def _parse_generated(value: Any) -> list[str]:
    if not isinstance(value, str):
        raise ValueError("generated reranker output must be text")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("invalid generated JSON") from exc
    if set(payload) != {"selected_codes"} or not isinstance(
        payload["selected_codes"], list
    ):
        raise ValueError("generated JSON does not match schema")
    selected = payload["selected_codes"]
    if (
        len(selected) > 2
        or any(not isinstance(code, str) or not code for code in selected)
        or len(selected) != len(set(selected))
    ):
        raise ValueError("generated selected_codes are invalid")
    return selected


def reranker_generation_metrics(
    outputs: Sequence[str],
    examples: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    if not outputs or len(outputs) != len(examples):
        raise ValueError("outputs and examples must have equal non-zero length")
    invalid_json = 0
    out_of_pool = 0
    gold_pairs: set[tuple[int, str]] = set()
    predicted_pairs: set[tuple[int, str]] = set()
    for index, (output, example) in enumerate(zip(outputs, examples)):
        gold = target_json(example)
        gold_codes = json.loads(gold)["selected_codes"]
        gold_pairs.update((index, code) for code in gold_codes)
        try:
            selected = _parse_generated(output)
        except (TypeError, ValueError):
            invalid_json += 1
            continue
        pool = set(_candidate_codes(example))
        if any(code not in pool for code in selected):
            out_of_pool += 1
            continue
        predicted_pairs.update((index, code) for code in selected)
    exact = exact_fbeta(gold_pairs, predicted_pairs, beta=0.5)
    count = len(examples)
    return {
        "invalid_json_rate": invalid_json / count,
        "out_of_pool_rate": out_of_pool / count,
        "precision": exact.precision,
        "recall": exact.recall,
        "f0_5": exact.f_beta,
    }


__all__ = [
    "FrozenCandidateResult",
    "build_reranker_prompt",
    "completion_only_features",
    "freeze_candidate_examples",
    "load_frozen_candidate_dataset",
    "reranker_generation_metrics",
    "select_reranker_examples",
    "target_json",
    "write_frozen_candidate_dataset",
]
