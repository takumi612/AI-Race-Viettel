"""Split-safe entity-code pairs, descriptions, and hard-negative mining."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sqlite3
from typing import Any, Callable, Mapping, Sequence

from src.training.metrics import exact_fbeta


class CodeDescriptionStore:
    def __init__(self, database: str | Path):
        self.database = Path(database).resolve()
        if not self.database.is_file() or self.database.stat().st_size == 0:
            raise ValueError(f"metadata database is missing or empty: {self.database}")
        try:
            with self._connect() as connection:
                connection.execute("SELECT 1 FROM icd10 LIMIT 1").fetchone()
                connection.execute("SELECT 1 FROM rxnorm LIMIT 1").fetchone()
        except sqlite3.Error as exc:
            raise ValueError(f"invalid metadata database: {exc}") from exc

    def _connect(self):
        return sqlite3.connect(f"{self.database.as_uri()}?mode=ro", uri=True)

    def get(self, entity_type: str, code: str) -> str:
        normalized = str(code).strip()
        with self._connect() as connection:
            if entity_type == "CHẨN_ĐOÁN":
                row = connection.execute(
                    "SELECT name_vi, name_en FROM icd10 WHERE code = ? LIMIT 1",
                    (normalized,),
                ).fetchone()
                if row:
                    names = [str(value).strip() for value in row if value]
                    return f"{normalized}: {' | '.join(names)}"
            elif entity_type == "THUỐC":
                row = connection.execute(
                    "SELECT name FROM rxnorm WHERE CAST(rxcui AS TEXT) = ? LIMIT 1",
                    (normalized,),
                ).fetchone()
                if row and row[0]:
                    return str(row[0]).strip()
            else:
                raise ValueError(f"unsupported embedding entity type: {entity_type}")
        raise KeyError(f"{entity_type} code is missing from metadata.db: {normalized}")


def load_embedding_seeds(path: str | Path) -> tuple[dict[str, Any], ...]:
    source = Path(path)
    if not source.is_file():
        raise ValueError(f"embedding seeds are missing: {source}")
    values: list[dict[str, Any]] = []
    try:
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not line.strip():
                raise ValueError(f"blank embedding JSONL line: {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"embedding seed {line_number} must be an object")
            values.append(value)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid embedding seeds: {source}") from exc
    return tuple(values)


def select_embedding_seeds(
    seeds: Sequence[Mapping[str, Any]],
    *,
    stage: str,
    role: str,
    fold: int | None = None,
) -> tuple[dict[str, Any], ...]:
    if stage not in {"synthetic", "trusted-fold", "trusted-final"}:
        raise ValueError(f"unsupported embedding stage: {stage}")
    if role not in {"train", "eval"}:
        raise ValueError("embedding role must be train or eval")
    if stage == "trusted-fold":
        if isinstance(fold, bool) or not isinstance(fold, int) or not 0 <= fold < 5:
            raise ValueError("trusted-fold requires fold in [0, 4]")
    elif fold is not None:
        raise ValueError(f"{stage} does not accept a fold")

    selected: list[dict[str, Any]] = []
    for seed in seeds:
        split = seed.get("split")
        if isinstance(split, str) and "pseudo" in split.casefold():
            raise ValueError("pseudo-label embedding seeds are forbidden")
        include = False
        if stage == "synthetic":
            include = split == (
                "synthetic_train" if role == "train" else "synthetic_validation"
            )
        elif stage == "trusted-fold" and split == "trusted_fold":
            include = (
                seed.get("fold") != fold
                if role == "train"
                else seed.get("fold") == fold
            )
        elif stage == "trusted-final":
            include = role == "train" and split == "trusted_fold"
        if include:
            selected.append(dict(seed))
    return tuple(
        sorted(
            selected,
            key=lambda value: (
                str(value.get("record_id", "")),
                str(value.get("example_id", "")),
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class EmbeddingMiningResult:
    examples: tuple[dict[str, Any], ...]
    retrieval_misses: tuple[str, ...]


def _candidate_code(candidate: Any) -> str:
    return str(getattr(candidate, "code", candidate)).strip()


def mine_hard_negative_examples(
    seeds: Sequence[Mapping[str, Any]],
    descriptions: CodeDescriptionStore,
    *,
    retrieve: Callable[[str, str, int], Sequence[Any]],
    negatives_per_example: int,
    retrieval_top_k: int,
) -> EmbeddingMiningResult:
    if negatives_per_example < 1 or retrieval_top_k < negatives_per_example:
        raise ValueError("invalid hard-negative mining limits")
    examples: list[dict[str, Any]] = []
    misses: list[str] = []
    for seed in seeds:
        entity_type = seed.get("entity_type")
        query = seed.get("query")
        context = seed.get("context")
        positive_codes = seed.get("positive_codes")
        if (
            entity_type not in {"CHẨN_ĐOÁN", "THUỐC"}
            or not isinstance(query, str)
            or not isinstance(context, str)
            or not isinstance(positive_codes, list)
            or not positive_codes
        ):
            raise ValueError(f"invalid embedding seed: {seed.get('example_id')}")
        positives = tuple(dict.fromkeys(str(code).strip() for code in positive_codes))
        ranked = tuple(
            dict.fromkeys(
                code
                for candidate in retrieve(entity_type, query, retrieval_top_k)
                if (code := _candidate_code(candidate))
            )
        )
        if not set(positives) & set(ranked):
            misses.append(str(seed.get("example_id")))
        negatives: list[str] = []
        for code in ranked:
            if code in positives:
                continue
            try:
                descriptions.get(entity_type, code)
            except KeyError:
                continue
            negatives.append(code)
            if len(negatives) >= negatives_per_example:
                break
        anchor = f"{query}\n[CONTEXT]\n{context}"
        for positive_code in positives:
            positive_text = descriptions.get(entity_type, positive_code)
            for negative_code in negatives:
                examples.append(
                    {
                        "example_id": str(seed.get("example_id")),
                        "record_id": str(seed.get("record_id")),
                        "anchor": anchor,
                        "positive": positive_text,
                        "negative": descriptions.get(entity_type, negative_code),
                        "positive_code": positive_code,
                        "positive_codes": list(positives),
                        "negative_code": negative_code,
                        "entity_type": entity_type,
                        "split": seed.get("split"),
                        **({"fold": seed["fold"]} if "fold" in seed else {}),
                    }
                )
    return EmbeddingMiningResult(tuple(examples), tuple(sorted(set(misses))))


def retrieval_ranking_metrics(
    rankings: Sequence[tuple[set[str], Sequence[str]]],
) -> dict[str, float]:
    if not rankings:
        raise ValueError("rankings must not be empty")
    recalls = {1: 0.0, 5: 0.0, 10: 0.0}
    reciprocal_rank = 0.0
    ndcg = 0.0
    downstream_gold: set[tuple[int, str]] = set()
    downstream_predicted: set[tuple[int, str]] = set()
    for example_index, (gold, ranked_values) in enumerate(rankings):
        gold_codes = {str(code) for code in gold}
        if not gold_codes:
            raise ValueError("each ranking requires at least one gold code")
        ranked = [str(code) for code in ranked_values[:10]]
        downstream_gold.update(
            (example_index, code) for code in gold_codes
        )
        downstream_predicted.update(
            (example_index, code) for code in ranked[:2]
        )
        for cutoff in recalls:
            recalls[cutoff] += float(bool(gold_codes & set(ranked[:cutoff])))
        first = next(
            (index for index, code in enumerate(ranked, start=1) if code in gold_codes),
            None,
        )
        if first is not None:
            reciprocal_rank += 1.0 / first
        dcg = sum(
            1.0 / math.log2(index + 1)
            for index, code in enumerate(ranked, start=1)
            if code in gold_codes
        )
        ideal_hits = min(len(gold_codes), 10)
        ideal = sum(
            1.0 / math.log2(index + 1)
            for index in range(1, ideal_hits + 1)
        )
        ndcg += dcg / ideal
    count = len(rankings)
    downstream = exact_fbeta(
        downstream_gold,
        downstream_predicted,
        beta=0.5,
    )
    return {
        "recall_at_1": recalls[1] / count,
        "recall_at_5": recalls[5] / count,
        "recall_at_10": recalls[10] / count,
        "mrr_at_10": reciprocal_rank / count,
        "ndcg_at_10": ndcg / count,
        "downstream_precision": downstream.precision,
        "downstream_recall": downstream.recall,
        "downstream_f0_5": downstream.f_beta,
    }
