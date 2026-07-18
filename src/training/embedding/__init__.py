"""BGE-M3 hard-negative data, LoRA training, and index contracts."""

from src.training.embedding.config import EmbeddingTrainingConfig
from src.training.embedding.data import (
    CodeDescriptionStore,
    EmbeddingMiningResult,
    mine_hard_negative_examples,
    retrieval_ranking_metrics,
    select_embedding_seeds,
)
from src.training.embedding.index_manifest import (
    EmbeddingIndexManifest,
    validate_index_manifest,
    write_index_manifest,
)

__all__ = [
    "CodeDescriptionStore",
    "EmbeddingIndexManifest",
    "EmbeddingMiningResult",
    "EmbeddingTrainingConfig",
    "mine_hard_negative_examples",
    "retrieval_ranking_metrics",
    "select_embedding_seeds",
    "validate_index_manifest",
    "write_index_manifest",
]
