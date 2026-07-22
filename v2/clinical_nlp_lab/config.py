from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "fast_dev_run": True,
    "input_zip": "input.zip",
    "input_dir": "input",
    "train_dir": "train",
    "output_dir": "output",
    "diagnostics_dir": "diagnostics",
    "artifact_dir": "artifacts",
    "icd10_path": "ICD10.xlsx",
    "icd10_sheet": "ICD10",
    "icd10_header_row": 3,
    "rxnorm_zip_path": "RxNorm_full_07062026.zip",
    "rxnorm_conso_member": "rrf/RXNCONSO.RRF",
    "rxnorm_rel_member": "rrf/RXNREL.RRF",
    "rxnorm_languages": ["ENG"],
    "rxnorm_sources": ["RXNORM"],
    "rxnorm_tty": ["IN", "PIN", "MIN", "BN", "SCD", "SBD", "GPCK", "BPCK", "DF", "DFG"],
    "rxnorm_suppress": ["N"],
    "rxnorm_relation_names": ["has_ingredient", "tradename_of", "has_dose_form", "consists_of"],
    "ner_model_name": "xlm-roberta-base",
    "assertion_model_name": "xlm-roberta-base",
    "relation_model_name": "xlm-roberta-base",
    "embedding_model_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "max_length": 512,
    "stride": 128,
    "batch_size": 8,
    "gradient_accumulation_steps": 2,
    "learning_rate": 2e-5,
    "ner_epochs": 3,
    "assertion_epochs": 3,
    "candidate_top_k": 20,
    "candidate_output_k": 1,
    "enable_regex_fallback": False,
    "relation_max_distance": 256,
    "validation_fraction": 0.2,
    "drop_unmapped_entity_types": True,
    "submission_keys": ["text", "type", "candidates", "assertions", "position"],
    "internal_entity_types": ["DISEASE", "DRUG", "SYMPTOM", "LAB_NAME", "LAB_RESULT", "PATIENT_INFO"],
    "thresholds": {
        "dictionary_exact": 0.99,
        "dictionary_phrase": 0.94,
        "regex_rule": 0.78,
        "candidate_min_score": 0.50,
        "candidate_min_margin": 0.05,
        "overlap_close_confidence": 0.05
    }
}


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        return copy.deepcopy(DEFAULT_CONFIG)
    config_path = Path(path)
    if not config_path.exists():
        return copy.deepcopy(DEFAULT_CONFIG)
    with config_path.open("r", encoding="utf-8") as stream:
        override = json.load(stream)
    return merge_config(DEFAULT_CONFIG, override)


def save_config(config: dict[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(config, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    return destination


def set_reproducible_seed(seed: int) -> dict[str, bool]:
    random.seed(seed)
    status = {"python": True, "numpy": False, "torch": False}
    try:
        import numpy as np

        np.random.seed(seed)
        status["numpy"] = True
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        status["torch"] = True
    except ImportError:
        pass
    return status

