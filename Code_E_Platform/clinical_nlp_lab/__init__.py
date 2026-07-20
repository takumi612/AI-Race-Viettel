"""Offline-first Clinical NLP lab package."""

from .config import DEFAULT_CONFIG, load_config, save_config
from .pipeline import ClinicalNLPPipeline, run_inference

__all__ = [
    "DEFAULT_CONFIG",
    "ClinicalNLPPipeline",
    "load_config",
    "run_inference",
    "save_config",
]

__version__ = "1.0.0"

