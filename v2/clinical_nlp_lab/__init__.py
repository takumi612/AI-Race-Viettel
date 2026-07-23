"""Offline-first Clinical NLP lab package."""

from typing import Any

from .config import DEFAULT_CONFIG, load_config, save_config

__all__ = [
    "DEFAULT_CONFIG",
    "ClinicalNLPPipeline",
    "load_config",
    "run_inference",
    "run_inference_with_bundle",
    "save_config",
]

__version__ = "1.0.0"


def __getattr__(name: str) -> Any:
    if name in {"ClinicalNLPPipeline", "run_inference", "run_inference_with_bundle"}:
        from .pipeline import ClinicalNLPPipeline, run_inference, run_inference_with_bundle

        return {
            "ClinicalNLPPipeline": ClinicalNLPPipeline,
            "run_inference": run_inference,
            "run_inference_with_bundle": run_inference_with_bundle,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

