from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence, Literal


class CurriculumError(ValueError):
    """Raised when curriculum transition or resume validation fails."""


@dataclass(frozen=True)
class StageSpec:
    name: Literal["stage1", "stage2", "stage3", "final_fit"]
    parent_stage: str | None
    max_epochs: int
    learning_rate: float
    organizer_fraction: float | None
    replay_fraction: float | None
    update_encoder: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "parent_stage": self.parent_stage,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "organizer_fraction": self.organizer_fraction,
            "replay_fraction": self.replay_fraction,
            "update_encoder": self.update_encoder,
        }


@dataclass(frozen=True, slots=True)
class StageManifest:
    schema_id: str
    schema_version: int
    stage_name: str
    parent_stage: str | None
    config_hash: str
    fingerprints: dict[str, str]
    checkpoint_sha256: str
    status: Literal["completed"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_id": self.schema_id,
            "schema_version": self.schema_version,
            "stage_name": self.stage_name,
            "parent_stage": self.parent_stage,
            "config_hash": self.config_hash,
            "fingerprints": dict(sorted(self.fingerprints.items())),
            "checkpoint_sha256": self.checkpoint_sha256,
            "status": self.status,
        }


def build_stage_manifest(
    stage_spec: StageSpec,
    fingerprints: Mapping[str, str],
    checkpoint_sha256: str,
) -> StageManifest:
    if len(checkpoint_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in checkpoint_sha256.lower()):
        raise ValueError("checkpoint_sha256 must be a 64-character hexadecimal hash")
    config_bytes = json.dumps(stage_spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return StageManifest(
        schema_id="clinical_nlp.stage_manifest",
        schema_version=1,
        stage_name=stage_spec.name,
        parent_stage=stage_spec.parent_stage,
        config_hash=hashlib.sha256(config_bytes).hexdigest(),
        fingerprints={str(key): str(value) for key, value in fingerprints.items()},
        checkpoint_sha256=checkpoint_sha256,
        status="completed",
    )


def validate_resume_manifest(
    manifest: StageManifest | Mapping[str, Any],
    stage_spec: StageSpec,
    current_fingerprints: Mapping[str, str],
) -> None:
    data = manifest.to_dict() if isinstance(manifest, StageManifest) else dict(manifest)
    if data.get("stage_name") != stage_spec.name:
        raise ValueError(f"stage mismatch: expected {stage_spec.name}")
    config_bytes = json.dumps(stage_spec.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    expected_config_hash = hashlib.sha256(config_bytes).hexdigest()
    if data.get("config_hash") != expected_config_hash:
        raise ValueError("stage config fingerprint mismatch")
    checkpoint = str(data.get("checkpoint_sha256", ""))
    if len(checkpoint) != 64:
        raise ValueError("resume manifest is missing checkpoint hash")
    saved_fingerprints = data.get("fingerprints")
    if not isinstance(saved_fingerprints, Mapping):
        raise ValueError("resume manifest is missing fingerprints")
    for key, current in current_fingerprints.items():
        if saved_fingerprints.get(key) != current:
            raise ValueError(f"resume fingerprint mismatch for {key}")


def plan_curriculum(
    run_mode: Literal["full", "resume", "inference_only"],
    frozen_config: Mapping[str, Any] | None = None,
) -> tuple[StageSpec, ...]:
    if run_mode == "inference_only":
        return ()

    stage1 = StageSpec(
        name="stage1",
        parent_stage=None,
        max_epochs=3,
        learning_rate=3e-5,
        organizer_fraction=0.0,
        replay_fraction=0.0,
        update_encoder=True,
    )

    stage2 = StageSpec(
        name="stage2",
        parent_stage="stage1",
        max_epochs=2,
        learning_rate=2e-5,
        organizer_fraction=0.35,
        replay_fraction=0.0,
        update_encoder=True,
    )

    stage3 = StageSpec(
        name="stage3",
        parent_stage="stage2",
        max_epochs=4,
        learning_rate=1e-5,
        organizer_fraction=0.80,
        replay_fraction=0.20,
        update_encoder=True,
    )

    final_fit = StageSpec(
        name="final_fit",
        parent_stage="stage3",
        max_epochs=2,
        learning_rate=5e-6,
        organizer_fraction=0.50,
        replay_fraction=0.15,
        update_encoder=True,
    )

    if run_mode == "full":
        return (stage1, stage2, stage3, final_fit)
    elif run_mode == "resume":
        return (stage2, stage3, final_fit)
    else:
        raise CurriculumError(f"Unsupported run mode: {run_mode}")


def validate_stage_transition(
    stage_spec: StageSpec,
    parent_manifest: Mapping[str, Any] | None,
    current_fingerprints: Mapping[str, str],
) -> None:
    if stage_spec.parent_stage is not None:
        if not parent_manifest:
            raise CurriculumError(
                f"Missing parent manifest for stage '{stage_spec.name}', expected parent '{stage_spec.parent_stage}'"
            )
        parent_name = parent_manifest.get("stage_name")
        if parent_name != stage_spec.parent_stage:
            raise CurriculumError(
                f"Parent stage name mismatch: expected '{stage_spec.parent_stage}', got '{parent_name}'"
            )
        if not parent_manifest.get("checkpoint_sha256"):
            raise CurriculumError(f"Parent stage manifest '{parent_name}' is missing checkpoint_sha256")

        parent_fps = parent_manifest.get("fingerprints", {})
        for key, curr_val in current_fingerprints.items():
            parent_val = parent_fps.get(key)
            if parent_val and parent_val != curr_val:
                raise CurriculumError(
                    f"Stale fingerprint '{key}' for resume in stage '{stage_spec.name}': "
                    f"parent={parent_val}, current={curr_val}"
                )
