"""Pydantic models for RunPod state, keyframes, branches, shots, stories, and jobs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field


def utc_now_iso() -> str:
    """Return an ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat()


class ActiveFactoryState(BaseModel):
    """Persisted metadata for an active cloud GPU session (mode 0600)."""

    provider: Literal["runpod"] = "runpod"
    pod_id: str
    gpu_name: str
    vram_gb: int
    hourly_rate_usd: float
    ssh_host: str
    ssh_port: int
    local_comfy_port: int = 8188
    tunnel_pid: int | None = None
    network_volume_id: str | None = None
    datacenter_id: str | None = None
    started_at: str = Field(default_factory=utc_now_iso)


class CharacterSpec(BaseModel):
    """Character identity, Master Portrait, and optional LoRA references."""

    char_id: str = ""
    name: str
    mode: Literal["lora", "kontext_portrait", "description_only"] = "kontext_portrait"
    trigger_word: str = ""
    lora_strength: float = 0.85
    appearance: str = ""
    default_wardrobe: str = ""
    default_prop_ids: list[str] = Field(default_factory=list)
    master_portrait_path: str | None = None
    mask_region: str | None = None


class KeyframeState(BaseModel):
    """Start/End keyframe generation, staging, and pre-render approval state."""

    status: Literal["staged", "generating", "awaiting_approval", "approved"] = "staged"
    start_frame_path: str | None = None
    end_frame_path: str | None = None
    has_end_frame: bool = False
    approved: bool = False
    start_prompt: str = ""
    end_prompt: str | None = None
    negative_prompt: str = "blurry, deformed hands, extra fingers, plastic skin, watermark"
    reference_anchor_path: str | None = None
    last_edit_suggestion: str | None = None
    seed: int = 42
    lora_strength: float = 0.85
    width: int = 1280
    height: int = 720
    updated_at: str = Field(default_factory=utc_now_iso)


class BranchRecord(BaseModel):
    """A single continuation branch created from a Frame K checkpoint."""

    branch_id: str
    mode: Literal["carry", "pivot", "lipsync"]
    continuation_prompt: str | None = None
    end_keyframe_prompt: str | None = None
    audio_wav_path: str | None = None
    driving_video_path: str | None = None
    branched_master_path: str
    branched_preview_path: str
    wan_frames_requested: int | None = None
    tail_frames_kept: int
    seed: int = 42
    created_at: str = Field(default_factory=utc_now_iso)


class BranchCheckpoint(BaseModel):
    """Cached lossless prefix and raw/repaired frames at a specific Frame K."""

    checkpoint_id: str
    checkpoint_dir: str
    source_video_path: str
    frame_index: int
    total_source_frames: int
    fps: float = 24.0
    locked_prefix_path: str
    seam_ref_frame_path: str | None = None
    raw_frame_path: str
    repaired_frame_path: str | None = None
    mask_path: str | None = None
    repair_engine: Literal["kontext", "fill"] = "kontext"
    repair_prompt: str = ""
    seam_continuity_score: float | None = None
    branches: list[BranchRecord] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


class ShotManifest(BaseModel):
    """Complete lineage and render metadata for a single shot."""

    project_name: str
    shot_id: str
    scene_id: str = "scene_01"
    scene_name: str = ""
    shot_order: int = 1
    character_name: str
    character_mode: Literal["lora", "kontext_portrait", "description_only"] = "kontext_portrait"
    characters: list[CharacterSpec] = Field(default_factory=list)
    prompt: str = ""
    video_motion_prompt: str = ""
    beat_summary: str = ""
    spoken_text: str | None = None
    audio_cue: str = ""
    inherited_wardrobe: str = ""
    inherited_props: list[str] = Field(default_factory=list)
    negative_prompt: str = "static, morphing face, extra limbs, jitter, flicker, low quality"
    duration_sec: float = 5.0
    fps: float = 24.0
    total_frames: int = 0
    keyframes: KeyframeState = Field(default_factory=KeyframeState)
    master_video_path: str | None = None
    preview_video_path: str | None = None
    contact_strip_path: str | None = None
    last_frame_path: str | None = None
    checkpoints: dict[str, BranchCheckpoint] = Field(default_factory=dict)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)


class StoryManifest(BaseModel):
    """Project-level continuity ledger tracking shot order, scene, wardrobe, and exports."""

    project_name: str
    scene_description: str = ""
    wardrobe_notes: str = ""
    lighting_notes: str = ""
    shot_order: list[str] = Field(default_factory=list)
    master_cut_path: str | None = None
    updated_at: str = Field(default_factory=utc_now_iso)


class AsyncJobRecord(BaseModel):
    """Local tracking record for detached remote jobs (e.g., LoRA training)."""

    job_id: str
    job_type: Literal["lora_training"] = "lora_training"
    character_name: str
    trigger_word: str
    train_flux: bool = True
    train_wan22: bool = True
    remote_job_dir: str
    remote_dataset_dir: str
    tmux_session: str
    status: Literal["running", "completed", "failed", "collected"] = "running"
    stage: str = "queued"
    progress_pct: float = 0.0
    error: str | None = None
    local_artifacts: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)
    updated_at: str = Field(default_factory=utc_now_iso)
