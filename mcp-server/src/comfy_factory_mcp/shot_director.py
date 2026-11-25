"""Shot Director & Pause-and-Pivot Branch Manager (Start/End Keyframe Gate + 4k+1 Wan math)."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Literal

from comfy_factory_mcp.comfy_client import ComfyUIClient
from comfy_factory_mcp.config import FactorySettings, resolve_safe_path
from comfy_factory_mcp.models import (
    BranchCheckpoint,
    BranchRecord,
    KeyframeState,
    ShotManifest,
    StoryManifest,
    utc_now_iso,
)
from comfy_factory_mcp.qa_metrics import (
    FFMPEG_BIN,
    build_contact_strip,
    compute_seam_continuity_score,
    concat_prefix_and_tail,
    extract_single_frame,
    probe_media_info,
    slice_lossless_prefix,
    slice_lossless_tail,
)
from comfy_factory_mcp.ssh_tunnel import SSHTunnelManager


def compute_wan_continuation_frames(remaining_frames: int) -> tuple[int, int]:
    """Compute Wan 2.2 4k+1 padded frame count and exact non-duplicate tail frames to keep.

    Example: remaining_frames=36 -> wan_frames=37 (4*9+1), after dropping duplicate index 0 we keep 36.
    """
    if remaining_frames <= 0:
        raise ValueError("remaining_frames must be >= 1")
    wan_frames = max(5, math.ceil(remaining_frames / 4) * 4 + 1)
    return wan_frames, remaining_frames


def compute_wan_shot_frames(duration_sec: float, fps: int = 24) -> int:
    """Compute a valid 4k+1 frame count for full-shot Wan 2.2 generation."""
    raw = max(5, round(duration_sec * fps))
    return max(5, math.ceil((raw - 1) / 4) * 4 + 1)


class ShotDirector:
    """Orchestrates Start/End Keyframe review, Wan 2.2 MoE rendering, and Pause-and-Pivot."""

    def __init__(
        self,
        settings: FactorySettings,
        comfy: ComfyUIClient,
        tunnel: SSHTunnelManager | None = None,
    ) -> None:
        self.settings = settings
        self.comfy = comfy
        self.tunnel = tunnel

    def project_dir(self, project_name: str) -> Path:
        safe = resolve_safe_path(self.settings.outputs_dir / project_name, self.settings)
        safe.mkdir(parents=True, exist_ok=True)
        return safe

    def shot_dir(self, project_name: str, shot_id: str) -> Path:
        safe = self.project_dir(project_name) / "shots" / shot_id
        safe.mkdir(parents=True, exist_ok=True)
        return safe

    def load_shot_manifest(self, shot_dir: Path) -> ShotManifest:
        path = shot_dir / "shot.manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"Shot manifest not found: {path}")
        return ShotManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def save_shot_manifest(self, shot_dir: Path, manifest: ShotManifest) -> Path:
        manifest.updated_at = utc_now_iso()
        path = shot_dir / "shot.manifest.json"
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        return path

    def load_or_create_story_manifest(self, project_name: str) -> StoryManifest:
        pdir = self.project_dir(project_name)
        path = pdir / "story_manifest.json"
        if path.exists():
            return StoryManifest.model_validate_json(path.read_text(encoding="utf-8"))
        story = StoryManifest(project_name=project_name)
        path.write_text(story.model_dump_json(indent=2), encoding="utf-8")
        return story

    def save_story_manifest(self, story: StoryManifest) -> Path:
        story.updated_at = utc_now_iso()
        path = self.project_dir(story.project_name) / "story_manifest.json"
        path.write_text(story.model_dump_json(indent=2), encoding="utf-8")
        return path

    def _find_character_lora(self, character_name: str, kind: Literal["flux", "wan22"]) -> Path | None:
        char_dir = self.settings.loras_dir / character_name
        if not char_dir.exists():
            return None
        patterns = ["*flux*.safetensors"] if kind == "flux" else ["*wan22*.safetensors", "*wan*.safetensors"]
        for pattern in patterns:
            matches = sorted(char_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _ensure_cloud_ready(self) -> SSHTunnelManager:
        if self.tunnel is None:
            raise RuntimeError("Cloud GPU factory is not active. Run factory_up first.")
        return self.tunnel

    async def _upload_character_lora(self, character_name: str, kind: Literal["flux", "wan22"], scratch_tag: str) -> str:
        tunnel = self._ensure_cloud_ready()
        local_lora = self._find_character_lora(character_name, kind)
        if local_lora is None:
            return f"scratch/{scratch_tag}/missing_{kind}.safetensors"
        cache_subdir = f"session_cache/{character_name}"
        remote_dir = f"/dev/shm/factory_scratch/active_loras/{cache_subdir}"
        remote_file = f"{remote_dir}/{local_lora.name}"
        if hasattr(tunnel, "run_remote"):
            expected_size = local_lora.stat().st_size
            check = await tunnel.run_remote(
                f"if [ \"$(stat -c%s '{remote_file}' 2>/dev/null || echo 0)\" = '{expected_size}' ]; then echo cached; fi"
            )
            if "cached" in check:
                return f"scratch/{cache_subdir}/{local_lora.name}"
        await tunnel.rsync_upload(local_lora, remote_dir)
        return f"scratch/{cache_subdir}/{local_lora.name}"

    @staticmethod
    def _bypass_missing_flux_lora_keyframes(patched: dict[str, Any], lora_rel: str) -> None:
        """When a character has no FLUX LoRA, rewire CLIP/UNET directly and strip LoraLoader nodes 4 & 13."""
        if "missing_" not in lora_rel:
            return
        if "4" in patched:
            for nid in ("5", "7"):
                if nid in patched and patched[nid]["inputs"].get("clip") == ["4", 1]:
                    patched[nid]["inputs"]["clip"] = ["2", 0]
            if "9" in patched and patched["9"]["inputs"].get("model") == ["4", 0]:
                patched["9"]["inputs"]["model"] = ["1", 0]
            patched.pop("4", None)
        if "13" in patched:
            if "15" in patched and patched["15"]["inputs"].get("clip") == ["13", 1]:
                patched["15"]["inputs"]["clip"] = ["2", 0]
            if "18" in patched and patched["18"]["inputs"].get("model") == ["13", 0]:
                patched["18"]["inputs"]["model"] = ["12", 0]
            patched.pop("13", None)

    @staticmethod
    def _inject_kontext_start_anchor(
        patched: dict[str, Any], remote_anchor_rel: str, denoise: float = 0.72
    ) -> None:
        """Anchor full-frame rebuilds to an existing frame via LoadImage + VAEEncode (compatible with ComfyUI v0.3.26)."""
        patched["21"] = {
            "class_type": "LoadImage",
            "inputs": {"image": remote_anchor_rel},
        }
        patched["22"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["21", 0], "vae": ["3", 0]},
        }
        if "9" in patched:
            patched["9"]["inputs"]["latent_image"] = ["22", 0]
            patched["9"]["inputs"]["denoise"] = denoise
        patched.pop("8", None)

    @staticmethod
    def _bypass_missing_flux_lora_repair(
        patched: dict[str, Any], lora_rel: str, repair_engine: Literal["kontext", "fill"]
    ) -> None:
        """When repairing a frame/keyframe for a non-LoRA character, bypass LoraLoader nodes 11 & 13."""
        if "missing_" not in lora_rel:
            return
        for nid in ("14", "15"):
            if nid in patched:
                patched[nid]["inputs"]["clip"] = ["8", 0]
        if "17" in patched:
            patched["17"]["inputs"]["model"] = ["12", 0] if repair_engine == "fill" else ["10", 0]
        patched.pop("11", None)
        patched.pop("13", None)

    async def generate_character_portrait(
        self,
        *,
        project_name: str,
        char_id: str,
        prompt: str | None = None,
        seed: int = 42,
        lora_strength: float = 0.85,
        width: int = 1280,
        height: int = 720,
    ) -> dict[str, Any]:
        """Generate or update a Master Character Portrait used to anchor character consistency across scenes."""
        pdir = self.project_dir(project_name)
        bible_path = pdir / "project.json"
        char_name = char_id.removeprefix("char_")
        appearance = ""
        wardrobe = ""
        if bible_path.exists():
            bible = json.loads(bible_path.read_text(encoding="utf-8"))
            for ch in bible.get("assets", {}).get("characters", []):
                if ch["id"] == char_id or ch.get("name", "").lower() == char_id.lower():
                    char_id = ch["id"]
                    char_name = ch.get("casting", {}).get("nairobiwood", {}).get("lora_name") or ch.get("name") or char_id
                    appearance = ch.get("appearance") or ""
                    wardrobe = ch.get("default_wardrobe") or appearance
                    break

        portrait_prompt = (
            prompt
            or f"Cinematic 35mm medium close-up master character portrait of {char_name}. {appearance}. Wardrobe: {wardrobe}. Neutral studio lighting, sharp focus on eyes and facial structure, Kodak Vision3 500T film grain."
        ).strip()

        cdir = pdir / "characters" / char_id
        cdir.mkdir(parents=True, exist_ok=True)
        master_local = cdir / "master_portrait.png"

        tunnel = self._ensure_cloud_ready()
        scratch_tag = f"mp_{project_name}_{char_id}_{uuid.uuid4().hex[:6]}"
        remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"

        try:
            lora_rel = await self._upload_character_lora(char_name, "flux", scratch_tag)
            workflow, node_map = self.comfy.load_workflow_and_map("wan22_keyframes")
            patched = self.comfy.patch_workflow(
                workflow,
                node_map,
                {
                    "character_flux_lora": lora_rel,
                    "character_flux_lora_end": lora_rel,
                    "lora_strength": lora_strength,
                    "lora_strength_clip": lora_strength,
                    "lora_strength_end": lora_strength,
                    "start_prompt": portrait_prompt,
                    "end_prompt": portrait_prompt,
                    "negative_prompt": "blurry, deformed hands, extra fingers, plastic skin, watermark",
                    "width": width,
                    "height": height,
                    "start_seed": seed,
                    "end_seed": seed + 1,
                    "start_prefix": f"scratch/{scratch_tag}/master_portrait",
                    "end_prefix": f"scratch/{scratch_tag}/unused_end",
                },
            )
            for end_nid in ("12", "13", "14", "15", "16", "17", "18", "19", "20", "901"):
                patched.pop(end_nid, None)
            self._bypass_missing_flux_lora_keyframes(patched, lora_rel)
            await self.comfy.queue_and_wait(patched)
            await self.comfy.free_vram()
            await tunnel.rsync_download(f"{remote_renders}/master_portrait*.png", master_local)
        finally:
            await tunnel.purge_remote(f"/dev/shm/factory_scratch/active_loras/{scratch_tag}")
            await tunnel.purge_remote(remote_renders)

        # Update project.json and all staged/existing shot manifests for this character
        if bible_path.exists():
            from comfy_factory_mcp.studio_bible import stage_bible_into_comfy_project

            bible = json.loads(bible_path.read_text(encoding="utf-8"))
            for ch in bible.get("assets", {}).get("characters", []):
                if ch["id"] == char_id:
                    ch.setdefault("casting", {})["master_portrait_path"] = str(master_local)
            stage_bible_into_comfy_project(bible, project_name, self.settings)

        return {
            "project_name": project_name,
            "char_id": char_id,
            "master_portrait_path": str(master_local),
            "prompt": portrait_prompt,
        }

    async def generate_shot_keyframes(
        self,
        *,
        project_name: str,
        shot_id: str,
        character_name: str,
        start_prompt: str,
        end_prompt: str | None = None,
        negative_prompt: str = "blurry, deformed hands, extra fingers, plastic skin, watermark",
        lora_strength: float = 0.85,
        seed: int = 42,
        width: int = 1280,
        height: int = 720,
        last_frame_handoff_shot_id: str | None = None,
    ) -> ShotManifest:
        """Generate Start Frame (and optional Kontext-anchored End Frame) for pre-render review."""
        sdir = self.shot_dir(project_name, shot_id)
        kf_dir = sdir / "keyframes"
        kf_dir.mkdir(parents=True, exist_ok=True)
        start_local = kf_dir / "start_frame.png"
        end_local = kf_dir / "end_frame.png"

        existing_manifest: ShotManifest | None = None
        if (sdir / "shot.manifest.json").exists():
            existing_manifest = self.load_shot_manifest(sdir)

        anchor_local: Path | None = None
        if existing_manifest and existing_manifest.keyframes.reference_anchor_path:
            cand = Path(existing_manifest.keyframes.reference_anchor_path)
            if cand.exists():
                anchor_local = cand

        tunnel = self._ensure_cloud_ready()
        scratch_tag = f"kf_{project_name}_{shot_id}_{uuid.uuid4().hex[:6]}"
        remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"

        try:
            if last_frame_handoff_shot_id:
                prev_dir = self.shot_dir(project_name, last_frame_handoff_shot_id)
                prev_manifest = self.load_shot_manifest(prev_dir)
                prev_master = Path(prev_manifest.master_video_path or prev_manifest.preview_video_path or "")
                info = probe_media_info(prev_master)
                last_idx = max(0, int(info["frame_count"]) - 1)
                extract_single_frame(prev_master, last_idx, start_local)

            lora_rel = await self._upload_character_lora(character_name, "flux", scratch_tag)
            workflow, node_map = self.comfy.load_workflow_and_map("wan22_keyframes")
            patched = self.comfy.patch_workflow(
                workflow,
                node_map,
                {
                    "character_flux_lora": lora_rel,
                    "character_flux_lora_end": lora_rel,
                    "lora_strength": lora_strength,
                    "lora_strength_clip": lora_strength,
                    "lora_strength_end": lora_strength,
                    "start_prompt": start_prompt,
                    "end_prompt": end_prompt or start_prompt,
                    "negative_prompt": negative_prompt,
                    "width": width,
                    "height": height,
                    "start_seed": seed,
                    "end_seed": seed + 1,
                    "start_prefix": f"scratch/{scratch_tag}/start_frame",
                    "end_prefix": f"scratch/{scratch_tag}/end_frame",
                },
            )
            if not end_prompt:
                for end_nid in ("12", "13", "14", "15", "16", "17", "18", "19", "20", "901"):
                    patched.pop(end_nid, None)
            self._bypass_missing_flux_lora_keyframes(patched, lora_rel)
            await self.comfy.queue_and_wait(patched)
            await self.comfy.free_vram()

            if not last_frame_handoff_shot_id:
                await tunnel.rsync_download(f"{remote_renders}/start_frame*.png", start_local)
            if end_prompt:
                await tunnel.rsync_download(f"{remote_renders}/end_frame*.png", end_local)
            elif end_local.exists():
                end_local.unlink()
        finally:
            await tunnel.purge_remote(f"/dev/shm/factory_scratch/active_loras/{scratch_tag}")
            await tunnel.purge_remote(remote_renders)

        kf_state = KeyframeState(
            status="awaiting_approval",
            start_frame_path=str(start_local),
            end_frame_path=str(end_local) if end_prompt else None,
            has_end_frame=bool(end_prompt),
            approved=False,
            start_prompt=start_prompt,
            end_prompt=end_prompt,
            negative_prompt=negative_prompt,
            reference_anchor_path=str(anchor_local) if anchor_local else (existing_manifest.keyframes.reference_anchor_path if existing_manifest else None),
            seed=seed,
            lora_strength=lora_strength,
            width=width,
            height=height,
        )

        if existing_manifest is not None:
            manifest = existing_manifest
            manifest.character_name = character_name
            if not manifest.video_motion_prompt:
                manifest.prompt = start_prompt
            manifest.negative_prompt = negative_prompt
            manifest.keyframes = kf_state
        else:
            manifest = ShotManifest(
                project_name=project_name,
                shot_id=shot_id,
                character_name=character_name,
                prompt=start_prompt,
                negative_prompt=negative_prompt,
                keyframes=kf_state,
            )
        self.save_shot_manifest(sdir, manifest)

        story = self.load_or_create_story_manifest(project_name)
        if shot_id not in story.shot_order:
            story.shot_order.append(shot_id)
            self.save_story_manifest(story)
        return manifest

    async def regenerate_shot_keyframe(
        self,
        *,
        project_name: str,
        shot_id: str,
        which: Literal["start", "end"] = "start",
        prompt: str | None = None,
        edit_suggestion: str | None = None,
        mask_image_path: str | None = None,
        mask_target_text: str | None = None,
        repair_engine: Literal["kontext", "fill"] = "kontext",
        denoise: float = 0.65,
        seed: int | None = None,
    ) -> ShotManifest:
        """Re-roll, rebuild with editing suggestions (Kontext-anchored), or mask-inpaint a Start or End keyframe."""
        sdir = self.shot_dir(project_name, shot_id)
        manifest = self.load_shot_manifest(sdir)
        tunnel = self._ensure_cloud_ready()
        scratch_tag = f"rkf_{project_name}_{shot_id}_{uuid.uuid4().hex[:6]}"
        remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"

        target_path = Path(
            manifest.keyframes.start_frame_path if which == "start" else (manifest.keyframes.end_frame_path or sdir / "keyframes" / "end_frame.png")
        )
        new_seed = seed if seed is not None else manifest.keyframes.seed + (7 if which == "start" else 13)
        base_prompt = (
            manifest.keyframes.start_prompt if which == "start" else (manifest.keyframes.end_prompt or manifest.keyframes.start_prompt)
        )
        if edit_suggestion and edit_suggestion.strip():
            active_prompt = edit_suggestion.strip()
        elif prompt and prompt.strip():
            active_prompt = prompt.strip()
        else:
            active_prompt = base_prompt

        try:
            lora_rel = await self._upload_character_lora(manifest.character_name, "flux", scratch_tag)
            if (mask_image_path or mask_target_text) and target_path.exists():
                await tunnel.rsync_upload(target_path, remote_renders)
                mask_rel = f"scratch/{scratch_tag}/unused_mask.png"
                if mask_image_path:
                    safe_mask = resolve_safe_path(mask_image_path, self.settings, must_exist=True)
                    await tunnel.rsync_upload(safe_mask, remote_renders)
                    mask_rel = f"scratch/{scratch_tag}/{safe_mask.name}"
                wf, nm = self.comfy.load_workflow_and_map("wan22_frame_repair")
                capped_fill = min(0.50, manifest.keyframes.lora_strength)
                patched = self.comfy.patch_workflow(
                    wf,
                    nm,
                    {
                        "frame_raw_image": f"scratch/{scratch_tag}/{target_path.name}",
                        "mask_override_image": mask_rel,
                        "mask_target_text": mask_target_text or "selected area",
                        "repair_prompt": active_prompt,
                        "character_flux_lora_kontext": lora_rel,
                        "character_flux_lora_fill": lora_rel,
                        "lora_strength_kontext": manifest.keyframes.lora_strength,
                        "lora_strength_fill": capped_fill,
                        "inpaint_denoise": denoise,
                        "seed": new_seed,
                        "output_prefix": f"scratch/{scratch_tag}/kf_{which}",
                    },
                )
                if repair_engine == "fill":
                    patched["16"]["inputs"]["positive"] = ["14", 0]
                    patched["17"]["inputs"]["model"] = ["13", 0]
                if mask_image_path:
                    # Use explicit brushed mask and skip Florence-2/SAM2 nodes
                    patched["7"]["inputs"]["mask"] = ["2", 0]
                    for auto_nid in ("3", "4", "5", "6"):
                        patched.pop(auto_nid, None)
                else:
                    patched.pop("2", None)
                self._bypass_missing_flux_lora_repair(patched, lora_rel, repair_engine)
                await self.comfy.queue_and_wait(patched)
                await tunnel.rsync_download(f"{remote_renders}/kf_{which}*.png", target_path)
            else:
                wf, nm = self.comfy.load_workflow_and_map("wan22_keyframes")
                patched = self.comfy.patch_workflow(
                    wf,
                    nm,
                    {
                        "character_flux_lora": lora_rel,
                        "character_flux_lora_end": lora_rel,
                        "lora_strength": manifest.keyframes.lora_strength,
                        "start_prompt": active_prompt if which == "start" else manifest.keyframes.start_prompt,
                        "end_prompt": active_prompt if which == "end" else (manifest.keyframes.end_prompt or active_prompt),
                        "negative_prompt": manifest.keyframes.negative_prompt,
                        "width": manifest.keyframes.width,
                        "height": manifest.keyframes.height,
                        "start_seed": new_seed if which == "start" else manifest.keyframes.seed,
                        "end_seed": new_seed if which == "end" else manifest.keyframes.seed + 1,
                        "start_prefix": f"scratch/{scratch_tag}/start_frame",
                        "end_prefix": f"scratch/{scratch_tag}/end_frame",
                    },
                )
                if which == "start" and not manifest.keyframes.has_end_frame:
                    for end_nid in ("12", "13", "14", "15", "16", "17", "18", "19", "20", "901"):
                        patched.pop(end_nid, None)

                # If editing suggestion is provided on an existing frame, anchor rebuild to the current frame via VAEEncode
                if edit_suggestion and target_path.exists() and which == "start":
                    await tunnel.rsync_upload(target_path, remote_renders)
                    self._inject_kontext_start_anchor(
                        patched, f"scratch/{scratch_tag}/{target_path.name}", denoise=denoise
                    )

                self._bypass_missing_flux_lora_keyframes(patched, lora_rel)
                await self.comfy.queue_and_wait(patched)
                file_glob = "start_frame*.png" if which == "start" else "end_frame*.png"
                await tunnel.rsync_download(f"{remote_renders}/{file_glob}", target_path)
        finally:
            await tunnel.purge_remote(f"/dev/shm/factory_scratch/active_loras/{scratch_tag}")
            await tunnel.purge_remote(remote_renders)

        if which == "start":
            if prompt and not edit_suggestion:
                manifest.keyframes.start_prompt = active_prompt
            manifest.keyframes.start_frame_path = str(target_path)
            manifest.keyframes.seed = new_seed
        else:
            if prompt and not edit_suggestion:
                manifest.keyframes.end_prompt = active_prompt
            manifest.keyframes.end_frame_path = str(target_path)
            manifest.keyframes.has_end_frame = True
        if edit_suggestion:
            manifest.keyframes.last_edit_suggestion = edit_suggestion
        manifest.keyframes.status = "awaiting_approval"
        manifest.keyframes.approved = False
        manifest.keyframes.updated_at = utc_now_iso()
        self.save_shot_manifest(sdir, manifest)
        return manifest

    async def generate_scene_keyframes(
        self,
        *,
        project_name: str,
        scene_id: str | None = None,
        shot_ids: list[str] | None = None,
        force: bool = False,
    ) -> list[ShotManifest]:
        """Generate first frames across a scene (or the whole storyboard), updating each shot manifest as it completes."""
        story = self.load_or_create_story_manifest(project_name)
        candidates = shot_ids or story.shot_order
        completed: list[ShotManifest] = []
        for sid in candidates:
            sdir = self.shot_dir(project_name, sid)
            if not (sdir / "shot.manifest.json").exists():
                continue
            manifest = self.load_shot_manifest(sdir)
            if scene_id and manifest.scene_id != scene_id:
                continue
            has_existing = bool(
                manifest.keyframes.start_frame_path and Path(manifest.keyframes.start_frame_path).exists()
            )
            if has_existing and not force:
                completed.append(manifest)
                continue
            updated = await self.generate_shot_keyframes(
                project_name=project_name,
                shot_id=sid,
                character_name=manifest.character_name,
                start_prompt=manifest.keyframes.start_prompt,
                end_prompt=manifest.keyframes.end_prompt if manifest.keyframes.has_end_frame else None,
                negative_prompt=manifest.keyframes.negative_prompt,
                lora_strength=manifest.keyframes.lora_strength,
                seed=manifest.keyframes.seed,
                width=manifest.keyframes.width,
                height=manifest.keyframes.height,
            )
            completed.append(updated)
        return completed

    async def approve_and_render_video(
        self,
        *,
        project_name: str,
        shot_id: str,
        prompt: str | None = None,
        duration_sec: float = 5.0,
        driving_video_path: str | None = None,
        audio_wav_path: str | None = None,
        use_fast_distill_lora: bool = False,
        lora_strength: float | None = None,
        seed: int | None = None,
    ) -> ShotManifest:
        """Approve Start/End keyframes, unload FLUX VRAM, and render Wan 2.2 MoE video."""
        sdir = self.shot_dir(project_name, shot_id)
        manifest = self.load_shot_manifest(sdir)
        if not manifest.keyframes.start_frame_path or not Path(manifest.keyframes.start_frame_path).exists():
            raise FileNotFoundError("Start keyframe must exist before approving video render.")

        tunnel = self._ensure_cloud_ready()
        await self.comfy.free_vram(unload_models=True, free_memory=True)

        effective_duration = duration_sec
        if audio_wav_path:
            safe_wav = resolve_safe_path(audio_wav_path, self.settings, must_exist=True)
            wav_info = probe_media_info(safe_wav)
            if float(wav_info["duration_sec"]) > 0:
                effective_duration = max(duration_sec, float(wav_info["duration_sec"]))

        wan_frames = compute_wan_shot_frames(effective_duration, fps=24)
        active_seed = seed if seed is not None else manifest.keyframes.seed
        active_lora_strength = lora_strength if lora_strength is not None else manifest.keyframes.lora_strength
        active_prompt = prompt or manifest.video_motion_prompt or manifest.prompt or manifest.keyframes.start_prompt

        scratch_tag = f"vid_{project_name}_{shot_id}_{uuid.uuid4().hex[:6]}"
        remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"

        master_local = sdir / f"{shot_id}_master.mkv"
        preview_local = sdir / f"{shot_id}_preview.mp4"
        strip_local = sdir / f"{shot_id}_contact_strip.png"
        last_frame_local = sdir / f"{shot_id}_last_frame.png"

        try:
            start_local = Path(manifest.keyframes.start_frame_path)
            start_rel = await self.comfy.upload_image(start_local, subfolder=f"scratch/{scratch_tag}")
            end_rel = start_rel
            if manifest.keyframes.end_frame_path and Path(manifest.keyframes.end_frame_path).exists():
                end_local = Path(manifest.keyframes.end_frame_path)
                end_rel = await self.comfy.upload_image(end_local, subfolder=f"scratch/{scratch_tag}")

            driving_rel = f"scratch/{scratch_tag}/unused.mp4"
            fun_strength = 0.0
            if driving_video_path:
                safe_drv = resolve_safe_path(driving_video_path, self.settings, must_exist=True)
                await tunnel.rsync_upload(safe_drv, remote_renders)
                driving_rel = f"scratch/{scratch_tag}/{safe_drv.name}"
                fun_strength = 0.85

            lora_rel = await self._upload_character_lora(manifest.character_name, "wan22", scratch_tag)
            steps = 4 if use_fast_distill_lora else 24
            handoff = 2 if use_fast_distill_lora else 12
            lightx2v = 1.0 if use_fast_distill_lora else 0.0

            wf, nm = self.comfy.load_workflow_and_map("wan22_video_from_keyframes")
            patched = self.comfy.patch_workflow(
                wf,
                nm,
                {
                    "start_image": start_rel,
                    "end_image": end_rel,
                    "positive_prompt": active_prompt,
                    "negative_prompt": manifest.negative_prompt,
                    "character_wan22_lora": lora_rel,
                    "lora_strength": active_lora_strength,
                    "lightx2v_high_strength": lightx2v,
                    "lightx2v_low_strength": lightx2v,
                    "driving_video": driving_rel,
                    "fun_control_strength": fun_strength,
                    "width": manifest.keyframes.width,
                    "height": manifest.keyframes.height,
                    "wan_frames": wan_frames,
                    "seed_high": active_seed,
                    "seed_low": active_seed,
                    "steps_high": steps,
                    "steps_low": steps,
                    "handoff_step_high": handoff,
                    "handoff_step_low": handoff,
                    "lossless_prefix": f"scratch/{scratch_tag}/shot_master",
                    "preview_prefix": f"scratch/{scratch_tag}/shot_preview",
                },
            )
            if "missing_" in lora_rel and "10" in patched and "11" in patched:
                patched["11"]["inputs"]["model"] = patched["10"]["inputs"]["model"]
                patched.pop("10", None)
            if not use_fast_distill_lora:
                if "9" in patched and "12" in patched:
                    patched["12"]["inputs"]["model"] = patched["9"]["inputs"]["model"]
                    patched.pop("9", None)
                if "11" in patched and "13" in patched:
                    patched["13"]["inputs"]["model"] = patched["11"]["inputs"]["model"]
                    patched.pop("11", None)
            if not driving_video_path:
                if "19" in patched:
                    patched["19"]["inputs"]["positive"] = ["17", 0]
                    patched["19"]["inputs"]["negative"] = ["17", 1]
                if "20" in patched:
                    patched["20"]["inputs"]["positive"] = ["17", 0]
                    patched["20"]["inputs"]["negative"] = ["17", 1]
                for fun_nid in ("14", "15", "16", "18"):
                    patched.pop(fun_nid, None)
            await self.comfy.queue_and_wait(patched)
            await tunnel.rsync_download(f"{remote_renders}/shot_master*.mkv", master_local)
            await tunnel.rsync_download(f"{remote_renders}/shot_preview*.mp4", preview_local)
        finally:
            await tunnel.purge_remote(f"/dev/shm/factory_scratch/active_loras/{scratch_tag}")
            await tunnel.purge_remote(remote_renders)

        info = probe_media_info(master_local)
        manifest.keyframes.approved = True
        manifest.keyframes.status = "approved"
        manifest.prompt = active_prompt
        manifest.video_motion_prompt = active_prompt
        manifest.duration_sec = float(info["duration_sec"] or effective_duration)
        manifest.total_frames = int(info["frame_count"] or wan_frames)
        manifest.master_video_path = str(master_local)
        manifest.preview_video_path = str(preview_local)
        build_contact_strip(master_local, strip_local, num_samples=6)
        extract_single_frame(master_local, max(0, manifest.total_frames - 1), last_frame_local)
        manifest.contact_strip_path = str(strip_local)
        manifest.last_frame_path = str(last_frame_local)
        self.save_shot_manifest(sdir, manifest)
        return manifest

    async def extract_and_repair_frame(
        self,
        *,
        source_video_path: str,
        frame_index: int,
        character_name: str,
        repair_prompt: str,
        mask_target_text: str | None = None,
        mask_image_path: str | None = None,
        repair_engine: Literal["kontext", "fill"] = "kontext",
        inpaint_denoise: float = 0.65,
        project_name: str | None = None,
        shot_id: str | None = None,
    ) -> BranchCheckpoint:
        """Slice lossless prefix 0..K-1 locally, repair Frame K on RunPod, and compute seam score."""
        safe_video = resolve_safe_path(source_video_path, self.settings, must_exist=True)
        info = probe_media_info(safe_video)
        total_frames = int(info["frame_count"])
        fps = float(info["fps"] or 24.0)
        if not (0 < frame_index < total_frames):
            raise ValueError(f"frame_index={frame_index} must satisfy 0 < frame_index < {total_frames}.")

        ckpt_id = f"f{frame_index:04d}_checkpoint"
        if project_name and shot_id:
            ckpt_dir = self.shot_dir(project_name, shot_id) / "branches" / ckpt_id
        else:
            ckpt_dir = safe_video.parent / "branches" / ckpt_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        prefix_mkv = ckpt_dir / f"locked_prefix_0000_{frame_index - 1:04d}.mkv"
        seam_ref_png = ckpt_dir / f"frame_{frame_index - 1:04d}_seam_ref.png"
        raw_png = ckpt_dir / f"frame_{frame_index:04d}_raw.png"
        repaired_png = ckpt_dir / f"frame_{frame_index:04d}_repaired.png"

        slice_lossless_prefix(safe_video, frame_index, prefix_mkv)
        extract_single_frame(safe_video, frame_index - 1, seam_ref_png)
        extract_single_frame(safe_video, frame_index, raw_png)

        tunnel = self._ensure_cloud_ready()
        scratch_tag = f"rep_{ckpt_id}_{uuid.uuid4().hex[:6]}"
        remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"

        try:
            await tunnel.rsync_upload(raw_png, remote_renders)
            lora_rel = await self._upload_character_lora(character_name, "flux", scratch_tag)
            mask_rel = f"scratch/{scratch_tag}/unused_mask.png"
            if mask_image_path:
                safe_mask = resolve_safe_path(mask_image_path, self.settings, must_exist=True)
                await tunnel.rsync_upload(safe_mask, remote_renders)
                mask_rel = f"scratch/{scratch_tag}/{safe_mask.name}"

            wf, nm = self.comfy.load_workflow_and_map("wan22_frame_repair")
            patched = self.comfy.patch_workflow(
                wf,
                nm,
                {
                    "frame_raw_image": f"scratch/{scratch_tag}/{raw_png.name}",
                    "mask_override_image": mask_rel,
                    "mask_target_text": mask_target_text or "subject face and hands",
                    "character_flux_lora_kontext": lora_rel,
                    "character_flux_lora_fill": lora_rel,
                    "lora_strength_kontext": 0.85,
                    "lora_strength_fill": 0.45,
                    "repair_prompt": repair_prompt,
                    "inpaint_denoise": inpaint_denoise,
                    "output_prefix": f"scratch/{scratch_tag}/frame_repaired",
                },
            )
            if mask_image_path:
                patched["7"]["inputs"]["mask"] = ["2", 0]
                for auto_nid in ("3", "4", "5", "6"):
                    patched.pop(auto_nid, None)
            else:
                patched.pop("2", None)
            if repair_engine == "fill":
                patched["17"]["inputs"]["model"] = ["13", 0]
            self._bypass_missing_flux_lora_repair(patched, lora_rel, repair_engine)
            await self.comfy.queue_and_wait(patched)
            await tunnel.rsync_download(f"{remote_renders}/frame_repaired*.png", repaired_png)
        finally:
            await tunnel.purge_remote(f"/dev/shm/factory_scratch/active_loras/{scratch_tag}")
            await tunnel.purge_remote(remote_renders)

        seam_score = compute_seam_continuity_score(seam_ref_png, repaired_png)
        checkpoint = BranchCheckpoint(
            checkpoint_id=ckpt_id,
            checkpoint_dir=str(ckpt_dir),
            source_video_path=str(safe_video),
            frame_index=frame_index,
            total_source_frames=total_frames,
            fps=fps,
            locked_prefix_path=str(prefix_mkv),
            seam_ref_frame_path=str(seam_ref_png),
            raw_frame_path=str(raw_png),
            repaired_frame_path=str(repaired_png),
            mask_path=mask_image_path,
            repair_engine=repair_engine,
            repair_prompt=repair_prompt,
            seam_continuity_score=seam_score,
        )
        (ckpt_dir / "checkpoint.json").write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")

        if project_name and shot_id:
            sdir = self.shot_dir(project_name, shot_id)
            if (sdir / "shot.manifest.json").exists():
                manifest = self.load_shot_manifest(sdir)
                manifest.checkpoints[ckpt_id] = checkpoint
                self.save_shot_manifest(sdir, manifest)

        return checkpoint

    async def continue_from_repaired_frame(
        self,
        *,
        checkpoint_dir: str,
        character_name: str,
        mode: Literal["carry", "pivot", "lipsync"],
        continuation_prompt: str | None = None,
        audio_wav_path: str | None = None,
        end_keyframe_prompt: str | None = None,
        driving_video_path: str | None = None,
        target_total_frames: int | None = None,
        lora_strength: float = 0.85,
        seed: int = 42,
    ) -> BranchRecord:
        """Continue forward from Frame K via Wan 2.2 (4k+1 padded) or LatentSync bypass."""
        ckpt_path = resolve_safe_path(checkpoint_dir, self.settings, must_exist=True)
        ckpt_json = ckpt_path / "checkpoint.json"
        if not ckpt_json.exists():
            raise FileNotFoundError(f"checkpoint.json missing in {ckpt_path}")
        checkpoint = BranchCheckpoint.model_validate_json(ckpt_json.read_text(encoding="utf-8"))

        if mode == "pivot" and not continuation_prompt:
            raise ValueError("continuation_prompt is required when mode='pivot'.")
        if mode == "lipsync" and not audio_wav_path:
            raise ValueError("audio_wav_path is required when mode='lipsync'.")

        tunnel = self._ensure_cloud_ready()
        await self.comfy.free_vram(unload_models=True, free_memory=True)

        desired_total = target_total_frames or checkpoint.total_source_frames
        remaining_frames = max(1, desired_total - checkpoint.frame_index)
        branch_id = f"{mode}_{uuid.uuid4().hex[:6]}"
        branch_master = ckpt_path / f"branch_{branch_id}_master.mkv"
        branch_preview = ckpt_path / f"branch_{branch_id}_preview.mp4"
        tail_local = ckpt_path / f".tail_{branch_id}.mkv"

        scratch_tag = f"cont_{branch_id}"
        remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"
        wan_frames_req: int | None = None
        safe_wav: Path | None = None

        try:
            if mode == "lipsync":
                safe_wav = resolve_safe_path(audio_wav_path or "", self.settings, must_exist=True)
                wav_info = probe_media_info(safe_wav)
                wav_frames = max(1, round(float(wav_info["duration_sec"]) * checkpoint.fps))
                remaining_frames = max(remaining_frames, wav_frames)

                raw_tail = ckpt_path / f".src_tail_{branch_id}.mkv"
                slice_lossless_tail(Path(checkpoint.source_video_path), checkpoint.frame_index, raw_tail)
                await tunnel.rsync_upload(raw_tail, remote_renders)
                await tunnel.rsync_upload(safe_wav, remote_renders)
                wf, nm = self.comfy.load_workflow_and_map("wan22_frame_continue")
                patched = self.comfy.patch_workflow(
                    wf,
                    nm,
                    {
                        "lipsync_tail_video": f"scratch/{scratch_tag}/{raw_tail.name}",
                        "lipsync_audio_wav": f"scratch/{scratch_tag}/{safe_wav.name}",
                        "lipsync_seed": seed,
                        "lipsync_prefix": f"scratch/{scratch_tag}/branch_lipsync",
                    },
                )
                # Strip Wan 2.2 nodes so ComfyUI only executes the LatentSync bypass subgraph
                for wan_node_id in [str(i) for i in range(1, 19)]:
                    patched.pop(wan_node_id, None)
                await self.comfy.queue_and_wait(patched)
                await tunnel.rsync_download(f"{remote_renders}/branch_lipsync*.mkv", tail_local)
                raw_tail.unlink(missing_ok=True)
            else:
                wan_frames_req, keep_tail = compute_wan_continuation_frames(remaining_frames)
                repaired_local = Path(checkpoint.repaired_frame_path or checkpoint.raw_frame_path)
                await tunnel.rsync_upload(repaired_local, remote_renders)
                lora_rel = await self._upload_character_lora(character_name, "wan22", scratch_tag)
                wf, nm = self.comfy.load_workflow_and_map("wan22_frame_continue")
                patched = self.comfy.patch_workflow(
                    wf,
                    nm,
                    {
                        "repaired_start_image": f"scratch/{scratch_tag}/{repaired_local.name}",
                        "pivot_end_image": f"scratch/{scratch_tag}/{repaired_local.name}",
                        "continuation_prompt": continuation_prompt or checkpoint.repair_prompt or "smooth natural motion",
                        "character_wan22_lora": lora_rel,
                        "lora_strength": lora_strength,
                        "wan_frames": wan_frames_req,
                        "trim_remaining_frames": keep_tail,
                        "seed_high": seed,
                        "seed_low": seed,
                        "tail_lossless_prefix": f"scratch/{scratch_tag}/branch_tail",
                    },
                )
                # Strip LatentSync nodes so ComfyUI only executes the Wan 2.2 MoE continuation subgraph
                for ls_node_id in ("19", "20", "21", "22"):
                    patched.pop(ls_node_id, None)
                if "missing_" in lora_rel and "9" in patched and "11" in patched:
                    patched["11"]["inputs"]["model"] = patched["9"]["inputs"]["model"]
                    patched.pop("9", None)
                await self.comfy.queue_and_wait(patched)
                await tunnel.rsync_download(f"{remote_renders}/branch_tail*.mkv", tail_local)
        finally:
            await tunnel.purge_remote(f"/dev/shm/factory_scratch/active_loras/{scratch_tag}")
            await tunnel.purge_remote(remote_renders)

        audio_offset = checkpoint.frame_index / max(checkpoint.fps, 1.0)
        concat_prefix_and_tail(
            Path(checkpoint.locked_prefix_path),
            tail_local,
            branch_master,
            branch_preview,
            audio_wav_path=safe_wav,
            audio_offset_sec=audio_offset,
        )
        tail_local.unlink(missing_ok=True)

        record = BranchRecord(
            branch_id=branch_id,
            mode=mode,
            continuation_prompt=continuation_prompt,
            end_keyframe_prompt=end_keyframe_prompt,
            audio_wav_path=audio_wav_path,
            driving_video_path=driving_video_path,
            branched_master_path=str(branch_master),
            branched_preview_path=str(branch_preview),
            wan_frames_requested=wan_frames_req,
            tail_frames_kept=remaining_frames,
            seed=seed,
        )
        checkpoint.branches.append(record)
        ckpt_json.write_text(checkpoint.model_dump_json(indent=2), encoding="utf-8")
        return record

    async def export_or_compose_story(
        self,
        *,
        project_name: str,
        shot_ids: list[str] | None = None,
        apply_rife: bool = True,
        target_fps: int = 48,
        add_mmaudio_prompt: str | None = None,
    ) -> Path:
        """Assemble project shots into a unified master cut, optionally applying RIFE + MMAudio."""
        story = self.load_or_create_story_manifest(project_name)
        chosen_ids = shot_ids or story.shot_order
        if not chosen_ids:
            raise ValueError(f"No shots found in project '{project_name}'.")

        inputs: list[Path] = []
        for sid in chosen_ids:
            manifest = self.load_shot_manifest(self.shot_dir(project_name, sid))
            candidate = Path(manifest.master_video_path or manifest.preview_video_path or "")
            if not candidate.exists():
                raise FileNotFoundError(f"Shot '{sid}' has no rendered video.")
            inputs.append(candidate)

        pdir = self.project_dir(project_name)
        concat_list = pdir / ".concat_shots.txt"
        raw_cut = pdir / f"{project_name}_raw_cut.mkv"
        final_cut = pdir / f"{project_name}_master_cut.mp4"

        concat_list.write_text(
            "\n".join(f"file '{p}'" for p in inputs) + "\n", encoding="utf-8"
        )
        try:
            subprocess.run(
                [FFMPEG_BIN, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(concat_list), "-c:v", "ffv1", str(raw_cut)],
                check=True,
            )
            if self.tunnel is not None and (apply_rife or add_mmaudio_prompt):
                scratch_tag = f"exp_{project_name}_{uuid.uuid4().hex[:6]}"
                remote_renders = f"/dev/shm/factory_scratch/temp_renders/{scratch_tag}"
                try:
                    info = probe_media_info(raw_cut)
                    await self.tunnel.rsync_upload(raw_cut, remote_renders)
                    wf, nm = self.comfy.load_workflow_and_map("wan22_post_export")
                    patched = self.comfy.patch_workflow(
                        wf,
                        nm,
                        {
                            "master_video": f"scratch/{scratch_tag}/{raw_cut.name}",
                            "rife_multiplier": 2 if apply_rife else 1,
                            "mmaudio_duration": float(info["duration_sec"] or 5.0),
                            "mmaudio_prompt": add_mmaudio_prompt or "subtle cinematic room tone",
                            "export_fps": target_fps if apply_rife else int(info["fps"] or 24),
                            "export_prefix": f"scratch/{scratch_tag}/final_export",
                        },
                    )
                    await self.comfy.queue_and_wait(patched)
                    await self.tunnel.rsync_download(f"{remote_renders}/final_export*.mp4", final_cut)
                finally:
                    await self.tunnel.purge_remote(remote_renders)
            else:
                subprocess.run(
                    [FFMPEG_BIN, "-y", "-v", "error", "-i", str(raw_cut), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "17", str(final_cut)],
                    check=True,
                )
        finally:
            concat_list.unlink(missing_ok=True)
            raw_cut.unlink(missing_ok=True)

        story.master_cut_path = str(final_cut)
        self.save_story_manifest(story)
        return final_cut
