"""Shared service layer backing both MCP tools (Antigravity) and HTTP REST (Browser Studio)."""

from __future__ import annotations

import base64
import io
import json
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Literal

from PIL import Image

from comfy_factory_mcp.cloud_providers import (
    RunPodProvider,
    clear_active_factory_state,
    load_active_factory_state,
    save_active_factory_state,
)
from comfy_factory_mcp.comfy_client import ComfyUIClient
from comfy_factory_mcp.config import FactorySettings, ensure_data_dirs, resolve_safe_path
from comfy_factory_mcp.lora_trainer import LoRATrainer
from comfy_factory_mcp.models import ActiveFactoryState, ShotManifest
from comfy_factory_mcp.qa_metrics import extract_single_frame, probe_media_info
from comfy_factory_mcp.shot_director import ShotDirector
from comfy_factory_mcp.ssh_tunnel import SSHTunnelManager
from comfy_factory_mcp.studio_bible import (
    parse_raw_story_into_bible,
    resolve_scene_continuity,
    stage_bible_into_comfy_project,
)


class FactoryService:
    """Unified controller for RunPod lifecycle, LoRA training, rendering, and Director UI state."""

    def __init__(self, settings: FactorySettings | None = None) -> None:
        self.settings = settings or FactorySettings()
        ensure_data_dirs(self.settings)
        self.comfy = ComfyUIClient(
            self.settings.resolved_workflows_dir(),
            base_url=f"http://127.0.0.1:{self.settings.comfy_local_port}",
        )

    def _active_tunnel(self) -> SSHTunnelManager | None:
        state = load_active_factory_state(self.settings)
        if state is None:
            return None
        return SSHTunnelManager(
            state.ssh_host,
            state.ssh_port,
            self.settings.resolved_ssh_key(),
            local_port=state.local_comfy_port,
        )

    def _director(self) -> ShotDirector:
        return ShotDirector(self.settings, self.comfy, tunnel=self._active_tunnel())

    async def factory_status(self) -> dict[str, Any]:
        state = load_active_factory_state(self.settings)
        if state is None:
            return {"active": False, "provider": "runpod", "comfy_url": None}
        tunnel_alive = SSHTunnelManager.is_pid_alive(state.tunnel_pid)
        comfy_stats: dict[str, Any] | None = None
        if tunnel_alive:
            try:
                comfy_stats = await self.comfy.system_stats()
            except Exception:  # noqa: BLE001
                comfy_stats = None
        return {
            "active": True,
            "provider": state.provider,
            "pod_id": state.pod_id,
            "gpu_name": state.gpu_name,
            "vram_gb": state.vram_gb,
            "hourly_rate_usd": state.hourly_rate_usd,
            "tunnel_alive": tunnel_alive,
            "comfy_reachable": comfy_stats is not None,
            "comfy_url": f"http://127.0.0.1:{state.local_comfy_port}",
            "network_volume_id": state.network_volume_id,
        }

    async def factory_up(
        self,
        gpu_filter: str = "A100|H100|RTX PRO 6000|L40S|RTX 6000 Ada",
        max_price_hr: float = 2.29,
        network_volume_id: str | None = None,
        bootstrap_models: bool = True,
    ) -> dict[str, Any]:
        ssh_pub = self.settings.resolved_ssh_key().with_suffix(".pub")
        if not ssh_pub.exists():
            raise FileNotFoundError(f"SSH public key missing: {ssh_pub}")
        vol_id = network_volume_id if network_volume_id is not None else self.settings.runpod_network_volume_id
        provider = RunPodProvider(self.settings.runpod_api_key)
        gpu = await provider.find_best_gpu(gpu_filter, max_price_hr)
        pod = await provider.provision_pod(
            gpu_id=gpu["id"],
            image_name=self.settings.container_image,
            ssh_public_key=ssh_pub.read_text(encoding="utf-8"),
            hf_token=self.settings.hf_token,
            network_volume_id=vol_id,
            datacenter_id=self.settings.runpod_datacenter_id,
        )
        # Persist provisional state immediately so factory_down can terminate orphan pods on failure
        state = ActiveFactoryState(
            pod_id=pod["id"],
            gpu_name=gpu.get("displayName", gpu["id"]),
            vram_gb=int(gpu.get("memoryInGb", 48)),
            hourly_rate_usd=float(gpu["effective_price"]),
            ssh_host="",
            ssh_port=22,
            local_comfy_port=self.settings.comfy_local_port,
            tunnel_pid=None,
            network_volume_id=vol_id or None,
            datacenter_id=self.settings.runpod_datacenter_id or None,
        )
        save_active_factory_state(state, self.settings)

        ssh_host, ssh_port = await provider.wait_for_ssh_endpoint(pod["id"])
        state.ssh_host = ssh_host
        state.ssh_port = ssh_port
        save_active_factory_state(state, self.settings)

        tunnel = SSHTunnelManager(
            ssh_host,
            ssh_port,
            self.settings.resolved_ssh_key(),
            local_port=self.settings.comfy_local_port,
        )
        await tunnel.bootstrap_remote_host(
            self.settings.nairobiwood_root / "infra",
            bootstrap_models=bootstrap_models,
        )
        pid = tunnel.start_tunnel()
        await tunnel.wait_for_comfyui()
        state.tunnel_pid = pid
        save_active_factory_state(state, self.settings)
        return {**state.model_dump(), "bootstrap_models_started": bootstrap_models}

    async def factory_down(self) -> dict[str, Any]:
        state = load_active_factory_state(self.settings)
        if state is None:
            return {"status": "idle", "message": "No active factory pod."}
        SSHTunnelManager.stop_tunnel(state.tunnel_pid)
        if self.settings.runpod_api_key:
            provider = RunPodProvider(self.settings.runpod_api_key)
            await provider.terminate_pod(state.pod_id)
        clear_active_factory_state(self.settings)
        return {"status": "terminated", "pod_id": state.pod_id}

    async def list_local_assets(self) -> dict[str, Any]:
        datasets = [
            {"name": d.name, "image_count": len([f for f in d.iterdir() if f.is_file()])}
            for d in sorted(self.settings.datasets_dir.iterdir())
            if d.is_dir()
        ]
        loras = {
            d.name: [f.name for f in sorted(d.glob("*.safetensors"))]
            for d in sorted(self.settings.loras_dir.iterdir())
            if d.is_dir()
        }
        character_map: dict[str, dict[str, Any]] = {}
        for char_name, lora_files in loras.items():
            character_map[char_name] = {
                "name": char_name,
                "has_flux_lora": any("flux" in f.lower() for f in lora_files),
                "has_wan22_lora": any("wan" in f.lower() for f in lora_files),
                "lora_files": lora_files,
                "dataset_images": 0,
            }
        for ds in datasets:
            cname = ds["name"]
            if cname in character_map:
                character_map[cname]["dataset_images"] = ds["image_count"]
            else:
                character_map[cname] = {
                    "name": cname,
                    "has_flux_lora": False,
                    "has_wan22_lora": False,
                    "lora_files": [],
                    "dataset_images": ds["image_count"],
                }
        characters = [character_map[k] for k in sorted(character_map)]
        workflows = [f.name for f in sorted(self.settings.resolved_workflows_dir().glob("*_pipeline.json"))]
        projects: list[dict[str, Any]] = []
        for pdir in sorted(self.settings.outputs_dir.iterdir()):
            if not pdir.is_dir():
                continue
            shots_root = pdir / "shots"
            shot_ids = [s.name for s in sorted(shots_root.iterdir()) if s.is_dir()] if shots_root.exists() else []
            has_bible = (pdir / "project.json").exists()
            projects.append({"project_name": pdir.name, "shots": shot_ids, "has_bible": has_bible})
        return {
            "data_dir": str(self.settings.resolved_data_dir()),
            "characters": characters,
            "datasets": datasets,
            "loras": loras,
            "workflows": workflows,
            "projects": projects,
        }

    async def start_lora_training(
        self,
        *,
        character_name: str,
        local_dataset_dir: str,
        trigger_word: str,
        train_flux: bool = True,
        train_wan22: bool = True,
    ) -> dict[str, Any]:
        trainer = LoRATrainer(self.settings, tunnel=self._active_tunnel())
        record = await trainer.start_lora_training(
            character_name=character_name,
            local_dataset_dir=local_dataset_dir,
            trigger_word=trigger_word,
            train_flux=train_flux,
            train_wan22=train_wan22,
        )
        return record.model_dump()

    async def get_job_status(self, job_id: str, collect_if_complete: bool = False) -> dict[str, Any]:
        trainer = LoRATrainer(self.settings, tunnel=self._active_tunnel())
        record = await trainer.get_job_status(job_id)
        if collect_if_complete and record.status == "completed":
            record = await trainer.collect_job_artifacts(job_id)
        return record.model_dump()

    async def generate_shot_keyframes(self, **kwargs: Any) -> dict[str, Any]:
        manifest = await self._director().generate_shot_keyframes(**kwargs)
        return {"status": "awaiting_keyframe_approval", "manifest": manifest.model_dump()}

    async def generate_scene_keyframes(self, **kwargs: Any) -> dict[str, Any]:
        manifests = await self._director().generate_scene_keyframes(**kwargs)
        return {
            "status": "scene_keyframes_ready",
            "count": len(manifests),
            "shot_ids": [m.shot_id for m in manifests],
        }

    async def generate_character_portrait(self, **kwargs: Any) -> dict[str, Any]:
        res = await self._director().generate_character_portrait(**kwargs)
        return {"status": "portrait_ready", **res}

    async def import_or_init_story_bible(
        self,
        *,
        project_name: str,
        story_or_bible_path: str | None = None,
        raw_story_json: dict[str, Any] | str | None = None,
        title: str | None = None,
    ) -> dict[str, Any]:
        """Ingest a StudioMode project.json or raw Story JSON, resolve scene wardrobe/prop continuity, and stage the full storyboard."""
        source_dir: Path | None = None
        if story_or_bible_path:
            safe_src = resolve_safe_path(story_or_bible_path, self.settings, must_exist=True)
            source_dir = safe_src.parent
            payload = json.loads(safe_src.read_text(encoding="utf-8"))
        elif isinstance(raw_story_json, str):
            payload = json.loads(raw_story_json)
        elif isinstance(raw_story_json, dict):
            payload = raw_story_json
        else:
            # Re-sync existing project.json inside outputs/<project_name>/project.json
            existing_bible = self.settings.outputs_dir / project_name / "project.json"
            if not existing_bible.exists():
                raise ValueError(
                    "Provide story_or_bible_path or raw_story_json, or ensure outputs/<project>/project.json exists."
                )
            source_dir = existing_bible.parent
            payload = json.loads(existing_bible.read_text(encoding="utf-8"))

        if "scenes" in payload and "assets" in payload:
            bible = payload
            bible.setdefault("project", {})["id"] = project_name
            if title:
                bible["project"]["title"] = title
        else:
            bible = parse_raw_story_into_bible(payload, project_slug=project_name, title=title)

        staged = stage_bible_into_comfy_project(
            bible=bible,
            project_name=project_name,
            settings=self.settings,
            source_file_dir=source_dir,
        )
        return {"status": "bible_staged", **staged}

    async def regenerate_shot_keyframe(self, **kwargs: Any) -> dict[str, Any]:
        manifest = await self._director().regenerate_shot_keyframe(**kwargs)
        return {"status": "awaiting_keyframe_approval", "manifest": manifest.model_dump()}

    async def approve_and_render_video(self, **kwargs: Any) -> dict[str, Any]:
        manifest = await self._director().approve_and_render_video(**kwargs)
        return {"status": "rendered", "manifest": manifest.model_dump()}

    async def render_shot(
        self,
        *,
        project_name: str,
        shot_id: str,
        character_name: str,
        start_prompt: str,
        end_prompt: str | None = None,
        negative_prompt: str = "blurry, deformed hands, extra fingers, plastic skin, watermark",
        duration_sec: float = 5.0,
        driving_video_path: str | None = None,
        audio_wav_path: str | None = None,
        lora_strength: float = 0.85,
        seed: int = 42,
        require_keyframe_approval: bool = True,
        use_fast_distill_lora: bool = False,
        last_frame_handoff_shot_id: str | None = None,
    ) -> dict[str, Any]:
        director = self._director()
        manifest = await director.generate_shot_keyframes(
            project_name=project_name,
            shot_id=shot_id,
            character_name=character_name,
            start_prompt=start_prompt,
            end_prompt=end_prompt,
            negative_prompt=negative_prompt,
            lora_strength=lora_strength,
            seed=seed,
            last_frame_handoff_shot_id=last_frame_handoff_shot_id,
        )
        if require_keyframe_approval:
            return {
                "status": "awaiting_keyframe_approval",
                "message": "Start/End keyframes ready for review in Director Studio before video render.",
                "manifest": manifest.model_dump(),
            }
        rendered = await director.approve_and_render_video(
            project_name=project_name,
            shot_id=shot_id,
            prompt=start_prompt,
            duration_sec=duration_sec,
            driving_video_path=driving_video_path,
            audio_wav_path=audio_wav_path,
            use_fast_distill_lora=use_fast_distill_lora,
            lora_strength=lora_strength,
            seed=seed,
        )
        return {"status": "rendered", "manifest": rendered.model_dump()}

    async def extract_and_repair_frame(self, **kwargs: Any) -> dict[str, Any]:
        ckpt = await self._director().extract_and_repair_frame(**kwargs)
        return {"status": "frame_repaired", "checkpoint": ckpt.model_dump()}

    async def continue_from_repaired_frame(self, **kwargs: Any) -> dict[str, Any]:
        branch = await self._director().continue_from_repaired_frame(**kwargs)
        return {"status": "branch_rendered", "branch": branch.model_dump()}

    async def export_or_compose_story(self, **kwargs: Any) -> dict[str, Any]:
        cut_path = await self._director().export_or_compose_story(**kwargs)
        return {"status": "exported", "master_cut_path": str(cut_path)}

    async def inspect_or_patch_node(
        self,
        workflow_name: str,
        node_id: str,
        input_key: str,
        new_value: Any,
        save_copy_name: str | None = None,
    ) -> dict[str, Any]:
        wf, _ = self.comfy.load_workflow_and_map(workflow_name)
        if node_id not in wf:
            raise KeyError(f"Node '{node_id}' not found in {workflow_name}")
        wf[node_id]["inputs"][input_key] = new_value
        saved_path: str | None = None
        if save_copy_name:
            target = resolve_safe_path(
                self.settings.resolved_workflows_dir() / f"{save_copy_name}.json",
                self.settings,
                allowed_roots=[self.settings.resolved_workflows_dir()],
            )
            target.write_text(json.dumps(wf, indent=2), encoding="utf-8")
            saved_path = str(target)
        return {"workflow_name": workflow_name, "node_id": node_id, "input_key": input_key, "saved_path": saved_path}

    async def pull_live_workflow(self, output_name: str) -> dict[str, Any]:
        safe_out = resolve_safe_path(
            self.settings.resolved_workflows_dir() / f"{output_name.removesuffix('.json')}.json",
            self.settings,
            allowed_roots=[self.settings.resolved_workflows_dir()],
        )
        written = await self.comfy.pull_live_workflow(safe_out)
        return {"saved_path": str(written)}

    # --- App-Only & Studio Helper Methods ---

    async def get_studio_state(
        self, project_name: str | None = None, shot_id: str | None = None
    ) -> dict[str, Any]:
        assets = await self.list_local_assets()
        status = await self.factory_status()
        projects = assets["projects"]
        active_project = project_name or (projects[0]["project_name"] if projects else None)
        active_shot: str | None = shot_id
        manifest_data: dict[str, Any] | None = None
        bible_summary: dict[str, Any] | None = None
        storyboard: list[dict[str, Any]] = []

        if active_project:
            pdir = self.settings.outputs_dir / active_project
            bible_path = pdir / "project.json"
            if bible_path.exists():
                try:
                    raw_bible = json.loads(bible_path.read_text(encoding="utf-8"))
                    resolve_scene_continuity(raw_bible)
                    bible_summary = {
                        "title": raw_bible.get("project", {}).get("title") or active_project,
                        "logline": raw_bible.get("project", {}).get("logline") or "",
                        "workflow_stage": raw_bible.get("project", {}).get("workflow_stage") or "storyboarding",
                        "style_guide": raw_bible.get("style_guide", {}),
                        "characters": raw_bible.get("assets", {}).get("characters", []),
                        "props": raw_bible.get("assets", {}).get("props", []),
                        "scenes": [
                            {
                                "id": sc.get("id"),
                                "name": sc.get("name"),
                                "order": sc.get("order", idx),
                                "plot_summary": sc.get("plot_summary", ""),
                                "dop_direction": sc.get("dop_direction", {}),
                                "character_states": sc.get("character_states", []),
                                "shot_ids": [sh.get("id") for sh in sc.get("shots", [])],
                            }
                            for idx, sc in enumerate(raw_bible.get("scenes", []), start=1)
                        ],
                    }
                    # Merge bible characters into assets.characters so the UI dropdown always lists story characters
                    existing_cnames = {c["name"].lower() for c in assets["characters"]}
                    for bchar in bible_summary["characters"]:
                        cname = bchar.get("name") or bchar.get("id")
                        if cname and cname.lower() not in existing_cnames:
                            nw = bchar.get("casting", {}).get("nairobiwood", {})
                            assets["characters"].append(
                                {
                                    "name": cname,
                                    "char_id": bchar.get("id"),
                                    "mode": nw.get("mode", "kontext_portrait"),
                                    "master_portrait_path": bchar.get("casting", {}).get("master_portrait_path"),
                                    "has_flux_lora": False,
                                    "has_wan22_lora": False,
                                    "lora_files": [],
                                    "dataset_images": len(bchar.get("casting", {}).get("user_reference_photos") or []),
                                }
                            )
                except Exception:  # noqa: BLE001
                    bible_summary = None

            matching = next((p for p in projects if p["project_name"] == active_project), None)
            shot_ids = matching["shots"] if matching else []
            # Order shot_ids according to story_manifest.json if available
            sm_path = pdir / "story_manifest.json"
            if sm_path.exists():
                try:
                    sm = json.loads(sm_path.read_text(encoding="utf-8"))
                    ordered = [s for s in sm.get("shot_order", []) if s in shot_ids]
                    remaining = [s for s in shot_ids if s not in ordered]
                    shot_ids = ordered + remaining
                except Exception:  # noqa: BLE001
                    pass

            for sid in shot_ids:
                mp = pdir / "shots" / sid / "shot.manifest.json"
                if mp.exists():
                    sm_obj = ShotManifest.model_validate_json(mp.read_text(encoding="utf-8"))
                    storyboard.append(sm_obj.model_dump())

            if not active_shot and shot_ids:
                active_shot = shot_ids[0]
            if active_shot:
                m_path = pdir / "shots" / active_shot / "shot.manifest.json"
                if m_path.exists():
                    manifest_data = ShotManifest.model_validate_json(m_path.read_text(encoding="utf-8")).model_dump()

        return {
            "factory": status,
            "studio_url": f"http://127.0.0.1:{self.settings.studio_http_port}",
            "assets": assets,
            "active_project": active_project,
            "active_shot": active_shot,
            "shot_manifest": manifest_data,
            "bible": bible_summary,
            "storyboard": storyboard,
        }

    def get_media_data_url(self, file_path: str, max_width: int = 1280) -> dict[str, Any]:
        safe = resolve_safe_path(file_path, self.settings, must_exist=True)
        mime, _ = mimetypes.guess_type(str(safe))
        if mime and mime.startswith("image/"):
            with Image.open(safe) as im:
                if im.width > max_width:
                    ratio = max_width / im.width
                    im = im.resize((max_width, max(1, round(im.height * ratio))))
                buf = io.BytesIO()
                im.save(buf, format="PNG")
                encoded = base64.b64encode(buf.getvalue()).decode("ascii")
            return {"data_url": f"data:image/png;base64,{encoded}", "mime_type": "image/png", "path": str(safe)}
        raw = safe.read_bytes()
        encoded = base64.b64encode(raw).decode("ascii")
        resolved_mime = mime or "application/octet-stream"
        return {"data_url": f"data:{resolved_mime};base64,{encoded}", "mime_type": resolved_mime, "path": str(safe)}

    def extract_video_frame_data_url(self, video_path: str, frame_index: int) -> dict[str, Any]:
        safe_video = resolve_safe_path(video_path, self.settings, must_exist=True)
        info = probe_media_info(safe_video)
        total = int(info.get("frame_count") or 1)
        clamped = max(0, min(frame_index, total - 1))
        tmp_png = safe_video.parent / f".preview_f{clamped:04d}.png"
        extract_single_frame(safe_video, clamped, tmp_png)
        try:
            res = self.get_media_data_url(str(tmp_png))
        finally:
            tmp_png.unlink(missing_ok=True)
        res.update({"frame_index": clamped, "total_frames": total, "fps": info.get("fps", 24.0)})
        return res

    def save_painted_mask(
        self,
        *,
        project_name: str,
        shot_id: str,
        mask_data_url: str,
        label: str = "brush_mask",
    ) -> dict[str, str]:
        if "," not in mask_data_url:
            raise ValueError("Invalid data URL for mask PNG.")
        _, b64_part = mask_data_url.split(",", 1)
        raw_bytes = base64.b64decode(b64_part)
        sdir = self._director().shot_dir(project_name, shot_id)
        masks_dir = sdir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        out_path = masks_dir / f"{label}_{uuid.uuid4().hex[:6]}.png"
        with Image.open(io.BytesIO(raw_bytes)) as im:
            im.convert("L").save(out_path, format="PNG")
        return {"mask_image_path": str(out_path)}
