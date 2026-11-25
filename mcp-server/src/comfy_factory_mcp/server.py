"""FastMCP / MCPServer entrypoint exposing Nairobiwood tools and the Director Studio MCP App."""

from __future__ import annotations

import json
from typing import Any, Literal

from mcp.server import MCPServer
from mcp.types import CallToolResult, TextContent

from comfy_factory_mcp.config import FactorySettings
from comfy_factory_mcp.service import FactoryService
from comfy_factory_mcp.web_studio import ensure_background_studio_server, load_director_html

APP_URI = "ui://comfy-factory/director.html"
INLINE_META: dict[str, Any] = {"ui": {"resourceUri": APP_URI}}
PANE_META: dict[str, Any] = {
    "ui": {"resourceUri": APP_URI},
    "com.google.antigravity/ui": {
        "entrypoints": [
            {"type": "aux_pane", "title": "Director Studio"},
            {"type": "global", "title": "Director Studio"},
        ]
    },
}
APP_ONLY_META: dict[str, Any] = {"ui": {"visibility": ["app"]}}


def _tool_result(summary_text: str, payload: dict[str, Any]) -> CallToolResult:
    """Keep model-visible text compact while delivering structured state to the UI app."""
    return CallToolResult(
        content=[TextContent(type="text", text=summary_text)],
        structuredContent=payload,
    )


def create_mcp_server(settings: FactorySettings | None = None) -> tuple[MCPServer, FactoryService]:
    """Build and wire the Nairobiwood MCPServer instance and underlying FactoryService."""
    svc = FactoryService(settings=settings)
    mcp = MCPServer("comfy-factory")

    @mcp.resource(
        APP_URI,
        name="Nairobiwood Director Studio",
        description="Unified Start/End Keyframe Review and Pause-and-Pivot Timeline Studio",
        mime_type="text/html;profile=mcp-app",
        meta={"ui": {"prefersBorder": False}},
    )
    def director_studio_html() -> str:
        return load_director_html(svc.settings)

    @mcp.tool(
        name="open_director_studio",
        title="Director Studio",
        description=(
            "Open the unified Director Studio UI (Antigravity side pane and optional localhost browser "
            "server) to review Start/End keyframes before video creation or scrub/repair/branch clips."
        ),
        meta=PANE_META,
    )
    async def open_director_studio(
        project_name: str | None = None,
        shot_id: str | None = None,
        tab: Literal["keyframes", "timeline"] = "keyframes",
        start_browser_server: bool = True,
    ) -> CallToolResult:
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        if start_browser_server:
            state["studio_url"] = ensure_background_studio_server(svc)
        state["requested_tab"] = tab
        return _tool_result(
            f"Director Studio ready (project={state.get('active_project')}, "
            f"shot={state.get('active_shot')}, browser_url={state.get('studio_url')}).",
            state,
        )

    @mcp.tool(
        name="factory_status",
        description="Return active RunPod GPU status, hourly rate, SSH tunnel health, and ComfyUI reachability.",
    )
    async def factory_status() -> str:
        return json.dumps(await svc.factory_status(), indent=2)

    @mcp.tool(
        name="factory_up",
        description="Provision a RunPod Secure Cloud GPU pod (>=48GB VRAM, SSH-only), bootstrap ComfyUI over SSH, and start localhost:8188 tunnel.",
    )
    async def factory_up(
        gpu_filter: str = "A100|H100|RTX PRO 6000|L40S|RTX 6000 Ada",
        max_price_hr: float = 2.29,
        network_volume_id: str | None = None,
        bootstrap_models: bool = True,
    ) -> str:
        return json.dumps(
            await svc.factory_up(
                gpu_filter=gpu_filter,
                max_price_hr=max_price_hr,
                network_volume_id=network_volume_id,
                bootstrap_models=bootstrap_models,
            ),
            indent=2,
        )

    @mcp.tool(
        name="factory_down",
        description="Terminate the active RunPod GPU pod and close local SSH tunnels immediately.",
    )
    async def factory_down() -> str:
        return json.dumps(await svc.factory_down(), indent=2)

    @mcp.tool(
        name="list_local_assets",
        description="List local datasets, trained character LoRAs, workflows, and rendered shot projects.",
    )
    async def list_local_assets() -> str:
        return json.dumps(await svc.list_local_assets(), indent=2)

    @mcp.tool(
        name="start_lora_training",
        description="Upload 15-40 reference images to remote /dev/shm and launch detached FLUX + Wan 2.2 LoRA training.",
    )
    async def start_lora_training(
        character_name: str,
        local_dataset_dir: str,
        trigger_word: str,
        train_flux: bool = True,
        train_wan22: bool = True,
    ) -> str:
        res = await svc.start_lora_training(
            character_name=character_name,
            local_dataset_dir=local_dataset_dir,
            trigger_word=trigger_word,
            train_flux=train_flux,
            train_wan22=train_wan22,
        )
        return json.dumps(res, indent=2)

    @mcp.tool(
        name="get_job_status",
        description="Poll detached LoRA training progress and optionally Pull-and-Purge finished .safetensors weights.",
    )
    async def get_job_status(job_id: str, collect_if_complete: bool = True) -> str:
        return json.dumps(
            await svc.get_job_status(job_id=job_id, collect_if_complete=collect_if_complete),
            indent=2,
        )

    @mcp.tool(
        name="import_or_init_story_bible",
        title="Import Story or StudioMode Bible",
        description=(
            "Ingest a raw Story JSON or a StudioMode project.json into ~/ComfyFactory/outputs/<project_name>/, "
            "resolve scene-to-scene wardrobe & prop continuity, and stage the full storyboard in Director Studio."
        ),
        meta=INLINE_META,
    )
    async def import_or_init_story_bible(
        project_name: str,
        story_or_bible_path: str | None = None,
        raw_story_json: str | None = None,
        title: str | None = None,
    ) -> CallToolResult:
        res = await svc.import_or_init_story_bible(
            project_name=project_name,
            story_or_bible_path=story_or_bible_path,
            raw_story_json=raw_story_json,
            title=title,
        )
        state = await svc.get_studio_state(project_name=project_name)
        state["requested_tab"] = "keyframes"
        state["last_action"] = res
        return _tool_result(
            f"Staged '{project_name}' storyboard ({res['scene_count']} scenes, {res['staged_shot_count']} shots).",
            state,
        )

    @mcp.tool(
        name="generate_character_portrait",
        title="Generate Master Character Portrait",
        description="Generate or update a character's Master Portrait (used by FLUX.1-Kontext-dev to anchor identity across scenes).",
        meta=INLINE_META,
    )
    async def generate_character_portrait(
        project_name: str,
        char_id: str,
        prompt: str | None = None,
        seed: int = 42,
        lora_strength: float = 0.85,
    ) -> CallToolResult:
        res = await svc.generate_character_portrait(
            project_name=project_name,
            char_id=char_id,
            prompt=prompt,
            seed=seed,
            lora_strength=lora_strength,
        )
        state = await svc.get_studio_state(project_name=project_name)
        state["requested_tab"] = "keyframes"
        state["last_action"] = res
        return _tool_result(
            f"Master portrait ready for '{char_id}' at {res['master_portrait_path']}.",
            state,
        )

    @mcp.tool(
        name="generate_scene_keyframes",
        title="Generate Scene First Frames",
        description="Generate first frames (start_frame) for all shots in a scene (or whole storyboard) under Director supervision.",
        meta=INLINE_META,
    )
    async def generate_scene_keyframes(
        project_name: str,
        scene_id: str | None = None,
        shot_ids: list[str] | None = None,
        force: bool = False,
    ) -> CallToolResult:
        res = await svc.generate_scene_keyframes(
            project_name=project_name,
            scene_id=scene_id,
            shot_ids=shot_ids,
            force=force,
        )
        first_shot = res["shot_ids"][0] if res.get("shot_ids") else None
        state = await svc.get_studio_state(project_name=project_name, shot_id=first_shot)
        state["requested_tab"] = "keyframes"
        state["last_action"] = res
        return _tool_result(
            f"Generated {res['count']} keyframe(s) for '{project_name}' (scene={scene_id or 'all'}).",
            state,
        )

    @mcp.tool(
        name="generate_shot_keyframes",
        title="Generate Start/End Keyframes",
        description="Generate FLUX Start Frame (and optional Kontext-anchored End Frame) for pre-render review.",
        meta=INLINE_META,
    )
    async def generate_shot_keyframes(
        project_name: str,
        shot_id: str,
        character_name: str,
        start_prompt: str,
        end_prompt: str | None = None,
        negative_prompt: str = "blurry, deformed hands, extra fingers, plastic skin, watermark",
        lora_strength: float = 0.85,
        seed: int = 42,
        last_frame_handoff_shot_id: str | None = None,
    ) -> CallToolResult:
        res = await svc.generate_shot_keyframes(
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
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        state["requested_tab"] = "keyframes"
        state["last_action"] = res
        return _tool_result(
            f"Keyframes ready for '{project_name}/{shot_id}' (awaiting approval before video render).",
            state,
        )

    @mcp.tool(
        name="regenerate_shot_keyframe",
        title="Regenerate Keyframe",
        description="Re-roll, rebuild with editing suggestions (FLUX Kontext), or mask-inpaint (Brush/SAM2) a Start or End keyframe.",
        meta=INLINE_META,
    )
    async def regenerate_shot_keyframe(
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
    ) -> CallToolResult:
        res = await svc.regenerate_shot_keyframe(
            project_name=project_name,
            shot_id=shot_id,
            which=which,
            prompt=prompt,
            edit_suggestion=edit_suggestion,
            mask_image_path=mask_image_path,
            mask_target_text=mask_target_text,
            repair_engine=repair_engine,
            denoise=denoise,
            seed=seed,
        )
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        state["requested_tab"] = "keyframes"
        state["last_action"] = res
        return _tool_result(f"Regenerated {which} keyframe for '{project_name}/{shot_id}'.", state)

    @mcp.tool(
        name="approve_and_render_video",
        title="Approve & Render Video",
        description="Approve Start/End keyframes, unload FLUX VRAM, and render the Wan 2.2 MoE video shot.",
        meta=INLINE_META,
    )
    async def approve_and_render_video(
        project_name: str,
        shot_id: str,
        prompt: str | None = None,
        duration_sec: float = 5.0,
        driving_video_path: str | None = None,
        audio_wav_path: str | None = None,
        use_fast_distill_lora: bool = False,
        lora_strength: float | None = None,
        seed: int | None = None,
    ) -> CallToolResult:
        res = await svc.approve_and_render_video(
            project_name=project_name,
            shot_id=shot_id,
            prompt=prompt,
            duration_sec=duration_sec,
            driving_video_path=driving_video_path,
            audio_wav_path=audio_wav_path,
            use_fast_distill_lora=use_fast_distill_lora,
            lora_strength=lora_strength,
            seed=seed,
        )
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        state["requested_tab"] = "timeline"
        state["last_action"] = res
        return _tool_result(f"Rendered Wan 2.2 shot '{project_name}/{shot_id}'.", state)

    @mcp.tool(
        name="render_shot",
        title="Render Shot",
        description=(
            "Generate Start/End keyframes and either pause for review in Director Studio "
            "(require_keyframe_approval=True) or render Wan 2.2 immediately."
        ),
        meta=INLINE_META,
    )
    async def render_shot(
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
    ) -> CallToolResult:
        res = await svc.render_shot(
            project_name=project_name,
            shot_id=shot_id,
            character_name=character_name,
            start_prompt=start_prompt,
            end_prompt=end_prompt,
            negative_prompt=negative_prompt,
            duration_sec=duration_sec,
            driving_video_path=driving_video_path,
            audio_wav_path=audio_wav_path,
            lora_strength=lora_strength,
            seed=seed,
            require_keyframe_approval=require_keyframe_approval,
            use_fast_distill_lora=use_fast_distill_lora,
            last_frame_handoff_shot_id=last_frame_handoff_shot_id,
        )
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        state["requested_tab"] = "keyframes" if require_keyframe_approval else "timeline"
        state["last_action"] = res
        return _tool_result(
            f"render_shot status={res['status']} for '{project_name}/{shot_id}'.",
            state,
        )

    @mcp.tool(
        name="extract_and_repair_frame",
        title="Extract & Repair Frame K",
        description="Lock frames 0..K-1 losslessly and repair Frame K via Kontext or Fill + Florence-2/SAM2.",
        meta=INLINE_META,
    )
    async def extract_and_repair_frame(
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
    ) -> CallToolResult:
        res = await svc.extract_and_repair_frame(
            source_video_path=source_video_path,
            frame_index=frame_index,
            character_name=character_name,
            repair_prompt=repair_prompt,
            mask_target_text=mask_target_text,
            mask_image_path=mask_image_path,
            repair_engine=repair_engine,
            inpaint_denoise=inpaint_denoise,
            project_name=project_name,
            shot_id=shot_id,
        )
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        state["requested_tab"] = "timeline"
        state["last_action"] = res
        return _tool_result(
            f"Repaired frame {frame_index} (seam_score={res['checkpoint']['seam_continuity_score']}).",
            state,
        )

    @mcp.tool(
        name="continue_from_repaired_frame",
        title="Continue Branch From Frame K",
        description="Continue forward from repaired Frame K via carry, pivot (Wan 2.2 4k+1), or lipsync (LatentSync).",
        meta=INLINE_META,
    )
    async def continue_from_repaired_frame(
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
        project_name: str | None = None,
        shot_id: str | None = None,
    ) -> CallToolResult:
        res = await svc.continue_from_repaired_frame(
            checkpoint_dir=checkpoint_dir,
            character_name=character_name,
            mode=mode,
            continuation_prompt=continuation_prompt,
            audio_wav_path=audio_wav_path,
            end_keyframe_prompt=end_keyframe_prompt,
            driving_video_path=driving_video_path,
            target_total_frames=target_total_frames,
            lora_strength=lora_strength,
            seed=seed,
        )
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        state["requested_tab"] = "timeline"
        state["last_action"] = res
        return _tool_result(
            f"Created branch {res['branch']['branch_id']} (mode={mode}).",
            state,
        )

    @mcp.tool(
        name="export_or_compose_story",
        description="Concatenate shots and optionally apply RIFE VFI + MMAudio foley into final project MP4.",
    )
    async def export_or_compose_story(
        project_name: str,
        shot_ids: list[str] | None = None,
        apply_rife: bool = True,
        target_fps: int = 48,
        add_mmaudio_prompt: str | None = None,
    ) -> str:
        return json.dumps(
            await svc.export_or_compose_story(
                project_name=project_name,
                shot_ids=shot_ids,
                apply_rife=apply_rife,
                target_fps=target_fps,
                add_mmaudio_prompt=add_mmaudio_prompt,
            ),
            indent=2,
        )

    @mcp.tool(
        name="inspect_or_patch_node",
        description="Surgically patch a single node input in a ComfyUI workflow JSON.",
    )
    async def inspect_or_patch_node(
        workflow_name: str,
        node_id: str,
        input_key: str,
        new_value: Any,
        save_copy_name: str | None = None,
    ) -> str:
        return json.dumps(
            await svc.inspect_or_patch_node(
                workflow_name=workflow_name,
                node_id=node_id,
                input_key=input_key,
                new_value=new_value,
                save_copy_name=save_copy_name,
            ),
            indent=2,
        )

    @mcp.tool(
        name="pull_live_workflow",
        description="Pull the active or most recent ComfyUI API graph from localhost:8188 into workflows/.",
    )
    async def pull_live_workflow(output_name: str) -> str:
        return json.dumps(await svc.pull_live_workflow(output_name), indent=2)

    # --- App-Only Tools (Hidden from LLM trajectory; keep large media payloads out of model context) ---

    @mcp.tool(name="app_get_studio_state", meta=APP_ONLY_META)
    async def app_get_studio_state(
        project_name: str | None = None,
        shot_id: str | None = None,
    ) -> CallToolResult:
        state = await svc.get_studio_state(project_name=project_name, shot_id=shot_id)
        return _tool_result("studio state", state)

    @mcp.tool(name="app_get_media_data_url", meta=APP_ONLY_META)
    def app_get_media_data_url(file_path: str, max_width: int = 1280) -> CallToolResult:
        res = svc.get_media_data_url(file_path=file_path, max_width=max_width)
        return _tool_result("media payload", res)

    @mcp.tool(name="app_extract_video_frame", meta=APP_ONLY_META)
    def app_extract_video_frame(video_path: str, frame_index: int) -> CallToolResult:
        res = svc.extract_video_frame_data_url(video_path=video_path, frame_index=frame_index)
        return _tool_result("frame payload", res)

    @mcp.tool(name="app_save_painted_mask", meta=APP_ONLY_META)
    def app_save_painted_mask(
        project_name: str,
        shot_id: str,
        mask_data_url: str,
        label: str = "brush_mask",
    ) -> CallToolResult:
        res = svc.save_painted_mask(
            project_name=project_name,
            shot_id=shot_id,
            mask_data_url=mask_data_url,
            label=label,
        )
        return _tool_result("mask saved", res)

    return mcp, svc


def main() -> None:
    mcp, _ = create_mcp_server()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
