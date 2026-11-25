"""Localhost-only Starlette backend serving the unified Director Studio in Safari/Chrome."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from comfy_factory_mcp.config import FactorySettings, resolve_safe_path
from comfy_factory_mcp.service import FactoryService

_STUDIO_THREAD: threading.Thread | None = None


def load_director_html(settings: FactorySettings) -> str:
    html_path = settings.resolved_director_html()
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return (
        "<!doctype html><html><body><h1>Director Studio Not Built Yet</h1>"
        "<p>Run <code>pnpm --dir mcp-app build</code> to generate dist/director.html.</p>"
        "</body></html>"
    )


def create_web_studio_app(
    service: FactoryService | None = None,
    settings: FactorySettings | None = None,
) -> Starlette:
    """Create a localhost-only Starlette app exposing the Director Studio UI and JSON/media API."""
    svc = service or FactoryService(settings=settings)
    cfg = svc.settings

    async def index(_: Request) -> HTMLResponse:
        return HTMLResponse(load_director_html(cfg))

    async def health(_: Request) -> JSONResponse:
        return JSONResponse({"ok": True, "port": cfg.studio_http_port})

    async def media_proxy(request: Request) -> Response:
        raw_path = request.query_params.get("path", "")
        if not raw_path:
            return JSONResponse({"error": "Missing 'path' query parameter"}, status_code=400)
        try:
            safe = resolve_safe_path(
                raw_path,
                cfg,
                allowed_roots=[cfg.resolved_data_dir()],
                must_exist=True,
            )
        except (ValueError, FileNotFoundError) as exc:
            status = 404 if isinstance(exc, FileNotFoundError) else 403
            return JSONResponse({"error": str(exc)}, status_code=status)

        # FileResponse in Starlette natively supports HTTP Range requests for video scrubbing
        return FileResponse(safe)

    async def call_tool_api(request: Request) -> JSONResponse:
        tool_name = request.path_params["tool_name"]
        try:
            body: dict[str, Any] = await request.json() if await request.body() else {}
        except Exception:  # noqa: BLE001
            body = {}

        async def _with_studio_state(
            action_coro: Any,
            project_name: str | None = None,
            shot_id: str | None = None,
            tab: str | None = None,
        ) -> dict[str, Any]:
            res = await action_coro
            resolved_shot = shot_id
            if not resolved_shot and isinstance(res, dict) and isinstance(res.get("shot_ids"), list) and res["shot_ids"]:
                resolved_shot = res["shot_ids"][0]
            state = await svc.get_studio_state(project_name=project_name, shot_id=resolved_shot)
            if tab:
                state["requested_tab"] = tab
            state["last_action"] = res
            return state

        dispatch: dict[str, Any] = {
            "factory_status": lambda p: svc.factory_status(),
            "factory_up": lambda p: svc.factory_up(**p),
            "factory_down": lambda p: svc.factory_down(),
            "list_local_assets": lambda p: svc.list_local_assets(),
            "start_lora_training": lambda p: svc.start_lora_training(**p),
            "get_job_status": lambda p: svc.get_job_status(**p),
            "import_or_init_story_bible": lambda p: _with_studio_state(
                svc.import_or_init_story_bible(**p), p.get("project_name"), None, "keyframes"
            ),
            "generate_character_portrait": lambda p: _with_studio_state(
                svc.generate_character_portrait(**p), p.get("project_name"), None, "keyframes"
            ),
            "generate_scene_keyframes": lambda p: _with_studio_state(
                svc.generate_scene_keyframes(**p), p.get("project_name"), None, "keyframes"
            ),
            "generate_shot_keyframes": lambda p: _with_studio_state(
                svc.generate_shot_keyframes(**p), p.get("project_name"), p.get("shot_id"), "keyframes"
            ),
            "regenerate_shot_keyframe": lambda p: _with_studio_state(
                svc.regenerate_shot_keyframe(**p), p.get("project_name"), p.get("shot_id"), "keyframes"
            ),
            "approve_and_render_video": lambda p: _with_studio_state(
                svc.approve_and_render_video(**p), p.get("project_name"), p.get("shot_id"), "timeline"
            ),
            "render_shot": lambda p: _with_studio_state(
                svc.render_shot(**p), p.get("project_name"), p.get("shot_id"), None
            ),
            "extract_and_repair_frame": lambda p: _with_studio_state(
                svc.extract_and_repair_frame(**p), p.get("project_name"), p.get("shot_id"), "timeline"
            ),
            "continue_from_repaired_frame": lambda p: _with_studio_state(
                svc.continue_from_repaired_frame(**p), p.get("project_name"), p.get("shot_id"), "timeline"
            ),
            "export_or_compose_story": lambda p: svc.export_or_compose_story(**p),
            "app_get_studio_state": lambda p: svc.get_studio_state(**p),
            "app_get_media_data_url": lambda p: _sync_wrap(svc.get_media_data_url(**p)),
            "app_extract_video_frame": lambda p: _sync_wrap(svc.extract_video_frame_data_url(**p)),
            "app_save_painted_mask": lambda p: _sync_wrap(svc.save_painted_mask(**p)),
        }
        handler = dispatch.get(tool_name)
        if handler is None:
            return JSONResponse({"error": f"Unknown tool: {tool_name}"}, status_code=404)
        try:
            result = await handler(body)
            return JSONResponse({"structuredContent": result})
        except Exception as exc:  # noqa: BLE001
            return JSONResponse({"error": str(exc)}, status_code=400)

    routes = [
        Route("/", index, methods=["GET"]),
        Route("/api/health", health, methods=["GET"]),
        Route("/api/media", media_proxy, methods=["GET"]),
        Route("/api/tool/{tool_name}", call_tool_api, methods=["POST"]),
    ]
    return Starlette(routes=routes)


async def _sync_wrap(val: Any) -> Any:
    return val


def ensure_background_studio_server(service: FactoryService) -> str:
    """Start the localhost browser Director Studio on 127.0.0.1:8190 in a daemon thread if idle."""
    global _STUDIO_THREAD
    port = service.settings.studio_http_port
    url = f"http://127.0.0.1:{port}"
    if _STUDIO_THREAD is not None and _STUDIO_THREAD.is_alive():
        return url

    app = create_web_studio_app(service=service)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, name="nairobiwood-studio", daemon=True)
    thread.start()
    _STUDIO_THREAD = thread
    time.sleep(0.15)
    return url


def main() -> None:
    settings = FactorySettings()
    uvicorn.run(
        create_web_studio_app(settings=settings),
        host="127.0.0.1",
        port=settings.studio_http_port,
    )


if __name__ == "__main__":
    main()
