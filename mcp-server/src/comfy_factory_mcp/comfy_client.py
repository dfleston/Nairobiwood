"""ComfyUI HTTP/WebSocket client, deterministic node patcher, and VRAM manager."""

from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path
from typing import Any

import httpx
import websockets


class ComfyUIError(RuntimeError):
    """Raised when ComfyUI rejects a workflow or fails during execution."""


VIRTUAL_UI_NODE_TYPES = frozenset({"Note", "MarkdownNote"})


def strip_virtual_ui_nodes(workflow: dict[str, Any]) -> dict[str, Any]:
    """Strip frontend-only virtual annotation nodes (Note / MarkdownNote) before /prompt submission."""
    return {
        node_id: node
        for node_id, node in workflow.items()
        if isinstance(node, dict) and node.get("class_type") not in VIRTUAL_UI_NODE_TYPES
    }


class ComfyUIClient:
    """Async client communicating exclusively with localhost-forwarded ComfyUI."""

    def __init__(self, workflows_dir: Path, base_url: str = "http://127.0.0.1:8188") -> None:
        self.workflows_dir = workflows_dir
        self.base_url = base_url.rstrip("/")

    async def upload_image(self, image_path: Path, subfolder: str = "", overwrite: bool = True) -> str:
        """Upload an image directly to ComfyUI's /upload/image endpoint (bypasses symlink restrictions)."""
        data = {"overwrite": "true" if overwrite else "false"}
        if subfolder:
            data["subfolder"] = subfolder
        async with httpx.AsyncClient(timeout=30.0) as client:
            with image_path.open("rb") as f:
                resp = await client.post(
                    f"{self.base_url}/upload/image",
                    files={"image": (image_path.name, f, "image/png")},
                    data=data,
                )
            resp.raise_for_status()
            body = resp.json()
            name = body.get("name", image_path.name)
            sub = body.get("subfolder", "")
            return f"{sub}/{name}" if sub else name

    def load_workflow_and_map(self, workflow_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load an API-format workflow JSON and its companion *_node_map.json."""
        stem = workflow_name.removesuffix(".json").removesuffix("_pipeline")
        wf_path = self.workflows_dir / f"{stem}_pipeline.json"
        map_path = self.workflows_dir / f"{stem}_node_map.json"
        if not wf_path.exists():
            raise FileNotFoundError(f"Workflow not found: {wf_path}")
        if not map_path.exists():
            raise FileNotFoundError(f"Node map not found: {map_path}")
        workflow = json.loads(wf_path.read_text(encoding="utf-8"))
        node_map = json.loads(map_path.read_text(encoding="utf-8"))
        return workflow, node_map

    @staticmethod
    def patch_workflow(
        workflow: dict[str, Any],
        node_map: dict[str, Any],
        semantic_params: dict[str, Any],
        raw_overrides: dict[tuple[str, str], Any] | None = None,
    ) -> dict[str, Any]:
        """Deterministically patch semantic parameters and optional raw node inputs."""
        patched = copy.deepcopy(workflow)
        for param_name, value in semantic_params.items():
            if value is None:
                continue
            if param_name not in node_map:
                raise KeyError(f"Unknown semantic parameter '{param_name}' for workflow.")
            spec = node_map[param_name]
            node_id = str(spec["node_id"])
            input_key = str(spec["input_key"])
            if node_id not in patched:
                raise KeyError(f"Node '{node_id}' from node_map missing in workflow.")
            patched[node_id]["inputs"][input_key] = value

        if raw_overrides:
            for (node_id, input_key), value in raw_overrides.items():
                if node_id not in patched:
                    raise KeyError(f"Node '{node_id}' missing in workflow.")
                patched[node_id]["inputs"][input_key] = value
        return patched

    async def system_stats(self) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{self.base_url}/system_stats")
            resp.raise_for_status()
            return resp.json()

    async def free_vram(self, unload_models: bool = True, free_memory: bool = True) -> None:
        """Explicitly unload FLUX/SAM2/Wan models between heavy stages to avoid 48GB OOMs."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self.base_url}/free",
                json={"unload_models": unload_models, "free_memory": free_memory},
            )
            resp.raise_for_status()

    async def queue_and_wait(
        self,
        workflow: dict[str, Any],
        *,
        timeout_sec: float = 900.0,
    ) -> dict[str, Any]:
        """Submit API prompt, monitor WebSocket for completion/OOM errors, and return history."""
        client_id = uuid.uuid4().hex
        executable_workflow = strip_virtual_ui_nodes(workflow)
        ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        ws_endpoint = f"{ws_url}/ws?clientId={client_id}"

        async with websockets.connect(ws_endpoint, open_timeout=15.0) as ws:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/prompt",
                    json={"prompt": executable_workflow, "client_id": client_id},
                )
                if resp.status_code >= 400:
                    raise ComfyUIError(f"ComfyUI rejected prompt ({resp.status_code}): {resp.text}")
                prompt_id = resp.json()["prompt_id"]

            while True:
                raw_msg = await ws.recv()
                if isinstance(raw_msg, bytes):
                    continue
                msg = json.loads(raw_msg)
                msg_type = msg.get("type")
                data = msg.get("data") or {}
                if data.get("prompt_id") not in (None, prompt_id):
                    continue

                if msg_type == "execution_error":
                    exc_msg = data.get("exception_message", "Unknown execution error")
                    node_type = data.get("node_type", "unknown")
                    raise ComfyUIError(f"ComfyUI node '{node_type}' failed: {exc_msg}")
                if msg_type == "executing" and data.get("node") is None:
                    break

        async with httpx.AsyncClient(timeout=30.0) as client:
            hist_resp = await client.get(f"{self.base_url}/history/{prompt_id}")
            hist_resp.raise_for_status()
            history = hist_resp.json()
        return history.get(prompt_id, history)

    async def pull_live_workflow(self, output_path: Path) -> Path:
        """Fetch the active/most-recent workflow JSON from ComfyUI queue/history."""
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(f"{self.base_url}/queue")
            resp.raise_for_status()
            queue_data = resp.json()
            running = queue_data.get("queue_running") or []
            pending = queue_data.get("queue_pending") or []
            prompt_graph = None
            if running:
                prompt_graph = running[0][2]
            elif pending:
                prompt_graph = pending[0][2]
            else:
                hist_resp = await client.get(f"{self.base_url}/history?max_items=1")
                hist_resp.raise_for_status()
                hist = hist_resp.json()
                if hist:
                    latest = next(iter(hist.values()))
                    prompt_graph = latest.get("prompt", [None, None, None])[2]

        if not prompt_graph:
            raise ComfyUIError("No running, pending, or recent workflow found on ComfyUI.")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(prompt_graph, indent=2), encoding="utf-8")
        return output_path
