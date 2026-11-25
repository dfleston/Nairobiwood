"""Tests for MCPServer tool/resource metadata, APP_ONLY visibility, and Web Studio ASGI routes."""

from __future__ import annotations

from pathlib import Path

from starlette.testclient import TestClient

from comfy_factory_mcp.config import FactorySettings
from comfy_factory_mcp.server import APP_URI, create_mcp_server
from comfy_factory_mcp.web_studio import create_web_studio_app


async def test_mcp_server_registers_ui_resource_and_app_only_tools(tmp_path: Path) -> None:
    html_file = tmp_path / "director.html"
    html_file.write_text("<!doctype html><html><body>OK</body></html>", encoding="utf-8")

    settings = FactorySettings(factory_data_dir=tmp_path / "data", director_app_html=html_file)
    mcp, _ = create_mcp_server(settings=settings)

    resources = await mcp.list_resources()
    ui_res = next(r for r in resources if str(r.uri) == APP_URI)
    assert ui_res.mime_type == "text/html;profile=mcp-app"
    assert ui_res.meta == {"ui": {"prefersBorder": False}}

    contents = await mcp.read_resource(APP_URI)
    assert "OK" in contents[0].content

    tools = {t.name: t for t in await mcp.list_tools()}
    assert "open_director_studio" in tools
    assert tools["open_director_studio"].meta["com.google.antigravity/ui"]["entrypoints"][0]["type"] == "aux_pane"
    assert tools["generate_shot_keyframes"].meta == {"ui": {"resourceUri": APP_URI}}
    assert tools["app_get_media_data_url"].meta == {"ui": {"visibility": ["app"]}}
    assert tools["app_save_painted_mask"].meta == {"ui": {"visibility": ["app"]}}


def test_web_studio_serves_html_tools_and_blocks_media_traversal(tmp_path: Path) -> None:
    html_file = tmp_path / "director.html"
    html_file.write_text("<!doctype html><html><body>Studio</body></html>", encoding="utf-8")
    data_dir = tmp_path / "data"
    (data_dir / "loras" / "D_Leston").mkdir(parents=True, exist_ok=True)
    (data_dir / "loras" / "D_Leston" / "D_Leston_flux.safetensors").write_bytes(b"flux")
    (data_dir / "datasets" / "D_Leston").mkdir(parents=True, exist_ok=True)
    (data_dir / "datasets" / "D_Leston" / "ref_01.png").write_bytes(b"png")
    (data_dir / "datasets" / "Elena").mkdir(parents=True, exist_ok=True)
    (data_dir / "datasets" / "Elena" / "ref_01.png").write_bytes(b"png")

    settings = FactorySettings(factory_data_dir=data_dir, director_app_html=html_file)
    app = create_web_studio_app(settings=settings)

    client = TestClient(app)
    root_resp = client.get("/")
    assert root_resp.status_code == 200
    assert "Studio" in root_resp.text

    status_resp = client.post("/api/tool/factory_status", json={})
    assert status_resp.status_code == 200
    assert status_resp.json()["structuredContent"]["active"] is False

    state_resp = client.post("/api/tool/app_get_studio_state", json={})
    assert state_resp.status_code == 200
    chars = state_resp.json()["structuredContent"]["assets"]["characters"]
    assert [c["name"] for c in chars] == ["D_Leston", "Elena"]
    assert chars[0]["has_flux_lora"] is True
    assert chars[0]["has_wan22_lora"] is False
    assert chars[0]["dataset_images"] == 1
    assert chars[1]["has_flux_lora"] is False
    assert chars[1]["dataset_images"] == 1

    bad_media = client.get("/api/media", params={"path": "../../etc/passwd"})
    assert bad_media.status_code == 403

    sample_file = data_dir / "outputs" / "sample.txt"
    sample_file.parent.mkdir(parents=True, exist_ok=True)
    sample_file.write_text("hello-video", encoding="utf-8")
    ok_media = client.get("/api/media", params={"path": str(sample_file)})
    assert ok_media.status_code == 200
    assert ok_media.text == "hello-video"
