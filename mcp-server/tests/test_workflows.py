"""Validate all 5 ComfyUI API-format workflows and companion node maps."""

from __future__ import annotations

import json
from pathlib import Path

from comfy_factory_mcp.comfy_client import strip_virtual_ui_nodes

WORKFLOWS_DIR = Path(__file__).resolve().parents[2] / "workflows"

EXPECTED_PAIRS = [
    "wan22_keyframes",
    "wan22_video_from_keyframes",
    "wan22_frame_repair",
    "wan22_frame_continue",
    "wan22_post_export",
]


def test_all_workflows_and_node_maps_resolve_cleanly() -> None:
    for stem in EXPECTED_PAIRS:
        wf_path = WORKFLOWS_DIR / f"{stem}_pipeline.json"
        map_path = WORKFLOWS_DIR / f"{stem}_node_map.json"
        assert wf_path.exists(), f"Missing workflow: {wf_path}"
        assert map_path.exists(), f"Missing node map: {map_path}"

        wf = json.loads(wf_path.read_text(encoding="utf-8"))
        node_map = json.loads(map_path.read_text(encoding="utf-8"))
        assert isinstance(wf, dict) and len(wf) > 0

        # Verify every link [node_id, slot] points to an existing node
        for nid, node in wf.items():
            assert "class_type" in node
            assert "inputs" in node
            for inp_key, val in node["inputs"].items():
                if isinstance(val, list) and len(val) == 2 and isinstance(val[0], str):
                    assert val[0] in wf, f"{stem}: node {nid}.{inp_key} references missing node {val[0]}"
                    assert isinstance(val[1], int)

        # Verify every semantic node map pointer exists
        for param, spec in node_map.items():
            target_id = str(spec["node_id"])
            input_key = spec["input_key"]
            assert target_id in wf, f"{stem} map '{param}' references missing node {target_id}"
            assert input_key in wf[target_id]["inputs"], (
                f"{stem} map '{param}' references missing input '{input_key}' on node {target_id}"
            )


def test_wan22_dual_moe_and_florence_sam2_present() -> None:
    vid_wf = json.loads((WORKFLOWS_DIR / "wan22_video_from_keyframes_pipeline.json").read_text())
    unets = [
        node["inputs"].get("unet_name", "")
        for node in vid_wf.values()
        if node.get("class_type") == "UNETLoader"
    ]
    assert "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors" in unets
    assert "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors" in unets

    repair_wf = json.loads((WORKFLOWS_DIR / "wan22_frame_repair_pipeline.json").read_text())
    classes = {node["class_type"] for node in repair_wf.values()}
    assert {"DownloadAndLoadFlorence2Model", "Florence2Run", "Sam2Segmentation"}.issubset(classes)


def test_markdown_notes_present_in_files_and_stripped_for_api_execution() -> None:
    for stem in EXPECTED_PAIRS:
        wf = json.loads((WORKFLOWS_DIR / f"{stem}_pipeline.json").read_text(encoding="utf-8"))
        note_nodes = [n for n in wf.values() if n.get("class_type") == "MarkdownNote"]
        assert len(note_nodes) >= 1, f"{stem} should include at least one MarkdownNote guide card"
        for nid, node in wf.items():
            meta = node.get("_meta", {})
            assert meta.get("title"), f"{stem} node {nid} missing _meta.title"
            assert meta.get("description"), f"{stem} node {nid} missing _meta.description"

        stripped = strip_virtual_ui_nodes(wf)
        assert all(n.get("class_type") != "MarkdownNote" for n in stripped.values())
        assert len(stripped) == len(wf) - len(note_nodes)
