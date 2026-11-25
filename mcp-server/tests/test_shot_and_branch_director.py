"""End-to-end director tests using real FFmpeg synthetic video/audio + mocked cloud execution."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from comfy_factory_mcp.comfy_client import ComfyUIClient
from comfy_factory_mcp.config import FactorySettings, ensure_data_dirs
from comfy_factory_mcp.qa_metrics import FFMPEG_BIN, probe_media_info
from comfy_factory_mcp.shot_director import (
    ShotDirector,
    compute_wan_continuation_frames,
    compute_wan_shot_frames,
)


def _make_synthetic_clip(out_mp4: Path, duration_sec: float = 3.375, fps: int = 24) -> None:
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration_sec}:size=320x180:rate={fps}",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(out_mp4),
        ],
        check=True,
    )


def _make_synthetic_wav(out_wav: Path, duration_sec: float = 2.0) -> None:
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            FFMPEG_BIN,
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={duration_sec}",
            str(out_wav),
        ],
        check=True,
    )


class FakeTunnel:
    """Mock SSH tunnel transferring synthetic test artifacts on upload/download."""

    def __init__(self, work_tmp: Path) -> None:
        self.work_tmp = work_tmp
        self.purged_paths: list[str] = []

    async def rsync_upload(self, local_path: Path, remote_dir: str) -> None:
        assert local_path.exists()

    async def rsync_download(self, remote_path: str, local_dest: Path) -> None:
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        if local_dest.suffix == ".png":
            color = (30, 120, 220) if "start" in local_dest.name else (220, 110, 40)
            Image.new("RGB", (320, 180), color).save(local_dest)
        elif local_dest.suffix == ".mkv":
            tmp_mp4 = self.work_tmp / f"gen_{local_dest.stem}.mp4"
            _make_synthetic_clip(tmp_mp4, duration_sec=1.5, fps=24)
            subprocess.run(
                [FFMPEG_BIN, "-y", "-v", "error", "-i", str(tmp_mp4), "-c:v", "ffv1", str(local_dest)],
                check=True,
            )
        elif local_dest.suffix == ".mp4":
            _make_synthetic_clip(local_dest, duration_sec=3.375, fps=24)

    async def purge_remote(self, remote_path: str) -> None:
        self.purged_paths.append(remote_path)


class FakeComfyClient(ComfyUIClient):
    def __init__(self, workflows_dir: Path) -> None:
        super().__init__(workflows_dir)
        self.freed_vram = 0
        self.queued: list[dict] = []

    async def free_vram(self, unload_models: bool = True, free_memory: bool = True) -> None:
        self.freed_vram += 1

    async def queue_and_wait(self, workflow: dict, *, timeout_sec: float = 900.0) -> dict:
        self.queued.append(workflow)
        return {"ok": True}


def test_wan_4k_plus_1_frame_math() -> None:
    assert compute_wan_continuation_frames(36) == (37, 36)
    assert compute_wan_continuation_frames(1) == (5, 1)
    assert compute_wan_continuation_frames(37) == (41, 37)
    assert compute_wan_continuation_frames(40) == (41, 40)
    assert (compute_wan_shot_frames(5.0, 24) - 1) % 4 == 0


async def test_keyframe_gate_and_pause_and_pivot_flow(tmp_path: Path) -> None:
    workflows_dir = Path(__file__).resolve().parents[2] / "workflows"
    settings = FactorySettings(factory_data_dir=tmp_path / "factory")
    ensure_data_dirs(settings)
    tunnel = FakeTunnel(tmp_path)
    comfy = FakeComfyClient(workflows_dir)
    director = ShotDirector(settings, comfy, tunnel=tunnel)  # type: ignore[arg-type]

    # 1. Start/End Keyframe Generation (Pre-Creation Gate)
    kf_manifest = await director.generate_shot_keyframes(
        project_name="pilot",
        shot_id="shot_01",
        character_name="elena",
        start_prompt="elena standing by window",
        end_prompt="elena looking toward doorway",
    )
    assert kf_manifest.keyframes.approved is False
    assert kf_manifest.keyframes.has_end_frame is True
    assert Path(kf_manifest.keyframes.start_frame_path or "").exists()
    assert Path(kf_manifest.keyframes.end_frame_path or "").exists()

    # 2. Re-roll Start Frame only
    rerolled = await director.regenerate_shot_keyframe(
        project_name="pilot",
        shot_id="shot_01",
        which="start",
        prompt="elena standing by sunlit window",
    )
    assert rerolled.keyframes.approved is False
    assert rerolled.keyframes.seed != kf_manifest.keyframes.seed

    # 3. Approve & Render Video
    rendered = await director.approve_and_render_video(
        project_name="pilot",
        shot_id="shot_01",
        duration_sec=3.375,
    )
    assert comfy.freed_vram >= 1
    assert rendered.keyframes.approved is True
    assert Path(rendered.master_video_path or "").exists()
    assert Path(rendered.contact_strip_path or "").exists()

    # Replace synthetic master with exact 81-frame clip to verify frame-exact slicing
    master = Path(rendered.master_video_path or "")
    tmp_src = tmp_path / "exact_81.mp4"
    _make_synthetic_clip(tmp_src, duration_sec=81 / 24.0, fps=24)
    subprocess.run([FFMPEG_BIN, "-y", "-v", "error", "-i", str(tmp_src), "-c:v", "ffv1", str(master)], check=True)
    assert probe_media_info(master)["frame_count"] == 81

    # 4. Extract & Repair Frame 45
    ckpt = await director.extract_and_repair_frame(
        source_video_path=str(master),
        frame_index=45,
        character_name="elena",
        repair_prompt="fix right hand gripping mug",
        mask_target_text="right hand",
        project_name="pilot",
        shot_id="shot_01",
    )
    assert Path(ckpt.locked_prefix_path).exists()
    assert probe_media_info(Path(ckpt.locked_prefix_path))["frame_count"] == 45
    assert ckpt.seam_continuity_score is not None and 0.0 <= ckpt.seam_continuity_score <= 1.0

    # 5. Continue forward (pivot + lipsync)
    pivot_branch = await director.continue_from_repaired_frame(
        checkpoint_dir=ckpt.checkpoint_dir,
        character_name="elena",
        mode="pivot",
        continuation_prompt="looks suspiciously toward the door",
    )
    assert pivot_branch.wan_frames_requested == 37
    assert pivot_branch.tail_frames_kept == 36
    assert Path(pivot_branch.branched_preview_path).exists()

    wav_file = settings.voices_dir / "line_01.wav"
    _make_synthetic_wav(wav_file, duration_sec=2.0)
    lip_branch = await director.continue_from_repaired_frame(
        checkpoint_dir=ckpt.checkpoint_dir,
        character_name="elena",
        mode="lipsync",
        audio_wav_path=str(wav_file),
    )
    assert lip_branch.mode == "lipsync"
    assert Path(lip_branch.branched_preview_path).exists()
    assert len(tunnel.purged_paths) > 0
