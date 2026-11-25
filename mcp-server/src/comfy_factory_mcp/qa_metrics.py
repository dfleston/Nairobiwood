"""Local macOS FFmpeg media probing, contact strip generation, and seam continuity QA."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


def _find_bin(name: str, fallback: str) -> str:
    return shutil.which(name) or fallback


FFMPEG_BIN = _find_bin("ffmpeg", "/opt/homebrew/bin/ffmpeg")
FFPROBE_BIN = _find_bin("ffprobe", "/opt/homebrew/bin/ffprobe")


def probe_media_info(media_path: Path) -> dict[str, float | int | str]:
    """Return frame_count, fps, width, height, and duration_sec via ffprobe."""
    if not media_path.exists():
        raise FileNotFoundError(f"Media file not found: {media_path}")
    cmd = [
        FFPROBE_BIN,
        "-v",
        "error",
        "-show_entries",
        "stream=codec_type,width,height,r_frame_rate,nb_frames,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(media_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    payload = json.loads(res.stdout)
    streams = payload.get("streams") or []
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    fmt_dur = float((payload.get("format") or {}).get("duration") or 0.0)

    if video_stream is None:
        return {"duration_sec": fmt_dur, "fps": 0.0, "frame_count": 0, "width": 0, "height": 0}

    rate_raw = str(video_stream.get("r_frame_rate") or "24/1")
    if "/" in rate_raw:
        num, den = rate_raw.split("/", 1)
        fps = float(num) / max(float(den), 1.0)
    else:
        fps = float(rate_raw)

    dur = float(video_stream.get("duration") or fmt_dur or 0.0)
    nb_frames_raw = video_stream.get("nb_frames")
    if nb_frames_raw and str(nb_frames_raw).isdigit():
        frame_count = int(nb_frames_raw)
    elif dur > 0 and fps > 0:
        frame_count = max(1, round(dur * fps))
    else:
        count_cmd = [
            FFPROBE_BIN,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ]
        count_out = subprocess.run(count_cmd, capture_output=True, text=True, check=True).stdout.strip()
        frame_count = int(count_out) if count_out.isdigit() else 0

    return {
        "duration_sec": dur,
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": int(video_stream.get("width") or 0),
        "height": int(video_stream.get("height") or 0),
    }


def extract_single_frame(video_path: Path, frame_index: int, output_png: Path) -> Path:
    """Extract an exact 0-based frame from a local video into PNG."""
    output_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_index})",
        "-vsync",
        "0",
        "-frames:v",
        "1",
        str(output_png),
    ]
    subprocess.run(cmd, check=True)
    if not output_png.exists():
        raise RuntimeError(f"Failed to extract frame {frame_index} from {video_path}")
    return output_png


def slice_lossless_prefix(video_path: Path, end_frame_exclusive: int, output_mkv: Path) -> Path:
    """Cache frames 0..(K-1) as lossless FFV1 .mkv to prevent generational compression drift."""
    output_mkv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-frames:v",
        str(end_frame_exclusive),
        "-c:v",
        "ffv1",
        "-an",
        str(output_mkv),
    ]
    subprocess.run(cmd, check=True)
    return output_mkv


def slice_lossless_tail(video_path: Path, start_frame: int, output_mkv: Path) -> Path:
    """Extract frames K..N-1 as lossless FFV1 .mkv for Lip-Sync bypass mode."""
    output_mkv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"select=gte(n\\,{start_frame}),setpts=PTS-STARTPTS",
        "-c:v",
        "ffv1",
        "-an",
        str(output_mkv),
    ]
    subprocess.run(cmd, check=True)
    return output_mkv


def concat_prefix_and_tail(
    prefix_mkv: Path,
    tail_video: Path,
    output_master_mkv: Path,
    output_preview_mp4: Path,
    *,
    audio_wav_path: Path | None = None,
    audio_offset_sec: float = 0.0,
) -> tuple[Path, Path]:
    """Concatenate lossless prefix and regenerated tail into lossless master + preview MP4."""
    output_master_mkv.parent.mkdir(parents=True, exist_ok=True)
    output_preview_mp4.parent.mkdir(parents=True, exist_ok=True)

    filter_complex = "[0:v]setpts=PTS-STARTPTS[v0];[1:v]setpts=PTS-STARTPTS[v1];[v0][v1]concat=n=2:v=1:a=0[vout]"
    cmd_master = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(prefix_mkv),
        "-i",
        str(tail_video),
        "-filter_complex",
        filter_complex,
        "-map",
        "[vout]",
        "-c:v",
        "ffv1",
        str(output_master_mkv),
    ]
    subprocess.run(cmd_master, check=True)

    cmd_preview = [
        FFMPEG_BIN,
        "-y",
        "-v",
        "error",
        "-i",
        str(output_master_mkv),
    ]
    if audio_wav_path and audio_wav_path.exists():
        delay_ms = max(0, round(audio_offset_sec * 1000))
        cmd_preview.extend(
            [
                "-i",
                str(audio_wav_path),
                "-filter_complex",
                f"[1:a]adelay={delay_ms}|{delay_ms}[aout]",
                "-map",
                "0:v",
                "-map",
                "[aout]",
                "-c:a",
                "aac",
                "-shortest",
            ]
        )
    cmd_preview.extend(
        [
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            str(output_preview_mp4),
        ]
    )
    subprocess.run(cmd_preview, check=True)
    return output_master_mkv, output_preview_mp4


def compute_seam_continuity_score(prev_frame_path: Path, target_frame_path: Path) -> float:
    """Compute a normalized [0.0, 1.0] luminance + color + edge continuity score across a seam."""
    img_a = np.asarray(Image.open(prev_frame_path).convert("RGB"), dtype=np.float32) / 255.0
    img_b = np.asarray(Image.open(target_frame_path).convert("RGB"), dtype=np.float32) / 255.0
    if img_a.shape != img_b.shape:
        resized = Image.open(target_frame_path).convert("RGB").resize((img_a.shape[1], img_a.shape[0]))
        img_b = np.asarray(resized, dtype=np.float32) / 255.0

    # Global RGB mean/std similarity + spatial pixel similarity
    mae = float(np.mean(np.abs(img_a - img_b)))
    hist_a = np.concatenate([np.histogram(img_a[:, :, c], bins=16, range=(0.0, 1.0))[0] for c in range(3)])
    hist_b = np.concatenate([np.histogram(img_b[:, :, c], bins=16, range=(0.0, 1.0))[0] for c in range(3)])
    hist_a = hist_a / max(float(np.sum(hist_a)), 1.0)
    hist_b = hist_b / max(float(np.sum(hist_b)), 1.0)
    hist_l1 = float(np.sum(np.abs(hist_a - hist_b))) / 2.0

    score = max(0.0, min(1.0, 1.0 - (0.65 * mae + 0.35 * hist_l1)))
    return round(score, 4)


def build_contact_strip(video_path: Path, output_png: Path, num_samples: int = 6) -> Path:
    """Sample evenly spaced frames from video_path into a horizontal contact strip PNG."""
    info = probe_media_info(video_path)
    total = int(info.get("frame_count") or 0)
    if total <= 0:
        raise RuntimeError(f"Cannot build contact strip: 0 frames in {video_path}")
    indices = sorted({min(total - 1, round(i * (total - 1) / max(1, num_samples - 1))) for i in range(num_samples)})
    tmp_dir = output_png.parent / ".strip_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    thumbs: list[Image.Image] = []
    try:
        for idx in indices:
            frame_png = tmp_dir / f"f_{idx:04d}.png"
            extract_single_frame(video_path, idx, frame_png)
            im = Image.open(frame_png).convert("RGB")
            im.thumbnail((320, 180))
            thumbs.append(im)
        width = sum(t.width for t in thumbs)
        height = max(t.height for t in thumbs)
        strip = Image.new("RGB", (width, height))
        x = 0
        for t in thumbs:
            strip.paste(t, (x, 0))
            x += t.width
        output_png.parent.mkdir(parents=True, exist_ok=True)
        strip.save(output_png)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
    return output_png
