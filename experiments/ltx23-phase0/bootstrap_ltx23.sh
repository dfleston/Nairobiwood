#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/venv/bin:/root/.local/bin:${PATH}"
PYTHON_BIN="/opt/venv/bin/python3"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

MANIFEST="/opt/ltx23_models_manifest.json"
COMFY_ROOT="/opt/ComfyUI"

if [[ -d "/workspace/ComfyUI" && ! -L "/workspace/ComfyUI" ]]; then
  COMFY_ROOT="/workspace/ComfyUI"
elif [[ -d "/workspace" && ! -e "/workspace/ComfyUI" ]]; then
  ln -sfn /opt/ComfyUI /workspace/ComfyUI
fi

# Choose model destination with enough free disk (>= 80 GB on /workspace if mounted, else /opt/ComfyUI/models)
TARGET_DIR="$COMFY_ROOT/models"
if [[ -d "/workspace" ]]; then
  WS_AVAIL_KB=$(df -Pk /workspace 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)
  if [[ "${WS_AVAIL_KB:-0}" -ge 83886080 ]]; then
    TARGET_DIR="/workspace/models"
  fi
fi

mkdir -p "$TARGET_DIR"/{diffusion_models,unet,text_encoders,clip,vae,loras,latent_upscale_models}

echo "=== [LTX-2.3 Bootstrap] Configuring scratch directories ==="
SHM_AVAIL_KB=$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)
if [[ "${SHM_AVAIL_KB:-0}" -gt 4194304 ]]; then
  SCRATCH_BASE="/dev/shm/factory_scratch"
  echo "Using /dev/shm scratch ($(( SHM_AVAIL_KB / 1024 )) MB free)"
else
  SCRATCH_BASE="/tmp/factory_scratch"
  echo "Using /tmp/factory_scratch (/dev/shm=${SHM_AVAIL_KB} KB)"
fi

mkdir -p "$SCRATCH_BASE/active_loras" "$SCRATCH_BASE/temp_renders"
chmod -R 700 "$SCRATCH_BASE"

# Physical input/scratch and output/scratch required by ComfyUI 0.46+ realpath checks
mkdir -p "$COMFY_ROOT/models/loras" "$COMFY_ROOT/input/scratch" "$COMFY_ROOT/output/scratch"
ln -sfn "$SCRATCH_BASE/active_loras" "$COMFY_ROOT/models/loras/scratch"

echo "=== [LTX-2.3 Bootstrap] Pulling manifest weights into $TARGET_DIR ==="
export HF_HUB_ENABLE_HF_TRANSFER=1

"$PYTHON_BIN" - "$MANIFEST" "$TARGET_DIR" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

manifest_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
if not manifest_path.exists():
    print(f"[WARN] Manifest {manifest_path} missing; skipping.")
    sys.exit(0)

entries = json.loads(manifest_path.read_text(encoding="utf-8"))
for idx, entry in enumerate(entries, start=1):
    repo_id = entry["repo_id"]
    filename = entry["filename"]
    subdir = entry["target_subdir"]
    target_name = entry["target_filename"]

    dest_path = target_dir / subdir / target_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists() and dest_path.stat().st_size > 1024 * 1024:
        print(f"[{idx}/{len(entries)}] [CACHED] {subdir}/{target_name} ({dest_path.stat().st_size / 1e9:.2f} GB)", flush=True)
        continue

    print(f"[{idx}/{len(entries)}] [DOWNLOADING] {subdir}/{target_name} from {repo_id}...", flush=True)
    stage = dest_path.parent / f".tmp_{target_name}"
    downloaded = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(stage),
    )
    shutil.move(downloaded, dest_path)
    shutil.rmtree(stage, ignore_errors=True)
    print(f"[{idx}/{len(entries)}] [READY] {subdir}/{target_name} ({dest_path.stat().st_size / 1e9:.2f} GB)", flush=True)
PY

# Link /workspace/models -> ComfyUI/models if needed
if [[ "$TARGET_DIR" != "$COMFY_ROOT/models" ]]; then
  for sub in diffusion_models unet text_encoders clip vae loras latent_upscale_models; do
    mkdir -p "$COMFY_ROOT/models/$sub"
    for f in "$TARGET_DIR/$sub"/*; do
      [[ -e "$f" ]] || continue
      ln -sfn "$f" "$COMFY_ROOT/models/$sub/$(basename "$f")"
    done
  done
fi

# Cross-link diffusion_models <-> unet and text_encoders <-> clip so any loader finds weights
for f in "$COMFY_ROOT/models/diffusion_models"/*; do
  [[ -e "$f" ]] || continue
  ln -sfn "$f" "$COMFY_ROOT/models/unet/$(basename "$f")"
done
for f in "$COMFY_ROOT/models/text_encoders"/*; do
  [[ -e "$f" ]] || continue
  ln -sfn "$f" "$COMFY_ROOT/models/clip/$(basename "$f")"
done

echo "=== [LTX-2.3 Bootstrap] Complete ==="
