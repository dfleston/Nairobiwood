#!/usr/bin/env bash
set -euo pipefail

MANIFEST="/opt/wan22_models_manifest.json"
COMFY_ROOT="/opt/ComfyUI"

# If ComfyUI is in /workspace/ComfyUI, adapt COMFY_ROOT; otherwise ensure /workspace/ComfyUI symlink points to /opt/ComfyUI
if [[ -d "/workspace/ComfyUI" && ! -L "/workspace/ComfyUI" ]]; then
  COMFY_ROOT="/workspace/ComfyUI"
elif [[ -d "/workspace" && ! -e "/workspace/ComfyUI" ]]; then
  ln -sfn /opt/ComfyUI /workspace/ComfyUI
fi

# Ensure base storage directory for persistent weights
TARGET_DIR="/workspace/models"
if [[ ! -d "/workspace" ]]; then
  TARGET_DIR="$COMFY_ROOT/models"
fi

mkdir -p "$TARGET_DIR"/{diffusion_models,clip,vae,loras}

# -------------------------------------------------------------------------
# 1. Scratch Automation (/dev/shm vs /workspace/scratch)
# -------------------------------------------------------------------------
echo "=== [Wan 2.2 Bootstrap] Configuring scratch directories ==="
# Check available space in /dev/shm (KB)
SHM_AVAIL_KB=$(df -k /dev/shm 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)
# Require at least 4 GB (4194304 KB) for high-speed RAM-disk scratch
if [[ "$SHM_AVAIL_KB" -gt 4194304 ]]; then
  SCRATCH_BASE="/dev/shm/factory_scratch"
  echo "Using /dev/shm for ultra-fast scratch ($(( SHM_AVAIL_KB / 1024 )) MB available)"
else
  SCRATCH_BASE="/workspace/factory_scratch"
  echo "Using /workspace for scratch (/dev/shm too small: $(( SHM_AVAIL_KB / 1024 )) MB)"
fi

mkdir -p "$SCRATCH_BASE/active_loras" "$SCRATCH_BASE/temp_renders"

# Ensure ComfyUI directories exist
mkdir -p "$COMFY_ROOT/models/loras"
mkdir -p "$COMFY_ROOT/output"
# NOTE: input/scratch must be a real physical directory (NOT a symlink) for ComfyUI 0.46+ path validation
mkdir -p "$COMFY_ROOT/input/scratch"

# Symlink scratch dirs
ln -sfn "$SCRATCH_BASE/active_loras" "$COMFY_ROOT/models/loras/scratch"
ln -sfn "$SCRATCH_BASE/temp_renders" "$COMFY_ROOT/output/scratch"

echo "Scratch setup complete: loras/scratch -> $SCRATCH_BASE/active_loras, output/scratch -> $SCRATCH_BASE/temp_renders"

# -------------------------------------------------------------------------
# 2. Base Model & Fast-Distill LoRA Bootstrapper
# -------------------------------------------------------------------------
echo "=== [Wan 2.2 Bootstrap] Checking base weights & LoRAs in $TARGET_DIR ==="

export HF_HUB_ENABLE_HF_TRANSFER=1

python3 - "$MANIFEST" "$TARGET_DIR" <<'PY'
import json, sys, os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

manifest_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
if not manifest_path.exists():
    print(f"[WARN] Manifest {manifest_path} not found. Skipping weight pull.")
    sys.exit(0)

entries = json.loads(manifest_path.read_text(encoding="utf-8"))

for entry in entries:
    repo_id = entry["repo_id"]
    filename = entry["filename"]
    subdir = entry["target_subdir"]
    target_name = entry["target_filename"]
    
    dest_path = target_dir / subdir / target_name
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    if dest_path.exists() and dest_path.stat().st_size > 1024*1024:
        print(f"[CACHED] {target_name} is already present ({dest_path.stat().st_size / 1e9:.2f} GB)")
    else:
        print(f"[DOWNLOADING] {target_name} from {repo_id}...")
        stage = dest_path.parent / f".tmp_{target_name}"
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(stage),
            local_dir_use_symlinks=False
        )
        # Move final file
        shutil.move(downloaded, dest_path)
        shutil.rmtree(stage, ignore_errors=True)
        print(f"[READY] {target_name} downloaded successfully.")

PY

# Symlink from TARGET_DIR to COMFY_ROOT/models
if [[ "$TARGET_DIR" != "$COMFY_ROOT/models" ]]; then
  echo "Linking models into ComfyUI..."
  for sub in diffusion_models clip vae loras; do
    mkdir -p "$COMFY_ROOT/models/$sub"
    for f in "$TARGET_DIR/$sub"/*; do
      if [[ -f "$f" ]]; then
        ln -sf "$f" "$COMFY_ROOT/models/$sub/$(basename "$f")"
      fi
    done
  done
fi

# Ensure unet symlinks to diffusion_models
mkdir -p "$COMFY_ROOT/models/unet"
for f in "$COMFY_ROOT/models/diffusion_models"/*; do
  if [[ -f "$f" ]]; then
    ln -sf "$f" "$COMFY_ROOT/models/unet/$(basename "$f")"
  fi
done

echo "=== [Wan 2.2 Bootstrap] All base models and fast-distill LoRAs ready ==="
