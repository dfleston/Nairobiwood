#!/usr/bin/env bash
set -euo pipefail

MANIFEST="/opt/wan22_models_manifest.json"
# If /workspace is mounted (RunPod volume or container disk), store models there so they persist
TARGET_DIR="/workspace/models"
if [[ ! -d "/workspace" ]]; then
  TARGET_DIR="/opt/ComfyUI/models"
fi

mkdir -p "$TARGET_DIR"/{diffusion_models,clip,vae,loras}

echo "=== [Wan 2.2 Bootstrap] Checking base weights in $TARGET_DIR ==="

export HF_HUB_ENABLE_HF_TRANSFER=1

python3 - "$MANIFEST" "$TARGET_DIR" <<'PY'
import json, sys, os, shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

manifest_path = Path(sys.argv[1])
target_dir = Path(sys.argv[2])
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

# Symlink from TARGET_DIR to /opt/ComfyUI/models
if [[ "$TARGET_DIR" != "/opt/ComfyUI/models" ]]; then
  echo "Linking models into ComfyUI..."
  for sub in diffusion_models clip vae loras; do
    mkdir -p "/opt/ComfyUI/models/$sub"
    for f in "$TARGET_DIR/$sub"/*; do
      if [[ -f "$f" ]]; then
        ln -sf "$f" "/opt/ComfyUI/models/$sub/$(basename "$f")"
      fi
    done
  done
fi

# Ensure unet symlinks to diffusion_models
mkdir -p /opt/ComfyUI/models/unet
for f in /opt/ComfyUI/models/diffusion_models/*; do
  if [[ -f "$f" ]]; then
    ln -sf "$f" "/opt/ComfyUI/models/unet/$(basename "$f")"
  fi
done

echo "=== [Wan 2.2 Bootstrap] All base models ready ==="
