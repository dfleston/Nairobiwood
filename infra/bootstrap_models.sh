#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST_PATH="${MODELS_MANIFEST:-${SCRIPT_DIR}/models_manifest.json}"
MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
SKIP_GATED=0

for arg in "$@"; do
  case "$arg" in
    --skip-gated)
      SKIP_GATED=1
      ;;
    --manifest=*)
      MANIFEST_PATH="${arg#*=}"
      ;;
    --models-root=*)
      MODELS_ROOT="${arg#*=}"
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

if [[ ! -f "$MANIFEST_PATH" ]]; then
  echo "Manifest not found: $MANIFEST_PATH" >&2
  exit 1
fi

mkdir -p "$MODELS_ROOT"
export HF_HUB_ENABLE_HF_TRANSFER=1

python3 - "$MANIFEST_PATH" "$MODELS_ROOT" "$SKIP_GATED" <<'PY'
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

manifest_path = Path(sys.argv[1])
models_root = Path(sys.argv[2])
skip_gated = sys.argv[3] == "1"
hf_token = os.environ.get("HF_TOKEN", "").strip() or None

entries = json.loads(manifest_path.read_text(encoding="utf-8"))
gated_needed = [e for e in entries if e.get("gated")]
if gated_needed and not skip_gated and not hf_token:
    print(
        "ERROR: HF_TOKEN is required for gated FLUX models in models_manifest.json "
        "(or pass --skip-gated).",
        file=sys.stderr,
    )
    sys.exit(1)

for entry in entries:
    entry_id = entry["id"]
    repo_id = entry["repo_id"]
    filename = entry["filename"]
    target_subdir = entry["target_subdir"]
    target_filename = entry.get("target_filename", "")
    is_gated = bool(entry.get("gated", False))

    if is_gated and skip_gated:
        print(f"[skip-gated] {entry_id}", flush=True)
        continue

    dest_dir = models_root / target_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    tok = hf_token if is_gated else None

    if filename == "*":
        marker = dest_dir / ".download_complete"
        if marker.exists():
            print(f"[cached] {entry_id} -> {dest_dir}", flush=True)
            continue
        print(f"[download-repo] {entry_id}: {repo_id} -> {dest_dir}", flush=True)
        snapshot_download(repo_id=repo_id, local_dir=str(dest_dir), token=tok)
        marker.touch()
    else:
        final_name = target_filename or Path(filename).name
        final_path = dest_dir / final_name
        if final_path.exists() and final_path.stat().st_size > 0:
            print(f"[cached] {entry_id} -> {final_path}", flush=True)
            continue

        print(f"[download-file] {entry_id}: {repo_id}/{filename} -> {final_path}", flush=True)
        tmp_stage = dest_dir / ".hf_stage"
        tmp_stage.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(tmp_stage),
            token=tok,
        )
        shutil.move(str(downloaded), str(final_path))
        shutil.rmtree(tmp_stage, ignore_errors=True)

print("Model bootstrap complete.", flush=True)
PY
