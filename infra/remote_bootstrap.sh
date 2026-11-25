#!/usr/bin/env bash
set -euo pipefail

BOOTSTRAP_MODELS=0
for arg in "$@"; do
  case "$arg" in
    --bootstrap-models)
      BOOTSTRAP_MODELS=1
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      exit 2
      ;;
  esac
done

PERSIST_ROOT="${PERSIST_ROOT:-/workspace/nairobiwood}"
MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
MIN_SHM_MB=8192

mkdir -p "$PERSIST_ROOT" "$MODELS_ROOT"

# 1. Ensure system CLI tools are installed
MISSING_PKGS=()
for cmd_pkg in "tmux:tmux" "aria2c:aria2" "ffmpeg:ffmpeg" "jq:jq" "rsync:rsync" "gocryptfs:gocryptfs"; do
  cmd="${cmd_pkg%%:*}"
  pkg="${cmd_pkg#*:}"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    MISSING_PKGS+=("$pkg")
  fi
done

if [[ ${#MISSING_PKGS[@]} -gt 0 ]]; then
  echo "[remote_bootstrap] Installing system packages: ${MISSING_PKGS[*]}"
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq
  apt-get install -y --no-install-recommends \
    "${MISSING_PKGS[@]}" coreutils libgl1 libglib2.0-0
fi

# 2. Persistent ComfyUI + custom nodes + LoRA trainers installation under /workspace/nairobiwood
if [[ ! -f "$PERSIST_ROOT/.installed_v1" ]]; then
  echo "[remote_bootstrap] Installing Python dependencies, ComfyUI, custom nodes, and trainers into $PERSIST_ROOT..."
  python3 -m pip install --no-cache-dir \
    "huggingface_hub[hf_transfer]" \
    "sageattention==1.0.6" \
    accelerate \
    transformers \
    safetensors \
    einops \
    opencv-python-headless \
    pillow \
    soundfile \
    scipy

  if [[ ! -d "$PERSIST_ROOT/ComfyUI" ]]; then
    git clone --branch v0.3.26 --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$PERSIST_ROOT/ComfyUI"
  fi
  python3 -m pip install --upgrade --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
  # comfyui-frontend-package==1.11.8 was yanked or unavailable; install latest compatible release first
  python3 -m pip install --no-cache-dir "comfyui-frontend-package>=1.12.0"
  grep -v "comfyui-frontend-package" "$PERSIST_ROOT/ComfyUI/requirements.txt" | python3 -m pip install --no-cache-dir -r /dev/stdin

  mkdir -p "$PERSIST_ROOT/ComfyUI/custom_nodes"
  CUSTOM_NODES=(
    "https://github.com/kijai/ComfyUI-WanVideoWrapper.git"
    "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
    "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
    "https://github.com/kijai/ComfyUI-Florence2.git"
    "https://github.com/kijai/ComfyUI-segment-anything-2.git"
    "https://github.com/shmuelronen/ComfyUI-LatentSyncWrapper.git"
    "https://github.com/kijai/ComfyUI-MMAudio.git"
    "https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git"
    "https://github.com/kijai/ComfyUI-KJNodes.git"
  )
  for repo in "${CUSTOM_NODES[@]}"; do
    name="$(basename "$repo" .git)"
    target="$PERSIST_ROOT/ComfyUI/custom_nodes/$name"
    if [[ ! -d "$target/.git" ]]; then
      git clone --depth 1 "$repo" "$target"
    fi
    if [[ -f "$target/requirements.txt" ]]; then
      python3 -m pip install --no-cache-dir -r "$target/requirements.txt" || true
    fi
  done
  sed -i 's/from comfy.ldm.flux.math import apply_rope1 as apply_rope_comfy1/from comfy.ldm.flux.math import apply_rope as apply_rope_comfy1/' \
    "$PERSIST_ROOT/ComfyUI/custom_nodes/ComfyUI-WanVideoWrapper/wanvideo/modules/model.py" 2>/dev/null || true

  if [[ ! -d "$PERSIST_ROOT/ai-toolkit/.git" ]]; then
    git clone --depth 1 https://github.com/ostris/ai-toolkit.git "$PERSIST_ROOT/ai-toolkit"
  fi
  python3 -m pip install --no-cache-dir -r "$PERSIST_ROOT/ai-toolkit/requirements.txt" || true
  python3 -m pip install --no-cache-dir "mediapipe<0.10.20" || true

  if [[ ! -d "$PERSIST_ROOT/musubi-tuner/.git" ]]; then
    git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git "$PERSIST_ROOT/musubi-tuner"
  fi
  python3 -m pip install --no-cache-dir -e "$PERSIST_ROOT/musubi-tuner" || true

  touch "$PERSIST_ROOT/.installed_v1"
fi

ln -sfn "$PERSIST_ROOT/ComfyUI" /opt/ComfyUI
ln -sfn "$PERSIST_ROOT/ai-toolkit" /opt/ai-toolkit
ln -sfn "$PERSIST_ROOT/musubi-tuner" /opt/musubi-tuner

# 3. Initialize ephemeral RAM / encrypted scratch workspace
SHM_AVAIL_MB="$(df -Pm /dev/shm | awk 'NR==2 {print $4}')"
SCRATCH_ROOT=""

if [[ "${SHM_AVAIL_MB:-0}" -ge "$MIN_SHM_MB" ]]; then
  SCRATCH_ROOT="/dev/shm/factory_scratch"
  mkdir -p "$SCRATCH_ROOT"
  chmod 700 "$SCRATCH_ROOT"
  echo "[remote_bootstrap] Using RAM-backed tmpfs scratch at $SCRATCH_ROOT (${SHM_AVAIL_MB} MB free)"
else
  echo "[remote_bootstrap] WARNING: /dev/shm only has ${SHM_AVAIL_MB:-0} MB free (< ${MIN_SHM_MB} MB)." >&2
  SCRATCH_ROOT="/tmp/factory_scratch"
  if [[ -e /dev/fuse ]] && command -v gocryptfs >/dev/null 2>&1; then
    CIPHER_DIR="/tmp/.factory_cipher"
    KEY_FILE="/dev/shm/.factory_key"
    mkdir -p "$CIPHER_DIR" "$SCRATCH_ROOT"
    chmod 700 "$CIPHER_DIR" "$SCRATCH_ROOT"
    if [[ ! -f "$KEY_FILE" ]]; then
      head -c 32 /dev/urandom | base64 > "$KEY_FILE"
      chmod 600 "$KEY_FILE"
    fi
    if [[ ! -f "$CIPHER_DIR/gocryptfs.conf" ]]; then
      gocryptfs -init -passfile "$KEY_FILE" "$CIPHER_DIR"
    fi
    if ! mountpoint -q "$SCRATCH_ROOT" 2>/dev/null; then
      gocryptfs -passfile "$KEY_FILE" "$CIPHER_DIR" "$SCRATCH_ROOT"
    fi
    echo "[remote_bootstrap] Mounted ephemeral gocryptfs overlay at $SCRATCH_ROOT"
  else
    mkdir -p "$SCRATCH_ROOT"
    chmod 700 "$SCRATCH_ROOT"
    echo "[remote_bootstrap] WARNING: /dev/fuse unavailable; falling back to chmod 700 $SCRATCH_ROOT" >&2
  fi
fi

mkdir -p "$SCRATCH_ROOT"/{datasets,active_loras,temp_renders,branches,jobs}
chmod -R 700 "$SCRATCH_ROOT"

mkdir -p /opt/ComfyUI/models/loras /opt/ComfyUI/input /opt/ComfyUI/output
ln -sfn "$SCRATCH_ROOT/active_loras" /opt/ComfyUI/models/loras/scratch
ln -sfn "$SCRATCH_ROOT/temp_renders" /opt/ComfyUI/input/scratch
ln -sfn "$SCRATCH_ROOT/temp_renders" /opt/ComfyUI/output/scratch

for sub in diffusion_models text_encoders vae loras sam2 controlnet depthanything latentsync mmaudio rife LLM; do
  mkdir -p "$MODELS_ROOT/$sub" "/opt/ComfyUI/models/$sub"
  for item in "$MODELS_ROOT/$sub"/*; do
    [[ -e "$item" ]] || continue
    base="$(basename "$item")"
    ln -sfn "$item" "/opt/ComfyUI/models/$sub/$base"
  done
done

# 4. Headless localhost-only ComfyUI startup
if ! curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo "[remote_bootstrap] Starting ComfyUI strictly on 127.0.0.1:8188 in tmux session nw_comfy..."
  if ! tmux has-session -t nw_comfy 2>/dev/null; then
    tmux new-session -d -s nw_comfy \
      "COMFYUI_DISABLE_ANALYTICS=1 python3 /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-metadata > /tmp/comfyui.log 2>&1"
  fi
  for _ in $(seq 1 45); do
    if curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done
fi

if ! curl -sf http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo "[remote_bootstrap] ERROR: ComfyUI failed to become ready on 127.0.0.1:8188" >&2
  tail -n 40 /tmp/comfyui.log >&2 || true
  exit 1
fi
echo "[remote_bootstrap] ComfyUI is listening on 127.0.0.1:8188."

# 5. Optional background model bootstrap (inherits HF_TOKEN from container environment)
if [[ "$BOOTSTRAP_MODELS" == "1" ]]; then
  if ! tmux has-session -t nw_models 2>/dev/null; then
    echo "[remote_bootstrap] Starting background model bootstrap in tmux session nw_models..."
    tmux new-session -d -s nw_models \
      "bash -lc '/workspace/nairobiwood/infra/bootstrap_models.sh --models-root=\"$MODELS_ROOT\" > /tmp/model_bootstrap.log 2>&1'"
  else
    echo "[remote_bootstrap] Background model bootstrap session nw_models already running."
  fi
fi
