#!/usr/bin/env bash
set -euo pipefail

COMFY_DIR="${COMFY_DIR:-/opt/ComfyUI}"
MODELS_ROOT="${MODELS_ROOT:-/workspace/models}"
MIN_SHM_MB=8192

# 1. Configure SSH authorized_keys from RunPod PUBLIC_KEY
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

mkdir -p /run/sshd
/usr/sbin/sshd

# 2. Initialize ephemeral encrypted / RAM scratch workspace
SHM_AVAIL_MB="$(df -Pm /dev/shm | awk 'NR==2 {print $4}')"
SCRATCH_ROOT=""

if [[ "${SHM_AVAIL_MB:-0}" -ge "$MIN_SHM_MB" ]]; then
  SCRATCH_ROOT="/dev/shm/factory_scratch"
  mkdir -p "$SCRATCH_ROOT"
  chmod 700 "$SCRATCH_ROOT"
  echo "[entrypoint] Using RAM-backed tmpfs scratch at $SCRATCH_ROOT (${SHM_AVAIL_MB} MB free)"
else
  echo "[entrypoint] WARNING: /dev/shm only has ${SHM_AVAIL_MB:-0} MB free (< ${MIN_SHM_MB} MB)." >&2
  SCRATCH_ROOT="/tmp/factory_scratch"
  if [[ -e /dev/fuse ]] && command -v gocryptfs >/dev/null 2>&1; then
    CIPHER_DIR="/tmp/.factory_cipher"
    KEY_FILE="/dev/shm/.factory_key"
    mkdir -p "$CIPHER_DIR" "$SCRATCH_ROOT"
    chmod 700 "$CIPHER_DIR" "$SCRATCH_ROOT"
    head -c 32 /dev/urandom | base64 > "$KEY_FILE"
    chmod 600 "$KEY_FILE"
    if [[ ! -f "$CIPHER_DIR/gocryptfs.conf" ]]; then
      gocryptfs -init -passfile "$KEY_FILE" "$CIPHER_DIR"
    fi
    gocryptfs -passfile "$KEY_FILE" "$CIPHER_DIR" "$SCRATCH_ROOT"
    echo "[entrypoint] Mounted ephemeral gocryptfs overlay at $SCRATCH_ROOT"
  else
    mkdir -p "$SCRATCH_ROOT"
    chmod 700 "$SCRATCH_ROOT"
    echo "[entrypoint] WARNING: /dev/fuse unavailable; falling back to chmod 700 $SCRATCH_ROOT" >&2
  fi
fi

mkdir -p "$SCRATCH_ROOT"/{datasets,active_loras,temp_renders,branches,jobs}
chmod -R 700 "$SCRATCH_ROOT"
export FACTORY_SCRATCH_ROOT="$SCRATCH_ROOT"

# 3. Symlink scratch & persistent /workspace/models into ComfyUI
mkdir -p "$COMFY_DIR/models/loras" "$COMFY_DIR/input" "$COMFY_DIR/output"
ln -sfn "$SCRATCH_ROOT/active_loras" "$COMFY_DIR/models/loras/scratch"
ln -sfn "$SCRATCH_ROOT/temp_renders" "$COMFY_DIR/input/scratch"
ln -sfn "$SCRATCH_ROOT/temp_renders" "$COMFY_DIR/output/scratch"

if [[ -d "$MODELS_ROOT" ]]; then
  for sub in diffusion_models text_encoders vae loras sam2 controlnet depthanything latentsync mmaudio rife LLM; do
    if [[ -d "$MODELS_ROOT/$sub" ]]; then
      mkdir -p "$COMFY_DIR/models/$sub"
      for item in "$MODELS_ROOT/$sub"/*; do
        [[ -e "$item" ]] || continue
        base="$(basename "$item")"
        ln -sfn "$item" "$COMFY_DIR/models/$sub/$base"
      done
    fi
  done
fi

# 4. Optional on-boot model bootstrap when AUTO_BOOTSTRAP_MODELS=1
if [[ "${AUTO_BOOTSTRAP_MODELS:-0}" == "1" ]]; then
  /opt/nairobiwood/infra/bootstrap_models.sh --models-root="$MODELS_ROOT"
fi

# 5. Start ComfyUI strictly bound to localhost (never 0.0.0.0)
echo "[entrypoint] Starting ComfyUI bound strictly to 127.0.0.1:8188 (--disable-metadata)"
exec python3 "$COMFY_DIR/main.py" --listen 127.0.0.1 --port 8188 --disable-metadata
