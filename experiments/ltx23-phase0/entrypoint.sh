#!/usr/bin/env bash
set -uo pipefail

export PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

# 1. Configure SSH authorized_keys from PUBLIC_KEY (Vast.ai / Runpod)
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null || \
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Start sshd only if Runpod/host runtime has not already started it on :22
mkdir -p /run/sshd
if ! pgrep -x sshd >/dev/null 2>&1; then
  /usr/sbin/sshd || true
fi

echo "=============================================="
echo "   Nairobiwood LTX-2.3 + FaceID Phase 0 Pod   "
echo "=============================================="
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Avail:', torch.cuda.is_available())" || true

# 2. Bootstrap LTX-2.3 FP8 + Best-Face-ID weights and launch ComfyUI in background
# Keep PID 1 (and sshd) alive on failure so logs can be inspected before teardown.
(
  set -euo pipefail
  /opt/bootstrap_ltx23.sh > /tmp/bootstrap_ltx23.log 2>&1
  cd /opt/ComfyUI
  exec /opt/venv/bin/python3 /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-metadata > /tmp/comfyui.log 2>&1
) &
CHILD_PID=$!

trap 'kill -TERM "$CHILD_PID" 2>/dev/null || true' TERM INT
wait "$CHILD_PID" || {
  echo "[entrypoint] Bootstrap or ComfyUI exited early; keeping SSH open for diagnostics." >&2
  sleep 600
}

