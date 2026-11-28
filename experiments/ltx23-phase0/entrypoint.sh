#!/usr/bin/env bash
set -euo pipefail

export PATH="/opt/venv/bin:/root/.local/bin:${PATH}"

# 1. Configure SSH authorized_keys from PUBLIC_KEY (Vast.ai / Runpod)
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  grep -qxF "$PUBLIC_KEY" /root/.ssh/authorized_keys 2>/dev/null || \
    printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

mkdir -p /run/sshd
/usr/sbin/sshd

echo "=============================================="
echo "   Nairobiwood LTX-2.3 + FaceID Phase 0 Pod   "
echo "=============================================="
python3 -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Avail:', torch.cuda.is_available())"

# 2. Bootstrap LTX-2.3 FP8 + Best-Face-ID weights
/opt/bootstrap_ltx23.sh

# 3. Start ComfyUI bound strictly to 127.0.0.1:8188
echo "Starting ComfyUI on 127.0.0.1:8188 (--disable-metadata)..."
cd /opt/ComfyUI
exec /opt/venv/bin/python3 /opt/ComfyUI/main.py --listen 127.0.0.1 --port 8188 --disable-metadata
