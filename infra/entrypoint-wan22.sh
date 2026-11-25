#!/usr/bin/env bash
set -e

# 1. Setup RunPod SSH Key if provided
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" > /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Start SSH daemon
mkdir -p /run/sshd
/usr/sbin/sshd

echo "=============================================="
echo "    Nairobiwood Wan 2.2 ComfyUI Container    "
echo "=============================================="
echo "CUDA Version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch:      $(python3 -c 'import torch; print(torch.__version__)')"

# 2. Bootstrap Wan 2.2 models from HF if not already cached
/opt/bootstrap_wan22.sh

# 3. Start ComfyUI on 0.0.0.0:8188
echo "Starting ComfyUI on port 8188..."
cd /opt/ComfyUI
exec python3 main.py --listen 0.0.0.0 --port 8188 --disable-metadata
