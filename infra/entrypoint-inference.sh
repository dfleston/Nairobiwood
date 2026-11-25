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

echo "=== Nairobiwood Wan 2.2 Ready Container ==="
echo "CUDA Version: $(python3 -c 'import torch; print(torch.version.cuda)')"
echo "PyTorch: $(python3 -c 'import torch; print(torch.__version__)')"

# 2. Symlink persistent /workspace models if attached
if [[ -d "/workspace/models/loras" ]]; then
  echo "[entrypoint] Symlinking persistent /workspace/models/loras into ComfyUI..."
  mkdir -p /workspace/models/loras
  ln -sfn /workspace/models/loras/* /opt/ComfyUI/models/loras/ 2>/dev/null || true
fi

# 3. Start ComfyUI in foreground (or managed)
cd /opt/ComfyUI
exec python3 main.py --listen 0.0.0.0 --port 8188 --disable-metadata
