# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    COMFYUI_DISABLE_ANALYTICS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    aria2 \
    ffmpeg \
    tmux \
    jq \
    gocryptfs \
    openssh-server \
    coreutils \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Harden SSH: key-only auth
RUN sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config

# PyTorch CUDA 12.4 + attention acceleration
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install \
      torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
      --index-url https://download.pytorch.org/whl/cu124 && \
    python3 -m pip install \
      huggingface_hub[hf_transfer] \
      sageattention \
      flash-attn \
      accelerate \
      transformers \
      safetensors \
      einops \
      opencv-python-headless \
      pillow \
      soundfile \
      scipy

# Clone ComfyUI (pinned tag) & required custom nodes
WORKDIR /opt
RUN git clone --branch v0.3.26 --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    python3 -m pip install -r /opt/ComfyUI/requirements.txt

WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/kijai/ComfyUI-WanVideoWrapper.git && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    git clone --depth 1 https://github.com/Fannovel16/comfyui_controlnet_aux.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-Florence2.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-SAM2.git && \
    git clone --depth 1 https://github.com/shmuelronen/ComfyUI-LatentSyncWrapper.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-MMAudio.git && \
    git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    for req in */requirements.txt; do \
      if [ -f "$req" ]; then python3 -m pip install -r "$req" || true; fi; \
    done

# Install LoRA trainers (ai-toolkit for FLUX, musubi-tuner for Wan 2.2)
WORKDIR /opt
RUN git clone --depth 1 https://github.com/ostris/ai-toolkit.git /opt/ai-toolkit && \
    python3 -m pip install -r /opt/ai-toolkit/requirements.txt || true && \
    git clone --depth 1 https://github.com/kohya-ss/musubi-tuner.git /opt/musubi-tuner && \
    python3 -m pip install -e /opt/musubi-tuner || true

# Copy manifest, bootstrapper, and entrypoint
RUN mkdir -p /opt/nairobiwood/infra
COPY models_manifest.json /opt/nairobiwood/infra/models_manifest.json
COPY bootstrap_models.sh /opt/nairobiwood/infra/bootstrap_models.sh
COPY entrypoint.sh /opt/nairobiwood/infra/entrypoint.sh
RUN chmod +x /opt/nairobiwood/infra/bootstrap_models.sh /opt/nairobiwood/infra/entrypoint.sh

# Strictly expose SSH only; ComfyUI 8188 is never publicly exposed
EXPOSE 22

ENTRYPOINT ["/opt/nairobiwood/infra/entrypoint.sh"]
