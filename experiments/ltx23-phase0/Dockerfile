# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_NO_CACHE=1 \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    COMFYUI_DISABLE_ANALYTICS=1

# 1. System packages (includes libsndfile1 & libgomp1 for librosa/onnxruntime/BFSNodes)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    wget \
    aria2 \
    ffmpeg \
    tmux \
    jq \
    rsync \
    openssh-server \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    libgomp1 \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Hardened SSH for cloud GPU access (key auth only)
RUN mkdir -p /run/sshd && \
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config && \
    sed -i 's/^#\?AllowTcpForwarding.*/AllowTcpForwarding yes/' /etc/ssh/sshd_config

# 3. Install uv + Python 3.11 virtualenv (required by colour-science>=0.4.7 & modern transformers)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/opt/venv/bin:/root/.local/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"
RUN uv venv /opt/venv --python 3.11

# 4. PyTorch 2.6.0 + CUDA 12.4
RUN uv pip install --no-cache \
      torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 \
      --index-url https://download.pytorch.org/whl/cu124

# 5. Clone ComfyUI (latest main for native LTX-2.3 nodes) & install requirements
WORKDIR /opt
RUN git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git /opt/ComfyUI && \
    uv pip install --no-cache -r /opt/ComfyUI/requirements.txt

# 6. Install required LTX-2.3 + Best-Face-ID custom nodes
WORKDIR /opt/ComfyUI/custom_nodes
RUN git clone --depth 1 https://github.com/Lightricks/ComfyUI-LTXVideo.git && \
    git clone --depth 1 https://github.com/alisson-anjos/ComfyUI-BFSNodes.git && \
    git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git && \
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git && \
    for req in */requirements.txt; do \
      if [ -f "$req" ]; then uv pip install --no-cache -r "$req" || true; fi; \
    done && \
    uv pip install --no-cache \
      "transformers>=4.55.0,<5.0.0" \
      "huggingface_hub[hf_transfer]>=0.26.0" \
      hf-transfer \
      accelerate \
      safetensors \
      einops \
      "colour-science>=0.4.4" \
      openimageio \
      kornia \
      diffusers \
      librosa \
      scenedetect \
      scipy \
      soundfile \
      opencv-python-headless \
      insightface \
      onnxruntime-gpu \
      matplotlib

# 7. Verify Python 3.11 runtime and all critical imports at build time
RUN python3 -c "import sys, torch, transformers, OpenImageIO, colour, librosa, insightface; \
    assert sys.version_info[:2] == (3, 11), sys.version; \
    print('LTX-2.3 Phase 0 runtime verified:', torch.__version__, transformers.__version__)"

# 8. Setup model directories & real physical scratch dirs (required by ComfyUI 0.46+ realpath checks)
RUN mkdir -p /opt/ComfyUI/models/diffusion_models \
             /opt/ComfyUI/models/unet \
             /opt/ComfyUI/models/text_encoders \
             /opt/ComfyUI/models/clip \
             /opt/ComfyUI/models/vae \
             /opt/ComfyUI/models/loras \
             /opt/ComfyUI/models/latent_upscale_models \
             /opt/ComfyUI/input/scratch \
             /opt/ComfyUI/output/scratch && \
    touch /opt/ComfyUI/models/loras/put_loras_here

COPY patch_torch_schema.py /opt/patch_torch_schema.py
RUN python3 /opt/patch_torch_schema.py && \
    cd /opt/ComfyUI && python3 -c "import sys, asyncio; sys.path.insert(0, '/opt/ComfyUI'); sys.argv = ['main.py', '--cpu']; \
    import comfy.options; comfy.options.enable_args_parsing(); \
    import nodes; \
    res = nodes.init_extra_nodes(); \
    asyncio.run(res) if asyncio.iscoroutine(res) else None; \
    req = ['UNETLoader','LoraLoaderModelOnly','DualCLIPLoader','VAELoader','LatentUpscaleModelLoader', \
           'LTXVConditioning','EmptyLTXVLatentVideo','LTXVEmptyLatentAudio','LTXVConcatAVLatent', \
           'LTXVSeparateAVLatent','LTXVCropGuides','LTXVLatentUpsampler','LTXVAudioVAEDecode', \
           'LTXIdentityOverlapConditioning','VHS_VideoCombine']; \
    missing = [n for n in req if n not in nodes.NODE_CLASS_MAPPINGS]; \
    assert not missing, f'Missing nodes: {missing}'; \
    print('Verified all 15 LTX-2.3 + BFSNodes + VHS nodes in CPU mode! Total:', len(nodes.NODE_CLASS_MAPPINGS))"

COPY ltx23_models_manifest.json /opt/ltx23_models_manifest.json
COPY bootstrap_ltx23.sh /opt/bootstrap_ltx23.sh
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /opt/bootstrap_ltx23.sh /entrypoint.sh && \
    ln -sf /opt/bootstrap_ltx23.sh /opt/bootstrap_wan22.sh

WORKDIR /opt/ComfyUI
EXPOSE 22 8188

ENTRYPOINT ["/entrypoint.sh"]
