### [2026-11-25] - Wan 2.2 ComfyUI Inference Pod Setup
- **Context:** Deploying an NVIDIA A40 inference pod on RunPod to run ComfyUI with Wan 2.2 base models and custom LoRAs.
- **Friction Points:**
  1. Default base image (`runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`) ships with PyTorch 2.2.0, whereas `ComfyUI-WanVideoWrapper` requires `@torch.library.custom_op` (available in PyTorch >= 2.4.0).
  2. Upgrading to PyTorch 2.6.0 triggered strict typing schema validation in `torch._library.infer_schema`, which rejected Python 3.10 PEP 585/604 built-in generics (`list[int]`, `Tensor | None`) used in `comfy_kitchen`.
  3. Multiple manual patching steps were required on the remote container (updating `SUPPORTED_PARAM_TYPES`, installing frontend packages, and syncing LoRA weights via rsync).
- **Impact:** Increased spin-up time and required interactive troubleshooting to prevent node import failures.
- **Suggested Simplification:** Build and publish a pre-baked Docker image containing PyTorch 2.6.0+cu124, ComfyUI v0.46+, `ComfyUI-WanVideoWrapper`, `ComfyUI-VideoHelperSuite`, and pre-cached base weights, so any new pod starts immediately in ready state.
