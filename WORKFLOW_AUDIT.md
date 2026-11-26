### [2026-11-25] - Wan 2.2 ComfyUI Inference Pod Setup
- **Context:** Deploying an NVIDIA A40 inference pod on RunPod to run ComfyUI with Wan 2.2 base models and custom LoRAs.
- **Friction Points:**
  1. Default base image (`runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04`) ships with PyTorch 2.2.0, whereas `ComfyUI-WanVideoWrapper` requires `@torch.library.custom_op` (available in PyTorch >= 2.4.0).
  2. Upgrading to PyTorch 2.6.0 triggered strict typing schema validation in `torch._library.infer_schema`, which rejected Python 3.10 PEP 585/604 built-in generics (`list[int]`, `Tensor | None`) used in `comfy_kitchen`.
  3. Multiple manual patching steps were required on the remote container (updating `SUPPORTED_PARAM_TYPES`, installing frontend packages, and syncing LoRA weights via rsync).
- **Impact:** Increased spin-up time and required interactive troubleshooting to prevent node import failures.
- **Suggested Simplification:** Build and publish a pre-baked Docker image containing PyTorch 2.6.0+cu124, ComfyUI v0.46+, `ComfyUI-WanVideoWrapper`, `ComfyUI-VideoHelperSuite`, and pre-cached base weights, so any new pod starts immediately in ready state.

### [2026-11-26] - GitHub Actions Docker Image Build Space Exhaustion
- **Context:** Building `ghcr.io/dfleston/nairobiwood-wan22:latest` using GitHub Actions `ubuntu-latest` runner to provide an automated pre-baked inference image.
- **Friction Points:** Standard GitHub Actions runners have only ~14 GB of free root disk because pre-installed language runtimes (.NET, Android SDK, Haskell, CodeQL) take over 40 GB. Installing PyTorch 2.6 CUDA 12.4 wheels and large pip packages triggered `OSError: [Errno 28] No space left on device`.
- **Impact:** Failed initial automated container build in CI.
- **Suggested Simplification:** Added a runner disk space liberation step (`sudo rm -rf /usr/share/dotnet /usr/local/lib/android /opt/ghc /opt/hostedtoolcache/CodeQL`) liberating ~40 GB before buildx executes, and added `--no-cache-dir` to all pip commands in `Dockerfile.wan22`.
