# Nairobiwood — Portable RunPod Video Factory & Director Studio

A privacy-hardened, local-first AI video production factory controlled from macOS via Antigravity (`comfy-factory-mcp`) and rendered on ephemeral **RunPod Secure Cloud** GPUs (`>= 48 GB` VRAM).

## Architecture Split
- **Code & Workflows (`/Users/leston/code/Nairobiwood`):**
  - `infra/` — Immutable CUDA 12.4 Docker image, `/dev/shm` scratch entrypoint, model manifest & parallel bootstrapper, and CLI wrappers (`factory_up.sh`, `factory_down.sh`).
  - `workflows/` — Five deterministic ComfyUI API-format DAGs (`wan22_keyframes_pipeline.json`, `wan22_video_from_keyframes_pipeline.json`, `wan22_frame_repair_pipeline.json`, `wan22_frame_continue_pipeline.json`, `wan22_post_export_pipeline.json`) with companion `*_node_map.json` files.
  - `mcp-server/` — Local Python `FastMCP` server, RunPod GraphQL provisioner (`22/tcp` SSH-only, never exposing `8188`), SSH tunnel + `rsync` + `shred` Pull-and-Purge lifecycle, async LoRA training (`Qwen2.5-VL-7B` + `ai-toolkit` + `musubi-tuner`), Start/End Keyframe gate, Pause-and-Pivot branch manager (`FFV1` lossless prefixes + `4k + 1` Wan frame math), and localhost browser studio backend.
  - `mcp-app/` — Single-file bundled **Director Studio** UI (`mcp-app/dist/director.html`) supporting both the Antigravity side pane (`ui://comfy-factory/director.html`) and Safari/Chrome (`http://127.0.0.1:8190`).
- **Biometric & Media Assets (`~/ComfyFactory`, via `FACTORY_DATA_DIR`):**
  - `datasets/`, `loras/`, `voices/`, `outputs/`, `jobs/`, `.active_factory.json`

## Quick Start (Option A: Zero-Docker Official RunPod PyTorch + SSH Auto-Bootstrap)
1. Copy `.env.example` to `.env` and set `RUNPOD_API_KEY`, `HF_TOKEN`, and optionally `RUNPOD_NETWORK_VOLUME_ID` + `RUNPOD_DATACENTER_ID`.
   - Default `CONTAINER_IMAGE` uses RunPod's official `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` image; `factory_up` automatically uploads `infra/remote_bootstrap.sh` over SSH to install/cache ComfyUI + custom nodes + trainers under `/workspace/nairobiwood`, initializes RAM `/dev/shm/factory_scratch`, starts ComfyUI strictly on `127.0.0.1:8188`, and downloads weights in the background. *(An optional pre-baked `infra/Dockerfile` is also included for custom registry builds.)*
2. Generate an SSH key if needed (`ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519_factory -N ""`).
3. Start the factory via Antigravity (`factory_up`) or CLI:
   ```bash
   ./infra/factory_up.sh --gpu-filter "L40S|RTX 6000 Ada|A100" --max-price-hr 1.69
   ```
4. Launch the **Director Studio** in the Antigravity side pane (`open_director_studio`) or open `http://127.0.0.1:8190` in Safari/Chrome (`uv --directory mcp-server run nairobiwood-studio`).
5. Review **Start / End Keyframes** in Tab 1 before approving Wan 2.2 rendering, or scrub & repair any frame in Tab 2 (**Pause-and-Pivot**: `carry`, `pivot`, `lipsync`).
6. Tear down the pod and close tunnels when finished (`factory_down` or `./infra/factory_down.sh`).
