# Antigravity Build Plan: ComfyFactory (macOS + MCP + Ephemeral Cloud GPU)

This plan translates the **Project Brief: Agentic Hybrid Video Pipeline (ComfyUI + Wan 2.2)**, the **Local MCP Server Specification (`comfy-factory-mcp`)**, and the **Interactive Frame Scrub, Repair & Continuation Utility Spec (`frame-branch-director`)** into a concrete, ordered implementation blueprint for an Antigravity execution agent on macOS.

**Core Objective:** Lock character identities across shots and produce cohesive, multi-shot stories under Antigravity agent control (with optional human "Pause-and-Pivot" visual overrides).

---

## 1. Pre-Build Review: Gaps & Conflicts Resolved

Before writing code, the execution agent must apply these architectural reconciliations across the three source documents:

| Area | Discrepancy / Gap in Source Docs | Resolution for Build |
| :--- | :--- | :--- |
| **MCP Tool Surface** | Doc 1 lists 4 generic tools; Doc 2 defines 7 concrete tools; Doc 3 adds 2 frame-branching tools (`extract_and_repair_frame`, `continue_from_repaired_frame`). | Standardize on **10 MCP tools**: Doc 2's 7 tools + `pull_live_workflow` (browser-to-baseline sync) + Doc 3's 2 `frame-branch-director` tools. |
| **Ephemeral Encryption (`tmpfs` vs. LUKS/FUSE)** | Doc 1 mentions LUKS and `gocryptfs`; many Vast.ai and RunPod containers lack `CAP_SYS_ADMIN` or `/dev/fuse`. | Default to RAM-backed `tmpfs` (`/dev/shm/factory_scratch` with `0700` permissions, explicit `df -h /dev/shm` size check, and immediate `shred -u`/`rm -rf` pull-and-purge). Use `gocryptfs` only when `/dev/fuse` is detected. |
| **ComfyUI JSON Format** | ComfyUI uses UI format in the browser and **API format** (keyed by string node IDs) for `POST /prompt`. | Store both `wan22_master_pipeline.json` and `wan22_frame_branch_pipeline.json` in **API format** with companion `*_node_map.json` files documenting every controllable node ID. |
| **Gated Model Weights & New Models** | `FLUX.1-dev` and `FLUX.1-Fill-dev` require a Hugging Face license acceptance and `HF_TOKEN`; Doc 3 also requires `SAM2` and `LatentSync`/`MuseTalk`. | Never bake tokens into the Docker image. Pass `HF_TOKEN` at runtime to `bootstrap_models.sh`, and include `FLUX.1-Fill-dev`, `SAM2`, and `LatentSync`/`MuseTalk` in `models_manifest.json`. |
| **MCP Config Placeholders** | Doc 2 uses `<YOUR_MAC_USER>` in `mcp_config.json`. | Dynamically resolve `$HOME` (`os.path.expanduser("~")`) when generating Antigravity's MCP configuration. |

---

## 2. Target Workspace Layout (`~/ComfyFactory`)

```text
~/ComfyFactory/
├── .env.example                          # VAST_API_KEY, RUNPOD_API_KEY, HF_TOKEN, SSH_KEY_PATH
├── .gitignore                            # Exclude .env, datasets/, loras/, outputs/, .venv/
├── datasets/                             # Local character reference folders (<character_name>/)
├── loras/                                # Downloaded .safetensors (<character_name>/)
├── outputs/
│   └── <project_name>/
│       ├── shot_01_v1.mp4                # Root render
│       ├── shot_01_v1.manifest.json      # Prompts, seeds, LoRAs, and branch lineage
│       └── branches/
│           └── f0045_checkpoint/         # Locked prefix, raw/repaired frames, & branch .mp4s
├── workflows/
│   ├── wan22_master_pipeline.json        # ComfyUI API-format master DAG
│   ├── wan22_master_node_map.json        # Named parameter -> (node_id, input_key)
│   ├── wan22_frame_branch_pipeline.json  # ComfyUI API-format split-graph (SAM2 + FLUX-Fill + Wan/LatentSync)
│   └── wan22_frame_branch_node_map.json  # Node map for carry / pivot / lipsync modes
├── infra/
│   ├── Dockerfile                        # Immutable CUDA 12.4 + ComfyUI + trainers + SAM2/LatentSync image
│   ├── entrypoint.sh                     # /dev/shm scratch setup + 127.0.0.1:8188 ComfyUI launch
│   ├── bootstrap_models.sh               # Parallel hf_transfer / aria2c model downloader
│   ├── models_manifest.json              # Pinned HF repos, filenames, and ComfyUI target dirs
│   ├── factory_up.sh                     # CLI wrapper for Vast.ai / RunPod provisioning + SSH tunnel
│   └── factory_down.sh                   # Instance termination + SSH tunnel cleanup
└── mcp-server/
    ├── pyproject.toml                    # Python dependencies (mcp / FastMCP, httpx, websockets, etc.)
    ├── .venv/                            # Isolated local Python virtual environment
    ├── server.py                         # FastMCP entrypoint exposing all 10 tools
    ├── cloud_providers.py                # Vast.ai & RunPod API/CLI adapters + state tracking
    ├── ssh_tunnel.py                     # SSH tunnel lifecycle, rsync/scp, & remote shred/purge
    ├── comfy_client.py                   # ComfyUI HTTP/WebSocket client & deterministic JSON patcher
    ├── lora_trainer.py                   # Remote VLM captioning + ai-toolkit & musubi-tuner runner
    ├── branch_director.py                # Local branch manifest, prefix caching, & frame extraction
    └── tests/                            # Unit tests for JSON patching, branching, & path safety
```

---

## 3. Phased Execution Plan

### Phase 1: Local Workspace & Security Scaffolding
1. Create `~/ComfyFactory/` and all subdirectories (`datasets/`, `loras/`, `workflows/`, `outputs/`, `infra/`, `mcp-server/tests/`).
2. Generate `.env.example` containing `FACTORY_HOME`, `VAST_API_KEY`, `RUNPOD_API_KEY`, `HF_TOKEN`, and `SSH_KEY_PATH`.
3. Generate `.gitignore` so biometric datasets, trained LoRAs, `.env`, and rendered videos are never committed to version control.
4. Check for `~/.ssh/id_ed25519_factory`; if absent, document or generate an ed25519 keypair for passwordless cloud SSH authentication.

**Acceptance Gate:** Directory tree exists; `.gitignore` blocks sensitive directories; `.env.example` documents all required keys.

---

### Phase 2: Portable Container & Model Bootstrapper (`infra/`)
1. **Write `infra/Dockerfile`:**
   - Base: Ubuntu 22.04 + CUDA 12.4 + PyTorch.
   - Install `SageAttention` and `FlashAttention`.
   - Clone **ComfyUI** at a pinned release tag/commit and install locked custom nodes:
     - `ComfyUI-WanVideoWrapper`
     - `ComfyUI-VideoHelperSuite`
     - `comfyui_controlnet_aux`
     - `ComfyUI-MMAudio`
     - `ComfyUI-Frame-Interpolation`
     - `ComfyUI-SAM2` (shared spatial isolation for frame repair and lip-sync)
     - `ComfyUI-LatentSync` / `ComfyUI-MuseTalk` (fast in-place audio lip-sync)
   - Install LoRA training suites (`ai-toolkit` for FLUX, `musubi-tuner` for Wan 2.2) and VLM auto-captioning dependencies (JoyCaption / Qwen-VL).
   - Disable ComfyUI Manager analytics/telemetry.
2. **Write `infra/entrypoint.sh`:**
   - Verify `/dev/shm` available space (`df -h /dev/shm`) and create `/dev/shm/factory_scratch/{datasets,active_loras,temp_renders,branches}` with `chmod 700`.
   - Optionally mount `gocryptfs` if `/dev/fuse` is available; otherwise enforce RAM-only `tmpfs` + strict post-job `shred`.
   - Symlink scratch subdirectories into ComfyUI's `loras/`, `input/`, and `output/` directories.
   - Start ComfyUI strictly on `127.0.0.1:8188` with `--disable-metadata` (never `0.0.0.0`).
3. **Write `infra/models_manifest.json` & `infra/bootstrap_models.sh`:**
   - Pin open-weight repos and target paths under `/workspace/models`: `Wan 2.2 14B` (`fp8_e4m3fn` / `bf16`), `UMT5-XXL`, `wan_2.1_vae`, `FLUX.1-dev` + `FLUX.1-Fill-dev` (gated, require `HF_TOKEN`), `SAM2`, `LatentSync`/`MuseTalk`, `DWPose`, `DepthAnythingV2`, `RIFE`, `MMAudio`, and optional `LightX2V` distillation LoRA.
   - Download in parallel using `hf_transfer` and `aria2c`, skipping files already present (supports persistent RunPod Network Volumes).

**Acceptance Gate:** `docker build` syntax and shellcheck (`bash -n` / `shellcheck`) pass on all `infra/` scripts; `models_manifest.json` parses cleanly.

---

### Phase 3: Cloud Provisioning & Zero-Trust Tunnel (`infra/` + `mcp-server/`)
1. **Implement `mcp-server/cloud_providers.py` (and `infra/factory_up.sh` / `factory_down.sh`):**
   - **Vast.ai adapter:** Query verified **Datacenter / Secure Cloud** offers only, filtering for `VRAM >= 48 GB` (`L40S`, `RTX 6000 Ada`, `A100 80GB`) sorted by lowest hourly cost under `max_price_hr`.
   - **RunPod adapter:** Launch Secure Cloud pod with `wan22-comfy-factory:latest` and optional persistent Network Volume mounted at `/workspace/models`.
   - Persist active instance metadata (provider, instance ID, SSH host/port, hourly rate, tunnel PID) in `~/ComfyFactory/.active_factory.json` (`0600`).
2. **Implement `mcp-server/ssh_tunnel.py`:**
   - Wait for SSH readiness, then spawn `ssh -N -L 8188:127.0.0.1:8188 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30` in the background.
   - Provide `rsync`/`scp` upload and download helpers plus a `purge_remote(remote_path)` helper that runs `find <path> -type f -exec shred -u {} + && rm -rf <path>`.
   - Verify tunnel health by polling `http://127.0.0.1:8188/system_stats`.

**Acceptance Gate:** Unit tests verify Vast.ai/RunPod filter logic (rejecting non-datacenter/unverified hosts), state file locking, and clean tunnel/process teardown.

---

### Phase 4: Master & Frame-Branch ComfyUI Workflows (`workflows/`)
1. **Author `workflows/wan22_master_pipeline.json` (API format)** covering all 4 pipeline blocks:
   - **Block 1 — Identity & Keyframe:** `FLUX.1-dev` + `Character_FLUX_LoRA.safetensors` generating deterministic Start Frame (and optional End Frame).
   - **Block 2 — Spatial & Physics Conditioning (Toggleable):** `DWPose` + `DepthAnythingV2` routed into `Wan VACE` / `Fun Control` for driving reference videos.
   - **Block 3 — Core Video Render:** `Wan 2.2 14B` (`fp8_e4m3fn` or `bf16`) + `UMT5-XXL` + `wan_2.1_vae` + `Character_Wan22_LoRA.safetensors` (`strength: 0.75–0.90`) + `ModelSamplingSD3` (`shift: 6.0`) + `KSampler` (`SageAttention` + optional `LightX2V` LoRA).
   - **Block 4 — Post-Production & Audio:** `RIFE VFI` (30/60fps) + `MMAudio` / `LatentSync` + `VHS_VideoCombine` (`.mp4`).
2. **Author `workflows/wan22_frame_branch_pipeline.json` (API format)** implementing the 4-stage tensor split-and-stitch graph:
   - **Stage 1 — Ingestion & Slicing:** `VHS_LoadVideo` + `ImageFromBatch` slicing `Batch_A` (`0..K-1`, locked) and `Frame_K` (`K`, 1 frame) or `Frames K..N`.
   - **Stage 2 — Shared Spatial Isolation & LoRA Inpainting:** `SAM2Segment` (from `mask_target_text` with `8–16px` dilation + Gaussian feather, or user-brushed mask PNG) feeding `FLUX.1-Fill-dev` + `Character_FLUX_LoRA` (`strength: 0.85`, `denoise: 0.55–0.75`) to produce `Frame_K_Repaired.png`.
   - **Stage 3 — Three-Mode Continuation:**
     - `carry`: `WanImageToVideo` (`start_image = Frame_K_Repaired`) keeping original prompt/shift (`6.0`).
     - `pivot`: `WanImageToVideo` or `WanFirstLastFrameToVideo` (optional `end_keyframe_prompt` via FLUX+LoRA, optional `DWPose` + `Wan VACE`) with new `continuation_prompt`.
     - `lipsync`: Bypasses Wan 2.2 completely; routes `Frames K..N` + `.wav` audio through `LatentSync` / `MuseTalk` masked to the lower face/jaw via the shared SAM2 sub-graph.
   - **Stage 4 — Seam-Free Stitching:** `ImageFromBatch` (dropping duplicate `Frame_K` index `0` on `carry`/`pivot`, optional 1-frame `0.5` alpha blend) + `ImageBatch` concat + `RIFE VFI` / `MMAudio` + `VHS_VideoCombine`.
3. **Author `wan22_master_node_map.json` & `wan22_frame_branch_node_map.json`:** Map every semantic parameter to exact `(node_id, input_key)` pairs.

**Acceptance Gate:** JSON schema validation confirms all node references (`["node_id", slot]`) in both workflows resolve to valid nodes and every entry in both node maps exists.

---

### Phase 5: Local MCP Server (`mcp-server/`)
1. **Implement `mcp-server/comfy_client.py`:**
   - Deterministic JSON loader & patcher using the node maps (plus raw `inspect_or_patch_node` overrides).
   - Async submission to `POST http://127.0.0.1:8188/prompt`, WebSocket progress/error listener on `ws://127.0.0.1:8188/ws` (catching CUDA OOM and node execution errors), and `GET /history/{prompt_id}` output retrieval.
   - Support `POST /free` (`{"unload_models": true}`) between heavy stages (e.g., unloading `FLUX.1-Fill-dev` before loading `Wan 2.2 14B`).
2. **Implement `mcp-server/lora_trainer.py`:**
   - Validate local dataset (15–40 images in `~/ComfyFactory/datasets/<character_name>/`), `rsync` to `/dev/shm/factory_scratch/datasets/<character_name>/`, run VLM captioning with `trigger_word`, execute `ai-toolkit` (FLUX LoRA) and `musubi-tuner` (Wan 2.2 LoRA), stream `.safetensors` back to `~/ComfyFactory/loras/<character_name>/`, and immediately purge remote dataset and training artifacts.
3. **Implement `mcp-server/branch_director.py`:**
   - Manage `shot_XX_vY.manifest.json` lineage trees and `branches/fXXXX_checkpoint/` directories locally on macOS.
   - Handle local prefix caching (`locked_prefix_0000_XXXX`), frame extraction, and keyframe preview comparison (`frame_XXXX_raw.png`, `frame_XXXX_repaired.png`, and `frame_(X-1).png`).
4. **Implement `mcp-server/server.py` (`FastMCP`)** exposing all 10 tools:
   - `factory_status()`
   - `factory_up(provider="vast", gpu_filter="L40S|RTX_6000_Ada", max_price_hr=0.95)`
   - `factory_down()`
   - `train_character_lora(character_name, local_dataset_dir, trigger_word, train_flux=True, train_wan22=True)`
   - `list_local_assets()`
   - `render_shot(workflow_template, character_name, prompt, negative_prompt, duration_sec=5, driving_video_path=None, lora_strength=0.85, seed=None)`
   - `inspect_or_patch_node(node_id, input_key, new_value)`
   - `pull_live_workflow(output_name)`
   - `extract_and_repair_frame(source_video_path, frame_index, character_name, repair_prompt, mask_target_text=None, mask_image_path=None, inpaint_denoise=0.65)`
   - `continue_from_repaired_frame(checkpoint_dir, character_name, mode, continuation_prompt=None, audio_wav_path=None, end_keyframe_prompt=None, driving_video_path=None, target_total_frames=None, lora_strength=0.85, seed=None)`

**Acceptance Gate:** All `pytest` unit tests pass (mocking ComfyUI HTTP/WS, SSH/rsync, branch manifest lineage, and cloud provider APIs); MCP server starts cleanly over `stdio` and responds to `tools/list`.

---

### Phase 6: Antigravity Registration & End-to-End Verification
1. Create `mcp-server/.venv`, install dependencies, and generate Antigravity's `mcp_config.json` with resolved `$HOME` paths (no `<YOUR_MAC_USER>` placeholders).
2. Run a local MCP `stdio` verification to confirm Antigravity can discover all 10 tools.
3. Document a manual live-cloud smoke test checklist (`factory_up` -> verify `http://localhost:8188` -> `train_character_lora` -> `render_shot` -> `extract_and_repair_frame` -> `continue_from_repaired_frame` -> `factory_down`).

---

## 4. Recommended Architectural Adjustments (For Character-Locked Agentic Storytelling)

To fully achieve the objective of **locking characters and producing multi-shot stories via agent control**, consider incorporating these six architectural adjustments (either during the initial build or as an immediate Phase 7):

1. **Multi-Character Scene Support (Preventing LoRA Identity Bleed):**
   - *Current limitation:* `render_shot`, `extract_and_repair_frame`, and `continue_from_repaired_frame` accept only a single `character_name: string`.
   - *Adjustment:* Accept a `characters` list (`[{name, trigger_word, strength, mask_region}]`) and wire **regional conditioning / attention masking** (via SAM2 or bounding boxes) in the FLUX keyframe stage so two or more locked characters can share a shot without facial or wardrobe blending.
2. **Cross-Shot Storyboard & Continuity Ledger (`story_manifest.json`):**
   - *Current limitation:* `shot_01_v1.manifest.json` tracks branches inside a single clip, not continuity across sequential shots.
   - *Adjustment:* Add a project-level `story_manifest.json` (tracking scene, wardrobe, lighting, active props, and shot order), a `last_frame_handoff` parameter in `render_shot` (using the final frame of Shot `N` to anchor Shot `N+1`), and a `compose_story_sequence` tool (FFmpeg concatenation + crossfade/audio ducking) so the agent can assemble full scenes.
3. **Automated Agent-Side QA Gate (Identity & Seam Scoring):**
   - *Current limitation:* Pause-and-Pivot relies heavily on a human scrubbing the timeline to spot glitches.
   - *Adjustment:* Have `render_shot` return a sampled **keyframe contact strip** plus a lightweight **face-embedding similarity score** (e.g., InsightFace/ArcFace cosine distance against the character's reference centroid) computed before remote purge. If identity drops below threshold at Frame `K`, the Antigravity agent can autonomously trigger `extract_and_repair_frame` + `continue_from_repaired_frame(mode="carry")`.
4. **Lossless Prefix Caching to Prevent Generational Compression Drift:**
   - *Current limitation:* Caching `locked_prefix_0000_0044.mp4` as a lossy `.mp4` introduces re-encoding degradation and color shifts at the seam when branching multiple times (`f45 -> f68 -> f75`).
   - *Adjustment:* Store `locked_prefix` locally as **lossless PNGs or FFV1/ProRes `.mkv`** and apply a `ColorMatch` node across the `Frame_K` seam, encoding to `.mp4` only on final export.
5. **Explicit VRAM Stage Unloading on 48 GB GPUs:**
   - *Current limitation:* Keeping `FLUX.1-dev`, `FLUX.1-Fill-dev`, `SAM2`, `Wan 2.2 14B`, `UMT5-XXL`, `LatentSync`, `RIFE`, and `MMAudio` loaded simultaneously risks CUDA OOM on 48 GB cards (L40S / RTX 6000 Ada).
   - *Adjustment:* Enforce `POST /free` (`{"unload_models": true}`) between `extract_and_repair_frame` and `continue_from_repaired_frame`, and keep `Wan 2.2 14B` in `fp8_e4m3fn` with `--lowvram` / async offload fallback.
6. **Audio-First / Dialogue-Driven Shot Duration:**
   - *Current limitation:* `lipsync` mode assumes the pre-rendered clip length already matches the `.wav` dialogue track.
   - *Adjustment:* Auto-probe `.wav` duration in `continue_from_repaired_frame` and `render_shot` to set `target_total_frames` (or pad/extend tail frames) so agent-generated dialogue lines never clip mid-word.
