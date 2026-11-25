# Local MCP Server Specification (`comfy-factory-mcp`)
**Role:** Local Control Plane Bridge between Antigravity (macOS), Ephemeral Cloud GPUs, and ComfyUI  
**Target Runtime:** Python (`FastMCP` / `mcp` SDK) running locally on macOS

---

## 1. Is ComfyUI MCP Not Open-Source? (Clarifying the Ecosystem)

**Yes—standard ComfyUI MCP servers ARE open-source** (such as `comfyui-mcp-server`, `mcp-comfyui`, etc.). However, out-of-the-box open-source ComfyUI MCP servers only wrap **ComfyUI's internal HTTP/WebSocket API** (`/prompt`, `/queue`, `/history`, `/view`). 

They assume:
1. ComfyUI is already running on a static IP/port.
2. All models and LoRAs are already sitting inside ComfyUI's local folders.
3. No infrastructure provisioning, SSH tunneling, LoRA training, non-destructive video frame branching, or secure file purging is needed.

### Why You Still Need a Custom Local MCP Wrapper (`comfy-factory-mcp`)
In our **"Moveable Factory"** architecture, ComfyUI is only **one component** running inside an ephemeral, encrypted cloud container that spins up on demand and shuts down afterward. A standard open-source ComfyUI MCP cannot:

| Capability Gap | Standard Open-Source ComfyUI MCP | Custom `comfy-factory-mcp` (Local macOS) |
| :--- | :--- | :--- |
| **Cloud GPU Lifecycle** | Cannot rent, boot, or destroy cloud GPUs. | Provisions/terminates **Vast.ai / RunPod** instances via CLI/API on demand so you stop paying when idle. |
| **Zero-Trust SSH Tunneling** | Expects an exposed HTTP port (`http://ip:8188`). | Automatically opens and monitors an encrypted **SSH tunnel (`localhost:8188`)** so ComfyUI never touches the public internet. |
| **Automated LoRA Training** | Only runs inference graphs; cannot trigger `ai-toolkit` or `musubi-tuner`. | Encrypts your local macOS photo folder, pushes it to cloud RAM (`tmpfs`), runs FLUX + Wan 2.2 LoRA training, and pulls back the `.safetensors`. |
| **Mid-Clip Frame Scrub, Repair & Branching** | Treats video generation as an all-or-nothing single pass. | Splits rendered clips at any user-selected frame, repairs that frame via LoRA-locked inpainting, and regenerates only subsequent frames (`Carry` or `Pivot` mode). |
| **Local-First Asset Sync** | Leaves generated `.mp4` and `.safetensors` files stranded on the remote server. | Implements **Pull-and-Purge**: streams finished weights/videos back to `~/ComfyFactory/` on macOS and runs `shred`/`rm -rf` remotely. |
| **Semantic Graph Patching** | Often requires the LLM to emit raw 500-line ComfyUI JSONs (prone to syntax/link errors). | Exposes high-level semantic tools (`set_character_lora`, `toggle_dwpose`, `branch_from_frame`) that deterministically patch node IDs in ComfyUI graphs. |

> **Design Pattern:** Our local MCP server **incorporates/extends** the standard open-source ComfyUI API client patterns for graph execution, while wrapping them in infrastructure, LoRA training, interactive frame-branching, and privacy orchestration tailored for macOS + Antigravity.

---

## 2. How the Local MCP Operates (End-to-End Lifecycle)

```
┌─────────────────────────────────────────────────────────────────────────┐
│ LOCAL CONTROL PLANE (macOS)                                             │
│                                                                         │
│  ┌──────────────┐   MCP (stdio)   ┌──────────────────────────────────┐  │
│  │ Antigravity  │ ◄─────────────► │ comfy-factory-mcp (Python)       │  │
│  │ (IDE/Agent)  │                 │  ├─ Cloud Provisioner (Vast/RP)  │  │
│  └──────────────┘                 │  ├─ SSH Tunnel & SFTP/Rsync      │  │
│                                   │  ├─ LoRA Training Orchestrator   │  │
│  ┌──────────────┐                 │  ├─ Frame-Branch State Manager   │  │
│  │ Safari/Chrome│                 │  └─ ComfyUI API & JSON Patcher   │  │
│  │ localhost:   │                 └────────────────┬─────────────────┘  │
│  │ 8188 (Visual)│ ◄─────── Encrypted SSH Tunnel ───┼─────────────────┐  │
│  └──────────────┘          (127.0.0.1:8188)        │                 │  │
└────────────────────────────────────────────────────┼─────────────────┼──┘
                                                     ▼                 │
┌──────────────────────────────────────────────────────────────────────┼──┐
│ EPHEMERAL COMPUTE PLANE (Cloud GPU Container - Vast.ai / RunPod)     │  │
│                                                                      │  │
│  ┌──────────────────────┐    ┌────────────────────────────────────┐  │  │
│  │ Encrypted tmpfs RAM  │    │ ComfyUI Headless Engine            │◄─┘  │
│  │ /dev/shm/workspace   │◄──►│ (Bound strictly to 127.0.0.1:8188) │     │
│  │ ├─ raw_dataset/      │    │ ├─ FLUX + Character LoRA (Key/Fix) │     │
│  │ ├─ active_loras/     │    │ ├─ Wan 2.2 14B + Character LoRA    │     │
│  │ └─ temp_renders/     │    │ └─ ControlNet / VACE / MMAudio     │     │
│  └──────────────────────┘    └────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────┘
```

### Phase A: Boot & Tunnel Establishment (`factory_start`)
1. You tell Antigravity: *"Spin up the video factory on an L40S."*
2. `comfy-factory-mcp` queries Vast.ai / RunPod API for the cheapest verified datacenter GPU matching your VRAM threshold (`>= 48 GB`), launches `wan22-comfy-factory:latest`, and waits for SSH readiness.
3. It mounts an in-memory encrypted workspace (`/dev/shm/factory_scratch`) on the remote host and establishes a background SSH tunnel mapping remote `127.0.0.1:8188` to local `127.0.0.1:8188`.
4. **Result:** Both the Antigravity agent (via API) and you (via local browser at `http://localhost:8188`) have direct, private access to ComfyUI.

### Phase B: Character LoRA Creation (`train_character_lora`)
1. You place 20–30 reference images into `~/ComfyFactory/datasets/elena/`.
2. Antigravity invokes `train_character_lora(dataset="elena", trigger="elena_v1")`.
3. The local MCP server streams the images via `rsync` over SSH directly into the remote RAM disk (`/dev/shm/factory_scratch/datasets/elena`), runs automated VLM captioning, and executes:
   * `ai-toolkit` → produces `elena_v1_flux.safetensors`
   * `musubi-tuner` → produces `elena_v1_wan22.safetensors`
4. Immediately upon completion, `comfy-factory-mcp` downloads both `.safetensors` files to `~/ComfyFactory/loras/elena/` on your Mac and **wipes the remote dataset and training artifacts** from `/dev/shm`.

### Phase C: Hybrid Node Editing & Interactive Frame Branching (Required Utility)
* **Initial Shot Generation:** When you ask Antigravity to generate a shot, the MCP server loads `wan22_master_pipeline.json`, injects your local LoRA, patches prompts/switches, renders the clip, and pulls the `.mp4` locally.
* **Interactive Scrub, Stop & Branch (`edit_and_branch_from_frame`):**
  1. **UX Trigger:** You preview the local `.mp4` in Antigravity, scrub to an exact timestamp (e.g., **Frame 45**), and instruct the agent: *"Pause at Frame 45, fix the hand holding the cup, and pivot her expression to suspicious for the rest of the clip."*
  2. **Tensor Slicing (`ImageFromBatch`):** The MCP uploads the local clip to `/dev/shm/factory_scratch/`, slices `Batch_A` (`frames 0..44`) to lock the approved opening segment, and extracts `frame_45.png`.
  3. **LoRA-Locked Inpainting (`FLUX-Fill` + Character LoRA + SAM2):** The MCP runs an inpainting branch on `frame_45.png` (using either a user-supplied brush mask or automated SAM2 object segmentation from text), producing `frame_45_fixed.png` with pixel-accurate lighting and facial consistency. It returns `frame_45_fixed.png` to Antigravity for instant visual confirmation.
  4. **Forward Regeneration (`WanImageToVideo` / `WanFirstLastFrameToVideo`):** Once confirmed (or in a single pass), the MCP routes `frame_45_fixed.png` as the `start_image` for `Batch_B` (`frames 45..N`), applying either:
     * **`mode="carry"`:** Maintains original prompt and motion vectors so the visual repair carries smoothly through the remainder of the shot.
     * **`mode="pivot"`:** Injects a new narrative prompt, target `end_image`, or `DWPose` driving sequence to redirect the story from Frame 45 onward.
  5. **Seamless Stitching & Branch History:** `Batch_A` + `Batch_B` are concatenated via `ImageBatch` (dropping the duplicate seam frame), encoded to `.mp4`, downloaded to `~/ComfyFactory/outputs/<project>/branches/`, and purged from cloud RAM. Because `Batch_A` is preserved locally, you can generate multiple alternate story branches from the exact same Frame 45 checkpoint.
* **Manual Visual Tweak:** At any point, you can open `http://localhost:8188` in your Mac browser to rewire nodes manually and pull the updated graph JSON back into Antigravity.

### Phase D: Pull-and-Purge Output & Teardown (`fetch_and_clean` / `factory_stop`)
1. Via WebSocket (`ws://127.0.0.1:8188/ws`), `comfy-factory-mcp` monitors step-by-step sampling progress and catches any CUDA OOM or node errors automatically.
2. Once `VHS_VideoCombine` finishes writing the `.mp4`, the MCP server downloads the file to `~/ComfyFactory/outputs/<project>/`, deletes the remote copy, and returns the local Mac file path (plus extracted keyframe thumbnails so Antigravity's vision model can inspect quality).
3. When you finish your session, `factory_stop` destroys the cloud container and closes the SSH tunnel—halting all hourly billing and guaranteeing zero residual data in the cloud.

---

## 3. Core MCP Tool Surface (Exposed to Antigravity)

The local Python MCP server exposes these clean, deterministic tools to Antigravity:

1. **`factory_status()`**  
   Returns active cloud GPU provider, GPU model, hourly burn rate, SSH tunnel health, VRAM utilization, and currently loaded models.
2. **`factory_up(provider="vast", gpu_filter="L40S|RTX_6000_Ada", max_price_hr=0.95)`**  
   Provisions the containerized factory, runs `bootstrap_models.sh` if needed, mounts `tmpfs`, and opens `localhost:8188`.
3. **`factory_down()`**  
   Terminates the active cloud GPU instance and closes local SSH tunnels.
4. **`train_character_lora(character_name, local_dataset_dir, trigger_word, train_flux=True, train_wan22=True)`**  
   Uploads dataset to RAM disk, runs captioning + dual LoRA training, downloads `.safetensors` to `~/ComfyFactory/loras/`, and purges remote files.
5. **`list_local_assets()`**  
   Lists available character LoRAs, driving pose videos, rendered clips, and base JSON workflow templates stored locally on macOS.
6. **`render_shot(workflow_template, character_name, prompt, negative_prompt, duration_sec=5, driving_video_path=None, lora_strength=0.85, seed=None)`**  
   Uploads the required character LoRA (and optional driving video) into the ephemeral container, patches the ComfyUI node graph, executes the render, downloads the `.mp4` to `~/ComfyFactory/outputs/`, purges remote temp files, and returns the local path + preview frames.
7. **`extract_and_repair_frame(source_video_path, frame_index, character_name, repair_prompt, mask_target_text=None, mask_image_path=None)`**  
   Extracts `frame_index` from a local video, locks preceding frames (`0..frame_index-1`), applies LoRA-locked FLUX inpainting to `frame_index`, and returns the repaired preview PNG locally before continuing video generation.
8. **`continue_from_repaired_frame(source_video_path, frame_index, repaired_frame_path, character_name, mode="carry", continuation_prompt=None, end_keyframe_prompt=None, driving_video_path=None)`**  
   Runs Wan 2.2 forward from `repaired_frame_path` for the remaining frame count, stitches it seamlessly onto locked frames `0..(frame_index - 1)`, saves the branched `.mp4` locally, and purges remote scratch files.
9. **`inspect_or_patch_node(node_id, input_key, new_value)`**  
   Allows surgical parameter overrides on any specific node ID in the active workflow without rewriting the entire graph.

---

## 4. Antigravity MCP Configuration (`mcp_config.json`)

To wire this local MCP server into **Antigravity** on macOS, the execution agent will generate the server at `~/ComfyFactory/mcp-server/server.py` and register it in Antigravity's MCP settings:

```json
{
  "mcpServers": {
    "comfy-factory": {
      "command": "/Users/<YOUR_MAC_USER>/ComfyFactory/mcp-server/.venv/bin/python",
      "args": [
        "/Users/<YOUR_MAC_USER>/ComfyFactory/mcp-server/server.py"
      ],
      "env": {
        "FACTORY_HOME": "/Users/<YOUR_MAC_USER>/ComfyFactory",
        "VAST_API_KEY": "${VAST_API_KEY}",
        "RUNPOD_API_KEY": "${RUNPOD_API_KEY}",
        "SSH_KEY_PATH": "/Users/<YOUR_MAC_USER>/.ssh/id_ed25519_factory"
      }
    }
  }
}
```