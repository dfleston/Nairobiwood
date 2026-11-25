# Project Brief: Portable, Private & Agent-Orchestrated Video Factory (macOS + Antigravity + Cloud GPU)

## 1. Executive Summary & Objective
Build a **portable, privacy-hardened AI video production factory** controlled locally from **macOS** via **Antigravity** (IDE & Agentic Runtime) using the **Model Context Protocol (MCP)**, while offloading heavy training and inference workloads to the **most cost-effective cloud GPUs**.

The system operates on two core pillars:
1. **Automated Character LoRA Factory:** Converts a local folder of character reference photos into downloadable, owned `.safetensors` LoRA weights (for both **FLUX** keyframes and **Wan 2.2** native video).
2. **Agent-Orchestrated Video Pipeline:** Uses ComfyUI node graphs (`JSON` DAGs) manipulated by the Antigravity agent (or visually by the user in a local browser) to render consistent, physically accurate video clips with Wan 2.2, including **non-destructive mid-clip frame repair and story branching**.

### Architectural Philosophy: "The Moveable Factory"
* **Local Control Plane (macOS):** Hosts Antigravity, the MCP Server, your raw reference datasets, downloaded `.safetensors` LoRAs, final `.mp4` renders, and the ComfyUI browser interface (`localhost:8188`). **Zero sensitive data remains permanently on cloud disks.**
* **Portable Compute Plane (Containerized Cloud GPU):** Everything required to run the factory (CUDA drivers, PyTorch, `SageAttention`, ComfyUI, custom nodes, `ai-toolkit`/`musubi-tuner` for LoRA training, and model bootstrap scripts) is encapsulated in a **single portable Docker image + Infrastructure-as-Code (IaC) script**. You can spin up, tear down, or migrate the entire factory between **Vast.ai**, **RunPod**, or **TensorDock** in under 5 minutes.

---

## 2. System Architecture (Input Specification for Execution Agent)

### Layer 1: Local Control Plane (macOS + Antigravity)
* **Host OS:** macOS (Apple Silicon / Intel) running **Antigravity**.
* **MCP Server (`comfy-factory-mcp`):** Local Python/Node MCP server registered in Antigravity's MCP configuration. Exposes high-level agent tools:
  * `provision_factory(provider, gpu_type)`: Launches the Docker container on the cheapest available cloud GPU and establishes an encrypted tunnel.
  * `train_character_lora(local_folder_path, character_trigger, target_models=["flux", "wan2.2"])`: Encrypts and uploads the local dataset to ephemeral GPU RAM/scratch storage, triggers automated VLM captioning + LoRA training, downloads the resulting `.safetensors` files directly to `~/ComfyFactory/loras/`, and purges remote copies.
  * `load_and_patch_workflow(workflow_name, params_dict)`: Modifies node parameters (prompts, LoRA strength, ControlNet toggles, frame counts) in `wan22_master_pipeline.json`.
  * `execute_and_fetch(workflow_json)`: Queues the render on the remote ComfyUI backend, streams progress, downloads the output `.mp4` to `~/ComfyFactory/outputs/`, and deletes remote output artifacts.
  * `edit_and_branch_from_frame(source_video_path, frame_index, edit_prompt, continuation_prompt, mask_path=None, mode="carry|pivot")`: Core non-destructive editing utility that locks frames `0..(frame_index - 1)`, repairs `frame_index` via LoRA-locked inpainting, and regenerates only the remaining timeline forward.
* **Visual Node Override:** User opens `http://localhost:8188` in Safari/Chrome on macOS—routed over an encrypted SSH/Tailscale tunnel—to visually inspect or manually rewire nodes at any time.

### Layer 2: Automated LoRA Training & Extraction Pipeline (Cloud Ephemeral Scratch)
* **Input:** User drops 15–40 clean reference images into `~/ComfyFactory/datasets/<character_name>/`.
* **Automated Preprocessing:** Local or container-side VLM (e.g., JoyCaption / Qwen-VL) auto-generates trigger-word captions separating immutable facial traits from variable clothing/backgrounds.
* **Dual-Target Training Engine:**
  * **FLUX LoRA (`ai-toolkit`):** Trains high-res identity weights for generating locked Start/End keyframes.
  * **Wan 2.2 Video LoRA (`musubi-tuner` / `diffusion-pipe`):** Trains temporal cross-attention weights so the character maintains 3D facial consistency during head turns and dynamic lighting.
* **Artifact Delivery:** Immediately upon completion, `.safetensors` weights are streamed back via `scp`/`rsync` over SSH to macOS (`~/ComfyFactory/loras/`), giving the user permanent, portable ownership.

### Layer 3: ComfyUI Master Execution Graph (`wan22_master_pipeline.json`) & Split-Graph Continuation
Executed on the cloud GPU container:
1. **Identity & Keyframe Block:** `FLUX.1-dev` + `Character_FLUX_LoRA.safetensors` generates deterministic Start Frame (and optional End Frame).
2. **Spatial & Physics Conditioning Block (Toggleable):** `DWPose` (skeletal tracking) + `DepthAnythingV2` fed into `Wan VACE` / `Fun Control` to enforce exact body physics from driving reference videos.
3. **Core Video Render Block:** `Wan 2.2 14B` (`fp8_e4m3fn` or `bf16`) + `UMT5-XXL` + `wan_2.1_vae` + `Character_Wan22_LoRA.safetensors` (`strength: 0.75–0.90`) + `ModelSamplingSD3` (`shift: 6.0`) + `KSampler` (accelerated with `SageAttention` + optional `LightX2V` distillation LoRA).
4. **Interactive Mid-Clip Repair & Branching Utility (Required Production Capability):**
   * **Frame Splitting (`ImageFromBatch`):** Slices an existing video tensor at user-selected `frame_index` (e.g., Frame 45), locking `Batch_A` (`0..44`) with zero recalculation.
   * **LoRA-Locked Frame Repair (`FLUX Inpaint` + SAM2/Mask):** Repairs localized flaws (hand anatomy, facial expression, lighting, prop interaction) on `frame_index` using the character's FLUX LoRA so visual continuity with `frame_index - 1` is pixel-exact.
   * **Forward Continuation (`WanImageToVideo` / `WanFirstLastFrameToVideo`):** Feeds `frame_index_fixed` as the new `start_image` for `Batch_B` (`45..N`), supporting two directorial modes:
     * **Carry Mode:** Propagates the visual fix seamlessly while continuing the original action/camera trajectory.
     * **Pivot Mode:** Alters the prompt, target `end_image`, or `DWPose` skeleton from `frame_index` onward to branch into a new narrative beat or emotion without jump cuts.
   * **Temporal Stitching (`ImageBatch`):** Concatenates locked `Batch_A` + regenerated `Batch_B` (dropping the duplicate seam frame) before encoding.
5. **Post-Production & Audio Block:** `RIFE VFI` (frame interpolation to 30/60fps) + `MMAudio` / `LatentSync` + `VHS_VideoCombine` (`.mp4` output).

---

## 3. Containerization Strategy ("The Moveable Factory")

To avoid vendor lock-in and eliminate manual model re-installation every time you rent a cheap GPU, the compute plane is packaged into **three decoupled layers**:

1. **Immutable Base Docker Image (`ghcr.io/<user>/wan22-comfy-factory:latest`):**
   * Pre-baked Ubuntu 22.04 + CUDA 12.4 + PyTorch + `SageAttention` + `FlashAttention`.
   * Pre-installed **ComfyUI** + locked versions of all required custom nodes (`ComfyUI-WanVideoWrapper`, `ComfyUI-VideoHelperSuite`, `comfyui_controlnet_aux`, `ComfyUI-MMAudio`, `ComfyUI-Frame-Interpolation`, `ComfyUI-SAM2`).
   * Pre-installed LoRA training suites (`ai-toolkit` and `musubi-tuner`).
   * *Result:* Zero dependency conflicts; container boots in <15 seconds on any cloud provider.
2. **Fast Model Bootstrapper (`bootstrap_models.sh`):**
   * Instead of bloating the Docker image to 90 GB, base open-weights (`Wan 2.2 14B`, `FLUX.1-dev`, `FLUX.1-Fill-dev` for inpainting, `UMT5`, `VAE`, `ControlNets`) are pulled in parallel at high speed (1–10 Gbps datacenter bandwidth) from Hugging Face or a private S3/Backblaze B2 bucket into a mounted volume (`/workspace/models`) using `hf_transfer` / `aria2c`.
   * If staying within one provider (e.g., RunPod Network Volume), models persist across sessions at ~$0.07/GB/month (~$10/mo for 150 GB) for instant 0-second startup.
3. **One-Command Cloud Agnostic Launcher (`factory_up.sh`):**
   * CLI script runnable by Antigravity that provisions the instance via provider API (Vast.ai CLI / RunPod API), injects your SSH public key, mounts an encrypted RAM/LUKS scratch volume, and opens the local tunnel.

---

## 4. Most Affordable Cloud Providers & Privacy Comparison

To run `FLUX` + `Wan 2.2 14B` + LoRA training comfortably, target **48 GB VRAM (RTX 6000 Ada / L40S)** or **80 GB VRAM (A100 SXM)** using `fp8` quantization.

| Provider | Best Value GPU | VRAM | Est. Cost / Hr | Portability & Convenience | Privacy & Security Posture |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vast.ai** *(Top Pick for Lowest Cost)* | **RTX 6000 Ada** or **A100 80GB** (Datacenter / Verified tier) | 48–80 GB | **$0.45 – $0.95/hr** | Native custom Docker image support; spin up via `vastai` CLI from Antigravity. | **Filter strictly for "Datacenter / Secure Cloud" verified hosts** to prevent physical host snooping; use encrypted `tmpfs` RAM disk for datasets. |
| **RunPod** *(Top Pick for Convenience)* | **NVIDIA L40S** or **A100 80GB** (Secure Cloud) | 48–80 GB | **$0.79 – $1.69/hr** | Persistent **Network Volumes** mean models install once; instant pod boot in ~10s with your Docker image. | SOC2-compliant Secure Cloud datacenters; isolated containers; zero data shared with third-party model APIs. |
| **TensorDock** | **NVIDIA L40S / A100** | 48–80 GB | **$0.55 – $1.15/hr** | Full root VM/container control; easy SSH port forwarding and custom startup scripts. | Enterprise tier-3/4 datacenters; full disk encryption supported on VM instances. |
| **Lambda Labs** | **1x NVIDIA H100 PCIe** | 80 GB | **~$2.49/hr** | Fastest LoRA training (~8–12 mins) and rapid batch rendering; great for short burst jobs. | Strict enterprise isolation; zero retention upon instance termination. |

---

## 5. End-to-End Privacy & Zero-Retention Protocol

Because character reference photos and trained LoRA weights represent sensitive biometric/IP assets, the execution agent must enforce these rules:

1. **Zero Public Endpoints:** ComfyUI on the cloud container must bind strictly to `127.0.0.1:8188` (never `0.0.0.0` or public proxy URLs). All traffic travels exclusively over **SSH Port Forwarding (`ssh -L 8188:127.0.0.1:8188`)** or an authenticated **Tailscale** wireguard mesh.
2. **Ephemeral Encrypted Scratch Space (`tmpfs` / `gocryptfs`):** Reference photos uploaded for LoRA training and intermediate rendered frames are stored inside an in-memory `tmpfs` filesystem (or encrypted overlay) on the cloud GPU—never written unencrypted to shared host disks.
3. **Pull-and-Purge Lifecycle:** As soon as a LoRA `.safetensors` or final `.mp4` render completes, the local MCP server downloads it to macOS and immediately executes `shred` / `rm -rf` on the remote scratch directory.
4. **No Telemetry / Offline-Capable Nodes:** Disable ComfyUI Manager analytics and telemetry flags (`--disable-metadata` or stripped custom metadata if distributing outputs) so prompts and local file paths are not leaked.

---

## 6. Execution Agent Action Plan

The Antigravity execution agent should build the project in the following order:
1. **Scaffold Local Workspace (`~/ComfyFactory`):** Create directories (`datasets/`, `loras/`, `workflows/`, `outputs/`, `infra/`, `mcp-server/`).
2. **Build Portable Container & Bootstrap Scripts (`infra/`):** Write `Dockerfile`, `bootstrap_models.sh` (using `hf_transfer` for Wan 2.2 14B, FLUX, FLUX-Fill, UMT5, VAE, ControlNets), and `factory_up.sh` / `factory_down.sh` (supporting Vast.ai & RunPod APIs + SSH tunnel).
3. **Implement Local MCP Server (`mcp-server/`):** Build the Python MCP server exposing dataset upload, LoRA training trigger (`ai-toolkit` + `musubi-tuner`), workflow JSON manipulation, render queuing, **mid-clip frame repair & story branching**, and automatic download-and-purge cleanup; register it in Antigravity.
4. **Generate Master & Continuation Workflows (`workflows/`):** Output `wan22_master_pipeline.json` and `wan22_frame_branch_pipeline.json` with clean node IDs for automated agent control and visual browser editing.