# Utility Specification: Interactive Frame Scrub, Repair & Continuation ("Pause-and-Pivot")

**Module Name:** `frame-branch-director`  
**Scope:** Local UX (macOS + Antigravity), Local MCP State Manager (`comfy-factory-mcp`), and Cloud ComfyUI Split-Graph Execution (`wan22_frame_branch_pipeline.json`)

---

## 1. Purpose & Problem Statement
In standard text-to-video or image-to-video generation, a 5-to-10 second clip is rendered as an atomic, all-or-nothing block. If an anatomical glitch (e.g., hand clipping an object), lighting error, or flat facial expression appears at **Second 2.5 (Frame 60)**, traditional pipelines force the creator to discard the entire clip—including an otherwise flawless opening—and gamble on a full re-render with a new random seed.

The **Interactive Frame Scrub, Repair & Continuation ("Pause-and-Pivot")** utility transforms video generation from a linear slot machine into a **non-destructive, timeline-aware directing tool**. It enables the creator or AI agent to:
1. **Scrub and pause** on any exact frame of a previewed `.mp4`.
2. **Surgically repair** that specific frame using LoRA-locked inpainting (preserving character identity, lighting, and costume).
3. **Branch forward** from that repaired frame—either **carrying** the visual fix seamlessly into the original action, **pivoting** the story/camera movement into an alternate timeline, or applying **fast localized region/lip-sync edits** without recalculating or paying GPU compute for prior frames.

---

## 2. User Experience (UX) Flow in Antigravity

### Step 1: Preview & Timeline Scrub
* A draft clip (e.g., `shot_01_v1.mp4`, 81 frames at 24 fps) finishes rendering and downloads automatically to `~/ComfyFactory/outputs/<project>/`.
* The creator plays the video inside Antigravity's preview pane, scrubs the timeline, and stops at an exact frame—for example, **Frame 45 (`00:01.87`)**, where the character's right hand distorts while picking up a glass, or where a line of spoken dialogue begins.

### Step 2: Frame-Level Directing (Mask + Prompt or Audio)
* The creator clicks **"Pause & Edit at Frame 45"** (or invokes the Antigravity agent directly in chat):
  > *"Stop at Frame 45. Fix the right hand gripping the glass, and from here onward pivot her expression from calm to suspicious as she looks toward the door."*
* **Optional Masking:** The creator can either brush a quick mask over the flawed region directly on the paused frame or supply a text target (e.g., `"right hand and glass"` or `"mouth and jaw"`) for automated segmentation via **SAM2**.

### Step 3: 2-Second Keyframe Confirmation (Human-in-the-Loop Gate)
* Before spending GPU compute on multi-second video diffusion, the utility runs a rapid (~2-second) **LoRA-locked image inpaint** on `Frame 45` alone (skipped if running direct audio lip-sync).
* `frame_0045_repaired.png` appears in Antigravity alongside the original `frame_0045_raw.png` (and `frame_0044.png` for seam comparison).
* Once confirmed (or auto-approved in agent autonomous mode), the utility proceeds to video continuation.

### Step 4: Continuation & Branch Tree Management
* Frames `0–44` (`Batch_A`) are **cryptographically/hash-locked** locally on macOS.
* Only Frames `45–81` (`Batch_B`) are synthesized on the cloud GPU, anchored strictly to `frame_0045_repaired.png`.
* The stitched clip saves as a versioned branch (`shot_01_v1_f45_branchA.mp4`), leaving `shot_01_v1.mp4` and the `Frame 45` checkpoint intact so the creator can test multiple alternate endings (`branchB`, `branchC`) from the exact same midpoint.

---

## 3. Shared Paradigm: Single-Frame Repair vs. Multi-Frame Localized Inpainting (Lip-Sync)

Both **Pause-and-Pivot Frame Repair** and **Post-Production Voice / Lip-Sync** operate on the exact same **region-masked temporal inpainting** architecture and share the **SAM2 spatial isolation sub-graph** inside `wan22_frame_branch_pipeline.json`:

| Dimension | Single-Frame Repair + Wan 2.2 Continuation (`carry` / `pivot`) | Multi-Frame Localized Inpainting (`lipsync` / `region_patch`) |
| :--- | :--- | :--- |
| **Shared Masking Engine** | **SAM2** isolates a small target area (hand, prop, facial expression) while locking >90% of background/body pixels. | **SAM2 / Face Landmarks** isolates the mouth/jaw region (or small static prop) across a continuous frame window (`K..N`). |
| **Diffusion Scope** | Denoises **1 pivot frame (`Frame_K`)** via `FLUX-Fill + Character LoRA`, then lets **Wan 2.2** propagate new motion forward. | Denoises **only the masked region across `Frames K..N`** via `LatentSync` / `MuseTalk` (driven by `.wav` audio), keeping existing body/camera motion 100% intact. |
| **When to Use** | Changing pose, fixing distorted limbs, altering lighting, or pivoting narrative trajectory from Frame `K` onward. | Adding spoken dialogue/voice from Frame `K` onward or patching a localized defect without altering body movement. |
| **GPU Compute Cost** | Medium (renders `Batch_B` via Wan 2.2). | **Ultra-Low / Fast** (bypasses Wan 2.2 completely; runs in seconds). |

---

## 4. Technical Architecture: Split-Graph ComfyUI Mechanics

Because ComfyUI cannot natively pause mid-denoising inside a single `KSampler` pass, the MCP server orchestrates a **4-stage tensor split-and-stitch graph** (`wan22_frame_branch_pipeline.json`):

```
[Local: shot_01_v1.mp4] ──(Encrypted Rsync to Cloud tmpfs)──► [VHS_LoadVideo]
                                                                    │
                     ┌──────────────────────────────────────────────┴──────────────────────────────────────────────┐
                     ▼ (Frames 0 .. K-1)                                                                           ▼ (Frames K .. N)
        [ImageFromBatch: Batch_A]                                                                   [SAM2 Spatial Isolation]
        (LOCKED - Zero Recompute)                                                                   (Reused for Repair & Lip-Sync)
                     │                                                                                             │
                     │                                                                    ┌────────────────────────┴────────────────────────┐
                     │                                                                    ▼ (Mode: carry / pivot)                           ▼ (Mode: lipsync / region_patch)
                     │                                                       [FLUX-Fill + Char LoRA on Frame_K]               [LatentSync / MuseTalk + Audio .wav]
                     │                                                                    │                                   (Bypasses Wan 2.2; keeps body motion)
                     │                                                                    ▼                                                 │
                     │                                                       [Frame_K_Repaired.png] (UX Gate)                               │
                     │                                                                    │                                                 │
                     │                                                                    ▼                                                 │
                     │                                                       [WanImageToVideo / WanFirstLastFrame]                          │
                     │                                                       [Wan 2.2 14B + Char LoRA]                                     │
                     │                                                                    │                                                 │
                     │                                                                    ▼                                                 │
                     │                                                       [ImageFromBatch: Batch_B]                                     │
                     │                                                       (Drop index 0 seam duplicate)                                  │
                     │                                                                    │                                                 │
                     └──────────────────────────────────────► [ImageBatch Concat] ◄───────┴─────────────────────────────────────────────────┘
                                                                        │
                                                                        ▼
                                                             [RIFE VFI / MMAudio]
                                                                        │
                                                                        ▼
                                                             [VHS_VideoCombine (.mp4)] ──► Pull to macOS & Purge Remote
```

### Stage-by-Stage Node Breakdown

1. **Video Tensor Ingestion & Slicing (`VHS_LoadVideo` + `ImageFromBatch`):**
   * Loads the source video into GPU RAM as an image tensor batch `[N, H, W, C]`.
   * Branch 1 slices `start_index = 0`, `length = K` (`Batch_A`, frames `0` through `K-1`).
   * Branch 2 extracts `start_index = K`, `length = 1` (`Frame_K`) for keyframe repair, or `start_index = K`, `length = (N - K)` for direct in-place lip-sync/region patching.

2. **Shared Spatial Isolation & LoRA-Locked Inpainting (`SAM2` + `FLUX.1-Fill-dev` + `Character_FLUX_LoRA`):**
   * **Mask Generation:** If no manual mask PNG is provided, `SAM2Segment` generates a binary mask from `mask_target_text` (dilated by `8–16px` with Gaussian feathering to prevent hard seams). This exact same segmentation node serves both keyframe repair and mouth/jaw isolation for lip-sync.
   * **Identity-Preserving Repair:** `FLUX.1-Fill-dev` combined with the character's `.safetensors` FLUX LoRA (`strength: 0.85`) denoises **only** the masked area (`denoise: 0.55–0.75`). Because unmasked pixels are preserved at 100% fidelity, `Frame_K_Repaired` matches `Frame_(K-1)` in background geometry, film grain, and ambient lighting.

3. **Three-Mode Forward Continuation (`Wan 2.2 14B` or `LatentSync` Bypass):**
   * **Mode A — `"carry"` (Defect Propagation / Continuity):**
     * Encodes `Frame_K_Repaired` into `start_image` of `WanImageToVideo` while retaining the original text prompt, negative prompt, and `ModelSamplingSD3` shift (`6.0`). Wan 2.2 carries the fixed hand, prop, or lighting forward naturally through the rest of the clip.
   * **Mode B — `"pivot"` (Narrative / Emotional Redirection):**
     * Replaces the positive prompt from frame `K` onward (e.g., changing emotion, action beat, or camera movement), optionally injects a new `end_image` keyframe (generated via FLUX + Character LoRA) into `WanFirstLastFrameToVideo`, or attaches a new `DWPose` driving video into `Wan VACE` to redirect body choreography mid-shot.
   * **Mode C — `"lipsync"` (Fast In-Place Dialogue / Voice Synchronization):**
     * When the existing body motion and camera trajectory from `Frame K` onward are already approved and the user only wants the character to speak a `.wav` audio track starting at `Frame K`, the MCP **bypasses Wan 2.2 completely**. It routes `Frames K..N` + `.wav` audio directly through `LatentSync` / `MuseTalk` (masked strictly to the lower face/jaw), rendering synced lip movements in seconds while preserving original motion pixel-for-pixel.

4. **Seam-Free Temporal Stitching (`ImageFromBatch` + `ImageBatch` + `VHS_VideoCombine`):**
   * For Wan 2.2 continuations (`carry`/`pivot`), `Batch_B[1:]` is appended directly onto `Batch_A` (dropping the duplicate `Frame_K` seam frame, with optional 1-frame `alpha = 0.5` cross-blend).
   * For `lipsync` mode, the lip-synced `Frames K..N` slice replaces the silent tail frames directly onto `Batch_A` and muxes the `.wav` dialogue track with ambient `MMAudio` foley in `VHS_VideoCombine`.

---

## 5. Local State & Branch Tree Persistence (macOS)

To enable rapid experimentation without re-uploading or re-slicing source videos repeatedly, `comfy-factory-mcp` maintains a lightweight JSON manifest alongside your outputs on macOS:

```text
~/ComfyFactory/outputs/<project_name>/
├── shot_01_v1.mp4                        # Original root render (81 frames)
├── shot_01_v1.manifest.json              # Tracks prompts, seeds, LoRAs, and branch lineage
└── branches/
    └── f0045_checkpoint/
        ├── locked_prefix_0000_0044.mp4   # Cached lossless prefix (Batch_A)
        ├── frame_0045_raw.png            # Original extracted frame
        ├── frame_0045_repaired.png       # Approved FLUX+LoRA repaired frame
        ├── shot_01_v1_f45_carry.mp4      # Branch 1: Carry fix with original action
        ├── shot_01_v1_f45_pivot_door.mp4 # Branch 2: Pivot expression & look toward door
        └── shot_01_v1_f45_lipsync.mp4    # Branch 3: Fast in-place voice/lip-sync from frame 45
```

---

## 6. MCP Tool Contracts (`comfy-factory-mcp`)

### Tool 1: `extract_and_repair_frame`
Extracts a specific frame from a local clip, caches the locked prefix, performs LoRA-locked inpainting on the target frame, and returns the preview PNG to Antigravity.

```json
{
  "name": "extract_and_repair_frame",
  "parameters": {
    "source_video_path": "string (required) - Local path to .mp4 in ~/ComfyFactory/outputs/",
    "frame_index": "integer (required) - 0-based frame index to pause and edit",
    "character_name": "string (required) - Character identifier to load matching FLUX LoRA",
    "repair_prompt": "string (required) - Description of the fix on this specific frame",
    "mask_target_text": "string (optional) - Object/region for SAM2 auto-masking (e.g., 'right hand')",
    "mask_image_path": "string (optional) - Local path to user-brushed binary mask PNG",
    "inpaint_denoise": "number (default: 0.65) - Denoising strength for FLUX-Fill"
  },
  "returns": {
    "checkpoint_dir": "string - Local path to cached prefix & frame artifacts",
    "raw_frame_path": "string - Local path to frame_XXXX_raw.png",
    "repaired_frame_path": "string - Local path to frame_XXXX_repaired.png"
  }
}
```

### Tool 2: `continue_from_repaired_frame`
Generates `Batch_B` forward from an approved `repaired_frame_path` (via Wan 2.2 `carry`/`pivot`) OR applies fast in-place `lipsync` from `frame_index` onward, stitches it onto `locked_prefix`, downloads the final `.mp4`, and purges remote GPU scratch files.

```json
{
  "name": "continue_from_repaired_frame",
  "parameters": {
    "checkpoint_dir": "string (required) - Directory returned by extract_and_repair_frame (or raw slice checkpoint)",
    "character_name": "string (required) - Character identifier to load matching Wan 2.2 LoRA",
    "mode": "string (enum: ['carry', 'pivot', 'lipsync'], required) - Carry original motion, pivot story, or apply fast in-place lip-sync",
    "continuation_prompt": "string (optional) - Required if mode='pivot'; overrides prompt from frame_index onward",
    "audio_wav_path": "string (optional) - Required if mode='lipsync'; local .wav dialogue track starting at frame_index",
    "end_keyframe_prompt": "string (optional) - Generates a target end frame via FLUX+LoRA to lock the final pose",
    "driving_video_path": "string (optional) - Local pose/depth reference video for the pivoted segment",
    "target_total_frames": "integer (optional) - Defaults to original video length; can extend or shorten clip",
    "lora_strength": "number (default: 0.85)",
    "seed": "integer (optional)"
  },
  "returns": {
    "branched_video_path": "string - Local path to stitched .mp4 on macOS",
    "branch_id": "string - Unique identifier logged in project manifest"
  }
}
```