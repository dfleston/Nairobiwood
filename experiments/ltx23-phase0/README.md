# LTX-2.3 + `Alissonerdx/LTX-Best-Face-ID` Phase 0 Validation

Isolated test harness to benchmark **LTX-2.3 (22B FP8)** + **`Best_FaceID_v1.0_LoRA`** (`Alissonerdx/LTX-Best-Face-ID` + `ComfyUI-BFSNodes`) against the locked-in **Wan 2.2 (4-step LightX2V) + `edge-tts` + `LatentSync 1.6`** *Midnight Signal* baseline (`329.3s + 63.1s = 6.54 min` GPU, `~$0.087`/shot on Vast.ai L40S).

Zero changes are made to production `infra/`, `mcp-server/`, or `workflows/`.

## Contents

- [Dockerfile](./Dockerfile) — CUDA 12.4.1 + `uv` Python 3.11 venv + PyTorch 2.6.0 + latest ComfyUI + `ComfyUI-LTXVideo` + `ComfyUI-BFSNodes` + `ComfyUI-KJNodes` + `ComfyUI-VideoHelperSuite`. Published via [.github/workflows/docker-publish-ltx23.yml](../../.github/workflows/docker-publish-ltx23.yml) as `ghcr.io/dfleston/nairobiwood-ltx23:latest`.
- [ltx23_models_manifest.json](./ltx23_models_manifest.json) — Un-gated HF weights (`~47.03 GB` total) pulled at runtime onto ephemeral cloud disk.
- [bootstrap_ltx23.sh](./bootstrap_ltx23.sh) & [entrypoint.sh](./entrypoint.sh) — Bootstraps weights, sets physical `/opt/ComfyUI/{input,output}/scratch`, binds ComfyUI strictly to `127.0.0.1:8188 --disable-metadata`.
- [workflows/ltx23_faceid_1stage_api.json](./workflows/ltx23_faceid_1stage_api.json) — Direct 8-step `1280x704 -> 1280x720` (`121` frames @ `24 fps`) Ref-to-Video + joint audio graph.
- [workflows/ltx23_faceid_2stage_upscale_api.json](./workflows/ltx23_faceid_2stage_upscale_api.json) — 2-stage spatial upscale (`640x352` 8 steps -> `ltx-2.3-spatial-upscaler-x2-1.1` -> `1280x704 -> 1280x720` 3 steps).
- [run_phase0.py](./run_phase0.py) — Provisions an ephemeral L40S, verifies live nodes, runs Warmup + Tests 1–3, pulls outputs to `~/ComfyFactory/outputs/ltx23_phase0/`, shreds remote scratch, and terminates the GPU in `finally:`.

## Quick Run

```bash
/Users/leston/code/Nairobiwood/mcp-server/.venv/bin/python \
  experiments/ltx23-phase0/run_phase0.py
```
