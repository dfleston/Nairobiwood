#!/usr/bin/env python3
"""Ephemeral cloud GPU runner for LTX-2.3 + Alissonerdx/LTX-Best-Face-ID Phase 0 benchmarks.

Keeps production Wan 2.2 files untouched: uses FactorySettings, VastAIProvider,
RunPodProvider, SSHTunnelManager, and ComfyUIClient from mcp-server/src.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import shlex
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

PHASE0_DIR = Path(__file__).resolve().parent
REPO_ROOT = PHASE0_DIR.parents[1]
MCP_SRC = REPO_ROOT / "mcp-server" / "src"
if str(MCP_SRC) not in sys.path:
    sys.path.insert(0, str(MCP_SRC))

from comfy_factory_mcp.cloud_providers import (  # noqa: E402
    CloudProviderError,
    RunPodProvider,
    VastAIProvider,
)
from comfy_factory_mcp.comfy_client import ComfyUIClient  # noqa: E402
from comfy_factory_mcp.config import FactorySettings  # noqa: E402
from comfy_factory_mcp.ssh_tunnel import SSHTunnelManager  # noqa: E402

DEFAULT_IMAGE = "ghcr.io/dfleston/nairobiwood-ltx23:latest"
OUT_DIR = Path("~/ComfyFactory/outputs/ltx23_phase0").expanduser()
CHAR_BASE = Path("~/ComfyFactory/outputs/midnight_signal/characters").expanduser()
DEFAULT_D_LESTON_PORTRAIT = CHAR_BASE / "char_d_leston" / "master_portrait.png"
DEFAULT_ZURI_PORTRAIT = CHAR_BASE / "char_zuri_vance" / "master_portrait.png"


class LTX23VastProvider(VastAIProvider):
    """Vast.ai provider subclass configured for LTX-2.3 + Best-Face-ID Phase 0."""

    def __init__(self, *args: Any, disk_gb: int = 160, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.disk_gb = int(disk_gb)

    @staticmethod
    def _build_onstart_script() -> str:
        return (
            "mkdir -p /root/.ssh && chmod 700 /root/.ssh && "
            "if [ -n \"${PUBLIC_KEY:-}\" ]; then "
            "grep -qxF \"$PUBLIC_KEY\" /root/.ssh/authorized_keys 2>/dev/null || "
            "printf '%s\\n' \"$PUBLIC_KEY\" >> /root/.ssh/authorized_keys; "
            "chmod 600 /root/.ssh/authorized_keys; fi; "
            "env | grep -E '^(HF_|COMFYUI_|PATH=|VIRTUAL_ENV=)' >> /etc/environment || true; "
            "if [ -f /opt/bootstrap_ltx23.sh ] && "
            "! pgrep -f 'main\\.py --listen|bootstrap_ltx23\\.sh' >/dev/null 2>&1; then "
            "nohup bash -c 'PATH=/opt/venv/bin:/root/.local/bin:$PATH /opt/bootstrap_ltx23.sh "
            "> /tmp/bootstrap_ltx23.log 2>&1 && "
            "cd /opt/ComfyUI && exec /opt/venv/bin/python3 main.py --listen 127.0.0.1 "
            "--port 8188 --disable-metadata > /tmp/comfyui.log 2>&1' >/dev/null 2>&1 & fi"
        )

    async def provision_pod(
        self,
        *,
        gpu_id: str,
        image_name: str,
        ssh_public_key: str,
        hf_token: str = "",
        network_volume_id: str = "",
        datacenter_id: str = "",
        registry_auth_id: str = "",
        name: str = "nairobiwood-ltx23-phase0",
    ) -> dict[str, Any]:
        await self.ensure_ssh_key(ssh_public_key)
        env_map: dict[str, str] = {
            "PUBLIC_KEY": ssh_public_key.strip(),
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
        }
        if hf_token.strip():
            env_map["HF_TOKEN"] = hf_token.strip()

        payload: dict[str, Any] = {
            "client_id": "me",
            "image": image_name,
            "disk": self.disk_gb,
            "label": name,
            "runtype": "ssh_direc ssh_proxy",
            "cancel_unavail": True,
            "env": env_map,
            "onstart": self._build_onstart_script(),
        }
        if "ghcr.io" in image_name and self.ghcr_token:
            payload["image_login"] = f"-u {self.ghcr_username} -p {self.ghcr_token} ghcr.io"

        body = await self._request("PUT", f"/asks/{gpu_id}/", json_data=payload)
        if not isinstance(body, dict) or not body.get("success") or not body.get("new_contract"):
            raise CloudProviderError(f"Vast.ai instance creation failed for offer {gpu_id}: {body}")
        return {"id": str(body["new_contract"])}


def _read_ssh_public_key(settings: FactorySettings) -> str:
    priv = settings.resolved_ssh_key()
    pub = Path(f"{priv}.pub")
    if not pub.exists():
        raise FileNotFoundError(f"SSH public key not found: {pub}")
    return pub.read_text(encoding="utf-8").strip()


async def wait_for_ltx23_comfy(tunnel: SSHTunnelManager, timeout_sec: float = 1800.0) -> None:
    """Wait for SSH, model bootstrap (~47 GB), and ComfyUI 127.0.0.1:8188 readiness."""
    await tunnel.wait_for_ssh_ready(timeout_sec=360.0)
    print("[phase0] SSH is up; waiting for bootstrap + ComfyUI :8188...", flush=True)

    loop = asyncio.get_running_loop()
    start_t = loop.time()
    deadline = start_t + timeout_sec
    kicked = False
    last_log_tail = ""

    while loop.time() < deadline:
        stats = await tunnel.run_remote(
            "curl -s -m 3 http://127.0.0.1:8188/system_stats || echo not_ready"
        )
        if '"system"' in stats or '"devices"' in stats:
            print("[phase0] Remote ComfyUI :8188 is ready.", flush=True)
            return

        tail = await tunnel.run_remote(
            "tail -n 6 /tmp/bootstrap_ltx23.log /tmp/comfyui.log 2>/dev/null || true"
        )
        tail_clean = tail.strip()
        if tail_clean and tail_clean != last_log_tail:
            lines = [ln for ln in tail_clean.splitlines() if ln.strip()][-4:]
            print(f"[bootstrap +{int(loop.time() - start_t)}s] " + " | ".join(lines), flush=True)
            last_log_tail = tail_clean

        if loop.time() - start_t >= 12.0:
            proc_state = await tunnel.run_remote(
                "if pgrep -f 'bootstrap_ltx23\\.sh|ComfyUI/main\\.py|main\\.py --listen' >/dev/null 2>&1; then "
                "echo running; "
                "elif [ -f /tmp/bootstrap_ltx23.log ] || [ -f /tmp/comfyui.log ]; then "
                "echo failed; "
                "else echo unstarted; fi"
            )
            if "failed" in proc_state or ("unstarted" in proc_state and kicked):
                err_tail = await tunnel.run_remote(
                    "tail -n 50 /tmp/bootstrap_ltx23.log /tmp/comfyui.log 2>/dev/null || true"
                )
                raise RuntimeError(f"Bootstrap/ComfyUI exited unexpectedly:\n{err_tail}")
            if "unstarted" in proc_state and not kicked:
                kicked = True
                print("[phase0] Starting /opt/bootstrap_ltx23.sh + ComfyUI...", flush=True)
                await tunnel.run_remote(
                    "nohup bash -c 'PATH=/opt/venv/bin:/root/.local/bin:$PATH "
                    "/opt/bootstrap_ltx23.sh > /tmp/bootstrap_ltx23.log 2>&1 && "
                    "cd /opt/ComfyUI && exec /opt/venv/bin/python3 main.py --listen 127.0.0.1 "
                    "--port 8188 --disable-metadata > /tmp/comfyui.log 2>&1' >/dev/null 2>&1 &"
                )
        await asyncio.sleep(8.0)

    raise TimeoutError(f"Timed out after {timeout_sec:.0f}s waiting for ComfyUI startup.")


async def verify_workflow_nodes(base_url: str, workflows: dict[str, dict[str, Any]]) -> None:
    """Validate that every node class_type used by Phase 0 workflows exists on ComfyUI."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(f"{base_url}/object_info")
        resp.raise_for_status()
        obj_info = resp.json()

    needed = {
        node["class_type"]
        for wf in workflows.values()
        for node in wf.values()
        if isinstance(node, dict) and "class_type" in node
    }
    missing = sorted(cls for cls in needed if cls not in obj_info)
    if missing:
        raise RuntimeError(f"ComfyUI is missing required nodes: {missing}")
    print(f"[phase0] Verified {len(needed)} workflow node types in live /object_info.", flush=True)


def _extract_video_output(history: dict[str, Any], vhs_node_id: str) -> tuple[str, str]:
    """Extract (filename, subfolder) from VHS_VideoCombine history output."""
    outputs = history.get("outputs") or {}
    node_out = outputs.get(vhs_node_id) or {}
    candidates: list[dict[str, Any]] = []
    for key in ("gifs", "videos", "images"):
        items = node_out.get(key)
        if isinstance(items, list):
            candidates.extend(it for it in items if isinstance(it, dict) and it.get("filename"))

    mp4s = [c for c in candidates if str(c["filename"]).lower().endswith(".mp4")]
    audio_mp4s = [c for c in mp4s if "-audio" in str(c["filename"]).lower()]
    chosen = (audio_mp4s or mp4s or candidates)[0] if (audio_mp4s or mp4s or candidates) else None
    if not chosen:
        raise RuntimeError(f"No video output found in history for node {vhs_node_id}: {node_out}")
    return str(chosen["filename"]), str(chosen.get("subfolder") or "")


async def _download_comfy_output(
    base_url: str,
    tunnel: SSHTunnelManager,
    remote_filename: str,
    remote_subfolder: str,
    local_dest: Path,
) -> Path:
    """Download rendered output over SSH rsync or fallback /view HTTP tunnel."""
    local_dest.parent.mkdir(parents=True, exist_ok=True)
    rel = f"{remote_subfolder}/{remote_filename}" if remote_subfolder else remote_filename
    remote_path = f"/opt/ComfyUI/output/{rel}"
    if remote_path.startswith("/opt/ComfyUI/output/scratch/"):
        await tunnel.rsync_download(remote_path, local_dest)
        return local_dest

    params = {"filename": remote_filename, "subfolder": remote_subfolder, "type": "output"}
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.get(f"{base_url}/view", params=params)
        resp.raise_for_status()
        local_dest.write_bytes(resp.content)
    return local_dest


def _patch_1stage(
    template: dict[str, Any],
    *,
    ref_image: str,
    prompt: str,
    prefix: str,
    seed: int = 1042,
    width: int = 1280,
    height: int = 704,
    out_width: int = 1280,
    out_height: int = 720,
    length: int = 121,
    sigmas: str | None = None,
) -> dict[str, Any]:
    wf = copy.deepcopy(template)
    wf["7"]["inputs"]["image"] = ref_image
    wf["8"]["inputs"]["width"] = width
    wf["8"]["inputs"]["height"] = height
    wf["9"]["inputs"]["text"] = prompt
    wf["12"]["inputs"]["width"] = width
    wf["12"]["inputs"]["height"] = height
    wf["12"]["inputs"]["length"] = length
    wf["13"]["inputs"]["frames_number"] = length
    wf["16"]["inputs"]["noise_seed"] = seed
    if sigmas is not None:
        wf["18"]["inputs"]["sigmas"] = sigmas
    wf["25"]["inputs"]["width"] = out_width
    wf["25"]["inputs"]["height"] = out_height
    wf["26"]["inputs"]["filename_prefix"] = prefix
    return wf


def _patch_2stage(
    template: dict[str, Any],
    *,
    ref_image: str,
    prompt: str,
    prefix: str,
    seed_stage1: int = 1042,
    seed_stage2: int = 42,
) -> dict[str, Any]:
    wf = copy.deepcopy(template)
    wf["8"]["inputs"]["image"] = ref_image
    wf["11"]["inputs"]["text"] = prompt
    wf["18"]["inputs"]["noise_seed"] = seed_stage1
    wf["28"]["inputs"]["noise_seed"] = seed_stage2
    wf["38"]["inputs"]["filename_prefix"] = prefix
    return wf


async def run_benchmarks(args: argparse.Namespace) -> dict[str, Any]:
    settings = FactorySettings()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"ltx23_{uuid.uuid4().hex[:8]}"

    d_leston_ref = Path(args.d_leston_ref).expanduser().resolve()
    zuri_ref = Path(args.zuri_ref).expanduser().resolve()
    for p in (d_leston_ref, zuri_ref):
        if not p.exists():
            raise FileNotFoundError(f"Missing required reference portrait: {p}")

    wf_1stage = json.loads(
        (PHASE0_DIR / "workflows" / "ltx23_faceid_1stage_api.json").read_text(encoding="utf-8")
    )
    wf_2stage = json.loads(
        (PHASE0_DIR / "workflows" / "ltx23_faceid_2stage_upscale_api.json").read_text(encoding="utf-8")
    )

    provider: VastAIProvider | RunPodProvider | None = None
    provider_name = ""
    pod_id = args.pod_id or ""
    ssh_host = args.ssh_host or ""
    ssh_port = int(args.ssh_port or 0)
    price_hr = 0.0
    gpu_name = "reused-pod"
    tunnel: SSHTunnelManager | None = None
    tunnel_pid: int | None = None

    try:
        if not (pod_id and ssh_host and ssh_port):
            ssh_pub = _read_ssh_public_key(settings)
            vast_key = settings.resolved_vast_api_key()
            runpod_key = settings.runpod_api_key.strip()

            order = ["vastai", "runpod"] if args.provider == "auto" else [args.provider]
            last_err: Exception | None = None
            for cand in order:
                try:
                    if cand == "vastai" and vast_key:
                        prov = LTX23VastProvider(
                            vast_key,
                            ghcr_username=settings.ghcr_username,
                            ghcr_token=settings.ghcr_token,
                            disk_gb=args.disk_gb,
                        )
                        offer = await prov.find_best_gpu(
                            gpu_filter=args.gpu_filter,
                            max_price_hr=args.max_price,
                            min_vram_gb=48,
                            disk_gb=args.disk_gb,
                        )
                        price_hr = float(offer["effective_price"])
                        gpu_name = str(offer["displayName"])
                        print(
                            f"[phase0] Provisioning Vast.ai offer {offer['id']} ({gpu_name}, "
                            f"${price_hr:.4f}/hr) with {args.image}...",
                            flush=True,
                        )
                        pod = await prov.provision_pod(
                            gpu_id=offer["id"],
                            image_name=args.image,
                            ssh_public_key=ssh_pub,
                            hf_token=settings.hf_token,
                            name=f"nw-{run_id}",
                        )
                        pod_id = str(pod["id"])
                        provider = prov
                        provider_name = "vastai"
                        ssh_host, ssh_port = await prov.wait_for_ssh_endpoint(pod_id, timeout_sec=420.0)
                        break
                    if cand == "runpod" and runpod_key:
                        prov_rp = RunPodProvider(runpod_key)
                        gpu = await prov_rp.find_best_gpu(
                            gpu_filter=args.gpu_filter,
                            max_price_hr=max(args.max_price, 1.69),
                            min_vram_gb=48,
                        )
                        price_hr = float(gpu["effective_price"])
                        gpu_name = str(gpu["displayName"])
                        print(
                            f"[phase0] Provisioning Runpod GPU {gpu['id']} ({gpu_name}, "
                            f"${price_hr:.4f}/hr) with {args.image}...",
                            flush=True,
                        )
                        pod = await prov_rp.provision_pod(
                            gpu_id=gpu["id"],
                            image_name=args.image,
                            ssh_public_key=ssh_pub,
                            hf_token=settings.hf_token,
                            registry_auth_id=settings.runpod_registry_auth_id,
                            name=f"nw-{run_id}",
                        )
                        pod_id = str(pod["id"])
                        provider = prov_rp
                        provider_name = "runpod"
                        ssh_host, ssh_port = await prov_rp.wait_for_ssh_endpoint(pod_id, timeout_sec=600.0)
                        break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    print(f"[phase0] Provider {cand} unavailable: {exc}", flush=True)
                    if pod_id and provider is not None:
                        await provider.terminate_pod(pod_id)
                        pod_id = ""
            if not pod_id or not ssh_host:
                raise RuntimeError(f"Failed to provision GPU pod: {last_err}")

        tunnel = SSHTunnelManager(
            ssh_host=ssh_host,
            ssh_port=ssh_port,
            ssh_key_path=settings.resolved_ssh_key(),
            local_port=settings.comfy_local_port,
        )
        await wait_for_ltx23_comfy(tunnel, timeout_sec=1800.0)
        tunnel_pid = tunnel.start_tunnel()
        base_url = f"http://127.0.0.1:{settings.comfy_local_port}"
        stats = await tunnel.wait_for_comfyui(timeout_sec=60.0)
        print(f"[phase0] Tunnel active (PID {tunnel_pid}); ComfyUI devices={stats.get('devices')}", flush=True)

        await verify_workflow_nodes(base_url, {"1stage": wf_1stage, "2stage": wf_2stage})

        sub = f"scratch/{run_id}"
        remote_d_leston = await comfy_upload(base_url, d_leston_ref, sub)
        remote_zuri = await comfy_upload(base_url, zuri_ref, sub)
        comfy = ComfyUIClient(PHASE0_DIR / "workflows", base_url=base_url)

        prompt_d_leston = (
            "ref_t2v: A Hispanic Mediterranean man in his late 30s with loose dark wavy hair "
            "with silver strands, salt-and-pepper beard, intense dark eyes, wearing a charcoal wool "
            "trench coat over a matte black turtleneck with matte-black studio headphones around his neck, "
            "in a tight cinematic 35mm anamorphic close-up on a rain-blurred night overpass. He holds a "
            "vintage brass field transceiver near his chin, turns the knurled brass dial as warm amber "
            "vacuum-tube light pulses onto his cheekbones, narrows his eyes with recognition, and speaks "
            'clearly in quiet intense Spanish: "Ahí está. Pulso analógico puro a trescientos metros." '
            "Warm analog vacuum-tube hum and soft night rain."
        )
        prompt_zuri = (
            "ref_t2v: An East African woman in her early 30s with a sculpted high braided updo woven "
            "with subtle copper thread, high cheekbones, and calm piercing dark eyes, wearing a tailored "
            "ochre-gold leather field jacket over an ivory silk shirt, medium close-up inside a modernist "
            "rain-streaked glass transit atrium at night. Slow poised camera push-in as she turns her head "
            "smoothly toward the atrium entrance while holding a copper-bound ledger, speaking in calm "
            'poised English: "You followed the carrier wave all the way through the storm." '
            "Soft glass atrium reverb and distant rain."
        )

        results: dict[str, Any] = {
            "run_id": run_id,
            "provider": provider_name,
            "pod_id": pod_id,
            "gpu_name": gpu_name,
            "price_hr": price_hr,
            "image": args.image,
            "benchmarks": {},
        }

        # Warmup pass (loads 22B FP8 + Gemma-3 12B once cleanly before timing steady-state shots)
        print("[phase0] Running 9-frame warmup pass to measure cold weight load time...", flush=True)
        warmup_wf = _patch_1stage(
            wf_1stage,
            ref_image=remote_d_leston,
            prompt=prompt_d_leston,
            prefix=f"scratch/{run_id}/warmup",
            seed=99,
            width=256,
            height=256,
            out_width=256,
            out_height=256,
            length=9,
            sigmas="1.0, 0.725, 0.421875, 0.0",
        )
        t0 = time.monotonic()
        await asyncio.wait_for(comfy.queue_and_wait(warmup_wf), timeout=900.0)
        results["warmup_cold_load_sec"] = round(time.monotonic() - t0, 2)
        print(f"[phase0] Warmup complete in {results['warmup_cold_load_sec']}s", flush=True)

        tests = [
            (
                "test1_s01_sh02_d_leston_1stage",
                _patch_1stage(
                    wf_1stage,
                    ref_image=remote_d_leston,
                    prompt=prompt_d_leston,
                    prefix=f"scratch/{run_id}/test1_d_leston_1stage",
                    seed=1042,
                ),
                "26",
                OUT_DIR / "test1_s01_sh02_d_leston_1stage.mp4",
            ),
            (
                "test2_s02_sh01_zuri_1stage",
                _patch_1stage(
                    wf_1stage,
                    ref_image=remote_zuri,
                    prompt=prompt_zuri,
                    prefix=f"scratch/{run_id}/test2_zuri_1stage",
                    seed=2043,
                ),
                "26",
                OUT_DIR / "test2_s02_sh01_zuri_1stage.mp4",
            ),
            (
                "test3_s01_sh02_d_leston_2stage",
                _patch_2stage(
                    wf_2stage,
                    ref_image=remote_d_leston,
                    prompt=prompt_d_leston,
                    prefix=f"scratch/{run_id}/test3_d_leston_2stage",
                    seed_stage1=1042,
                    seed_stage2=42,
                ),
                "38",
                OUT_DIR / "test3_s01_sh02_d_leston_2stage.mp4",
            ),
        ]

        for name, wf_payload, vhs_id, local_mp4 in tests:
            print(f"[phase0] Executing {name}...", flush=True)
            t_start = time.monotonic()
            hist = await asyncio.wait_for(comfy.queue_and_wait(wf_payload), timeout=900.0)
            gpu_sec = round(time.monotonic() - t_start, 2)
            r_file, r_sub = _extract_video_output(hist, vhs_id)
            t_dl0 = time.monotonic()
            await _download_comfy_output(base_url, tunnel, r_file, r_sub, local_mp4)
            dl_sec = round(time.monotonic() - t_dl0, 2)
            wall_sec = round(gpu_sec + dl_sec, 2)
            est_cost = round((gpu_sec / 3600.0) * price_hr, 4) if price_hr else 0.0
            results["benchmarks"][name] = {
                "gpu_sec": gpu_sec,
                "download_sec": dl_sec,
                "wall_sec": wall_sec,
                "est_gpu_cost_usd": est_cost,
                "local_mp4": str(local_mp4),
                "size_mb": round(local_mp4.stat().st_size / 1e6, 2),
            }
            print(
                f"[phase0] {name} done: gpu={gpu_sec}s wall={wall_sec}s cost=${est_cost:.4f} -> {local_mp4}",
                flush=True,
            )

        out_json = OUT_DIR / "benchmark_results.json"
        out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"[phase0] Saved benchmark report -> {out_json}", flush=True)
        return results
    finally:
        if tunnel is not None:
            for remote_dir in (
                f"/opt/ComfyUI/output/scratch/{run_id}/",
                f"/opt/ComfyUI/input/scratch/{run_id}/",
                "/dev/shm/factory_scratch/active_loras/",
                "/dev/shm/factory_scratch/temp_renders/",
            ):
                try:
                    await tunnel.purge_remote(remote_dir)
                except Exception as exc:  # noqa: BLE001
                    print(f"[phase0] WARN scratch purge {remote_dir}: {exc}", flush=True)
            tunnel.stop_tunnel(tunnel_pid)
        if pod_id and provider is not None and not args.keep_warm:
            print(f"[phase0] Tearing down {provider_name} pod {pod_id}...", flush=True)
            await provider.terminate_pod(pod_id)
            print(f"[phase0] Pod {pod_id} terminated.", flush=True)


async def comfy_upload(base_url: str, local_path: Path, subfolder: str) -> str:
    comfy = ComfyUIClient(PHASE0_DIR / "workflows", base_url=base_url)
    return await comfy.upload_image(local_path, subfolder=subfolder, overwrite=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run isolated LTX-2.3 + Best-Face-ID Phase 0 benchmarks.")
    p.add_argument("--provider", choices=("auto", "vastai", "runpod"), default="auto")
    p.add_argument("--image", default=DEFAULT_IMAGE)
    p.add_argument("--gpu-filter", default="L40S|RTX 6000Ada|RTX 6000 Ada|PRO 6000")
    p.add_argument("--max-price", type=float, default=1.49)
    p.add_argument("--disk-gb", type=int, default=160)
    p.add_argument("--d-leston-ref", default=str(DEFAULT_D_LESTON_PORTRAIT))
    p.add_argument("--zuri-ref", default=str(DEFAULT_ZURI_PORTRAIT))
    p.add_argument("--pod-id", default="")
    p.add_argument("--ssh-host", default="")
    p.add_argument("--ssh-port", type=int, default=0)
    p.add_argument("--keep-warm", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run_benchmarks(parse_args()))
