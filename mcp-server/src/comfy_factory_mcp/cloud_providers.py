"""Cloud GPU provider protocol and RunPod Secure Cloud GraphQL implementation."""

from __future__ import annotations

import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

import httpx

from comfy_factory_mcp.config import FactorySettings
from comfy_factory_mcp.models import ActiveFactoryState

RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql"


class CloudProviderError(RuntimeError):
    """Raised when cloud GPU provisioning or API queries fail."""


class CloudProvider(Protocol):
    """Provider-agnostic interface so Vast.ai or other providers can be plugged in cleanly."""

    async def find_best_gpu(
        self, gpu_filter: str, max_price_hr: float, min_vram_gb: int = 48
    ) -> dict[str, Any]: ...

    async def provision_pod(
        self,
        *,
        gpu_id: str,
        image_name: str,
        ssh_public_key: str,
        hf_token: str = "",
        network_volume_id: str = "",
        datacenter_id: str = "",
        name: str = "nairobiwood-factory",
    ) -> dict[str, Any]: ...

    async def wait_for_ssh_endpoint(
        self, pod_id: str, timeout_sec: float = 300.0, poll_interval: float = 3.0
    ) -> tuple[str, int]: ...

    async def terminate_pod(self, pod_id: str) -> None: ...


class RunPodProvider:
    """RunPod Secure Cloud GraphQL adapter (strictly SSH 22/tcp, never exposing 8188)."""

    def __init__(
        self,
        api_key: str,
        *,
        graphql_url: str = RUNPOD_GRAPHQL_URL,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise CloudProviderError("RUNPOD_API_KEY is required.")
        self.api_key = api_key.strip()
        self.graphql_url = graphql_url
        self._external_client = client

    async def _graphql(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"query": query, "variables": variables or {}}

        if self._external_client is not None:
            resp = await self._external_client.post(self.graphql_url, json=payload, headers=headers)
        else:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(self.graphql_url, json=payload, headers=headers)

        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            raise CloudProviderError(f"RunPod GraphQL error: {body['errors']}")
        return body.get("data", {})

    async def find_best_gpu(
        self,
        gpu_filter: str = "L40S|RTX 6000 Ada|A100",
        max_price_hr: float = 1.69,
        min_vram_gb: int = 48,
    ) -> dict[str, Any]:
        """Select the cheapest RunPod Secure Cloud GPU matching VRAM, regex filter, and budget."""
        query = """
        query GpuTypes {
          gpuTypes {
            id
            displayName
            memoryInGb
            secureCloud
            securePrice
            lowestPrice(input: {gpuCount: 1}) {
              minimumBidPrice
              uninterruptablePrice
            }
          }
        }
        """
        data = await self._graphql(query)
        gpu_types = data.get("gpuTypes", [])
        pattern = re.compile(gpu_filter, re.IGNORECASE)

        candidates: list[dict[str, Any]] = []
        for gpu in gpu_types:
            if not gpu.get("secureCloud"):
                continue
            vram = int(gpu.get("memoryInGb") or 0)
            if vram < min_vram_gb:
                continue
            name = f"{gpu.get('id', '')} {gpu.get('displayName', '')}"
            if not pattern.search(name):
                continue
            price = gpu.get("securePrice")
            if price is None:
                price = (gpu.get("lowestPrice") or {}).get("uninterruptablePrice")
            if price is None:
                continue
            price_f = float(price)
            if price_f <= max_price_hr:
                candidates.append({**gpu, "effective_price": price_f})

        if not candidates:
            raise CloudProviderError(
                f"No RunPod Secure Cloud GPU matched filter='{gpu_filter}', "
                f"vram>={min_vram_gb}GB, max_price_hr<=${max_price_hr:.2f}."
            )
        candidates.sort(key=lambda item: (item["effective_price"], -int(item["memoryInGb"])))
        return candidates[0]

    async def provision_pod(
        self,
        *,
        gpu_id: str,
        image_name: str,
        ssh_public_key: str,
        hf_token: str = "",
        network_volume_id: str = "",
        datacenter_id: str = "",
        name: str = "nairobiwood-factory",
    ) -> dict[str, Any]:
        """Deploy an on-demand RunPod Secure Cloud pod exposing ONLY 22/tcp."""
        env_vars = [{"key": "PUBLIC_KEY", "value": ssh_public_key.strip()}]
        if hf_token.strip():
            env_vars.append({"key": "HF_TOKEN", "value": hf_token.strip()})

        pod_input: dict[str, Any] = {
            "cloudType": "SECURE",
            "gpuCount": 1,
            "volumeInGb": 0 if network_volume_id.strip() else 120,
            "containerDiskInGb": 40,
            "minVcpuCount": 8,
            "minMemoryInGb": 48,
            "gpuTypeId": gpu_id,
            "name": name,
            "imageName": image_name,
            "ports": "22/tcp",
            "volumeMountPath": "/workspace",
            "startSsh": True,
            "env": env_vars,
        }
        if network_volume_id.strip():
            pod_input["networkVolumeId"] = network_volume_id.strip()
        if datacenter_id.strip():
            pod_input["dataCenterId"] = datacenter_id.strip()

        mutation = """
        mutation DeployPod($input: PodFindAndDeployOnDemandInput!) {
          podFindAndDeployOnDemand(input: $input) {
            id
            name
            costPerHr
            desiredStatus
            machine {
              podHostId
            }
          }
        }
        """
        data = await self._graphql(mutation, {"input": pod_input})
        pod = data.get("podFindAndDeployOnDemand")
        if not pod or not pod.get("id"):
            raise CloudProviderError("RunPod podFindAndDeployOnDemand returned no pod id.")
        return pod

    async def wait_for_ssh_endpoint(
        self, pod_id: str, timeout_sec: float = 300.0, poll_interval: float = 3.0
    ) -> tuple[str, int]:
        """Poll RunPod until public IP + mapped port for privatePort 22 are ready."""
        query = """
        query PodStatus($input: PodFilter!) {
          pod(input: $input) {
            id
            desiredStatus
            runtime {
              uptimeInSeconds
              ports {
                ip
                isIpPublic
                privatePort
                publicPort
                type
              }
            }
          }
        }
        """
        deadline = asyncio.get_running_loop().time() + timeout_sec
        while asyncio.get_running_loop().time() < deadline:
            data = await self._graphql(query, {"input": {"podId": pod_id}})
            pod = data.get("pod") or {}
            runtime = pod.get("runtime") or {}
            for port_info in runtime.get("ports") or []:
                if (
                    int(port_info.get("privatePort") or 0) == 22
                    and port_info.get("isIpPublic")
                    and port_info.get("ip")
                    and port_info.get("publicPort")
                ):
                    return str(port_info["ip"]), int(port_info["publicPort"])
            await asyncio.sleep(poll_interval)
        raise CloudProviderError(f"Timed out waiting for RunPod SSH endpoint on pod {pod_id}.")

    async def terminate_pod(self, pod_id: str) -> None:
        """Terminate an active RunPod pod immediately to stop hourly billing."""
        mutation = """
        mutation TerminatePod($input: PodTerminateInput!) {
          podTerminate(input: $input)
        }
        """
        await self._graphql(mutation, {"input": {"podId": pod_id}})


def save_active_factory_state(state: ActiveFactoryState, settings: FactorySettings) -> Path:
    """Persist active RunPod state atomically with 0600 file permissions."""
    path = settings.active_factory_path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(state.model_dump_json(indent=2), encoding="utf-8")
    os.chmod(tmp_path, 0o600)
    tmp_path.replace(path)
    os.chmod(path, 0o600)
    return path


def load_active_factory_state(settings: FactorySettings) -> ActiveFactoryState | None:
    """Load persisted RunPod state if present."""
    path = settings.active_factory_path
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ActiveFactoryState.model_validate(raw)


def clear_active_factory_state(settings: FactorySettings) -> None:
    """Delete persisted RunPod state file if it exists."""
    path = settings.active_factory_path
    if path.exists():
        path.unlink()
