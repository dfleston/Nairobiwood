"""Tests for RunPod Secure Cloud GPU selection, 22/tcp-only provisioning, and 0600 state."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import httpx
import pytest
import respx

from comfy_factory_mcp.cloud_providers import (
    RUNPOD_GRAPHQL_URL,
    CloudProviderError,
    RunPodProvider,
    load_active_factory_state,
    save_active_factory_state,
)
from comfy_factory_mcp.config import FactorySettings
from comfy_factory_mcp.models import ActiveFactoryState


@respx.mock
async def test_find_best_gpu_filters_secure_cloud_and_budget() -> None:
    respx.post(RUNPOD_GRAPHQL_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "data": {
                    "gpuTypes": [
                        {
                            "id": "NVIDIA GeForce RTX 4090",
                            "displayName": "RTX 4090",
                            "memoryInGb": 24,
                            "secureCloud": True,
                            "securePrice": 0.44,
                        },
                        {
                            "id": "NVIDIA L40S-Unsecure",
                            "displayName": "L40S Community",
                            "memoryInGb": 48,
                            "secureCloud": False,
                            "securePrice": 0.50,
                        },
                        {
                            "id": "NVIDIA L40S",
                            "displayName": "L40S",
                            "memoryInGb": 48,
                            "secureCloud": True,
                            "securePrice": 0.79,
                        },
                        {
                            "id": "NVIDIA A100 80GB PCIe",
                            "displayName": "A100 PCIe",
                            "memoryInGb": 80,
                            "secureCloud": True,
                            "securePrice": 1.49,
                        },
                    ]
                }
            },
        )
    )
    provider = RunPodProvider("rp_test_key")
    best = await provider.find_best_gpu("L40S|A100", max_price_hr=1.69, min_vram_gb=48)
    assert best["id"] == "NVIDIA L40S"
    assert best["effective_price"] == pytest.approx(0.79)


@respx.mock
async def test_provision_pod_exposes_only_ssh_port_22() -> None:
    captured_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        body = json.loads(request.content.decode())
        captured_payloads.append(body)
        if "podFindAndDeployOnDemand" in body["query"]:
            return httpx.Response(
                200,
                json={"data": {"podFindAndDeployOnDemand": {"id": "pod_abc123", "desiredStatus": "RUNNING"}}},
            )
        if "podTerminate" in body["query"]:
            return httpx.Response(200, json={"data": {"podTerminate": None}})
        return httpx.Response(
            200,
            json={
                "data": {
                    "pod": {
                        "id": "pod_abc123",
                        "runtime": {
                            "ports": [
                                {"ip": "203.0.113.10", "isIpPublic": True, "privatePort": 22, "publicPort": 40022}
                            ]
                        },
                    }
                }
            },
        )

    respx.post(RUNPOD_GRAPHQL_URL).mock(side_effect=handler)
    provider = RunPodProvider("rp_test_key")
    pod = await provider.provision_pod(
        gpu_id="NVIDIA L40S",
        image_name="ghcr.io/nairobiwood/wan22-comfy-factory:latest",
        ssh_public_key="ssh-ed25519 AAAA_TEST_KEY",
        hf_token="hf_secret",
        network_volume_id="vol_xyz",
    )
    assert pod["id"] == "pod_abc123"
    deploy_input = captured_payloads[0]["variables"]["input"]
    assert deploy_input["cloudType"] == "SECURE"
    assert deploy_input["ports"] == "22/tcp"
    assert "8188" not in deploy_input["ports"]
    assert deploy_input["networkVolumeId"] == "vol_xyz"

    ip, port = await provider.wait_for_ssh_endpoint("pod_abc123", timeout_sec=5.0, poll_interval=0.01)
    assert (ip, port) == ("203.0.113.10", 40022)
    await provider.terminate_pod("pod_abc123")


def test_active_factory_state_saved_with_0600_mode(tmp_path: Path) -> None:
    settings = FactorySettings(factory_data_dir=tmp_path)
    state = ActiveFactoryState(
        pod_id="pod_999",
        gpu_name="L40S",
        vram_gb=48,
        hourly_rate_usd=0.79,
        ssh_host="203.0.113.10",
        ssh_port=40022,
    )
    saved = save_active_factory_state(state, settings)
    mode = stat.S_IMODE(os.stat(saved).st_mode)
    assert mode == 0o600
    loaded = load_active_factory_state(settings)
    assert loaded is not None and loaded.pod_id == "pod_999"


@respx.mock
async def test_find_best_gpu_raises_when_over_budget() -> None:
    respx.post(RUNPOD_GRAPHQL_URL).mock(
        return_value=httpx.Response(
            200,
            json={"data": {"gpuTypes": [{"id": "L40S", "displayName": "L40S", "memoryInGb": 48, "secureCloud": True, "securePrice": 2.99}]}},
        )
    )
    provider = RunPodProvider("rp_test_key")
    with pytest.raises(CloudProviderError):
        await provider.find_best_gpu("L40S", max_price_hr=1.00)


def test_default_container_image_is_official_runpod_pytorch() -> None:
    settings = FactorySettings(_env_file=None)
    assert settings.container_image == "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"


async def test_factory_up_bootstraps_before_tunnel_and_persists_provisional_pod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from comfy_factory_mcp.service import FactoryService

    ssh_priv = tmp_path / "id_ed25519_test"
    ssh_priv.write_text("PRIVATE", encoding="utf-8")
    ssh_priv.with_suffix(".pub").write_text("ssh-ed25519 AAAA_TEST", encoding="utf-8")

    settings = FactorySettings(
        _env_file=None,
        factory_data_dir=tmp_path / "data",
        runpod_api_key="rp_test_key",
        ssh_key_path=ssh_priv,
    )
    svc = FactoryService(settings=settings)

    call_order: list[str] = []

    class FakeProvider:
        def __init__(self, api_key: str) -> None:
            pass

        async def find_best_gpu(self, gpu_filter: str, max_price_hr: float) -> dict:
            call_order.append("find_best_gpu")
            return {"id": "NVIDIA L40S", "displayName": "L40S", "memoryInGb": 48, "effective_price": 0.79}

        async def provision_pod(self, **kwargs: object) -> dict:
            call_order.append("provision_pod")
            return {"id": "pod_orphan_safe"}

        async def wait_for_ssh_endpoint(self, pod_id: str) -> tuple[str, int]:
            # Provisional state must already exist on disk before waiting for SSH
            saved = load_active_factory_state(settings)
            assert saved is not None and saved.pod_id == "pod_orphan_safe"
            call_order.append("wait_for_ssh_endpoint")
            return ("203.0.113.55", 22022)

    class FakeTunnel:
        def __init__(self, host: str, port: int, key: Path, local_port: int = 8188) -> None:
            pass

        async def bootstrap_remote_host(self, infra_dir: Path, *, bootstrap_models: bool = False) -> str:
            call_order.append(f"bootstrap:{bootstrap_models}")
            return "bootstrapped"

        def start_tunnel(self) -> int:
            call_order.append("start_tunnel")
            return 4242

        async def wait_for_comfyui(self) -> dict:
            call_order.append("wait_for_comfyui")
            return {"system": "ok"}

    monkeypatch.setattr("comfy_factory_mcp.service.RunPodProvider", FakeProvider)
    monkeypatch.setattr("comfy_factory_mcp.service.SSHTunnelManager", FakeTunnel)

    result = await svc.factory_up(bootstrap_models=True)
    assert call_order == [
        "find_best_gpu",
        "provision_pod",
        "wait_for_ssh_endpoint",
        "bootstrap:True",
        "start_tunnel",
        "wait_for_comfyui",
    ]
    assert result["pod_id"] == "pod_orphan_safe"
    assert result["tunnel_pid"] == 4242
    assert result["bootstrap_models_started"] is True

    # Verify provisional pod_id survives if bootstrap raises mid-flight
    class FailingTunnel(FakeTunnel):
        async def bootstrap_remote_host(self, infra_dir: Path, *, bootstrap_models: bool = False) -> str:
            raise RuntimeError("SSH bootstrap interrupted")

    monkeypatch.setattr("comfy_factory_mcp.service.SSHTunnelManager", FailingTunnel)
    with pytest.raises(RuntimeError, match="SSH bootstrap interrupted"):
        await svc.factory_up()
    persisted = load_active_factory_state(settings)
    assert persisted is not None and persisted.pod_id == "pod_orphan_safe"
