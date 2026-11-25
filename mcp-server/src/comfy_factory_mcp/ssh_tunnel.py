"""Zero-trust SSH tunnel, rsync asset transfer, tmux job runner, and shred purge."""

from __future__ import annotations

import asyncio
import os
import shlex
import signal
import subprocess
from pathlib import Path

import httpx

ALLOWED_REMOTE_SCRATCH_PREFIXES = (
    "/dev/shm/factory_scratch/",
    "/tmp/factory_scratch/",
)


class SSHTunnelError(RuntimeError):
    """Raised when SSH tunnel, rsync, or remote execution fails."""


def validate_remote_scratch_path(remote_path: str) -> str:
    """Ensure remote purge/write operations stay strictly inside ephemeral scratch."""
    normalized = os.path.normpath(remote_path)
    if not any(normalized.startswith(prefix) for prefix in ALLOWED_REMOTE_SCRATCH_PREFIXES):
        raise ValueError(
            f"Refusing remote operation on '{remote_path}': must reside under "
            f"{ALLOWED_REMOTE_SCRATCH_PREFIXES}."
        )
    return normalized


class SSHTunnelManager:
    """Manages encrypted SSH forwarding to remote 127.0.0.1:8188 and file transfers."""

    def __init__(self, ssh_host: str, ssh_port: int, ssh_key_path: Path, local_port: int = 8188):
        self.ssh_host = ssh_host
        self.ssh_port = ssh_port
        self.ssh_key_path = ssh_key_path
        self.local_port = local_port

    def _base_ssh_opts(self) -> list[str]:
        return [
            "-i",
            str(self.ssh_key_path),
            "-p",
            str(self.ssh_port),
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "BatchMode=yes",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=3",
        ]

    def _ssh_cmd_str(self) -> str:
        return (
            f"ssh -i {shlex.quote(str(self.ssh_key_path))} "
            f"-p {self.ssh_port} "
            "-o StrictHostKeyChecking=accept-new -o BatchMode=yes "
            "-o ServerAliveInterval=15 -o ServerAliveCountMax=6 -o ConnectTimeout=20"
        )

    def start_tunnel(self) -> int:
        """Spawn background SSH port forward mapping local_port -> remote 127.0.0.1:8188."""
        cmd = [
            "ssh",
            "-N",
            "-L",
            f"127.0.0.1:{self.local_port}:127.0.0.1:8188",
            "-o",
            "ExitOnForwardFailure=yes",
            *self._base_ssh_opts(),
            f"root@{self.ssh_host}",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        return proc.pid

    @staticmethod
    def is_pid_alive(pid: int | None) -> bool:
        if pid is None or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def stop_tunnel(pid: int | None) -> None:
        if pid is None or pid <= 0:
            return
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    async def wait_for_comfyui(self, timeout_sec: float = 120.0) -> dict:
        """Poll local forwarded ComfyUI /system_stats until responsive."""
        url = f"http://127.0.0.1:{self.local_port}/system_stats"
        deadline = asyncio.get_running_loop().time() + timeout_sec
        last_err: Exception | None = None
        async with httpx.AsyncClient(timeout=5.0) as client:
            while asyncio.get_running_loop().time() < deadline:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return resp.json()
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                await asyncio.sleep(2.0)
        raise SSHTunnelError(f"ComfyUI tunnel health check failed at {url}: {last_err}")

    async def run_remote(self, command: str) -> str:
        """Execute a remote shell command over SSH and return stdout (with transient retry)."""
        last_err = ""
        for attempt in range(3):
            proc = await asyncio.create_subprocess_exec(
                "ssh",
                *self._base_ssh_opts(),
                f"root@{self.ssh_host}",
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                return stdout.decode()
            last_err = stderr.decode().strip()
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
        raise SSHTunnelError(
            f"Remote SSH command failed ({proc.returncode}): {last_err}"
        )

    async def wait_for_ssh_ready(
        self, timeout_sec: float = 240.0, poll_interval: float = 3.0
    ) -> None:
        """Poll remote SSH daemon until key-based authentication succeeds."""
        deadline = asyncio.get_running_loop().time() + timeout_sec
        last_err: Exception | None = None
        while asyncio.get_running_loop().time() < deadline:
            try:
                out = await self.run_remote("echo ready")
                if "ready" in out:
                    return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
            await asyncio.sleep(poll_interval)
        raise SSHTunnelError(
            f"Timed out waiting for SSH readiness on {self.ssh_host}:{self.ssh_port}: {last_err}"
        )

    async def bootstrap_remote_host(
        self,
        infra_dir: Path,
        *,
        bootstrap_models: bool = False,
        timeout_sec: float = 900.0,
    ) -> str:
        """Upload infra scripts and bootstrap ComfyUI + scratch + ComfyUI process over SSH."""
        await self.wait_for_ssh_ready()
        # Ensure rsync is installed on base image before uploading
        await self.run_remote("which rsync >/dev/null 2>&1 || (apt-get update -y && apt-get install -y rsync tmux)")
        remote_infra = "/workspace/nairobiwood/infra"
        await self.run_remote(
            f"mkdir -p {shlex.quote(remote_infra)} && chmod 700 {shlex.quote(remote_infra)}"
        )

        files = [
            infra_dir / "remote_bootstrap.sh",
            infra_dir / "bootstrap_models.sh",
            infra_dir / "models_manifest.json",
        ]
        for f in files:
            if not f.exists():
                raise FileNotFoundError(f"Missing bootstrap file: {f}")

        proc = await asyncio.create_subprocess_exec(
            "rsync",
            "-rlptDz",
            "--no-owner",
            "--no-group",
            "-e",
            self._ssh_cmd_str(),
            *[str(f) for f in files],
            f"root@{self.ssh_host}:{remote_infra}/",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise SSHTunnelError(f"Failed uploading bootstrap scripts: {stderr.decode().strip()}")

        flag = " --bootstrap-models" if bootstrap_models else ""
        cmd = (
            f"chmod +x {remote_infra}/remote_bootstrap.sh {remote_infra}/bootstrap_models.sh && "
            f"bash {remote_infra}/remote_bootstrap.sh{flag}"
        )
        return await asyncio.wait_for(self.run_remote(cmd), timeout=timeout_sec)

    async def rsync_upload(self, local_path: Path, remote_dir: str) -> None:
        """Upload local file or directory contents into remote scratch directory (with retry)."""
        safe_remote = validate_remote_scratch_path(remote_dir)
        await self.run_remote(f"mkdir -p {shlex.quote(safe_remote)} && chmod 700 {shlex.quote(safe_remote)}")
        src = f"{local_path}/" if local_path.is_dir() else str(local_path)
        last_err = ""
        for attempt in range(3):
            proc = await asyncio.create_subprocess_exec(
                "rsync",
                "-rlptDz",
                "--partial",
                "--no-owner",
                "--no-group",
                "-e",
                self._ssh_cmd_str(),
                src,
                f"root@{self.ssh_host}:{safe_remote}/",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return
            last_err = stderr.decode().strip()
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
        raise SSHTunnelError(f"rsync upload failed: {last_err}")

    async def rsync_download(self, remote_path: str, local_dest: Path) -> None:
        """Download remote artifact or directory back to macOS (with retry)."""
        safe_remote = validate_remote_scratch_path(remote_path)
        local_dest.parent.mkdir(parents=True, exist_ok=True)
        last_err = ""
        for attempt in range(3):
            proc = await asyncio.create_subprocess_exec(
                "rsync",
                "-rlptDz",
                "--partial",
                "--no-owner",
                "--no-group",
                "-e",
                self._ssh_cmd_str(),
                f"root@{self.ssh_host}:{safe_remote}",
                f"{local_dest}/" if local_dest.is_dir() else str(local_dest),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode == 0:
                return
            last_err = stderr.decode().strip()
            if attempt < 2:
                await asyncio.sleep(2.0 * (attempt + 1))
        raise SSHTunnelError(f"rsync download failed: {last_err}")

    async def purge_remote(self, remote_path: str) -> None:
        """Securely shred and remove remote scratch artifacts immediately after pull."""
        safe_remote = validate_remote_scratch_path(remote_path)
        cmd = (
            f"if [ -e {shlex.quote(safe_remote)} ]; then "
            f"find {shlex.quote(safe_remote)} -type f -exec shred -u {{}} + && "
            f"rm -rf {shlex.quote(safe_remote)}; fi"
        )
        await self.run_remote(cmd)
