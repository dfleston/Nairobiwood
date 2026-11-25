"""Configuration, directory setup, and path-traversal guards for Nairobiwood."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]


class FactorySettings(BaseSettings):
    """Runtime settings loaded from environment variables and optional .env file."""

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    nairobiwood_root: Path = Field(default=DEFAULT_REPO_ROOT)
    factory_data_dir: Path = Field(default_factory=lambda: Path("~/ComfyFactory").expanduser())
    runpod_api_key: str = Field(default="")
    runpod_network_volume_id: str = Field(default="")
    runpod_datacenter_id: str = Field(default="")
    hf_token: str = Field(default="")
    ssh_key_path: Path = Field(
        default_factory=lambda: Path("~/.ssh/id_ed25519_factory").expanduser()
    )
    container_image: str = Field(default="runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04")
    comfy_local_port: int = Field(default=8188)
    studio_http_port: int = Field(default=8190)
    director_app_html: Path | None = Field(default=None)

    def resolved_data_dir(self) -> Path:
        """Return the expanded, resolved local media/biometric directory."""
        return Path(os.path.expanduser(str(self.factory_data_dir))).resolve()

    def resolved_ssh_key(self) -> Path:
        """Return the expanded private SSH key path."""
        return Path(os.path.expanduser(str(self.ssh_key_path))).resolve()

    def resolved_workflows_dir(self) -> Path:
        """Return the workflows directory inside Nairobiwood."""
        return (self.nairobiwood_root.resolve() / "workflows").resolve()

    def resolved_director_html(self) -> Path:
        """Return the single-file bundled Director Studio HTML path."""
        if self.director_app_html is not None:
            return Path(os.path.expanduser(str(self.director_app_html))).resolve()
        return (self.nairobiwood_root.resolve() / "mcp-app" / "dist" / "director.html").resolve()

    @property
    def active_factory_path(self) -> Path:
        return self.resolved_data_dir() / ".active_factory.json"

    @property
    def datasets_dir(self) -> Path:
        return self.resolved_data_dir() / "datasets"

    @property
    def loras_dir(self) -> Path:
        return self.resolved_data_dir() / "loras"

    @property
    def voices_dir(self) -> Path:
        return self.resolved_data_dir() / "voices"

    @property
    def outputs_dir(self) -> Path:
        return self.resolved_data_dir() / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.resolved_data_dir() / "jobs"


def ensure_data_dirs(settings: FactorySettings) -> None:
    """Create local ComfyFactory runtime directories with restrictive permissions."""
    data_dir = settings.resolved_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)
    for sub, mode in (
        (settings.datasets_dir, 0o700),
        (settings.loras_dir, 0o700),
        (settings.voices_dir, 0o755),
        (settings.outputs_dir, 0o700),
        (settings.jobs_dir, 0o700),
    ):
        sub.mkdir(parents=True, exist_ok=True)
        os.chmod(sub, mode)


def resolve_safe_path(
    path: str | Path,
    settings: FactorySettings,
    *,
    allowed_roots: Iterable[Path] | None = None,
    must_exist: bool = False,
) -> Path:
    """Resolve a user/agent-supplied path and ensure it stays inside allowed roots."""
    candidate_raw = os.path.expanduser(str(path))
    candidate = Path(candidate_raw)
    if not candidate.is_absolute():
        candidate = settings.resolved_data_dir() / candidate
    resolved = candidate.resolve()

    roots = (
        [r.resolve() for r in allowed_roots]
        if allowed_roots is not None
        else [
            settings.resolved_data_dir(),
            settings.resolved_workflows_dir(),
            settings.nairobiwood_root.resolve(),
            (settings.nairobiwood_root.resolve().parent / "StudioMode").resolve(),
        ]
    )
    if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
        raise ValueError(
            f"Path '{path}' resolves to '{resolved}', which is outside allowed directories."
        )
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Required path does not exist: {resolved}")
    return resolved
