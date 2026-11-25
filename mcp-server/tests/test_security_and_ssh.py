"""Security boundary tests for path traversal and remote scratch shredding."""

from __future__ import annotations

from pathlib import Path

import pytest

from comfy_factory_mcp.config import FactorySettings, ensure_data_dirs, resolve_safe_path
from comfy_factory_mcp.ssh_tunnel import validate_remote_scratch_path


def test_resolve_safe_path_blocks_traversal(tmp_path: Path) -> None:
    settings = FactorySettings(factory_data_dir=tmp_path)
    ensure_data_dirs(settings)

    ok = resolve_safe_path("outputs/pilot/shot_01", settings)
    assert ok == (tmp_path / "outputs/pilot/shot_01").resolve()

    with pytest.raises(ValueError):
        resolve_safe_path("../../etc/passwd", settings)

    with pytest.raises(ValueError):
        resolve_safe_path("/etc/passwd", settings)


def test_validate_remote_scratch_path_enforces_ephemeral_prefixes() -> None:
    assert (
        validate_remote_scratch_path("/dev/shm/factory_scratch/datasets/elena")
        == "/dev/shm/factory_scratch/datasets/elena"
    )
    assert (
        validate_remote_scratch_path("/tmp/factory_scratch/jobs/j1")
        == "/tmp/factory_scratch/jobs/j1"
    )

    for bad in [
        "/workspace/models",
        "/dev/shm/factory_scratch/../root/.ssh",
        "/root",
        "/",
    ]:
        with pytest.raises(ValueError):
            validate_remote_scratch_path(bad)
