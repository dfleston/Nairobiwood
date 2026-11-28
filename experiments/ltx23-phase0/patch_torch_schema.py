#!/usr/bin/env python3
"""Patch PyTorch 2.6 torch._library.infer_schema and optionally verify ComfyUI nodes."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path

import torch
import torch._library.infer_schema as s

PATCH_MARKER = "# Nairobiwood: PEP 585 / 604 full infer_schema compatibility patch"
PATCH_CODE = f"""
{PATCH_MARKER}
import collections.abc as _cabc
import typing as _typing
import torch as _torch

SUPPORTED_PARAM_TYPES.update({{
    list[_torch.Tensor]: SUPPORTED_PARAM_TYPES[_typing.List[_torch.Tensor]],
    list[_typing.Optional[_torch.Tensor]]: SUPPORTED_PARAM_TYPES[_typing.List[_typing.Optional[_torch.Tensor]]],
    list[_torch.Tensor | None]: SUPPORTED_PARAM_TYPES[_typing.List[_typing.Optional[_torch.Tensor]]],
    list[int]: SUPPORTED_PARAM_TYPES[_typing.List[int]],
    _typing.Optional[list[int]]: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.List[int]]],
    list[int] | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.List[int]]],
    list[float]: SUPPORTED_PARAM_TYPES[_typing.List[float]],
    _typing.Optional[list[float]]: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.List[float]]],
    list[float] | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.List[float]]],
    list[bool]: SUPPORTED_PARAM_TYPES[_typing.List[bool]],
    _typing.Optional[list[bool]]: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.List[bool]]],
    list[bool] | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.List[bool]]],
    list[int | float | bool]: SUPPORTED_PARAM_TYPES[_typing.List[_typing.Union[int, float, bool]]],
    _cabc.Sequence[_torch.Tensor]: SUPPORTED_PARAM_TYPES[_typing.Sequence[_torch.Tensor]],
    _cabc.Sequence[_torch.Tensor | None]: SUPPORTED_PARAM_TYPES[_typing.Sequence[_typing.Optional[_torch.Tensor]]],
    _cabc.Sequence[int]: SUPPORTED_PARAM_TYPES[_typing.Sequence[int]],
    _cabc.Sequence[int] | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.Sequence[int]]],
    _cabc.Sequence[float]: SUPPORTED_PARAM_TYPES[_typing.Sequence[float]],
    _cabc.Sequence[float] | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.Sequence[float]]],
    _cabc.Sequence[bool]: SUPPORTED_PARAM_TYPES[_typing.Sequence[bool]],
    _cabc.Sequence[bool] | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_typing.Sequence[bool]]],
    _torch.Tensor | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_torch.Tensor]],
    int | None: SUPPORTED_PARAM_TYPES[_typing.Optional[int]],
    float | None: SUPPORTED_PARAM_TYPES[_typing.Optional[float]],
    bool | None: SUPPORTED_PARAM_TYPES[_typing.Optional[bool]],
    str | None: SUPPORTED_PARAM_TYPES[_typing.Optional[str]],
    _torch.dtype | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_torch.dtype]],
    _torch.device | None: SUPPORTED_PARAM_TYPES[_typing.Optional[_torch.device]],
    int | float | bool: SUPPORTED_PARAM_TYPES[_typing.Union[int, float, bool]],
    int | float | bool | None: SUPPORTED_PARAM_TYPES[_typing.Union[int, float, bool, type(None)]],
}})
"""

REQUIRED_NODES = [
    "UNETLoader",
    "LoraLoaderModelOnly",
    "DualCLIPLoader",
    "VAELoader",
    "LatentUpscaleModelLoader",
    "LoadImage",
    "ImageScale",
    "CLIPTextEncode",
    "LTXVConditioning",
    "EmptyLTXVLatentVideo",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LTXIdentityOverlapConditioning",
    "RandomNoise",
    "KSamplerSelect",
    "ManualSigmas",
    "CFGGuider",
    "SamplerCustomAdvanced",
    "LTXVSeparateAVLatent",
    "LTXVCropGuides",
    "LTXVLatentUpsampler",
    "VAEDecodeTiled",
    "LTXVAudioVAEDecode",
    "VHS_VideoCombine",
]


def apply_and_verify() -> None:
    target = Path(s.__file__)
    content = target.read_text(encoding="utf-8")
    if PATCH_MARKER not in content:
        target.write_text(content.rstrip() + "\n" + PATCH_CODE + "\n", encoding="utf-8")
        print(f"Patched {target}")
    else:
        print(f"Already patched: {target}")

    reloaded = importlib.reload(s)

    def _test_op(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        residual: torch.Tensor | None,
        stride: list[int],
        out: torch.Tensor,
    ) -> None:
        pass

    schema = reloaded.infer_schema(_test_op, mutates_args=("out",))
    print("Self-test infer_schema passed:", schema)


async def _verify_comfy_nodes(comfy_dir: str = "/opt/ComfyUI") -> None:
    os.chdir(comfy_dir)
    if comfy_dir not in sys.path:
        sys.path.insert(0, comfy_dir)
    sys.argv = ["main.py", "--cpu"]

    import comfy.options

    comfy.options.enable_args_parsing()

    from app.assets.manager import default_asset_manager
    import nodes
    import server

    asset_manager = default_asset_manager()
    srv = server.PromptServer(asyncio.get_running_loop(), asset_manager)
    server.PromptServer.instance = srv
    res = nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False)
    if inspect.isawaitable(res):
        await res

    missing = [n for n in REQUIRED_NODES if n not in nodes.NODE_CLASS_MAPPINGS]
    assert not missing, f"Missing required ComfyUI nodes: {missing}"
    print(
        f"Verified all {len(REQUIRED_NODES)} LTX-2.3 + BFSNodes + VHS nodes in CPU mode! "
        f"Total registered: {len(nodes.NODE_CLASS_MAPPINGS)}"
    )


if __name__ == "__main__":
    apply_and_verify()
    if "--verify-nodes" in sys.argv:
        asyncio.run(_verify_comfy_nodes())
