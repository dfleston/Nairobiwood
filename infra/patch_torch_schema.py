#!/usr/bin/env python3
"""Patch PyTorch 2.6 schema inference to support Python 3.10 typing."""

import inspect
import typing
import torch
import torch._library.infer_schema as s

patch_str = """
# Nairobiwood: PEP 585 / 604 typing compatibility patch for Python 3.10
import typing
SUPPORTED_PARAM_TYPES.update({
    list[int]: "SymInt[]",
    list[float]: "float[]",
    list[bool]: "bool[]",
    list[torch.Tensor]: "Tensor[]",
    typing.Optional[torch.Tensor]: "Tensor?",
    typing.Optional[int]: "SymInt?",
    typing.Optional[float]: "float?",
    typing.Optional[str]: "str?",
    typing.Optional[bool]: "bool?",
})
"""

target_file = s.__file__
with open(target_file, "r", encoding="utf-8") as f:
    content = f.read()

if "Nairobiwood: PEP 585" not in content:
    with open(target_file, "a", encoding="utf-8") as f:
        f.write("\n" + patch_str + "\n")
    print(f"Successfully patched {target_file}")
else:
    print(f"Already patched: {target_file}")
