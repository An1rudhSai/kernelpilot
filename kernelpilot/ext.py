"""Loads the CMake-built HIP extension (build/libkernelpilot_ops.so) as torch.ops.kp."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
_LIB_NAMES = ("libkernelpilot_ops.so", "libkernelpilot_ops.dylib", "kernelpilot_ops.dll")


def library_path() -> Path | None:
    env = os.environ.get("KP_EXT_PATH")
    if env:
        return Path(env)
    for name in _LIB_NAMES:
        p = REPO_ROOT / "build" / name
        if p.exists():
            return p
    return None


@lru_cache(maxsize=1)
def load() -> bool:
    """Load the extension once. Returns False (instead of raising) when it is absent."""
    path = library_path()
    if path is None or not path.exists() or not torch.cuda.is_available():
        return False
    torch.ops.load_library(str(path))
    return True


def ops():
    if not load():
        raise RuntimeError(
            "HIP extension not built or no GPU available; run `make build` first "
            "(or set KP_EXT_PATH to the built library)."
        )
    return torch.ops.kp
