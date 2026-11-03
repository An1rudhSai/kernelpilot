"""Every implementation of every kernel, under one naming scheme:

    torch            PyTorch built-in (the baseline)
    hip:<stage>      hand-written HIP kernel, stages ordered naive -> best
    triton:<name>    hand-written Triton kernel
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Callable

from . import ext
from .specs import get_spec

# HIP stages per kernel, in order; the index is the `variant` passed to the op.
HIP_STAGES: dict[str, list[str]] = {
    "vector_add": ["naive", "gridstride", "vectorized"],
    "reduce_sum": ["naive", "seq_addressing", "vec_warp_shuffle"],
    "softmax": ["naive", "block_per_row", "online_fused"],
    "matmul": ["naive", "smem_tiled", "reg_tiled", "vectorized"],
}

# Extra trailing arguments some ops take besides the spec inputs.
_HIP_EXTRA_ARGS: dict[str, tuple] = {}


def triton_available() -> bool:
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    import torch

    return torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1"


def _hip_impl(kernel: str, variant: int) -> Callable:
    extra = _HIP_EXTRA_ARGS.get(kernel, ())

    def run(*inputs):
        return getattr(ext.ops(), kernel)(*inputs, *extra, variant)

    run.__name__ = f"hip_{kernel}_{HIP_STAGES[kernel][variant]}"
    return run


def hip_impls(kernel: str) -> dict[str, Callable]:
    if not ext.load():
        return {}
    return {f"hip:{stage}": _hip_impl(kernel, i)
            for i, stage in enumerate(HIP_STAGES.get(kernel, []))}


def triton_impls(kernel: str) -> dict[str, Callable]:
    if not triton_available():
        return {}
    try:
        mod = importlib.import_module(f"kernelpilot.triton_kernels.{kernel}")
    except ModuleNotFoundError:
        return {}
    return {f"triton:{name}": fn for name, fn in mod.IMPLS.items()}


def get_impls(kernel: str, include: tuple[str, ...] = ("torch", "hip", "triton")
              ) -> dict[str, Callable]:
    spec = get_spec(kernel)
    impls: dict[str, Callable] = {}
    if "torch" in include:
        impls["torch"] = spec.baseline
    if "hip" in include:
        impls.update(hip_impls(kernel))
    if "triton" in include:
        impls.update(triton_impls(kernel))
    return impls
