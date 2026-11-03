"""Kernel specifications: the single source of truth shared by the correctness
harness, the benchmark harness and the agent.

A spec says what a kernel computes (a PyTorch reference evaluated in high
precision), which shapes and dtypes it must handle, how much work it does
(FLOPs and ideal bytes moved) and how close an answer has to be.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass

import torch

Shape = tuple[int, ...]
Inputs = tuple[torch.Tensor, ...]


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


@dataclass
class KernelSpec:
    name: str
    description: str
    # The function a candidate kernel must implement, as Python source for prompts.
    signature: str
    make_inputs: Callable[[Shape, torch.dtype, torch.device, int], Inputs]
    # High-precision reference. Returns float32/float64; compared after upcasting.
    reference: Callable[..., torch.Tensor]
    # PyTorch's own fast implementation, used as the benchmark baseline.
    baseline: Callable[..., torch.Tensor]
    test_shapes: list[Shape]
    bench_shapes: list[Shape]
    dtypes: list[torch.dtype]
    flops: Callable[[Shape], float]
    bytes_moved: Callable[[Shape, torch.dtype], float]
    tolerance: Callable[[Shape, torch.dtype], Tolerance]
    shape_names: tuple[str, ...]
    # dtype the kernel must return for a given input dtype (default: the same).
    output_dtype: Callable[[torch.dtype], torch.dtype] = lambda dt: dt

    def reference_source(self) -> str:
        return inspect.getsource(self.reference)

    def format_shape(self, shape: Shape) -> str:
        return ", ".join(f"{n}={v}" for n, v in zip(self.shape_names, shape, strict=True))


def _esize(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _randn(shape: Shape, dtype: torch.dtype, device: torch.device, gen: torch.Generator,
           scale: float = 1.0) -> torch.Tensor:
    # Generate in float32 on the target device, then cast, so fp16 and fp32 runs
    # with the same seed see the same values (up to rounding).
    x = torch.randn(shape, generator=gen, device=device, dtype=torch.float32) * scale
    return x.to(dtype)


def _gen(device: torch.device, seed: int) -> torch.Generator:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    return g


# --------------------------------------------------------------------------- vector add

def vector_add_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.double() + b.double()


def _vector_add_inputs(shape, dtype, device, seed):
    g = _gen(device, seed)
    return _randn(shape, dtype, device, g), _randn(shape, dtype, device, g)


VECTOR_ADD = KernelSpec(
    name="vector_add",
    description="Elementwise c = a + b over 1-D tensors.",
    signature="def kernel_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor  # same dtype as a",
    make_inputs=_vector_add_inputs,
    reference=vector_add_reference,
    baseline=torch.add,
    test_shapes=[(1,), (7,), (1000,), (1024,), (65537,), (1 << 20,), (3_000_001,)],
    bench_shapes=[(1 << 24,), (1 << 26,)],
    dtypes=[torch.float32, torch.float16],
    flops=lambda s: float(s[0]),
    bytes_moved=lambda s, dt: 3.0 * s[0] * _esize(dt),
    tolerance=lambda s, dt: Tolerance(1e-6, 1e-6) if dt == torch.float32
    else Tolerance(1e-3, 1e-3),
    shape_names=("n",),
)


# --------------------------------------------------------------------------- reduction

def reduce_sum_reference(x: torch.Tensor) -> torch.Tensor:
    return x.double().sum()


def _reduce_sum_inputs(shape, dtype, device, seed):
    return (_randn(shape, dtype, device, _gen(device, seed)),)


def _reduce_sum_tol(shape, dtype):
    # fp32 accumulation of n O(1) values: rounding error grows ~ sqrt(n) * eps.
    n = shape[0]
    return Tolerance(atol=1e-5 * math.sqrt(n) + 1e-5, rtol=1e-5)


REDUCE_SUM = KernelSpec(
    name="reduce_sum",
    description="Sum of all elements of a 1-D tensor, returned as a float32 scalar "
                "(0-dim tensor), accumulating in float32 regardless of input dtype.",
    signature="def kernel_fn(x: torch.Tensor) -> torch.Tensor  # 0-dim float32",
    make_inputs=_reduce_sum_inputs,
    reference=reduce_sum_reference,
    baseline=lambda x: x.sum(dtype=torch.float32),
    test_shapes=[(1,), (7,), (1000,), (1024,), (65537,), (1 << 20,), (3_000_001,)],
    bench_shapes=[(1 << 24,), (1 << 26,)],
    dtypes=[torch.float32, torch.float16],
    flops=lambda s: float(s[0]),
    bytes_moved=lambda s, dt: float(s[0] * _esize(dt) + 4),
    tolerance=_reduce_sum_tol,
    shape_names=("n",),
    output_dtype=lambda dt: torch.float32,
)


# --------------------------------------------------------------------------- softmax

def softmax_reference(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.double(), dim=-1)


def _softmax_inputs(shape, dtype, device, seed):
    # Scale 30 makes exp() overflow (fp32 overflows past ~88) unless the kernel
    # subtracts the row max first, so unstable implementations fail loudly.
    return (_randn(shape, dtype, device, _gen(device, seed), scale=30.0),)


SOFTMAX = KernelSpec(
    name="softmax",
    description="Numerically stable softmax over the last dimension of a contiguous "
                "(M, N) tensor. Output has the input dtype.",
    signature="def kernel_fn(x: torch.Tensor) -> torch.Tensor  # same shape/dtype as x",
    make_inputs=_softmax_inputs,
    reference=softmax_reference,
    baseline=lambda x: torch.softmax(x, dim=-1),
    test_shapes=[(1, 1), (1, 7), (3, 1000), (17, 1024), (128, 4097), (64, 12288),
                 (4, 32768), (1000, 129)],
    bench_shapes=[(4096, 1024), (4096, 4096), (1024, 32768)],
    dtypes=[torch.float32, torch.float16],
    flops=lambda s: 5.0 * s[0] * s[1],
    bytes_moved=lambda s, dt: 2.0 * s[0] * s[1] * _esize(dt),
    tolerance=lambda s, dt: Tolerance(1e-6, 1e-5) if dt == torch.float32
    else Tolerance(1e-3, 1e-2),
    shape_names=("M", "N"),
)


# --------------------------------------------------------------------------- matmul

def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return a.double() @ b.double()


def _matmul_inputs(shape, dtype, device, seed):
    M, N, K = shape
    g = _gen(device, seed)
    return _randn((M, K), dtype, device, g), _randn((K, N), dtype, device, g)


def _matmul_tol(shape, dtype):
    # Outputs are sums of K products of N(0,1) values, so magnitudes grow like
    # sqrt(K) and fp32 accumulation error grows roughly like K * eps. A real bug
    # (wrong index, missing tile) gives errors of order sqrt(K), far above these.
    K = shape[2]
    if dtype == torch.float32:
        return Tolerance(atol=2e-6 * K + 1e-5, rtol=1e-4)
    return Tolerance(atol=1e-3 * math.sqrt(K) + 1e-3, rtol=1e-2)


MATMUL = KernelSpec(
    name="matmul",
    description="C = A @ B for row-major A (M, K) and B (K, N). Output has the input dtype; "
                "accumulate in float32. TF32 is not allowed for float32 inputs.",
    signature="def kernel_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor  # (M, N)",
    make_inputs=_matmul_inputs,
    reference=matmul_reference,
    baseline=torch.matmul,
    test_shapes=[(1, 1, 1), (7, 13, 5), (64, 64, 64), (127, 129, 65), (256, 512, 128),
                 (33, 1000, 77), (512, 256, 1024), (1024, 1024, 1024)],
    bench_shapes=[(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)],
    dtypes=[torch.float32, torch.float16],
    flops=lambda s: 2.0 * s[0] * s[1] * s[2],
    bytes_moved=lambda s, dt: float((s[0] * s[2] + s[2] * s[1] + s[0] * s[1]) * _esize(dt)),
    tolerance=_matmul_tol,
    shape_names=("M", "N", "K"),
)


SPECS: dict[str, KernelSpec] = {s.name: s for s in [VECTOR_ADD, REDUCE_SUM, SOFTMAX, MATMUL]}


def get_spec(name: str) -> KernelSpec:
    if name not in SPECS:
        raise KeyError(f"unknown kernel {name!r}; choose from {sorted(SPECS)}")
    return SPECS[name]

