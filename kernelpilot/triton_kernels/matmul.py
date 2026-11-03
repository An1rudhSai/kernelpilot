import os

import torch
import triton
import triton.language as tl


@triton.jit
def _matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                   GROUP_M: tl.constexpr, IS_FP32: tl.constexpr):
    # Grouped program ordering: consecutive programs share rows of A, which
    # improves L2 reuse over plain row-major tile order.
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    group_m = min(num_m - first_m, GROUP_M)
    pid_m = first_m + (pid % group) % group_m
    pid_n = (pid % group) // group_m

    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a_ptrs = a_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = b_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        k_left = K - k * BK
        a = tl.load(a_ptrs, mask=(rm[:, None] < M) & (rk[None, :] < k_left), other=0.0)
        b = tl.load(b_ptrs, mask=(rk[:, None] < k_left) & (rn[None, :] < N), other=0.0)
        if IS_FP32:
            acc = tl.dot(a, b, acc, input_precision="ieee")  # no TF32
        else:
            acc = tl.dot(a, b, acc)
        a_ptrs += BK * stride_ak
        b_ptrs += BK * stride_bk
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(c_ptr.dtype.element_ty), mask=mask)


def _configs():
    out = []
    for bm, bn, bk, w, s in [(128, 128, 32, 8, 3), (128, 64, 32, 4, 4), (64, 128, 32, 4, 4),
                             (64, 64, 32, 4, 4), (128, 128, 64, 8, 3), (64, 64, 64, 4, 2)]:
        out.append(triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8},
                                 num_warps=w, num_stages=s))
    return out


_autotuned = triton.autotune(configs=_configs(), key=["M", "N", "K", "IS_FP32"])(_matmul_kernel)


def _launch(kernel, a, b, **meta):
    M, K = a.shape
    N = b.shape[1]
    c = torch.empty((M, N), dtype=a.dtype, device=a.device)
    if M == 0 or N == 0:
        return c
    if K == 0:
        return c.zero_()

    def grid(m):
        return (triton.cdiv(M, m["BM"]) * triton.cdiv(N, m["BN"]),)

    kernel[grid](a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                 c.stride(0), c.stride(1), IS_FP32=a.dtype == torch.float32, **meta)
    return c


def matmul_tiled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _launch(_matmul_kernel, a, b, BM=64, BN=64, BK=32, GROUP_M=8, num_warps=4)


def matmul_autotuned(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return _launch(_autotuned, a, b)


IMPLS = {"tiled": matmul_tiled}
# Autotuning benchmarks every config, which the CPU interpreter cannot do usefully.
if os.environ.get("TRITON_INTERPRET") != "1":
    IMPLS["autotuned"] = matmul_autotuned
