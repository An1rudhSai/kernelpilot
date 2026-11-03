import torch
import triton
import triton.language as tl

EPS = 1e-5


@triton.jit
def _ln_row_kernel(x_ptr, w_ptr, b_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
    # Whole row in registers: one read of x, centered variance, fused affine.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    xc = tl.where(mask, x - mean, 0.0)
    var = tl.sum(xc * xc, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = xc * rstd * w + b
    tl.store(y_ptr + row * N + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _ln_loop_kernel(x_ptr, w_ptr, b_ptr, y_ptr, N, eps, BLOCK: tl.constexpr):
    # Long rows: chunked passes for mean, centered variance, then normalize.
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + base + offs, mask=offs < N, other=0.0).to(tl.float32)
    mean = tl.sum(acc, axis=0) / N
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        xc = tl.where(mask, x - mean, 0.0)
        acc += xc * xc
    rstd = 1.0 / tl.sqrt(tl.sum(acc, axis=0) / N + eps)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


MAX_ROW_BLOCK = 16384


def layernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    N = x.shape[-1]
    M = x.numel() // N if N else 0
    if M == 0:
        return y
    block = triton.next_power_of_2(N)
    if block <= MAX_ROW_BLOCK:
        num_warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
        _ln_row_kernel[(M,)](x, weight, bias, y, N, EPS, BLOCK=block, num_warps=num_warps)
    else:
        _ln_loop_kernel[(M,)](x, weight, bias, y, N, EPS, BLOCK=4096, num_warps=8)
    return y


IMPLS = {"fused": layernorm}
