import torch
import triton
import triton.language as tl


@triton.jit
def _softmax_row_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    # Whole row in registers: one read, one write, max/sum/normalize fused.
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + row * N + offs, mask=mask, other=-float("inf")).to(tl.float32)
    x = x - tl.max(x, axis=0)
    e = tl.exp(x)
    y = e / tl.sum(e, axis=0)
    tl.store(y_ptr + row * N + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _softmax_online_kernel(x_ptr, y_ptr, N, BLOCK: tl.constexpr):
    # Rows too long for registers: online (max, sum) over chunks, then a write pass.
    row = tl.program_id(0).to(tl.int64)
    base = row * N
    m = tl.full([BLOCK], -float("inf"), dtype=tl.float32)
    d = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        x = tl.load(x_ptr + base + offs, mask=offs < N, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, x)
        alpha = tl.where(m_new == -float("inf"), 0.0, tl.exp(m - m_new))
        beta = tl.where(x == -float("inf"), 0.0, tl.exp(x - m_new))
        d = d * alpha + beta
        m = m_new
    m_row = tl.max(m, axis=0)
    d_row = tl.sum(tl.where(m == -float("inf"), 0.0, d * tl.exp(m - m_row)), axis=0)
    inv = 1.0 / d_row
    for start in range(0, N, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        y = tl.exp(x - m_row) * inv
        tl.store(y_ptr + base + offs, y.to(y_ptr.dtype.element_ty), mask=mask)


MAX_ROW_BLOCK = 16384


def softmax(x: torch.Tensor) -> torch.Tensor:
    y = torch.empty_like(x)
    N = x.shape[-1]
    M = x.numel() // N if N else 0
    if M == 0:
        return y
    block = triton.next_power_of_2(N)
    if block <= MAX_ROW_BLOCK:
        num_warps = 4 if block <= 2048 else (8 if block <= 8192 else 16)
        _softmax_row_kernel[(M,)](x, y, N, BLOCK=block, num_warps=num_warps)
    else:
        _softmax_online_kernel[(M,)](x, y, N, BLOCK=4096, num_warps=8)
    return y


IMPLS = {"fused": softmax}
