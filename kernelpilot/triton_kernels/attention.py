import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(q_ptr, k_ptr, v_ptr, o_ptr, S, D, scale,
              BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr, IS_FP32: tl.constexpr):
    # Flash-attention forward: one program per (query block, batch*head). K/V
    # blocks are streamed with an online softmax; scores never leave registers.
    pid_m = tl.program_id(0)
    bh = tl.program_id(1).to(tl.int64)
    base = bh * S * D
    rm = pid_m * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    rd = tl.arange(0, BD)
    q_mask = (rm[:, None] < S) & (rd[None, :] < D)
    q = tl.load(q_ptr + base + rm[:, None] * D + rd[None, :], mask=q_mask, other=0.0)
    q = (q.to(tl.float32) * scale).to(q.dtype)

    m_i = tl.full([BM], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BM], dtype=tl.float32)
    acc = tl.zeros([BM, BD], dtype=tl.float32)
    for start in range(0, S, BN):
        kn = start + rn
        kv_mask = (kn[:, None] < S) & (rd[None, :] < D)
        k = tl.load(k_ptr + base + kn[:, None] * D + rd[None, :], mask=kv_mask, other=0.0)
        if IS_FP32:
            s = tl.dot(q, tl.trans(k), input_precision="ieee")
        else:
            s = tl.dot(q, tl.trans(k))
        s = tl.where(kn[None, :] < S, s, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        v = tl.load(v_ptr + base + kn[:, None] * D + rd[None, :], mask=kv_mask, other=0.0)
        if IS_FP32:
            acc = acc * alpha[:, None] + tl.dot(p, v, input_precision="ieee")
        else:
            acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    out = acc / l_i[:, None]
    tl.store(o_ptr + base + rm[:, None] * D + rd[None, :], out.to(o_ptr.dtype.element_ty),
             mask=q_mask)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, H, S, D = q.shape
    o = torch.empty_like(q)
    if q.numel() == 0:
        return o
    BD = max(16, triton.next_power_of_2(D))
    BM = 64 if S > 32 else 16
    BN = 32 if BD <= 64 else 16
    grid = (triton.cdiv(S, BM), B * H)
    _attn_fwd[grid](q, k, v, o, S, D, 1.0 / math.sqrt(D), BM=BM, BN=BN, BD=BD,
                    IS_FP32=q.dtype == torch.float32, num_warps=4)
    return o


IMPLS = {"flash": attention}
