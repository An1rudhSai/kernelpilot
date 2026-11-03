import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(a_ptr, b_ptr, c_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(a_ptr + offs, mask=mask)
    b = tl.load(b_ptr + offs, mask=mask)
    tl.store(c_ptr + offs, a + b, mask=mask)


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    c = torch.empty_like(a)
    n = a.numel()
    if n == 0:
        return c
    BLOCK = 4096
    _add_kernel[(triton.cdiv(n, BLOCK),)](a, b, c, n, BLOCK=BLOCK, num_warps=8)
    return c


IMPLS = {"blocked": vector_add}
