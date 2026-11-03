import torch
import triton
import triton.language as tl


@triton.jit
def _sum_kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
    # Persistent-style: each program accumulates a strided set of blocks in a
    # BLOCK-wide fp32 register vector, reduces it once, and does one atomic.
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for start in range(pid * BLOCK, n, nprog * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(x_ptr + offs, mask=offs < n, other=0.0).to(tl.float32)
    tl.atomic_add(out_ptr, tl.sum(acc, axis=0))


def _max_programs(device: torch.device) -> int:
    if device.type == "cuda":
        return 4 * torch.cuda.get_device_properties(device).multi_processor_count
    return 64


def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros((), dtype=torch.float32, device=x.device)
    n = x.numel()
    if n == 0:
        return out
    BLOCK = 2048
    grid = (min(triton.cdiv(n, BLOCK), _max_programs(x.device)),)
    _sum_kernel[grid](x, out, n, BLOCK=BLOCK, num_warps=8)
    return out


IMPLS = {"persistent_atomic": reduce_sum}
