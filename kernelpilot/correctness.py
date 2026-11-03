"""Correctness harness: compare a kernel against the high-precision PyTorch
reference and fail loudly, with max absolute/relative error, on any mismatch."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from .specs import KernelSpec, Shape


class KernelMismatch(AssertionError):
    pass


@dataclass
class CheckResult:
    kernel: str
    impl: str
    shape: Shape
    dtype: str
    ok: bool
    max_abs_err: float = 0.0
    max_rel_err: float = 0.0
    num_mismatched: int = 0
    numel: int = 0
    worst_index: tuple[int, ...] | None = None
    got: float | None = None
    expected: float | None = None
    atol: float = 0.0
    rtol: float = 0.0
    error: str | None = None

    def summary(self) -> str:
        head = f"{self.kernel}[{self.impl}] dtype={self.dtype} shape={self.shape}"
        if self.error:
            return f"{head}: {self.error}"
        status = "OK" if self.ok else "MISMATCH"
        msg = (f"{head}: {status} max_abs_err={self.max_abs_err:.3e} "
               f"max_rel_err={self.max_rel_err:.3e} (atol={self.atol:.1e}, rtol={self.rtol:.1e})")
        if not self.ok and self.worst_index is not None:
            msg += (f"; {self.num_mismatched}/{self.numel} elements out of tolerance, worst at "
                    f"index {self.worst_index}: got {self.got!r}, expected {self.expected!r}")
        return msg

    def to_dict(self) -> dict:
        return asdict(self)


def compare(out: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> dict:
    """Elementwise |out - ref| <= atol + rtol * |ref|, with NaN/Inf treated as errors
    unless the reference has the same non-finite value at that position."""
    if tuple(out.shape) != tuple(ref.shape):
        raise KernelMismatch(f"shape mismatch: got {tuple(out.shape)}, "
                             f"expected {tuple(ref.shape)}")
    o = out.detach().to(torch.float64)
    r = ref.detach().to(torch.float64)
    diff = (o - r).abs()
    same_nonfinite = (~torch.isfinite(r)) & (o == r)
    diff = torch.where(same_nonfinite, torch.zeros_like(diff), diff)
    diff = torch.where(torch.isnan(diff), torch.full_like(diff, float("inf")), diff)
    bad = diff > (atol + rtol * r.abs())
    rel = diff / r.abs().clamp_min(1e-12)
    info = {
        "max_abs_err": float(diff.max()) if diff.numel() else 0.0,
        "max_rel_err": float(rel.max()) if rel.numel() else 0.0,
        "num_mismatched": int(bad.sum()),
        "numel": int(diff.numel()),
    }
    if info["num_mismatched"]:
        flat = int(torch.argmax(torch.where(bad, diff, torch.zeros_like(diff))).item())
        idx = tuple(int(i) for i in torch.unravel_index(torch.tensor(flat), diff.shape)) \
            if diff.dim() else ()
        info.update(worst_index=idx, got=float(o.reshape(-1)[flat]),
                    expected=float(r.reshape(-1)[flat]))
    return info


def check_impl(spec: KernelSpec, impl_name: str, fn: Callable, shape: Shape,
               dtype: torch.dtype, device: str | torch.device = "cuda",
               seed: int = 0) -> CheckResult:
    device = torch.device(device)
    inputs = spec.make_inputs(shape, dtype, device, seed)
    tol = spec.tolerance(shape, dtype)
    res = CheckResult(kernel=spec.name, impl=impl_name, shape=tuple(shape),
                      dtype=str(dtype).replace("torch.", ""), ok=False,
                      atol=tol.atol, rtol=tol.rtol)
    snapshot = [x.clone() for x in inputs]
    try:
        out = fn(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    except Exception as e:  # a crash is a failed check, reported, not raised
        res.error = f"{type(e).__name__}: {e}"
        return res
    if not isinstance(out, torch.Tensor):
        res.error = f"kernel returned {type(out).__name__}, expected torch.Tensor"
        return res
    expected_dtype = spec.output_dtype(dtype)
    if out.dtype != expected_dtype:
        res.error = f"kernel returned dtype {out.dtype}, expected {expected_dtype}"
        return res
    if any(not torch.equal(a, b) for a, b in zip(inputs, snapshot, strict=True)):
        res.error = "kernel modified its inputs in place"
        return res
    ref = spec.reference(*inputs)
    try:
        info = compare(out, ref, tol.atol, tol.rtol)
    except KernelMismatch as e:
        res.error = str(e)
        return res
    for k, v in info.items():
        setattr(res, k, v)
    res.ok = info["num_mismatched"] == 0
    return res


def assert_correct(spec: KernelSpec, impl_name: str, fn: Callable, shape: Shape,
                   dtype: torch.dtype, device: str | torch.device = "cuda") -> CheckResult:
    res = check_impl(spec, impl_name, fn, shape, dtype, device)
    if not res.ok:
        raise KernelMismatch(res.summary())
    return res


def check_all(spec: KernelSpec, impl_name: str, fn: Callable,
              shapes: list[Shape] | None = None, dtypes: list[torch.dtype] | None = None,
              device: str | torch.device = "cuda", stop_on_fail: bool = False
              ) -> list[CheckResult]:
    results = []
    for dtype in dtypes or spec.dtypes:
        for shape in shapes or spec.test_shapes:
            r = check_impl(spec, impl_name, fn, shape, dtype, device)
            results.append(r)
            if stop_on_fail and not r.ok:
                return results
    return results
