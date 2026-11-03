"""Benchmark harness: GPU-event timing with warm-up, L2 flushing and the median
of N iterations; reports time, GB/s, TFLOPS and % of the GPU's theoretical peak."""

from __future__ import annotations

import json
import statistics
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .correctness import check_impl
from .device import GPUInfo, get_gpu_info
from .specs import KernelSpec, Shape

_L2_FLUSH_BYTES = 256 << 20  # larger than the L2 / Infinity Cache of any target GPU
_flush_buf: torch.Tensor | None = None


def _flush_l2() -> None:
    global _flush_buf
    if _flush_buf is None:
        _flush_buf = torch.empty(_L2_FLUSH_BYTES // 4, dtype=torch.float32, device="cuda")
    _flush_buf.zero_()


def time_fn(fn: Callable, args: tuple, warmup: int = 10, iters: int = 100,
            budget_s: float = 3.0, flush_l2: bool = True) -> list[float]:
    """Per-iteration GPU time in milliseconds.

    Each iteration is bracketed by its own pair of events so the L2 flush (done
    before the start event) is excluded. Iterations are capped so that slow
    kernels (e.g. a naive matmul) stay within roughly `budget_s` seconds.
    """
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    # Estimate a single call to bound the iteration count.
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn(*args)
    end.record()
    torch.cuda.synchronize()
    est_ms = max(start.elapsed_time(end), 1e-3)
    n = max(5, min(iters, int(budget_s * 1e3 / est_ms)))

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(n)]
    for i in range(n):
        if flush_l2:
            _flush_l2()
        starts[i].record()
        fn(*args)
        ends[i].record()
    torch.cuda.synchronize()
    return [s.elapsed_time(e) for s, e in zip(starts, ends, strict=True)]


@dataclass
class BenchResult:
    kernel: str
    impl: str
    dtype: str
    shape: Shape
    ms: float | None
    gbs: float | None = None
    tflops: float | None = None
    pct_peak_bw: float | None = None
    pct_peak_flops: float | None = None
    arithmetic_intensity: float | None = None
    speedup_vs_torch: float | None = None
    correct: bool = True
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def bench_one(spec: KernelSpec, impl: str, fn: Callable, shape: Shape, dtype: torch.dtype,
              gpu: GPUInfo, warmup: int = 10, iters: int = 100, verify: bool = True
              ) -> BenchResult:
    dname = str(dtype).replace("torch.", "")
    if verify:
        chk = check_impl(spec, impl, fn, shape, dtype, "cuda")
        if not chk.ok:
            return BenchResult(spec.name, impl, dname, tuple(shape), ms=None, correct=False,
                               note=chk.summary())
    inputs = spec.make_inputs(shape, dtype, torch.device("cuda"), 0)
    times = time_fn(fn, inputs, warmup=warmup, iters=iters)
    ms = statistics.median(times)
    flops = spec.flops(shape)
    nbytes = spec.bytes_moved(shape, dtype)
    gbs = nbytes / (ms / 1e3) / 1e9
    tflops = flops / (ms / 1e3) / 1e12
    peak_fl = gpu.peak_for(dtype)
    return BenchResult(
        spec.name, impl, dname, tuple(shape), ms=ms, gbs=gbs, tflops=tflops,
        pct_peak_bw=100 * gbs / gpu.peak_bw_gbs if gpu.peak_bw_gbs else None,
        pct_peak_flops=100 * tflops / peak_fl if peak_fl else None,
        arithmetic_intensity=flops / nbytes,
    )


def bench_kernel(spec: KernelSpec, impls: dict[str, Callable],
                 shapes: list[Shape] | None = None, dtypes: list[torch.dtype] | None = None,
                 gpu: GPUInfo | None = None, **kw) -> list[BenchResult]:
    gpu = gpu or get_gpu_info()
    results: list[BenchResult] = []
    for dtype in dtypes or spec.dtypes:
        for shape in shapes or spec.bench_shapes:
            row = [bench_one(spec, name, fn, shape, dtype, gpu, **kw)
                   for name, fn in impls.items()]
            base = next((r for r in row if r.impl == "torch" and r.ms), None)
            for r in row:
                if base and r.ms:
                    r.speedup_vs_torch = base.ms / r.ms
            results.extend(row)
    return results


def _fmt(v: float | None, spec: str) -> str:
    return "–" if v is None else format(v, spec)


def to_markdown(results: list[BenchResult], gpu: GPUInfo | None = None) -> str:
    lines: list[str] = []
    if gpu:
        lines += [f"GPU: {gpu.describe()}", ""]
    by_kernel: dict[str, list[BenchResult]] = {}
    for r in results:
        by_kernel.setdefault(r.kernel, []).append(r)
    for kernel, rows in by_kernel.items():
        lines += [f"### {kernel}", "",
                  "| impl | dtype | shape | time (ms) | GB/s | TFLOPS | % peak BW "
                  "| % peak FLOPS | vs torch |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if not r.correct:
                lines.append(f"| {r.impl} | {r.dtype} | {r.shape} | FAILED correctness "
                             "| | | | | |")
                continue
            lines.append(
                f"| {r.impl} | {r.dtype} | {r.shape} | {_fmt(r.ms, '.4f')} | "
                f"{_fmt(r.gbs, '.1f')} | {_fmt(r.tflops, '.3f')} | "
                f"{_fmt(r.pct_peak_bw, '.1f')} | {_fmt(r.pct_peak_flops, '.1f')} | "
                f"{_fmt(r.speedup_vs_torch, '.2f')}x |")
        lines.append("")
    return "\n".join(lines)


def save_results(results: list[BenchResult], gpu: GPUInfo, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"gpu": asdict(gpu), "results": [r.to_dict() for r in results]}
    (out / "bench.json").write_text(json.dumps(payload, indent=2))
    (out / "bench.md").write_text(to_markdown(results, gpu))


def load_results(path: str | Path) -> tuple[GPUInfo, list[BenchResult]]:
    payload = json.loads(Path(path).read_text())
    gpu = GPUInfo(**payload["gpu"])
    results = []
    for d in payload["results"]:
        d["shape"] = tuple(d["shape"])
        results.append(BenchResult(**d))
    return gpu, results
