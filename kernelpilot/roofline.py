"""Roofline plot: where each benchmarked kernel sits relative to the GPU's
memory-bandwidth and compute ceilings."""

from __future__ import annotations

from pathlib import Path

from .benchmark import BenchResult
from .device import GPUInfo

# Fixed categorical order: a kernel keeps its color no matter which subset is plotted.
KERNEL_COLORS = {
    "vector_add": "#2a78d6",
    "reduce_sum": "#eb6834",
    "softmax": "#1baf7a",
    "matmul": "#eda100",
    "layernorm": "#e87ba4",
    "attention": "#6250d6",
}
# Implementation family is encoded by marker shape, so identity is never color alone.
FAMILY_MARKERS = {"torch": "o", "hip": "s", "triton": "^", "agent": "D"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _family(impl: str) -> str:
    return impl.split(":", 1)[0]


def plot_roofline(results: list[BenchResult], gpu: GPUInfo, out_path: str | Path,
                  title: str | None = None) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pts = [r for r in results if r.correct and r.ms and r.arithmetic_intensity and r.tflops]
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    ax.set_xscale("log")
    ax.set_yscale("log")

    xs = [r.arithmetic_intensity for r in pts] or [0.1, 100]
    x_lo, x_hi = min(xs) / 3, max(xs) * 3
    if gpu.peak_bw_gbs:
        bw_tbs = gpu.peak_bw_gbs / 1e3
        grid_x = [x_lo * (x_hi / x_lo) ** (i / 200) for i in range(201)]
        for dtype_key, style in (("float32", "-"), ("float16", "--")):
            peak = gpu.peak_tflops.get(dtype_key)
            if not peak:
                continue
            ys = [min(peak, bw_tbs * x) for x in grid_x]
            ax.plot(grid_x, ys, style, color=MUTED, linewidth=2)
            ax.annotate(f"{dtype_key} peak {peak:.0f} TFLOPS", xy=(grid_x[-1], peak),
                        xytext=(-4, 4), textcoords="offset points", ha="right",
                        fontsize=8, color=MUTED)
        # Label the bandwidth slope halfway (in log space) to the fp32 ridge point.
        ridge_x = gpu.peak_tflops.get("float32", bw_tbs * x_hi) / bw_tbs
        lx = (x_lo * min(ridge_x, x_hi)) ** 0.5
        ax.annotate(f"memory bound: {gpu.peak_bw_gbs:.0f} GB/s", xy=(lx, bw_tbs * lx),
                    xytext=(-8, 8), textcoords="offset points", ha="right", fontsize=8,
                    color=MUTED)

    for r in pts:
        ax.scatter(r.arithmetic_intensity, r.tflops, s=64,
                   color=KERNEL_COLORS.get(r.kernel, MUTED),
                   marker=FAMILY_MARKERS.get(_family(r.impl), "o"),
                   edgecolors="#fcfcfb", linewidths=2, zorder=3)

    # Label only the best implementation per kernel (selective direct labels).
    best: dict[str, BenchResult] = {}
    for r in pts:
        if r.kernel not in best or r.tflops > best[r.kernel].tflops:
            best[r.kernel] = r
    for r in best.values():
        ax.annotate(f"{r.kernel} ({r.impl}, {r.dtype})", xy=(r.arithmetic_intensity, r.tflops),
                    xytext=(6, -10), textcoords="offset points", fontsize=8, color=INK)

    ax.set_xlabel("Arithmetic intensity (FLOP / byte, ideal traffic)", color=INK)
    ax.set_ylabel("Achieved TFLOPS", color=INK)
    ax.set_title(title or f"Roofline: {gpu.name}", color=INK, loc="left")
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)

    order = list(KERNEL_COLORS)
    kernels = sorted({r.kernel for r in pts},
                     key=lambda k: order.index(k) if k in order else len(order))
    handles = [Line2D([], [], marker="o", linestyle="", color=KERNEL_COLORS.get(k, MUTED),
                      label=k) for k in kernels]
    handles += [Line2D([], [], marker=m, linestyle="", color=MUTED, label=f"{fam} impl")
                for fam, m in FAMILY_MARKERS.items() if any(_family(r.impl) == fam for r in pts)]
    if handles:
        ax.legend(handles=handles, loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out
