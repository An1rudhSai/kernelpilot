"""Evaluates one candidate kernel. Always run in its own subprocess (see
sandbox.py) so a hang, segfault or GPU fault in generated code cannot take down
the agent, and so each candidate starts from a clean CUDA context.

    python -m kernelpilot.agent.worker --kernel softmax --lang triton \
        --dtype float32 --code cand.py --out result.json
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
import traceback
import uuid
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from .result import EvalResult

CSRC = Path(__file__).resolve().parents[2] / "csrc"
MAX_LOG = 6000


def _tail(s: str, n: int = MAX_LOG) -> str:
    return s if len(s) <= n else "...(truncated)...\n" + s[-n:]


def _load_triton(code_path: Path):
    spec = importlib.util.spec_from_file_location(f"candidate_{uuid.uuid4().hex[:8]}", code_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, "kernel_fn"):
        raise AttributeError("candidate module does not define kernel_fn")
    return mod.kernel_fn, mod


def _load_hip(code_path: Path, arity: int, build_dir: Path):
    import torch
    from torch.utils.cpp_extension import load_inline

    decl = "torch::Tensor kernel_fn(" + ", ".join(f"torch::Tensor x{i}" for i in range(arity)) \
        + ");"
    source = '#include <torch/extension.h>\n#include "compat.h"\n' + code_path.read_text()
    if torch.version.hip:
        flags = ["-O3", "-Rpass-analysis=kernel-resource-usage"]
    else:
        flags = ["-O3", "-Xptxas=-v", "-lineinfo"]
    build_dir.mkdir(parents=True, exist_ok=True)
    ext = load_inline(name=f"kp_candidate_{uuid.uuid4().hex[:8]}", cpp_sources=decl,
                      cuda_sources=source, functions=["kernel_fn"],
                      extra_include_paths=[str(CSRC)], extra_cuda_cflags=flags,
                      build_directory=str(build_dir), verbose=True)
    return ext.kernel_fn, None


def _triton_metadata(mod) -> list[dict]:
    """Registers/spills/shared memory of compiled Triton kernels, best effort
    (the cache layout differs between Triton versions)."""
    out = []
    if mod is None:
        return out
    for name, obj in vars(mod).items():
        caches = getattr(obj, "device_caches", None)
        if not caches:
            continue
        try:
            for entry in caches.values():
                kcache = entry[0] if isinstance(entry, tuple) else entry
                for k in kcache.values():
                    md = getattr(k, "metadata", None)
                    out.append({
                        "kernel": name,
                        "n_regs": getattr(k, "n_regs", None),
                        "n_spills": getattr(k, "n_spills", None),
                        "shared_bytes": getattr(md, "shared", None),
                        "num_warps": getattr(md, "num_warps", None),
                    })
        except Exception:
            continue
    return out[:10]


def _profile(fn, inputs) -> list[dict]:
    import torch
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn(*inputs)
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages():
        t = getattr(e, "device_time_total", None) or getattr(e, "cuda_time_total", 0)
        if t:
            rows.append({"name": e.key[:120], "calls": e.count, "device_us": round(t, 2)})
    rows.sort(key=lambda r: -r["device_us"])
    return rows[:8]


def evaluate(kernel: str, lang: str, dtype_name: str, code_path: Path, build_dir: Path,
             bench_shapes: list[tuple] | None = None) -> EvalResult:
    import torch

    from ..benchmark import bench_one
    from ..correctness import check_impl
    from ..device import get_gpu_info
    from ..specs import get_spec

    t0 = time.monotonic()
    spec = get_spec(kernel)
    dtype = getattr(torch, dtype_name)
    if not torch.cuda.is_available():
        return EvalResult(stage="crash", error="no GPU available in the worker")

    # 1. Compile / load.
    log = io.StringIO()
    mod = None
    try:
        with redirect_stdout(log), redirect_stderr(log):
            if lang == "triton":
                fn, mod = _load_triton(code_path)
            else:
                arity = len(spec.make_inputs(spec.test_shapes[0], dtype, torch.device("cuda"), 0))
                fn, mod = _load_hip(code_path, arity, build_dir)
    except Exception as e:
        return EvalResult(stage="compile", error=_tail(f"{type(e).__name__}: {e}"),
                          compile_log=_tail(log.getvalue()), duration_s=time.monotonic() - t0)
    compile_log = _tail(log.getvalue(), 3000)

    # 2. Correctness on every test shape. Triton compiles lazily on first launch,
    #    so a Triton compilation error surfaces here and is re-labelled.
    passed = 0
    for shape in spec.test_shapes:
        res = check_impl(spec, "candidate", fn, shape, dtype, "cuda")
        if res.error and "Compil" in res.error.split(":", 1)[0]:
            return EvalResult(stage="compile", error=_tail(res.error), compile_log=compile_log,
                              duration_s=time.monotonic() - t0)
        if not res.ok:
            return EvalResult(stage="correctness", error=res.summary(), failures=[res.to_dict()],
                              checks_passed=passed, compile_log=compile_log,
                              duration_s=time.monotonic() - t0)
        passed += 1

    # 3. Benchmark against PyTorch on the benchmark shapes.
    gpu = get_gpu_info()
    rows = []
    try:
        for shape in bench_shapes or spec.bench_shapes:
            cand = bench_one(spec, "candidate", fn, shape, dtype, gpu, verify=True)
            if not cand.correct:  # bench shapes are also verified, never trusted
                return EvalResult(stage="correctness", error=cand.note, checks_passed=passed,
                                  compile_log=compile_log, duration_s=time.monotonic() - t0)
            base = bench_one(spec, "torch", spec.baseline, shape, dtype, gpu, verify=False)
            rows.append({
                "shape": list(shape), "ms": cand.ms, "torch_ms": base.ms,
                "speedup": base.ms / cand.ms, "gbs": cand.gbs, "tflops": cand.tflops,
                "pct_peak_bw": cand.pct_peak_bw, "pct_peak_flops": cand.pct_peak_flops,
            })
    except Exception:
        return EvalResult(stage="benchmark", error=_tail(traceback.format_exc()),
                          checks_passed=passed, compile_log=compile_log,
                          duration_s=time.monotonic() - t0)

    # 4. Profile one call at the first benchmark shape.
    profile_info: dict = {}
    try:
        shape = (bench_shapes or spec.bench_shapes)[0]
        inputs = spec.make_inputs(shape, dtype, torch.device("cuda"), 0)
        profile_info["gpu_kernels"] = _profile(fn, inputs)
        profile_info["triton"] = _triton_metadata(mod)
        profile_info["gpu"] = gpu.describe()
    except Exception as e:
        profile_info["error"] = f"{type(e).__name__}: {e}"

    return EvalResult(stage="ok", checks_passed=passed, bench=rows, profile=profile_info,
                      compile_log=compile_log, duration_s=time.monotonic() - t0)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--kernel", required=True)
    p.add_argument("--lang", choices=["triton", "hip"], required=True)
    p.add_argument("--dtype", default="float32")
    p.add_argument("--code", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--build-dir", default=None)
    p.add_argument("--bench-shapes", default=None, help="JSON list of shapes")
    a = p.parse_args(argv)
    code = Path(a.code)
    shapes = [tuple(s) for s in json.loads(a.bench_shapes)] if a.bench_shapes else None
    try:
        res = evaluate(a.kernel, a.lang, a.dtype, code,
                       Path(a.build_dir or code.parent / "build"), shapes)
    except Exception:
        res = EvalResult(stage="crash", error=_tail(traceback.format_exc()))
    Path(a.out).write_text(json.dumps(res.to_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
