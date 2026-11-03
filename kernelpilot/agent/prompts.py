"""Prompt construction. The model sees: the spec (reference code, shapes,
dtype), the target GPU, the contract its code must satisfy, the best kernel so
far with its numbers, and the most recent attempts with their compiler errors,
correctness failures, timings and profiler output."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from ..specs import KernelSpec
from .result import EvalResult

SYSTEM = """You are an expert GPU kernel engineer. You write fast, correct kernels in \
{lang_name} and improve them using benchmark and profiler feedback.

Rules:
- Reply with a short plan (2-5 sentences: what you will change and why it should be \
faster), then exactly ONE fenced code block containing the complete source file.
- Correctness is checked against a float64 PyTorch reference on many shapes, including \
size 1, non-power-of-2 sizes and large sizes. An incorrect kernel is always rejected, \
however fast it is.
- Do not call the PyTorch operator being implemented (or any library that implements \
it) to produce the result; the work must be done by your own kernel(s). Allocating \
outputs with torch.empty/zeros is fine.
- Never modify the input tensors.
{contract}"""

TRITON_CONTRACT = """- Language: Triton (Python). Import torch, triton and triton.language as tl.
- Define `kernel_fn` with this exact signature, launching your @triton.jit kernel(s):
    {signature}
- Inputs are contiguous CUDA tensors of dtype {dtype}. Return a new tensor.
- Use tl.dot for matrix products (not the @ operator). For float32 tl.dot, pass \
input_precision="ieee" (TF32 is not accurate enough)."""

HIP_CONTRACT = """- Language: HIP C++ (C++17). The file is compiled with PyTorch's \
cpp_extension.load_inline. These are already included before your code: \
<torch/extension.h> and "compat.h" (which includes the HIP or CUDA runtime and fp16 \
headers and defines KP_SHFL_DOWN/KP_SHFL_XOR(v, delta) warp shuffles; use `warpSize`, \
never hard-code 32 or 64).
- Use HIP-style names (hipStream_t etc.) and <<<grid, block, smem, stream>>> launches.
- Define this host function (all arguments are torch::Tensor, in this order):
    torch::Tensor kernel_fn({hip_args})
  Python-side semantics: {signature}
- Inputs are contiguous GPU tensors of dtype {dtype} (at::Half for float16; \
reinterpret data_ptr as __half*). Launch on the current stream: include \
<ATen/cuda/CUDAContext.h> and use `at::cuda::getCurrentCUDAStream()` (PyTorch maps this \
to HIP on ROCm builds)."""


def system_prompt(spec: KernelSpec, lang: str, dtype: str) -> str:
    if lang == "triton":
        contract = TRITON_CONTRACT.format(signature=spec.signature, dtype=dtype)
        lang_name = "Triton"
    else:
        n = spec.signature.split("->")[0].count("torch.Tensor")
        contract = HIP_CONTRACT.format(
            signature=spec.signature, dtype=dtype,
            hip_args=", ".join(f"torch::Tensor x{i}" for i in range(n)))
        lang_name = "HIP C++"
    return SYSTEM.format(lang_name=lang_name, contract=contract)


@dataclass
class AttemptView:
    """What the prompt needs to know about a past attempt."""

    iteration: int
    code: str
    result: EvalResult


def _fmt_bench(res: EvalResult) -> str:
    lines = []
    for r in res.bench:
        pct = []
        if r.get("pct_peak_bw") is not None:
            pct.append(f"{r['pct_peak_bw']:.0f}% peak BW")
        if r.get("pct_peak_flops") is not None:
            pct.append(f"{r['pct_peak_flops']:.0f}% peak FLOPS")
        lines.append(f"  shape={tuple(r['shape'])}: {r['ms']:.4f} ms vs PyTorch "
                     f"{r['torch_ms']:.4f} ms -> {r['speedup']:.2f}x; {r['gbs']:.0f} GB/s, "
                     f"{r['tflops']:.2f} TFLOPS ({', '.join(pct) or 'peak unknown'})")
    return "\n".join(lines)


def _fmt_profile(res: EvalResult) -> str:
    p = res.profile or {}
    parts = []
    if p.get("gpu_kernels"):
        parts.append("GPU kernels launched for one call (torch.profiler):\n" + "\n".join(
            f"  {k['name']}: {k['calls']} call(s), {k['device_us']} us"
            for k in p["gpu_kernels"]))
    if p.get("triton"):
        parts.append("Compiled Triton kernels: " + json.dumps(p["triton"]))
    return "\n".join(parts)


def _fmt_result(res: EvalResult) -> str:
    if res.stage == "ok":
        out = f"Correct on all {res.checks_passed} test shapes. Score (geomean speedup vs " \
              f"PyTorch): {res.score:.3f}x\n{_fmt_bench(res)}"
        prof = _fmt_profile(res)
        if prof:
            out += "\n" + prof
        if res.compile_log and ("ptxas" in res.compile_log or "VGPRs" in res.compile_log):
            out += "\nCompiler resource usage:\n" + _tail_lines(res.compile_log, 25)
        return out
    label = {
        "no_code": "No code block found in your reply.",
        "rejected": "REJECTED before running.",
        "compile": "COMPILE ERROR.",
        "correctness": "WRONG RESULT (rejected).",
        "benchmark": "Crashed while benchmarking.",
        "timeout": "TIMEOUT.",
        "crash": "Process crashed.",
    }.get(res.stage, res.stage)
    out = f"{label}\n{res.error or ''}"
    if res.stage == "correctness" and res.checks_passed:
        out += f"\n(It passed {res.checks_passed} smaller shapes before failing.)"
    if res.stage == "compile" and res.compile_log and res.compile_log not in (res.error or ""):
        out += "\nCompiler output:\n" + _tail_lines(res.compile_log, 60)
    return out


def _tail_lines(s: str, n: int) -> str:
    return "\n".join(s.strip().splitlines()[-n:])


def user_prompt(spec: KernelSpec, lang: str, dtype: str, gpu_desc: str,
                history: list[AttemptView], best: AttemptView | None,
                bench_shapes: list[tuple], recent: int = 3) -> str:
    fence = "python" if lang == "triton" else "cpp"
    parts = [
        f"## Task: {spec.name}",
        spec.description,
        f"Target GPU: {gpu_desc}",
        f"dtype: {dtype}",
        "Reference semantics (evaluated in float64):",
        f"```python\n{spec.reference_source().strip()}\n```",
        "Correctness shapes (" + ", ".join(spec.shape_names) + "): "
        + ", ".join(str(s) for s in spec.test_shapes),
        "Benchmark shapes (what you are optimizing for): "
        + ", ".join(str(tuple(s)) for s in bench_shapes),
    ]
    if best is not None:
        parts += [f"## Best correct kernel so far (attempt {best.iteration})",
                  _fmt_result(best.result), f"```{fence}\n{best.code.strip()}\n```"]
    tail = [a for a in history if best is None or a.iteration != best.iteration][-recent:]
    if tail:
        parts.append("## Most recent attempts")
        for a in tail:
            parts.append(f"### Attempt {a.iteration}\n{_fmt_result(a.result)}")
        last = history[-1]
        if not last.result.ok and (best is None or last.iteration != best.iteration):
            parts.append(f"Code of attempt {last.iteration} (fix it, or take another "
                         f"approach):\n```{fence}\n{last.code.strip()}\n```")
    if not history:
        parts.append("## Instructions\nWrite a correct, reasonably fast first version.")
    elif best is None:
        parts.append("## Instructions\nNo correct kernel yet. Fix the errors above first; "
                     "correctness before speed.")
    else:
        parts.append("## Instructions\nWrite a kernel that beats the best one above. Use the "
                     "benchmark and profiler numbers: decide whether it is memory or compute "
                     "bound and attack that limit.")
    return "\n\n".join(parts)


_FENCE = re.compile(r"```([a-zA-Z0-9_+\-]*)[ \t]*\n(.*?)```", re.DOTALL)
_LANG_TAGS = {"triton": {"python", "py", "triton"},
              "hip": {"cpp", "c++", "hip", "cuda", "cu", "c"}}


def extract_code(text: str, lang: str) -> str | None:
    """Return the code block for `lang` (the longest one if there are several),
    falling back to an untagged block. None if the reply has no code."""
    blocks = [(tag.lower(), body) for tag, body in _FENCE.findall(text)]
    tagged = [b for t, b in blocks if t in _LANG_TAGS[lang]]
    pool = tagged or [b for t, b in blocks if not t]
    if not pool:
        return None
    code = max(pool, key=len).strip()
    return code or None
