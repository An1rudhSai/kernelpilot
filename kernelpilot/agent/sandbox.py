"""Runs a candidate kernel in a separate Python process with a hard timeout.

This is isolation, not a security sandbox: the subprocess can still touch the
filesystem and network. It protects the agent from hangs, crashes and GPU
faults in generated code, and gives every candidate a fresh CUDA context.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from .result import EvalResult

REPO_ROOT = Path(__file__).resolve().parents[2]


def _tail(s: str, n: int = 4000) -> str:
    return s if len(s) <= n else "...(truncated)...\n" + s[-n:]


def run_candidate(code: str, kernel: str, lang: str, dtype: str, timeout_s: float = 300.0,
                  bench_shapes: list[tuple] | None = None, workdir: Path | None = None
                  ) -> EvalResult:
    tmp = Path(workdir or tempfile.mkdtemp(prefix="kp_cand_"))
    tmp.mkdir(parents=True, exist_ok=True)
    code_path = tmp / ("candidate.py" if lang == "triton" else "candidate.hip")
    code_path.write_text(code)
    out_path = tmp / "result.json"
    cmd = [sys.executable, "-m", "kernelpilot.agent.worker", "--kernel", kernel, "--lang", lang,
           "--dtype", dtype, "--code", str(code_path), "--out", str(out_path),
           "--build-dir", str(tmp / "build")]
    if bench_shapes:
        cmd += ["--bench-shapes", json.dumps([list(s) for s in bench_shapes])]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPO_ROOT), env.get("PYTHONPATH")]))
    env.setdefault("TORCH_EXTENSIONS_DIR", str(tmp / "torch_ext"))
    t0 = time.monotonic()
    try:
        proc = subprocess.run(cmd, cwd=tmp, env=env, capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) \
            else (e.stdout or "")
        return EvalResult(stage="timeout",
                          error=f"candidate did not finish within {timeout_s:.0f}s "
                                "(infinite loop, deadlock, or far too slow)",
                          compile_log=_tail(out), duration_s=time.monotonic() - t0)
    output = _tail((proc.stdout or "") + (proc.stderr or ""))
    if not out_path.exists():
        return EvalResult(stage="crash",
                          error=f"worker exited with code {proc.returncode} without a result "
                                f"(segfault or GPU fault?)\n{output}",
                          duration_s=time.monotonic() - t0)
    res = EvalResult.from_dict(json.loads(out_path.read_text()))
    if not res.compile_log:
        res.compile_log = output
    res.duration_s = time.monotonic() - t0
    return res
