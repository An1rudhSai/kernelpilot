"""The agent loop:

    for i in 1..N:
        ask the LLM for a kernel (spec + best so far + recent attempts + feedback)
        statically reject code that just calls the PyTorch op
        compile -> correctness -> benchmark -> profile, in a subprocess with a timeout
        log the attempt to JSONL; keep it only if correct and faster than the best

A kernel that fails correctness is never accepted, however fast it is: only
EvalResult.stage == "ok" (all correctness checks passed, including on the
benchmark shapes themselves) can become the best kernel.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from ..specs import KernelSpec, get_spec
from .llm import LLM
from .prompts import AttemptView, extract_code, system_prompt, user_prompt
from .result import EvalResult
from .sandbox import run_candidate

# Calls that would make the "kernel" a thin wrapper around PyTorch's own
# implementation. A cheap textual guard; the profiler output in the log lets a
# human double-check which GPU kernels actually ran.
_MATMUL_OP = r"[\w\)\]][ \t]*@[ \t]*[\w\(]"  # the @ operator, not a decorator
FORBIDDEN: dict[str, list[str]] = {
    "vector_add": [r"torch\.add\b", r"(?<!tl)\.add_?\("],
    "reduce_sum": [r"torch\.(sum|cumsum)\b", r"(?<!tl)\.sum\("],
    "softmax": [r"torch\.(log_)?softmax\b", r"\bF\.(log_)?softmax\b",
                r"functional\.(log_)?softmax\b", r"(?<!tl)\.(log_)?softmax\("],
    "matmul": [r"torch\.(matmul|mm|bmm|einsum|addmm|tensordot)\b", r"(?<!tl)\.(matmul|mm)\(",
               r"\bF\.linear\b", r"functional\.linear\b", _MATMUL_OP],
    "layernorm": [r"layer_norm", r"LayerNorm", r"torch\.(var|std|var_mean)\b"],
    "attention": [r"scaled_dot_product_attention", r"torch\.(matmul|bmm|einsum|softmax)\b",
                  r"\bF\.softmax\b", r"(?<!tl)\.(matmul|softmax)\(", _MATMUL_OP],
}

Evaluator = Callable[[str], EvalResult]


def forbidden_call(kernel: str, code: str, lang: str) -> str | None:
    if lang != "triton":
        # HIP sources go through C++; only flag obvious ATen calls.
        hits = re.findall(r"\bat::(matmul|mm|softmax|layer_norm|sum|add|scaled_dot_product)",
                          code)
        return f"calls at::{hits[0]}" if hits else None
    for pat in FORBIDDEN.get(kernel, []):
        m = re.search(pat, _strip_comments(code))
        if m:
            return f"uses `{m.group(0).strip()}`, which delegates to PyTorch"
    return None


def _strip_comments(code: str) -> str:
    code = re.sub(r'"""[\s\S]*?"""', "", code)
    return re.sub(r"#.*", "", code)


@dataclass
class Attempt:
    iteration: int
    code: str | None
    reply: str
    result: EvalResult
    llm_latency_s: float
    is_best: bool = False
    is_regression: bool = False  # correct, but slower than the best at that time


@dataclass
class AgentRun:
    run_id: str
    kernel: str
    dtype: str
    lang: str
    model: str
    attempts: list[Attempt] = field(default_factory=list)
    best: Attempt | None = None

    @property
    def best_score(self) -> float | None:
        return self.best.result.score if self.best else None

    def iterations_to_best(self) -> int | None:
        return self.best.iteration if self.best else None


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")


class Agent:
    def __init__(self, llm: LLM, kernel: str, lang: str = "triton", dtype: str = "float32",
                 log_path: str | Path | None = None, timeout_s: float = 300.0,
                 evaluator: Evaluator | None = None, gpu_desc: str | None = None,
                 bench_shapes: list[tuple] | None = None, verbose: bool = True):
        self.llm = llm
        self.spec: KernelSpec = get_spec(kernel)
        self.lang = lang
        self.dtype = dtype
        self.timeout_s = timeout_s
        self.bench_shapes = bench_shapes or list(self.spec.bench_shapes)
        self.evaluator = evaluator or self._sandbox_eval
        self.gpu_desc = gpu_desc or _gpu_description()
        self.verbose = verbose
        self.run = AgentRun(run_id=uuid.uuid4().hex[:12], kernel=kernel, dtype=dtype, lang=lang,
                            model=getattr(llm, "model", "?"))
        default_log = Path("runs") / f"{kernel}_{lang}_{dtype}_{self.run.run_id}.jsonl"
        self.logger = JsonlLogger(log_path or default_log)

    def _sandbox_eval(self, code: str) -> EvalResult:
        return run_candidate(code, self.spec.name, self.lang, self.dtype, self.timeout_s,
                             self.bench_shapes)

    def _say(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)

    def _messages(self) -> list[dict]:
        history = [AttemptView(a.iteration, a.code or "", a.result) for a in self.run.attempts]
        best = self.run.best
        best_view = AttemptView(best.iteration, best.code or "", best.result) if best else None
        return [
            {"role": "system", "content": system_prompt(self.spec, self.lang, self.dtype)},
            {"role": "user", "content": user_prompt(self.spec, self.lang, self.dtype,
                                                    self.gpu_desc, history, best_view,
                                                    self.bench_shapes)},
        ]

    def step(self, iteration: int) -> Attempt:
        messages = self._messages()
        try:
            completion = self.llm.complete(messages)
            reply, latency = completion.text, completion.latency_s
        except Exception as e:
            reply, latency = "", 0.0
            res = EvalResult(stage="crash", error=f"LLM call failed: {e}")
            return self._record(Attempt(iteration, None, reply, res, latency))

        code = extract_code(reply, self.lang)
        if code is None:
            res = EvalResult(stage="no_code",
                             error="Reply contained no fenced code block for this language.")
        elif why := forbidden_call(self.spec.name, code, self.lang):
            res = EvalResult(stage="rejected", error=f"Candidate {why}. Implement the "
                             "computation in your own kernel.")
        else:
            res = self.evaluator(code)
        return self._record(Attempt(iteration, code, reply, res, latency))

    def _record(self, att: Attempt) -> Attempt:
        prev_best = self.run.best
        if att.result.ok and att.result.score is not None:
            if prev_best is None or att.result.score > prev_best.result.score:
                att.is_best = True
                self.run.best = att
            else:
                att.is_regression = True
        self.run.attempts.append(att)
        self.logger.write({
            "run_id": self.run.run_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "kernel": self.spec.name, "dtype": self.dtype, "lang": self.lang,
            "model": self.run.model, "iteration": att.iteration,
            "stage": att.result.stage, "ok": att.result.ok, "score": att.result.score,
            "is_best": att.is_best, "is_regression": att.is_regression,
            "best_score_so_far": self.run.best_score,
            "error": att.result.error, "failures": att.result.failures,
            "checks_passed": att.result.checks_passed, "bench": att.result.bench,
            "profile": att.result.profile, "compile_log": att.result.compile_log,
            "eval_duration_s": att.result.duration_s, "llm_latency_s": att.llm_latency_s,
            "code": att.code, "reply": att.reply,
        })
        score = f"{att.result.score:.3f}x" if att.result.score else "-"
        tag = " NEW BEST" if att.is_best else (" (regression)" if att.is_regression else "")
        err = "" if att.result.ok else f": {(att.result.error or '').splitlines()[0][:120]}"
        self._say(f"[{att.iteration:>2}] {att.result.stage:<11} score={score}{tag}{err}")
        return att

    def optimize(self, iterations: int = 10) -> AgentRun:
        self._say(f"optimizing {self.spec.name} ({self.lang}, {self.dtype}) with "
                  f"{self.run.model} for {iterations} iterations; log: {self.logger.path}")
        t0 = time.monotonic()
        for i in range(1, iterations + 1):
            self.step(i)
        best = self.run.best
        if best:
            self._say(f"best: attempt {best.iteration}, {best.result.score:.3f}x vs PyTorch "
                      f"({time.monotonic() - t0:.0f}s total)")
        else:
            self._say("no correct kernel found")
        return self.run


def _gpu_description() -> str:
    try:
        from ..device import get_gpu_info

        return get_gpu_info().describe()
    except Exception:
        return "unknown GPU"


def save_best(run: AgentRun, out_dir: str | Path) -> Path | None:
    if not run.best or not run.best.code:
        return None
    ext = "py" if run.lang == "triton" else "hip"
    out = Path(out_dir) / f"{run.kernel}_{run.dtype}_{run.lang}_best.{ext}"
    out.parent.mkdir(parents=True, exist_ok=True)
    header = "#" if run.lang == "triton" else "//"
    out.write_text(f"{header} kernelpilot agent best: run {run.run_id}, attempt "
                   f"{run.best.iteration}, {run.best.result.score:.3f}x vs PyTorch, "
                   f"model {run.model}\n{run.best.code}\n")
    return out
