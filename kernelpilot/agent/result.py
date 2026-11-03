"""What evaluating one candidate kernel produces. Shared by the sandboxed worker
(which fills it in) and the agent loop (which reads it)."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

# How far a candidate got. Only "ok" candidates can ever become the best kernel.
STAGES = ("no_code", "rejected", "compile", "correctness", "benchmark", "timeout", "crash", "ok")


@dataclass
class EvalResult:
    stage: str
    error: str | None = None
    compile_log: str = ""
    # Failed correctness checks (CheckResult dicts), at most a few.
    failures: list[dict] = field(default_factory=list)
    checks_passed: int = 0
    # One row per benchmark shape: shape, ms, torch_ms, speedup, gbs, tflops, pct_*.
    bench: list[dict] = field(default_factory=list)
    profile: dict = field(default_factory=dict)
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.stage == "ok"

    @property
    def score(self) -> float | None:
        """Geometric-mean speedup over PyTorch across benchmark shapes."""
        sp = [r["speedup"] for r in self.bench if r.get("speedup")]
        if not self.ok or not sp:
            return None
        return math.exp(sum(math.log(s) for s in sp) / len(sp))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["score"] = self.score
        return d

    @classmethod
    def from_dict(cls, d: dict) -> EvalResult:
        d = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**d)
