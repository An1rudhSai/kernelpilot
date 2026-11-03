"""Evaluation report: PyTorch built-in vs. hand-written best vs. agent best per
kernel, plus an honest breakdown of how agent attempts failed."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .benchmark import BenchResult, load_results


def _geomean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(map(math.log, xs)) / len(xs)) if xs else None


def handwritten_best(results: list[BenchResult]) -> dict[tuple[str, str], tuple[str, float]]:
    """(kernel, dtype) -> (impl, geomean speedup vs torch) for the best non-torch impl.
    An impl that failed correctness on any shape is excluded."""
    by: dict[tuple[str, str, str], list[BenchResult]] = defaultdict(list)
    for r in results:
        if r.impl != "torch":
            by[(r.kernel, r.dtype, r.impl)].append(r)
    best: dict[tuple[str, str], tuple[str, float]] = {}
    for (kernel, dtype, impl), rows in by.items():
        if any(not r.correct for r in rows):
            continue
        g = _geomean([r.speedup_vs_torch for r in rows])
        if g and ((kernel, dtype) not in best or g > best[(kernel, dtype)][1]):
            best[(kernel, dtype)] = (impl, g)
    return best


def load_runs(runs_dir: str | Path) -> dict[str, list[dict]]:
    runs: dict[str, list[dict]] = defaultdict(list)
    for path in sorted(Path(runs_dir).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                runs[rec["run_id"]].append(rec)
    for recs in runs.values():
        recs.sort(key=lambda r: r["iteration"])
    return dict(runs)


def summarize_run(recs: list[dict]) -> dict:
    stages = Counter(r["stage"] for r in recs)
    best = None
    for r in recs:
        if r.get("is_best"):
            best = r
    first = recs[0]
    return {
        "run_id": first["run_id"], "kernel": first["kernel"], "dtype": first["dtype"],
        "lang": first["lang"], "model": first["model"], "attempts": len(recs),
        "best_score": best["score"] if best else None,
        "iterations_to_best": best["iteration"] if best else None,
        "first_correct": next((r["iteration"] for r in recs if r["ok"]), None),
        "stages": dict(stages),
        "regressions": sum(1 for r in recs if r.get("is_regression")),
    }


def _x(v: float | None) -> str:
    return "–" if v is None else f"{v:.2f}x"


def report(bench_json: str | Path | None, runs_dir: str | Path) -> str:
    hand = {}
    gpu_line = ""
    if bench_json and Path(bench_json).exists():
        gpu, results = load_results(bench_json)
        hand = handwritten_best(results)
        gpu_line = f"GPU: {gpu.describe()}\n"
    runs = [summarize_run(r) for r in load_runs(runs_dir).values()] \
        if Path(runs_dir).exists() else []
    agent_best: dict[tuple[str, str], dict] = {}
    for s in runs:
        key = (s["kernel"], s["dtype"])
        if s["best_score"] and (key not in agent_best
                                or s["best_score"] > agent_best[key]["best_score"]):
            agent_best[key] = s

    lines = ["# Evaluation", "", gpu_line,
             "Speedups are geometric means over the benchmark shapes, relative to the PyTorch "
             "built-in (= 1.00x). Higher is better.", "",
             "| kernel | dtype | PyTorch | hand-written best | agent best | agent iterations "
             "to best | agent model/lang |",
             "|---|---|---:|---:|---:|---:|---|"]
    for key in sorted(set(hand) | set(agent_best)):
        h = hand.get(key)
        a = agent_best.get(key)
        lines.append(
            f"| {key[0]} | {key[1]} | 1.00x | "
            f"{_x(h[1]) + f' ({h[0]})' if h else '–'} | {_x(a['best_score']) if a else '–'} | "
            f"{a['iterations_to_best'] if a else '–'} | "
            f"{a['model'] + ' / ' + a['lang'] if a else '–'} |")
    if not hand and not agent_best:
        lines.append("| (no results yet: run `make bench` and `kernelpilot optimize` on a GPU) "
                     "| | | | | | |")

    lines += ["", "## Agent failure modes", "",
              "Every attempt is logged. `correctness` = compiled and ran but produced wrong "
              "numbers (silent bugs caught by the harness); `regression` = correct but slower "
              "than the best kernel at that point; `rejected` = delegated to PyTorch.", "",
              "| run | kernel | dtype | lang | model | attempts | ok | compile | correctness "
              "| timeout | crash | rejected | no code | regressions | first correct |",
              "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in sorted(runs, key=lambda s: (s["kernel"], s["dtype"], s["run_id"])):
        st = s["stages"]
        lines.append(
            f"| {s['run_id']} | {s['kernel']} | {s['dtype']} | {s['lang']} | {s['model']} | "
            f"{s['attempts']} | {st.get('ok', 0)} | {st.get('compile', 0)} | "
            f"{st.get('correctness', 0)} | {st.get('timeout', 0)} | "
            f"{st.get('crash', 0) + st.get('benchmark', 0)} | {st.get('rejected', 0)} | "
            f"{st.get('no_code', 0)} | {s['regressions']} | {s['first_correct'] or '–'} |")
    if runs:
        total = Counter()
        for s in runs:
            total.update(s["stages"])
        n = sum(total.values())
        lines += ["", "Totals: " + ", ".join(f"{k} {v} ({100 * v / n:.0f}%)"
                                             for k, v in total.most_common())]
    return "\n".join(lines) + "\n"
