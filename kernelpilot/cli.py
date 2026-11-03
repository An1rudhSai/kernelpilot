"""kernelpilot command line.

    kernelpilot info
    kernelpilot check   --kernel softmax [--impl hip:best]
    kernelpilot bench   [--kernel softmax ...] [--out results]
    kernelpilot roofline --results results/bench.json --out results/roofline.png
    kernelpilot optimize --kernel softmax --iterations 10 [--lang triton|hip]
    kernelpilot evaluate --results results/bench.json --runs runs
"""

from __future__ import annotations

import argparse
import sys


def cmd_info(args: argparse.Namespace) -> int:
    from . import ext, registry
    from .device import get_gpu_info
    from .specs import SPECS

    print(get_gpu_info().describe())
    print(f"HIP extension loaded: {ext.load()}  (path: {ext.library_path()})")
    print(f"Triton available: {registry.triton_available()}")
    for name in SPECS:
        print(f"  {name}: {', '.join(registry.get_impls(name))}")
    return 0


def _select(impls: dict, patterns: list[str] | None) -> dict:
    if not patterns:
        return impls
    return {k: v for k, v in impls.items() if any(p in k for p in patterns)}


def cmd_check(args: argparse.Namespace) -> int:
    from . import registry
    from .correctness import check_all
    from .specs import SPECS

    failed = 0
    for name in args.kernel or list(SPECS):
        spec = SPECS[name]
        for impl, fn in _select(registry.get_impls(name), args.impl).items():
            for r in check_all(spec, impl, fn):
                if not r.ok or args.verbose:
                    print(r.summary())
                failed += not r.ok
    print("all checks passed" if not failed else f"{failed} checks FAILED")
    return 1 if failed else 0


def cmd_bench(args: argparse.Namespace) -> int:
    from . import registry
    from .benchmark import bench_kernel, save_results, to_markdown
    from .device import get_gpu_info
    from .specs import SPECS

    gpu = get_gpu_info()
    print(gpu.describe(), file=sys.stderr)
    results = []
    for name in args.kernel or list(SPECS):
        spec = SPECS[name]
        impls = _select(registry.get_impls(name), args.impl)
        if "torch" not in impls:
            impls = {"torch": spec.baseline, **impls}
        print(f"benchmarking {name}: {', '.join(impls)}", file=sys.stderr)
        results += bench_kernel(spec, impls, gpu=gpu, iters=args.iters)
    print(to_markdown(results, gpu))
    if args.out:
        from .roofline import plot_roofline

        save_results(results, gpu, args.out)
        plot_roofline(results, gpu, f"{args.out}/roofline.png")
        print(f"wrote {args.out}/bench.json, bench.md and roofline.png", file=sys.stderr)
    return 0


def cmd_roofline(args: argparse.Namespace) -> int:
    from .benchmark import load_results
    from .roofline import plot_roofline

    gpu, results = load_results(args.results)
    print(plot_roofline(results, gpu, args.out))
    return 0


def cmd_optimize(args: argparse.Namespace) -> int:
    from .agent.llm import OpenAICompatLLM
    from .agent.loop import Agent, save_best

    llm = OpenAICompatLLM.from_env(model=args.model, base_url=args.base_url,
                                   temperature=args.temperature)
    agent = Agent(llm, args.kernel, lang=args.lang, dtype=args.dtype, log_path=args.log,
                  timeout_s=args.timeout)
    run = agent.optimize(args.iterations)
    path = save_best(run, args.save_dir)
    if path:
        print(f"saved best kernel to {path}")
    return 0 if run.best else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    from pathlib import Path

    from .evaluation import report

    text = report(args.results, args.runs)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kernelpilot")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("info", help="show GPU, peaks and available implementations")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("check", help="run the correctness harness")
    s.add_argument("--kernel", action="append")
    s.add_argument("--impl", action="append", help="substring filter, e.g. hip: or triton")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("bench", help="run the benchmark harness")
    s.add_argument("--kernel", action="append")
    s.add_argument("--impl", action="append", help="substring filter, e.g. hip: or triton")
    s.add_argument("--iters", type=int, default=100)
    s.add_argument("--out", default="results")
    s.set_defaults(func=cmd_bench)

    s = sub.add_parser("roofline", help="plot a roofline from saved benchmark results")
    s.add_argument("--results", default="results/bench.json")
    s.add_argument("--out", default="results/roofline.png")
    s.set_defaults(func=cmd_roofline)

    s = sub.add_parser("optimize", help="run the LLM agent on one kernel")
    s.add_argument("--kernel", required=True)
    s.add_argument("--iterations", type=int, default=10)
    s.add_argument("--lang", choices=["triton", "hip"], default="triton")
    s.add_argument("--dtype", choices=["float32", "float16"], default="float32")
    s.add_argument("--model", help="model name (default: $KP_MODEL)")
    s.add_argument("--base-url", help="OpenAI-compatible base URL (default: $KP_BASE_URL)")
    s.add_argument("--temperature", type=float, default=0.7)
    s.add_argument("--timeout", type=float, default=300.0, help="seconds per candidate")
    s.add_argument("--log", help="JSONL log path (default: runs/<kernel>_..._<id>.jsonl)")
    s.add_argument("--save-dir", default="agent_kernels")
    s.set_defaults(func=cmd_optimize)

    s = sub.add_parser("evaluate", help="compare PyTorch vs hand-written vs agent")
    s.add_argument("--results", default="results/bench.json")
    s.add_argument("--runs", default="runs")
    s.add_argument("--out", default="results/evaluation.md")
    s.set_defaults(func=cmd_evaluate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
