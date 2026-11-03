"""kernelpilot command line.

    kernelpilot info
    kernelpilot check   --kernel softmax [--impl hip:best]
    kernelpilot bench   [--kernel softmax ...] [--out results]
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
        save_results(results, gpu, args.out)
        print(f"wrote {args.out}/bench.json and bench.md", file=sys.stderr)
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
