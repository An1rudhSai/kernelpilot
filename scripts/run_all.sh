#!/usr/bin/env bash
# Full evaluation on a GPU machine: build, correctness, benchmarks + roofline,
# agent runs for every kernel, then the comparison report.
#   KP_MODEL=... KP_API_KEY=... [KP_BASE_URL=...] ./scripts/run_all.sh
# ITERS (default 10), LANG (triton|hip, default triton), DTYPES (default "float32 float16").
set -euo pipefail
cd "$(dirname "$0")/.."
ITERS=${ITERS:-10}
LANG_=${LANG:-triton}
DTYPES=${DTYPES:-"float32 float16"}
PY=${PYTHON:-python}

make build
make test-gpu
$PY -m kernelpilot.cli bench --out results
if [[ -n "${KP_MODEL:-}" ]]; then
  for k in vector_add reduce_sum softmax matmul layernorm attention; do
    for dt in $DTYPES; do
      $PY -m kernelpilot.cli optimize --kernel "$k" --dtype "$dt" --lang "$LANG_" \
        --iterations "$ITERS" || true   # a run with no correct kernel is a result too
    done
  done
else
  echo "KP_MODEL not set: skipping agent runs" >&2
fi
$PY -m kernelpilot.cli evaluate --results results/bench.json --runs runs --out results/evaluation.md
