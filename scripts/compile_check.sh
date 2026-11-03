#!/usr/bin/env bash
# Compile every HIP kernel file with nvcc (as CUDA) without a GPU or PyTorch.
# Used in CI to catch syntax/type errors in the kernels on every push.
set -euo pipefail
cd "$(dirname "$0")/.."
NVCC=${NVCC:-nvcc}
ARCH=${ARCH:-sm_75}
OUT=$(mktemp -d)
mkdir -p "$OUT/kernels"
cp csrc/compat.h csrc/launchers.h "$OUT/"
status=0
for f in csrc/kernels/*.hip; do
  stem=$(basename "$f" .hip)
  cp "$f" "$OUT/kernels/$stem.cu"
  echo "== $stem"
  if ! "$NVCC" -std=c++17 -O3 -arch="$ARCH" -c "$OUT/kernels/$stem.cu" -o "$OUT/$stem.o" \
      ${NVCC_FLAGS:-}; then
    status=1
  fi
done
exit $status
