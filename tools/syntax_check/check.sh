#!/usr/bin/env bash
# Type-check every HIP kernel, device and host code, for both NVIDIA (CUDA mode)
# and AMD (HIP mode) using stub runtime headers. Needs a clang with the NVPTX
# and AMDGPU targets: e.g. Homebrew/upstream LLVM, or `pip install ziglang` and
# CLANG="python -m ziglang clang". No GPU or vendor toolkit required.
# This only proves the kernels parse and type-check; it does not run them.
set -uo pipefail
cd "$(dirname "$0")/../.."
CLANG=${CLANG:-clang++}
EXTRA=${EXTRA_FLAGS:-}
if [[ "$(uname)" == Darwin ]]; then
  SDK=$(xcrun --show-sdk-path)
  EXTRA="-isysroot $SDK -isystem $SDK/usr/include/c++/v1 $EXTRA"
fi
INC="-Itools/syntax_check/include -Icsrc"
status=0
for f in csrc/kernels/*.hip; do
  for mode in cuda hip; do
    if [[ $mode == cuda ]]; then
      flags="-x cuda --cuda-gpu-arch=sm_75 -nocudainc -nocudalib"
    else
      flags="-x hip --offload-arch=gfx90a -nogpuinc -nogpulib -DKP_MOCK_HIP"
    fi
    # shellcheck disable=SC2086
    if out=$($CLANG -std=c++17 -fsyntax-only $flags $INC $EXTRA "$f" 2>&1); then
      echo "ok    $mode  $f"
    else
      echo "FAIL  $mode  $f"; echo "$out" | head -40; status=1
    fi
  done
done
exit $status
