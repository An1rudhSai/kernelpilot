// Host-side launchers implemented in csrc/kernels/*.hip. This header is plain
// C++ (no GPU headers) so bindings.cpp can be compiled by the host compiler.
// `variant` selects the optimization stage: 0 = naive, 1 = optimized, 2 = best.
// `stream` is the raw cudaStream_t / hipStream_t of the current PyTorch stream.
#pragma once
#include <cstdint>

namespace kp {

enum class DType : int { F32 = 0, F16 = 1 };

void vector_add(int variant, DType dt, const void* a, const void* b, void* c, int64_t n,
                void* stream);

}  // namespace kp
