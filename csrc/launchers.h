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

// out must be a zero-initialised float32 scalar; accumulation is fp32.
void reduce_sum(int variant, DType dt, const void* x, float* out, int64_t n, void* stream);

// Row-wise softmax of a contiguous (M, N) matrix.
void softmax(int variant, DType dt, const void* x, void* y, int64_t M, int64_t N, void* stream);

// C = A @ B, row-major. variant 0..3: naive, smem tiled, register tiled, vectorized.
void matmul(int variant, DType dt, const void* A, const void* B, void* C, int64_t M, int64_t N,
            int64_t K, void* stream);

}  // namespace kp
