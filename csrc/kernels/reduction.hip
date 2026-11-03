// Sum reduction of a 1-D tensor into a float32 scalar (fp32 accumulation).
// Memory bound: the whole game is to read the input once at full bandwidth and
// keep the reduction tree cheap and divergence-free.
#include "../compat.h"
#include "../launchers.h"

namespace kp {
namespace {

constexpr int kThreads = 256;

// Stage 0 (naive): the textbook "interleaved addressing" tree in shared memory.
// One element per thread, `tid % (2*s)` makes warps diverge, and half of the
// threads are idle from the very first step. One atomicAdd per block.
template <typename T>
__global__ void sum_naive(const T* __restrict__ x, float* __restrict__ out, int64_t n) {
  __shared__ float sdata[kThreads];
  const int tid = threadIdx.x;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid;
  sdata[tid] = i < n ? to_float(x[i]) : 0.f;
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    if (tid % (2 * s) == 0) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

// Stage 1 (optimized): sequential addressing (contiguous active threads, no
// divergence within a warp, no bank conflicts), a grid-stride loop so each
// thread sums many elements in registers before touching shared memory, and a
// bounded grid so there are few atomics.
template <typename T>
__global__ void sum_seq(const T* __restrict__ x, float* __restrict__ out, int64_t n) {
  __shared__ float sdata[kThreads];
  const int tid = threadIdx.x;
  float acc = 0.f;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + tid; i < n; i += stride)
    acc += to_float(x[i]);
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

// Stage 2 (best): 128-bit vector loads, several independent accumulators per
// thread (ILP), then warp-shuffle reduction (registers only, no __syncthreads
// inside a warp) with a single shared-memory hop between warps.
__device__ __forceinline__ float sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }
__device__ __forceinline__ float sum_vec(float4 v, float) { return sum4(v); }
__device__ __forceinline__ float sum_vec(float4 v, __half) {
  const __half2* h = reinterpret_cast<const __half2*>(&v);
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    float2 f = __half22float2(h[k]);
    s += f.x + f.y;
  }
  return s;
}

template <typename T>
__global__ void sum_vec_shfl(const T* __restrict__ x, float* __restrict__ out, int64_t n) {
  __shared__ float scratch[kMaxWarps];
  constexpr int kPerVec = 16 / sizeof(T);
  const int64_t nvec = n / kPerVec;
  const float4* xv = reinterpret_cast<const float4*>(x);
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  float a0 = 0.f, a1 = 0.f;
  for (; i + stride < nvec; i += 2 * stride) {
    a0 += sum_vec(xv[i], T{});
    a1 += sum_vec(xv[i + stride], T{});
  }
  if (i < nvec) a0 += sum_vec(xv[i], T{});
  // Scalar tail (fewer than kPerVec elements), handled by the first block.
  if (blockIdx.x == 0) {
    for (int64_t j = nvec * kPerVec + threadIdx.x; j < n; j += blockDim.x) a1 += to_float(x[j]);
  }
  float total = block_reduce<false>(a0 + a1, scratch);
  if (threadIdx.x == 0) atomicAdd(out, total);
}

inline int bounded_blocks(int64_t work) {
  int64_t b = (work + kThreads - 1) / kThreads;
  if (b > 1024) b = 1024;
  return static_cast<int>(b < 1 ? 1 : b);
}

template <typename T>
void launch(int variant, const T* x, float* out, int64_t n, hipStream_t s) {
  // `out` is zero-initialised by the caller; empty input sums to 0.
  if (n == 0) return;
  if (variant == 0) {
    sum_naive<T><<<ceil_div(n, kThreads), kThreads, 0, s>>>(x, out, n);
  } else if (variant == 1) {
    sum_seq<T><<<bounded_blocks(n), kThreads, 0, s>>>(x, out, n);
  } else {
    if (reinterpret_cast<uintptr_t>(x) % 16 != 0) {
      sum_seq<T><<<bounded_blocks(n), kThreads, 0, s>>>(x, out, n);
    } else {
      // Each thread handles >= 2 vectors per pass; size the grid accordingly.
      const int64_t nvec = n / (16 / sizeof(T));
      sum_vec_shfl<T><<<bounded_blocks(nvec / 2 + 1), kThreads, 0, s>>>(x, out, n);
    }
  }
  KP_CHECK_LAUNCH();
}

}  // namespace

void reduce_sum(int variant, DType dt, const void* x, float* out, int64_t n, void* stream) {
  auto s = static_cast<hipStream_t>(stream);
  if (dt == DType::F32)
    launch(variant, static_cast<const float*>(x), out, n, s);
  else
    launch(variant, static_cast<const __half*>(x), out, n, s);
}

}  // namespace kp
