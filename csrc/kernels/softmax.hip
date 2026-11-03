// Row-wise softmax over a contiguous (M, N) tensor, numerically stable
// (subtract the row max before exp). Memory bound: ideal traffic is one read and
// one write of the matrix, which only the best stage achieves.
#include "../compat.h"
#include "../launchers.h"

namespace kp {
namespace {

// Stage 0 (naive): one thread per row, three sequential passes (max, sum,
// normalize). Neighbouring threads read addresses N elements apart, so loads
// are uncoalesced, and small M leaves most of the GPU idle.
template <typename T>
__global__ void softmax_naive(const T* __restrict__ x, T* __restrict__ y, int M, int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const T* xr = x + static_cast<int64_t>(row) * N;
  T* yr = y + static_cast<int64_t>(row) * N;
  float m = -INFINITY;
  for (int j = 0; j < N; ++j) m = fmaxf(m, to_float(xr[j]));
  float d = 0.f;
  for (int j = 0; j < N; ++j) d += expf(to_float(xr[j]) - m);
  const float inv = 1.f / d;
  for (int j = 0; j < N; ++j) yr[j] = from_float<T>(expf(to_float(xr[j]) - m) * inv);
}

// Stage 1 (optimized): one block per row. Threads stride across the row so loads
// coalesce, and max/sum use block-wide tree reductions. Still three passes over
// global memory: read for max, read for sum, read + write to normalize.
template <typename T>
__global__ void softmax_block(const T* __restrict__ x, T* __restrict__ y, int N) {
  __shared__ float scratch[kMaxWarps];
  const T* xr = x + static_cast<int64_t>(blockIdx.x) * N;
  T* yr = y + static_cast<int64_t>(blockIdx.x) * N;
  float m = -INFINITY;
  for (int j = threadIdx.x; j < N; j += blockDim.x) m = fmaxf(m, to_float(xr[j]));
  m = block_reduce<true>(m, scratch);
  float d = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) d += expf(to_float(xr[j]) - m);
  d = block_reduce<false>(d, scratch);
  const float inv = 1.f / d;
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    yr[j] = from_float<T>(expf(to_float(xr[j]) - m) * inv);
}

// Online softmax (Milakov & Gimelshein 2018): a running (max, sum) pair is
// updated per element, so max and sum come out of a single pass.
__device__ __forceinline__ void online_update(float& m, float& d, float x) {
  if (x > m) {
    d = d * expf(m - x) + 1.f;
    m = x;
  } else if (m != -INFINITY) {
    d += expf(x - m);
  }
}

__device__ __forceinline__ void online_merge(float& m, float& d, float m2, float d2) {
  const float mn = fmaxf(m, m2);
  if (mn == -INFINITY) return;  // both partial states are empty
  d = d * expf(m - mn) + d2 * expf(m2 - mn);
  m = mn;
}

// Stage 2 (best): fused max/sum/normalize. One block per row computes the
// online (max, sum) in one read pass, merging partial states with warp shuffles
// and one shared-memory hop. When the row fits in shared memory it is cached
// during that pass, so global memory sees exactly one read and one write.
template <typename T, bool CacheRow>
__global__ void softmax_online(const T* __restrict__ x, T* __restrict__ y, int N) {
  extern __shared__ float row_cache[];
  __shared__ float sm[kMaxWarps], sd[kMaxWarps];
  const T* xr = x + static_cast<int64_t>(blockIdx.x) * N;
  T* yr = y + static_cast<int64_t>(blockIdx.x) * N;

  float m = -INFINITY, d = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const float v = to_float(xr[j]);
    if (CacheRow) row_cache[j] = v;
    online_update(m, d, v);
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    const float m2 = KP_SHFL_XOR(m, off);
    const float d2 = KP_SHFL_XOR(d, off);
    online_merge(m, d, m2, d2);
  }
  const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
  const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) {
    sm[wid] = m;
    sd[wid] = d;
  }
  __syncthreads();
  if (wid == 0) {
    m = lane < nwarps ? sm[lane] : -INFINITY;
    d = lane < nwarps ? sd[lane] : 0.f;
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float m2 = KP_SHFL_XOR(m, off);
      const float d2 = KP_SHFL_XOR(d, off);
      online_merge(m, d, m2, d2);
    }
    if (lane == 0) {
      sm[0] = m;
      sd[0] = d;
    }
  }
  __syncthreads();
  m = sm[0];
  const float inv = 1.f / sd[0];
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const float v = CacheRow ? row_cache[j] : to_float(xr[j]);
    yr[j] = from_float<T>(expf(v - m) * inv);
  }
}

// Default static shared memory budget available on every target without opt-in.
constexpr size_t kRowCacheBytes = 48 * 1024 - 2 * kMaxWarps * sizeof(float);

inline int threads_for(int N) {
  if (N <= 512) return 128;
  if (N <= 4096) return 256;
  if (N <= 16384) return 512;
  return 1024;
}

template <typename T>
void launch(int variant, const T* x, T* y, int M, int N, hipStream_t s) {
  if (M == 0 || N == 0) return;
  if (variant == 0) {
    softmax_naive<T><<<ceil_div(M, 128), 128, 0, s>>>(x, y, M, N);
  } else if (variant == 1) {
    softmax_block<T><<<M, 256, 0, s>>>(x, y, N);
  } else {
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    if (bytes <= kRowCacheBytes)
      softmax_online<T, true><<<M, threads_for(N), bytes, s>>>(x, y, N);
    else
      softmax_online<T, false><<<M, threads_for(N), 0, s>>>(x, y, N);
  }
  KP_CHECK_LAUNCH();
}

}  // namespace

void softmax(int variant, DType dt, const void* x, void* y, int64_t M, int64_t N, void* stream) {
  auto s = static_cast<hipStream_t>(stream);
  if (dt == DType::F32)
    launch(variant, static_cast<const float*>(x), static_cast<float*>(y), static_cast<int>(M),
           static_cast<int>(N), s);
  else
    launch(variant, static_cast<const __half*>(x), static_cast<__half*>(y), static_cast<int>(M),
           static_cast<int>(N), s);
}

}  // namespace kp
