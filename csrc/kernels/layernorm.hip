// Fused LayerNorm over the last dimension of a contiguous (M, N) tensor:
//   y = (x - mean) / sqrt(var + eps) * weight + bias   (biased variance, like PyTorch)
// Memory bound, so like softmax the goal is one read and one write of x.
#include "../compat.h"
#include "../launchers.h"

namespace kp {
namespace {

// Stage 0 (naive): one thread per row, three passes (mean, variance, write).
// Uncoalesced like the naive softmax.
template <typename T>
__global__ void ln_naive(const T* __restrict__ x, const T* __restrict__ w,
                         const T* __restrict__ b, T* __restrict__ y, int M, int N, float eps) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= M) return;
  const T* xr = x + static_cast<int64_t>(row) * N;
  T* yr = y + static_cast<int64_t>(row) * N;
  float mean = 0.f;
  for (int j = 0; j < N; ++j) mean += to_float(xr[j]);
  mean /= N;
  float var = 0.f;
  for (int j = 0; j < N; ++j) {
    const float d = to_float(xr[j]) - mean;
    var += d * d;
  }
  const float rstd = rsqrtf(var / N + eps);
  for (int j = 0; j < N; ++j)
    yr[j] = from_float<T>((to_float(xr[j]) - mean) * rstd * to_float(w[j]) + to_float(b[j]));
}

// Stage 1 (optimized): one block per row, coalesced strided access, two block
// reductions (mean, then centered variance: numerically safe). Three reads of x.
template <typename T>
__global__ void ln_block(const T* __restrict__ x, const T* __restrict__ w,
                         const T* __restrict__ b, T* __restrict__ y, int N, float eps) {
  __shared__ float scratch[kMaxWarps];
  const T* xr = x + static_cast<int64_t>(blockIdx.x) * N;
  T* yr = y + static_cast<int64_t>(blockIdx.x) * N;
  float s = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) s += to_float(xr[j]);
  const float mean = block_reduce<false>(s, scratch) / N;
  float v = 0.f;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const float d = to_float(xr[j]) - mean;
    v += d * d;
  }
  const float rstd = rsqrtf(block_reduce<false>(v, scratch) / N + eps);
  for (int j = threadIdx.x; j < N; j += blockDim.x)
    yr[j] = from_float<T>((to_float(xr[j]) - mean) * rstd * to_float(w[j]) + to_float(b[j]));
}

// Welford's online mean/variance, with Chan et al.'s parallel merge. Gives mean
// and M2 = sum((x - mean)^2) from a single pass without the catastrophic
// cancellation of E[x^2] - E[x]^2.
struct Welford {
  float n = 0.f, mean = 0.f, m2 = 0.f;
  __device__ __forceinline__ void add(float x) {
    n += 1.f;
    const float d = x - mean;
    mean += d / n;
    m2 += d * (x - mean);
  }
  __device__ __forceinline__ void merge(float nb, float mb, float m2b) {
    if (nb == 0.f) return;
    const float nn = n + nb;
    const float d = mb - mean;
    mean += d * (nb / nn);
    m2 += m2b + d * d * (n * nb / nn);
    n = nn;
  }
};

// Stage 2 (best): one read pass computing Welford statistics per thread, merged
// with warp shuffles and one shared-memory hop. The row is cached in shared
// memory during that pass when it fits, so the normalize pass never re-reads x.
template <typename T, bool CacheRow>
__global__ void ln_welford(const T* __restrict__ x, const T* __restrict__ w,
                           const T* __restrict__ b, T* __restrict__ y, int N, float eps) {
  extern __shared__ float row_cache[];
  __shared__ float sn[kMaxWarps], smean[kMaxWarps], sm2[kMaxWarps];
  const T* xr = x + static_cast<int64_t>(blockIdx.x) * N;
  T* yr = y + static_cast<int64_t>(blockIdx.x) * N;

  Welford st;
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const float v = to_float(xr[j]);
    if (CacheRow) row_cache[j] = v;
    st.add(v);
  }
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    const float nb = KP_SHFL_XOR(st.n, off);
    const float mb = KP_SHFL_XOR(st.mean, off);
    const float m2b = KP_SHFL_XOR(st.m2, off);
    st.merge(nb, mb, m2b);
  }
  const int lane = threadIdx.x % warpSize, wid = threadIdx.x / warpSize;
  const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) {
    sn[wid] = st.n;
    smean[wid] = st.mean;
    sm2[wid] = st.m2;
  }
  __syncthreads();
  if (wid == 0) {
    Welford t;
    if (lane < nwarps) {
      t.n = sn[lane];
      t.mean = smean[lane];
      t.m2 = sm2[lane];
    }
    for (int off = warpSize / 2; off > 0; off >>= 1) {
      const float nb = KP_SHFL_XOR(t.n, off);
      const float mb = KP_SHFL_XOR(t.mean, off);
      const float m2b = KP_SHFL_XOR(t.m2, off);
      t.merge(nb, mb, m2b);
    }
    if (lane == 0) {
      smean[0] = t.mean;
      sm2[0] = t.m2;
    }
  }
  __syncthreads();
  const float mean = smean[0];
  const float rstd = rsqrtf(sm2[0] / N + eps);
  for (int j = threadIdx.x; j < N; j += blockDim.x) {
    const float v = CacheRow ? row_cache[j] : to_float(xr[j]);
    yr[j] = from_float<T>((v - mean) * rstd * to_float(w[j]) + to_float(b[j]));
  }
}

constexpr size_t kRowCacheBytes = 48 * 1024 - 3 * kMaxWarps * sizeof(float);

inline int threads_for(int N) {
  if (N <= 512) return 128;
  if (N <= 4096) return 256;
  if (N <= 16384) return 512;
  return 1024;
}

template <typename T>
void launch(int variant, const T* x, const T* w, const T* b, T* y, int M, int N, float eps,
            hipStream_t s) {
  if (M == 0 || N == 0) return;
  if (variant == 0) {
    ln_naive<T><<<ceil_div(M, 128), 128, 0, s>>>(x, w, b, y, M, N, eps);
  } else if (variant == 1) {
    ln_block<T><<<M, 256, 0, s>>>(x, w, b, y, N, eps);
  } else {
    const size_t bytes = static_cast<size_t>(N) * sizeof(float);
    if (bytes <= kRowCacheBytes)
      ln_welford<T, true><<<M, threads_for(N), bytes, s>>>(x, w, b, y, N, eps);
    else
      ln_welford<T, false><<<M, threads_for(N), 0, s>>>(x, w, b, y, N, eps);
  }
  KP_CHECK_LAUNCH();
}

}  // namespace

void layernorm(int variant, DType dt, const void* x, const void* w, const void* b, void* y,
               int64_t M, int64_t N, float eps, void* stream) {
  auto s = static_cast<hipStream_t>(stream);
  if (dt == DType::F32)
    launch(variant, static_cast<const float*>(x), static_cast<const float*>(w),
           static_cast<const float*>(b), static_cast<float*>(y), static_cast<int>(M),
           static_cast<int>(N), eps, s);
  else
    launch(variant, static_cast<const __half*>(x), static_cast<const __half*>(w),
           static_cast<const __half*>(b), static_cast<__half*>(y), static_cast<int>(M),
           static_cast<int>(N), eps, s);
}

}  // namespace kp
