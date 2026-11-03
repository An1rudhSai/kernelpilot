// Vector add: c = a + b. Purely memory bound (12 bytes moved per FLOP in fp32),
// so every stage is about moving bytes more efficiently, not about math.
#include "../compat.h"
#include "../launchers.h"

namespace kp {
namespace {

// Stage 0 (naive): one thread per element.
template <typename T>
__global__ void add_naive(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                          int64_t n) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = from_float<T>(to_float(a[i]) + to_float(b[i]));
}

// Stage 1 (optimized): grid-stride loop with 4 independent elements per iteration.
// A bounded grid amortises block scheduling cost, and the unrolled loads give
// each thread several memory requests in flight (instruction-level parallelism).
template <typename T>
__global__ void add_gridstride(const T* __restrict__ a, const T* __restrict__ b,
                               T* __restrict__ c, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    float x0 = to_float(a[i]) + to_float(b[i]);
    float x1 = to_float(a[i + stride]) + to_float(b[i + stride]);
    float x2 = to_float(a[i + 2 * stride]) + to_float(b[i + 2 * stride]);
    float x3 = to_float(a[i + 3 * stride]) + to_float(b[i + 3 * stride]);
    c[i] = from_float<T>(x0);
    c[i + stride] = from_float<T>(x1);
    c[i + 2 * stride] = from_float<T>(x2);
    c[i + 3 * stride] = from_float<T>(x3);
  }
  for (; i < n; i += stride) c[i] = from_float<T>(to_float(a[i]) + to_float(b[i]));
}

// Stage 2 (best): 128-bit vector loads/stores (float4 = 4 floats or 8 halves),
// grid-stride, scalar tail. Fewer, wider memory transactions.
__global__ void add_vec_f32(const float4* __restrict__ a, const float4* __restrict__ b,
                            float4* __restrict__ c, int64_t nvec) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nvec;
       i += stride) {
    float4 x = a[i], y = b[i];
    c[i] = make_float4(x.x + y.x, x.y + y.y, x.z + y.z, x.w + y.w);
  }
}

__global__ void add_vec_f16(const float4* __restrict__ a, const float4* __restrict__ b,
                            float4* __restrict__ c, int64_t nvec) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < nvec;
       i += stride) {
    float4 xa = a[i], xb = b[i], out;
    const __half2* ha = reinterpret_cast<const __half2*>(&xa);
    const __half2* hb = reinterpret_cast<const __half2*>(&xb);
    __half2* ho = reinterpret_cast<__half2*>(&out);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      float2 fa = __half22float2(ha[k]);
      float2 fb = __half22float2(hb[k]);
      ho[k] = __float22half2_rn(make_float2(fa.x + fb.x, fa.y + fb.y));
    }
    c[i] = out;
  }
}

template <typename T>
__global__ void add_tail(const T* __restrict__ a, const T* __restrict__ b, T* __restrict__ c,
                         int64_t start, int64_t n) {
  int64_t i = start + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) c[i] = from_float<T>(to_float(a[i]) + to_float(b[i]));
}

constexpr int kThreads = 256;
// Enough blocks to fill any current GPU several times over without launching
// one block per element.
constexpr int64_t kMaxBlocks = 4096;

inline int gridstride_blocks(int64_t n) {
  int64_t blocks = (n + kThreads - 1) / kThreads;
  if (blocks > kMaxBlocks) blocks = kMaxBlocks;
  return static_cast<int>(blocks < 1 ? 1 : blocks);
}

template <typename T>
void launch(int variant, const T* a, const T* b, T* c, int64_t n, hipStream_t s) {
  if (n == 0) return;
  if (variant == 0) {
    add_naive<T><<<ceil_div(n, kThreads), kThreads, 0, s>>>(a, b, c, n);
  } else if (variant == 1) {
    add_gridstride<T><<<gridstride_blocks(n), kThreads, 0, s>>>(a, b, c, n);
  } else {
    constexpr int per_vec = 16 / sizeof(T);
    const bool aligned = (reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(b) |
                          reinterpret_cast<uintptr_t>(c)) % 16 == 0;
    if (!aligned) {
      add_gridstride<T><<<gridstride_blocks(n), kThreads, 0, s>>>(a, b, c, n);
      KP_CHECK_LAUNCH();
      return;
    }
    const int64_t nvec = n / per_vec;
    if (nvec > 0) {
      auto va = reinterpret_cast<const float4*>(a);
      auto vb = reinterpret_cast<const float4*>(b);
      auto vc = reinterpret_cast<float4*>(c);
      if constexpr (sizeof(T) == 4) {
        add_vec_f32<<<gridstride_blocks(nvec), kThreads, 0, s>>>(va, vb, vc, nvec);
      } else {
        add_vec_f16<<<gridstride_blocks(nvec), kThreads, 0, s>>>(va, vb, vc, nvec);
      }
      KP_CHECK_LAUNCH();
    }
    const int64_t done = nvec * per_vec;
    if (done < n) add_tail<T><<<ceil_div(n - done, kThreads), kThreads, 0, s>>>(a, b, c, done, n);
  }
  KP_CHECK_LAUNCH();
}

}  // namespace

void vector_add(int variant, DType dt, const void* a, const void* b, void* c, int64_t n,
                void* stream) {
  auto s = static_cast<hipStream_t>(stream);
  if (dt == DType::F32) {
    launch(variant, static_cast<const float*>(a), static_cast<const float*>(b),
           static_cast<float*>(c), n, s);
  } else {
    launch(variant, static_cast<const __half*>(a), static_cast<const __half*>(b),
           static_cast<__half*>(c), n, s);
  }
}

}  // namespace kp
