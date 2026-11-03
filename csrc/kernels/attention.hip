// Scaled dot-product attention (non-causal) for small sequence lengths:
//   O = softmax(Q K^T / sqrt(D)) V,  Q, K, V, O contiguous (B, H, S, D).
// Stretch goal: two stages only.
#include "../compat.h"
#include "../launchers.h"

namespace kp {
namespace {

// Stage 0 (naive): one block per (batch*head, query row). The full score row
// (S floats) is materialised in shared memory, then a block softmax, then each
// thread computes some output dims by walking all S rows of V.
template <typename T>
__global__ void attn_naive(const T* __restrict__ q, const T* __restrict__ k,
                           const T* __restrict__ v, T* __restrict__ o, int S, int D,
                           float scale) {
  extern __shared__ float smem[];  // [D] query, then [S] scores
  __shared__ float scratch[kMaxWarps];
  float* qs = smem;
  float* p = smem + D;
  const int64_t bh = blockIdx.y;
  const int i = blockIdx.x;
  const T* qr = q + (bh * S + i) * D;
  const T* kb = k + bh * S * D;
  const T* vb = v + bh * S * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) qs[d] = to_float(qr[d]);
  __syncthreads();
  float m = -INFINITY;
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    float s = 0.f;
    for (int d = 0; d < D; ++d) s += qs[d] * to_float(kb[static_cast<int64_t>(j) * D + d]);
    s *= scale;
    p[j] = s;
    m = fmaxf(m, s);
  }
  m = block_reduce<true>(m, scratch);
  float l = 0.f;
  for (int j = threadIdx.x; j < S; j += blockDim.x) {
    const float e = expf(p[j] - m);
    p[j] = e;
    l += e;
  }
  l = block_reduce<false>(l, scratch);  // also a barrier: all p[] are written
  const float inv = 1.f / l;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < S; ++j) acc += p[j] * to_float(vb[static_cast<int64_t>(j) * D + d]);
    o[(bh * S + i) * D + d] = from_float<T>(acc * inv);
  }
}

// Stage 1 (flash-style tiling): one thread per query, kBQ queries per block.
// K and V are streamed through shared memory in tiles of kBK keys and reused by
// all kBQ queries in the block; every thread reads the same K/V element at the
// same time (a broadcast, so no bank conflicts). Softmax is computed online
// (running max m, running sum l, rescaled accumulator), so the S x S score
// matrix never exists anywhere. D is a template parameter so q and the output
// accumulator live in registers.
constexpr int kBQ = 64;
constexpr int kBK = 32;

template <typename T, int D>
__global__ void __launch_bounds__(kBQ) attn_flash(const T* __restrict__ q, const T* __restrict__ k,
                                                  const T* __restrict__ v, T* __restrict__ o,
                                                  int S, float scale) {
  __shared__ float Ks[kBK][D];
  __shared__ float Vs[kBK][D];
  const int64_t bh = blockIdx.y;
  const int i = blockIdx.x * kBQ + threadIdx.x;
  const bool active = i < S;
  const T* kb = k + bh * S * D;
  const T* vb = v + bh * S * D;

  float qr[D], acc[D];
#pragma unroll
  for (int d = 0; d < D; ++d) {
    qr[d] = active ? to_float(q[(bh * S + i) * D + d]) * scale : 0.f;
    acc[d] = 0.f;
  }
  float m = -INFINITY, l = 0.f;

  for (int j0 = 0; j0 < S; j0 += kBK) {
    const int nk = min(kBK, S - j0);
    for (int idx = threadIdx.x; idx < kBK * D; idx += kBQ) {
      const int r = idx / D, c = idx % D;
      const bool in = r < nk;
      Ks[r][c] = in ? to_float(kb[static_cast<int64_t>(j0 + r) * D + c]) : 0.f;
      Vs[r][c] = in ? to_float(vb[static_cast<int64_t>(j0 + r) * D + c]) : 0.f;
    }
    __syncthreads();
    for (int j = 0; j < nk; ++j) {
      float s = 0.f;
#pragma unroll
      for (int d = 0; d < D; ++d) s += qr[d] * Ks[j][d];
      const float m_new = fmaxf(m, s);
      const float alpha = expf(m - m_new);  // 0 on the first key (m = -inf)
      const float pj = expf(s - m_new);
      l = l * alpha + pj;
#pragma unroll
      for (int d = 0; d < D; ++d) acc[d] = acc[d] * alpha + pj * Vs[j][d];
      m = m_new;
    }
    __syncthreads();
  }
  if (active) {
    const float inv = 1.f / l;
#pragma unroll
    for (int d = 0; d < D; ++d) o[(bh * S + i) * D + d] = from_float<T>(acc[d] * inv);
  }
}

template <typename T, int D>
void launch_flash(const T* q, const T* k, const T* v, T* o, int BH, int S, float scale,
                  hipStream_t s) {
  dim3 grid(ceil_div(S, kBQ), BH);
  attn_flash<T, D><<<grid, kBQ, 0, s>>>(q, k, v, o, S, scale);
}

template <typename T>
void launch(int variant, const T* q, const T* k, const T* v, T* o, int BH, int S, int D,
            hipStream_t s) {
  if (BH == 0 || S == 0) return;
  const float scale = 1.f / sqrtf(static_cast<float>(D));
  bool done = false;
  if (variant == 1) {
    done = true;
    switch (D) {
      case 16: launch_flash<T, 16>(q, k, v, o, BH, S, scale, s); break;
      case 32: launch_flash<T, 32>(q, k, v, o, BH, S, scale, s); break;
      case 64: launch_flash<T, 64>(q, k, v, o, BH, S, scale, s); break;
      case 128: launch_flash<T, 128>(q, k, v, o, BH, S, scale, s); break;
      default: done = false;  // other head dims use the naive kernel
    }
  }
  if (!done) {
    dim3 grid(S, BH);
    const size_t bytes = static_cast<size_t>(S + D) * sizeof(float);
    attn_naive<T><<<grid, 128, bytes, s>>>(q, k, v, o, S, D, scale);
  }
  KP_CHECK_LAUNCH();
}

}  // namespace

void attention(int variant, DType dt, const void* q, const void* k, const void* v, void* o,
               int64_t BH, int64_t S, int64_t D, void* stream) {
  auto s = static_cast<hipStream_t>(stream);
  if (dt == DType::F32)
    launch(variant, static_cast<const float*>(q), static_cast<const float*>(k),
           static_cast<const float*>(v), static_cast<float*>(o), static_cast<int>(BH),
           static_cast<int>(S), static_cast<int>(D), s);
  else
    launch(variant, static_cast<const __half*>(q), static_cast<const __half*>(k),
           static_cast<const __half*>(v), static_cast<__half*>(o), static_cast<int>(BH),
           static_cast<int>(S), static_cast<int>(D), s);
}

}  // namespace kp
