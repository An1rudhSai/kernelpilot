// C = A @ B for row-major A (M x K), B (K x N), C (M x N), fp32 or fp16 inputs,
// always accumulating in fp32. At large sizes this is compute bound, so the
// stages are about raising arithmetic intensity at each level of the memory
// hierarchy: global -> shared memory -> registers.
//
// None of these stages use tensor cores / matrix cores (WMMA/MFMA); PyTorch's
// fp16 matmul does, so the hand-written fp16 kernels are expected to lose there.
#include "../compat.h"
#include "../launchers.h"

namespace kp {
namespace {

// Stage 0 (naive): one thread per output, K-loop straight from global memory.
// Every A and B element is re-read from DRAM/L2 by many threads: ~0.25 FLOP/byte.
template <typename T>
__global__ void mm_naive(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                         int M, int N, int K) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= M || col >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; ++k)
    acc += to_float(A[static_cast<int64_t>(row) * K + k]) * to_float(B[static_cast<int64_t>(k) * N + col]);
  C[static_cast<int64_t>(row) * N + col] = from_float<T>(acc);
}

// Stage 1 (shared-memory tiling): a 32x32 block stages 32x32 tiles of A and B
// in shared memory, so each global element is loaded once per tile and reused
// 32 times from on-chip memory.
constexpr int kTile = 32;

template <typename T>
__global__ void mm_smem(const T* __restrict__ A, const T* __restrict__ B, T* __restrict__ C,
                        int M, int N, int K) {
  __shared__ float As[kTile][kTile];
  __shared__ float Bs[kTile][kTile + 1];  // +1 pad: no bank conflicts on column reads
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int row = blockIdx.y * kTile + ty;
  const int col = blockIdx.x * kTile + tx;
  float acc = 0.f;
  for (int k0 = 0; k0 < K; k0 += kTile) {
    As[ty][tx] = (row < M && k0 + tx < K) ? to_float(A[static_cast<int64_t>(row) * K + k0 + tx]) : 0.f;
    Bs[ty][tx] = (k0 + ty < K && col < N) ? to_float(B[static_cast<int64_t>(k0 + ty) * N + col]) : 0.f;
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kTile; ++k) acc += As[ty][k] * Bs[k][tx];
    __syncthreads();
  }
  if (row < M && col < N) C[static_cast<int64_t>(row) * N + col] = from_float<T>(acc);
}

// Stage 2 (register tiling): a 64x64 output tile per 256-thread block, each
// thread owning a 4x4 sub-tile in registers. Per k step a thread reads 4 A and
// 4 B values from shared memory and does 16 FMAs, so shared-memory traffic per
// FLOP drops 4x versus stage 1. A is stored transposed in shared memory so the
// inner loop reads it with unit stride.
constexpr int kBM2 = 64, kBN2 = 64, kBK2 = 16, kTM2 = 4, kTN2 = 4;

template <typename T>
__global__ void __launch_bounds__(256) mm_regtile(const T* __restrict__ A,
                                                  const T* __restrict__ B,
                                                  T* __restrict__ C, int M, int N, int K) {
  __shared__ float As[kBK2][kBM2];
  __shared__ float Bs[kBK2][kBN2];
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  const int bm = blockIdx.y * kBM2, bn = blockIdx.x * kBN2;
  float acc[kTM2][kTN2] = {};

  for (int k0 = 0; k0 < K; k0 += kBK2) {
#pragma unroll
    for (int r = 0; r < (kBM2 * kBK2) / 256; ++r) {
      const int idx = tid + r * 256;
      const int ar = idx / kBK2, ac = idx % kBK2;
      const int gr = bm + ar, gc = k0 + ac;
      As[ac][ar] = (gr < M && gc < K) ? to_float(A[static_cast<int64_t>(gr) * K + gc]) : 0.f;
    }
#pragma unroll
    for (int r = 0; r < (kBK2 * kBN2) / 256; ++r) {
      const int idx = tid + r * 256;
      const int br = idx / kBN2, bc = idx % kBN2;
      const int gr = k0 + br, gc = bn + bc;
      Bs[br][bc] = (gr < K && gc < N) ? to_float(B[static_cast<int64_t>(gr) * N + gc]) : 0.f;
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kBK2; ++k) {
      float a[kTM2], b[kTN2];
      // Strided ownership (ty + 16*i, tx + 16*j) keeps a warp's B reads on
      // consecutive banks.
#pragma unroll
      for (int i = 0; i < kTM2; ++i) a[i] = As[k][ty + 16 * i];
#pragma unroll
      for (int j = 0; j < kTN2; ++j) b[j] = Bs[k][tx + 16 * j];
#pragma unroll
      for (int i = 0; i < kTM2; ++i)
#pragma unroll
        for (int j = 0; j < kTN2; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < kTM2; ++i) {
    const int r = bm + ty + 16 * i;
    if (r >= M) continue;
#pragma unroll
    for (int j = 0; j < kTN2; ++j) {
      const int c = bn + tx + 16 * j;
      if (c < N) C[static_cast<int64_t>(r) * N + c] = from_float<T>(acc[i][j]);
    }
  }
}

// Stage 3 (vectorized): 128x128 tile, 8x8 outputs per thread (64 FMAs per 16
// shared-memory reads), 128-bit global loads (float4, or 4 halves as 64 bits)
// and 128-bit shared-memory reads in the inner loop. Requires K % 4 == 0 and
// N % 4 == 0 so every 4-wide chunk is either fully in bounds or fully out.
constexpr int kBM3 = 128, kBN3 = 128, kBK3 = 8, kTM3 = 8, kTN3 = 8;

__device__ __forceinline__ void load4(const float* p, float* out) {
  const float4 v = *reinterpret_cast<const float4*>(p);
  out[0] = v.x; out[1] = v.y; out[2] = v.z; out[3] = v.w;
}
__device__ __forceinline__ void load4(const __half* p, float* out) {
  const float2 raw = *reinterpret_cast<const float2*>(p);  // 4 halves, 64-bit load
  const __half2* h = reinterpret_cast<const __half2*>(&raw);
  const float2 lo = __half22float2(h[0]), hi = __half22float2(h[1]);
  out[0] = lo.x; out[1] = lo.y; out[2] = hi.x; out[3] = hi.y;
}
__device__ __forceinline__ void store4(float* p, const float* v) {
  *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
}
__device__ __forceinline__ void store4(__half* p, const float* v) {
  float2 raw;
  __half2* h = reinterpret_cast<__half2*>(&raw);
  h[0] = __float22half2_rn(make_float2(v[0], v[1]));
  h[1] = __float22half2_rn(make_float2(v[2], v[3]));
  *reinterpret_cast<float2*>(p) = raw;
}

template <typename T>
__global__ void __launch_bounds__(256) mm_vec(const T* __restrict__ A, const T* __restrict__ B,
                                              T* __restrict__ C, int M, int N, int K) {
  __shared__ alignas(16) float As[kBK3][kBM3];
  __shared__ alignas(16) float Bs[kBK3][kBN3];
  const int tid = threadIdx.x;
  const int tx = tid % 16, ty = tid / 16;
  const int bm = blockIdx.y * kBM3, bn = blockIdx.x * kBN3;
  // Load mapping: A tile is 128 rows x 8 cols = 256 chunks of 4; B tile is
  // 8 rows x 128 cols = 256 chunks of 4. One chunk per thread for each.
  const int a_r = tid / 2, a_c = (tid % 2) * 4;
  const int b_r = tid / 32, b_c = (tid % 32) * 4;
  float acc[kTM3][kTN3] = {};

  for (int k0 = 0; k0 < K; k0 += kBK3) {
    float v[4];
    {
      const int gr = bm + a_r, gc = k0 + a_c;
      if (gr < M && gc < K) {
        load4(A + static_cast<int64_t>(gr) * K + gc, v);
      } else {
        v[0] = v[1] = v[2] = v[3] = 0.f;
      }
#pragma unroll
      for (int q = 0; q < 4; ++q) As[a_c + q][a_r] = v[q];
    }
    {
      const int gr = k0 + b_r, gc = bn + b_c;
      if (gr < K && gc < N) {
        load4(B + static_cast<int64_t>(gr) * N + gc, v);
      } else {
        v[0] = v[1] = v[2] = v[3] = 0.f;
      }
      store4(&Bs[b_r][b_c], v);
    }
    __syncthreads();
#pragma unroll
    for (int k = 0; k < kBK3; ++k) {
      float a[kTM3], b[kTN3];
      // Rows ty*4 .. +4 and 64 + ty*4 .. +4 (likewise for columns): two float4
      // reads each, and neighbouring threads read neighbouring 16-byte words.
      const float4 a0 = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
      const float4 a1 = *reinterpret_cast<const float4*>(&As[k][64 + ty * 4]);
      const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
      const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][64 + tx * 4]);
      a[0] = a0.x; a[1] = a0.y; a[2] = a0.z; a[3] = a0.w;
      a[4] = a1.x; a[5] = a1.y; a[6] = a1.z; a[7] = a1.w;
      b[0] = b0.x; b[1] = b0.y; b[2] = b0.z; b[3] = b0.w;
      b[4] = b1.x; b[5] = b1.y; b[6] = b1.z; b[7] = b1.w;
#pragma unroll
      for (int i = 0; i < kTM3; ++i)
#pragma unroll
        for (int j = 0; j < kTN3; ++j) acc[i][j] += a[i] * b[j];
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < kTM3; ++i) {
    const int r = bm + (i < 4 ? ty * 4 + i : 64 + ty * 4 + (i - 4));
    if (r >= M) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int c = bn + h * 64 + tx * 4;
      if (c < N) store4(C + static_cast<int64_t>(r) * N + c, &acc[i][h * 4]);
    }
  }
}

template <typename T>
void launch(int variant, const T* A, const T* B, T* C, int M, int N, int K, hipStream_t s) {
  if (M == 0 || N == 0) return;
  if (variant == 0) {
    dim3 block(16, 16);
    dim3 grid(ceil_div(N, 16), ceil_div(M, 16));
    mm_naive<T><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  } else if (variant == 1) {
    dim3 block(kTile, kTile);
    dim3 grid(ceil_div(N, kTile), ceil_div(M, kTile));
    mm_smem<T><<<grid, block, 0, s>>>(A, B, C, M, N, K);
  } else {
    const bool aligned = (reinterpret_cast<uintptr_t>(A) | reinterpret_cast<uintptr_t>(B) |
                          reinterpret_cast<uintptr_t>(C)) % 16 == 0;
    if (variant == 3 && K % 4 == 0 && N % 4 == 0 && aligned) {
      dim3 grid(ceil_div(N, kBN3), ceil_div(M, kBM3));
      mm_vec<T><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
    } else {
      dim3 grid(ceil_div(N, kBN2), ceil_div(M, kBM2));
      mm_regtile<T><<<grid, 256, 0, s>>>(A, B, C, M, N, K);
    }
  }
  KP_CHECK_LAUNCH();
}

}  // namespace

void matmul(int variant, DType dt, const void* A, const void* B, void* C, int64_t M, int64_t N,
            int64_t K, void* stream) {
  auto s = static_cast<hipStream_t>(stream);
  if (dt == DType::F32)
    launch(variant, static_cast<const float*>(A), static_cast<const float*>(B),
           static_cast<float*>(C), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K), s);
  else
    launch(variant, static_cast<const __half*>(A), static_cast<const __half*>(B),
           static_cast<__half*>(C), static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
           s);
}

}  // namespace kp
