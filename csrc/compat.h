// Portability layer: the kernels are written in HIP. On ROCm they compile as-is
// with hipcc/clang. On NVIDIA the build copies each .hip file to .cu and nvcc
// compiles it with the small set of aliases below.
#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define KP_SHFL_DOWN(v, d) __shfl_down((v), (d))
#define KP_SHFL_XOR(v, d) __shfl_xor((v), (d))
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
using hipStream_t = cudaStream_t;
using hipError_t = cudaError_t;
#define hipSuccess cudaSuccess
#define hipGetLastError cudaGetLastError
#define hipGetErrorString cudaGetErrorString
#define KP_SHFL_DOWN(v, d) __shfl_down_sync(0xffffffffu, (v), (d))
#define KP_SHFL_XOR(v, d) __shfl_xor_sync(0xffffffffu, (v), (d))
#endif

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#define KP_CHECK_LAUNCH()                                                           \
  do {                                                                              \
    hipError_t err_ = hipGetLastError();                                            \
    if (err_ != hipSuccess) {                                                       \
      throw std::runtime_error(std::string("kernel launch failed: ") +              \
                               hipGetErrorString(err_) + " at " + __FILE__ + ":" + \
                               std::to_string(__LINE__));                           \
    }                                                                               \
  } while (0)

namespace kp {

// Max warps per block we size shared scratch for. Works for both 32-wide (NVIDIA,
// RDNA) and 64-wide (CDNA) wavefronts since blockDim <= 1024.
constexpr int kMaxWarps = 32;

inline int ceil_div(int64_t a, int64_t b) { return static_cast<int>((a + b - 1) / b); }

__device__ __forceinline__ float to_float(float x) { return x; }
__device__ __forceinline__ float to_float(__half x) { return __half2float(x); }

template <typename T>
__device__ __forceinline__ T from_float(float x);
template <>
__device__ __forceinline__ float from_float<float>(float x) { return x; }
template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

// Warp-wide reductions. warpSize is 32 on NVIDIA/RDNA and 64 on CDNA; reading it
// at runtime keeps one source for both.
__device__ __forceinline__ float warp_sum(float v) {
  for (int off = warpSize / 2; off > 0; off >>= 1) v += KP_SHFL_XOR(v, off);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int off = warpSize / 2; off > 0; off >>= 1) v = fmaxf(v, KP_SHFL_XOR(v, off));
  return v;
}

// Block-wide reduction built from warp reductions. Every thread gets the result.
// `scratch` must hold kMaxWarps floats.
template <bool IsMax>
__device__ __forceinline__ float block_reduce(float v, float* scratch) {
  const int lane = threadIdx.x % warpSize;
  const int wid = threadIdx.x / warpSize;
  const int nwarps = (blockDim.x + warpSize - 1) / warpSize;
  v = IsMax ? warp_max(v) : warp_sum(v);
  __syncthreads();  // scratch may still be in use by a previous call
  if (lane == 0) scratch[wid] = v;
  __syncthreads();
  float init = IsMax ? -INFINITY : 0.f;
  v = (threadIdx.x < nwarps) ? scratch[threadIdx.x] : init;
  if (wid == 0) {
    v = IsMax ? warp_max(v) : warp_sum(v);
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
}

}  // namespace kp
