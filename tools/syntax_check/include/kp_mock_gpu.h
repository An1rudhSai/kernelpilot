// Minimal stand-ins for the CUDA/HIP runtime and fp16 headers, just enough for
// clang to parse and type-check the kernels (host and device side) in
// -fsyntax-only mode on a machine with no GPU toolkit. Never linked or run.
#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1)
      : x(x_), y(y_), z(z_) {}
};
struct __kp_uint3 { unsigned x, y, z; };
extern const __device__ __kp_uint3 threadIdx;
extern const __device__ __kp_uint3 blockIdx;
extern const __device__ dim3 blockDim;
extern const __device__ dim3 gridDim;
extern const __device__ int warpSize;

struct alignas(8) float2 { float x, y; };
struct alignas(16) float4 { float x, y, z, w; };
struct alignas(8) uint2 { unsigned x, y; };
struct alignas(16) uint4 { unsigned x, y, z, w; };
__host__ __device__ inline float2 make_float2(float x, float y) { return {x, y}; }
__host__ __device__ inline float4 make_float4(float x, float y, float z, float w) {
  return {x, y, z, w};
}

struct __half {
  unsigned short x;
  __host__ __device__ __half() : x(0) {}
};
struct alignas(4) __half2 { __half x, y; };
__host__ __device__ float __half2float(__half);
__host__ __device__ __half __float2half(float);
__host__ __device__ float2 __half22float2(__half2);
__host__ __device__ __half2 __float22half2_rn(float2);
__host__ __device__ __half2 __floats2half2_rn(float, float);

__device__ void __syncthreads();
__device__ float atomicAdd(float*, float);
__device__ int atomicAdd(int*, int);
__device__ float expf(float);
__device__ float __expf(float);
__device__ float fmaxf(float, float);
__device__ float fminf(float, float);
__device__ float sqrtf(float);
__device__ float rsqrtf(float);
__device__ float fabsf(float);
__device__ float __fdividef(float, float);

typedef struct __kp_stream* __kp_stream_t;

#if defined(KP_MOCK_HIP)
#define __HIP_PLATFORM_AMD__ 1
#define __HIPCC__ 1
typedef __kp_stream_t hipStream_t;
typedef int hipError_t;
#define hipSuccess 0
hipError_t hipGetLastError();
const char* hipGetErrorString(hipError_t);
__device__ float __shfl_down(float, unsigned, int = 64);
__device__ float __shfl_xor(float, int, int = 64);
__device__ int __shfl_xor(int, int, int = 64);
extern "C" int hipConfigureCall(dim3, dim3, size_t = 0, hipStream_t = 0);
extern "C" int __hipPushCallConfiguration(dim3, dim3, size_t = 0, hipStream_t = 0);
#else
typedef __kp_stream_t cudaStream_t;
typedef int cudaError_t;
#define cudaSuccess 0
cudaError_t cudaGetLastError();
const char* cudaGetErrorString(cudaError_t);
__device__ float __shfl_down_sync(unsigned, float, unsigned, int = 32);
__device__ float __shfl_xor_sync(unsigned, float, int, int = 32);
__device__ int __shfl_xor_sync(unsigned, int, int, int = 32);
extern "C" int cudaConfigureCall(dim3, dim3, size_t = 0, cudaStream_t = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, void* = 0);
#endif
