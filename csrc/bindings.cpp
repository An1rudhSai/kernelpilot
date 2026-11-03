// PyTorch C++ extension: registers the hand-written kernels as torch.ops.kp.*.
// Uses TORCH_LIBRARY (not pybind11) so the shared library is loaded with
// torch.ops.load_library and needs no Python headers at build time.
#include <torch/library.h>
#include <ATen/ATen.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "launchers.h"

namespace {

void* current_stream(const at::Tensor& t) {
  // Works for both CUDA and ROCm builds (ROCm PyTorch reuses the CUDA device type)
  // without including any CUDA/HIP headers here.
  auto impl = c10::impl::getDeviceGuardImpl(t.device().type());
  return impl->getStream(t.device()).native_handle();
}

kp::DType dtype_of(const at::Tensor& t) {
  if (t.scalar_type() == at::kFloat) return kp::DType::F32;
  if (t.scalar_type() == at::kHalf) return kp::DType::F16;
  TORCH_CHECK(false, "kernelpilot kernels support float32 and float16, got ", t.scalar_type());
}

void check_input(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be a GPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

int check_variant(int64_t v, int64_t num_variants = 3) {
  TORCH_CHECK(v >= 0 && v < num_variants, "variant must be in [0, ", num_variants, ")");
  return static_cast<int>(v);
}

at::Tensor vector_add(const at::Tensor& a, const at::Tensor& b, int64_t variant) {
  check_input(a, "a");
  check_input(b, "b");
  TORCH_CHECK(a.sizes() == b.sizes() && a.scalar_type() == b.scalar_type(),
              "a and b must match in shape and dtype");
  auto c = at::empty_like(a);
  kp::vector_add(check_variant(variant), dtype_of(a), a.data_ptr(), b.data_ptr(), c.data_ptr(),
                 a.numel(), current_stream(a));
  return c;
}

at::Tensor reduce_sum(const at::Tensor& x, int64_t variant) {
  check_input(x, "x");
  auto out = at::zeros({}, x.options().dtype(at::kFloat));
  kp::reduce_sum(check_variant(variant), dtype_of(x), x.data_ptr(), out.data_ptr<float>(),
                 x.numel(), current_stream(x));
  return out;
}

at::Tensor softmax(const at::Tensor& x, int64_t variant) {
  check_input(x, "x");
  TORCH_CHECK(x.dim() >= 1, "softmax needs at least 1 dimension");
  auto y = at::empty_like(x);
  const int64_t N = x.size(-1);
  const int64_t M = N == 0 ? 0 : x.numel() / N;
  TORCH_CHECK(M < (int64_t{1} << 31) && N < (int64_t{1} << 31), "softmax shape too large");
  kp::softmax(check_variant(variant), dtype_of(x), x.data_ptr(), y.data_ptr(), M, N,
              current_stream(x));
  return y;
}

}  // namespace

TORCH_LIBRARY(kp, m) {
  m.def("vector_add(Tensor a, Tensor b, int variant) -> Tensor");
  m.def("reduce_sum(Tensor x, int variant) -> Tensor");
  m.def("softmax(Tensor x, int variant) -> Tensor");
}

TORCH_LIBRARY_IMPL(kp, CUDA, m) {
  m.impl("vector_add", &vector_add);
  m.impl("reduce_sum", &reduce_sum);
  m.impl("softmax", &softmax);
}
