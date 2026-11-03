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

at::Tensor matmul(const at::Tensor& a, const at::Tensor& b, int64_t variant) {
  check_input(a, "a");
  check_input(b, "b");
  TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.size(1) == b.size(0),
              "matmul expects (M, K) @ (K, N), got ", a.sizes(), " and ", b.sizes());
  TORCH_CHECK(a.scalar_type() == b.scalar_type(), "a and b must have the same dtype");
  const int64_t M = a.size(0), K = a.size(1), N = b.size(1);
  TORCH_CHECK(M < (int64_t{1} << 31) && N < (int64_t{1} << 31) && K < (int64_t{1} << 31),
              "matmul shape too large");
  auto c = at::empty({M, N}, a.options());
  if (K == 0) return c.zero_();
  kp::matmul(check_variant(variant, 4), dtype_of(a), a.data_ptr(), b.data_ptr(), c.data_ptr(), M,
             N, K, current_stream(a));
  return c;
}

at::Tensor layernorm(const at::Tensor& x, const at::Tensor& w, const at::Tensor& b, double eps,
                     int64_t variant) {
  check_input(x, "x");
  check_input(w, "weight");
  check_input(b, "bias");
  const int64_t N = x.size(-1);
  TORCH_CHECK(w.numel() == N && b.numel() == N, "weight and bias must have size ", N);
  TORCH_CHECK(w.scalar_type() == x.scalar_type() && b.scalar_type() == x.scalar_type(),
              "x, weight and bias must share a dtype");
  auto y = at::empty_like(x);
  const int64_t M = N == 0 ? 0 : x.numel() / N;
  TORCH_CHECK(M < (int64_t{1} << 31), "layernorm shape too large");
  kp::layernorm(check_variant(variant), dtype_of(x), x.data_ptr(), w.data_ptr(), b.data_ptr(),
                y.data_ptr(), M, N, static_cast<float>(eps), current_stream(x));
  return y;
}

at::Tensor attention(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                     int64_t variant) {
  check_input(q, "q");
  check_input(k, "k");
  check_input(v, "v");
  TORCH_CHECK(q.dim() == 4 && q.sizes() == k.sizes() && q.sizes() == v.sizes(),
              "q, k, v must all be (B, H, S, D)");
  TORCH_CHECK(q.scalar_type() == k.scalar_type() && q.scalar_type() == v.scalar_type(),
              "q, k, v must share a dtype");
  const int64_t BH = q.size(0) * q.size(1), S = q.size(2), D = q.size(3);
  TORCH_CHECK(D <= 256 && BH < 65536, "attention supports D <= 256 and B*H < 65536");
  auto o = at::empty_like(q);
  kp::attention(check_variant(variant, 2), dtype_of(q), q.data_ptr(), k.data_ptr(), v.data_ptr(),
                o.data_ptr(), BH, S, D, current_stream(q));
  return o;
}

}  // namespace

TORCH_LIBRARY(kp, m) {
  m.def("vector_add(Tensor a, Tensor b, int variant) -> Tensor");
  m.def("reduce_sum(Tensor x, int variant) -> Tensor");
  m.def("softmax(Tensor x, int variant) -> Tensor");
  m.def("matmul(Tensor a, Tensor b, int variant) -> Tensor");
  m.def("layernorm(Tensor x, Tensor weight, Tensor bias, float eps, int variant) -> Tensor");
  m.def("attention(Tensor q, Tensor k, Tensor v, int variant) -> Tensor");
}

TORCH_LIBRARY_IMPL(kp, CUDA, m) {
  m.impl("vector_add", &vector_add);
  m.impl("reduce_sum", &reduce_sum);
  m.impl("softmax", &softmax);
  m.impl("matmul", &matmul);
  m.impl("layernorm", &layernorm);
  m.impl("attention", &attention);
}
