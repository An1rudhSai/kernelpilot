"""Every hand-written kernel against the PyTorch reference, across all test
shapes and dtypes. Runs only on a machine with a GPU (`make test`)."""

import pytest
import torch

from kernelpilot import registry
from kernelpilot.correctness import check_impl
from kernelpilot.specs import SPECS

pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("no GPU available", allow_module_level=True)


def _cases():
    for name, spec in SPECS.items():
        for impl in registry.get_impls(name):
            for dtype in spec.dtypes:
                for shape in spec.test_shapes:
                    yield pytest.param(name, impl, dtype, shape,
                                       id=f"{name}-{impl}-{str(dtype)[6:]}-{shape}")


@pytest.mark.parametrize("kernel,impl,dtype,shape", list(_cases()))
def test_kernel_matches_reference(kernel, impl, dtype, shape):
    spec = SPECS[kernel]
    fn = registry.get_impls(kernel)[impl]
    res = check_impl(spec, impl, fn, shape, dtype, "cuda")
    assert res.ok, res.summary()


def test_hip_extension_built():
    from kernelpilot import ext
    assert ext.load(), "HIP extension missing: run `make build`"
