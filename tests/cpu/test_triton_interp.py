"""Hand-written Triton kernels run on CPU via Triton's interpreter
(TRITON_INTERPRET=1), checked with the same harness the GPU tests use.
Shapes are capped because the interpreter is slow."""

import math
import os

import pytest
import torch

pytestmark = pytest.mark.triton_interp

if os.environ.get("TRITON_INTERPRET") != "1":
    pytest.skip("set TRITON_INTERPRET=1 to run Triton kernels on CPU", allow_module_level=True)
pytest.importorskip("triton")

from kernelpilot import registry  # noqa: E402
from kernelpilot.correctness import check_impl  # noqa: E402
from kernelpilot.specs import SPECS  # noqa: E402

MAX_NUMEL = 1 << 16


def _cases():
    for name, spec in SPECS.items():
        for impl in registry.triton_impls(name):
            for dtype in spec.dtypes:
                for shape in spec.test_shapes:
                    if math.prod(shape) <= MAX_NUMEL:
                        yield pytest.param(name, impl, dtype, shape,
                                           id=f"{name}-{impl}-{str(dtype)[6:]}-{shape}")


@pytest.mark.parametrize("kernel,impl,dtype,shape", list(_cases()))
def test_triton_kernel_interpreted(kernel, impl, dtype, shape):
    spec = SPECS[kernel]
    fn = registry.triton_impls(kernel)[impl]
    res = check_impl(spec, impl, fn, shape, dtype, "cpu")
    assert res.ok, res.summary()
