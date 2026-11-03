"""CPU tests of the harness logic itself: the comparator must catch wrong
answers, NaNs, shape/dtype mismatches and in-place mutation."""

import pytest
import torch

from kernelpilot.correctness import KernelMismatch, check_impl, compare
from kernelpilot.device import lookup_peaks
from kernelpilot.specs import SPECS, get_spec


def test_compare_exact():
    x = torch.randn(100)
    info = compare(x, x.clone(), 1e-6, 1e-6)
    assert info["num_mismatched"] == 0 and info["max_abs_err"] == 0


def test_compare_reports_worst_element():
    ref = torch.zeros(4, 5)
    out = ref.clone()
    out[2, 3] = 1.0
    info = compare(out, ref, 1e-3, 1e-3)
    assert info["num_mismatched"] == 1
    assert info["worst_index"] == (2, 3)
    assert info["max_abs_err"] == pytest.approx(1.0)


def test_compare_nan_is_error():
    ref = torch.ones(3)
    out = torch.tensor([1.0, float("nan"), 1.0])
    assert compare(out, ref, 1e-3, 1e-3)["num_mismatched"] == 1


def test_compare_matching_inf_is_ok():
    ref = torch.tensor([float("inf"), 1.0])
    assert compare(ref.clone(), ref, 0, 0)["num_mismatched"] == 0


def test_compare_shape_mismatch_raises():
    with pytest.raises(KernelMismatch):
        compare(torch.zeros(3), torch.zeros(4), 1, 1)


@pytest.mark.parametrize("name", list(SPECS))
def test_torch_baseline_passes_on_cpu(name):
    spec = get_spec(name)
    shape = spec.test_shapes[min(2, len(spec.test_shapes) - 1)]
    res = check_impl(spec, "torch", spec.baseline, shape, torch.float32, "cpu")
    assert res.ok, res.summary()


@pytest.mark.parametrize("name", list(SPECS))
def test_wrong_kernel_is_caught(name):
    spec = get_spec(name)
    shape = spec.test_shapes[min(2, len(spec.test_shapes) - 1)]

    def wrong(*xs):
        out = spec.baseline(*xs)
        out = out.clone()
        out.view(-1)[0] += 1.0  # a single wrong element must be caught
        return out

    res = check_impl(spec, "wrong", wrong, shape, torch.float32, "cpu")
    assert not res.ok
    assert "MISMATCH" in res.summary() and "max_abs_err" in res.summary()


def test_inplace_mutation_is_caught():
    spec = get_spec("vector_add")

    def mutating(a, b):
        a.add_(b)
        return a.clone()

    res = check_impl(spec, "mut", mutating, (16,), torch.float32, "cpu")
    assert not res.ok and "in place" in res.error


def test_wrong_dtype_is_caught():
    spec = get_spec("vector_add")
    res = check_impl(spec, "f32", lambda a, b: (a + b).float(), (16,), torch.float16, "cpu")
    assert not res.ok and "dtype" in res.error


def test_crash_is_reported_not_raised():
    spec = get_spec("vector_add")

    def boom(a, b):
        raise RuntimeError("illegal memory access")

    res = check_impl(spec, "boom", boom, (16,), torch.float32, "cpu")
    assert not res.ok and "illegal memory access" in res.error


def test_peak_lookup():
    assert lookup_peaks("Tesla T4") == (320.0, 8.1, 65.0)
    assert lookup_peaks("NVIDIA L40S") is None  # must not match "L4"
    assert lookup_peaks("NVIDIA H100 PCIe")[0] == 2000.0
    assert lookup_peaks("AMD Instinct MI300X")[0] == 5300.0


@pytest.mark.parametrize("name", list(SPECS))
def test_specs_are_consistent(name):
    spec = get_spec(name)
    for shape in spec.test_shapes + spec.bench_shapes:
        assert len(shape) == len(spec.shape_names)
    for shape in spec.bench_shapes:
        for dt in spec.dtypes:
            assert spec.flops(shape) > 0 and spec.bytes_moved(shape, dt) > 0
