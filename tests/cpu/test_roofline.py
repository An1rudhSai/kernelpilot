"""The roofline and markdown writers run on CPU with hand-made (fake) results."""

from kernelpilot.benchmark import BenchResult, load_results, save_results, to_markdown
from kernelpilot.device import GPUInfo
from kernelpilot.roofline import plot_roofline

GPU = GPUInfo(name="Fake GPU", vendor="nvidia", peak_bw_gbs=300.0,
              peak_tflops={"float32": 8.0, "float16": 65.0}, source="test")


def _fake_results():
    return [
        BenchResult("vector_add", "torch", "float32", (1 << 24,), ms=0.7, gbs=280, tflops=0.023,
                    arithmetic_intensity=1 / 12),
        BenchResult("matmul", "hip:vectorized", "float32", (2048, 2048, 2048), ms=4.0,
                    gbs=12, tflops=4.3, arithmetic_intensity=341),
        BenchResult("softmax", "hip:naive", "float32", (4096, 4096), ms=None, correct=False),
    ]


def test_roofline_png(tmp_path):
    out = plot_roofline(_fake_results(), GPU, tmp_path / "r.png")
    assert out.exists() and out.stat().st_size > 10_000


def test_results_roundtrip_and_markdown(tmp_path):
    save_results(_fake_results(), GPU, tmp_path)
    gpu, results = load_results(tmp_path / "bench.json")
    assert gpu.name == "Fake GPU" and len(results) == 3
    md = to_markdown(results, gpu)
    assert "FAILED correctness" in md and "### matmul" in md
