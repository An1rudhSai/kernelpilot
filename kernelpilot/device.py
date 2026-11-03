"""GPU detection and theoretical peaks used for %-of-peak and roofline plots."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

import torch

# Datasheet peaks: (memory GB/s, FP32 TFLOPS, FP16 TFLOPS).
# FP16 is the dense matrix-core (tensor core / MFMA) peak, which the hand-written
# kernels here do not use; it is still the honest ceiling PyTorch can reach.
# Matched as a whole word against torch.cuda.get_device_name(), first match wins.
KNOWN_GPUS: list[tuple[str, float, float, float]] = [
    ("T4", 320.0, 8.1, 65.0),
    ("V100", 900.0, 15.7, 125.0),
    ("A100-SXM4-80GB", 2039.0, 19.5, 312.0),
    ("A100 80GB", 1935.0, 19.5, 312.0),
    ("A100", 1555.0, 19.5, 312.0),
    ("L4", 300.0, 30.3, 121.0),
    ("H100 PCIe", 2000.0, 51.0, 756.0),
    ("H100", 3350.0, 67.0, 989.0),
    ("RTX 4090", 1008.0, 82.6, 165.2),
    ("MI300X", 5300.0, 163.4, 1307.4),
    ("MI250X", 1638.0, 47.9, 383.0),
    ("MI250", 1638.0, 45.3, 362.1),
    ("MI210", 1638.0, 22.6, 181.0),
    ("MI100", 1228.0, 23.1, 184.6),
    ("7900 XTX", 960.0, 61.4, 122.8),
]


@dataclass
class GPUInfo:
    name: str
    vendor: str  # "nvidia", "amd" or "none"
    peak_bw_gbs: float | None
    peak_tflops: dict[str, float] = field(default_factory=dict)
    source: str = "unknown"

    def peak_for(self, dtype: torch.dtype) -> float | None:
        key = "float16" if dtype == torch.float16 else "float32"
        return self.peak_tflops.get(key)

    def describe(self) -> str:
        bw = f"{self.peak_bw_gbs:.0f} GB/s" if self.peak_bw_gbs else "unknown BW"
        fl = ", ".join(f"{k} {v:.1f} TFLOPS" for k, v in self.peak_tflops.items()) or "unknown"
        return f"{self.name} ({self.vendor}): {bw}, {fl} [{self.source}]"


def gpu_available() -> bool:
    return torch.cuda.is_available()


def vendor() -> str:
    if not torch.cuda.is_available():
        return "none"
    return "amd" if torch.version.hip else "nvidia"


def lookup_peaks(name: str) -> tuple[float, float, float] | None:
    for key, bw, fp32, fp16 in KNOWN_GPUS:
        if re.search(rf"\b{re.escape(key)}\b", name, flags=re.IGNORECASE):
            return bw, fp32, fp16
    return None


def measure_bandwidth_gbs(nbytes: int = 1 << 28, iters: int = 20) -> float:
    """Device-to-device copy bandwidth, a practical stand-in for an unknown peak."""
    src = torch.empty(nbytes // 4, dtype=torch.float32, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src)
    end.record()
    torch.cuda.synchronize()
    seconds = start.elapsed_time(end) / 1e3 / iters
    return 2 * nbytes / seconds / 1e9  # read + write


def get_gpu_info() -> GPUInfo:
    if not torch.cuda.is_available():
        return GPUInfo(name="cpu", vendor="none", peak_bw_gbs=None, source="no GPU")
    name = torch.cuda.get_device_name()
    info = GPUInfo(name=name, vendor=vendor(), peak_bw_gbs=None)
    known = lookup_peaks(name)
    if known:
        bw, fp32, fp16 = known
        info.peak_bw_gbs = bw
        info.peak_tflops = {"float32": fp32, "float16": fp16}
        info.source = "datasheet"
    # Environment overrides win, so users can correct or supply unknown GPUs.
    env_bw = os.environ.get("KP_PEAK_BW_GBS")
    env_fp32 = os.environ.get("KP_PEAK_FP32_TFLOPS")
    env_fp16 = os.environ.get("KP_PEAK_FP16_TFLOPS")
    if env_bw:
        info.peak_bw_gbs, info.source = float(env_bw), "env"
    if env_fp32:
        info.peak_tflops["float32"], info.source = float(env_fp32), "env"
    if env_fp16:
        info.peak_tflops["float16"], info.source = float(env_fp16), "env"
    if info.peak_bw_gbs is None:
        info.peak_bw_gbs = measure_bandwidth_gbs()
        info.source = "measured copy bandwidth (set KP_PEAK_* to override)"
    return info
