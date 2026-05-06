from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import psutil
import torch

try:
    import pynvml
except Exception:  # pragma: no cover
    pynvml = None


@dataclass
class ResourceSnapshot:
    rss_mb: float
    gpu_mem_mb: float
    gpu_power_w: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "rss_mb": self.rss_mb,
            "gpu_mem_mb": self.gpu_mem_mb,
            "gpu_power_w": -1.0 if self.gpu_power_w is None else self.gpu_power_w,
        }


class ResourceMonitor:
    def __init__(self, device: str) -> None:
        self.device = device
        self.process = psutil.Process()
        self._gpu_index: int | None = None
        if device.startswith("cuda") and torch.cuda.is_available() and pynvml is not None:
            pynvml.nvmlInit()
            self._gpu_index = int(device.split(":")[1]) if ":" in device else 0

    def snapshot(self) -> ResourceSnapshot:
        rss_mb = self.process.memory_info().rss / (1024 * 1024)
        gpu_mem_mb = 0.0
        gpu_power_w: float | None = None
        if self._gpu_index is not None:
            handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_mem_mb = mem.used / (1024 * 1024)
            try:
                gpu_power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
            except Exception:
                gpu_power_w = None
        return ResourceSnapshot(rss_mb=rss_mb, gpu_mem_mb=gpu_mem_mb, gpu_power_w=gpu_power_w)
