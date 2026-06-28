"""Real system-info implementation using psutil + existing GpuInfo.

psutil is an optional import — if unavailable, CPU/RAM metrics are silently
``None`` rather than raising. VRAM metrics come from the existing ``GpuInfo``
service (pynvml-backed). ``sample()`` is best-effort and never raises.
"""

from __future__ import annotations

import logging

from services.gpu_info.gpu_info import GpuInfo
from services.system_info.system_info import SystemTelemetry

logger = logging.getLogger(__name__)


class SystemInfoImpl:
    """psutil + GpuInfo-backed telemetry sampler."""

    def __init__(self, gpu_info: GpuInfo) -> None:
        self._gpu_info = gpu_info
        try:
            import psutil  # type: ignore[import-untyped]

            self._psutil: object | None = psutil
        except ImportError:
            self._psutil = None

    def sample(self) -> SystemTelemetry:
        vram_used_mb: int | None = None
        vram_total_mb: int | None = None
        gpu_util_pct: float | None = None

        try:
            info = self._gpu_info.get_gpu_info()
            vram_total_mb = info.get("vram") or None
            vram_used_mb = info.get("vramUsed") or None
        except Exception:
            logger.debug("VRAM sample failed", exc_info=True)

        ram_used_mb: int | None = None
        ram_total_mb: int | None = None
        cpu_util_pct: float | None = None

        if self._psutil is not None:
            try:
                import psutil  # type: ignore[import-untyped]

                vm = psutil.virtual_memory()
                ram_total_mb = int(vm.total // (1024 * 1024)) or None
                ram_used_mb = int(vm.used // (1024 * 1024)) or None
                # cpu_percent with interval=None returns instantaneous since last call
                cpu_util_pct = psutil.cpu_percent(interval=None)
            except Exception:
                logger.debug("psutil sample failed", exc_info=True)

        return SystemTelemetry(
            vram_used_mb=vram_used_mb,
            vram_total_mb=vram_total_mb,
            gpu_util_pct=gpu_util_pct,
            ram_used_mb=ram_used_mb,
            ram_total_mb=ram_total_mb,
            cpu_util_pct=cpu_util_pct,
        )
