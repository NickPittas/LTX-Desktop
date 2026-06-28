"""System resource telemetry protocol (metrics/progress Phase).

Provides a ``SystemInfo`` protocol that samples live resource utilization
(VRAM, RAM, CPU, GPU utilization) for display during generation. The protocol
is intentionally minimal: a single ``sample()`` call returns a typed snapshot.

Per oracle decision: metrics are live-only (not persisted), sampled by a
background sampler at ~1 Hz, and never alter workflow algorithms.
"""

from __future__ import annotations

from typing import Protocol, TypedDict


class SystemTelemetry(TypedDict):
    """Resource utilization snapshot at a point in time.

    All values are ``None`` when unavailable (e.g. no GPU, psutil missing).
    VRAM/RAM are in megabytes; utilization values are percentages (0–100).
    """

    vram_used_mb: int | None
    vram_total_mb: int | None
    gpu_util_pct: float | None
    ram_used_mb: int | None
    ram_total_mb: int | None
    cpu_util_pct: float | None


class SystemInfo(Protocol):
    """Protocol for sampling system resource telemetry."""

    def sample(self) -> SystemTelemetry:
        """Return a best-effort resource snapshot. Never raises."""
        ...
