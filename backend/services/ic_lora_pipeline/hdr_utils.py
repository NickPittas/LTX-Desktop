"""HDR utilities adapted from upstream LTX-2 (Lightricks/LTX-2 commit
``780984275fd47128b02bef9b5c085404276866ee``, file ``packages/ltx-core/src/ltx_core/hdr.py``).

Provides:
- LogC3 compress / decompress (ARRI ACES LogC3 transfer function).
- ``apply_hdr_decode_postprocess`` — decompress LogC3 model output → linear
  scene-referred values for EXR storage or SDR proxy tonemapping.
- ``tonemap_for_sdr`` — simple Reinhard tonemap for SDR proxy generation.

The HDR pipeline produces video in **LogC3 compressed** space. This module
recovers linear values and optionally tonemaps for SDR display.
"""

from __future__ import annotations

import math

import torch

# ── ACES / ARRI LogC3 parameters (publicly documented) ──────────────────
# Forward (linear → LogC3):
#   y = log2(a * x + 1) + b        for x ≥ cut
#   y = x * slope                   for x < cut (C0-continuous extension)
# Inverse (LogC3 → linear):
#   x = (2^(y - b) - 1) / a         for y ≥ y_cut
#   x = y / slope                    for y < y_cut

_A: float = 5.67743
_B: float = 0.092499
_CUT: float = 0.010591

# Precomputed: LogC3 output value at the cut point.
_Y_AT_CUT: float = math.log2(_A * _CUT + 1.0) + _B

# Linear-region slope ensures C0 continuity at the cut point.
_LINEAR_SLOPE: float = _Y_AT_CUT / _CUT


def logc3_compress(x: torch.Tensor) -> torch.Tensor:
    """Compress linear scene-referred values to LogC3.

    Input is non-negative linear light (any range). Output is in LogC3
    compressed space (roughly [0, 1] for SDR-range linear inputs).
    """
    return torch.where(
        x >= _CUT,
        torch.log2(_A * x + 1.0) + _B,
        x * _LINEAR_SLOPE,
    )


def logc3_decompress(y: torch.Tensor) -> torch.Tensor:
    """Decompress LogC3 values to linear scene-referred values.

    This is the exact inverse of :func:`logc3_compress`.
    """
    return torch.where(
        y >= _Y_AT_CUT,
        (torch.pow(2.0, y - _B) - 1.0) / _A,
        y / _LINEAR_SLOPE,
    )


def apply_hdr_decode_postprocess(video: torch.Tensor) -> torch.Tensor:
    """Decompress LogC3-encoded HDR model output to linear scene-referred values.

    The HDR IC-LoRA pipeline produces output in LogC3 compressed space. This
    function applies LogC3 decompression to recover linear values suitable for
    linear EXR storage. Values are clamped to ≥ 0 (no negative light).

    Adapted from upstream ``apply_hdr_decode_postprocess``.
    """
    return logc3_decompress(video).clamp(min=0.0)


def tonemap_for_sdr(linear: torch.Tensor) -> torch.Tensor:
    """Simple Reinhard tonemap for SDR proxy generation from linear HDR.

    Maps [0, ∞) → [0, 1) via ``x / (x + 1)``. Suitable for generating a
    browser-playable SDR proxy from HDR linear output.
    """
    return (linear / (linear + 1.0)).clamp(0.0, 1.0)


__all__ = [
    "logc3_compress",
    "logc3_decompress",
    "apply_hdr_decode_postprocess",
    "tonemap_for_sdr",
]
