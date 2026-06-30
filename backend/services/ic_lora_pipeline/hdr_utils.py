"""HDR utilities mirroring official LTX-2 ``ltx_core.hdr`` semantics.

Provides the official ARRI ALEXA LogC3 transfer (forward ``compress`` /
inverse ``decompress``) and the ``apply_hdr_decode_postprocess`` entry point
used by the HDR IC-LoRA pipeline to recover linear scene-referred values
from the model's LogC3-domain VAE output.

Official LogC3 constants (``.slim/deepwork/hdr-v2v-input-repair.md`` oracle
blocker #3, mirroring ``ltx_core.hdr``):
    A = 5.555556, B = 0.052272, C = 0.247190, D = 0.385537,
    E = 5.367655, F = 0.092809, CUT = 0.010591.

Forward (linear scene-referred → LogC3), C0-continuous piecewise::

    y = C * log10(A*x + B) + D     for x ≥ CUT
    y = E*x + F                    for x < CUT

Inverse (LogC3 → linear scene-referred)::

    x = (10^((y - D)/C) - B) / A   for y ≥ y_cut
    x = (y - F) / E                for y < y_cut

where ``y_cut = E*CUT + F`` (the LogC3 value at the cut point, so both
branches agree at ``x = CUT``).

The LDR (SDR 8-bit source) domain is treated as clamp-identity over [0, 1]:
``compress_ldr`` and ``decompress_ldr`` are no-ops other than clamping. This
matches the official HDR workflow, where the 8-bit source video is fed to the
VAE in its native SDR domain and only the model OUTPUT is LogC3-encoded.
"""

from __future__ import annotations

from typing import Literal

import torch

# ── Official ARRI ALEXA LogC3 constants ─────────────────────────────────
_A: float = 5.555556
_B: float = 0.052272
_C: float = 0.247190
_D: float = 0.385537
_E: float = 5.367655
_F: float = 0.092809
_CUT: float = 0.010591

# LogC3 output value at the cut point (linear branch evaluated at x=CUT).
# Both branches agree here, so the piecewise join is C0-continuous.
_Y_AT_CUT: float = _E * _CUT + _F

#: Supported HDR decode transforms. ``logc3`` is the only official HDR path.
HdrTransform = Literal["logc3"]


class LogC3:
    """Official ARRI ALEXA LogC3 forward/inverse transfer (stateless).

    All methods are element-wise and preserve tensor shape/dtype/device.
    """

    @staticmethod
    def compress(hdr: torch.Tensor) -> torch.Tensor:
        """Map linear HDR ``[0, ∞)`` → LogC3 ``[0, 1]`` (approximate).

        Official forward (``ltx_core.hdr.LogC3.compress``). Uses log10 with
        the documented A/B/C/D constants for ``x ≥ CUT`` and the linear toe
        (E/F) below the cut so the curve is C0-continuous.
        """
        x = hdr
        log_branch = _C * torch.log10(_A * x + _B) + _D
        lin_branch = _E * x + _F
        return torch.where(x >= _CUT, log_branch, lin_branch)

    @staticmethod
    def decompress(logc: torch.Tensor) -> torch.Tensor:
        """Map LogC3 ``[0, 1]`` → linear HDR ``[0, ∞)``.

        Exact inverse of :meth:`compress` (official ``LogC3.decompress``).
        """
        y = logc
        log_branch = (torch.pow(10.0, (y - _D) / _C) - _B) / _A
        lin_branch = (y - _F) / _E
        return torch.where(y >= _Y_AT_CUT, log_branch, lin_branch)


def compress_ldr(ldr: torch.Tensor) -> torch.Tensor:
    """LDR (SDR 8-bit source) → model domain: clamp-identity over [0, 1].

    Official HDR treats the 8-bit source video as already in the model's LDR
    domain (no gamma/LogC3 transform on the conditioning path). This helper
    exists so the conditioning loader mirrors the official
    ``compress_ldr`` / ``hdr_transform`` plumbing.
    """
    return ldr.clamp(0.0, 1.0)


def decompress_ldr(ldr: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`compress_ldr` — clamp-identity over [0, 1]."""
    return ldr.clamp(0.0, 1.0)


def apply_hdr_decode_postprocess(
    decoded_video: torch.Tensor,
    transform: HdrTransform = "logc3",
) -> torch.Tensor:
    """Recover linear scene-referred HDR from the model's VAE-decoded output.

    Mirrors official ``apply_hdr_decode_postprocess``:

    - Input ``decoded_video`` is the raw VAE decoder output in ``[0, 1]``,
      shape ``[B, C, F, H, W]`` (model-domain, pre-EOTF).
    - ``transform="logc3"`` applies :meth:`LogC3.decompress` element-wise,
      returning a linear HDR float tensor (values may exceed 1.0). Negative
      values are clamped to 0 (no negative light).
    - Any other ``transform`` value raises ``ValueError`` — no silent fallback
      to a different transfer function.

    The returned linear tensor is what the EXR primary stores and what the
    SDR proxy tonemaps from.
    """
    if transform != "logc3":
        raise ValueError(
            f"Unsupported HDR decode transform: {transform!r}. Only 'logc3' is supported."
        )
    return LogC3.decompress(decoded_video).clamp(min=0.0)


def tonemap_for_sdr(linear: torch.Tensor) -> torch.Tensor:
    """Simple Reinhard tonemap for SDR proxy generation from linear HDR.

    Maps ``[0, ∞) → [0, 1)`` via ``x / (x + 1)``. Suitable for generating a
    browser-playable SDR proxy from HDR linear output.
    """
    return (linear / (linear + 1.0)).clamp(0.0, 1.0)


__all__ = [
    "HdrTransform",
    "LogC3",
    "compress_ldr",
    "decompress_ldr",
    "apply_hdr_decode_postprocess",
    "tonemap_for_sdr",
]
