"""Color-science helpers for the :class:`MediaEncoder`.

Working space = **Rec.709** (per the color-pipeline addendum §1): the model's
native domain is Rec.709 gamma-domain RGB. Two concerns live here:

* **BT.709 transfer pair** (ITU-R BT.709-6) — ``bt709_oetf`` (linear→Rec.709
  gamma) and ``bt709_eotf`` (Rec.709 gamma→linear; the inverse OETF, which is the
  compositing standard — NOT BT.1886 display EOTF). EXR is the de-facto linear-
  light container, so the EXR path linearizes decoded Rec.709-gamma frames via
  ``bt709_eotf`` before writing. ProRes/MP4 are display-referred gamma-domain and
  are NOT linearized (the Rec.709-gamma tensor maps directly to tagged Rec.709
  YUV). Per §9.1 this is the ~mid-gray fix: at V=0.5 the BT.709 EOTF yields
  0.259 (sRGB EOTF yielded 0.214 — ~17.5% low).

* Color metadata constants + ffmpeg flag builders for the BT.709 / D65 canonical
  colorspace.

Forward-looking notes for CM-1 (cross-primaries / ColorManager):

* **Matrix orientation** (§9.6): for row-vector arrays ``(..., 3)`` apply a 3×3
  matrix as ``rgb @ M.T`` (NOT ``M @ rgb``).
* **Gamut/range policy** (§9.8): cross-primaries conversions can go out-of-gamut;
  preserve floats with NO hard clip through the working space. Clipping happens
  only at the final encode quantization (a documented lossy step — hard-clipping
  there breaks identity, which is expected).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# BT.709 transfer functions (ITU-R BT.709-6)
# ---------------------------------------------------------------------------

def bt709_oetf(lin: torch.Tensor) -> torch.Tensor:
    """BT.709 OETF: linear scene light → Rec.709 gamma-domain (ITU-R BT.709-6).

    ``V = 4.5L`` for ``L < 0.018``, else ``V = 1.099·L^0.45 − 0.099``.
    Vectorized over any tensor shape. Clamp non-negative before the ``pow`` branch
    to avoid NaNs from fractional powers of negative numbers.
    """
    lin = lin.clamp(min=0.0)
    return torch.where(lin < 0.018, 4.5 * lin, 1.099 * lin.pow(0.45) - 0.099)


def bt709_eotf(v: torch.Tensor) -> torch.Tensor:
    """BT.709 EOTF (inverse OETF): Rec.709 gamma-domain → linear light.

    ``L = V/4.5`` for ``V < 0.081``, else ``L = ((V+0.099)/1.099)^(1/0.45)``.
    This is the compositing-standard inverse OETF — NOT the BT.1886 display EOTF
    (which would apply ~2.4 gamma on top). Used to linearize decoded Rec.709-gamma
    frames before writing linear EXR. Vectorized over any tensor shape.
    """
    return torch.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099).pow(1.0 / 0.45))


# ---------------------------------------------------------------------------
# BT.709 / D65 chromaticities
# ---------------------------------------------------------------------------

# BT.709 primaries + D65 white point, CIE xy chromaticity coordinates.
# These match the canonical SMPTE RP 431-2 / ITU-R BT.709 values used by every
# pro tool (Resolve / Premiere / Nuke / RV).
BT709_RED_XY: NDArray[np.float32] = np.array([0.6400, 0.3300], dtype=np.float32)
BT709_GREEN_XY: NDArray[np.float32] = np.array([0.3000, 0.6000], dtype=np.float32)
BT709_BLUE_XY: NDArray[np.float32] = np.array([0.1500, 0.0600], dtype=np.float32)
BT709_WHITE_XY: NDArray[np.float32] = np.array([0.3127, 0.3290], dtype=np.float32)

# OpenEXR's ``chromaticities`` header attribute is written as an 8-tuple of
# python floats ``(rx, ry, gx, gy, bx, by, wx, wy)`` — verified against the
# installed OpenEXR 3.x binding (``objectToChromaticities`` in PyOpenEXR.cpp;
# the "expected a 6-tuple" error string in the binding is a known misleading
# message — the real requirement is 8 floats).
BT709_CHROMATICITIES: tuple[float, float, float, float, float, float, float, float] = (
    float(BT709_RED_XY[0]),
    float(BT709_RED_XY[1]),
    float(BT709_GREEN_XY[0]),
    float(BT709_GREEN_XY[1]),
    float(BT709_BLUE_XY[0]),
    float(BT709_BLUE_XY[1]),
    float(BT709_WHITE_XY[0]),
    float(BT709_WHITE_XY[1]),
)

# adoptedNeutral (the assumed neutral / source white) — D65.
ADOPTED_NEUTRAL_D65: NDArray[np.float32] = BT709_WHITE_XY

# OIIO/Nuke colorSpace label written as a descriptive header attribute.
# Our output is MORE rigorously tagged than the reference fixtures (which are
# untagged Nuke defaults) — per §0B refined EXR encoder spec.
LINEAR_REC709_SCENE_COLORSPACE: str = "lin_rec709_scene"


# ---------------------------------------------------------------------------
# ffmpeg color flag builders
# ---------------------------------------------------------------------------

def ffmpeg_bt709_color_flags() -> list[str]:
    """Output color tags for an ffmpeg invocation forcing BT.709 / D65 limited range.

    Per §0A.C these tags alone do NOT force swscale's matrix/range — pair with
    :func:`ffmpeg_bt709_matrix_filter` for the explicit conversion.

    Note: ``-chroma_location topleft`` is intentionally omitted — the bundled
    imageio-ffmpeg static build (johnvansickle 7.0.2) does not expose that CLI
    flag and rejects it with "Unrecognized option". References are
    ``unspecified`` anyway (§0B), and assertions accept {unspecified, topleft,
    left}, so omitting it does not weaken the color gate.
    """
    return [
        "-color_range", "tv",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
    ]


def ffmpeg_bt709_matrix_filter() -> str:
    """Explicit RGB→YUV BT.709 matrix + limited-range conversion filter string.

    Per §0A.C, ``-colorspace bt709`` only tags the bitstream; swscale may still
    apply a default (often bt601) matrix and full-range mapping. This filter
    forces the correct limited-range BT.709 matrix on the RGB→YUV conversion so
    decoded luma round-trips without matrix/range drift. The output pixel format
    itself is selected separately via ffmpeg's ``-pix_fmt`` flag.

    Uses the ``scale`` filter's documented ``out_color_matrix`` / ``out_range``
    options (verified against the bundled imageio-ffmpeg 7.0.2 build — that build
    rejects the non-standard ``full_range`` option name).
    """
    return "scale=out_color_matrix=bt709:out_range=tv"


__all__ = [
    "ADOPTED_NEUTRAL_D65",
    "BT709_BLUE_XY",
    "BT709_CHROMATICITIES",
    "BT709_GREEN_XY",
    "BT709_RED_XY",
    "BT709_WHITE_XY",
    "LINEAR_REC709_SCENE_COLORSPACE",
    "bt709_eotf",
    "bt709_oetf",
    "ffmpeg_bt709_color_flags",
    "ffmpeg_bt709_matrix_filter",
]
