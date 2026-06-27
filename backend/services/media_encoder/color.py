"""Color-science helpers for the :class:`MediaEncoder`.

Working space = **Rec.709** (per the color-pipeline addendum §1): the model's
native domain is Rec.709 gamma-domain RGB. The BT.709 transfer pair, the
BT.709/D65 chromaticities, and ``ADOPTED_NEUTRAL_D65`` now live in
:mod:`services.color_management` — the single source of truth — and are
re-exported here for the encoder's convenience. ffmpeg-specific flag builders
(``ffmpeg_bt709_color_flags`` / ``ffmpeg_bt709_matrix_filter``) remain here.

Forward-looking notes (see color_management.py for the full ColorManager):

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

from services.color_management import (
    ADOPTED_NEUTRAL_D65,
    BT709_CHROMATICITIES,
    bt709_eotf,
    bt709_oetf,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


# BT.709 primary xy arrays (kept for backward-compat with existing consumers).
BT709_RED_XY: NDArray[np.float32] = np.array(
    [BT709_CHROMATICITIES[0], BT709_CHROMATICITIES[1]], dtype=np.float32
)
BT709_GREEN_XY: NDArray[np.float32] = np.array(
    [BT709_CHROMATICITIES[2], BT709_CHROMATICITIES[3]], dtype=np.float32
)
BT709_BLUE_XY: NDArray[np.float32] = np.array(
    [BT709_CHROMATICITIES[4], BT709_CHROMATICITIES[5]], dtype=np.float32
)
BT709_WHITE_XY: NDArray[np.float32] = np.array(
    [BT709_CHROMATICITIES[6], BT709_CHROMATICITIES[7]], dtype=np.float32
)

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
