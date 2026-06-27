"""Color-management math — single source of truth for color science.

Pure functions (numpy/torch), no pipeline/decode coupling. Consumed by both the
encoder (``media_encoder``) and the future input pipeline (CM-1b chokepoints).

Working space = **Rec.709** (gamma-domain RGB, per the color-pipeline addendum
§1). This module owns:

* **Transfer functions** (BT.709, sRGB, gamma-2.4; PQ/HLG gated stubs) — exact,
  vectorized over numpy OR torch.
* **Chromaticity tables** + :class:`ColorSpace` registry (Rec.709/sRGB,
  Display-P3-D65, Rec.2020, ACES AP0, ACEScg/AP1, BT.601).
* **Cross-primaries matrices** (SMPTE RP 177 + Bradford CAT).
* **Detection** (:func:`detect_colorspace`) for EXR (OpenEXR chromaticities) and
  video (ffprobe VUI).
* **ColorTransform** (``to_model_domain`` / ``from_model_domain``) — the input↔
  working-space and working↔output transforms.

Gamut/range policy (§9.8): cross-primaries conversions can go out-of-gamut; we
preserve floats through the working space with NO hard clip. Clipping happens
only at the final encode quantization (a documented, expected lossy step).

Matrix orientation (§9.6): all 3×3 matrices here map RGB column vectors to XYZ
(``M @ rgb_col``); apply to row-vector arrays ``(..., 3)`` as ``rgb @ M.T``.
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import torch

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Array type aliases
# ---------------------------------------------------------------------------

# A transfer function accepts either backend. Overloads below pin the return to
# the same backend that was passed in.
XY = tuple[float, float]
Primaries = tuple[XY, XY, XY, XY]  # (red, green, blue, white) — each (x, y)

TransferKey = Literal["linear", "bt709", "srgb", "gamma24", "pq", "hlg"]


# ---------------------------------------------------------------------------
# Transfer functions (exact)
# ---------------------------------------------------------------------------
#
# Each pair is the OETF (linear→gamma) and EOTF (gamma→linear). Polymorphic over
# torch.Tensor / np.ndarray; @overload pins the return type to the input backend
# so callers (e.g. media_encoder) keep precise torch types.

@overload
def bt709_oetf(lin: torch.Tensor) -> torch.Tensor: ...
@overload
def bt709_oetf(lin: np.ndarray) -> np.ndarray: ...
def bt709_oetf(lin: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """BT.709 OETF: linear scene light → Rec.709 gamma (ITU-R BT.709-6).

    ``V = 4.5L`` for ``L < 0.018``, else ``V = 1.099·L^0.45 − 0.099``.
    Clamp non-negative before the ``pow`` branch (fractional powers of negatives
    are NaN). At L=0.5 → V≈0.7055 (note: NOT 0.409; that value is erroneous —
    the authoritative inverse is bt709_eotf(0.5)=0.2596, §9.1).
    """
    if isinstance(lin, torch.Tensor):
        t = lin.clamp(min=0.0)
        return torch.where(t < 0.018, 4.5 * t, 1.099 * t.pow(0.45) - 0.099)
    a = np.clip(np.asarray(lin, dtype=np.float64), 0.0, None)
    return np.where(a < 0.018, 4.5 * a, 1.099 * np.power(a, 0.45) - 0.099)


@overload
def bt709_eotf(v: torch.Tensor) -> torch.Tensor: ...
@overload
def bt709_eotf(v: np.ndarray) -> np.ndarray: ...
def bt709_eotf(v: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """BT.709 EOTF (inverse OETF): Rec.709 gamma → linear (compositing standard).

    ``L = V/4.5`` for ``V < 0.081``, else ``L = ((V+0.099)/1.099)^(1/0.45)``.
    NOT the BT.1886 display EOTF. At V=0.5 → L≈0.2596 (§9.1).
    """
    if isinstance(v, torch.Tensor):
        return torch.where(v < 0.081, v / 4.5, ((v + 0.099) / 1.099).pow(1.0 / 0.45))
    a = np.asarray(v, dtype=np.float64)
    return np.where(a < 0.081, a / 4.5, np.power((a + 0.099) / 1.099, 1.0 / 0.45))


@overload
def srgb_oetf(lin: torch.Tensor) -> torch.Tensor: ...
@overload
def srgb_oetf(lin: np.ndarray) -> np.ndarray: ...
def srgb_oetf(lin: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """sRGB OETF (IEC 61966-2-1): linear → sRGB gamma. For sRGB-tagged image inputs."""
    if isinstance(lin, torch.Tensor):
        t = lin.clamp(min=0.0)
        return torch.where(t <= 0.0031308, 12.92 * t, 1.055 * t.pow(1.0 / 2.4) - 0.055)
    a = np.clip(np.asarray(lin, dtype=np.float64), 0.0, None)
    return np.where(a <= 0.0031308, 12.92 * a, 1.055 * np.power(a, 1.0 / 2.4) - 0.055)


@overload
def srgb_eotf(v: torch.Tensor) -> torch.Tensor: ...
@overload
def srgb_eotf(v: np.ndarray) -> np.ndarray: ...
def srgb_eotf(v: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """sRGB EOTF (IEC 61966-2-1): sRGB gamma → linear. Threshold 0.04045 / exp 2.4."""
    if isinstance(v, torch.Tensor):
        return torch.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055).pow(2.4))
    a = np.asarray(v, dtype=np.float64)
    return np.where(a <= 0.04045, a / 12.92, np.power((a + 0.055) / 1.055, 2.4))


@overload
def gamma24_oetf(lin: torch.Tensor) -> torch.Tensor: ...
@overload
def gamma24_oetf(lin: np.ndarray) -> np.ndarray: ...
def gamma24_oetf(lin: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Pure power-2.4 OETF: linear → gamma 2.4 (DCI/P3-gamma-style)."""
    if isinstance(lin, torch.Tensor):
        t = lin.clamp(min=0.0)
        return t.pow(1.0 / 2.4)
    a = np.clip(np.asarray(lin, dtype=np.float64), 0.0, None)
    return np.power(a, 1.0 / 2.4)


@overload
def gamma24_eotf(v: torch.Tensor) -> torch.Tensor: ...
@overload
def gamma24_eotf(v: np.ndarray) -> np.ndarray: ...
def gamma24_eotf(v: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Pure power-2.4 EOTF: gamma 2.4 → linear."""
    if isinstance(v, torch.Tensor):
        return v.clamp(min=0.0).pow(2.4)
    a = np.clip(np.asarray(v, dtype=np.float64), 0.0, None)
    return np.power(a, 2.4)


def _hdr_unsupported(name: str, *_args: object, **_kwargs: object) -> Any:
    raise NotImplementedError(
        f"{name} transfer is gated — HDR (PQ/HLG) is not supported; "
        "HDR IC-LoRA is a separate workstream. Do not silently mishandle HDR."
    )


# PQ/HLG are intentionally stubs (HDR deferred — §8, gated). They share a raising
# signature so accidental use fails loudly instead of silently mishandling HDR.
def pq_oetf(lin: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """PQ (SMPTE ST 2084) OETF — HDR, gated (raises NotImplementedError)."""
    return _hdr_unsupported("pq_oetf", lin)


def pq_eotf(v: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """PQ (SMPTE ST 2084) EOTF — HDR, gated (raises NotImplementedError)."""
    return _hdr_unsupported("pq_eotf", v)


def hlg_oetf(lin: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """HLG (BT.2100) OETF — HDR, gated (raises NotImplementedError)."""
    return _hdr_unsupported("hlg_oetf", lin)


def hlg_eotf(v: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """HLG (BT.2100) EOTF — HDR, gated (raises NotImplementedError)."""
    return _hdr_unsupported("hlg_eotf", v)


def _identity(rgb: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Identity transfer (linear → linear)."""
    return rgb


# Transfer registry — maps a ColorSpace.transfer key to its (OETF, EOTF) pair.
_TRANSFER_OETF: dict[str, Callable[..., Any]] = {
    "linear": _identity,
    "bt709": bt709_oetf,
    "srgb": srgb_oetf,
    "gamma24": gamma24_oetf,
    "pq": pq_oetf,
    "hlg": hlg_oetf,
}
_TRANSFER_EOTF: dict[str, Callable[..., Any]] = {
    "linear": _identity,
    "bt709": bt709_eotf,
    "srgb": srgb_eotf,
    "gamma24": gamma24_eotf,
    "pq": pq_eotf,
    "hlg": hlg_eotf,
}


def _apply_eotf(rgb: np.ndarray | torch.Tensor, transfer: TransferKey) -> np.ndarray | torch.Tensor:
    """Linearize ``rgb`` from ``transfer`` gamma-domain → linear (identity if linear)."""
    return _TRANSFER_EOTF[transfer](rgb)


def _apply_oetf(rgb: np.ndarray | torch.Tensor, transfer: TransferKey) -> np.ndarray | torch.Tensor:
    """Apply ``transfer`` OETF to linear ``rgb`` → gamma-domain (identity if linear)."""
    return _TRANSFER_OETF[transfer](rgb)


# ---------------------------------------------------------------------------
# Chromaticity tables + ColorSpace registry
# ---------------------------------------------------------------------------

# Standard CIE xy chromaticities. D65 white = (0.3127, 0.3290); ACES white ≈
# (0.32168, 0.33767).
_D65: XY = (0.3127, 0.3290)
_ACES_WHITE: XY = (0.32168, 0.33767)

_REC709_PRIMARIES: Primaries = (
    (0.6400, 0.3300),  # red
    (0.3000, 0.6000),  # green
    (0.1500, 0.0600),  # blue
    _D65,
)
_P3_D65_PRIMARIES: Primaries = (
    (0.680, 0.320),
    (0.265, 0.690),
    (0.150, 0.060),
    _D65,
)
_REC2020_PRIMARIES: Primaries = (
    (0.708, 0.292),
    (0.170, 0.797),
    (0.131, 0.046),
    _D65,
)
_ACES_AP0_PRIMARIES: Primaries = (
    (0.7347, 0.2653),
    (0.0000, 1.0000),
    (0.0001, -0.0770),
    _ACES_WHITE,
)
_ACES_CG_PRIMARIES: Primaries = (  # AP1
    (0.713, 0.293),
    (0.165, 0.830),
    (0.128, 0.044),
    _ACES_WHITE,
)
# BT.601 (525-line / SMPTE 170M, "smpte170m") — primaries DIFFER from BT.709 (§9.7).
_BT601_525_PRIMARIES: Primaries = (
    (0.640, 0.340),
    (0.310, 0.595),
    (0.155, 0.070),
    _D65,
)
# BT.601 (625-line / bt470bg) primaries.
_BT601_625_PRIMARIES: Primaries = (
    (0.64, 0.33),
    (0.29, 0.60),
    (0.15, 0.06),
    _D65,
)


@dataclass(frozen=True)
class ColorSpace:
    """A colorspace: CIE xy primaries+white + a transfer-function key.

    ``primaries`` is (red, green, blue, white) each (x, y); ``white`` is the
    white point (== primaries[3], duplicated for convenience). ``transfer`` is a
    key into the transfer registry: "linear" / "bt709" / "srgb" / "gamma24" /
    "pq" / "hlg".
    """

    name: str
    primaries: Primaries
    white: XY
    transfer: TransferKey


# Canonical ColorSpace instances.
LINEAR_REC709: ColorSpace = ColorSpace(
    name="lin_rec709", primaries=_REC709_PRIMARIES, white=_D65, transfer="linear"
)
REC709: ColorSpace = ColorSpace(
    name="rec709", primaries=_REC709_PRIMARIES, white=_D65, transfer="bt709"
)
SRGB: ColorSpace = ColorSpace(
    name="srgb", primaries=_REC709_PRIMARIES, white=_D65, transfer="srgb"
)
DISPLAY_P3_D65: ColorSpace = ColorSpace(
    name="display_p3_d65", primaries=_P3_D65_PRIMARIES, white=_D65, transfer="gamma24"
)
REC2020: ColorSpace = ColorSpace(
    name="rec2020", primaries=_REC2020_PRIMARIES, white=_D65, transfer="bt709"
)
LINEAR_REC2020: ColorSpace = ColorSpace(
    name="lin_rec2020", primaries=_REC2020_PRIMARIES, white=_D65, transfer="linear"
)
ACES_AP0: ColorSpace = ColorSpace(
    name="aces_ap0", primaries=_ACES_AP0_PRIMARIES, white=_ACES_WHITE, transfer="linear"
)
ACES_CG: ColorSpace = ColorSpace(
    name="aces_cg", primaries=_ACES_CG_PRIMARIES, white=_ACES_WHITE, transfer="linear"
)
BT601_525: ColorSpace = ColorSpace(
    name="bt601_525", primaries=_BT601_525_PRIMARIES, white=_D65, transfer="bt709"
)
BT601_625: ColorSpace = ColorSpace(
    name="bt601_625", primaries=_BT601_625_PRIMARIES, white=_D65, transfer="bt709"
)

# Registry of named spaces, used by detection (primaries/white matching).
COLOR_SPACES: dict[str, ColorSpace] = {
    cs.name: cs
    for cs in (
        LINEAR_REC709, REC709, SRGB, DISPLAY_P3_D65,
        REC2020, LINEAR_REC2020, ACES_AP0, ACES_CG,
        BT601_525, BT601_625,
    )
}


def _primaries_match(a: Primaries, b: Primaries, atol: float) -> bool:
    """True iff all 4 xy pairs of ``a`` match ``b`` within ``atol`` (per-coord)."""
    return all(
        abs(a[i][0] - b[i][0]) <= atol and abs(a[i][1] - b[i][1]) <= atol
        for i in range(4)
    )


def _match_named_primaries(primaries: Primaries, atol: float = 5e-4) -> ColorSpace | None:
    """Find a registry ColorSpace whose primaries match within ``atol``.

    Returns the linear variant when multiple share primaries (e.g. REC709/SRGB/
    LINEAR_REC709 all share Rec.709 primaries → returns LINEAR_REC709, since EXR
    detection prefers the linear representative). Detection tolerance per §9.5
    is 5e-4 (not 1e-4 — some rounded EXR metadata).
    """
    # Preferred representatives for shared-primaries sets.
    preferred_linear: tuple[str, ...] = (
        "lin_rec709", "lin_rec2020", "aces_ap0", "aces_cg",
    )
    candidates = [
        cs for cs in COLOR_SPACES.values() if _primaries_match(cs.primaries, primaries, atol)
    ]
    if not candidates:
        return None
    for name in preferred_linear:
        for cs in candidates:
            if cs.name == name:
                return cs
    return candidates[0]


# ---------------------------------------------------------------------------
# Cross-primaries matrices (SMPTE RP 177 + Bradford CAT)
# ---------------------------------------------------------------------------

# Bradford cone-response matrix (fixed) and its inverse.
_MA_BRADFORD: np.ndarray = np.array([
    [0.8951, 0.2664, -0.1614],
    [-0.7502, 1.7135, 0.0367],
    [0.0389, -0.0685, 1.0295],
], dtype=np.float64)
_MA_BRADFORD_INV: np.ndarray = np.linalg.inv(_MA_BRADFORD)


def xy_to_xyz(x: float, y: float) -> np.ndarray:
    """CIE xy chromaticity → XYZ tristimulus with Y=1 (shape (3,), float64)."""
    if y == 0.0:
        raise ValueError("chromaticity y must be non-zero")
    return np.array([x / y, 1.0, (1.0 - x - y) / y], dtype=np.float64)


def rgb_to_xyz_matrix(primaries: Primaries) -> np.ndarray:
    """RGB→XYZ matrix (3×3, float64) for the given primaries (SMPTE RP 177).

    Columns are the white-scaled primary XYZ vectors; for an RGB column vector,
    ``XYZ = M @ rgb_col``. Apply to row-vector arrays as ``rgb @ M.T`` (§9.6).
    White scaling solved via ``np.linalg.solve`` so the white point maps to the
    chromaticity-derived XYZ_w (Y=1).
    """
    primary_xyz = np.stack(
        [xy_to_xyz(primaries[i][0], primaries[i][1]) for i in range(3)], axis=1
    )  # 3×3, columns = unscaled primary XYZ (each Y=1)
    white_xyz = xy_to_xyz(primaries[3][0], primaries[3][1])
    # Solve primary_xyz @ s = white_xyz  →  s = solve(primary_xyz, white_xyz).
    s = np.linalg.solve(primary_xyz, white_xyz)
    return primary_xyz * s  # broadcast scales each column


def bradford_cat(src_white: XY, dst_white: XY) -> np.ndarray:
    """Bradford chromatic-adaptation matrix (3×3) src_white → dst_white.

    Von Kries diagonal in the Bradford cone-response space:
    ``CAT = MA^-1 · diag(d/s) · MA`` where s, d are the source/destination whites
    in cone-response space.
    """
    src = _MA_BRADFORD @ xy_to_xyz(src_white[0], src_white[1])
    dst = _MA_BRADFORD @ xy_to_xyz(dst_white[0], dst_white[1])
    diag = np.diag(dst / src)
    return _MA_BRADFORD_INV @ diag @ _MA_BRADFORD


def rgb_to_rgb_matrix(src: Primaries, dst: Primaries) -> np.ndarray:
    """RGB(src) → RGB(dst) matrix (3×3, float64).

    ``M = inv(M_dst→XYZ) · CAT(src_w→dst_w) · M_src→XYZ``. For an RGB column
    vector, ``rgb_dst = M @ rgb_src``; apply to row-vector arrays as
    ``rgb @ M.T`` (§9.6).
    """
    m_src = rgb_to_xyz_matrix(src)
    m_dst = rgb_to_xyz_matrix(dst)
    cat = bradford_cat(src[3], dst[3])
    return np.linalg.inv(m_dst) @ cat @ m_src


# ---------------------------------------------------------------------------
# ColorTransform — input ↔ working-space (Rec.709 gamma) ↔ output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ColorTransform:
    """Functional color transform between a source/dest ColorSpace and the model.

    The model domain is **Rec.709 gamma-domain RGB** (the working space, §1).
    """

    src: ColorSpace
    dst: ColorSpace

    def to_model_domain(self, rgb: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """src RGB (gamma or linear) → Rec.709 gamma (model domain).

        Pipeline: src EOTF (linearize) → src-linear→Rec.709-linear matrix →
        bt709 OETF (to gamma). For src==REC709 this is identity; for
        src==LINEAR_REC709 this is just bt709_oetf. No hard clip (§9.8).
        """
        linear = _apply_eotf(rgb, self.src.transfer)
        if not _primaries_match(self.src.primaries, _REC709_PRIMARIES, atol=5e-4):
            m = rgb_to_rgb_matrix(self.src.primaries, _REC709_PRIMARIES)
            linear = _apply_matrix(linear, m)
        return _apply_oetf(linear, "bt709")

    def from_model_domain(self, rgb_rec709_gamma: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Rec.709 gamma (model domain) → dst RGB.

        Pipeline: bt709 EOTF (linearize) → Rec.709-linear→dst-linear matrix →
        dst OETF (or stop at linear for a linear dst). For dst==REC709 identity;
        for dst==LINEAR_REC709 this is just bt709_eotf. No hard clip (§9.8).
        """
        linear = bt709_eotf(rgb_rec709_gamma)
        if not _primaries_match(self.dst.primaries, _REC709_PRIMARIES, atol=5e-4):
            m = rgb_to_rgb_matrix(_REC709_PRIMARIES, self.dst.primaries)
            linear = _apply_matrix(linear, m)
        return _apply_oetf(linear, self.dst.transfer)

    def model_to_output(self, rgb_rec709_gamma: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Alias for :meth:`from_model_domain` (working→output)."""
        return self.from_model_domain(rgb_rec709_gamma)


def _apply_matrix(rgb: np.ndarray | torch.Tensor, matrix: np.ndarray) -> np.ndarray | torch.Tensor:
    """Apply a 3×3 matrix to row-vector RGB array(s) as ``rgb @ M.T`` (§9.6).

    Preserves the input backend; out-of-gamut values pass through (no clip, §9.8).
    """
    if isinstance(rgb, torch.Tensor):
        m = torch.as_tensor(matrix, dtype=rgb.dtype, device=rgb.device)
        return rgb @ m.T
    return np.asarray(rgb, dtype=np.float64) @ matrix.T


def color_to_model_space(
    tensor: np.ndarray | torch.Tensor, src: ColorSpace
) -> np.ndarray | torch.Tensor:
    """Convenience wrapper: ``ColorTransform(src, REC709).to_model_domain(tensor)``."""
    return ColorTransform(src=src, dst=REC709).to_model_domain(tensor)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _resolve_ffprobe() -> str:
    """Resolve the ffprobe binary: PATH, else /usr/bin/ffprobe."""
    return shutil.which("ffprobe") or "/usr/bin/ffprobe"


def detect_colorspace(path: str) -> ColorSpace:
    """Detect the colorspace of an image/video file (§3 detection rules).

    * ``.exr`` → read OpenEXR ``chromaticities``; match the registry at 5e-4
      (§9.5). No chromaticities → :data:`LINEAR_REC709` (untagged EXR = linear
      Rec.709/D65). Unknown/custom chromaticities → a ``custom`` ColorSpace that
      preserves the as-read primaries/white (transfer="linear" — EXR is linear by
      convention).
    * video (mov/mp4/etc.) → ffprobe VUI. ``smpte170m``/``bt470bg`` → BT.601
      (NOT bt709; primaries differ — §9.7). ``bt709`` → REC709. ``unspecified``
      → REC709 (assume). HDR transfers (smpte2084/PQ, arib-std-b67/HLG) raise.
    """
    suffix = Path(path).suffix.lower()
    if suffix == ".exr":
        return _detect_exr_colorspace(path)
    return _detect_video_colorspace(path)


def _chroma_tuple_from_exr(chroma: Any) -> Primaries | None:
    """Normalize an OpenEXR ``chromaticities`` read value to a Primaries tuple.

    The OpenEXR v3 binding returns an 8-tuple ``(rx,ry,gx,gy,bx,by,wx,wy)``.
    Returns None if absent/malformed. Param is ``Any`` because OpenEXR ships no
    type stubs (``header.get(...)`` yields ``Any``); conversion is guarded by
    try/except so non-tuple/non-numeric values resolve to None without relying on
    isinstance narrowing (which would yield Unknown tuple elements).
    """
    try:
        if len(chroma) != 8:
            return None
        return (
            (float(chroma[0]), float(chroma[1])),
            (float(chroma[2]), float(chroma[3])),
            (float(chroma[4]), float(chroma[5])),
            (float(chroma[6]), float(chroma[7])),
        )
    except (TypeError, ValueError):
        return None


def _detect_exr_colorspace(path: str) -> ColorSpace:
    import OpenEXR

    openexr: Any = OpenEXR
    f = openexr.File(path, header_only=True)
    header = f.header()
    primaries = _chroma_tuple_from_exr(header.get("chromaticities"))
    if primaries is None:
        # Untagged EXR → linear Rec.709/D65 (§9.5).
        return LINEAR_REC709

    named = _match_named_primaries(primaries, atol=5e-4)
    if named is not None:
        # EXR is linear by convention — return the linear representative whose
        # primaries matched (REC709 primaries → LINEAR_REC709, ACES AP0 →
        # ACES_AP0, etc.).
        return named
    # Custom/unknown chromaticities — preserve them (§9.5), linear transfer.
    return ColorSpace(
        name="custom",
        primaries=primaries,
        white=primaries[3],
        transfer="linear",
    )


# ffprobe VUI → ColorSpace. Keys: color_primaries, color_transfer, color_space.
def _ffprobe_video_vui(path: str) -> dict[str, str | None]:
    """Probe color_primaries/color_transfer/color_space via ffprobe (strings)."""
    binary = _resolve_ffprobe()
    result = subprocess.run(  # noqa: S603 — ffprobe with controlled args
        [binary, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=color_primaries,color_transfer,color_space",
         "-of", "default=noprint_wrappers=1:nokey=0", path],
        capture_output=True,
        text=True,
        check=False,
    )
    vals: dict[str, str | None] = {"color_primaries": None, "color_transfer": None, "color_space": None}
    for line in result.stdout.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key in vals:
            vals[key] = val.strip() or None
    return vals


_HDR_TRANSFERS = {"smpte2084", "arib-std-b67"}


def _detect_video_colorspace(path: str) -> ColorSpace:
    vui = _ffprobe_video_vui(path)
    transfer = vui["color_transfer"]
    if transfer in _HDR_TRANSFERS:
        raise NotImplementedError(
            f"HDR video transfer {transfer!r} is gated — HDR IC-LoRA is separate; "
            "refusing to silently mishandle HDR."
        )

    primaries = vui["color_primaries"]
    # Map ffprobe primaries tag → ColorSpace.
    if primaries in ("smpte170m", "bt470bg", "smpte240m"):
        # BT.601 (525 = smpte170m, 625 = bt470bg). Primaries DIFFER from BT.709 (§9.7).
        return BT601_525 if primaries == "smpte170m" else BT601_625
    if primaries in ("bt2020nc", "bt2020"):
        return REC2020
    if primaries == "smpte432":
        return DISPLAY_P3_D65
    # bt709, unspecified, or unknown → REC709 (assume Rec.709 per §3).
    return REC709


# ---------------------------------------------------------------------------
# Re-exports consumed by media_encoder (single source of truth)
# ---------------------------------------------------------------------------

# OpenEXR ``chromaticities`` 8-tuple for BT.709/D65 (verified form — see
# media_encoder/color.py). Built from the authoritative REC709 primaries.
BT709_CHROMATICITIES: tuple[float, float, float, float, float, float, float, float] = (
    _REC709_PRIMARIES[0][0], _REC709_PRIMARIES[0][1],
    _REC709_PRIMARIES[1][0], _REC709_PRIMARIES[1][1],
    _REC709_PRIMARIES[2][0], _REC709_PRIMARIES[2][1],
    _REC709_PRIMARIES[3][0], _REC709_PRIMARIES[3][1],
)

# adoptedNeutral (D65) as a float32 numpy array (OpenEXR header value form).
ADOPTED_NEUTRAL_D65: np.ndarray = np.array(
    [_REC709_PRIMARIES[3][0], _REC709_PRIMARIES[3][1]], dtype=np.float32
)


# ---------------------------------------------------------------------------
# EXR-format converters (ColorSpace → OpenEXR header values)
# ---------------------------------------------------------------------------

def primaries_to_exr_chromaticities(primaries: Primaries) -> tuple[float, ...]:
    """Convert 4 xy pairs ``(r,g,b,w)`` to the OpenEXR 8-tuple
    ``(rx,ry,gx,gy,bx,by,wx,wy)`` used by the ``chromaticities`` header attr."""
    return tuple(float(v) for p in primaries for v in p)


def white_to_adopted_neutral(white: XY) -> np.ndarray:
    """Convert an xy white point to the OpenEXR ``adoptedNeutral`` float32 array."""
    return np.array([float(white[0]), float(white[1])], dtype=np.float32)


__all__ = [
    "ACES_AP0",
    "ACES_CG",
    "ADOPTED_NEUTRAL_D65",
    "BT601_525",
    "BT601_625",
    "BT709_CHROMATICITIES",
    "COLOR_SPACES",
    "ColorSpace",
    "ColorTransform",
    "DISPLAY_P3_D65",
    "LINEAR_REC709",
    "LINEAR_REC2020",
    "Primaries",
    "REC2020",
    "REC709",
    "SRGB",
    "TransferKey",
    "XY",
    "bradford_cat",
    "bt709_eotf",
    "bt709_oetf",
    "color_to_model_space",
    "detect_colorspace",
    "gamma24_eotf",
    "gamma24_oetf",
    "hlg_eotf",
    "hlg_oetf",
    "pq_eotf",
    "pq_oetf",
    "primaries_to_exr_chromaticities",
    "rgb_to_rgb_matrix",
    "rgb_to_xyz_matrix",
    "srgb_eotf",
    "srgb_oetf",
    "white_to_adopted_neutral",
    "xy_to_xyz",
]
