"""Pure-math tests for the color-management module (CM-1a).

Portable (numpy/torch only — NO /mnt fixture paths; the fixture check lives in
the untracked ``validate_color_math.py``). Verifies §9 authoritative math:
BT.709 transfers (incl. the V=0.5→0.2596 mid-gray value), SMPTE RP 177 +
Bradford matrix round-trips, the ``rgb @ M.T`` row-vector orientation, the
ColorTransform identity/linear paths, EXR + video detection, and the HDR gates.

Repo rules: no mocking libraries, pyright strict for the module under test
(tests dir is excluded from pyright).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from services.color_management import (
    ACES_AP0,
    BT601_525,
    COLOR_SPACES,
    DISPLAY_P3_D65,
    LINEAR_REC709,
    REC2020,
    REC709,
    ColorSpace,
    ColorTransform,
    bt709_eotf,
    bt709_oetf,
    color_to_model_space,
    detect_colorspace,
    gamma24_eotf,
    gamma24_oetf,
    hlg_eotf,
    hlg_oetf,
    pq_eotf,
    pq_oetf,
    rgb_to_rgb_matrix,
    rgb_to_xyz_matrix,
    srgb_eotf,
    srgb_oetf,
)


# ---------------------------------------------------------------------------
# Transfers
# ---------------------------------------------------------------------------

# Round-trip grid clear of the BT.709 OETF junction band. The piecewise OETF has
# an inherent ~0.00026 gap in V at the L=0.018 junction (linear branch 4.5·0.018
# = 0.081 vs power branch ≈ 0.08126), so the inverse round-trip carries a small
# residual across the neighborhood [~0.081, ~0.0813] (and symmetrically in L) —
# this is documented ITU-R BT.709 behavior, not a bug. We prove exact invertibility
# on a grid clear of that band, and characterize the junction separately below.
_GRID_SAFE = [0.0, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
_JUNCTION_GAMMA = [0.080, 0.081, 0.0811, 0.082]


def test_bt709_transfer_identity() -> None:
    """bt709_eotf(bt709_oetf(x)) == x and inverse — EXACT away from the junction band.

    float64 atol 1e-9 (proves exact mathematical invertibility), float32 atol
    1e-6, on a grid clear of [~0.081, ~0.0813]. Both numpy and torch backends.
    """
    for dtype, tol in ((np.float64, 1e-9), (np.float32, 1e-6)):
        x = np.array(_GRID_SAFE, dtype=dtype)
        np.testing.assert_allclose(bt709_oetf(bt709_eotf(x)), x, atol=tol,
                                   err_msg=f"oetf(eotf) numpy round-trip ({dtype})")
        lin = bt709_eotf(x)
        np.testing.assert_allclose(bt709_eotf(bt709_oetf(lin)), lin, atol=tol,
                                   err_msg=f"eotf(oetf) numpy round-trip ({dtype})")
        # torch backend
        xt = torch.tensor(_GRID_SAFE, dtype=torch.float32)
        np.testing.assert_allclose(
            bt709_oetf(bt709_eotf(xt)).cpu().numpy(),
            np.array(_GRID_SAFE, dtype=np.float32), atol=1e-6,
            err_msg="oetf(eotf) torch round-trip",
        )


def test_bt709_junction_gap_characterization() -> None:
    """Document the inherent BT.709 OETF junction gap (~3e-4 at the L=0.018 junction).

    The piecewise OETF's linear and power branches do not meet perfectly at
    L=0.018 (4.5·0.018=0.081 vs 1.099·0.018^0.45−0.099≈0.08126), so the inverse
    round-trip carries a residual < ~3e-4 across the junction neighborhood in V.
    This is a documented property of ITU-R BT.709 (not a bug); the bound below
    pins it so any drift in the transfer math surfaces here.
    """
    for v0 in _JUNCTION_GAMMA:
        v = np.array([v0], dtype=np.float64)
        residual = abs(float(bt709_oetf(bt709_eotf(v))[0]) - v0)
        assert residual < 5e-4, (
            f"BT.709 junction round-trip residual {residual} at V={v0} exceeds "
            f"the documented junction gap"
        )


def test_bt709_known_values() -> None:
    """§9.1: bt709_eotf(0.5)≈0.2596, bt709_oetf(0.2596)≈0.5.

    NOTE: the task body claimed bt709_oetf(0.5)=0.409 — that value is ERRONEOUS
    (it contradicts §9.1 and the ITU-R BT.709-6 formula, which give ~0.7055).
    §9 is authoritative, so we assert the correct values here.
    """
    assert bt709_eotf(np.array([0.5], dtype=np.float32))[0] == pytest.approx(0.2596, abs=1e-3)
    assert bt709_oetf(np.array([0.2596], dtype=np.float32))[0] == pytest.approx(0.5, abs=1e-3)
    # Authoritative forward value (documents the task-body error for the record).
    assert bt709_oetf(np.array([0.5], dtype=np.float32))[0] == pytest.approx(0.7055, abs=1e-3)


def test_srgb_and_gamma24_identity() -> None:
    """sRGB and gamma-2.4 transfer pairs round-trip (pure-power — no junction)."""
    x = np.array(_GRID_SAFE, dtype=np.float64)
    np.testing.assert_allclose(srgb_oetf(srgb_eotf(x)), x, atol=1e-6)
    np.testing.assert_allclose(gamma24_oetf(gamma24_eotf(x)), x, atol=1e-6)


# ---------------------------------------------------------------------------
# Matrices (SMPTE RP 177 + Bradford CAT)
# ---------------------------------------------------------------------------

_SPACE_PAIRS = [
    (REC709.primaries, DISPLAY_P3_D65.primaries),
    (REC709.primaries, REC2020.primaries),
    (REC709.primaries, ACES_AP0.primaries),
    (DISPLAY_P3_D65.primaries, REC2020.primaries),
]


@pytest.mark.parametrize("src,dst", _SPACE_PAIRS)
def test_matrix_identity(src: Any, dst: Any) -> None:
    """rgb_to_rgb_matrix(A,B) @ rgb_to_rgb_matrix(B,A) ≈ I (float64)."""
    ab = rgb_to_rgb_matrix(src, dst)
    ba = rgb_to_rgb_matrix(dst, src)
    product = ab @ ba
    np.testing.assert_allclose(product, np.eye(3), atol=1e-12,
                               err_msg="matrix round-trip not identity")
    assert np.max(np.abs(product - np.eye(3))) < 1e-12


def test_rec709_xyz_matrix_canonical_luma() -> None:
    """Rec.709 RGB→XYZ Y row = canonical luma coefficients (0.2126/0.7152/0.0722)."""
    m = rgb_to_xyz_matrix(REC709.primaries)
    np.testing.assert_allclose(m[1], [0.2126, 0.7152, 0.0722], atol=1e-4)


def test_matrix_orientation() -> None:
    """§9.6: apply as ``rgb @ M.T`` (row-vector) — equals ``M @ rgb_col``, NOT ``rgb @ M``."""
    m = rgb_to_rgb_matrix(REC709.primaries, DISPLAY_P3_D65.primaries)
    # A non-symmetric cross-primaries matrix.
    assert not np.allclose(m, m.T), "test needs a non-symmetric matrix"

    rgb = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    via_row = rgb @ m.T            # correct row-vector application
    via_col = m @ rgb              # equivalent column-vector application
    wrong = rgb @ m                # the WRONG orientation (M.T @ rgb_col)

    np.testing.assert_allclose(via_row, via_col, atol=1e-12,
                               err_msg="rgb @ M.T must equal M @ rgb_col")
    assert not np.allclose(via_row, wrong), (
        "rgb @ M.T must differ from rgb @ M — orientation matters"
    )


# ---------------------------------------------------------------------------
# ColorTransform (input ↔ working ↔ output)
# ---------------------------------------------------------------------------

def test_to_model_domain_identity_for_rec709() -> None:
    """REC709 (gamma) input → identity (model domain is already Rec.709 gamma)."""
    t = ColorTransform(src=REC709, dst=REC709)
    x = np.array([0.1, 0.4, 0.7], dtype=np.float64)
    out = t.to_model_domain(x)
    np.testing.assert_allclose(out, x, atol=1e-12)


def test_to_model_domain_linear_to_rec709() -> None:
    """LINEAR_REC709 input → bt709_oetf applied (0.5 linear → 0.7055 Rec.709 gamma).

    NOTE: the task body claimed 0.5 linear → 0.409; that is ERRONEOUS — the
    correct BT.709 OETF value is ~0.7055 (§9.1 + ITU-R BT.709-6). Asserted here.
    """
    t = ColorTransform(src=LINEAR_REC709, dst=REC709)
    x = np.array([0.5], dtype=np.float64)
    out = t.to_model_domain(x)
    expected = bt709_oetf(np.array([0.5], dtype=np.float64))
    np.testing.assert_allclose(out, expected, atol=1e-12)
    assert float(out[0]) == pytest.approx(0.7055, abs=1e-3)


def test_from_model_domain_rec709_to_linear() -> None:
    """REC709 gamma 0.7055 → LINEAR 0.5 (round-trip of the above)."""
    t = ColorTransform(src=REC709, dst=LINEAR_REC709)
    gamma = np.array([0.7055], dtype=np.float64)
    out = t.from_model_domain(gamma)
    assert float(out[0]) == pytest.approx(0.5, abs=1e-3)


def test_color_to_model_space_wrapper() -> None:
    """The convenience wrapper matches ColorTransform(src, REC709).to_model_domain."""
    x = np.array([0.3, 0.6, 0.9], dtype=np.float64)
    via_wrapper = color_to_model_space(x, LINEAR_REC709)
    via_transform = ColorTransform(src=LINEAR_REC709, dst=REC709).to_model_domain(x)
    np.testing.assert_allclose(via_wrapper, via_transform, atol=1e-12)


def test_gamut_policy_no_hard_clip() -> None:
    """§9.8: out-of-gamut values pass through the working space unclipped (floats preserved)."""
    t = ColorTransform(src=LINEAR_REC709, dst=REC709)
    x = np.array([1.5, -0.2, 0.5], dtype=np.float64)  # intentionally out-of-range
    out = t.to_model_domain(x)
    # No clipping to [0,1]: 1.5 maps above 1.0 (and -0.2 stays negative in linear).
    assert float(out[0]) > 1.0, "out-of-gamut value must not be clipped (gamut policy)"


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _write_exr(path: str, chromaticities: Any) -> None:
    """Write a tiny synthetic EXR with the given chromaticities (8-tuple or None)."""
    import OpenEXR

    ch = {
        "R": np.full((8, 8), 0.5, dtype="float16"),
        "G": np.full((8, 8), 0.5, dtype="float16"),
        "B": np.full((8, 8), 0.5, dtype="float16"),
    }
    hdr: dict[str, Any] = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    if chromaticities is not None:
        hdr["chromaticities"] = chromaticities
    OpenEXR.File(hdr, ch).write(path)


def _chroma_of(cs: ColorSpace) -> tuple[float, ...]:
    return tuple(v for p in cs.primaries for v in p)


def test_detect_exr_colorspace_tagged(tmp_path: Path) -> None:
    """EXR detection (§9.5, 5e-4 tolerance).

    EXR is ALWAYS linear-light (§9.3), so a Rec.709-primaries EXR detects as
    LINEAR_REC709 (NOT REC709 — the task body's "→ REC709" was imprecise for
    EXR; REC709 implies a bt709 transfer which is wrong for linear EXR).
    """
    # Rec.709 chromaticities → LINEAR_REC709 (linear + Rec.709 primaries).
    p = tmp_path / "r709.exr"
    _write_exr(str(p), _chroma_of(REC709))
    assert detect_colorspace(str(p)) == LINEAR_REC709

    # ACES AP0 chromaticities → ACES_AP0.
    p = tmp_path / "ap0.exr"
    _write_exr(str(p), _chroma_of(ACES_AP0))
    assert detect_colorspace(str(p)) == ACES_AP0

    # No chromaticities → LINEAR_REC709 (untagged EXR default, §9.5).
    p = tmp_path / "none.exr"
    _write_exr(str(p), None)
    assert detect_colorspace(str(p)) == LINEAR_REC709

    # Custom/unknown chromaticities → preserved (transfer="linear").
    p = tmp_path / "custom.exr"
    _write_exr(str(p), (0.5, 0.4, 0.3, 0.5, 0.2, 0.1, 0.31, 0.33))
    cs = detect_colorspace(str(p))
    assert cs.name == "custom"
    assert cs.transfer == "linear"
    # Primaries preserved as-read.
    np.testing.assert_allclose(
        [cs.primaries[0][0], cs.primaries[3][1]], [0.5, 0.33], atol=1e-6
    )


def _write_tagged_mp4(path: str, primaries: str, transfer: str, matrix: str) -> None:
    """Generate a tiny mp4 with explicit VUI tags via the bundled ffmpeg."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ff, "-y", "-f", "lavfi", "-i", "color=c=gray:s=16x16:d=0.5",
         "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-color_primaries", primaries, "-color_trc", transfer, "-colorspace", matrix,
         path],
        capture_output=True, check=True,
    )


def test_detect_video_colorspace(tmp_path: Path) -> None:
    """Video detection: bt709→REC709, smpte170m→BT.601 (§9.7, NOT bt709)."""
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not available — video detection probe needs ffprobe")

    bt709_mp4 = tmp_path / "bt709.mp4"
    _write_tagged_mp4(str(bt709_mp4), "bt709", "bt709", "bt709")
    assert detect_colorspace(str(bt709_mp4)) == REC709

    # smpte170m = BT.601 (primaries DIFFER from BT.709 — §9.7).
    bt601_mp4 = tmp_path / "bt601.mp4"
    _write_tagged_mp4(str(bt601_mp4), "smpte170m", "smpte170m", "smpte170m")
    detected = detect_colorspace(str(bt601_mp4))
    assert detected == BT601_525, (
        f"smpte170m must map to BT.601, not {detected.name} (§9.7)"
    )
    assert detected.primaries != REC709.primaries


def test_detect_video_unspecified_assumes_rec709(tmp_path: Path) -> None:
    """Unspecified/untagged video → REC709 (assume Rec.709 per §3)."""
    if not shutil.which("ffprobe"):
        pytest.skip("ffprobe not available")
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    mp4 = tmp_path / "untagged.mp4"
    subprocess.run(
        [ff, "-y", "-f", "lavfi", "-i", "color=c=gray:s=16x16:d=0.5",
         "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(mp4)],
        capture_output=True, check=True,
    )
    assert detect_colorspace(str(mp4)) == REC709


# ---------------------------------------------------------------------------
# HDR gates
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fn", [pq_oetf, pq_eotf, hlg_oetf, hlg_eotf])
def test_hdr_transfers_raise(fn: Any) -> None:
    """PQ/HLG are gated — must raise NotImplementedError (no silent HDR mishandling)."""
    x = np.array([0.5], dtype=np.float32)
    with pytest.raises(NotImplementedError, match="HDR"):
        fn(x)


# ---------------------------------------------------------------------------
# Registry sanity + refactor import-path guard
# ---------------------------------------------------------------------------

def test_color_space_registry_names() -> None:
    """Registry contains the canonical named spaces."""
    for name in ("lin_rec709", "rec709", "srgb", "display_p3_d65",
                 "rec2020", "lin_rec2020", "aces_ap0", "aces_cg",
                 "bt601_525", "bt601_625"):
        assert name in COLOR_SPACES, f"missing named ColorSpace {name!r}"


def test_output_path_byte_identical_regression() -> None:
    """The refactor must point media_encoder at the SAME transfer functions.

    bt709_eotf imported via media_encoder.color must be the exact function object
    defined in color_management (single source of truth) — guarantees the
    encoder's output math is unchanged by the refactor.
    """
    from services.media_encoder.color import bt709_eotf as me_eotf
    from services.media_encoder.color import bt709_oetf as me_oetf

    assert me_eotf is bt709_eotf, "media_encoder must re-export color_management.bt709_eotf"
    assert me_oetf is bt709_oetf, "media_encoder must re-export color_management.bt709_oetf"
