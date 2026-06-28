"""EXR single-file input + input→Rec.709 tests (CM-1b rework).

The dir-based EXR path (CM-1b + P0-3) is GONE — ``video_path`` is now a SINGLE
FILE from a sequence, resolved by :mod:`services.sequence_input` (see
``test_sequence_input.py``). This file keeps the EXR image-conditioning color
tests + the byte-identity gates for the reworked production branches.

SAFETY-FIRST: ``test_non_exr_inputs_are_byte_identical`` +
``test_bt709_video_input_byte_identical`` + the two ``test_production_inpaint_*``
gates prove the reworked branches yield bit-exact conditioning tensors for the
validated MP4 inpaint path. No /mnt paths in committed tests.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from services.color_management import bt709_oetf
from services.exr_input import (
    decode_exr_image,
    iter_video_frames_to_model_domain,
    resolve_image_input_path,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _write_srgb_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path, format="PNG")


def _write_mp4(path: Path, frames: np.ndarray, fps: int = 24) -> None:
    """Write a tiny MP4 from a ``(F, H, W, 3)`` uint8 array via ffmpeg."""
    ff = _ffmpeg()
    tmp_dir = path.parent / (path.stem + "_frames")
    tmp_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        _write_srgb_png(tmp_dir / f"frame_{i:05d}.png", frame)
    subprocess.run(
        [ff, "-y", "-framerate", str(fps), "-i", str(tmp_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


def _write_exr(path: Path, linear_rgb: np.ndarray, chromaticities: object = None) -> None:
    """Write an EXR (half) with the given linear RGB ``(H, W, 3)`` float."""
    import OpenEXR

    h, w, _ = linear_rgb.shape
    hdr: dict[str, object] = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    if chromaticities is not None:
        hdr["chromaticities"] = chromaticities
    channels = {
        "R": np.ascontiguousarray(linear_rgb[:, :, 0].astype(np.float16)),
        "G": np.ascontiguousarray(linear_rgb[:, :, 1].astype(np.float16)),
        "B": np.ascontiguousarray(linear_rgb[:, :, 2].astype(np.float16)),
    }
    OpenEXR.File(hdr, channels).write(str(path))


# ---------------------------------------------------------------------------
# THE inpaint-protection gates (mandatory, must not be weakened)
# ---------------------------------------------------------------------------

def test_non_exr_inputs_are_byte_identical(tmp_path: Path) -> None:
    """For NON-sequence video inputs the conditioning frames must be BIT-EXACT vs
    the legacy ``decode_video_by_frame``.

    The reworked production branch is a single ``iter_video_frames_to_model_domain``
    call; for a bt709/untagged MP4 it must passthrough byte-identical to the
    legacy decoder. This is the byte-identity guarantee for the validated inpaint.
    """
    from ltx_pipelines.utils.media_io import decode_video_by_frame

    device = torch.device("cpu")
    frames = np.stack([
        np.full((16, 16, 3), 50, dtype=np.uint8),
        np.full((16, 16, 3), 200, dtype=np.uint8),
    ])
    mp4 = tmp_path / "src.mp4"
    _write_mp4(mp4, frames)

    legacy = [t.cpu() for t in decode_video_by_frame(path=str(mp4), frame_cap=2, device=device)]
    branched = [
        t.cpu() for t in iter_video_frames_to_model_domain(str(mp4), frame_cap=2, device=device)
    ]

    assert len(legacy) == len(branched) == 2
    for a, b in zip(legacy, branched):
        assert torch.equal(a, b), "non-sequence video branch must be byte-identical to legacy"

    # Image-conditioning path: PNG resolves to ITSELF (identity), not a temp file.
    png = tmp_path / "src.png"
    _write_srgb_png(png, np.full((16, 16, 3), 128, dtype=np.uint8))
    assert resolve_image_input_path(str(png)) == str(png), (
        "non-EXR image input path must be returned UNCHANGED (identity)"
    )


# ---------------------------------------------------------------------------
# EXR image input → Rec.709 (model domain)
# ---------------------------------------------------------------------------

def test_exr_image_input_transferred_to_rec709(tmp_path: Path) -> None:
    """A linear-light EXR (value 0.5) → model domain uint8 ≈ bt709_oetf(0.5)=0.7055.

    The transfer: linear 0.5 → BT.709 OETF → 0.7055 → ×255 → 180 (uint8). This is
    the value the model receives (same Rec.709 gamma domain as MP4/sRGB inputs).
    """
    linear_val = 0.5
    exr = tmp_path / "lin.exr"
    _write_exr(exr, np.full((16, 16, 3), linear_val, dtype=np.float32))

    resolved = resolve_image_input_path(str(exr))
    assert resolved != str(exr), "EXR must resolve to a temp PNG"
    assert Path(resolved).suffix == ".png"

    from PIL import Image

    arr = np.asarray(Image.open(resolved).convert("RGB"), dtype=np.uint8)
    expected = int(round(float(bt709_oetf(np.array([linear_val], dtype=np.float32))[0]) * 255.0))
    assert arr[0, 0, 0] == expected, f"EXR→Rec.709: expected {expected}, got {int(arr[0,0,0])}"
    assert expected == 180, f"bt709_oetf(0.5)*255 should be 180, got {expected}"


def test_decode_exr_image_linear_value(tmp_path: Path) -> None:
    """decode_exr_image returns the raw linear value (no gamma applied here)."""
    exr = tmp_path / "raw.exr"
    _write_exr(exr, np.full((8, 8, 3), 0.3, dtype=np.float32))
    rgb = decode_exr_image(str(exr))
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.float32
    np.testing.assert_allclose(rgb[0, 0, :], 0.3, atol=2e-3)  # half rounding


# ---------------------------------------------------------------------------
# media_validation EXR allow-list
# ---------------------------------------------------------------------------

def test_media_validation_accepts_exr(tmp_path: Path) -> None:
    """An `.exr` passes validate_image_file; a non-exr is unchanged."""
    from server_utils.media_validation import validate_image_file

    exr = tmp_path / "ok.exr"
    _write_exr(exr, np.full((8, 8, 3), 0.5, dtype=np.float32))
    assert validate_image_file(str(exr)) == exr

    png = tmp_path / "ok.png"
    _write_srgb_png(png, np.full((8, 8, 3), 128, dtype=np.uint8))
    assert validate_image_file(str(png)) == png

    bad = tmp_path / "bad.exr"
    bad.write_bytes(b"not an exr")
    from _routes._errors import HTTPError

    with pytest.raises(HTTPError):
        validate_image_file(str(bad))


# ---------------------------------------------------------------------------
# Missing channels fail loudly
# ---------------------------------------------------------------------------

def test_exr_missing_channels_raise(tmp_path: Path) -> None:
    """An EXR missing R/G/B must raise (fail loudly, not default to a black plane)."""
    import OpenEXR

    arr = np.full((8, 8), 0.5, dtype=np.float16)
    channels = {"R": arr, "G": arr}  # no B
    hdr = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    exr = tmp_path / "rg_only.exr"
    OpenEXR.File(hdr, channels).write(str(exr))

    with pytest.raises(ValueError, match="missing required channel"):
        decode_exr_image(str(exr))


# ---------------------------------------------------------------------------
# CM-1c: tagged non-bt709 VIDEO input → Rec.709 correction
# ---------------------------------------------------------------------------

def _write_tagged_mp4(path: str, primaries: str, transfer: str, matrix: str, color: str = "0x8040C0") -> None:
    """Generate a tiny tagged mp4 via ffmpeg lavfi. Default color is non-gray
    (purple) so matrix corrections fire for tagged non-bt709 tests."""
    import imageio_ffmpeg

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run(
        [ff, "-y", "-f", "lavfi", "-i", f"color=c={color}:s=16x16:d=0.5",
         "-frames:v", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-color_primaries", primaries, "-color_trc", transfer, "-colorspace", matrix,
         path],
        capture_output=True, check=True,
    )


def test_bt709_video_input_byte_identical(tmp_path: Path) -> None:
    """THE inpaint-protection gate: bt709/untagged video → byte-identical passthrough."""
    from ltx_pipelines.utils.media_io import decode_video_by_frame

    device = torch.device("cpu")
    frames = np.stack([
        np.full((16, 16, 3), 50, dtype=np.uint8),
        np.full((16, 16, 3), 200, dtype=np.uint8),
    ])

    untagged = tmp_path / "untagged.mp4"
    _write_mp4(untagged, frames)
    legacy = [t.cpu() for t in decode_video_by_frame(path=str(untagged), frame_cap=2, device=device)]
    wrapped = [t.cpu() for t in iter_video_frames_to_model_domain(str(untagged), frame_cap=2, device=device)]
    assert len(legacy) == len(wrapped) == 2
    for a, b in zip(legacy, wrapped):
        assert torch.equal(a, b), "untagged video must be byte-identical (CM-1c passthrough)"

    bt709_mp4 = str(tmp_path / "bt709.mp4")
    _write_tagged_mp4(bt709_mp4, "bt709", "bt709", "bt709")
    legacy2 = [t.cpu() for t in decode_video_by_frame(path=bt709_mp4, frame_cap=3, device=device)]
    wrapped2 = [t.cpu() for t in iter_video_frames_to_model_domain(bt709_mp4, frame_cap=3, device=device)]
    assert len(legacy2) == len(wrapped2)
    for a, b in zip(legacy2, wrapped2):
        assert torch.equal(a, b), "bt709-tagged video must be byte-identical (CM-1c passthrough)"


def test_smpte170m_video_input_corrected(tmp_path: Path) -> None:
    """smpte170m (BT.601) video → correction applied, matches color_to_model_space."""
    from services.color_management import BT601_525, color_to_model_space, detect_colorspace

    mp4 = str(tmp_path / "smpte170m.mp4")
    _write_tagged_mp4(mp4, "smpte170m", "smpte170m", "smpte170m")

    cs = detect_colorspace(mp4)
    assert cs == BT601_525, f"expected BT601_525, got {cs.name}"

    device = torch.device("cpu")
    wrapped = list(iter_video_frames_to_model_domain(mp4, frame_cap=3, device=device))

    from ltx_pipelines.utils.media_io import decode_video_by_frame

    raw = [t.cpu() for t in decode_video_by_frame(path=mp4, frame_cap=3, device=device)]

    assert len(wrapped) == len(raw) > 0
    assert not torch.equal(wrapped[0], raw[0]), "smpte170m frames must be corrected (not passthrough)"

    framef = raw[0].float() / 255.0
    expected = color_to_model_space(framef, BT601_525)
    if isinstance(expected, torch.Tensor):
        expected_uint8 = (expected.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
    else:
        expected_uint8 = torch.from_numpy(
            (np.clip(expected, 0, 1) * 255).round().astype(np.uint8)
        )
    assert torch.equal(wrapped[0], expected_uint8), (
        "corrected frame must match color_to_model_space(raw, BT601_525)"
    )


# ---------------------------------------------------------------------------
# Production-path identity: full video_preprocess chain, not just raw frames
# ---------------------------------------------------------------------------

def test_production_inpaint_video_branch_identity(tmp_path: Path) -> None:
    """The production inpaint video branch (now a single iter_video_frames_to_model_domain
    call) on a NON-sequence input must yield a byte-identical *conditioning tensor*
    (after the full video_preprocess chain), vs the legacy decode."""
    from ltx_pipelines.utils.media_io import decode_video_by_frame, video_preprocess

    device = torch.device("cpu")
    dtype = torch.float32
    frames = np.stack([
        np.full((16, 16, 3), 30, dtype=np.uint8),
        np.full((16, 16, 3), 120, dtype=np.uint8),
        np.full((16, 16, 3), 220, dtype=np.uint8),
    ])
    mp4 = tmp_path / "inpaint_src.mp4"
    _write_mp4(mp4, frames)

    legacy = video_preprocess(
        decode_video_by_frame(path=str(mp4), frame_cap=3, device=device),
        16, 16, dtype, device,
    )

    # The EXACT production branch expression (generate_inpaint, reworked):
    # a single iter_video_frames_to_model_domain call (no EXR ternary).
    branched = video_preprocess(
        iter_video_frames_to_model_domain(str(mp4), frame_cap=3, device=device),
        16, 16, dtype, device,
    )

    assert torch.equal(legacy, branched), (
        "production inpaint branch must yield byte-identical conditioning tensor"
    )
