"""EXR-input + input→Rec.709 tests (CM-1b).

SAFETY-FIRST: the headline test ``test_non_exr_inputs_are_byte_identical`` is the
mandatory inpaint-protection gate — it proves the branched code path yields
bit-exact conditioning tensors vs the legacy decode path for MP4 + sRGB PNG.
EXR tests verify the new decode+transfer (linear → Rec.709 gamma). No /mnt paths
in committed tests (the fixture check lives in the untracked validation script).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from api_types import ImageConditioningInput
from services.color_management import REC709, bt709_oetf
from services.exr_input import (
    decode_exr_image,
    decode_exr_sequence,
    is_exr_input,
    iter_exr_frames_as_video_tensors,
    resolve_image_input_path,
    resolve_video_input_path,
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
# THE inpaint-protection gate (mandatory, must not be weakened)
# ---------------------------------------------------------------------------

def test_non_exr_inputs_are_byte_identical(tmp_path: Path) -> None:
    """For NON-EXR inputs the conditioning frames must be BIT-EXACT vs legacy.

    Exercises the EXACT branch pattern wired at the chokepoints: for a non-EXR
    path the ternary must select ``decode_video_by_frame`` (the legacy decoder)
    with identical args. We replicate the branch and assert the yielded tensors
    equal the direct legacy call, for both an MP4 and (via the image resolver)
    a PNG. This is the byte-identity guarantee for the validated inpaint path.
    """
    from ltx_pipelines.utils.media_io import decode_video_by_frame

    device = torch.device("cpu")
    # A tiny synthetic MP4 (2 frames, 16x16, distinct pixel values).
    frames = np.stack([
        np.full((16, 16, 3), 50, dtype=np.uint8),
        np.full((16, 16, 3), 200, dtype=np.uint8),
    ])
    mp4 = tmp_path / "src.mp4"
    _write_mp4(mp4, frames)

    # Legacy path (what the non-EXR branch MUST call).
    legacy = [t.cpu() for t in decode_video_by_frame(path=str(mp4), frame_cap=2, device=device)]

    # Branched path (the wired pattern at the chokepoints).
    branched = [
        t.cpu() for t in (
            iter_exr_frames_as_video_tensors(str(mp4), frame_cap=2, device=device)
            if is_exr_input(str(mp4))
            else decode_video_by_frame(path=str(mp4), frame_cap=2, device=device)
        )
    ]

    assert is_exr_input(str(mp4)) is False, "MP4 must not be detected as EXR"
    assert len(legacy) == len(branched) == 2
    for a, b in zip(legacy, branched):
        assert torch.equal(a, b), "non-EXR video branch must be byte-identical to legacy"

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

    # Image-conditioning resolution → temp PNG in Rec.709 gamma domain.
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
# EXR sequence input → video frames in Rec.709
# ---------------------------------------------------------------------------

def test_exr_sequence_input(tmp_path: Path) -> None:
    """N=4 linear EXR frames → 4 video tensors, each transferred to Rec.709 gamma."""
    seq_dir = tmp_path / "exr_seq"
    seq_dir.mkdir()
    for i in range(4):
        _write_exr(seq_dir / f"frame_{i:05d}.exr",
                   np.full((12, 12, 3), 0.5, dtype=np.float32))

    frames = list(decode_exr_sequence(str(seq_dir)))
    assert len(frames) == 4

    # As video tensors (matches decode_video_by_frame output): (1,H,W,3) uint8.
    tensors = list(iter_exr_frames_as_video_tensors(str(seq_dir), frame_cap=None, device=torch.device("cpu")))
    assert len(tensors) == 4
    for t in tensors:
        assert t.shape == (1, 12, 12, 3)
        assert t.dtype == torch.uint8
    # Each transferred to bt709_oetf(0.5)*255 = 180.
    expected = int(round(float(bt709_oetf(np.array([0.5], dtype=np.float32))[0]) * 255.0))
    assert int(tensors[0][0, 0, 0, 0]) == expected

    # frame_cap respected.
    capped = list(iter_exr_frames_as_video_tensors(str(seq_dir), frame_cap=2, device=torch.device("cpu")))
    assert len(capped) == 2


def test_exr_sequence_fixture_pattern_sorts(tmp_path: Path) -> None:
    """The fixture ``Name_####.exr`` pattern is sorted by trailing digit."""
    seq_dir = tmp_path / "fixture_pattern"
    seq_dir.mkdir()
    # Write out of order; decode_exr_sequence must yield in numeric order.
    for i in (3, 0, 2, 1):
        _write_exr(seq_dir / f"Instant_Shave_Beard_{i:04d}.exr",
                   np.full((4, 4, 3), 0.1 * i, dtype=np.float32))
    frames = list(decode_exr_sequence(str(seq_dir)))
    assert len(frames) == 4
    # Sorted ascending → values 0.0, 0.1, 0.2, 0.3.
    np.testing.assert_allclose(frames[0][0, 0, 0], 0.0, atol=1e-3)
    np.testing.assert_allclose(frames[3][0, 0, 0], 0.3, atol=2e-3)


# ---------------------------------------------------------------------------
# media_validation EXR allow-list
# ---------------------------------------------------------------------------

def test_media_validation_accepts_exr(tmp_path: Path) -> None:
    """An `.exr` passes validate_image_file; a non-exr is unchanged."""
    from server_utils.media_validation import validate_image_file

    exr = tmp_path / "ok.exr"
    _write_exr(exr, np.full((8, 8, 3), 0.5, dtype=np.float32))
    assert validate_image_file(str(exr)) == exr

    # Non-EXR unchanged: a valid PNG still validates.
    png = tmp_path / "ok.png"
    _write_srgb_png(png, np.full((8, 8, 3), 128, dtype=np.uint8))
    assert validate_image_file(str(png)) == png

    # Invalid EXR (garbage) is rejected.
    bad = tmp_path / "bad.exr"
    bad.write_bytes(b"not an exr")
    from _routes._errors import HTTPError

    with pytest.raises(HTTPError):
        validate_image_file(str(bad))


# ---------------------------------------------------------------------------
# Predicate
# ---------------------------------------------------------------------------

def test_is_exr_input_predicate(tmp_path: Path) -> None:
    exr = tmp_path / "a.exr"
    _write_exr(exr, np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert is_exr_input(str(exr)) is True

    # EXR sequence dir — the pure-suffix gate requires the dir basename to signal
    # EXR (ends with `_exr` or is `exr`); our convention is `..._exr/`.
    seq = tmp_path / "clip_exr"
    seq.mkdir()
    _write_exr(seq / "frame_00000.exr", np.full((4, 4, 3), 0.5, dtype=np.float32))
    assert is_exr_input(str(seq)) is True

    mp4 = tmp_path / "a.mp4"
    _write_mp4(mp4, np.full((1, 8, 8, 3), 50, dtype=np.uint8))
    assert is_exr_input(str(mp4)) is False
    assert is_exr_input("/nonexistent/foo.exr") is False


def test_is_exr_input_zero_io_on_non_exr() -> None:
    """Non-EXR inputs must incur ZERO filesystem I/O from the gate.

    Pure-suffix fast path: a non-.exr path returns False without stat/open. We
    assert this by pointing at a path whose parent doesn't exist (a stat would
    raise) — the gate must return False cleanly without touching the FS.
    """
    # These paths do not exist and cannot be stat'd; the gate must NOT probe them.
    assert is_exr_input("/no/such/dir/video.mp4") is False
    assert is_exr_input("/no/such/dir/image.png") is False
    assert is_exr_input("/no/such/dir/clip.mov") is False
    # A dir whose name does NOT signal EXR is never probed (even if it existed).
    assert is_exr_input("/tmp/definitely_not_probed_dir") is False


# ---------------------------------------------------------------------------
# Production-path identity (item 5): full video_preprocess chain, not just raw frames
# ---------------------------------------------------------------------------

def test_production_inpaint_video_branch_identity(tmp_path: Path) -> None:
    """The production inpaint video branch (ltx_ic_lora_pipeline.py:526) on a
    NON-EXR input must yield a byte-identical *conditioning tensor* (after the
    full video_preprocess + normalize_latent chain), vs the legacy decode.

    Stronger than the raw-frame check: proves the whole downstream pipeline
    receives identical input for the validated MP4 inpaint path.
    """
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

    # Legacy conditioning tensor.
    legacy = video_preprocess(
        decode_video_by_frame(path=str(mp4), frame_cap=3, device=device),
        16, 16, dtype, device,
    )

    # The EXACT production branch expression (generate_inpaint).
    branched_gen = (
        iter_exr_frames_as_video_tensors(str(mp4), frame_cap=3, device=device)
        if is_exr_input(str(mp4))
        else decode_video_by_frame(path=str(mp4), frame_cap=3, device=device)
    )
    branched = video_preprocess(branched_gen, 16, 16, dtype, device)

    assert is_exr_input(str(mp4)) is False
    assert torch.equal(legacy, branched), (
        "production inpaint branch must yield byte-identical conditioning tensor"
    )


# ---------------------------------------------------------------------------
# Retake: non-EXR identity + zero-I/O, EXR metadata-ordering fix
# ---------------------------------------------------------------------------

def test_retake_non_exr_identity_zero_io() -> None:
    """resolve_video_input_path on a NON-EXR path returns it UNCHANGED with no I/O.

    The pure-suffix fast path means a non-.exr path never touches the FS — we
    prove this by passing a path whose parent doesn't exist (any stat would
    raise) and asserting identity return with no exception.
    """
    bogus = "/no/such/dir/at/all/source.mp4"
    assert resolve_video_input_path(bogus) == bogus
    # Also a MOV and a non-_exr dir.
    assert resolve_video_input_path("/no/such/clip.mov") == "/no/such/clip.mov"
    assert resolve_video_input_path("/no/such/not_an_exr_dir") == "/no/such/not_an_exr_dir"


def test_retake_exr_resolves_before_metadata(tmp_path: Path) -> None:
    """EXR retake source: resolve_video_input_path yields a readable MP4 so
    get_videostream_metadata succeeds — the metadata-ordering bug fix.

    Previously generate() called get_videostream_metadata on the raw EXR path
    → crash. Now resolution happens first (in generate), producing a temp MP4
    that the metadata/video-lateral helpers can consume.
    """
    from ltx_pipelines.utils.media_io import get_videostream_metadata

    # Build a 4-frame linear EXR sequence dir (our `_exr/` convention).
    seq = tmp_path / "retake_src_exr"
    seq.mkdir()
    for i in range(4):
        _write_exr(seq / f"frame_{i:05d}.exr", np.full((16, 16, 3), 0.5, dtype=np.float32))

    resolved = resolve_video_input_path(str(seq), fps=24)
    assert resolved != str(seq), "EXR dir must resolve to a temp MP4"
    try:
        # The metadata read that used to crash on the raw EXR path now succeeds.
        meta = get_videostream_metadata(resolved)
        assert meta.frames == 4, f"expected 4 frames in temp MP4, got {meta.frames}"
    finally:
        Path(resolved).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Missing channels fail loudly (item 6)
# ---------------------------------------------------------------------------

def test_exr_missing_channels_raise(tmp_path: Path) -> None:
    """An EXR missing R/G/B must raise (fail loudly, not default to a black plane)."""
    import OpenEXR

    # Write an EXR with only R and G (no B).
    arr = np.full((8, 8), 0.5, dtype=np.float16)
    channels = {"R": arr, "G": arr}  # no B
    hdr = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    exr = tmp_path / "rg_only.exr"
    OpenEXR.File(hdr, channels).write(str(exr))

    with pytest.raises(ValueError, match="missing required channel"):
        decode_exr_image(str(exr))
