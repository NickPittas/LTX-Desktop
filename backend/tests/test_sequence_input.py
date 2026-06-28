"""Image-sequence input resolution + decode tests (CM-1b rework).

Portable — no /mnt paths; all sequences are synthesized in ``tmp_path``. Covers:

* Sequence resolution: strict version isolation (``_v01`` vs ``_v02``), padding,
  single-image-not-a-sequence, video-container fast-false.
* Decode: frame count + shape/dtype + Rec.709 color transfer (linear EXR → gamma).
* The ``decode_video_by_frame`` monkey-patch: byte-identity on a real MP4 (the
  identity invariant) + sequence routing on a sequence file.
* ``sequence_metadata``: dims/count/fps.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from services.color_management import bt709_oetf
from services.sequence_input import (
    decode_sequence_frames,
    is_sequence_file,
    resolve_sequence,
    sequence_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def _write_exr(path: Path, linear_rgb: np.ndarray, fps: tuple[int, int] | None = None) -> None:
    """Write an EXR (half) with the given linear RGB ``(H, W, 3)`` float."""
    import OpenEXR

    hdr: dict[str, object] = {
        "compression": OpenEXR.ZIP_COMPRESSION,
        "type": OpenEXR.scanlineimage,
    }
    if fps is not None:
        hdr["framesPerSecond"] = fps
    channels = {
        "R": np.ascontiguousarray(linear_rgb[:, :, 0].astype(np.float16)),
        "G": np.ascontiguousarray(linear_rgb[:, :, 1].astype(np.float16)),
        "B": np.ascontiguousarray(linear_rgb[:, :, 2].astype(np.float16)),
    }
    OpenEXR.File(hdr, channels).write(str(path))


def _write_png(path: Path, arr: np.ndarray) -> None:
    from PIL import Image

    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path, format="PNG")


def _write_mp4(path: Path, frames: np.ndarray, fps: int = 24) -> None:
    """Write a tiny MP4 from a ``(F, H, W, 3)`` uint8 array via ffmpeg."""
    tmp_dir = path.parent / (path.stem + "_frames")
    tmp_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        _write_png(tmp_dir / f"frame_{i:05d}.png", frame)
    subprocess.run(
        [_ffmpeg(), "-y", "-framerate", str(fps), "-i", str(tmp_dir / "frame_%05d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", str(path)],
        capture_output=True, check=True,
    )


# ---------------------------------------------------------------------------
# is_sequence_file — pure-string fast path (no directory scan)
# ---------------------------------------------------------------------------

def test_is_sequence_file_video_returns_false() -> None:
    """Video containers and unnumbered files return False with NO dir scan.

    Points at paths whose parent doesn't exist — any directory scan would raise.
    """
    assert is_sequence_file("/no/such/dir/clip.mp4") is False
    assert is_sequence_file("/no/such/dir/clip.mov") is False
    assert is_sequence_file("/no/such/dir/clip.mkv") is False
    # Unnumbered image (no trailing digit-run) → False.
    assert is_sequence_file("/no/such/dir/poster.png") is False
    assert is_sequence_file("/no/such/dir/render.exr") is False


def test_is_sequence_file_image_with_digits_is_potentially_true() -> None:
    """Image ext + trailing digits → True (potentially a sequence)."""
    assert is_sequence_file("/some/shot_0001.exr") is True
    assert is_sequence_file("/some/shot_0001.png") is True
    assert is_sequence_file("/some/Instant_Share_Beard_0001.exr") is True


# ---------------------------------------------------------------------------
# resolve_sequence — strict sibling match (version segments preserved LITERAL)
# ---------------------------------------------------------------------------

def test_resolve_sequence_strict_versions(tmp_path: Path) -> None:
    """``shot_v01_0001`` resolves to ONLY the v01 frames; v02 is excluded.

    Co-existing sequences in the same folder (differing by a LITERAL version
    segment) must never bleed into each other.
    """
    for v in ("v01", "v02"):
        for i in range(1, 4):
            _write_exr(
                tmp_path / f"shot_{v}_{i:04d}.exr",
                np.full((8, 8, 3), 0.5, dtype=np.float32),
            )

    spec = resolve_sequence(str(tmp_path / "shot_v01_0001.exr"))
    assert spec is not None
    assert len(spec.files) == 3
    # All resolved files are v01 — v02 must not bleed in.
    assert all("shot_v01_" in f for f in spec.files)
    assert spec.frame_numbers == (1, 2, 3)
    assert spec.pad == 4
    assert spec.prefix == "shot_v01_"
    assert spec.suffix_stem == ""


def test_resolve_sequence_padding(tmp_path: Path) -> None:
    """4-digit and 3-digit paddings are detected correctly (distinct sequences)."""
    # 4-digit sequence.
    for i in range(1, 4):
        _write_exr(tmp_path / f"a_{i:04d}.exr", np.full((4, 4, 3), 0.1, dtype=np.float32))
    spec4 = resolve_sequence(str(tmp_path / "a_0001.exr"))
    assert spec4 is not None
    assert spec4.pad == 4
    assert spec4.frame_numbers == (1, 2, 3)

    # 3-digit sequence in a different folder.
    d3 = tmp_path / "three"
    d3.mkdir()
    for i in range(1, 4):
        _write_exr(d3 / f"b_{i:03d}.exr", np.full((4, 4, 3), 0.1, dtype=np.float32))
    spec3 = resolve_sequence(str(d3 / "b_001.exr"))
    assert spec3 is not None
    assert spec3.pad == 3
    assert spec3.frame_numbers == (1, 2, 3)


def test_resolve_sequence_preserves_suffix_stem(tmp_path: Path) -> None:
    """A trailing non-digit suffix after the frame number stays LITERAL."""
    for i in range(1, 4):
        _write_png(tmp_path / f"render_{i:04d}_beauty.png", np.full((6, 6, 3), 100, dtype=np.uint8))
    spec = resolve_sequence(str(tmp_path / "render_0001_beauty.png"))
    assert spec is not None
    assert spec.prefix == "render_"
    assert spec.suffix_stem == "_beauty"
    assert spec.ext == ".png"
    assert len(spec.files) == 3


def test_resolve_sequence_tolerates_gaps(tmp_path: Path) -> None:
    """Non-contiguous frame numbers: the sequence is the sorted present set."""
    for i in (1, 2, 5, 8):  # gaps at 3,4,6,7
        _write_png(tmp_path / f"f_{i:04d}.png", np.full((4, 4, 3), 50, dtype=np.uint8))
    spec = resolve_sequence(str(tmp_path / "f_0001.png"))
    assert spec is not None
    assert spec.frame_numbers == (1, 2, 5, 8)
    assert len(spec.files) == 4


def test_single_image_not_a_sequence(tmp_path: Path) -> None:
    """One ``img_0001.png`` alone → resolve_sequence None (no siblings).

    ``is_sequence_file`` may be True (it's potentially a sequence) but with no
    matching siblings it is a single image, not a sequence.
    """
    single = tmp_path / "img_0001.png"
    _write_png(single, np.full((8, 8, 3), 200, dtype=np.uint8))

    # Potentially a sequence (image ext + trailing digits), but actually standalone.
    assert is_sequence_file(str(single)) is True
    assert resolve_sequence(str(single)) is None


def test_resolve_sequence_video_returns_none(tmp_path: Path) -> None:
    """A non-sequence file (video) → None instantly (no dir scan)."""
    mp4 = tmp_path / "clip.mp4"
    _write_mp4(mp4, np.full((1, 8, 8, 3), 50, dtype=np.uint8))
    assert resolve_sequence(str(mp4)) is None


# ---------------------------------------------------------------------------
# decode_sequence_frames — count, shape/dtype, Rec.709 color transfer
# ---------------------------------------------------------------------------

def test_decode_sequence_frames_count_and_color(tmp_path: Path) -> None:
    """Linear EXR sequence (value 0.5) → N frames, (1,H,W,3) uint8, transferred
    to Rec.709 gamma (bt709_oetf(0.5)*255 = 180)."""
    seq_dir = tmp_path / "exr_seq"
    seq_dir.mkdir()
    for i in range(3):
        _write_exr(seq_dir / f"frame_{i:04d}.exr", np.full((12, 12, 3), 0.5, dtype=np.float32))

    spec = resolve_sequence(str(seq_dir / "frame_0000.exr"))
    assert spec is not None
    frames = list(decode_sequence_frames(spec, frame_cap=None, device=torch.device("cpu")))
    assert len(frames) == 3
    for t in frames:
        assert t.shape == (1, 12, 12, 3)
        assert t.dtype == torch.uint8
    expected = int(round(float(bt709_oetf(np.array([0.5], dtype=np.float32))[0]) * 255.0))
    assert expected == 180
    assert int(frames[0][0, 0, 0, 0]) == expected


def test_decode_sequence_frames_frame_cap(tmp_path: Path) -> None:
    """frame_cap caps the count (mirrors decode_video_by_frame)."""
    seq_dir = tmp_path / "cap_seq"
    seq_dir.mkdir()
    for i in range(5):
        _write_png(seq_dir / f"f_{i:04d}.png", np.full((4, 4, 3), 128, dtype=np.uint8))
    spec = resolve_sequence(str(seq_dir / "f_0000.png"))
    assert spec is not None
    frames = list(decode_sequence_frames(spec, frame_cap=2, device=torch.device("cpu")))
    assert len(frames) == 2


def test_decode_sequence_frames_png_passthrough(tmp_path: Path) -> None:
    """sRGB/Rec.709-domain PNG (value v) → color_to_model_space is identity → v."""
    seq_dir = tmp_path / "png_seq"
    seq_dir.mkdir()
    val = 200
    for i in range(2):
        _write_png(seq_dir / f"f_{i:04d}.png", np.full((6, 6, 3), val, dtype=np.uint8))
    spec = resolve_sequence(str(seq_dir / "f_0000.png"))
    assert spec is not None
    frames = list(decode_sequence_frames(spec, frame_cap=None, device=torch.device("cpu")))
    assert len(frames) == 2
    # sRGB ≈ Rec.709 → color_to_model_space(identity) → value unchanged.
    assert int(frames[0][0, 0, 0, 0]) == val


# ---------------------------------------------------------------------------
# sequence_metadata — dims / count / fps
# ---------------------------------------------------------------------------

def test_sequence_metadata(tmp_path: Path) -> None:
    """3-frame 16x16 EXR sequence @24fps → (16, 16, 3, 24.0)."""
    seq_dir = tmp_path / "meta_seq"
    seq_dir.mkdir()
    for i in range(3):
        _write_exr(
            seq_dir / f"frame_{i:04d}.exr",
            np.full((16, 16, 3), 0.5, dtype=np.float32),
            fps=(24, 1),
        )
    width, height, count, fps = sequence_metadata(str(seq_dir / "frame_0000.exr"))
    assert (width, height) == (16, 16)
    assert count == 3
    assert fps == pytest.approx(24.0)


def test_sequence_metadata_png_default_fps(tmp_path: Path) -> None:
    """PNG sequence → default fps 24.0 (no EXR framesPerSecond header)."""
    seq_dir = tmp_path / "png_meta"
    seq_dir.mkdir()
    for i in range(2):
        _write_png(seq_dir / f"f_{i:04d}.png", np.full((10, 8, 3), 50, dtype=np.uint8))
    width, height, count, fps = sequence_metadata(str(seq_dir / "f_0000.png"))
    assert (width, height, count) == (8, 10, 2)
    assert fps == pytest.approx(24.0)


# ---------------------------------------------------------------------------
# decode_video_by_frame monkey-patch — identity + sequence routing
# ---------------------------------------------------------------------------

def test_decode_video_by_frame_patch_identity(tmp_path: Path) -> None:
    """THE identity gate: patched decode_video_by_frame on a real MP4 is
    byte-identical to the true original PyAV decode (torch.equal per frame)."""
    import services.patches.sequence_decode_patch as patch_mod

    patch_mod.install_sequence_decode_patch()  # idempotent
    from ltx_pipelines.utils import media_io

    device = torch.device("cpu")
    frames = np.stack([
        np.full((16, 16, 3), 50, dtype=np.uint8),
        np.full((16, 16, 3), 200, dtype=np.uint8),
    ])
    mp4 = tmp_path / "src.mp4"
    _write_mp4(mp4, frames)

    patched = [t.cpu() for t in media_io.decode_video_by_frame(path=str(mp4), frame_cap=2, device=device)]
    original = [
        t.cpu() for t in patch_mod.original_decode_video_by_frame(path=str(mp4), frame_cap=2, device=device)
    ]

    assert len(patched) == len(original) == 2
    for a, b in zip(patched, original):
        assert torch.equal(a, b), "patched decode_video_by_frame must be byte-identical on MP4"


def test_decode_video_by_frame_patch_sequence(tmp_path: Path) -> None:
    """A sequence file routed through the patched decode_video_by_frame yields
    the sequence frames (count matches resolve_sequence)."""
    import services.patches.sequence_decode_patch as patch_mod

    patch_mod.install_sequence_decode_patch()
    from ltx_pipelines.utils import media_io

    seq_dir = tmp_path / "patch_seq"
    seq_dir.mkdir()
    for i in range(3):
        _write_exr(seq_dir / f"shot_{i:04d}.exr", np.full((8, 8, 3), 0.5, dtype=np.float32))

    device = torch.device("cpu")
    seq_file = str(seq_dir / "shot_0000.exr")
    assert is_sequence_file(seq_file) is True

    frames = list(media_io.decode_video_by_frame(path=seq_file, frame_cap=None, device=device))
    assert len(frames) == 3
    for t in frames:
        assert t.shape == (1, 8, 8, 3)
        assert t.dtype == torch.uint8
