"""Real-encoder color/byte/luma tests for the MediaEncoder (Phase 1).

Integration-style: exercises the real ``MediaEncoderImpl`` against portable
synthetic tensors only (NO /mnt fixture paths — those live in the untracked
local validation script). Covers the §0A.N mandatory gates:
  * EXR→proxy luma check (not dark / no double-gamma),
  * ProRes color tags + explicit matrix/range,
and the §7 regression guard:
  * default MP4 path color tags STILL unspecified (byte-identical delegation).

Repo rules honored: no mocking libraries (the guardrail test scans for them),
fakes-only for handler-level DI (these tests use the REAL encoder), pyright
strict for the encoder sources themselves (tests dir is excluded from pyright).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from api_types import OutputFormat
from services.media_encoder.color import (
    ADOPTED_NEUTRAL_D65,
    BT709_CHROMATICITIES,
    bt709_eotf,
)
from services.media_encoder.media_encoder_impl import MediaEncoderImpl
from services.services_utils import AudioOrNone

# ---------------------------------------------------------------------------
# Capabilities + helpers
# ---------------------------------------------------------------------------


def test_openexr_capability() -> None:
    """OpenEXR dep must be importable (§0A.M — hard requirement, no skip)."""
    import OpenEXR  # noqa: F401

    assert OpenEXR.ZIP_COMPRESSION is not None
    assert OpenEXR.scanlineimage is not None
    assert hasattr(OpenEXR, "File")


def _ffprobe_streams(path: str) -> list[dict[str, Any]]:
    """Run ffprobe on PATH; return the parsed ``streams`` list.

    Prefers the system ``ffprobe`` (CI/dev both have it). Falls back to the
    imageio-ffmpeg-adjacent probe if needed.
    """
    ffprobe = shutil.which("ffprobe") or str(
        Path(__file__).resolve().parents[1]
        / ".venv/lib/python3.13/site-packages/imageio_ffmpeg/binaries"
        / "ffprobe"
    )
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-of", "json", path],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    streams = payload.get("streams", [])
    assert isinstance(streams, list), f"unexpected ffprobe payload: {payload}"
    return streams


def _decode_mean_luma_yuv(path: str) -> float:
    """Decode the first video frame via PyAV and return mean luma in 0..255.

    The frame is converted to YUV (Rec.709) so the value is directly comparable
    to a bt709 limited-range reference.
    """
    import av

    container = av.open(path)
    try:
        vs = next(s for s in container.streams if s.type == "video")
        frame = next(container.decode(vs))
        yuv = frame.to_ndarray(format="yuv444p")
    finally:
        container.close()
    y_plane = yuv[0].astype(np.float32)
    return float(y_plane.mean())


def _decode_per_frame_mean_luma_normalized(path: str) -> list[float]:
    """Decode EVERY video frame and return per-frame mean luma in 0..255.

    Normalizes by the stream's raw bit depth (10-bit primary vs 8-bit proxy) so
    primary and proxy luma are directly comparable — used by the ProRes
    round-trip matrix/range regression gate (§0A.N).
    """
    import av

    container = av.open(path)
    try:
        vs = next(s for s in container.streams if s.type == "video")
        bits = int(getattr(vs.codec_context, "bits_per_raw_sample", 0) or 8)
        denom = float((1 << bits) - 1)
        means: list[float] = []
        for frame in container.decode(vs):
            yuv = frame.to_ndarray(format="yuv444p")
            y = yuv[0].astype(np.float32)
            means.append(float(y.mean()) / denom * 255.0)
        return means
    finally:
        container.close()


def _ffprobe_video_stream(path: str) -> dict[str, Any]:
    """Return the first video stream dict via ffprobe."""
    streams = _ffprobe_streams(path)
    return next(s for s in streams if s.get("codec_type") == "video")


def _ffprobe_nb_frames(path: str) -> int:
    """Return the video stream frame count (nb_frames, falling back to counting)."""
    ffprobe = shutil.which("ffprobe") or "/usr/bin/ffprobe"
    r = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-count_frames", "-show_entries", "stream=nb_read_frames",
         "-of", "default=nokey=1:noprint_wrappers=1", path],
        capture_output=True, text=True, check=True,
    )
    return int(r.stdout.strip())


def _make_gray_frames(num_frames: int, h: int, w: int, gray: float = 0.5) -> torch.Tensor:
    """Return a uint8 ``(F, H, W, 3)`` tensor filled with a flat sRGB gray value."""
    val = int(round(gray * 255.0))
    arr = np.full((num_frames, h, w, 3), val, dtype=np.uint8)
    return torch.from_numpy(arr)


def _make_gradient_frames(num_frames: int, h: int, w: int) -> torch.Tensor:
    """Return a uint8 ``(F, H, W, 3)`` colored horizontal-gradient clip.

    R increases across width, G decreases, B constant. A matrix-sensitive
    gradient (non-gray) is required to catch RGB→YUV matrix drift in the ProRes
    round-trip gate (pure gray maps to the same Y regardless of matrix).
    """
    xs = np.linspace(0, 255, w, dtype=np.uint8)
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = xs
    frame[:, :, 1] = 255 - xs
    frame[:, :, 2] = 128
    return torch.from_numpy(np.stack([frame] * num_frames))


def _make_audio(sample_rate: int = 44100, samples: int = 4410) -> AudioOrNone:
    from ltx_core.types import Audio

    # Stereo silence — content irrelevant; only stream presence is asserted.
    waveform = torch.zeros(1, 2, samples, dtype=torch.float32)
    return Audio(waveform=waveform, sampling_rate=sample_rate)


# ---------------------------------------------------------------------------
# EXR primary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt,expected_dtype", [
    (OutputFormat.EXR_ZIP_HALF, np.float16),
    (OutputFormat.EXR_ZIP_FLOAT, np.float32),
])
def test_exr_writes_bt709_metadata(
    tmp_path: Path, fmt: OutputFormat, expected_dtype: Any
) -> None:
    import OpenEXR

    encoder = MediaEncoderImpl()
    primary = tmp_path / "exr_out"
    video = _make_gray_frames(3, 64, 64, gray=0.5)

    encoder.encode(
        video=video,
        audio=None,
        fps=24,
        primary_path=str(primary),
        output_format=fmt,
        proxy_path=None,
        video_chunks_number=1,
    )

    frames = sorted(primary.glob("frame_*.exr"))
    assert len(frames) == 3, f"expected 3 EXR frames, got {len(frames)}"

    f = OpenEXR.File(str(frames[0]), separate_channels=True)
    h = f.header()

    # chromaticities round-trips as an 8-tuple of floats ≈ BT.709 (atol 1e-4).
    chroma = h["chromaticities"]
    assert isinstance(chroma, tuple) and len(chroma) == 8
    np.testing.assert_allclose(np.array(chroma, dtype=np.float32),
                               np.array(BT709_CHROMATICITIES, dtype=np.float32), atol=1e-4)

    # adoptedNeutral ≈ D65.
    adopted = np.asarray(h["adoptedNeutral"], dtype=np.float32)
    np.testing.assert_allclose(adopted, ADOPTED_NEUTRAL_D65, atol=1e-4)

    # compression == ZIP; colorSpace label present.
    assert h["compression"] == OpenEXR.ZIP_COMPRESSION
    assert h["colorSpace"] == "lin_rec709_scene"

    # channel dtype selects HALF / FLOAT, shape matches.
    ch = f.channels()
    for name in ("R", "G", "B"):
        assert name in ch, f"missing channel {name}"
        pixels = ch[name].pixels
        assert pixels.dtype == expected_dtype
        assert pixels.shape == (64, 64)


def test_exr_linearization_applied(tmp_path: Path) -> None:
    """Input code 0.5 must store ≈ bt709_eotf(0.5) ≈ 0.259 (BT.709 EOTF, §9.1).

    Was sRGB EOTF ≈ 0.214 before the Rec.709-working-space swap. The ~17.5%
    mid-gray difference between sRGB and BT.709 inverse OETF is the core fix.
    Half-rounding tolerance applied.
    """
    import OpenEXR

    encoder = MediaEncoderImpl()
    primary = tmp_path / "exr_lin"
    video = _make_gray_frames(1, 16, 16, gray=0.5)

    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.EXR_ZIP_HALF, proxy_path=None, video_chunks_number=1,
    )

    frame = sorted(primary.glob("frame_*.exr"))[0]
    f = OpenEXR.File(str(frame), separate_channels=True)
    r = f.channels()["R"].pixels.astype(np.float32)
    expected = float(bt709_eotf(torch.tensor(0.5)).item())
    # half rounding tolerance (~ relative 5e-3 + absolute 2e-2 covers float16).
    np.testing.assert_allclose(r.mean(), expected, atol=2e-2)


def test_exr_writes_framespersecond(tmp_path: Path) -> None:
    import OpenEXR

    encoder = MediaEncoderImpl()
    primary = tmp_path / "exr_fps"
    video = _make_gray_frames(1, 16, 16)
    fps = 30

    encoder.encode(
        video=video, audio=None, fps=fps, primary_path=str(primary),
        output_format=OutputFormat.EXR_ZIP_HALF, proxy_path=None, video_chunks_number=1,
    )

    frame = sorted(primary.glob("frame_*.exr"))[0]
    h = OpenEXR.File(str(frame)).header()
    assert h["framesPerSecond"] == Fraction(fps, 1)


# ---------------------------------------------------------------------------
# ProRes primary
# ---------------------------------------------------------------------------

_PRORES_FORMATS = [
    OutputFormat.PRORES_PROXY,
    OutputFormat.PRORES_LT,
    OutputFormat.PRORES_422,
    OutputFormat.PRORES_422_HQ,
    OutputFormat.PRORES_4444,
    OutputFormat.PRORES_4444_XQ,
]

# Expected pix_fmt per profile (§0B): 0-3 → yuv422p10le, 4-5 → yuv444p12le.
_PRORES_EXPECTED_PIXFMT = {
    OutputFormat.PRORES_PROXY: "yuv422p10le",
    OutputFormat.PRORES_LT: "yuv422p10le",
    OutputFormat.PRORES_422: "yuv422p10le",
    OutputFormat.PRORES_422_HQ: "yuv422p10le",
    OutputFormat.PRORES_4444: "yuv444p12le",
    OutputFormat.PRORES_4444_XQ: "yuv444p12le",
}


@pytest.mark.parametrize("fmt", _PRORES_FORMATS)
def test_prores_pixfmt_and_color_tags(tmp_path: Path, fmt: OutputFormat) -> None:
    encoder = MediaEncoderImpl()
    primary = tmp_path / f"out_{fmt.value}.mov"
    video = _make_gray_frames(4, 16, 16, gray=0.5)

    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=fmt, proxy_path=None, video_chunks_number=1,
    )
    assert primary.exists() and primary.stat().st_size > 0

    streams = _ffprobe_streams(str(primary))
    assert len(streams) >= 1
    vs = streams[0]
    assert vs["codec_name"] == "prores"

    # Pix fmt per profile.
    assert vs["pix_fmt"] == _PRORES_EXPECTED_PIXFMT[fmt], (
        f"{fmt}: expected {_PRORES_EXPECTED_PIXFMT[fmt]}, got {vs['pix_fmt']}"
    )

    # Color tags — accept the ffmpeg-build string variance noted in §0B/§14.
    assert vs.get("color_range") in ("tv", "limited"), vs.get("color_range")
    assert vs.get("color_transfer") == "bt709", vs.get("color_transfer")
    assert vs.get("color_primaries") == "bt709", vs.get("color_primaries")
    assert vs.get("color_space") == "bt709", vs.get("color_space")
    # References are `unspecified`; we omit the flag (bundled ffmpeg lacks it),
    # so ffprobe returns the key absent (None) — all three are acceptable.
    assert vs.get("chroma_location") in (None, "unspecified", "topleft", "left"), (
        vs.get("chroma_location")
    )


@pytest.mark.parametrize("fmt", _PRORES_FORMATS)
def test_prores_is_video_only(tmp_path: Path, fmt: OutputFormat) -> None:
    """ProRes primary is video-only (audio lives in the proxy) — §0A audio decision."""
    encoder = MediaEncoderImpl()
    primary = tmp_path / f"vo_{fmt.value}.mov"
    video = _make_gray_frames(2, 16, 16)

    # Audio is intentionally passed; the ProRes path must still drop it.
    encoder.encode(
        video=video, audio=_make_audio(), fps=24, primary_path=str(primary),
        output_format=fmt, proxy_path=None, video_chunks_number=1,
    )

    streams = _ffprobe_streams(str(primary))
    assert len(streams) == 1, f"ProRes must be video-only; got {len(streams)} streams"
    assert streams[0]["codec_type"] == "video"


# ---------------------------------------------------------------------------
# Proxies
# ---------------------------------------------------------------------------


def test_proxy_from_prores_has_audio_and_tags(tmp_path: Path) -> None:
    encoder = MediaEncoderImpl()
    primary = tmp_path / "primary.mov"
    proxy = tmp_path / "proxy.mp4"
    video = _make_gray_frames(4, 16, 16, gray=0.5)

    encoder.encode(
        video=video, audio=_make_audio(), fps=24, primary_path=str(primary),
        output_format=OutputFormat.PRORES_422_HQ, proxy_path=str(proxy),
        video_chunks_number=1,
    )
    assert proxy.exists() and proxy.stat().st_size > 0

    streams = _ffprobe_streams(str(proxy))
    video_streams = [s for s in streams if s["codec_type"] == "video"]
    audio_streams = [s for s in streams if s["codec_type"] == "audio"]
    assert len(video_streams) == 1
    assert len(audio_streams) == 1, f"proxy must carry audio; streams={streams}"

    vs = video_streams[0]
    assert vs.get("color_range") in ("tv", "limited")
    assert vs.get("color_transfer") == "bt709"
    assert vs.get("color_primaries") == "bt709"
    assert vs.get("color_space") == "bt709"


def test_proxy_from_exr_not_dark(tmp_path: Path) -> None:
    """MANDATORY gate (§0A.N): EXR→proxy must apply linear→bt709 (not dark / no double-gamma).

    A mid-gray sRGB 0.5 input must produce a proxy whose mean luma sits in the
    same band as the ProRes→proxy reference (~limited-range Y for 0.5). A missing
    ``-apply_trc linear`` would render the linearized 0.214 ~ 30-40 units darker.
    """
    encoder = MediaEncoderImpl()
    # Same source frames for both paths → luma must match.
    gray_frames = lambda: _make_gray_frames(6, 64, 64, gray=0.5)

    # Reference: ProRes 422 HQ → proxy (gamma-domain, no transfer change).
    ref_primary = tmp_path / "ref.mov"
    ref_proxy = tmp_path / "ref_proxy.mp4"
    encoder.encode(
        video=gray_frames(), audio=None, fps=24, primary_path=str(ref_primary),
        output_format=OutputFormat.PRORES_422_HQ, proxy_path=str(ref_proxy),
        video_chunks_number=1,
    )

    # EXR → proxy (must re-apply the transfer).
    exr_primary = tmp_path / "exr"
    exr_proxy = tmp_path / "exr_proxy.mp4"
    encoder.encode(
        video=gray_frames(), audio=None, fps=24, primary_path=str(exr_primary),
        output_format=OutputFormat.EXR_ZIP_HALF, proxy_path=str(exr_proxy),
        video_chunks_number=1,
    )

    # ffprobe tag sanity (bt709 / tv).
    streams = _ffprobe_streams(str(exr_proxy))
    vs = next(s for s in streams if s["codec_type"] == "video")
    assert vs.get("color_range") in ("tv", "limited")
    assert vs.get("color_transfer") == "bt709"

    ref_luma = _decode_mean_luma_yuv(str(ref_proxy))
    exr_luma = _decode_mean_luma_yuv(str(exr_proxy))

    # Absolute floor: a 0.5 mid-gray must NOT decode dark (limited-range Y for
    # 0.5 ≈ 125; anything < 90 indicates a missing transfer / double-gamma bug).
    assert exr_luma > 90.0, (
        f"EXR proxy is dark (mean Y={exr_luma:.1f}); likely missing -apply_trc linear"
    )
    # Relative: EXR proxy luma must track the ProRes reference within a tight band
    # (catches a partial/double transfer application).
    assert abs(exr_luma - ref_luma) <= 20.0, (
        f"EXR proxy luma {exr_luma:.1f} drifts from ProRes reference {ref_luma:.1f} "
        f"(delta {abs(exr_luma - ref_luma):.1f}) — transfer mismatch"
    )


def test_proxy_from_exr_has_correct_frame_count(tmp_path: Path) -> None:
    """Regression for the single-frame bug: the proxy must encode the WHOLE sequence.

    Previously ``_proxy_from_exr`` passed a literal ``frame_00000.exr`` to ffmpeg,
    encoding exactly ONE frame regardless of sequence length. With the image-
    sequence PATTERN (``frame_%05d.exr``) the proxy must carry all input frames.
    """
    encoder = MediaEncoderImpl()
    num_frames = 5
    exr_primary = tmp_path / "exr"
    exr_proxy = tmp_path / "exr_proxy.mp4"
    encoder.encode(
        video=_make_gray_frames(num_frames, 32, 32, gray=0.5), audio=None, fps=24,
        primary_path=str(exr_primary), output_format=OutputFormat.EXR_ZIP_HALF,
        proxy_path=str(exr_proxy), video_chunks_number=1,
    )

    # The EXR sequence itself must have all frames.
    written = sorted(exr_primary.glob("frame_*.exr"))
    assert len(written) == num_frames

    # The proxy must contain exactly num_frames video frames (the bug gave 1).
    assert _ffprobe_nb_frames(str(exr_proxy)) == num_frames, (
        f"EXR proxy frame count mismatch — expected {num_frames} (sequence length), "
        f"got {_ffprobe_nb_frames(str(exr_proxy))} (single-frame bug?)"
    )


def test_prores_proxy_roundtrip_luma(tmp_path: Path) -> None:
    """MANDATORY ProRes matrix/range round-trip gate (§0A.N/§0A.C).

    Encode a matrix-sensitive colored gradient as ProRes 422 HQ, transcode to the
    H.264 proxy, decode BOTH via PyAV, and compare per-frame mean luma (normalized
    to 8-bit). A matrix/range drift or double-gamma would shift luma well outside
    a tight band. ProRes→proxy is YUV(bt709,tv)→YUV(bt709,tv) so no matrix
    conversion is mathematically required; this test PROVES the filter-free proxy
    transcode does not drift.
    """
    encoder = MediaEncoderImpl()
    primary = tmp_path / "rt.mov"
    proxy = tmp_path / "rt_proxy.mp4"
    video = _make_gradient_frames(4, 64, 64)

    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.PRORES_422_HQ, proxy_path=str(proxy),
        video_chunks_number=1,
    )

    primary_means = _decode_per_frame_mean_luma_normalized(str(primary))
    proxy_means = _decode_per_frame_mean_luma_normalized(str(proxy))

    # Same number of frames.
    assert len(primary_means) == len(proxy_means) == 4, (
        f"frame count mismatch: primary={len(primary_means)} proxy={len(proxy_means)}"
    )

    diffs = [abs(p - q) for p, q in zip(primary_means, proxy_means)]
    max_diff = max(diffs)
    mean_diff = float(np.mean(diffs))
    # Tight tolerance: 10→8-bit quantization + 4:2:2→4:2:0 are luma-preserving
    # (chroma-only subsampling), so per-frame mean Y must stay within ~3 in 8-bit.
    assert mean_diff < 3.0 and max_diff < 3.0, (
        f"ProRes→proxy luma drift: mean_diff={mean_diff:.2f} max_diff={max_diff:.2f} "
        f"primary_means={[round(m, 1) for m in primary_means]} "
        f"proxy_means={[round(m, 1) for m in proxy_means]} — matrix/range drift detected"
    )


# ---------------------------------------------------------------------------
# MP4 regression guard (§7)
# ---------------------------------------------------------------------------


def test_mp4_is_rec709_tagged(tmp_path: Path) -> None:
    """Default MP4 must be Rec.709-tagged (bt709 / tv) via post-tag remux (§9.2).

    The pixel stream still comes from the external ``encode_video`` (libx264 /
    yuv420p); only VUI/metadata is added by the remux pass (container tags +
    ``h264_metadata`` SPS VUI). This intentionally retags the previously-
    unspecified MP4 per the user rule "mp4 should be rec709".
    """
    encoder = MediaEncoderImpl()
    primary = tmp_path / "out.mp4"
    video = _make_gray_frames(3, 32, 32, gray=0.5)

    result = encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.MP4, proxy_path=None, video_chunks_number=1,
    )
    assert result.primary_path == str(primary)
    assert result.proxy_path is None
    assert primary.exists() and primary.stat().st_size > 0

    vs = _ffprobe_video_stream(str(primary))
    assert vs["pix_fmt"] == "yuv420p", vs["pix_fmt"]
    assert vs.get("color_range") in ("tv", "limited"), vs.get("color_range")
    assert vs.get("color_transfer") == "bt709", vs.get("color_transfer")
    assert vs.get("color_primaries") == "bt709", vs.get("color_primaries")
    assert vs.get("color_space") == "bt709", vs.get("color_space")


def test_mp4_pixels_byte_identical_to_external(tmp_path: Path) -> None:
    """The post-tag remux must NOT alter pixels — only VUI/metadata (§9.2).

    Encodes the same source twice: once directly via the external
    ``encode_video`` (untagged), once via our MP4 branch (remuxed+tagged), then
    decodes a frame from each via PyAV and asserts the raw RGB bytes are
    identical. Guards against any accidental pixel-touching reimplementation.
    """
    from ltx_pipelines.utils.media_io import encode_video
    import av

    video_a = _make_gray_frames(2, 32, 32, gray=0.5)
    video_b = _make_gray_frames(2, 32, 32, gray=0.5)

    external_mp4 = tmp_path / "external.mp4"
    encode_video(video=video_a, fps=24, audio=None, output_path=str(external_mp4),
                 video_chunks_number=1)

    tagged_mp4 = tmp_path / "tagged.mp4"
    MediaEncoderImpl().encode(
        video=video_b, audio=None, fps=24, primary_path=str(tagged_mp4),
        output_format=OutputFormat.MP4, proxy_path=None, video_chunks_number=1,
    )

    def _first_frame_rgb(path: str) -> np.ndarray:
        c = av.open(path)
        try:
            vs = next(s for s in c.streams if s.type == "video")
            return next(c.decode(vs)).to_ndarray(format="rgb24").tobytes()
        finally:
            c.close()

    assert _first_frame_rgb(str(external_mp4)) == _first_frame_rgb(str(tagged_mp4)), (
        "MP4 post-tag remux altered pixels — must be byte-identical (VUI-only change)"
    )


# ---------------------------------------------------------------------------
# Cleanup-on-failure (§0A.J) — comprehensive partial-output removal
# ---------------------------------------------------------------------------


def test_prores_cleanup_on_proxy_failure(tmp_path: Path) -> None:
    """On ANY encode failure, partial primary AND proxy must be removed (§0A.J).

    Forces a deterministic failure in the proxy step by pointing ``proxy_path``
    inside a path component that is already a FILE (so the output cannot be
    created). The ProRes primary encodes fine, then the proxy step raises; the
    ``encode()`` wrapper must clean up BOTH the partial primary and the proxy.
    """
    encoder = MediaEncoderImpl()
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")  # a file, not a directory

    primary = tmp_path / "primary.mov"
    proxy = tmp_path / "blocker" / "proxy.mp4"  # parent is a file → cannot create

    with pytest.raises(Exception):
        encoder.encode(
            video=_make_gray_frames(2, 16, 16), audio=None, fps=24,
            primary_path=str(primary),
            output_format=OutputFormat.PRORES_422_HQ,
            proxy_path=str(proxy),
            video_chunks_number=1,
        )

    assert not primary.exists(), "partial ProRes primary must be cleaned up on failure"
    assert not proxy.exists()


def test_exr_cleanup_on_proxy_failure(tmp_path: Path) -> None:
    """Same cleanup guarantee for the EXR path (removes the EXR directory)."""
    encoder = MediaEncoderImpl()
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"x")

    primary = tmp_path / "exr"
    proxy = tmp_path / "blocker" / "proxy.mp4"

    with pytest.raises(Exception):
        encoder.encode(
            video=_make_gray_frames(2, 16, 16), audio=None, fps=24,
            primary_path=str(primary),
            output_format=OutputFormat.EXR_ZIP_HALF,
            proxy_path=str(proxy),
            video_chunks_number=1,
        )

    assert not primary.exists(), "partial EXR directory must be cleaned up on failure"
    assert not proxy.exists()


# ---------------------------------------------------------------------------
# Phase 4a: encode progress (on_progress)
# ---------------------------------------------------------------------------


def test_exr_encode_progress_monotonic(tmp_path: Path) -> None:
    """EXR encode: on_progress called monotonically non-decreasing, final ≈ 1.0."""
    encoder = MediaEncoderImpl()
    progress: list[float] = []
    primary = tmp_path / "exr_prog"
    video = _make_gray_frames(4, 16, 16, gray=0.5)
    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.EXR_ZIP_HALF, proxy_path=None,
        video_chunks_number=1, total_frames=4,
        on_progress=lambda p: progress.append(p),
    )
    assert len(progress) > 0, "on_progress must be called during EXR encode"
    # Monotonic non-decreasing.
    for i in range(1, len(progress)):
        assert progress[i] >= progress[i - 1] - 1e-9, (
            f"progress decreased at index {i}: {progress[i - 1]} → {progress[i]}"
        )
    # Final ≈ 1.0.
    assert progress[-1] == pytest.approx(1.0, abs=0.01)


def test_prores_encode_progress(tmp_path: Path) -> None:
    """ProRes encode: on_progress called; final ≈ 1.0 (frame-based may be coarse)."""
    encoder = MediaEncoderImpl()
    progress: list[float] = []
    primary = tmp_path / "prores_prog.mov"
    video = _make_gray_frames(4, 16, 16, gray=0.5)
    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.PRORES_422_HQ, proxy_path=None,
        video_chunks_number=1, total_frames=4,
        on_progress=lambda p: progress.append(p),
    )
    assert len(progress) > 0, "on_progress must be called during ProRes encode"
    assert progress[-1] == pytest.approx(1.0, abs=0.01)


def test_mp4_encode_no_progress(tmp_path: Path) -> None:
    """MP4 encode: on_progress must NOT be called (external encode_video has no hook)."""
    encoder = MediaEncoderImpl()
    progress: list[float] = []
    primary = tmp_path / "mp4_prog.mp4"
    video = _make_gray_frames(3, 32, 32, gray=0.5)
    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.MP4, proxy_path=None,
        video_chunks_number=1, total_frames=3,
        on_progress=lambda p: progress.append(p),
    )
    assert len(progress) == 0, (
        "MP4 must not call on_progress (external encode_video has no hook)"
    )


# ---------------------------------------------------------------------------
# CM-2: output-CS preservation (tagged EXR input → EXR output)
# ---------------------------------------------------------------------------

def _encode_exr_with_inputcs(
    tmp_path: Path, input_cs: Any | None, gray: float = 0.5
) -> tuple[Any, str]:
    """Encode a 1-frame EXR with the given input_colorspace; return (OpenEXR.File, primary_dir)."""
    import OpenEXR

    encoder = MediaEncoderImpl()
    primary = tmp_path / f"cm2_{id(input_cs)}"
    video = _make_gray_frames(1, 16, 16, gray=gray)
    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.EXR_ZIP_HALF, proxy_path=None,
        video_chunks_number=1, input_colorspace=input_cs,
    )
    frame = sorted(primary.glob("frame_*.exr"))[0]
    return OpenEXR.File(str(frame), separate_channels=True), str(primary)


@pytest.mark.parametrize("input_cs,name", [
    ("ACES_AP0", "aces_ap0"),
    ("LINEAR_REC2020", "lin_rec2020"),
])
def test_exr_out_preserves_tagged_input_cs(tmp_path: Path, input_cs: str, name: str) -> None:
    """CM-2: tagged EXR input → output EXR in the input's colorspace."""
    from services.color_management import ACES_AP0, LINEAR_REC2020, primaries_to_exr_chromaticities

    cs_map = {"ACES_AP0": ACES_AP0, "LINEAR_REC2020": LINEAR_REC2020}
    cs = cs_map[input_cs]
    f, _ = _encode_exr_with_inputcs(tmp_path, cs)
    h = f.header()

    # Chromaticities match the input CS primaries.
    expected_chroma = primaries_to_exr_chromaticities(cs.primaries)
    actual_chroma = h["chromaticities"]
    assert isinstance(actual_chroma, tuple) and len(actual_chroma) == 8
    np.testing.assert_allclose(
        np.array(actual_chroma, dtype=np.float32),
        np.array(expected_chroma, dtype=np.float32), atol=5e-4,
    )

    # colorSpace attribute matches the input CS name.
    assert h["colorSpace"] == name


def test_exr_out_default_when_no_inputcs(tmp_path: Path) -> None:
    """input_colorspace=None → BT.709/D65 + lin_rec709_scene (today's behavior)."""
    from services.media_encoder.color import BT709_CHROMATICITIES, LINEAR_REC709_SCENE_COLORSPACE

    f, _ = _encode_exr_with_inputcs(tmp_path, None)
    h = f.header()

    np.testing.assert_allclose(
        np.array(h["chromaticities"], dtype=np.float32),
        np.array(BT709_CHROMATICITIES, dtype=np.float32), atol=1e-4,
    )
    assert h["colorSpace"] == LINEAR_REC709_SCENE_COLORSPACE


def test_exr_out_default_for_linear_rec709_input(tmp_path: Path) -> None:
    """input_colorspace=LINEAR_REC709 → still BT.709/D65 default (no-op preservation)."""
    from services.color_management import LINEAR_REC709
    from services.media_encoder.color import BT709_CHROMATICITIES, LINEAR_REC709_SCENE_COLORSPACE

    f, _ = _encode_exr_with_inputcs(tmp_path, LINEAR_REC709)
    h = f.header()

    np.testing.assert_allclose(
        np.array(h["chromaticities"], dtype=np.float32),
        np.array(BT709_CHROMATICITIES, dtype=np.float32), atol=1e-4,
    )
    assert h["colorSpace"] == LINEAR_REC709_SCENE_COLORSPACE


def test_prores_ignores_input_colorspace(tmp_path: Path) -> None:
    """ProRes with input_colorspace=ACES_AP0 → still Rec.709-tagged output (CM-1c deferral)."""
    from services.color_management import ACES_AP0

    encoder = MediaEncoderImpl()
    primary = tmp_path / "cm2_prores.mov"
    video = _make_gray_frames(2, 16, 16, gray=0.5)
    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.PRORES_422_HQ, proxy_path=None,
        video_chunks_number=1, input_colorspace=ACES_AP0,
    )
    vs = _ffprobe_video_stream(str(primary))
    assert vs.get("color_space") == "bt709", "ProRes must stay Rec.709 (CM-1c deferral)"
    assert vs.get("color_transfer") == "bt709"


def test_exr_linear_passthrough_preserves_hdr_values(tmp_path: Path) -> None:
    """HDR linear input (values > 1.0) must be preserved in EXR — NOT clamped.

    Regression for the clamp-before-linear-branch bug that clipped HDR values.
    With LINEAR_REC709 input_colorspace, the encoder writes values as-is without
    clamp or EOTF.
    """
    import OpenEXR

    from services.color_management import LINEAR_REC709

    encoder = MediaEncoderImpl()
    primary = tmp_path / "hdr_linear_exr"

    # Float tensor with HDR values > 1.0 (linear scene-referred).
    hdr_video = torch.full((1, 8, 8, 3), 2.5, dtype=torch.float32)  # 2.5 >> 1.0

    encoder.encode(
        video=hdr_video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.EXR_ZIP_FLOAT, proxy_path=None,
        video_chunks_number=1, input_colorspace=LINEAR_REC709,
    )

    frames = sorted(primary.glob("frame_*.exr"))
    assert len(frames) == 1
    f = OpenEXR.File(str(frames[0]), separate_channels=True)
    ch = f.channels()
    r_pixels = ch["R"].pixels
    # HDR value 2.5 must be preserved (> 1.0), NOT clamped to 1.0.
    assert r_pixels.max() > 1.5, f"HDR value clipped: max={r_pixels.max()}"
    np.testing.assert_allclose(r_pixels[0, 0], 2.5, atol=0.01)


def test_exr_default_path_still_clamps_and_applies_eotf(tmp_path: Path) -> None:
    """Non-HDR default EXR (input_colorspace=None) must STILL clamp + apply bt709_eotf.

    Regression: the clamp was moved inside the non-linear branches to fix HDR
    clipping. This verifies the default path still clamps to [0, 1] and applies
    BT.709 EOTF (not identity).
    """
    import OpenEXR

    encoder = MediaEncoderImpl()
    primary = tmp_path / "default_exr"

    # uint8 gray=0.5 (128/255 ≈ 0.502 in [0,1]).
    video = _make_gray_frames(1, 8, 8, gray=0.5)

    encoder.encode(
        video=video, audio=None, fps=24, primary_path=str(primary),
        output_format=OutputFormat.EXR_ZIP_FLOAT, proxy_path=None,
        video_chunks_number=1,  # no input_colorspace → default bt709_eotf path
    )

    frames = sorted(primary.glob("frame_*.exr"))
    assert len(frames) == 1
    f = OpenEXR.File(str(frames[0]), separate_channels=True)
    ch = f.channels()
    r_pixels = ch["R"].pixels
    # bt709_eotf(0.502) ≈ 0.255 — NOT identity (would be 0.502 if no EOTF).
    assert abs(r_pixels[0, 0] - 0.502) > 0.1, "EOTF not applied — value too close to input"
    # All values ≤ 1.0 (clamped).
    assert r_pixels.max() <= 1.0 + 1e-6
