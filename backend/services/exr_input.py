"""EXR-input decoding + input→Rec.709 (model-domain) transfer (CM-1b).

SAFETY INVARIANT (governing): for every NON-EXR input the conditioning pixels
reaching the model MUST be byte-identical to today, AND the gating fast path
incurs ZERO filesystem I/O on non-EXR inputs. Predicates/decoders here are
invoked ONLY on paths that look like EXR by a pure string check; non-EXR video
inputs (always files: .mp4/.mov/...) return from the gate without a single
stat/open. EXR is the only newly-decodable input; its frames are transferred
linear → Rec.709 gamma (the model working space) via
:mod:`services.color_management`.

Temp lifecycle: ``resolve_image_input_path`` writes ONE temp PNG (cleaned by the
caller after the external image reader consumes it); ``resolve_video_input_path``
STREAMS EXR frames → ffmpeg stdin → ONE temp MP4 (no frame materialization, no
PNG staging), cleaned by the retake caller after the lazy video iterator is
consumed. Callers own cleanup (compare returned path to the original).

Tagged-NON-bt709 *video* colorspace correction is explicitly DEFERRED to CM-1c —
not handled here. Only EXR inputs are decoded+transferred by this module.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
import threading
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from services.color_management import (
    color_to_model_space,
    detect_colorspace,
)

logger = logging.getLogger(__name__)

_EXR_EXT = ".exr"
# Sort EXR sequences by the trailing digit group (matches both our
# ``frame_%05d.exr`` output and the fixture ``Name_####.exr`` pattern).
_EXR_NUM_RE = re.compile(r"(\d+)\.exr$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Predicates — PURE-STRING fast path (zero I/O on non-EXR)
# ---------------------------------------------------------------------------

def _looks_like_exr(path: str) -> bool:
    """Pure-string EXR signal check — NO filesystem I/O.

    True iff the path ends with ``.exr`` (single-frame EXR file) OR the dir
    basename signals an EXR sequence (ends with ``_exr`` or is ``exr`` — our
    ``outputs/<prefix>_exr/`` convention and the fixture ``.../EXR/`` dir).
    """
    lower = path.lower()
    if lower.endswith(_EXR_EXT):
        return True
    stripped = lower.rstrip("/")
    return stripped.endswith("_exr") or stripped.endswith("/exr")


def is_exr_input(path: str) -> bool:
    """True iff ``path`` is an EXR source (file or sequence dir).

    Fast path is a PURE STRING check (``_looks_like_exr``) — non-EXR inputs
    (``.mp4``/``.mov``/``.png``/...) return False with ZERO filesystem I/O. The
    directory probe (``is_dir`` + ``glob *.exr``) fires ONLY when the path string
    already signals EXR, so a non-EXR path never reaches it. Non-EXR video
    inputs are always files, never directories, so they never need the dir case.
    """
    if not _looks_like_exr(path):
        return False
    # String says EXR — now (and only now) confirm on disk.
    p = Path(path)
    if p.is_file():
        return p.suffix.lower() == _EXR_EXT
    if p.is_dir():
        return any(p.glob("*" + _EXR_EXT))
    return False


# ---------------------------------------------------------------------------
# Low-level OpenEXR decode (linear float RGB) — R/G/B required
# ---------------------------------------------------------------------------

def _read_exr_rgb(path: str) -> np.ndarray:
    """Read a single EXR file → ``(H, W, 3)`` float32 linear RGB.

    Requires all of R, G, B (case-insensitive) — raises ValueError if any are
    missing (fail loudly rather than silently emitting a black channel). Half/
    float pixel types both upcast to float32. Negatives clamped to 0 (linear
    light is non-negative). Does NOT gamma-encode — keeps linear; the color
    transform applies the OETF.
    """
    import OpenEXR

    openexr: Any = OpenEXR
    f = openexr.File(path, separate_channels=True)
    channels: Any = f.channels()

    def _get(name: str) -> Any:
        ch = channels.get(name)
        if ch is not None:
            return ch
        # Case-insensitive fallback (some writers use lowercase).
        return next((c for k, c in channels.items() if k.lower() == name.lower()), None)

    planes_by_name: dict[str, np.ndarray] = {}
    for i, name in enumerate(("R", "G", "B")):
        ch = _get(name)
        if ch is None:
            raise ValueError(
                f"EXR {path} is missing required channel {name!r} — only RGB EXR "
                f"inputs are supported (found channels: {sorted(channels.keys())})"
            )
        pixels: np.ndarray = np.asarray(ch.pixels, dtype=np.float32)
        if i == 0:
            # Validate the R channel has a 2D pixel grid; G/B must match.
            if pixels.ndim != 2:
                raise ValueError(f"EXR {path} channel {name!r} is not a 2D plane")
        planes_by_name[name] = pixels
    rgb = np.stack([planes_by_name["R"], planes_by_name["G"], planes_by_name["B"]], axis=-1)
    np.clip(rgb, 0.0, None, out=rgb)  # linear light is non-negative
    return rgb


def decode_exr_image(path: str) -> np.ndarray:
    """Decode a single-frame EXR image → ``(H, W, 3)`` float32 linear RGB."""
    return _read_exr_rgb(path)


def _sorted_exr_frames(directory: Path) -> list[Path]:
    """List ``*.exr`` in ``directory`` sorted by trailing digit group."""
    frames = list(directory.glob("*" + _EXR_EXT))

    def _key(p: Path) -> int:
        m = _EXR_NUM_RE.search(p.name)
        return int(m.group(1)) if m is not None else 0

    frames.sort(key=_key)
    return frames


def _exr_sequence_files(path: str) -> list[Path]:
    """Resolve an EXR source path to its sorted frame file list."""
    p = Path(path)
    if p.is_file() and p.suffix.lower() == _EXR_EXT:
        return [p]
    return _sorted_exr_frames(p)


def decode_exr_sequence(
    path: str, max_frames: int | None = None
) -> Iterator[np.ndarray]:
    """Yield ``(H, W, 3)`` float32 linear frames from an EXR sequence.

    ``path`` is a directory containing ``*.exr`` or a single ``.exr`` file.
    Frames sorted by trailing digit (supports ``frame_%05d.exr`` and
    ``Name_####.exr``). ``max_frames`` caps the count (mirrors
    ``decode_video_by_frame``'s ``frame_cap``).
    """
    files = _exr_sequence_files(path)
    if max_frames is not None:
        files = files[:max_frames]
    for fp in files:
        yield _read_exr_rgb(str(fp))


# ---------------------------------------------------------------------------
# Linear → Rec.709 (model domain) transfer
# ---------------------------------------------------------------------------

def _to_model_domain_uint8(linear_rgb: np.ndarray, src_path: str) -> np.ndarray:
    """Transfer linear EXR RGB → Rec.709 gamma uint8 (the model input domain).

    Detects the EXR colorspace from ``src_path`` (→ LINEAR_REC709 for untagged,
    per §9.5) and applies ``color_to_model_space`` (linear → Rec.709 gamma via
    BT.709 OETF; matrix only if primaries differ). Clips to [0,1] and scales to
    uint8 — the same domain ``decode_video_by_frame`` yields for MP4 frames.
    """
    cs = detect_colorspace(src_path)
    transferred = color_to_model_space(linear_rgb, cs)
    if isinstance(transferred, torch.Tensor):
        transferred = transferred.cpu().numpy()
    arr = np.clip(np.asarray(transferred, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# Video-conditioning frame iterator (matches decode_video_by_frame output)
# ---------------------------------------------------------------------------

def iter_exr_frames_as_video_tensors(
    path: str,
    *,
    frame_cap: int | None,
    device: Any,
) -> Generator[torch.Tensor, None, None]:
    """Yield ``(1, H, W, 3)`` uint8 Rec.709-gamma tensors — drop-in for
    ``ltx_pipelines.utils.media_io.decode_video_by_frame`` on EXR inputs.

    Streams one frame at a time (no full materialization): decode linear → detect
    CS → ``color_to_model_space`` → uint8 Rec.709 gamma. Output shape/dtype
    matches ``decode_video_by_frame`` so the downstream ``video_preprocess`` +
    ``normalize_latent`` is unchanged.
    """
    files = _exr_sequence_files(path)
    if frame_cap is not None:
        files = files[:frame_cap]
    for fp in files:
        linear = _read_exr_rgb(str(fp))
        uint8 = _to_model_domain_uint8(linear, str(fp))
        # Match decode_video_by_frame: (1, H, W, 3) uint8 on device.
        yield torch.as_tensor(uint8, device=device).unsqueeze(0)


# ---------------------------------------------------------------------------
# Image-conditioning path resolution (EXR → temp PNG in Rec.709 gamma)
# ---------------------------------------------------------------------------

def _write_png(uint8_rgb: np.ndarray, path: str) -> None:
    """Write an ``(H, W, 3)`` uint8 RGB array as a PNG via PIL."""
    from PIL import Image

    Image.fromarray(uint8_rgb, mode="RGB").save(path, format="PNG")


def resolve_image_input_path(path: str) -> str:
    """Resolve an image-conditioning input path for the model domain.

    NON-EXR: returns ``path`` UNCHANGED (pure-suffix check — no I/O).
    CM-1c NOTE: sRGB image → BT.709 cross-transfer is deferred for v1 (§8
    assumption: sRGB ≈ Rec.709 within model tolerance). CM-1c is video-only.
    EXR: decodes → transfers linear → Rec.709 gamma → writes ONE temp PNG and
    returns its path, so the existing external image reader (which expects a
    readable raster path) consumes a normal sRGB/Rec.709-domain PNG unchanged.

    Temp lifecycle: the caller owns cleanup — after the external reader consumes
    the returned path, unlink it if it differs from the original ``path``.
    """
    # Pure-suffix gate — zero I/O for non-EXR image inputs (PNG/JPEG/...).
    if not path.lower().endswith(_EXR_EXT):
        return path
    uint8 = _to_model_domain_uint8(decode_exr_image(path), path)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.close()
    _write_png(uint8, tmp.name)
    logger.info("EXR image input %s materialized to Rec.709 PNG %s", path, tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# Retake / path-based-video-helper resolution (STREAM EXR → temp MP4)
# ---------------------------------------------------------------------------

def _drain_pipe(proc: subprocess.Popen[bytes]) -> tuple[list[str], list[str]]:
    """Drain both stdout+stderr of ``proc`` in reader threads (avoid deadlock)."""
    stdout_lines: list[str] = []
    stderr_lines: list[str] = []

    def _read(stream: Any, sink: list[str]) -> None:
        if stream is None:
            return
        for raw in stream:
            sink.append(raw.decode("utf-8", errors="replace"))

    out_t = threading.Thread(target=_read, args=(proc.stdout, stdout_lines), daemon=True)
    err_t = threading.Thread(target=_read, args=(proc.stderr, stderr_lines), daemon=True)
    out_t.start()
    err_t.start()
    proc.wait()
    out_t.join(timeout=5)
    err_t.join(timeout=5)
    return stdout_lines, stderr_lines


def resolve_video_input_path(path: str, *, fps: int = 24) -> str:
    """Resolve a video-source path for path-based consumers (retake).

    NON-EXR: returns ``path`` UNCHANGED (pure-suffix gate via ``_looks_like_exr``
    — zero I/O for .mp4/.mov/...).

    EXR sequence (dir or ``.exr``): STREAMS frames → ffmpeg stdin → ONE temp MP4
    (linear → Rec.709 gamma per frame, raw rgb24 piped; no full materialization,
    no PNG staging). The temp MP4 is returned so downstream
    ``get_videostream_metadata`` / ``video_latent_from_file`` /
    ``audio_latent_from_file`` consume a normal video file. EXR has no audio →
    the temp MP4 is video-only (retake handles absent audio).

    Temp lifecycle: the caller owns cleanup — unlink the returned path if it
    differs from the original ``path`` (after the lazy video iterator is consumed).
    """
    # Pure-suffix fast path — non-EXR returns immediately with no I/O.
    if not _looks_like_exr(path):
        return path
    files = _exr_sequence_files(path)
    if not files:
        return path

    import imageio_ffmpeg

    # Peek the first frame to learn H/W (single read; the body re-reads from the
    # streaming generator below so the iterator stays single-pass over files).
    first = _read_exr_rgb(str(files[0]))
    height, width = int(first.shape[0]), int(first.shape[1])

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    ff = imageio_ffmpeg.get_ffmpeg_exe()

    # rawvideo rgb24 on stdin → libx264 yuv420p MP4. Frames streamed per-file:
    # decode → transfer to Rec.709 gamma → uint8 rgb24 bytes → stdin.
    cmd: list[str] = [
        ff, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{width}x{height}", "-pix_fmt", "rgb24",
        "-framerate", str(int(fps)),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-vf", "format=yuv420p",
        tmp.name,
    ]
    proc = subprocess.Popen(  # noqa: S603 — ffmpeg subprocess
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    try:
        assert proc.stdin is not None
        for fp in files:
            linear = _read_exr_rgb(str(fp))
            uint8 = _to_model_domain_uint8(linear, str(fp))
            proc.stdin.write(uint8.tobytes())
        proc.stdin.close()
    except Exception:
        if proc.poll() is None:
            proc.kill()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    finally:
        _drain_pipe(proc)

    if proc.returncode != 0:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"EXR→MP4 ffmpeg failed (returncode={proc.returncode})")

    logger.info("EXR sequence %s (%d frames) streamed to temp MP4 %s", path, len(files), tmp.name)
    return tmp.name


# ---------------------------------------------------------------------------
# CM-1c: tagged non-bt709 VIDEO input → Rec.709 correction
# ---------------------------------------------------------------------------

def iter_video_frames_to_model_domain(
    path: str,
    *,
    frame_cap: int | None,
    device: Any,
) -> Generator[torch.Tensor, None, None]:
    """Decode video frames and correct to the Rec.709 model domain if needed.

    HARD INVARIANT: for bt709-tagged or untagged video (Rec.709-assumed) → EXACT
    passthrough (``yield from decode_video_by_frame`` — byte-identical, zero
    per-frame work). This is the identity fast path; no transform object, no
    per-frame math. The validated inpaint MP4 path is an exact no-op.

    For tagged non-bt709 video (BT.601/smpte170m, Rec.2020, etc.) → for each
    frame, apply ``color_to_model_space(frame, src=detected_CS)`` (linearize via
    the CS transfer → matrix to Rec.709-linear → bt709_oetf → re-quantize to
    uint8). This is the ONLY active path.

    Output shape/dtype/device matches ``decode_video_by_frame`` exactly
    (``(1, H, W, 3) uint8``) so downstream ``video_preprocess`` + ``normalize_latent``
    is unchanged.
    """
    from ltx_pipelines.utils.media_io import decode_video_by_frame
    from services.color_management import REC709, color_to_model_space, detect_colorspace

    cs = detect_colorspace(path)
    if cs == REC709:
        # Identity fast path: bt709-tagged or untagged (Rec.709-assumed) →
        # pure passthrough, byte-identical to today. No CS object, no per-frame math.
        yield from decode_video_by_frame(path=path, frame_cap=frame_cap, device=device)
        return

    # Tagged non-bt709: apply color_to_model_space per frame (linearize → matrix →
    # bt709_oetf → re-quantize to uint8). The model sees Rec.709-domain pixels.
    for frame in decode_video_by_frame(path=path, frame_cap=frame_cap, device=device):
        # frame is (1, H, W, 3) uint8 on device. Normalize to [0,1] float first.
        framef = frame.float() / 255.0
        corrected = color_to_model_space(framef, cs)
        if isinstance(corrected, torch.Tensor):
            yield (corrected.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        else:
            # numpy fallback (shouldn't happen for torch input, but be safe).
            arr = np.clip(corrected, 0.0, 1.0)
            yield torch.as_tensor((arr * 255.0).round().astype(np.uint8), device=device)


__all__ = [
    "decode_exr_image",
    "decode_exr_sequence",
    "is_exr_input",
    "iter_exr_frames_as_video_tensors",
    "iter_video_frames_to_model_domain",
    "resolve_image_input_path",
    "resolve_video_input_path",
]
