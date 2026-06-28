"""EXR single-file decode + input→Rec.709 (model-domain) transfer.

The dir-based EXR input path (CM-1b + P0-3) has been REPLACED by
:mod:`services.sequence_input` (sequence resolution by filename + numeric
padding) and the ``decode_video_by_frame`` monkey-patch in
:mod:`services.patches.sequence_decode_patch`. ``video_path`` is now always a
SINGLE FILE from a sequence — never a directory, never a temp MP4.

This module keeps only what is still used directly:

* :func:`decode_exr_image` — single-file EXR read (linear float RGB), reused
  per-frame by :mod:`services.sequence_input`.
* :func:`resolve_image_input_path` — EXR image-conditioning input → temp PNG in
  Rec.709 gamma (the external image reader expects a readable raster path).
* :func:`iter_video_frames_to_model_domain` — CM-1c tagged-NON-bt709 VIDEO
  correction (byte-identical passthrough for bt709/untagged). Sequence files
  pass through unchanged here — their color transfer happens inside
  ``decode_sequence_frames`` via the patched ``decode_video_by_frame``.

SAFETY INVARIANT: for every NON-sequence, bt709/untagged input the conditioning
pixels reaching the model are byte-identical to the legacy decode path.
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Generator
from typing import Any

import numpy as np
import torch

from services.color_management import (
    REC709,
    color_to_model_space,
    detect_colorspace,
)

logger = logging.getLogger(__name__)

_EXR_EXT = ".exr"


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


# ---------------------------------------------------------------------------
# Linear → Rec.709 (model domain) transfer
# ---------------------------------------------------------------------------

def _to_model_domain_uint8(linear_rgb: np.ndarray, src_path: str) -> np.ndarray:
    """Transfer linear EXR RGB → Rec.709 gamma uint8 (the model input domain).

    Detects the EXR colorspace from ``src_path`` (→ LINEAR_REC709 for untagged)
    and applies ``color_to_model_space`` (linear → Rec.709 gamma via BT.709 OETF;
    matrix only if primaries differ). Clips to [0,1] and scales to uint8 — the
    same domain ``decode_video_by_frame`` yields for MP4 frames.
    """
    cs = detect_colorspace(src_path)
    transferred = color_to_model_space(linear_rgb, cs)
    if isinstance(transferred, torch.Tensor):
        transferred = transferred.cpu().numpy()
    arr = np.clip(np.asarray(transferred, dtype=np.float32), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


# ---------------------------------------------------------------------------
# CM-1c: tagged non-bt709 VIDEO input → Rec.709 correction
# (sequence files pass through; their color transfer is inside decode_sequence_frames)
# ---------------------------------------------------------------------------

def iter_video_frames_to_model_domain(
    path: str,
    *,
    frame_cap: int | None,
    device: Any,
) -> Generator[torch.Tensor, None, None]:
    """Decode video frames and correct to the Rec.709 model domain if needed.

    Sequence files (image sequences) pass through UNCHANGED — their color
    transfer happens inside :func:`decode_sequence_frames` via the patched
    ``decode_video_by_frame`` (a sequence path routes there automatically).

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
    from services.sequence_input import is_sequence_file

    # Sequence files: color correction happens INSIDE decode_sequence_frames
    # (invoked by the patched decode_video_by_frame). Pure passthrough here so
    # the per-frame CM-1c correction below never double-transfers them.
    if is_sequence_file(path):
        from ltx_pipelines.utils.media_io import decode_video_by_frame

        yield from decode_video_by_frame(path=path, frame_cap=frame_cap, device=device)
        return

    from ltx_pipelines.utils.media_io import decode_video_by_frame

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


__all__ = [
    "decode_exr_image",
    "iter_video_frames_to_model_domain",
    "resolve_image_input_path",
]
