"""Monkey-patch: make image-sequence files decode through the video path.

The external ``ICLoraPipeline._create_conditionings`` (and the retake path's
``video_latent_from_file``) decode a user-supplied video via PyAV
(``decode_video_by_frame`` / ``decode_video_from_file``) and read container
metadata via ``get_videostream_metadata`` / ``get_videostream_fps``. None of
those can open a single frame from an image sequence (an ``.exr``/``.dpx``/…).

This patch installs thin wrappers on those four functions in
``ltx_pipelines.utils.media_io`` and rebinds the module-level
``decode_video_by_frame`` reference imported by ``ltx_pipelines.ic_lora``:

* For a **sequence file** (``is_sequence_file`` true + resolvable) the wrappers
  route to :func:`services.sequence_input.decode_sequence_frames` (decode +
  color → Rec.709) / :func:`sequence_metadata` — no temp file, no directory.
* For **every other path** (video containers, single unnumbered images) the
  wrappers delegate to the ORIGINAL function with IDENTICAL args + return —
  byte-identical. This is the identity invariant: the validated inpaint/retake
  MP4 paths are an exact no-op.

Install is idempotent and applied at import (mirroring the existing
``services.patches.*`` modules imported once in ``ltx2_server.py``). It is also
callable via :func:`install_sequence_decode_patch` for tests.

Usage:
    import services.patches.sequence_decode_patch  # noqa: F401
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

import torch

logger = logging.getLogger(__name__)

_installed: bool = False
# Captured ONCE, before patching, so the wrappers delegate to the true originals
# without infinite recursion.
_original_decode_video_by_frame: Any = None
_original_decode_video_from_file: Any = None
_original_get_videostream_metadata: Any = None
_original_get_videostream_fps: Any = None


def install_sequence_decode_patch() -> None:
    """Install the sequence-aware wrappers (idempotent). Safe to call many times."""
    global _installed, _original_decode_video_by_frame, _original_decode_video_from_file
    global _original_get_videostream_metadata, _original_get_videostream_fps
    if _installed:
        return

    try:
        from ltx_pipelines.utils import media_io  # noqa: PLC0415
    except ImportError:  # pragma: no cover — ltx_pipelines always present in runtime/tests
        logger.warning("Could not install sequence decode patch: ltx_pipelines unavailable")
        return

    try:
        import ltx_pipelines.ic_lora as ic_lora_module  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        ic_lora_module = None

    from services.sequence_input import (  # noqa: PLC0415
        decode_sequence_frames,
        is_sequence_file,
        resolve_sequence,
        sequence_metadata,
    )

    if _original_decode_video_by_frame is None:
        _original_decode_video_by_frame = media_io.decode_video_by_frame
    if _original_decode_video_from_file is None:
        _original_decode_video_from_file = media_io.decode_video_from_file
    if _original_get_videostream_metadata is None:
        _original_get_videostream_metadata = media_io.get_videostream_metadata
    if _original_get_videostream_fps is None:
        _original_get_videostream_fps = media_io.get_videostream_fps

    def wrapped_decode_video_by_frame(
        path: str,
        device: Any,
        starting_frame: int = 0,
        frame_cap: int | None = None,
    ) -> Iterator[torch.Tensor]:
        if is_sequence_file(path):
            spec = resolve_sequence(path)
            if spec is not None:
                return decode_sequence_frames(spec, frame_cap=frame_cap, device=device)
        # Non-sequence OR unresolvable single image → original (byte-identical).
        # starting_frame honored only on the original path (sequences decode in
        # order; frame_cap caps the count).
        return _original_decode_video_by_frame(
            path=path, device=device, starting_frame=starting_frame, frame_cap=frame_cap
        )

    def wrapped_decode_video_from_file(
        path: str,
        device: Any,
        start_time: float = 0.0,
        max_duration: float | None = None,
    ) -> Iterator[torch.Tensor]:
        if is_sequence_file(path):
            spec = resolve_sequence(path)
            if spec is not None:
                # Sequences have no PTS; honor max_duration via a frame cap when
                # known, else yield all frames. start_time is ignored (sequences
                # are indexed, not PTS-seekable).
                frame_cap: int | None = None
                if max_duration is not None:
                    _, _, _, fps = sequence_metadata(path)
                    frame_cap = max(0, round(float(max_duration) * fps))
                return decode_sequence_frames(spec, frame_cap=frame_cap, device=device)
        return _original_decode_video_from_file(
            path=path, device=device, start_time=start_time, max_duration=max_duration
        )

    def wrapped_get_videostream_metadata(path: str) -> Any:
        if is_sequence_file(path):
            spec = resolve_sequence(path)
            if spec is not None:
                from ltx_core.types import VideoPixelShape  # noqa: PLC0415

                width, height, count, fps = sequence_metadata(path)
                # VideoPixelShape(batch, frames, height, width, fps).
                return VideoPixelShape(1, count, height, width, fps)
        return _original_get_videostream_metadata(path)

    def wrapped_get_videostream_fps(path: str) -> float:
        if is_sequence_file(path):
            spec = resolve_sequence(path)
            if spec is not None:
                _, _, _, fps = sequence_metadata(path)
                return fps
        return _original_get_videostream_fps(path)

    # Replace on the source module so future `from media_io import ...` callers
    # and lazy in-function imports pick up the wrapper.
    media_io.decode_video_by_frame = wrapped_decode_video_by_frame
    media_io.decode_video_from_file = wrapped_decode_video_from_file
    media_io.get_videostream_metadata = wrapped_get_videostream_metadata
    media_io.get_videostream_fps = wrapped_get_videostream_fps

    # Rebind the module-level reference the external ic_lora pipeline imported
    # at its import time (Python binds `from m import f` to the function object).
    if ic_lora_module is not None and hasattr(ic_lora_module, "decode_video_by_frame"):
        ic_lora_module.decode_video_by_frame = wrapped_decode_video_by_frame

    _installed = True
    logger.info("Installed sequence decode patch on ltx_pipelines.utils.media_io")


# The original (un-patched) decode_video_by_frame — exposed for the byte-identity
# test (compares patched output vs the true original on a real MP4).
def original_decode_video_by_frame(
    path: str,
    device: Any,
    starting_frame: int = 0,
    frame_cap: int | None = None,
) -> Iterator[torch.Tensor]:
    """Forward to the captured original PyAV decoder (for identity tests)."""
    if _original_decode_video_by_frame is None:  # pragma: no cover
        from ltx_pipelines.utils.media_io import decode_video_by_frame  # noqa: PLC0415

        return decode_video_by_frame(
            path=path, device=device, starting_frame=starting_frame, frame_cap=frame_cap
        )
    return _original_decode_video_by_frame(
        path=path, device=device, starting_frame=starting_frame, frame_cap=frame_cap
    )


# Apply on import (mirrors services.patches.pinned_pool_fix etc.).
install_sequence_decode_patch()
