"""``MediaEncoder`` Protocol + ``EncoderResult`` dataclass.

Real implementation lives in :mod:`services.media_encoder.media_encoder_impl`;
test double is ``FakeMediaEncoder`` in ``tests/fakes/services.py``.

The encoder is the single chokepoint that turns decoded VAE frame tensors into a
primary output (MP4 / ProRes MOV / EXR sequence) plus an optional browser-
playable H.264 proxy MP4. Color-science metadata is embedded on every non-MP4
format so Resolve / Premiere / Nuke / RV interpret color correctly (§2).
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Protocol

import torch

from api_types import OutputFormat
from services.services_utils import AudioOrNone


@dataclass(frozen=True)
class EncoderResult:
    """Outcome of an encode: absolute primary path and optional proxy path.

    ``proxy_path`` is ``None`` iff ``OutputFormat.MP4`` (the default MP4 path has
    no sidecar proxy — it IS the browser-playable file).
    """

    primary_path: str
    proxy_path: str | None


class MediaEncoder(Protocol):
    """Encode decoded VAE frame tensors into a primary + optional proxy.

    ``video`` is either a single tensor of shape ``(F, H, W, 3)`` or (as the
    tiled decoder emits) a single-pass ``Iterator[torch.Tensor]`` of chunks
    ``(N, H, W, 3)``. Values are sRGB-transfer-domain, either uint8 or float in
    [0,1] (per §0A.A). The encoder MUST NOT iterate the iterator twice.

    ``primary_path`` is a file path for MP4/MOV, or a directory path (created by
    the encoder) for an EXR sequence. ``proxy_path`` is required for non-MP4
    formats and ``None`` for MP4.

    ``on_progress`` (0.0–1.0) is invoked where computable; the handler converts
    to an integer percent before forwarding to the generation-progress channel
    (§0A.E). MP4 leaves progress unchanged (external ``encode_video`` has no
    hook) to protect the validated path.
    """

    def encode(
        self,
        *,
        video: "torch.Tensor | Iterator[torch.Tensor]",
        audio: AudioOrNone,
        fps: int,
        primary_path: str,
        output_format: OutputFormat,
        proxy_path: str | None,
        video_chunks_number: int,
        on_progress: Callable[[float], None] | None = None,
        total_frames: int | None = None,
    ) -> EncoderResult:
        ...


__all__ = ["EncoderResult", "MediaEncoder"]
