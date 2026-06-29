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
from enum import Enum
from typing import TYPE_CHECKING, Protocol

import torch

from api_types import OutputFormat
from services.services_utils import AudioOrNone

if TYPE_CHECKING:
    from services.color_management import ColorSpace


@dataclass(frozen=True)
class EncoderResult:
    """Outcome of an encode: absolute primary path and optional proxy path.

    ``proxy_path`` is ``None`` iff ``OutputFormat.MP4`` (the default MP4 path has
    no sidecar proxy — it IS the browser-playable file).
    """

    primary_path: str
    proxy_path: str | None


class HdrProxyPolicy(Enum):
    """How the browser-playable H.264 proxy is derived from a (possibly HDR) primary.

    The proxy is ALWAYS derived from the on-disk primary via ffmpeg (§14
    single-pass: the in-memory VAE iterator is never re-traversed). This policy
    only selects the transfer math applied on that ffmpeg path — it does NOT
    create a second encoder framework.

    * ``OFF`` — the default for every SDR path (MP4 / ProRes / SDR EXR). The
      proxy assumes the primary is gamma-domain or ≤1.0 linear and applies a
      single linear→BT.709 transfer (no tonemap). Used by every non-HDR caller.
    * ``SDR_TONEMAP_REINHARD`` — selected only by the HDR (linear scene-referred
      EXR) path, whose primary legitimately stores values >1.0. A plain
      linear→BT.709 OETF would hard-clip all highlights (bt709_oetf(2.5)≈1.56 →
      clamped to 1.0 → a blown-out proxy). Instead the proxy tonemaps with the
      deterministic global Reinhard operator ``x/(x+1)`` (mapping
      ``[0, ∞) → [0, 1)``) and THEN applies the Rec.709 OETF for an SDR
      BT.709 display-referred proxy that is playable and preserves highlight
      roll-off rather than clipping. The HDR linear EXR primary is untouched
      (values >1.0 preserved).
    """

    OFF = "off"
    SDR_TONEMAP_REINHARD = "sdr_tonemap_reinhard"


class MediaEncoder(Protocol):
    """Encode decoded VAE frame tensors into a primary + optional proxy.

    ``video`` is either a single tensor of shape ``(F, H, W, 3)`` or (as the
    tiled decoder emits) a single-pass ``Iterator[torch.Tensor]`` of chunks
    ``(N, H, W, 3)``. Values are sRGB-transfer-domain, either uint8 or float in
    [0,1] (per §0A.A). The encoder MUST NOT iterate the iterator twice.

    ``primary_path`` is a file path for MP4/MOV, or a directory path (created by
    the encoder) for an EXR sequence. ``proxy_path`` is required for non-MP4
    formats and ``None`` for MP4.

    ``hdr_proxy_policy`` selects the proxy transfer math for HDR linear
    primaries (values >1.0). It defaults to :data:`HdrProxyPolicy.OFF` (the SDR
    single-transfer proxy) and is only set to ``SDR_TONEMAP_REINHARD`` by the
    HDR linear-EXR path. MP4/ProRes ignore it (no proxy / gamma-domain primary).

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
        input_colorspace: ColorSpace | None = None,
        hdr_proxy_policy: "HdrProxyPolicy | None" = None,
    ) -> EncoderResult:
        ...


__all__ = ["EncoderResult", "HdrProxyPolicy", "MediaEncoder"]
