"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import torch

from api_types import ImageConditioningInput, OutputFormat
from services.exr_input import resolve_image_input_path
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_pipelines.utils.args import ImageConditioningInput as LtxImageConditioningInput
    from services.color_management import ColorSpace
    from services.media_encoder.media_encoder import HdrProxyPolicy, MediaEncoder


def default_tiling_config() -> TilingConfigType:
    from ltx_core.model.video_vae import TilingConfig

    return TilingConfig.default()


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(cfg_scale=3.0), MultiModalGuiderParams(cfg_scale=3.0)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


# CRF (H.264 compression quality) used for every app-created image conditioning
# input. Upstream ``ltx_pipelines`` defaults image conditioning CRF to
# ``DEFAULT_IMAGE_CRF`` (=33, lossy). The app targets a near-lossless CRF so
# conditioning frames are not recompressed/degraded before the VAE sees them.
# Centralized here (plan §11): all image-conditioning entry points must route
# construction through :func:`make_ltx_image_conditioning_input`.
IMAGE_CONDITIONING_CRF: int = 18


def make_ltx_image_conditioning_input(
    path: str, frame_idx: int, strength: float
) -> LtxImageConditioningInput:
    """Build an upstream ``ltx_pipelines`` image conditioning input with the
    app-wide CRF override applied.

    Upstream ``ltx_pipelines.utils.args.ImageConditioningInput`` is a
    ``NamedTuple`` of ``(path, frame_idx, strength, crf)`` whose ``crf`` field
    defaults to ``DEFAULT_IMAGE_CRF`` (=33). Every app entry point that builds
    image conditioning for the upstream pipelines (fast/distilled native,
    IC-LoRA) must use this helper so CRF is overridden consistently to
    :data:`IMAGE_CONDITIONING_CRF` (18). The upstream import is lazy so this
    module does not gain an import-time dependency on ``ltx_pipelines``.
    """
    from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

    return _LtxImageInput(path, frame_idx, strength, crf=IMAGE_CONDITIONING_CRF)


def make_primary_output_path(
    outputs_dir: str, prefix: str, output_format: OutputFormat, gen_id: str
) -> str:
    """Build the primary output path (file or EXR dir) by format.

    MP4 → ``<outputs>/<prefix>_<ts>_<id>.mp4``; ProRes → ``.mov``; EXR → an
    ``<prefix>_<ts>_<id>_exr`` directory. Used by handlers (Phase 2) so path
    construction is uniform & DRY.
    """
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{prefix}_{timestamp}_{gen_id}"
    if output_format in (OutputFormat.EXR_ZIP_HALF, OutputFormat.EXR_ZIP_FLOAT):
        name = f"{stem}_exr"
    elif output_format == OutputFormat.MP4:
        name = f"{stem}.mp4"
    else:
        name = f"{stem}.mov"  # all ProRes profiles
    return f"{outputs_dir}/{name}"


def make_proxy_output_path(primary_path: str, output_format: OutputFormat) -> str | None:
    """Build the proxy MP4 path, or ``None`` for MP4 (no proxy).

    For non-MP4 formats the proxy sits alongside the primary:
    ``<primary_name>_proxy.mp4``. For an EXR dir primary (``<stem>_exr``) the
    proxy is ``<stem>_exr_proxy.mp4``.
    """
    if output_format == OutputFormat.MP4:
        return None
    from pathlib import Path

    p = Path(primary_path)
    return str(p.parent / f"{p.name}_proxy.mp4")


# Encode progress split: the encoder's primary encode covers [0, _ENCODE_FRACTION]
# of the combined 0→1 range; the proxy pass covers [_ENCODE_FRACTION, 1.0]. Must
# match the encoder's ``_ENCODE_FRACTION`` constant (documented coupling).
_ENCODE_FRACTION: float = 0.6


def make_encode_progress_callback(
    update_progress: Callable[[str, int, int | None, int | None], None],
) -> Callable[[float], None]:
    """Create an ``on_progress`` callback for the encoder.

    Maps the encoder's combined 0→1 progress (encode=[0, _ENCODE_FRACTION],
    proxy=[_ENCODE_FRACTION, 1.0]) to ``update_progress(stage, int_pct, None, None)``
    with stages ``"encoding"`` (0→100% within encode) then ``"writing_proxy"``
    (0→100% within proxy). ``update_progress`` is the handler's
    ``GenerationHandler.update_progress`` method (takes integer percent).
    """
    def _cb(p: float) -> None:
        if p < _ENCODE_FRACTION:
            pct = int(round(p / _ENCODE_FRACTION * 100))
            update_progress("encoding", pct, None, None)
        else:
            pct = int(round((p - _ENCODE_FRACTION) / (1.0 - _ENCODE_FRACTION) * 100))
            update_progress("writing_proxy", pct, None, None)

    return _cb


def encode_video_output(
    *,
    video: torch.Tensor | Iterator[torch.Tensor],
    audio: AudioOrNone,
    fps: int,
    output_path: str,
    video_chunks_number_value: int,
    output_format: OutputFormat = OutputFormat.MP4,
    proxy_path: str | None = None,
    encoder: "MediaEncoder | None" = None,
    on_progress: "Callable[[float], None] | None" = None,
    total_frames: int | None = None,
    input_colorspace: "ColorSpace | None" = None,
    hdr_proxy_policy: "HdrProxyPolicy | None" = None,
) -> None:
    """Dispatch decoded VAE frames to the media encoder.

    ``output_path`` keeps its name (not ``primary_path``) so the 3 other pipeline
    call sites are unchanged. For the default ``OutputFormat.MP4`` path the call is
    byte-identical to the previous direct ``encode_video`` delegation (the encoder
    delegates to the external, validated ``encode_video`` with no color tags) —
    this guards the visually-validated default output (§7 non-goal).

    If ``encoder is None`` (legacy callers / the retake bypass), a
    ``MediaEncoderImpl`` singleton is lazily constructed. Non-MP4 formats require
    ``proxy_path`` to be set by the caller (handlers, Phase 2).

    ``on_progress`` (0.0→1.0 combined encode+proxy budget) and ``total_frames``
    are forwarded to the encoder for save-side progress (Phase 4a). MP4 ignores
    them (no hook from external ``encode_video``).

    ``hdr_proxy_policy`` threads the HDR proxy decision through the existing
    encode path without creating a second encoder framework. It defaults to
    ``None`` (→ :data:`HdrProxyPolicy.OFF`, the SDR single-transfer proxy) for
    every SDR caller; only the HDR linear-EXR path passes
    :data:`HdrProxyPolicy.SDR_TONEMAP_REINHARD` so the sidecar H.264 proxy is
    Reinhard-tonemapped instead of hard-clipping HDR highlights. The HDR linear
    EXR primary is always preserved (values >1.0); only the proxy is affected.
    """
    if encoder is None:
        encoder = _get_default_encoder()
    encoder.encode(
        video=video,
        audio=audio,
        fps=fps,
        primary_path=output_path,
        output_format=output_format,
        proxy_path=proxy_path,
        video_chunks_number=video_chunks_number_value,
        on_progress=on_progress,
        total_frames=total_frames,
        input_colorspace=input_colorspace,
        hdr_proxy_policy=hdr_proxy_policy,
    )


_default_encoder_instance: MediaEncoder | None = None


def _get_default_encoder() -> "MediaEncoder":
    """Lazily build a singleton ``MediaEncoderImpl`` for callers that don't inject.

    Heavy import (ffmpeg binary resolution, OpenEXR) stays out of module import
    time — consistent with the repo's lazy-import pattern.
    """
    global _default_encoder_instance
    if _default_encoder_instance is None:
        from services.media_encoder.media_encoder_impl import MediaEncoderImpl

        _default_encoder_instance = MediaEncoderImpl()
    return _default_encoder_instance


class DistilledNativePipeline:
    """Fast native pipeline implementation moved from ltx2_server.py."""

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str | None,
        device: torch.device | None = None,
        fp8transformer: bool = False,
    ) -> None:
        from ltx_core.quantization.fp8_cast import build_policy
        from ltx_pipelines.utils.blocks import (
            AudioDecoder,
            DiffusionStage,
            ImageConditioner,
            PromptEncoder,
            VideoDecoder,
        )
        from ltx_pipelines.utils.helpers import get_device

        if device is None:
            device = get_device()

        self.device = device
        self.dtype = torch.bfloat16

        self.prompt_encoder = PromptEncoder(
            checkpoint_path, gemma_root or "", self.dtype, device,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path, self.dtype, device,
        )
        self.stage = DiffusionStage(
            checkpoint_path,
            self.dtype,
            device,
            quantization=build_policy(checkpoint_path) if fp8transformer and device_supports_fp8(device) else None,
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, device)
        self.audio_decoder = AudioDecoder(checkpoint_path, self.dtype, device)

    @torch.inference_mode()
    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfigType | None = None,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import image_conditionings_by_replacing_latent
        from ltx_pipelines.utils.types import ModalitySpec

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        (ctx_p,) = self.prompt_encoder([prompt])
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        # CM-1b: EXR image inputs are pre-decoded → linear → Rec.709 gamma
        # (model domain) → temp PNG, so the external reader consumes a normal
        # sRGB/Rec.709-domain raster. NON-EXR paths are returned UNCHANGED
        # (literal identity — byte-identical to today). The temp PNGs are owned
        # here: cleaned up in the `finally` once image_conditioner has consumed
        # them (no leak across a generation).
        resolved_paths = [resolve_image_input_path(img.path) for img in images]
        ltx_images = [
            make_ltx_image_conditioning_input(rp, img.frame_idx, img.strength)
            for rp, img in zip(resolved_paths, images)
        ]
        try:
            conditionings = self.image_conditioner(
                lambda enc: image_conditionings_by_replacing_latent(
                    images=ltx_images,
                    height=height,
                    width=width,
                    video_encoder=enc,
                    dtype=dtype,
                    device=self.device,
                )
            )
        finally:
            # Unlink the temp PNGs produced for EXR image inputs (if any).
            from pathlib import Path as _Path

            for rp, img in zip(resolved_paths, images):
                if rp != img.path:
                    _Path(rp).unlink(missing_ok=True)

        video_state, audio_state = self.stage(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=conditionings),
            audio=ModalitySpec(context=audio_context) if audio_context is not None else None,
        )

        assert video_state is not None
        decoded_video = self.video_decoder(video_state.latent, tiling_config)
        decoded_audio = self.audio_decoder(audio_state.latent) if audio_state is not None else None
        return decoded_video, decoded_audio
