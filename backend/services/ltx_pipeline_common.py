"""Shared helpers and primitives for LTX video pipeline wrappers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

import torch

from api_types import ImageConditioningInput, OutputFormat
from services.exr_input import resolve_image_input_path
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_core.components.guiders import MultiModalGuiderParams
    from services.media_encoder.media_encoder import MediaEncoder


def default_tiling_config() -> TilingConfigType:
    from ltx_core.model.video_vae import TilingConfig

    return TilingConfig.default()


def default_guiders() -> tuple[MultiModalGuiderParams, MultiModalGuiderParams]:
    from ltx_core.components.guiders import MultiModalGuiderParams

    return MultiModalGuiderParams(cfg_scale=3.0), MultiModalGuiderParams(cfg_scale=3.0)


def video_chunks_number(num_frames: int, tiling_config: TilingConfigType | None) -> int:
    from ltx_core.model.video_vae import get_video_chunks_number

    return int(get_video_chunks_number(num_frames, tiling_config))


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
        from ltx_core.quantization import QuantizationPolicy
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
            quantization=QuantizationPolicy.fp8_cast() if fp8transformer and device_supports_fp8(device) else None,
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
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
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
            _LtxImageInput(rp, img.frame_idx, img.strength)
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
