"""LTX-Desktop HDR IC-LoRA pipeline wrapper (Phase 2: upstream replacement).

Thin subclass of the upstream ``ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline``.

Generation math (stage 1 / spatial upsampler / stage 2 / decode) is delegated
to upstream **unchanged**. This wrapper overrides **only**
``_create_conditionings`` to honor the app invariants that upstream's
file-path + frame-cap conditioning loader cannot:

- decode **all** source frames (no container-metadata count, no frame cap);
- never trim source frames;
- pad **in memory** with duplicate copies of the genuine final decoded frame
  until the count is ``8n + 1`` (single-frame input stays ``1``);
- never write a temp/recompressed video to disk.

Everything else — sampling, tiling, resize, embeddings, HDR decode — is
upstream's. The ``generate`` wrapper writes the linear HDR tensor as an EXR
primary sequence (no EOTF / tonemap / clamp — linear passthrough) via
``save_exr_tensor`` and then an SDR proxy MP4 via
``encode_exr_sequence_to_mp4`` (strictly after the primary).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Literal

import torch

from api_types import OutputFormat
from ltx_core.conditioning import ConditioningItem, VideoConditionByReferenceLatent
from ltx_core.hdr import LogC3
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_pipelines.hdr_ic_lora import HDRICLoraPipeline
from ltx_pipelines.utils.media_io import (
    decode_video_by_frame,
    encode_exr_sequence_to_mp4,
    resize_and_reflect_pad,
    save_exr_tensor,
    to_vae_range,
)
from ltx_pipelines.utils.types import OffloadMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from services.color_management import ColorSpace
    from services.ltx_components import BaseFamily, ResolvedLtxComponents, TransformerFormat
    from services.media_encoder.media_encoder import MediaEncoder

logger = logging.getLogger(__name__)


class LTXHdrIcLoraPipeline(HDRICLoraPipeline):
    """Thin app wrapper over the upstream HDR IC-LoRA two-stage pipeline.

    The class attribute ``pipeline_kind`` discriminates HDR pipeline state
    from other GPU-slot pipeline types. Only the conditioning-load path is
    overridden; all generation math is upstream's.
    """

    pipeline_kind: ClassVar[Literal["hdr_ic_lora"]] = "hdr_ic_lora"

    @staticmethod
    def create(
        checkpoint_path: str | tuple[str, ...],
        upsampler_path: str,
        hdr_lora_path: str,
        device: str | torch.device,
        components: ResolvedLtxComponents | None = None,
        transformer_format: TransformerFormat = "safetensors",
        base_family: BaseFamily = "distilled",
        distilled_lora_path: str | None = None,
        scene_embeddings_path: str | None = None,
        offload_mode: OffloadMode | None = None,
        *,
        gemma_root: str | None = None,
    ) -> "LTXHdrIcLoraPipeline":
        """Construct the HDR IC-LoRA pipeline.

        Phase 2 restricts HDR to the **official distilled safetensors** base
        checkpoint (single file). ``components``, ``distilled_lora_path`` and
        ``gemma_root`` are accepted for Protocol compatibility but are not
        used for HDR generation math. ``scene_embeddings_path`` is required
        and mapped to upstream ``text_embeddings_path``; it is never
        substituted with prompt/text embeddings. Phase 3 surfaces
        user-facing model gating and forwards ``scene_embeddings_path`` from
        the handler.
        """
        # Protocol-compat params unused by HDR generation math.
        del components, distilled_lora_path, gemma_root

        if isinstance(checkpoint_path, tuple):
            raise ValueError(
                "HDR IC-LoRA initial support requires a single official distilled "
                "safetensors checkpoint; split/tuple checkpoint paths are not supported."
            )
        if transformer_format != "safetensors":
            raise ValueError(
                "HDR IC-LoRA initial support requires the official distilled "
                "safetensors base checkpoint; non-safetensors formats are not supported."
            )
        if base_family != "distilled":
            raise ValueError(
                "HDR IC-LoRA initial support requires the official distilled base "
                f"family; base_family={base_family!r} is not supported."
            )
        if not scene_embeddings_path:
            raise ValueError("scene_embeddings_path is required for HDR IC-LoRA")

        resolved_offload = offload_mode if offload_mode is not None else OffloadMode.NONE
        if resolved_offload == OffloadMode.DISK:
            raise ValueError("OffloadMode.DISK is not supported for HDR IC-LoRA.")

        resolved_device = device if isinstance(device, torch.device) else torch.device(device)
        return LTXHdrIcLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=upsampler_path,
            hdr_lora=hdr_lora_path,
            text_embeddings_path=scene_embeddings_path,
            device=resolved_device,
            offload_mode=resolved_offload,
        )

    def _create_conditionings(  # type: ignore[override]
        self,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        tiling_config: TilingConfig | None = None,
        high_quality_hdr: bool = False,
    ) -> list[ConditioningItem]:
        """Override upstream conditioning load to use in-memory decoded frames.

        Mirrors upstream's per-frame transform exactly
        (``resize_and_reflect_pad`` for ``ResizeMode.REFLECT_PAD`` →
        ``/255`` clamp → ``LogC3().compress_ldr`` → ``to_vae_range`` →
        dtype/device), the same tiled-vs-direct encode decision, and the same
        ``VideoConditionByReferenceLatent`` downscale/strength logic. Only the
        source of frames differs: all decoded source frames in source order
        plus duplicate-final-frame padding to ``8n + 1`` — never a recompressed
        temp video.
        """
        if high_quality_hdr:
            raise NotImplementedError(
                "high_quality_hdr=True is not supported by the LTX-Desktop HDR wrapper "
                "(app invariant: use all source frames + duplicate-final-frame padding)."
            )

        scale = self.reference_downscale_factor
        if scale != 1 and (height % scale != 0 or width % scale != 0):
            raise ValueError(
                f"Output dimensions ({height}x{width}) must be divisible by "
                f"reference_downscale_factor ({scale})"
            )
        ref_height = height // scale
        ref_width = width // scale

        logc3 = LogC3()
        conditionings: list[ConditioningItem] = []

        for video_path, strength in video_conditioning:
            # Decode ALL source frames in memory (frame_cap=None). Never use
            # container metadata for the count; never write a temp video.
            source_frames = list(
                decode_video_by_frame(path=video_path, frame_cap=None, device=self.device)
            )
            source_count = len(source_frames)
            if source_count == 0:
                raise ValueError(
                    f"HDR conditioning source video decoded zero frames: {video_path!r}"
                )

            padded_count = _padded_frame_count(source_count)
            if padded_count != num_frames:
                raise ValueError(
                    "HDR conditioning padded frame count does not match the requested "
                    f"num_frames: decoded {source_count} -> padded {padded_count}, "
                    f"requested {num_frames}."
                )

            # Append duplicate copies of the genuine final decoded frame only.
            padded_frames = list(source_frames)
            last_frame = source_frames[-1]
            while len(padded_frames) < padded_count:
                padded_frames.append(last_frame)

            # Same per-frame transform as upstream load_video_conditioning_hdr
            # with ResizeMode.REFLECT_PAD; concatenate along the frame dim.
            transformed: list[torch.Tensor] = []
            for frame in padded_frames:
                resized = resize_and_reflect_pad(frame.to(torch.float32), ref_height, ref_width)
                ldr = (resized / 255.0).clamp(0.0, 1.0)
                compressed = logc3.compress_ldr(ldr)
                transformed.append(
                    to_vae_range(compressed).to(device=self.device, dtype=self.dtype)
                )

            video = torch.cat(transformed, dim=2)

            if tiling_config is not None and ref_height * ref_width > self._tiled_vae_encode_threshold:
                encoded_video = video_encoder.tiled_encode(video, tiling_config)
            else:
                encoded_video = video_encoder(video)

            conditionings.append(
                VideoConditionByReferenceLatent(
                    latent=encoded_video,
                    downscale_factor=scale,
                    strength=strength,
                )
            )

        if video_conditioning:
            logger.info("[HDR IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

        return conditionings

    def generate(
        self,
        source_video_path: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,  # noqa: ARG002  # HDR is source-driven; padded count is computed from the source.
        frame_rate: float,
        output_path: str,
        output_format: OutputFormat = OutputFormat.EXR_ZIP_HALF,
        encoder: MediaEncoder | None = None,  # noqa: ARG002  # unused: upstream writers own HDR output, not the app encoder.
        proxy_path: str | None = None,
        input_colorspace: ColorSpace | None = None,  # noqa: ARG002  # HDR source is 8-bit SDR; no colorspace detection.
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        """Run the official HDR IC-LoRA two-stage flow on the source video.

        Computes the padded frame count from the decoded source video and
        delegates generation to upstream ``__call__`` (stage 1 / upsampler /
        stage 2 / decode) unchanged. Then writes the returned linear HDR
        tensor as the EXR primary sequence (no EOTF / tonemap / clamp) and,
        after the primary, the SDR proxy MP4.
        """
        source_frames = list(
            decode_video_by_frame(path=source_video_path, frame_cap=None, device=self.device)
        )
        source_count = len(source_frames)
        if source_count == 0:
            raise ValueError(
                f"HDR source video decoded zero frames: {source_video_path!r}"
            )
        padded_num_frames = _padded_frame_count(source_count)

        video: torch.Tensor = self(
            seed=seed,
            height=height,
            width=width,
            num_frames=padded_num_frames,
            frame_rate=frame_rate,
            video_conditioning=[(source_video_path, 1.0)],
            high_quality_hdr=False,
        )

        # Phase 4 writer: linear HDR tensor -> EXR primary sequence
        # (linear passthrough; no EOTF / tonemap / clamp), then SDR proxy MP4
        # strictly after the primary. The proxy must never alter the primary.
        if output_format == OutputFormat.EXR_ZIP_HALF:
            half = True
        elif output_format == OutputFormat.EXR_ZIP_FLOAT:
            half = False
        else:
            raise ValueError(
                f"HDR IC-LoRA primary output must be EXR; got output_format={output_format!r}."
            )

        out_dir = Path(output_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(int(video.shape[0])):
            save_exr_tensor(video[idx], out_dir / f"frame_{idx:06d}.exr", half=half)
        if on_progress is not None:
            on_progress(0.9)

        if proxy_path is not None:
            encode_exr_sequence_to_mp4(out_dir, Path(proxy_path), frame_rate)
        if on_progress is not None:
            on_progress(1.0)


def _padded_frame_count(source_count: int) -> int:
    """Return the ``8n + 1`` padded frame count for ``source_count`` decoded frames.

    - ``source_count == 1`` -> ``1`` (trivially ``8*0 + 1``).
    - already ``(source_count - 1) % 8 == 0`` -> unchanged.
    - otherwise -> next value strictly greater than ``source_count`` of the
      form ``8n + 1``.

    Source frames are never trimmed.
    """
    if source_count <= 0:
        raise ValueError(f"source_count must be >= 1, got {source_count}")
    if source_count == 1:
        return 1
    if (source_count - 1) % 8 == 0:
        return source_count
    n = (source_count - 1) // 8 + 1
    return 8 * n + 1
