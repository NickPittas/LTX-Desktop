"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from api_types import ImageConditioningInput, OutputFormat
from services.exr_input import iter_video_frames_to_model_domain
from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import (
    default_tiling_config,
    encode_video_output,
    make_ltx_image_conditioning_input,
    video_chunks_number,
)
from services.media_encoder.media_encoder import HdrProxyPolicy
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace

_fp8_lora_fuse_patched = False

# ponytail: mask_grow_px controls LTXVDilateVideoMask radii only (derive_stage_radii).
# INPAINT_BLEND1_LOW_RES_DILATION=5 for bridge blend (stage1, node 5266, linked input) to soften edge ghosting at stage1.
# Stage2 blend uses user-controlled laplacian_blend_grow param directly.
INPAINT_BLEND1_LOW_RES_DILATION = 5


def derive_stage_radii(mask_grow_px: int) -> tuple[int, int]:
    """Derive stage1 (half-res) and stage2 (full-res) mask dilation radii from mask_grow_px.

    Effective runtime radii come from linked workflow inputs:
      - Stage1 (half-res): spatial_radius=15 (node 5382, linked via 5400 PrimitiveInt)
      - Stage2 (full-res): spatial_radius=30 (node 5379, computed as 2*15 via 5372 ComfyMathExpression)

    For mask_grow_px=30: returns (15, 30).
    For mask_grow_px=0: returns (0, 0) — no dilation on either stage.
    For mask_grow_px=1: returns (1, 1) — minimal unit dilation.

    Node widget defaults are ignored while linked; this function derives
    radii from the configurable mask_grow_px parameter.
    """
    if mask_grow_px == 0:
        return (0, 0)
    # ponytail: stage2 = mask_grow_px (full-res); stage1 = ceil-div (half-res → ~half radius)
    stage2 = mask_grow_px
    stage1 = (mask_grow_px + 1) // 2
    return (stage1, stage2)


def _quantize_fp8_same_layout(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    weight_fp32 = weight.to(torch.float32)
    fp8 = torch.finfo(torch.float8_e4m3fn)
    max_abs = torch.amax(torch.abs(weight_fp32))
    scale = fp8.max / max_abs.clamp_min(1e-12)
    quantized = torch.clamp(weight_fp32 * scale, min=fp8.min, max=fp8.max).to(torch.float8_e4m3fn)
    return quantized, scale.reciprocal()


def _install_kijai_fp8_lora_fuse_patch() -> None:
    global _fp8_lora_fuse_patched  # noqa: PLW0603
    if _fp8_lora_fuse_patched:
        return

    import ltx_core.loader.fuse_loras as fuse_loras  # noqa: PLC0415
    from ltx_core.quantization.fp8_scaled_mm import quantize_weight_to_fp8_per_tensor  # noqa: PLC0415

    def _layout_aware_fuse_delta_with_scaled_fp8(
        deltas: torch.Tensor,
        weight: torch.Tensor,
        key: str,
        scale_key: str,
        model_sd: Any,
    ) -> dict[str, torch.Tensor]:
        weight_scale = model_sd.sd[scale_key]
        if weight.shape == deltas.shape:
            original_weight = weight.to(torch.float32) * weight_scale
            new_weight = original_weight + deltas.to(torch.float32)
            new_fp8_weight, new_weight_scale = _quantize_fp8_same_layout(new_weight)
            return {key: new_fp8_weight, scale_key: new_weight_scale}

        original_weight = weight.t().to(torch.float32) * weight_scale
        new_weight = original_weight + deltas.to(torch.float32)
        new_fp8_weight, new_weight_scale = quantize_weight_to_fp8_per_tensor(new_weight)
        return {key: new_fp8_weight, scale_key: new_weight_scale}

    setattr(fuse_loras, "_fuse_delta_with_scaled_fp8", _layout_aware_fuse_delta_with_scaled_fp8)
    _fp8_lora_fuse_patched = True


def _vae_compatible_frame_count(num_frames: int) -> int:
    """Max 1+8*k frame count <= num_frames for VAE latent compatibility."""
    return 1 + 8 * max(0, (num_frames - 1) // 8)


def _vae_padded_frame_count(num_frames: int) -> int:
    """Next 1+8*k frame count >= num_frames for VAE latent compatibility.

    Official LTX pads to next valid 8n+1 frame count then crops output back.
    """
    return ((num_frames - 2) // 8 + 1) * 8 + 1


# ── HDR scene-embedding prompt-encoder injection ──────────────────────
#
# The pinned ``ltx_pipelines.ic_lora.ICLoraPipeline.__call__`` does NOT accept
# ``video_context``/``audio_context`` kwargs. It constructs them internally via
# ``self.prompt_encoder([prompt], ...)``. For HDR we inject pre-computed scene
# embeddings by temporarily replacing ``prompt_encoder`` with a wrapper that
# returns the HDR tensors instead of encoding text. This preserves the pinned
# pipeline's flow entirely — no unsupported kwargs are passed.


class _HDRPromptContext:
    """Minimal stand-in for ``PromptContext`` carrying HDR scene embeddings.

    The pinned pipeline reads ``ctx.video_encoding`` and ``ctx.audio_encoding``
    from the prompt-encoder return value (line 180 of ``ic_lora.py``).
    """

    __slots__ = ("video_encoding", "audio_encoding")

    def __init__(self, video_encoding: torch.Tensor, audio_encoding: torch.Tensor | None) -> None:
        self.video_encoding = video_encoding
        self.audio_encoding = audio_encoding


class _HDRPromptEncoderWrapper:
    """Replaces ``pipeline.prompt_encoder`` for HDR inference.

    When called (matching the ``PromptEncoder.__call__`` signature), returns a
    single-element tuple of :class:`_HDRPromptContext` carrying the HDR
    ``video_context`` scene-embedding tensor (moved/cast to the pipeline's
    device/dtype).

    HDR is video-only, so ``audio_context`` is normally ``None``. However, the
    pinned ``ICLoraPipeline.__call__`` UNCONDITIONALLY builds an audio modality
    from whatever ``audio_encoding`` the prompt-encoder yields — yielding
    ``None`` crashes the transformer's audio args preprocessor
    (``AttributeError: 'NoneType' object has no attribute 'view'`` at
    ``audio_args_preprocessor.prepare(audio, video)``). When no explicit
    ``audio_context`` is supplied, this wrapper borrows a valid
    ``audio_encoding`` from the real prompt encoder so upstream's audio
    modality builds and runs. The resulting audio output is discarded by
    ``generate()`` via ``_is_hdr_video_only_path`` — the HDR file remains
    video-only. The HDR ``video_encoding`` always comes from scene embeddings.
    """

    def __init__(
        self,
        video_context: torch.Tensor,
        audio_context: torch.Tensor | None,
        device: torch.device,
        dtype: torch.dtype,
        original_encoder: Any,
    ) -> None:
        self._video = video_context.to(device=device, dtype=dtype)
        self._audio = (
            audio_context.to(device=device, dtype=dtype)
            if audio_context is not None
            else None
        )
        self._original_encoder = original_encoder
        self._device = device
        self._dtype = dtype

    def __call__(self, prompts: list[str], **kwargs: Any) -> tuple[_HDRPromptContext, ...]:
        audio = self._audio
        if audio is None:
            # Borrow a valid audio_encoding from the real prompt encoder so the
            # pinned pipeline's unconditional audio modality builds without
            # crashing. enhance_first_prompt is forced off — we only need a
            # validly-shaped audio tensor (the HDR video_encoding from scene
            # embeddings overrides whatever the encoder produces for video), so
            # skip the heavy GGUF/Gemma enhance path. Generated audio is later
            # discarded by generate() (_is_hdr_video_only_path).
            fallback_kwargs = dict(kwargs)
            fallback_kwargs["enhance_first_prompt"] = False
            (real_ctx,) = self._original_encoder(prompts, **fallback_kwargs)
            real_audio = getattr(real_ctx, "audio_encoding", None)
            if real_audio is not None:
                audio = real_audio.to(device=self._device, dtype=self._dtype)
        return (_HDRPromptContext(self._video, audio),)


@contextmanager
def _swap_prompt_encoder_for_hdr(
    pipeline: Any,
    video_context: torch.Tensor,
    audio_context: torch.Tensor | None,
) -> Generator[None, None, None]:
    """Temporarily replace ``pipeline.prompt_encoder`` with an HDR injector.

    The original encoder is restored in the ``finally`` block so non-HDR calls
    are unaffected even if the HDR inference raises.
    """
    original = pipeline.prompt_encoder
    wrapper = _HDRPromptEncoderWrapper(
        video_context,
        audio_context,
        pipeline.device,
        getattr(pipeline, "dtype", torch.bfloat16),
        original_encoder=original,
    )
    pipeline.prompt_encoder = wrapper
    try:
        yield
    finally:
        pipeline.prompt_encoder = original


class LTXIcLoraPipeline:
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        lora_paths: list[str],
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
        lora_strength: float = 1.0,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_paths=lora_paths,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            components=components,
            lora_strength=lora_strength,
        )

    def __init__(
        self,
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        lora_paths: list[str],
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
        lora_strength: float = 1.0,
    ) -> None:
        self._components = components
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.ic_lora import ICLoraPipeline

        is_gguf = components is not None and components.transformer_format == "gguf"
        is_split = (
            components is not None
            and components.transformer_format == "safetensors"
            and components.video_vae_path is not None
        )

        if components is not None and components.gemma_root is not None:
            from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

            install_gguf_prompt_encoder_patch()

        if is_gguf:
            quantization = None
        elif is_split and device_supports_fp8(device):
            from services.patches.gguf_loader_fix import kijai_fp8_quantization_policy

            quantization = kijai_fp8_quantization_policy()
        else:
            from ltx_core.quantization import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None

        # ponytail: split safetensors 22B does not fit full residency on 32GB;
        # stream 2 layers at a time unless explicit mode set.
        if is_split and streaming_prefetch_count is None:
            streaming_prefetch_count = 2
        self._streaming_prefetch_count = streaming_prefetch_count
        # ponytail: one strength applies uniformly to all LoRAs in the stack;
        # split per-LoRA only if product needs it.
        lora_entries = [
            LoraPathStrengthAndSDOps(path=lp, strength=lora_strength, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
            for lp in lora_paths
        ]
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            spatial_upsampler_path=upsampler_path,
            gemma_root=gemma_root or "",
            loras=lora_entries,
            device=device,
            quantization=quantization,
        )

        if is_gguf:
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_gguf_loader

            install_gguf_loader(self.pipeline)
            c = self._components
            install_gguf_component_paths(
                self.pipeline,
                checkpoint_path,
                video_vae_path=c.video_vae_path if c else None,
                audio_vae_path=c.audio_vae_path if c else None,
            )

        if is_split:
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_kijai_transformer_config_patch

            c = self._components
            assert c is not None  # is_split guarantees this
            install_kijai_transformer_config_patch(self.pipeline, checkpoint_path)
            _install_kijai_fp8_lora_fuse_patch()
            install_gguf_component_paths(
                self.pipeline,
                checkpoint_path,
                video_vae_path=c.video_vae_path,
                audio_vae_path=c.audio_vae_path,
            )

    def _inpaint_streaming_prefetch_count(self, frames: int) -> int | None:
        """Return streaming_prefetch_count for inpaint based on frame count.

        If self._streaming_prefetch_count is explicitly set (not None), return it unchanged
        (env override does not override explicit pipeline mode).

        Otherwise, try LTX_INPAINT_STREAM_PREFETCH env var for long inpaint:
        - unset/empty/invalid/<=0: default 2 for frames >=97, None for short frames.
        - valid positive int >=1: use that value for frames >=97.
        """
        if self._streaming_prefetch_count is not None:
            return self._streaming_prefetch_count

        if frames >= 97:
            import os

            raw = os.environ.get("LTX_INPAINT_STREAM_PREFETCH", "")
            if raw:
                try:
                    val = int(raw)
                    if val >= 1:
                        return val
                except (ValueError, TypeError):
                    pass
            # ponytail: default 2 streams more layers than safest 1, better VRAM
            # utilization but higher risk of OOM on small GPUs. Tune after measurement.
            return 2
        return None

    def _run_inference(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        tiling_config: TilingConfigType,
        mask_path: str | None = None,
        conditioning_strength: float = 1.0,
        original_video_path: str | None = None,
        hdr_video_context: torch.Tensor | None = None,
        hdr_audio_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        import ltx_pipelines.ic_lora as ic_lora_module

        load_mask_video = cast(Callable[..., Any], getattr(ic_lora_module, "_load_mask_video"))
        # ponytail: trim mask frames to match VAE-compatible count (1+8*k)
        num_frames_vae = _vae_compatible_frame_count(num_frames)
        mask: Any | None = (
            load_mask_video(
                mask_path=mask_path,
                height=height // 2,
                width=width // 2,
                num_frames=num_frames_vae,
            )
            if mask_path is not None
            else None
        )

        # Build inference kwargs for the pinned ICLoraPipeline.__call__.
        # The pinned pipeline does NOT accept video_context/audio_context kwargs —
        # it constructs them internally via self.prompt_encoder. For HDR we inject
        # scene embeddings by temporarily swapping prompt_encoder (approach A).
        inference_kwargs: dict[str, Any] = dict(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[
                make_ltx_image_conditioning_input(img.path, img.frame_idx, img.strength)
                for img in images
            ],
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            streaming_prefetch_count=self._streaming_prefetch_count,
            conditioning_attention_mask=mask,
            conditioning_attention_strength=conditioning_strength,
        )

        if hdr_video_context is not None:
            # HDR: swap prompt_encoder with scene-embedding injector so the
            # pinned pipeline's __call__ receives the HDR contexts via its
            # normal prompt_encoder flow — no unsupported kwargs passed.
            with _swap_prompt_encoder_for_hdr(
                self.pipeline, hdr_video_context, hdr_audio_context
            ):
                return self.pipeline(**inference_kwargs)
        else:
            return self.pipeline(**inference_kwargs)

    @staticmethod
    def _is_hdr_video_only_path(
        hdr_video_context: torch.Tensor | None,
        output_postprocess: Callable[[torch.Tensor], torch.Tensor] | None,
    ) -> bool:
        """True when the HDR video-only generation path is active.

        HDR is strictly video-only. Both ``hdr_video_context`` (the HDR
        scene-embedding path) and ``output_postprocess`` (the HDR LogC3 →
        linear decode) are only ever supplied together by the HDR handler and
        are absent for every other generation mode.

        Even when ``hdr_audio_context=None`` is passed, the pinned
        ``ICLoraPipeline.__call__`` may still build and return an audio
        modality internally (its ``__call__`` constructs audio from whatever
        its prompt-encoder yields). When this returns True, ``generate()``
        intentionally discards any such ``audio`` before encoding so the HDR
        output is video-only (linear scene-referred EXR frames, no audio
        mux). Non-HDR generation is unaffected: audio flows through unchanged.
        """
        return hdr_video_context is not None or output_postprocess is not None

    @staticmethod
    def _composite_in_outpainting(
        video: torch.Tensor,
        original_video_path: str,
        mask_path: str,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Blend generated video tensor with original video using mask video.

        White mask (255) = keep generated region, black mask (0) = preserve original.
        Reads original/mask via decode_video_by_frame + video_preprocess, composites
        in float [0, 1] space, returns tensor in same shape/dtype as input video.
        """
        from ltx_pipelines.utils.media_io import decode_video_by_frame, video_preprocess

        # Read original video: decode yields (1, H, W, 3) uint8 → preprocess gives (1, 3, F, H, W) in [-1, 1]
        orig_gen = decode_video_by_frame(path=original_video_path, frame_cap=num_frames, device=device)
        orig_norm = video_preprocess(orig_gen, height, width, torch.float32, device)
        orig_01 = (orig_norm[0].permute(1, 2, 3, 0) + 1.0) / 2.0  # (F, H, W, 3) in [0, 1]

        # Read mask
        mask_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames, device=device)
        mask_norm = video_preprocess(mask_gen, height, width, torch.float32, device)
        mask_01 = (mask_norm[0].mean(dim=0) + 1.0) / 2.0  # (F, H, W) in [0, 1], grayscale

        # Handle short original/mask: repeat last frame or black fallback
        F = video.shape[0]
        if orig_01.shape[0] < F:
            last = orig_01[-1:, ...]
            pad = F - orig_01.shape[0]
            orig_01 = torch.cat([orig_01, last.expand(pad, -1, -1, -1)], dim=0)
        if mask_01.shape[0] < F:
            pad = F - mask_01.shape[0]
            mask_01 = torch.cat([mask_01, torch.zeros(pad, height, width, device=device, dtype=torch.float32)], dim=0)

        # Convert generated to [0, 1] float
        gen_01 = video.to(dtype=torch.float32, device=device)
        if gen_01.max() > 1.0:
            gen_01 = gen_01 / 255.0  # assume uint8 [0, 255]

        # Composite: result = gen * mask + orig * (1 - mask)
        mask_3ch = mask_01.unsqueeze(-1).expand(-1, -1, -1, 3).to(device=device)
        composite_01 = gen_01 * mask_3ch + orig_01.to(device=device) * (1.0 - mask_3ch)

        # Convert back to input dtype/range
        if video.dtype == torch.uint8:
            return (composite_01.clamp(0, 1) * 255).to(dtype=torch.uint8, device=device)
        return composite_01.clamp(0, 1).to(dtype=video.dtype, device=device)

    @staticmethod
    def _apply_raw_mask_guard(
        blend: torch.Tensor,
        raw_mask: torch.Tensor,
        original_frames: torch.Tensor,
        blur_radius: int = 0,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """Clamp generated pixels outside raw (undilated) user mask back to original.

        Processes frames in chunks over the temporal dimension to avoid OOM on
        long videos (196f 1080p = ~4.7 GiB full-frame temporary). Returns CPU
        tensor; caller downstream (encode_video_output) accepts CPU.

        When blur_radius > 0, threshold raw_mask > 0.5 then spatial-box-blur the
        binary alpha with avg_pool2d for a soft feathered edge at final output.
        blur_radius=0 preserves exact grayscale compositing.

        mask is (F, H, W) grayscale [0, 1]; blend and original_frames are (F, H, W, 3) [0, 1].
        Returns CPU tensor (F, H, W, 3) in [0, 1].
        """
        # ponytail: chunking avoids ~4.7 GiB full-frame temporary;
        # blur is per-frame avg_pool2d over H/W only (no temporal dependency),
        # so chunking over frames is exact. GPU → CPU per chunk keeps peak low.
        num_frames = blend.shape[0]
        device = blend.device
        dtype = blend.dtype
        orig = original_frames[:num_frames]
        alpha_raw = raw_mask[:num_frames]

        chunks: list[torch.Tensor] = []
        for start in range(0, num_frames, chunk_size):
            end = min(start + chunk_size, num_frames)
            alpha_chunk = alpha_raw[start:end].to(device=device, dtype=dtype)
            if blur_radius > 0:
                alpha_chunk = (alpha_chunk > 0.5).to(dtype)
                k = 2 * blur_radius + 1
                alpha_chunk = F.avg_pool2d(
                    alpha_chunk.unsqueeze(1),
                    kernel_size=k,
                    stride=1,
                    padding=blur_radius,
                    count_include_pad=False,
                ).squeeze(1)
                alpha_chunk = alpha_chunk.clamp(0.0, 1.0)
            alpha_4d = alpha_chunk.unsqueeze(-1)  # (C, H, W, 1)
            chunk_result = (
                blend[start:end] * alpha_4d
                + orig[start:end].to(device=device, dtype=dtype) * (1.0 - alpha_4d)
            )
            chunks.append(chunk_result.detach().cpu())

        return torch.cat(chunks, dim=0)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_conditioning: list[tuple[str, float]],
        output_path: str,
        mask_path: str | None = None,
        conditioning_strength: float = 1.0,
        original_video_path: str | None = None,
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
        hdr_video_context: torch.Tensor | None = None,
        hdr_audio_context: torch.Tensor | None = None,
        output_postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        tiling_config = default_tiling_config()
        result = self._run_inference(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            video_conditioning=video_conditioning,
            tiling_config=tiling_config,
            mask_path=mask_path,
            conditioning_strength=conditioning_strength,
            original_video_path=original_video_path,
            hdr_video_context=hdr_video_context,
            hdr_audio_context=hdr_audio_context,
        )
        video, audio = result

        # HDR is video-only: the pinned ICLoraPipeline.__call__ may still
        # build and return an audio modality internally even when we passed
        # hdr_audio_context=None (it constructs audio from whatever the
        # prompt-encoder yields). When the HDR scene-context / output-postprocess
        # path is active, intentionally discard any such audio so
        # encode_video_output receives audio=None and writes no audio stream
        # (linear scene-referred EXR frames only). Non-HDR generation is
        # untouched: audio flows through unchanged.
        if self._is_hdr_video_only_path(hdr_video_context, output_postprocess):
            audio = None

        if original_video_path is not None and mask_path is not None:
            if isinstance(video, Iterator):
                video = torch.cat(list(video), dim=0)
            video = self._composite_in_outpainting(
                video=video,
                original_video_path=original_video_path,
                mask_path=mask_path,
                height=height,
                width=width,
                num_frames=num_frames,
                device=self.pipeline.device,
            )

        # HDR output postprocess: apply LogC3 → linear decode (or any transform)
        # to the decoded video tensor before encoding. Applied once, before
        # the encoder, so EXR receives linear and proxy receives SDR-tonemapped.
        if output_postprocess is not None:
            if isinstance(video, Iterator):
                video = torch.cat(list(video), dim=0)
            video = output_postprocess(video)

        # HDR proxy policy: when the HDR linear (scene-referred, values >1.0)
        # path is active, the sidecar H.264 proxy must be SDR-tonemapped
        # (Reinhard) rather than hard-clipped. Threaded through the existing
        # encode path (single encoder framework) — only the proxy transfer math
        # changes. The HDR linear EXR primary is always preserved. Non-HDR
        # generation passes None → the encoder's SDR default (HdrProxyPolicy.OFF).
        hdr_proxy_policy: HdrProxyPolicy | None = (
            HdrProxyPolicy.SDR_TONEMAP_REINHARD
            if self._is_hdr_video_only_path(hdr_video_context, output_postprocess)
            else None
        )

        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=video, audio=audio, fps=int(frame_rate), output_path=output_path,
            video_chunks_number_value=chunks, output_format=output_format,
            encoder=encoder, proxy_path=proxy_path,
            on_progress=on_progress,
            input_colorspace=input_colorspace, total_frames=num_frames,
            hdr_proxy_policy=hdr_proxy_policy,
        )

    # ------------------------------------------------------------------ #
    # Official two-stage IC-LoRA inpaint (LTX-2.3_ICLoRA_Inpaint_Two_Stage)
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def generate_inpaint(  # noqa: PLR0913,PLR0915
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        video_path: str,
        mask_path: str,
        output_path: str,
        conditioning_strength: float = 1.0,
        mask_grow_px: int = 30,
        laplacian_blend_grow: int = 12,
        final_mask_blur_px: int = 6,
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
    ) -> None:
        """Official two-stage IC-LoRA inpaint pipeline.

        White mask = inpaint region, black mask = keep original.
        Uses green composite (#66FF00) conditioning, Laplacian pyramid
        inter-stage blending, and two denoising passes at half then full res.
        """
        import logging

        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
        from ltx_pipelines.utils.denoisers import SimpleDenoiser
        from ltx_pipelines.utils.helpers import (
            assert_resolution,
            combined_image_conditionings,
        )
        from ltx_pipelines.utils.media_io import (
            decode_video_by_frame,
            video_preprocess,
        )
        from ltx_pipelines.utils.types import ModalitySpec

        from .official_inpaint import (
            dilate_video_mask,
            green_composite_preprocess,
            laplacian_pyramid_blend,
        )

        logger = logging.getLogger(__name__)

        assert_resolution(height=height, width=width, is_two_stage=True)

        # Derive mask dilation radii from configurable mask_grow_px
        stage1_radius, stage2_radius = derive_stage_radii(mask_grow_px)

        half_h, half_w = height // 2, width // 2
        num_frames_vae_padded = _vae_padded_frame_count(num_frames)
        denoise_streaming_prefetch_count = self._inpaint_streaming_prefetch_count(num_frames_vae_padded)
        device = self.pipeline.device
        dtype = torch.bfloat16  # ponytail: matches ICLoraPipeline.dtype

        # ------------------------------------------------------------------ #
        # 1. Encode prompt and create contexts
        # ------------------------------------------------------------------ #
        # ponytail: encode prompt before loading full video/mask tensors to avoid
        # GGUF Gemma + 196f video tensor VRAM overlap (~31.6GB peak). Prompt encoder
        # builds/uses/frees, then video preprocessing allocates full tensors.
        logger.info("[inpaint] Encoding prompt")
        # Official: stage2=seed (42), stage1=seed+1 (43)
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        noiser = GaussianNoiser(generator=generator)

        (ctx_p,) = self.pipeline.prompt_encoder(
            [prompt],
            enhance_first_prompt=False,
            streaming_prefetch_count=self._streaming_prefetch_count,
        )
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding
        assert video_context is not None and audio_context is not None

        # ------------------------------------------------------------------ #
        # 2. Load input video and mask
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Loading video and mask")

        # Load video frames at half res (for stage 1 conditioning) and full res.
        # Sequence files (image sequences) decode through the patched
        # decode_video_by_frame inside iter_video_frames_to_model_domain (their
        # color transfer happens in decode_sequence_frames). CM-1c: tagged
        # non-bt709 VIDEO is corrected to Rec.709 here (byte-identical
        # passthrough for bt709/untagged — the validated inpaint MP4 path is an
        # exact no-op).
        video_gen = iter_video_frames_to_model_domain(video_path, frame_cap=num_frames, device=device)
        video_full = video_preprocess(video_gen, height, width, dtype, device)  # (1, 3, F, H, W) in [-1,1]
        num_actual_frames = video_full.shape[2]

        # ponytail: pad to 8n+1 by repeating last frame; crop output back after generation
        if video_full.shape[2] < num_frames_vae_padded:
            last_frame = video_full[:, :, -1:, :, :]
            pad_count = num_frames_vae_padded - video_full.shape[2]
            video_full = torch.cat([video_full, last_frame.expand(-1, -1, pad_count, -1, -1)], dim=2)

        # Downscale to half res for stage 1
        video_half = self._resize_video_spatial(
            video_full[:, :, :num_frames_vae_padded, :, :],
            height=half_h,
            width=half_w,
            mode="bilinear",
            align_corners=False,
        )

        # Full res for stage 2
        video_full = video_full[:, :, :num_frames_vae_padded, :, :]

        # Load mask frames, pad if fewer than needed
        mask_gen = decode_video_by_frame(path=mask_path, frame_cap=num_frames_vae_padded, device=device)
        mask_video = video_preprocess(mask_gen, height, width, dtype, device)  # (1, 3, F, H, W) in [-1,1]
        if mask_video.shape[2] < num_frames_vae_padded:
            last_mask = mask_video[:, :, -1:, :, :]
            mask_pad = num_frames_vae_padded - mask_video.shape[2]
            mask_video = torch.cat([mask_video, last_mask.expand(-1, -1, mask_pad, -1, -1)], dim=2)
        mask_video = mask_video[:, :, :num_frames_vae_padded, :, :]
        mask_gray = mask_video.mean(dim=1, keepdim=True)  # (1, 1, F, H, W) grayscale in [-1,1]
        mask_gray = (mask_gray + 1.0) / 2.0  # → [0, 1]
        # ------------------------------------------------------------------ #
        # 3. Dilate masks (node 5382: r=15 stage1, node 5379: r=30 stage2)
        #    Effective runtime radii from linked workflow inputs:
        #    - Stage1: node 5400 PrimitiveInt [15] feeds node 5382
        #    - Stage2: node 5372 ComfyMathExpression (2*a) feeds node 5379
        #    Node widget defaults are ignored while linked.
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Dilating masks")
        # Official: dilate at full res, then downscale for half-res stage 1 uses
        mask_full_gray = mask_gray[0, 0]  # (F, H_full, W_full)
        mask_stage1_full = dilate_video_mask(mask_full_gray.clone(), spatial_radius=stage1_radius, temporal_radius=0)
        mask_stage2_full = dilate_video_mask(mask_full_gray.clone(), spatial_radius=stage2_radius, temporal_radius=0)
        # Downscale stage1 mask to half res for stage 1 (bridge blend + green prep)
        mask_stage1_half = self._resize_video_mask_spatial(mask_stage1_full, half_h, half_w)  # (F, H_half, W_half)

        # ------------------------------------------------------------------ #
        # 4. Create green composites (official #66FF00)
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Creating green composite frames")
        # Official: stage 1 green prep uses stage1 (r=15) mask at half res
        # ponytail: only half-res green used for guide conditioning; full-res
        # green was for official blend but we blend against original video at both stages.
        green_half = green_composite_preprocess(video_half[:, :, :num_frames_vae_padded], mask_stage1_half)

        # ────────────────────────────────────────────────────────────────────── #
        # 5. Green composite guide conditioning — direct tensor encode, no file I/O
        # ────────────────────────────────────────────────────────────────────── #
        # ponytail: encode green_half tensor directly inside image_conditioner using
        # existing encoder; create VideoConditionByReferenceLatent with downscale_factor=1
        # (matching LTXAddVideoICLoRAGuideAdvanced latent_downscale_factor=1 widget) and
        # strength=conditioning_strength. No temp mp4 roundtrip or file decode.



        # ------------------------------------------------------------------ #
        # 6. Stage 1: denoising at half resolution
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Stage 1 denoising (half res, %d x %d)", half_w, half_h)

        # Create conditionings: image (from images param) + video (green composite)
        stage1_ltx_images = [
            make_ltx_image_conditioning_input(img.path, img.frame_idx, img.strength)
            for img in images
        ]

        # Encode green composite as video conditioning — direct tensor path
        tiling_config = default_tiling_config()
        stage1_conditionings = self.pipeline.image_conditioner(
            lambda enc: (
                combined_image_conditionings(
                    images=stage1_ltx_images,
                    height=half_h,
                    width=half_w,
                    video_encoder=enc,
                    dtype=dtype,
                    device=device,
                )
                + self._encode_green_guide_conditioning(
                    enc=enc,
                    tensor=green_half,
                    strength=conditioning_strength,
                )
            )
        )

        # ponytail: offload originals to CPU before stage1 denoising — 196f 1080p originals
        # (~4.7GB @ bf16) + GGUF transformer raw load OOMs on 8/12GB. Back for blend only.
        del green_half, mask_video, mask_gray, mask_stage1_full
        video_half = video_half.cpu()
        video_full = video_full.cpu()
        mask_stage1_half = mask_stage1_half.cpu()
        mask_stage2_full = mask_stage2_full.cpu()
        mask_full_gray = mask_full_gray.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        stage1_sigmas = DISTILLED_SIGMAS.to(dtype=torch.float32, device=device)

        video_state_s1, audio_state_s1 = self.pipeline.stage_1(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage1_sigmas,
            noiser=noiser,
            width=half_w,
            height=half_h,
            frames=num_frames_vae_padded,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage1_conditionings,
            ),
            audio=ModalitySpec(
                context=audio_context,
            ),
            streaming_prefetch_count=denoise_streaming_prefetch_count,
        )

        # ------------------------------------------------------------------ #
        # 7. Decode stage 1 and Laplacian blend at half res
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Decoding stage 1 and blending")
        assert video_state_s1 is not None
        decoded_s1_iter = self.pipeline.video_decoder(video_state_s1.latent, tiling_config, generator)
        decoded_s1_frames = self._collect_frames(decoded_s1_iter)
        # decoded_s1_frames: (F, H_half, W_half, 3) in [0, 1]

        # Move originals back to GPU for stage 1 blend
        video_half = video_half.to(device=device, dtype=dtype)
        mask_stage1_half = mask_stage1_half.to(device=device)

        # Original video as [0, 1] pixel frames for stage 1 blend
        # video_half is (1, 3, F, H, W) in [-1,1] → (F, H, W, 3) in [0, 1]
        video_half_frames = video_half[0].permute(1, 2, 3, 0)  # (F, H, W, 3) in [-1, 1]
        video_half_frames_01 = (video_half_frames + 1.0) / 2.0  # → [0, 1]

        # Bridge blend [5266]: image_a=stage1_decoded, image_b=original video.
        # ponytail: green remains guide conditioning; Laplacian blend preserves original
        # unmasked content and avoids green bleed. Not official parity.
        mask_s1_blend = mask_stage1_half[:decoded_s1_frames.shape[0]]
        blend_stage1 = laplacian_pyramid_blend(
            decoded_s1_frames,
            video_half_frames_01[:decoded_s1_frames.shape[0]],
            mask_s1_blend,
            max_level=7,
            mask_low_res_dilation=INPAINT_BLEND1_LOW_RES_DILATION,
            device=device,
        )  # (F, H_half, W_half, 3) in [0, 1]

        # ------------------------------------------------------------------ #
        # 8. Upscale 2× and VAE encode tiled for stage 2
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Upscaling and re-encoding blend for stage 2")
        # Resize blend to full res via GPU bicubic (replaced CPU OpenCV lanczos)
        # ponytail: explicit device before interpolate, dtype after — GPU path
        blend_stage1_bchw = blend_stage1.permute(0, 3, 1, 2).to(device=device)  # (F, 3, H, W)
        blend_full = F.interpolate(
            blend_stage1_bchw, size=(height, width), mode="bicubic", align_corners=False,
        ).clamp(0.0, 1.0).to(dtype=dtype)

        # VAEEncodeTiled the upscaled blend
        # VideoEncoder expects (B, C, F, H, W) in [-1, 1]
        blend_full_bcfhw = self._frames_chw_to_bcfhw(blend_full) * 2.0 - 1.0  # (1, 3, F, H_full, W_full) in [-1, 1]

        # Trim to VAE-compatible frame count
        blend_vae_frames = _vae_compatible_frame_count(blend_full_bcfhw.shape[2])
        blend_full_bcfhw = blend_full_bcfhw[:, :, :blend_vae_frames]

        # tiling_config from earlier declaration
        encoded_blend = self.pipeline.image_conditioner(
            lambda enc: enc.tiled_encode(blend_full_bcfhw, tiling_config)
        )  # (1, 128, F', H'_full, W'_full)
        # ponytail: free large intermediates before stage 2 denoising — no longer needed
        del blend_stage1, blend_stage1_bchw, blend_full, blend_full_bcfhw
        # ponytail: offload half-res originals back to CPU before stage 2 denoising
        video_half = video_half.cpu()
        mask_stage1_half = mask_stage1_half.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ------------------------------------------------------------------ #
        # 9. Stage 2: denoising at full resolution
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Stage 2 denoising (full res, %d x %d)", width, height)

        # ponytail: [1:] drops 0.909375 → official 3-step [0.725, 0.421875, 0.0]
        stage2_sigmas = STAGE_2_DISTILLED_SIGMAS[1:].to(dtype=torch.float32, device=device)

        # ponytail: scale sigma schedule proportionally so stage2_sigmas_video[0] ≈ 0.55,
        # visually tuned to reduce stage2 structural drift. Promote to UI only if users
        # need per-scene tuning.
        stage2_sigmas_video = stage2_sigmas * (0.55 / stage2_sigmas[0].item())
        stage2_noise_scale = 0.55

        # Recreate conditionings at full res for stage 2
        # ponytail: sampler uses default Euler (SimpleDenoiser/DiffusionStage).
        # Official Comfy workflow uses euler_cfg_pp (stage1) and
        # euler_ancestral_cfg_pp (stage2) via LTXVCFGGuider.
        # Not implemented in installed ltx_pipelines. Add when available.
        # Stage2 is guided by encoded_blend initial latent (VAE-encoded stage1 blend at full res).
        # Do NOT inject the half-res green guide (green_half) here — it is already used in stage1
        # conditionings and re-encoding it for stage2 at full res produces an incorrect
        # half-res embedding in the full frame.
        stage2_conditionings = self.pipeline.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=stage1_ltx_images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=device,
            )
        )

        # Official: stage2 uses seed (not seed+1)
        generator_s2 = torch.Generator(device=device).manual_seed(seed)
        noiser_s2 = GaussianNoiser(generator=generator_s2)

        video_state_s2, audio_state_s2 = self.pipeline.stage_2(
            denoiser=SimpleDenoiser(video_context, audio_context),
            sigmas=stage2_sigmas_video,
            noiser=noiser_s2,
            width=width,
            height=height,
            frames=num_frames_vae_padded,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=stage2_conditionings,
                noise_scale=stage2_noise_scale,
                initial_latent=encoded_blend,
            ),
            audio=ModalitySpec(
                context=audio_context,
                # ponytail: audio noise_scale intentionally left on stage2_sigmas[0].item()
                # to isolate video-side variable for testing.
                noise_scale=stage2_sigmas[0].item(),
                initial_latent=audio_state_s1.latent,  # type: ignore[union-attr]
            ),
            streaming_prefetch_count=denoise_streaming_prefetch_count,
        )

        # ------------------------------------------------------------------ #
        # 10. Decode stage 2 and Laplacian blend at full res
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Decoding stage 2 and final blend")
        assert video_state_s2 is not None
        decoded_s2_iter = self.pipeline.video_decoder(video_state_s2.latent, tiling_config, generator_s2)
        decoded_s2_frames = self._collect_frames(decoded_s2_iter)

        # Move full-res originals back to GPU for final blend
        video_full = video_full.to(device=device, dtype=dtype)
        mask_stage2_full = mask_stage2_full.to(device=device)
        mask_full_gray = mask_full_gray.to(device=device)

        # Original video as [0, 1] pixel frames for stage 2 blend
        # video_full is (1, 3, F, H, W) in [-1,1] → (F, H, W, 3) in [0, 1]
        video_full_frames = video_full[0].permute(1, 2, 3, 0)  # (F, H, W, 3) in [-1, 1]
        video_full_frames_01 = (video_full_frames + 1.0) / 2.0  # → [0, 1]

        # Final blend [5226]: image_a=stage2_decoded, image_b=original video.
        # ponytail: green remains guide conditioning; Laplacian blend preserves original
        # unmasked content and avoids green bleed. Not official parity.
        mask_s2_blend = mask_stage2_full[:decoded_s2_frames.shape[0]]
        blend_stage2 = laplacian_pyramid_blend(
            decoded_s2_frames,
            video_full_frames_01[:decoded_s2_frames.shape[0]],
            mask_s2_blend,
            max_level=7,
            mask_low_res_dilation=laplacian_blend_grow,
            device=device,
        )

        # Final raw-mask guard: clamp anything outside user mask back to original.
        # Blurs the raw mask threshold to feather the final composite edge.
        # ponytail: final guard uses raw user mask blurred for final feather;
        # dilation remains for model context only.
        # laplacian_blend_grow controls Laplacian pyramid dilation only;
        # final_mask_blur_px separately controls the raw-mask edge feather.
        blend_stage2 = self._apply_raw_mask_guard(
            blend_stage2,
            mask_full_gray[:blend_stage2.shape[0]],
            video_full_frames_01[:blend_stage2.shape[0]],
            blur_radius=final_mask_blur_px,
        )

        # ------------------------------------------------------------------ #
        # 11. Encode output video with audio
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Encoding output with audio")
        # Audio context is always set up in inpaint (see prompt encoding above)
        assert audio_state_s2 is not None, "stage 2 audio state is None — audio context may not be set"
        decoded_audio = self.pipeline.audio_decoder(audio_state_s2.latent)
        # ponytail: crop generated frames back to original input count;
        # extra padded frames (repeat-last-frame) discarded to preserve duration
        blend_stage2 = blend_stage2[:num_actual_frames]
        chunks = video_chunks_number(num_actual_frames, tiling_config)
        encode_video_output(
            video=(blend_stage2.clamp(0, 1) * 255).to(torch.uint8),
            audio=decoded_audio,
            fps=int(frame_rate),
            output_path=output_path,
            video_chunks_number_value=chunks,
            output_format=output_format,
            encoder=encoder,
            proxy_path=proxy_path,
            on_progress=on_progress,
            input_colorspace=input_colorspace,
            total_frames=num_actual_frames,
        )
        logger.info("[inpaint] Done — %s", output_path)

    def _encode_video_conditioning(
        self,
        enc: Any,
        video_path: str,
        height: int,
        width: int,
        num_frames: int,
        strength: float,
    ) -> list[Any]:
        """Encode a video file and create a VideoConditionByReferenceLatent item."""
        from ltx_core.conditioning import VideoConditionByReferenceLatent
        from ltx_pipelines.utils.media_io import video_preprocess

        # Sequence files decode + color-transfer inside decode_sequence_frames
        # (via the patched decode_video_by_frame that iter_video_frames_to_model_domain
        # routes to). CM-1c: tagged non-bt709 VIDEO is corrected to Rec.709 here
        # (byte-identical passthrough for bt709/untagged).
        frame_gen = iter_video_frames_to_model_domain(
            video_path, frame_cap=num_frames, device=self.pipeline.device
        )
        video = video_preprocess(frame_gen, height, width, self.pipeline.dtype, self.pipeline.device)
        encoded_video = enc(video)

        return [
            VideoConditionByReferenceLatent(
                latent=encoded_video,
                downscale_factor=self.pipeline.reference_downscale_factor,
                strength=strength,
            )
        ]

    @staticmethod
    def _collect_frames(iterator: Iterator[torch.Tensor]) -> torch.Tensor:
        """Collect all decoded frames from a video decoder iterator into (F, H, W, 3)."""
        chunks: list[torch.Tensor] = []
        for chunk in iterator:
            chunks.append(chunk.cpu())
        return torch.cat(chunks, dim=0)

    @staticmethod
    def _frames_chw_to_bcfhw(frames: torch.Tensor) -> torch.Tensor:
        """Convert (F, C, H, W) frame stack to (1, C, F, H, W) for VAE encode."""
        return frames.permute(1, 0, 2, 3).unsqueeze(0)

    @staticmethod
    def _resize_video_spatial(
        video: torch.Tensor,
        height: int,
        width: int,
        mode: str = "bilinear",
        align_corners: bool | None = None,
    ) -> torch.Tensor:
        """Resize a (B, C, F, H, W) video tensor spatially to (B, C, F, height, width).

        F.interpolate treats 5D as (B, C, D, H, W) — reshape to (B*F, C, H, W)
        for 2D spatial interpolation, then restore frame dimension.
        """
        b, c, f, h, w = video.shape
        # ponytail: reshape to 4D for 2D spatial resize
        flat = video.permute(0, 2, 1, 3, 4).reshape(b * f, c, h, w)  # (B*F, C, H, W)
        if mode in ("bilinear", "bicubic", "trilinear"):
            ac = align_corners if align_corners is not None else False
            resized = F.interpolate(flat, size=(height, width), mode=mode, align_corners=ac)
        else:
            resized = F.interpolate(flat, size=(height, width), mode=mode)
        _, _, rh, rw = resized.shape
        return resized.reshape(b, f, c, rh, rw).permute(0, 2, 1, 3, 4)  # (B, C, F, rh, rw)

    @staticmethod
    def _resize_video_mask_spatial(
        mask: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Resize a (F, H, W) mask spatially to (F, height, width) with nearest neighbor."""
        return F.interpolate(
            mask.unsqueeze(1),  # (F, 1, H, W)
            size=(height, width),
            mode="nearest",
        ).squeeze(1)  # (F, height, width)

    def _encode_green_guide_conditioning(
        self,
        enc: Any,
        tensor: torch.Tensor,
        strength: float,
    ) -> list[Any]:
        """Encode a green composite tensor directly, no file I/O.

        Accepts a (1, 3, F, H, W) tensor already on the correct device, encodes
        via the image_conditioner's encoder, and returns a list with one
        VideoConditionByReferenceLatent using downscale_factor=1 (matching
        LTXAddVideoICLoRAGuideAdvanced latent_downscale_factor=1 widget) and
        the given conditioning strength.

        This replaces the old temp-mp4 roundtrip path.
        """
        from ltx_core.conditioning import VideoConditionByReferenceLatent

        # ponytail: cast to pipeline dtype/device — green_composite_preprocess may promote
        # bfloat16 to float32 (via float32 _bg_tensor), but VAE conv expects matching dtype.
        tensor = tensor.to(dtype=self.pipeline.dtype, device=self.pipeline.device)
        # ponytail: tiled VAE encode avoids 196f 1080p green guide OOM; direct call if encoder lacks tiled API
        if hasattr(enc, "tiled_encode"):
            encoded = enc.tiled_encode(tensor, default_tiling_config())
        else:
            encoded = enc(tensor)
        return [
            VideoConditionByReferenceLatent(
                latent=encoded,
                downscale_factor=1,
                strength=strength,
            )
        ]
