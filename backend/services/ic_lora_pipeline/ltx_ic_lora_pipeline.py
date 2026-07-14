"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, cast

import torch

from api_types import ImageConditioningInput, OutputFormat
from services.block_offload import attach_memory_plan_to_stages
from services.exr_input import iter_video_frames_to_model_domain
from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import (
    default_tiling_config,
    encode_video_output,
    make_ltx_image_conditioning_input,
    video_chunks_number,
)
from services.media_encoder.media_encoder import HdrProxyPolicy
from services import memory_trace
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from ltx_pipelines.utils.types import OffloadMode
    from services.local_memory_plan import LocalMemoryPlan
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace

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


def _vae_compatible_frame_count(num_frames: int) -> int:
    """Max 1+8*k frame count <= num_frames for VAE latent compatibility."""
    return 1 + 8 * max(0, (num_frames - 1) // 8)


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
        offload_mode: OffloadMode,
        components: ResolvedLtxComponents | None = None,
        lora_strength: float = 1.0,
        *,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_paths=lora_paths,
            device=device,
            offload_mode=offload_mode,
            components=components,
            lora_strength=lora_strength,
            memory_plan=memory_plan,
        )

    def __init__(
        self,
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        lora_paths: list[str],
        device: torch.device,
        offload_mode: OffloadMode,
        components: ResolvedLtxComponents | None = None,
        lora_strength: float = 1.0,
        *,
        memory_plan: LocalMemoryPlan | None = None,
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
            from ltx_core.quantization.fp8_cast import build_policy

            quantization = build_policy(checkpoint_path) if device_supports_fp8(device) else None  # type: ignore[arg-type]  # non-split branch → str checkpoint

        # Phase 2: trust the memory plan's offload decision when provided; no
        # internal split/GGUF coercion. Transitional handler path
        # (memory_plan=None) uses the caller's offload_mode as-is.
        if memory_plan is not None:
            offload_mode = memory_plan.offload_mode
        self._offload_mode = offload_mode
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
            offload_mode=self._offload_mode,
        )
        stage_2 = cast(Any, self.pipeline.stage_2)
        stage_2._transformer_builder = stage_2._transformer_builder.with_loras(tuple(lora_entries))
        if hasattr(stage_2, "_streaming_builder"):
            stage_2._streaming_builder = stage_2._streaming_builder.with_loras(tuple(lora_entries))
        # Phase 3B: stamp the memory plan onto the upstream pipeline's
        # DiffusionStage(s) (stage_1/stage_2) so the block-offload build patch
        # can read it when the transformer is built lazily at generation time.
        if memory_plan is not None:
            attach_memory_plan_to_stages(self.pipeline, memory_plan)

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
            install_gguf_component_paths(
                self.pipeline,
                checkpoint_path,
                video_vae_path=c.video_vae_path,
                audio_vae_path=c.audio_vae_path,
            )

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
                with memory_trace.phase("ic_lora_denoise_decode"):
                    return self.pipeline(**inference_kwargs)
        else:
            with memory_trace.phase("ic_lora_denoise_decode"):
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
        on_phase_update: Callable[[str, str | None], None] | None = None,
    ) -> None:
        tiling_config = default_tiling_config()
        if on_phase_update is not None:
            on_phase_update("inference", "Sampling / VAE decode via upstream IC-LoRA pipeline")
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

    @torch.inference_mode()
    def generate_inpaint(
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
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
        on_phase_update: Callable[[str, str | None], None] | None = None,
        save_stage_1_preview: bool = False,
        inpaint_context_window_px: int | None = None,
        inpaint_context_overlap_px: int | None = None,
    ) -> str | None:
        """Run the canonical latent-bridge inpaint pipeline."""
        from .latent_inpaint import generate_inpaint

        return generate_inpaint(
            self, prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate, images=images,
            video_path=video_path, mask_path=mask_path, output_path=output_path,
            conditioning_strength=conditioning_strength, mask_grow_px=mask_grow_px,
            output_format=output_format, encoder=encoder, proxy_path=proxy_path,
            on_progress=on_progress, input_colorspace=input_colorspace,
            on_phase_update=on_phase_update, save_stage_1_preview=save_stage_1_preview,
            inpaint_context_window_px=inpaint_context_window_px,
            inpaint_context_overlap_px=inpaint_context_overlap_px,
        )

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
