"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any, cast

import torch
import torch.nn.functional as F

from api_types import ImageConditioningInput
from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

_fp8_lora_fuse_patched = False

# ponytail: mask_grow_px controls LTXVDilateVideoMask radii only (derive_stage_radii).
# Blend low-res dilation constants are separate controls — NOT related to mask dilation radii.
# INPAINT_BLEND1_LOW_RES_DILATION=5 for bridge blend (stage1, node 5266, linked input).
# INPAINT_BLEND2_LOW_RES_DILATION=6 for final blend (stage2, node 5226, linked input).
INPAINT_BLEND1_LOW_RES_DILATION = 5
INPAINT_BLEND2_LOW_RES_DILATION = 6


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
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_paths=lora_paths,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            components=components,
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

        self._streaming_prefetch_count = streaming_prefetch_count
        lora_entries = [
            LoraPathStrengthAndSDOps(path=lp, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
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
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        import ltx_pipelines.ic_lora as ic_lora_module
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

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

        return self.pipeline(
                prompt=prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=[_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images],
                video_conditioning=video_conditioning,
                tiling_config=tiling_config,
                streaming_prefetch_count=self._streaming_prefetch_count,
                conditioning_attention_mask=mask,
                conditioning_attention_strength=conditioning_strength,
            )

    @staticmethod
    def _composite_in_outpainting(
        output_path: str,
        original_video_path: str,
        mask_path: str,
    ) -> None:
        """Blend generated output with original video using mask video.

        White mask (255) = keep generated region, black mask (0) = preserve original.
        Uses grayscale alpha compositing: result = gen * (mask/255) + orig * (1 - mask/255).
        """
        import cv2
        import numpy as np

        out_cap = cv2.VideoCapture(output_path)
        orig_cap = cv2.VideoCapture(original_video_path)
        mask_cap = cv2.VideoCapture(mask_path)

        out_fps = out_cap.get(cv2.CAP_PROP_FPS)
        out_w = int(out_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        out_h = int(out_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_count = int(out_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore[attr-defined]
        tmp_path = output_path + ".composite_tmp.mp4"
        writer = cv2.VideoWriter(tmp_path, fourcc, out_fps, (out_w, out_h))  # type: ignore[arg-type]

        for _ in range(out_count):
            ret_out, out_frame = out_cap.read()
            if not ret_out:
                break

            ret_orig, orig_frame = orig_cap.read()
            if not ret_orig:
                orig_frame = out_frame.copy()

            ret_mask, mask_frame = mask_cap.read()
            if not ret_mask:
                mask_frame = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            if orig_frame.shape[:2] != (out_h, out_w):
                orig_frame = cv2.resize(orig_frame, (out_w, out_h))
            if mask_frame.shape[:2] != (out_h, out_w):
                mask_frame = cv2.resize(mask_frame, (out_w, out_h))

            mask_gray = (
                cv2.cvtColor(mask_frame, cv2.COLOR_BGR2GRAY)
                if mask_frame.ndim == 3 and mask_frame.shape[2] >= 3
                else mask_frame.reshape(out_h, out_w)
            )
            mask_norm = mask_gray.astype(np.float32) / 255.0
            mask_3ch = np.stack([mask_norm] * 3, axis=2)

            composite = (
                out_frame.astype(np.float32) * mask_3ch
                + orig_frame.astype(np.float32) * (1.0 - mask_3ch)
            )
            writer.write(np.clip(composite, 0, 255).astype(np.uint8))

        out_cap.release()
        orig_cap.release()
        mask_cap.release()
        writer.release()

        import os
        os.replace(tmp_path, output_path)

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
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
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
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

        if original_video_path is not None and mask_path is not None:
            self._composite_in_outpainting(output_path, original_video_path, mask_path)

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
        laplacian_blend_grow: int = 6,
    ) -> None:
        """Official two-stage IC-LoRA inpaint pipeline.

        White mask = inpaint region, black mask = keep original.
        Uses green composite (#66FF00) conditioning, Laplacian pyramid
        inter-stage blending, and two denoising passes at half then full res.
        """
        import logging

        import cv2
        import numpy as np
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput
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
        device = self.pipeline.device
        dtype = torch.bfloat16  # ponytail: matches ICLoraPipeline.dtype

        # ------------------------------------------------------------------ #
        # 1. Load input video and mask
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Loading video and mask")

        # Load video frames at half res (for stage 1 conditioning) and full res
        video_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=device)
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
        # 2. Dilate masks (node 5382: r=15 stage1, node 5379: r=30 stage2)
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
        # 3. Create green composites (official #66FF00)
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Creating green composite frames")
        # Official: stage 1 green prep uses stage1 (r=15) mask at half res
        # ponytail: only half-res green used for guide conditioning; full-res
        # green was for official blend but we blend against original video at both stages.
        green_half = green_composite_preprocess(video_half[:, :, :num_frames_vae_padded], mask_stage1_half)

        # ────────────────────────────────────────────────────────────────────── #
        # 4. Green composite guide conditioning — direct tensor encode, no file I/O
        # ────────────────────────────────────────────────────────────────────── #
        # ponytail: encode green_half tensor directly inside image_conditioner using
        # existing encoder; create VideoConditionByReferenceLatent with downscale_factor=1
        # (matching LTXAddVideoICLoRAGuideAdvanced latent_downscale_factor=1 widget) and
        # strength=conditioning_strength. No temp mp4 roundtrip or file decode.

        # ------------------------------------------------------------------ #
        # 5. Encode prompt and create contexts
        # ------------------------------------------------------------------ #
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
        # 6. Stage 1: denoising at half resolution
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Stage 1 denoising (half res, %d x %d)", half_w, half_h)

        # Create conditionings: image (from images param) + video (green composite)
        stage1_ltx_images = [_LtxImageInput(img.path, img.frame_idx, img.strength) for img in images]

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
            streaming_prefetch_count=self._streaming_prefetch_count,
        )

        # ------------------------------------------------------------------ #
        # 7. Decode stage 1 and Laplacian blend at half res
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Decoding stage 1 and blending")
        assert video_state_s1 is not None
        decoded_s1_iter = self.pipeline.video_decoder(video_state_s1.latent, tiling_config, generator)
        decoded_s1_frames = self._collect_frames(decoded_s1_iter)
        # decoded_s1_frames: (F, H_half, W_half, 3) in [0, 1]

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
        # Resize blend to full res (official: lanczos inter-stage upsample)
        blend_stage1_bchw = blend_stage1.permute(0, 3, 1, 2)  # (F, 3, H, W)
        blend_np = blend_stage1_bchw.cpu().float().numpy()
        frames_up: list[np.ndarray] = []
        for i in range(blend_np.shape[0]):
            frame_hwc = blend_np[i].transpose(1, 2, 0)  # (3, H, W) -> (H, W, 3)
            resized = cv2.resize(frame_hwc, (width, height), interpolation=cv2.INTER_LANCZOS4)  # type: ignore
            frames_up.append(resized.transpose(2, 0, 1))  # type: ignore
        blend_full = torch.from_numpy(np.stack(frames_up))  # (F, 3, H_full, W_full)  # type: ignore[arg-type]
        blend_full = blend_full.clamp(0, 1)  # safety

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
        stage2_conditionings = self.pipeline.image_conditioner(
            lambda enc: (
                combined_image_conditionings(
                    images=stage1_ltx_images,
                    height=height,
                    width=width,
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
        # ponytail: official node 5114 uses same half-res green guide (node 5378, mask r=15)
        # for both stages. Stage2 passes through LTXVCropGuides (temporal keyframe-crop) in official
        # workflow; installed clear_conditioning() already handles it.

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
            streaming_prefetch_count=self._streaming_prefetch_count,
        )

        # ------------------------------------------------------------------ #
        # 10. Decode stage 2 and Laplacian blend at full res
        # ------------------------------------------------------------------ #
        logger.info("[inpaint] Decoding stage 2 and final blend")
        assert video_state_s2 is not None
        decoded_s2_iter = self.pipeline.video_decoder(video_state_s2.latent, tiling_config, generator_s2)
        decoded_s2_frames = self._collect_frames(decoded_s2_iter)

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
        from ltx_pipelines.utils.media_io import decode_video_by_frame, video_preprocess

        frame_gen = decode_video_by_frame(path=video_path, frame_cap=num_frames, device=self.pipeline.device)
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
        encoded = enc(tensor)
        return [
            VideoConditionByReferenceLatent(
                latent=encoded,
                downscale_factor=1,
                strength=strength,
            )
        ]
