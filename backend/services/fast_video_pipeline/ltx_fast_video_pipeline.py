"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Callable, Iterator
import logging
import os
from typing import TYPE_CHECKING, Final

import torch

from api_types import ImageConditioningInput, OutputFormat
from services.ltx_components import BaseFamily, CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import (
    default_tiling_config,
    encode_video_output,
    make_ltx_image_conditioning_input,
    video_chunks_number,
)
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8

if TYPE_CHECKING:
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace

logger = logging.getLogger(__name__)


class LTXFastVideoPipeline:
    pipeline_kind: Final = "fast"

    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
        *,
        transformer_format: str = "safetensors",
        distilled_lora_path: str | None = None,
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            components=components,
            transformer_format=transformer_format,
            distilled_lora_path=distilled_lora_path,
        )

    def __init__(
        self,
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
        *,
        transformer_format: str = "safetensors",
        distilled_lora_path: str | None = None,
    ) -> None:
        self._components = components
        from services.patches.gguf_loader_fix import install_gguf_t2v_conditioning_patch

        install_gguf_t2v_conditioning_patch()

        if transformer_format == "gguf" or gemma_root is not None:
            from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

            install_gguf_prompt_encoder_patch()

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._device = device
        self._transformer_format = transformer_format
        # Phase 3D (plan §12): base family routes the upstream pipeline class.
        # Distilled => ``DistilledPipeline`` (existing fast path).
        # Dev => ``TI2VidTwoStagesPipeline`` with distilled LoRA + upstream
        # ``LTX_2_3_PARAMS`` guider/step params (CFG via negative_prompt).
        self._base_family: BaseFamily = (
            components.base_family if components is not None else "distilled"
        )
        self._distilled_lora_path = distilled_lora_path
        # GGUF is already quantized; the FP8 policy's sd_ops/module_ops would
        # downcast and overwrite the lazy QParam/GgufLinear path, so disable it.
        is_split = (
            transformer_format == "safetensors"
            and self._components is not None
            and self._components.video_vae_path is not None
        )
        # ponytail: split safetensors 22B does not fit full residency on 32GB;
        # stream 2 layers at a time unless explicit mode set.
        if is_split and streaming_prefetch_count is None:
            streaming_prefetch_count = 2
        self._streaming_prefetch_count = streaming_prefetch_count
        if transformer_format == "gguf":
            self._quantization = None
        elif is_split and device_supports_fp8(device):
            from services.patches.gguf_loader_fix import kijai_fp8_quantization_policy

            self._quantization = kijai_fp8_quantization_policy()
        else:
            from ltx_core.quantization import QuantizationPolicy

            self._quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None

        self.pipeline = self._build_upstream_pipeline()
        self._install_post_build_patches(is_split=is_split, transformer_format=transformer_format)

    def _build_upstream_pipeline(self) -> object:
        """Construct the upstream pipeline based on ``base_family``.

        - distilled (default for back-compat when no components): DistilledPipeline
        - dev: TI2VidTwoStagesPipeline with distilled LoRA + LTX_2_3_PARAMS
        """
        if self._base_family == "dev":
            return self._build_dev_pipeline()
        return self._build_distilled_pipeline()

    def _build_distilled_pipeline(self, *, torch_compile: bool = False) -> object:
        from ltx_pipelines.distilled import DistilledPipeline

        return DistilledPipeline(
            distilled_checkpoint_path=self._checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            gemma_root=self._gemma_root or "",
            spatial_upsampler_path=self._upsampler_path,
            loras=[],
            device=self._device,
            quantization=self._quantization,
            torch_compile=torch_compile,
        )

    def _build_dev_pipeline(self) -> object:
        """Construct ``TI2VidTwoStagesPipeline`` with distilled LoRA.

        ``self._distilled_lora_path`` is required (handler validates before
        calling create). We build the distilled LoRA entry with the upstream
        comfy renaming map and default strength.
        """
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.constants import DEFAULT_LORA_STRENGTH

        distilled_lora_path = self._distilled_lora_path
        # Defensive: handler validates before construction; if a dev route
        # reaches here without a LoRA path, raise a clear error instead of
        # silently building a LoRA-less dev pipeline.
        if not distilled_lora_path:
            raise RuntimeError(
                "Dev base family requires a distilled LoRA path; "
                "handler must resolve one before constructing the dev pipeline."
            )

        distilled_lora = [
            LoraPathStrengthAndSDOps(
                distilled_lora_path,
                DEFAULT_LORA_STRENGTH,
                LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]
        return TI2VidTwoStagesPipeline(
            checkpoint_path=self._checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            distilled_lora=distilled_lora,
            spatial_upsampler_path=self._upsampler_path,
            gemma_root=self._gemma_root or "",
            loras=[],
            device=self._device,
            quantization=self._quantization,
        )

    def _install_post_build_patches(self, *, is_split: bool, transformer_format: str) -> None:
        """Apply GGUF / split-safetensors patches after upstream construction.

        Both route types (dev and distilled) may need the GGUF component-path
        remap and the Kijai split transformer-config patch.
        """
        if transformer_format == "gguf":
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_gguf_loader

            install_gguf_loader(self.pipeline)
            c = self._components
            install_gguf_component_paths(
                self.pipeline,
                self._checkpoint_path,
                video_vae_path=c.video_vae_path if c is not None else None,
                audio_vae_path=c.audio_vae_path if c is not None else None,
                mmproj_path=c.mmproj_path if c is not None else None,
            )

        # ponytail: split safetensors also needs VAE path remap with Kijai key filters.
        # install_gguf_component_paths is format-agnostic.
        if is_split:
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_kijai_transformer_config_patch

            c = self._components
            assert c is not None  # guarded above
            install_kijai_transformer_config_patch(self.pipeline, self._checkpoint_path)
            install_gguf_component_paths(
                self.pipeline,
                self._checkpoint_path,
                video_vae_path=c.video_vae_path,
                audio_vae_path=c.audio_vae_path,
                mmproj_path=c.mmproj_path,
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
        tiling_config: TilingConfigType,
        enhance_prompt: bool,
        negative_prompt: str = "",
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        ltx_images = [
            make_ltx_image_conditioning_input(img.path, img.frame_idx, img.strength)
            for img in images
        ]
        if self._base_family == "dev":
            # Dev route: TI2VidTwoStagesPipeline — needs negative_prompt (CFG),
            # explicit num_inference_steps, guider params, and CRF=18 image
            # conditioning (already applied via make_ltx_image_conditioning_input).
            from ltx_pipelines.utils.constants import LTX_2_3_PARAMS

            return self.pipeline(  # type: ignore[no-any-return]
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                num_inference_steps=LTX_2_3_PARAMS.num_inference_steps,
                video_guider_params=LTX_2_3_PARAMS.video_guider_params,
                audio_guider_params=LTX_2_3_PARAMS.audio_guider_params,
                images=ltx_images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
                streaming_prefetch_count=self._streaming_prefetch_count,
            )
        # Distilled route: existing call shape.
        return self.pipeline(  # type: ignore[no-any-return]
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=ltx_images,
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            streaming_prefetch_count=self._streaming_prefetch_count,
        )

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
        output_path: str,
        enhance_prompt: bool = False,
        negative_prompt: str = "",
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
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
            tiling_config=tiling_config,
            enhance_prompt=enhance_prompt,
            negative_prompt=negative_prompt,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(
            video=video, audio=audio, fps=int(frame_rate), output_path=output_path,
            video_chunks_number_value=chunks, output_format=output_format,
            encoder=encoder, proxy_path=proxy_path,
            on_progress=on_progress,
            input_colorspace=input_colorspace, total_frames=num_frames,
        )

    @torch.inference_mode()
    def warmup(self, output_path: str) -> None:
        warmup_frames = 9
        tiling_config = default_tiling_config()

        try:
            video, audio = self._run_inference(
                prompt="test warmup",
                seed=42,
                height=256,
                width=384,
                num_frames=warmup_frames,
                frame_rate=8,
                images=[],
                tiling_config=tiling_config,
                enhance_prompt=False,
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        if not self.supports_torch_compile():
            # GGUF (untraceable dequant) and dev route (TI2VidTwoStagesPipeline
            # does not accept torch_compile in the dev constructor) are both
            # skipped at the handler via supports_torch_compile(); reaching here
            # means a caller ignored the gate. Raise instead of silently
            # rebuilding the dev route as a DistilledPipeline.
            raise RuntimeError(
                f"torch.compile is not supported for base_family={self._base_family!r}"
                f" transformer_format={self._transformer_format!r}"
            )
        # Distilled route is the only compile-capable route; rebuild with
        # torch_compile=True using the existing upstream pipeline class.
        self.pipeline = self._build_distilled_pipeline(torch_compile=True)
        self._install_post_build_patches(
            is_split=(
                self._transformer_format == "safetensors"
                and self._components is not None
                and self._components.video_vae_path is not None
            ),
            transformer_format=self._transformer_format,
        )

    def supports_torch_compile(self) -> bool:
        """Whether torch.compile is supported for the active route/format.

        Two skips (oracle strategy, Phase 3D):
        - GGUF transformers use lazy per-forward dequantization
          (numpy/GGUF-based) which is untracable by ``torch.compile``.
        - Dev route (``TI2VidTwoStagesPipeline``) is not torch.compile-enabled
          in the initial wiring; skip silently like GGUF.
        Callers should check this before invoking :meth:`compile_transformer`
        to skip silently (info log) rather than relying on the ``RuntimeError``
        guard inside compile.
        """
        if self._transformer_format == "gguf":
            return False
        if self._base_family == "dev":
            return False
        return True
