"""LTX IC-LoRA pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import torch

from api_types import ImageConditioningInput
from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class LTXIcLoraPipeline:
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
    ) -> "LTXIcLoraPipeline":
        return LTXIcLoraPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            lora_path=lora_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            components=components,
        )

    def __init__(
        self,
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        lora_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
    ) -> None:
        self._components = components
        from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
        from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
        from ltx_pipelines.ic_lora import ICLoraPipeline

        is_gguf = components is not None and components.transformer_format == "gguf"

        if is_gguf:
            from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

            install_gguf_prompt_encoder_patch()
            quantization = None
        else:
            from ltx_core.quantization import QuantizationPolicy

            quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None

        self._streaming_prefetch_count = streaming_prefetch_count
        lora_entry = LoraPathStrengthAndSDOps(path=lora_path, strength=1.0, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)
        self.pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            spatial_upsampler_path=upsampler_path,
            gemma_root=cast(str, gemma_root),
            loras=[lora_entry],
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
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        from ltx_pipelines.utils.args import ImageConditioningInput as _LtxImageInput

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
        video_conditioning: list[tuple[str, float]],
        output_path: str,
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
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
