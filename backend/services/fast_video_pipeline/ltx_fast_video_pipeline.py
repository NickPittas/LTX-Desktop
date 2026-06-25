"""LTX fast video pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator
import os
from typing import Final, cast

import torch

from api_types import ImageConditioningInput
from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


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
    ) -> "LTXFastVideoPipeline":
        return LTXFastVideoPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            components=components,
            transformer_format=transformer_format,
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
    ) -> None:
        self._components = components
        from services.patches.gguf_loader_fix import install_gguf_t2v_conditioning_patch

        install_gguf_t2v_conditioning_patch()

        if transformer_format == "gguf":
            from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

            install_gguf_prompt_encoder_patch()

        from ltx_pipelines.distilled import DistilledPipeline

        self._checkpoint_path = checkpoint_path
        self._gemma_root = gemma_root
        self._upsampler_path = upsampler_path
        self._device = device
        self._streaming_prefetch_count = streaming_prefetch_count
        self._transformer_format = transformer_format
        # GGUF is already quantized; the FP8 policy's sd_ops/module_ops would
        # downcast and overwrite the lazy QParam/GgufLinear path, so disable it.
        if transformer_format == "gguf":
            self._quantization = None
        else:
            from ltx_core.quantization import QuantizationPolicy

            self._quantization = QuantizationPolicy.fp8_cast() if device_supports_fp8(device) else None

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            gemma_root=cast(str, gemma_root),
            spatial_upsampler_path=upsampler_path,
            loras=[],
            device=device,
            quantization=self._quantization,
        )
        if transformer_format == "gguf":
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_gguf_loader

            install_gguf_loader(self.pipeline)
            c = self._components
            install_gguf_component_paths(
                self.pipeline,
                checkpoint_path,
                video_vae_path=c.video_vae_path if c is not None else None,
                audio_vae_path=c.audio_vae_path if c is not None else None,
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
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)

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
            )
            chunks = video_chunks_number(warmup_frames, tiling_config)
            encode_video_output(video=video, audio=audio, fps=8, output_path=output_path, video_chunks_number_value=chunks)
        finally:
            if os.path.exists(output_path):
                os.unlink(output_path)

    def compile_transformer(self) -> None:
        if self._transformer_format == "gguf":
            # GGUF transformer compile is not supported yet; lazy dequant uses
            # runtime GGUF dequantization (numpy-based, untracable by torch.compile).
            raise RuntimeError(
                "GGUF transformer compile is not supported yet; lazy dequant uses"
                " runtime GGUF dequantization"
            )
        from ltx_pipelines.distilled import DistilledPipeline

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=self._checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            gemma_root=cast(str, self._gemma_root),
            spatial_upsampler_path=self._upsampler_path,
            loras=[],
            device=self._device,
            quantization=self._quantization,
            torch_compile=True,
        )
        if self._transformer_format == "gguf":
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_gguf_loader

            install_gguf_loader(self.pipeline)
            c = self._components
            install_gguf_component_paths(
                self.pipeline,
                self._checkpoint_path,
                video_vae_path=c.video_vae_path if c is not None else None,
                audio_vae_path=c.audio_vae_path if c is not None else None,
            )
