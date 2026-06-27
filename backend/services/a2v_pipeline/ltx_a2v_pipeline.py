"""LTX A2V (Audio-to-Video) pipeline wrapper."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from api_types import ImageConditioningInput
from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import default_tiling_config, encode_video_output, video_chunks_number
from services.services_utils import AudioOrNone, TilingConfigType, device_supports_fp8


class LTXa2vPipeline:
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
    ) -> "LTXa2vPipeline":
        return LTXa2vPipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            upsampler_path=upsampler_path,
            device=device,
            streaming_prefetch_count=streaming_prefetch_count,
            components=components,
        )

    def __init__(
        self,
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
    ) -> None:
        self._components = components
        from services.a2v_pipeline.distilled_a2v_pipeline import DistilledA2VPipeline

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
        self.pipeline = DistilledA2VPipeline(
            distilled_checkpoint_path=checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            gemma_root=gemma_root or "",
            spatial_upsampler_path=upsampler_path,
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
            install_gguf_component_paths(
                self.pipeline,
                checkpoint_path,
                video_vae_path=c.video_vae_path,
                audio_vae_path=c.audio_vae_path,
            )

    def _run_inference(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        tiling_config: TilingConfigType,
    ) -> tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]:
        return self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[(img.path, img.frame_idx, img.strength) for img in images],
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
            streaming_prefetch_count=self._streaming_prefetch_count,
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        images: list[ImageConditioningInput],
        audio_path: str,
        audio_start_time: float,
        audio_max_duration: float | None,
        output_path: str,
    ) -> None:
        tiling_config = default_tiling_config()
        video, audio = self._run_inference(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            images=images,
            audio_path=audio_path,
            audio_start_time=audio_start_time,
            audio_max_duration=audio_max_duration,
            tiling_config=tiling_config,
        )
        chunks = video_chunks_number(num_frames, tiling_config)
        encode_video_output(video=video, audio=audio, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=chunks)
