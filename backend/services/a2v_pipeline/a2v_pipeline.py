"""A2V (Audio-to-Video) pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import ImageConditioningInput, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from services.ltx_components import CheckpointPath, ResolvedLtxComponents
    from services.media_encoder.media_encoder import MediaEncoder
    import torch


class A2VPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
    ) -> "A2VPipeline": ...

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
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> None: ...
