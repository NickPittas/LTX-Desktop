"""IC-LoRA pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import ImageConditioningInput, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from ltx_pipelines.utils.types import OffloadMode
    from services.ltx_components import CheckpointPath, ResolvedLtxComponents
    from services.local_memory_plan import LocalMemoryPlan
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace
    import torch


class IcLoraPipeline(Protocol):
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
        # Phase 2: memory plan owns the offload/quantization decision. None is a
        # narrow compatibility default until the handler slice passes explicit plans.
        memory_plan: LocalMemoryPlan | None = None,
    ) -> "IcLoraPipeline":
        ...

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
        ...

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
    ) -> str | None:
        ...
