"""IC-LoRA pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import ImageConditioningInput

if TYPE_CHECKING:
    from services.ltx_components import CheckpointPath, ResolvedLtxComponents
    import torch


class IcLoraPipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        lora_paths: list[str],
        device: torch.device,
        streaming_prefetch_count: int | None,
        components: ResolvedLtxComponents | None = None,
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
        laplacian_blend_grow: int = 12,
        final_mask_blur_px: int = 6,
    ) -> None:
        ...
