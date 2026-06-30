"""Fast video pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import ImageConditioningInput, OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from ltx_pipelines.utils.types import OffloadMode
    from services.ltx_components import CheckpointPath, ResolvedLtxComponents
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace
    import torch


class FastVideoPipeline(Protocol):
    pipeline_kind: ClassVar[Literal["fast"]]

    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        upsampler_path: str,
        device: torch.device,
        offload_mode: OffloadMode,
        components: ResolvedLtxComponents | None = None,
        *,
        transformer_format: str = "safetensors",
        distilled_lora_path: str | None = None,
    ) -> "FastVideoPipeline":
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
        output_path: str,
        enhance_prompt: bool = False,
        negative_prompt: str = "",
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
    ) -> None:
        ...

    def warmup(self, output_path: str) -> None:
        ...

    def compile_transformer(self) -> None:
        ...

    def supports_torch_compile(self) -> bool:
        ...
