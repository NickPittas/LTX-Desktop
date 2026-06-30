"""Retake pipeline protocol definitions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from api_types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from ltx_pipelines.utils.types import OffloadMode
    from services.ltx_components import CheckpointPath, ResolvedLtxComponents
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace
    import torch
    from ltx_core.components.guiders import MultiModalGuiderParams
    from ltx_core.loader import LoraPathStrengthAndSDOps
    from ltx_core.quantization import QuantizationPolicy


class RetakePipeline(Protocol):
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        device: "torch.device",
        offload_mode: OffloadMode,
        components: ResolvedLtxComponents | None = None,
        *,
        loras: list["LoraPathStrengthAndSDOps"] | None = None,
        quantization: "QuantizationPolicy | None" = None,
    ) -> "RetakePipeline": ...

    def generate(
        self,
        *,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        output_path: str,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: "MultiModalGuiderParams | None" = None,
        audio_guider_params: "MultiModalGuiderParams | None" = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
    ) -> None: ...
