"""HDR IC-LoRA pipeline protocol definitions.

Dedicated Protocol for the HDR IC-LoRA workflow: an 8-bit source video (or
image sequence) is VAE-encoded as the IC-LoRA reference, the official LTX-2
two-stage HDR flow runs (Stage 1 half-res → Stage 2 full-res), the HDR decode
postprocess is applied, and the output is encoded as linear EXR with an SDR
Reinhard-tonemapped proxy. This is distinct from the generic
``services.ic_lora_pipeline.IcLoraPipeline`` path, which is T2V/I2V-oriented
and not source-video-driven for HDR.

The HDR workflow is video-only: audio is never conditioned or produced.
Scene-embeddings replace text prompt encoding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal, Protocol

from api_types import OutputFormat

if TYPE_CHECKING:
    from collections.abc import Callable

    from ltx_pipelines.utils.types import OffloadMode
    from services.color_management import ColorSpace
    from services.ltx_components import BaseFamily, ResolvedLtxComponents, TransformerFormat
    from services.media_encoder.media_encoder import MediaEncoder

    import torch


class HdrIcLoraPipeline(Protocol):
    """HDR IC-LoRA two-stage pipeline Protocol.

    ``pipeline_kind`` discriminates HDR pipeline state from other GPU slot
    pipeline types (``"hdr_ic_lora"``).
    """

    pipeline_kind: ClassVar[Literal["hdr_ic_lora"]]

    @staticmethod
    def create(
        checkpoint_path: str | tuple[str, ...],
        upsampler_path: str,
        hdr_lora_path: str,
        device: str | torch.device,
        components: ResolvedLtxComponents | None = None,
        transformer_format: TransformerFormat = "safetensors",
        base_family: BaseFamily = "distilled",
        distilled_lora_path: str | None = None,
        scene_embeddings_path: str | None = None,
        offload_mode: OffloadMode | None = None,
        *,
        gemma_root: str | None = None,
    ) -> "HdrIcLoraPipeline":
        """Construct a dedicated HDR IC-LoRA two-stage pipeline.

        Phase 2 restricts initial HDR support to the **official distilled
        safetensors** base checkpoint (single file); the wrapper fails closed
        for tuple checkpoints, non-safetensors formats, or non-distilled base
        families. ``scene_embeddings_path`` is required and mapped to upstream
        ``text_embeddings_path`` (never substituted with prompt/text
        embeddings); it is defaulted here so the current handler typechecks,
        and Phase 3 forwards it explicitly. ``offload_mode`` defaults to
        ``OffloadMode.NONE`` (never ``OffloadMode.DISK``). ``components``,
        ``distilled_lora_path`` and ``gemma_root`` are accepted for Protocol
        compatibility but are not used for HDR generation math.
        """
        ...

    def generate(
        self,
        source_video_path: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        output_path: str,
        output_format: OutputFormat = OutputFormat.EXR_ZIP_HALF,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        input_colorspace: ColorSpace | None = None,
        on_progress: Callable[[float], None] | None = None,
    ) -> None:
        """Run the official HDR IC-LoRA two-stage flow on the source video.

        Threads ``video_conditioning=[(source_video_path, 1.0)]`` through the
        official two-stage path, applies HDR decode postprocess, and encodes
        the result via ``encode_video_output`` with
        ``HdrProxyPolicy.SDR_TONEMAP_REINHARD``. HDR is video-only: no audio
        is conditioned or produced.
        """
        ...
