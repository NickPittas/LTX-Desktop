"""LTX retake pipeline wrapper.

Forked orchestration of the retake pipeline flow from ``ltx_pipelines.retake``
with the following adjustments:

* ``@torch.no_grad()`` instead of ``@torch.inference_mode()`` — the
  transformer checkpoint uses custom autograd functions incompatible with
  inference-mode tensors.
* Tiled source-video encoding via a local helper that decodes through
  ``iter_video_frames_to_model_domain`` (tagged non-Rec.709 → Rec.709 model
  domain) then ``tiled_encode`` — the vendored ``video_latent_from_file``
  encodes all frames in a single pass (OOMs) and skips colorspace transfer.
* Tiled video decoding via ``VideoDecoder(..., tiling_config)`` — the
  original omits the tiling argument.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, Any
import torch

from api_types import OutputFormat
from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import get_videostream_metadata
from ltx_pipelines.utils.types import OffloadMode

from services.ltx_components import CheckpointPath, ResolvedLtxComponents
from services.ltx_pipeline_common import encode_video_output
from services.retake_pipeline.retake_pipeline import RetakePipeline

if TYPE_CHECKING:
    from services.media_encoder.media_encoder import MediaEncoder
    from services.color_management import ColorSpace
    from services.local_memory_plan import LocalMemoryPlan



def _encode_source_video_latent_model_domain(
    video_encoder: Any,
    file_path: str,
    output_shape: Any,
    device: torch.device,
    dtype: torch.dtype,
    tiling_config: TilingConfig | None = None,
) -> torch.Tensor:
    """Encode a source video to a VAE latent, transferring tagged non-Rec.709
    video into the Rec.709 model domain before the VAE.

    Mirrors ``ltx_pipelines.utils.helpers.video_latent_from_file`` but decodes
    via ``services.exr_input.iter_video_frames_to_model_domain`` so a tagged
    non-bt709 source (BT.601 / Rec.2020 / ...) is colour-transferred to Rec.709
    before ``video_preprocess`` + ``tiled_encode``. Untagged/bt709 video is an
    exact passthrough (byte-identical to the vendored helper). FPS is validated
    against ``output_shape.fps`` exactly as the vendored helper does.
    """
    from ltx_core.types import VideoLatentShape
    from ltx_pipelines.utils.media_io import get_videostream_fps, video_preprocess
    from services.exr_input import iter_video_frames_to_model_domain

    fps = get_videostream_fps(file_path)
    if fps != output_shape.fps:
        raise ValueError(
            f"Input video FPS {fps} does not match output FPS {output_shape.fps}, not supported"
        )
    frame_gen = iter_video_frames_to_model_domain(
        file_path, frame_cap=output_shape.frames, device=device,
    )
    frames = video_preprocess(frame_gen, output_shape.height, output_shape.width, dtype, device)
    latents = video_encoder.tiled_encode(frames, tiling_config or TilingConfig.default())
    # Conform latent length to the required VAE frame count (trim or zero-pad on
    # dim 2), matching the vendored video_latent_from_file helper's behavior
    # (inlined here to avoid importing a private vendored symbol).
    required_latent_frames = VideoLatentShape.from_pixel_shape(output_shape).frames
    actual_frames = latents.shape[2]
    if actual_frames > required_latent_frames:
        return latents[:, :, :required_latent_frames]
    if actual_frames < required_latent_frames:
        pad_shape = list(latents.shape)
        pad_shape[2] = required_latent_frames - actual_frames
        latents = torch.cat(
            [latents, torch.zeros(pad_shape, device=latents.device, dtype=latents.dtype)],
            dim=2,
        )
    return latents


class LTXRetakePipeline:
    @staticmethod
    def create(
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        device: torch.device,
        offload_mode: OffloadMode,
        components: ResolvedLtxComponents | None = None,
        *,
        loras: list[LoraPathStrengthAndSDOps] | None = None,
        quantization: QuantizationPolicy | None = None,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> RetakePipeline:
        return LTXRetakePipeline(
            checkpoint_path=checkpoint_path,
            gemma_root=gemma_root,
            device=device,
            offload_mode=offload_mode,
            components=components,
            loras=loras or [],
            quantization=quantization,
            memory_plan=memory_plan,
        )

    def __init__(
        self,
        checkpoint_path: CheckpointPath,
        gemma_root: str | None,
        device: torch.device,
        offload_mode: OffloadMode,
        components: ResolvedLtxComponents | None = None,
        *,
        loras: list[LoraPathStrengthAndSDOps],
        quantization: QuantizationPolicy | None,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> None:
        self._components = components
        from ltx_pipelines.utils.blocks import (
            AudioConditioner,
            AudioDecoder,
            DiffusionStage,
            ImageConditioner,
            PromptEncoder,
            VideoDecoder,
        )

        is_gguf = components is not None and components.transformer_format == "gguf"
        is_split = (
            components is not None
            and components.transformer_format == "safetensors"
            and components.video_vae_path is not None
        )

        if components is not None and components.gemma_root is not None:
            from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

            install_gguf_prompt_encoder_patch()

        # Phase 2: trust the memory plan's offload decision when provided; no
        # internal split/GGUF coercion. Transitional handler path
        # (memory_plan=None) uses the caller's offload_mode as-is.
        if memory_plan is not None:
            offload_mode = memory_plan.offload_mode
        self.device = device
        self.dtype = torch.bfloat16
        self._offload_mode = offload_mode

        self.prompt_encoder = PromptEncoder(
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]  # ponytail: ltx_pipelines accepts tuple per M5 spec
            gemma_root=gemma_root or "",
            dtype=self.dtype,
            device=device,
            offload_mode=self._offload_mode,
        )
        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            dtype=self.dtype,
            device=device,
        )
        self.audio_conditioner = AudioConditioner(
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            dtype=self.dtype,
            device=device,
        )
        if is_gguf:
            stage_quantization = None
        elif is_split and quantization is not None:
            from services.patches.gguf_loader_fix import kijai_fp8_quantization_policy

            stage_quantization = kijai_fp8_quantization_policy()
        else:
            stage_quantization = quantization

        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            dtype=self.dtype,
            device=device,
            loras=tuple(loras),
            quantization=stage_quantization,
            offload_mode=self._offload_mode,
        )
        # Phase 3B: stamp the memory plan onto the DiffusionStage so the
        # block-offload build patch can read it when the transformer is built
        # lazily at generation time. ``memory_plan`` is a dynamic attribute
        # (read via ``getattr`` in the patch); not declared on DiffusionStage.
        if memory_plan is not None:
            self.stage.memory_plan = memory_plan  # type: ignore[attr-defined]
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            dtype=self.dtype,
            device=device,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,  # type: ignore[arg-type]
            dtype=self.dtype,
            device=device,
        )

        if is_gguf:
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_gguf_loader

            install_gguf_loader(self)
            c = self._components
            install_gguf_component_paths(
                self,
                checkpoint_path,
                video_vae_path=c.video_vae_path if c else None,
                audio_vae_path=c.audio_vae_path if c else None,
            )

        if is_split:
            from services.patches.gguf_loader_fix import install_gguf_component_paths, install_kijai_transformer_config_patch

            c = self._components
            assert c is not None  # is_split guarantees this
            install_kijai_transformer_config_patch(self, checkpoint_path)
            install_gguf_component_paths(
                self,
                checkpoint_path,
                video_vae_path=c.video_vae_path,
                audio_vae_path=c.audio_vae_path,
            )

    @torch.no_grad()
    def _run(  # noqa: PLR0913, PLR0915
        self,
        video_path: str,
        prompt: str,
        start_time: float,
        end_time: float,
        seed: int,
        *,
        negative_prompt: str = "",
        num_inference_steps: int = 40,
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        from ltx_core.components.guiders import MultiModalGuider
        from ltx_core.components.noisers import GaussianNoiser
        from ltx_core.components.schedulers import LTX2Scheduler
        from ltx_core.conditioning.types.noise_mask_cond import TemporalRegionMask
        from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES as _distilled_sigmas
        from ltx_pipelines.utils.denoisers import GuidedDenoiser, SimpleDenoiser
        from ltx_pipelines.utils.helpers import audio_latent_from_file
        from ltx_pipelines.utils.types import ModalitySpec

        if start_time >= end_time:
            raise ValueError(f"start_time ({start_time}) must be less than end_time ({end_time})")

        # Image-sequence input: ``video_path`` is a SINGLE FILE from a sequence
        # (passed through unchanged by the handler — no temp file, no directory).
        # get_videostream_metadata / decode_video_from_file are monkey-patched to
        # resolve the sequence transparently; audio is absent for image sequences.
        from services.sequence_input import is_sequence_file

        is_sequence = is_sequence_file(video_path)

        effective_seed = int(torch.randint(0, 2**31, (1,)).item()) if seed < 0 else seed
        generator = torch.Generator(device=self.device).manual_seed(effective_seed)
        noiser = GaussianNoiser(generator=generator)
        from ltx_core.model.video_vae import SpatialTilingConfig, TemporalTilingConfig

        dtype = self.dtype
        tiling = TilingConfig.default()
        # Smaller tiles for source video encoding to reduce peak VRAM allocation
        # during the VAE encoder forward pass.
        encoding_tiling = TilingConfig(
            spatial_config=SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64),
            temporal_config=TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16),
        )

        # --- Encode source video (tiled) ---
        output_shape = get_videostream_metadata(video_path)

        # ponytail: LTX VAE needs spatial dims that are multiples of 64; source
        # videos can be any size (e.g. 1280x720). Align width/height UP to the
        # next multiple of 64 before generation (matches the other LTX paths).
        # The API/request dimensions are unchanged; only the internal generation
        # shape is aligned.
        _align = 64
        _h = ((output_shape.height + _align - 1) // _align) * _align
        _w = ((output_shape.width + _align - 1) // _align) * _align

        # ponytail: snap source video frames down to nearest 8n+1 for VAE compatibility.
        from ltx_core.types import SpatioTemporalScaleFactors
        _vae_time = SpatioTemporalScaleFactors.default().time
        if (output_shape.frames - 1) % _vae_time != 0:
            _frames = ((output_shape.frames - 1) // _vae_time) * _vae_time + 1
        else:
            _frames = output_shape.frames
        if (_h, _w, _frames) != (output_shape.height, output_shape.width, output_shape.frames):
            output_shape = type(output_shape)(
                output_shape.batch, _frames, _h, _w, output_shape.fps,
            )

        initial_video_latent = self.image_conditioner(
            lambda enc: _encode_source_video_latent_model_domain(
                video_encoder=enc,
                file_path=video_path,
                output_shape=output_shape,
                dtype=dtype,
                device=self.device,
                tiling_config=encoding_tiling,
            )
        )


        # --- Encode source audio ---
        # Image sequences have no audio stream (av.open on a frame file would
        # raise); skip audio conditioning for sequence inputs. Video files are
        # unchanged (byte-identical path through audio_latent_from_file).
        if is_sequence:
            initial_audio_latent = None
        else:
            initial_audio_latent = self.audio_conditioner(
                lambda enc: audio_latent_from_file(
                    audio_encoder=enc,
                    file_path=video_path,
                    output_shape=output_shape,
                    dtype=dtype,
                    device=self.device,
                )
            )


        # --- Text encoding ---

        prompts_to_encode = [prompt] if distilled else [prompt, negative_prompt]
        contexts = self.prompt_encoder(
            prompts_to_encode,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_seed=effective_seed,
        )


        v_context_p, a_context_p = contexts[0].video_encoding, contexts[0].audio_encoding

        # --- Build modality specs ---
        video_modality_spec = ModalitySpec(
            context=v_context_p,
            conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
            if regenerate_video
            else [],
            initial_latent=initial_video_latent,
            frozen=not regenerate_video,
        )
        audio_modality_spec: ModalitySpec | None = None
        if a_context_p is not None:
            audio_modality_spec = ModalitySpec(
                context=a_context_p,
                conditionings=[TemporalRegionMask(start_time=start_time, end_time=end_time, fps=output_shape.fps)]
                if (initial_audio_latent is not None and regenerate_audio)
                else [],
                initial_latent=initial_audio_latent,
                frozen=initial_audio_latent is not None and not regenerate_audio,
            )

        # --- Build denoiser ---
        if distilled:
            sigmas = torch.tensor(_distilled_sigmas).to(dtype=torch.float32, device=self.device)
            denoiser = SimpleDenoiser(v_context=v_context_p, a_context=a_context_p)
        else:
            sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)  # type: ignore[no-untyped-call]
            assert video_guider_params is not None, "video_guider_params required for non-distilled"
            assert audio_guider_params is not None, "audio_guider_params required for non-distilled"
            v_context_n, a_context_n = contexts[1].video_encoding, contexts[1].audio_encoding
            denoiser = GuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider=MultiModalGuider(params=video_guider_params, negative_context=v_context_n),
                audio_guider=MultiModalGuider(params=audio_guider_params, negative_context=a_context_n),
            )

        # --- Run diffusion stage ---

        video_state, audio_state = self.stage(
            denoiser=denoiser,
            sigmas=sigmas,
            noiser=noiser,
            width=output_shape.width,
            height=output_shape.height,
            frames=output_shape.frames,
            fps=output_shape.fps,
            video=video_modality_spec,
            audio=audio_modality_spec,
        )


        # --- Decode audio first (eager, small) ---
        assert audio_state is not None
        decoded_audio = self.audio_decoder(audio_state.latent)

        # --- Decode video (lazy generator, tiled) ---
        assert video_state is not None
        decoded_video = self.video_decoder(video_state.latent, tiling, generator)

        return decoded_video, decoded_audio

    @torch.no_grad()
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
        video_guider_params: MultiModalGuiderParams | None = None,
        audio_guider_params: MultiModalGuiderParams | None = None,
        regenerate_video: bool = True,
        regenerate_audio: bool = True,
        enhance_prompt: bool = False,
        distilled: bool = True,
        output_format: OutputFormat = OutputFormat.MP4,
        encoder: MediaEncoder | None = None,
        proxy_path: str | None = None,
        on_progress: Callable[[float], None] | None = None,
        input_colorspace: ColorSpace | None = None,
    ) -> None:
        # Image-sequence input: the ORIGINAL single-file video_path is passed
        # through unchanged (no temp MP4, no directory). get_videostream_metadata
        # is monkey-patched to resolve sequences transparently; NON-sequence
        # inputs are byte-identical (patched metadata delegates to the original).
        meta = get_videostream_metadata(video_path)
        fps, num_frames = meta.fps, meta.frames
        video_iter, audio = self._run(
            video_path=video_path,
            prompt=prompt,
            start_time=start_time,
            end_time=end_time,
            seed=seed,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            audio_guider_params=audio_guider_params,
            regenerate_video=regenerate_video,
            regenerate_audio=regenerate_audio,
            enhance_prompt=enhance_prompt,
            distilled=distilled,
        )
        audio_out: Audio | None = audio
        tiling_config = TilingConfig.default()
        video_chunks = get_video_chunks_number(num_frames, tiling_config)
        # Routed through the shared encode_video_output dispatcher (was a direct
        # encode_video call). Relies on MP4 defaults → byte-identical to today,
        # but now funnels through the MediaEncoder so format/proxy can be added
        # by handlers in a later phase without touching this pipeline again.
        encode_video_output(
            video=video_iter,
            audio=audio_out,
            fps=int(fps),
            output_path=output_path,
            video_chunks_number_value=video_chunks,
            output_format=output_format,
            encoder=encoder,
            proxy_path=proxy_path,
            on_progress=on_progress,
            input_colorspace=input_colorspace,
            total_frames=num_frames,
        )
