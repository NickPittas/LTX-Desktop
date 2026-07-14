"""Video-only latent bridge for IC-LoRA inpainting."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F

from api_types import ImageConditioningInput, OutputFormat
from services import memory_trace
from services.ltx_pipeline_common import (
    default_tiling_config,
    encode_video_output,
    make_ltx_image_conditioning_input,
    video_chunks_number,
)

from .ltx_ic_lora_pipeline import derive_stage_radii
from .official_inpaint import dilate_video_mask, green_composite_preprocess

OFFICIAL_NEGATIVE_PROMPT = "pc game, console game, video game, cartoon, childish, ugly"

if TYPE_CHECKING:
    from services.color_management import ColorSpace
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    from services.media_encoder.media_encoder import MediaEncoder


def mask_to_latent_denoise_mask(mask: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
    """Causally map F×H×W generation pixels to B×1×F_lat×H_lat×W_lat."""
    if mask.ndim != 3 or latent.ndim != 5:
        raise ValueError("Latent inpaint mask/latent shapes must be (F,H,W) and (B,C,F,H,W)")
    _, _, latent_frames, latent_h, latent_w = latent.shape
    binary = (mask >= 0.5).to(torch.float32)
    spatial = F.interpolate(binary.unsqueeze(1), size=(latent_h, latent_w), mode="nearest").squeeze(1)
    groups = [spatial[0]]
    for index in range(1, latent_frames):
        start, end = 1 + 8 * (index - 1), 1 + 8 * index
        if start >= spatial.shape[0]:
            groups.append(spatial[-1])
        else:
            groups.append(spatial[start:min(end, spatial.shape[0])].amax(dim=0))
    converted = torch.stack(groups, dim=0).to(torch.float32)
    return converted.unsqueeze(0).unsqueeze(0)


class InpaintTargetGuideAttention:
    """Append a green guide blocking masked target queries from guide keys."""

    def __init__(self, inner: Any, target_mask: torch.Tensor) -> None:
        self.inner = inner
        self.target_mask = target_mask

    def apply_to(self, latent_state: Any, latent_tools: Any) -> Any:
        if latent_state.latent.shape[0] != 1 or self.target_mask.shape[0] != 1:
            raise ValueError("Latent inpaint guide attention requires batch size one")
        if latent_state.attention_mask is not None:
            raise ValueError("Latent inpaint guide attention requires no existing attention mask")
        target_count = latent_tools.target_shape.token_count()
        if latent_state.latent.shape[1] != target_count:
            raise ValueError("Latent inpaint guide attention target token count does not match latent tools")
        try:
            target_tokens = latent_tools.patchifier.patchify(self.target_mask)
        except (IndexError, RuntimeError) as exc:
            raise ValueError("Latent inpaint guide attention cannot patchify target mask") from exc
        if target_tokens.shape != (1, target_count, 1):
            raise ValueError("Latent inpaint guide attention patchified mask shape is invalid")
        if not torch.all((target_tokens == 0) | (target_tokens == 1)):
            raise ValueError("Latent inpaint guide attention requires a binary target mask")
        conditioned = self.inner.apply_to(latent_state, latent_tools)
        if conditioned.latent.shape[0] != 1 or conditioned.latent.shape[1] != 2 * target_count:
            raise ValueError("Latent inpaint guide attention requires exactly one guide token per target token")
        if not torch.equal(conditioned.latent[:, :target_count], latent_state.latent):
            raise ValueError("Latent inpaint guide attention requires guide-last token ordering")
        attention = torch.ones(
            (1, 2 * target_count, 2 * target_count),
            device=conditioned.latent.device,
            dtype=torch.bool,
        )
        masked_targets = target_tokens[..., 0].to(device=attention.device, dtype=torch.bool)
        attention[:, :target_count, target_count:] &= ~masked_targets[:, :, None]
        return replace(conditioned, attention_mask=attention)


def blend_half_latents(generated: torch.Tensor, source: torch.Tensor, denoise_mask: torch.Tensor) -> torch.Tensor:
    """Use generated values inside white mask and clean source outside it."""
    expected_mask = (generated.shape[0], 1, *generated.shape[2:])
    if generated.shape != source.shape or denoise_mask.shape != expected_mask:
        raise ValueError("Latent inpaint half latent/source/mask shapes do not agree exactly")
    if generated.dtype != source.dtype or generated.device != source.device:
        raise ValueError("Latent inpaint half latents must share dtype and device")
    return generated * denoise_mask.to(device=generated.device, dtype=generated.dtype) + source * (
        1 - denoise_mask.to(device=generated.device, dtype=generated.dtype)
    )


def build_inpaint_context_config(window_px: int, overlap_px: int, latent_frames: int) -> tuple[int, int]:
    window = min(((window_px - 1) // 8) + 1, latent_frames)
    overlap = min(overlap_px // 8, window - 1)
    if not 0 <= overlap < window <= latent_frames:
        raise ValueError("Invalid latent inpaint context window/overlap")
    return window, overlap


def _windows(frames: int, window: int, overlap: int) -> tuple[tuple[int, int], ...]:
    if window >= frames:
        return ((0, frames),)
    stride, result, start = window - overlap, [], 0
    result: list[tuple[int, int]]
    while start + window <= frames:
        result.append((start, start + window))
        start += stride
    last = (frames - window, frames)
    if result[-1] != last:
        result.append(last)
    return tuple(result)


def _collect_frames(iterator: Any) -> torch.Tensor:
    return torch.cat([chunk.cpu() for chunk in iterator], dim=0)


def _context_loop(window: int, overlap: int, frames: int, height: int, width: int) -> Any:
    """Call-scoped target-only video Euler loop; no stage mutation or HDR patch."""
    from ltx_pipelines.utils.samplers import post_process_latent

    tokens_per_frame = height * width
    windows = _windows(frames, window, overlap)

    def loop(sigmas: torch.Tensor, video_state: Any, audio_state: Any, stepper: Any, transformer: Any, denoiser: Any) -> tuple[Any, None]:
        if video_state is None:
            return None, None
        target_tokens = frames * tokens_per_frame
        if video_state.attention_mask is not None or video_state.latent.shape[1] != target_tokens:
            raise ValueError("Latent inpaint context loop supports target-only video tokens with no attention mask")
        fused = torch.zeros_like(video_state.latent)
        weights = torch.zeros(frames, device=video_state.latent.device, dtype=video_state.latent.dtype)
        for step_index in range(len(sigmas) - 1):
            fused.zero_(); weights.zero_()
            for start, end in windows:
                token_start, token_end = start * tokens_per_frame, end * tokens_per_frame
                state = replace(
                    video_state,
                    latent=video_state.latent[:, token_start:token_end],
                    denoise_mask=video_state.denoise_mask[:, token_start:token_end],
                    clean_latent=video_state.clean_latent[:, token_start:token_end],
                    positions=video_state.positions[:, :, token_start:token_end, :],
                )
                video_result, _ = denoiser(transformer, state, None, sigmas, step_index)
                if video_result is None:
                    raise RuntimeError("Latent inpaint context denoiser returned no video result")
                denoised = video_result.denoised
                length = end - start
                local = torch.minimum(
                    torch.minimum(torch.arange(length, device=weights.device, dtype=weights.dtype) + 1, torch.arange(length, 0, -1, device=weights.device, dtype=weights.dtype)),
                    torch.full((length,), float(max(overlap, 1) + 1), device=weights.device, dtype=weights.dtype),
                )
                token_weights = local.repeat_interleave(tokens_per_frame)
                fused[:, token_start:token_end] += denoised * token_weights[None, :, None]
                weights[start:end] += local
            if not bool(torch.all(weights > 0)):
                raise ValueError("Latent inpaint context fusion left an uncovered latent frame")
            processed = post_process_latent(
                fused / weights.repeat_interleave(tokens_per_frame)[None, :, None],
                video_state.denoise_mask, video_state.clean_latent,
            )
            video_state = replace(video_state, latent=stepper.step(video_state.latent, processed, sigmas, step_index))
        return video_state, None

    return loop


def _stage1_cfg_pp_ancestral_loop(
    positive_video_context: torch.Tensor, negative_video_context: torch.Tensor, generator: torch.Generator,
) -> Any:
    """Return the official CFG++ ancestral Stage-1 loop for the masked guide path."""
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.samplers import EulerCfgPpDiffusionStep, post_process_latent

    cfg_pp_stepper = EulerCfgPpDiffusionStep(eta=1.0, s_noise=1.0)
    positive_denoiser = SimpleDenoiser(positive_video_context, None)
    negative_denoiser = SimpleDenoiser(negative_video_context, None)

    def loop(sigmas: torch.Tensor, video_state: Any, audio_state: Any, stepper: Any, transformer: Any, denoiser: Any) -> tuple[Any, None]:
        _ = stepper, denoiser  # DiffusionStage loop contract; CFG++ owns these locally.
        if video_state is None:
            raise RuntimeError("Latent inpaint CFG++ Stage 1 requires a video state")
        if audio_state is not None:
            raise ValueError("Latent inpaint CFG++ Stage 1 is video-only")
        for step_index in range(len(sigmas) - 1):
            state = video_state
            positive_result, _ = positive_denoiser(transformer, state, None, sigmas, step_index)
            negative_result, _ = negative_denoiser(transformer, state, None, sigmas, step_index)
            if (
                positive_result is None
                or negative_result is None
                or not isinstance(getattr(positive_result, "denoised", None), torch.Tensor)
                or not isinstance(getattr(negative_result, "denoised", None), torch.Tensor)
            ):
                raise RuntimeError("Latent inpaint CFG++ Stage 1 denoiser did not return video denoised tensors")
            positive = post_process_latent(positive_result.denoised, state.denoise_mask, state.clean_latent)
            if bool(sigmas[step_index + 1] == 0):
                return replace(state, latent=positive), None
            noise = torch.randn(state.latent.shape, generator=generator, device=state.latent.device, dtype=state.latent.dtype)
            step = cast(Callable[..., torch.Tensor], getattr(cfg_pp_stepper, "step"))
            x_next = step(
                sample=state.latent,
                denoised_sample=positive,
                uncond_denoised=negative_result.denoised,
                sigmas=sigmas,
                step_index=step_index,
                noise=noise,
            )
            video_state = replace(
                state,
                latent=post_process_latent(x_next, state.denoise_mask, state.clean_latent),
            )
        return video_state, None

    return loop


def generate_inpaint(  # noqa: PLR0913, PLR0915
    pipeline: LTXIcLoraPipeline, *, prompt: str, seed: int, height: int, width: int,
    num_frames: int, frame_rate: float, images: list[ImageConditioningInput], video_path: str,
    mask_path: str, output_path: str, conditioning_strength: float = 1.0, mask_grow_px: int = 30,
    output_format: OutputFormat = OutputFormat.MP4, encoder: MediaEncoder | None = None,
    proxy_path: str | None = None, on_progress: Callable[[float], None] | None = None,
    input_colorspace: ColorSpace | None = None,
    on_phase_update: Callable[[str, str | None], None] | None = None,
    save_stage_1_preview: bool = False, inpaint_context_window_px: int | None = None,
    inpaint_context_overlap_px: int | None = None,
) -> str | None:
    """Video-only latent inpaint: source/generation latent blend → one learned upsample → masked Stage 2."""
    from ltx_core.components.noisers import GaussianNoiser
    from ltx_core.conditioning import VideoConditionByMask
    from ltx_pipelines.utils.constants import DISTILLED_SIGMAS, STAGE_2_DISTILLED_SIGMAS
    from ltx_pipelines.utils.denoisers import SimpleDenoiser
    from ltx_pipelines.utils.helpers import assert_resolution, combined_image_conditionings
    from ltx_pipelines.utils.media_io import decode_video_by_frame, video_preprocess
    from ltx_pipelines.utils.types import ModalitySpec
    from services.exr_input import iter_video_frames_to_model_domain

    assert_resolution(height=height, width=width, is_two_stage=True)
    device, dtype, cpu = pipeline.pipeline.device, pipeline.pipeline.dtype, torch.device("cpu")
    half_h, half_w = height // 2, width // 2
    padded = ((num_frames - 2) // 8 + 1) * 8 + 1
    stage1_radius, stage2_radius = derive_stage_radii(mask_grow_px)
    tiling = default_tiling_config()
    if on_phase_update is not None:
        on_phase_update("inference", "Latent inpaint")
    memory_trace.snapshot("inpaint:start")

    source_full = video_preprocess(iter_video_frames_to_model_domain(video_path, frame_cap=num_frames, device=cpu), height, width, torch.float32, cpu)
    actual = source_full.shape[2]
    if actual < padded:
        source_full = torch.cat((source_full, source_full[:, :, -1:].expand(-1, -1, padded - actual, -1, -1)), dim=2)
    source_full = source_full[:, :, :padded]
    batch, channels, frames, _, _ = source_full.shape
    source_half = F.interpolate(
        source_full.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width),
        size=(half_h, half_w), mode="bilinear", align_corners=False,
    ).reshape(batch, frames, channels, half_h, half_w).permute(0, 2, 1, 3, 4)
    mask_video = video_preprocess(decode_video_by_frame(path=mask_path, frame_cap=padded, device=cpu), height, width, torch.float32, cpu)
    if mask_video.shape[2] < padded:
        mask_video = torch.cat((mask_video, mask_video[:, :, -1:].expand(-1, -1, padded - mask_video.shape[2], -1, -1)), dim=2)
    raw_mask = ((mask_video[:, :, :padded].mean(dim=1)[0] + 1) / 2).to(torch.float32)
    mask_stage1 = dilate_video_mask(raw_mask, spatial_radius=stage1_radius, temporal_radius=0)
    mask_stage2 = dilate_video_mask(raw_mask, spatial_radius=stage2_radius, temporal_radius=0)
    guide_mask_half = F.interpolate(mask_stage1.unsqueeze(1), size=(half_h, half_w), mode="nearest").squeeze(1)
    green_half = green_composite_preprocess(source_half, guide_mask_half)
    ltx_images = [make_ltx_image_conditioning_input(item.path, item.frame_idx, item.strength) for item in images]
    source_latent_half: Any = None
    half_mask: torch.Tensor | None = None
    stage1_cfg_pp_enabled = False
    negative_video_context: torch.Tensor | None = None
    with memory_trace.phase("inpaint:source_vae_encode_half"):
        def encode_half(enc: Any) -> list[Any]:
            nonlocal source_latent_half, half_mask, stage1_cfg_pp_enabled
            source_latent_half = enc.tiled_encode(source_half.to(device=device, dtype=dtype), tiling)
            half_mask = mask_to_latent_denoise_mask(mask_stage1, source_latent_half)
            from ltx_core.conditioning import VideoConditionByKeyframeIndex
            encoded_guide = enc.tiled_encode(green_half.to(device=device, dtype=dtype), tiling)
            guide: Any = VideoConditionByKeyframeIndex(keyframes=encoded_guide, frame_idx=0, strength=conditioning_strength, num_pixel_frames=padded)
            if prompt.strip() and not images and bool(torch.any(half_mask)):
                guide = InpaintTargetGuideAttention(guide, half_mask)
                stage1_cfg_pp_enabled = True
            return combined_image_conditionings(images=ltx_images, height=half_h, width=half_w, video_encoder=enc, dtype=dtype, device=device) + [guide]
        stage1_conditionings = pipeline.pipeline.image_conditioner(encode_half)
    del source_full, source_half, green_half, guide_mask_half
    if stage1_cfg_pp_enabled:
        positive_prompt_context, negative_prompt_context = pipeline.pipeline.prompt_encoder(
            [prompt, OFFICIAL_NEGATIVE_PROMPT], enhance_first_prompt=False,
        )
        video_context, negative_video_context = positive_prompt_context.video_encoding, negative_prompt_context.video_encoding
    else:
        (prompt_context,) = pipeline.pipeline.prompt_encoder([prompt], enhance_first_prompt=False)
        video_context = prompt_context.video_encoding
    generator1 = torch.Generator(device=device).manual_seed(seed + 1)
    ancestral_generator = torch.Generator(device=device).manual_seed(seed + 1) if stage1_cfg_pp_enabled else None
    stage1_kwargs: dict[str, Any] = {}
    if ancestral_generator is not None and negative_video_context is not None:
        stage1_kwargs["loop"] = _stage1_cfg_pp_ancestral_loop(video_context, negative_video_context, ancestral_generator)
    with memory_trace.phase("inpaint:stage1_denoise"):
        video_state1, _ = pipeline.pipeline.stage_1(
            denoiser=SimpleDenoiser(video_context, None), sigmas=DISTILLED_SIGMAS.to(dtype=torch.float32, device=device),
            noiser=GaussianNoiser(generator=generator1), width=half_w, height=half_h, frames=padded, fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage1_conditionings),
            audio=None,
            **stage1_kwargs,
        )
    if video_state1 is None:
        raise RuntimeError("Latent inpaint Stage 1 did not produce a video latent")
    if save_stage_1_preview:
        preview = str(Path(output_path).parent / f"{Path(output_path).stem}_stage1_preview.mp4")
        frames = _collect_frames(pipeline.pipeline.video_decoder(video_state1.latent, tiling, generator1))
        encode_video_output(video=frames.clamp(0, 1), audio=None, fps=int(frame_rate), output_path=preview, video_chunks_number_value=video_chunks_number(frames.shape[0], tiling), output_format=OutputFormat.MP4, encoder=encoder, total_frames=frames.shape[0])
    else:
        preview = None
    if half_mask is None:
        raise RuntimeError("Latent inpaint half mask was not encoded")
    with memory_trace.phase("inpaint:half_latent_blend"):
        blended = blend_half_latents(video_state1.latent, source_latent_half, half_mask)
    del video_state1, source_latent_half, half_mask, stage1_conditionings
    if not hasattr(pipeline.pipeline, "upsampler"):
        raise RuntimeError("Latent inpaint requires the loaded spatial upsampler")
    with memory_trace.phase("inpaint:learned_upsample"):
        stage2_initial = pipeline.pipeline.upsampler(blended[:1])
    del blended
    full_mask = mask_to_latent_denoise_mask(mask_stage2, stage2_initial)
    if full_mask.shape[2:] != stage2_initial.shape[2:]:
        raise ValueError("Latent inpaint full latent mask does not match upsampler output")
    preserve = 1 - full_mask
    stage2_conditionings: list[Any] = [
        VideoConditionByMask(
            latent=stage2_initial,
            mask=preserve.squeeze(1).to(device=device, dtype=dtype),
            strength=1.0,
        )
    ]
    stage2_sigmas = STAGE_2_DISTILLED_SIGMAS[1:].to(dtype=torch.float32, device=device)
    sigmas = stage2_sigmas * (0.55 / stage2_sigmas[0].item())
    loop = None
    if inpaint_context_window_px is not None or inpaint_context_overlap_px is not None:
        if inpaint_context_window_px is None or inpaint_context_overlap_px is None:
            raise ValueError("Latent inpaint context policy requires both window and overlap")
        window, overlap = build_inpaint_context_config(inpaint_context_window_px, inpaint_context_overlap_px, stage2_initial.shape[2])
        loop = _context_loop(window, overlap, stage2_initial.shape[2], stage2_initial.shape[3], stage2_initial.shape[4])
    generator2 = torch.Generator(device=device).manual_seed(seed)
    with memory_trace.phase("inpaint:stage2_denoise"):
        video_state2, _ = pipeline.pipeline.stage_2(
            denoiser=SimpleDenoiser(video_context, None), sigmas=sigmas, noiser=GaussianNoiser(generator=generator2),
            width=width, height=height, frames=padded, fps=frame_rate,
            video=ModalitySpec(context=video_context, conditionings=stage2_conditionings, noise_scale=sigmas[0].item(), initial_latent=stage2_initial),
            audio=None, loop=loop,
        )
    if video_state2 is None:
        raise RuntimeError("Latent inpaint Stage 2 did not produce a video latent")
    with memory_trace.phase("inpaint:stage2_decode"):
        output = _collect_frames(pipeline.pipeline.video_decoder(video_state2.latent, tiling, generator2))[:actual]
    encode_video_output(video=output.clamp(0, 1), audio=None, fps=int(frame_rate), output_path=output_path, video_chunks_number_value=video_chunks_number(actual, tiling), output_format=output_format, encoder=encoder, proxy_path=proxy_path, on_progress=on_progress, input_colorspace=input_colorspace, total_frames=actual)
    memory_trace.snapshot("inpaint:end")
    return preview
