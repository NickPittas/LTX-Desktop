"""LTX-Desktop HDR IC-LoRA pipeline wrapper (Phase 2: upstream replacement).

Thin subclass of the upstream ``ltx_pipelines.hdr_ic_lora.HDRICLoraPipeline``.

Generation math (stage 1 / spatial upsampler / stage 2 / decode) is delegated
to upstream **unchanged**. This wrapper overrides **only**
``_create_conditionings`` to honor the app invariants that upstream's
file-path + frame-cap conditioning loader cannot:

- decode **all** source frames (no container-metadata count, no frame cap);
- never trim source frames;
- pad **in memory** with duplicate copies of the genuine final decoded frame
  until the count is ``8n + 1`` (single-frame input stays ``1``);
- never write a temp/recompressed video to disk.

Everything else — sampling, tiling, resize, embeddings, HDR decode — is
upstream's. The ``generate`` wrapper writes the linear HDR tensor as an EXR
primary sequence (no EOTF / tonemap / clamp — linear passthrough) via
``save_exr_tensor`` and then an SDR proxy MP4 via
``encode_exr_sequence_to_mp4`` (strictly after the primary).
"""

from __future__ import annotations

import functools
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, NamedTuple, cast

import torch
from safetensors import safe_open

from api_types import OutputFormat
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.conditioning import ConditioningItem, VideoConditionByReferenceLatent
from ltx_core.hdr import LogC3
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.sd_ops import LTXV_LORA_COMFY_RENAMING_MAP
from ltx_core.model.video_vae import TilingConfig, VideoEncoder
from ltx_core.model.video_vae.tiling import (
    SpatialTilingConfig,
    TemporalTilingConfig,
)
from ltx_core.modality_tiling import VideoLatentShape, VideoModalityTilingHelper
from ltx_core.quantization import QuantizationPolicy
from ltx_core.tools import VideoLatentPatchifier, VideoLatentTools
from ltx_core.types import LatentState
from ltx_pipelines.hdr_ic_lora import (
    ALIGNMENT_DIVISOR,
    HDRICLoraPipeline,
    HdrLoraConfig,
    MIN_RESOLUTION,
    TileCountConfig,
    read_hdr_lora_config,
)
from ltx_pipelines.utils.blocks import (
    DiffusionStage,
    ImageConditioner,
    VideoDecoder,
    VideoUpsampler,
)
from ltx_pipelines.utils.constants import DISTILLED_SIGMA_VALUES
from ltx_pipelines.utils.denoisers import SimpleDenoiser
from ltx_pipelines.utils.helpers import modality_from_latent_state
from ltx_pipelines.utils.media_io import (
    ResizeMode,
    align_resolution,
    decode_video_by_frame,
    encode_exr_sequence_to_mp4,
    resize_and_reflect_pad,
    save_exr_tensor,
    to_vae_range,
)
from ltx_pipelines.utils.samplers import post_process_latent
from ltx_pipelines.utils.types import ModalitySpec
from ltx_pipelines.utils.types import OffloadMode
from services import memory_trace
from services.services_utils import device_supports_fp8

if TYPE_CHECKING:
    from collections.abc import Callable

    from ltx_core.loader.registry import Registry

    from services.color_management import ColorSpace
    from services.ltx_components import BaseFamily, ResolvedLtxComponents, TransformerFormat
    from services.local_memory_plan import LocalMemoryPlan
    from services.media_encoder.media_encoder import MediaEncoder

logger = logging.getLogger(__name__)


# Harness/debug knobs for HDR VAE tiling (NOT production UI policy).
# Encode tiling is chosen by a runtime policy from VRAM + padded frames (31GiB:
# no tile <=121f, tile512 >121f; lower VRAM tiles earlier) unless
# LTX_HDR_VAE_TILE_SIZE explicitly overrides. Decode tiling is ALWAYS on, with
# its own separate config (default spatial 768 / overlap 128).
_LTX_HDR_VAE_TILE_SIZE_ENV = "LTX_HDR_VAE_TILE_SIZE"
_LTX_HDR_VAE_TILE_THRESHOLD_ENV = "LTX_HDR_VAE_TILE_THRESHOLD"
_LTX_HDR_DECODE_VAE_TILE_SIZE_ENV = "LTX_HDR_DECODE_VAE_TILE_SIZE"
_LTX_HDR_TIMING_ENV = "LTX_HDR_TIMING"
# Process-local context knob set around generation so the resident-block policy
# (services.block_offload) can see the padded frame count at transformer build.
_LTX_HDR_PADDED_FRAMES_ENV = "LTX_HDR_PADDED_FRAMES"


def _hdr_timing_enabled() -> bool:
    """True when ``LTX_HDR_TIMING=1`` (coarse HDR phase timing; debug/harness knob)."""
    return os.environ.get(_LTX_HDR_TIMING_ENV) == "1"


# Encode overlap + temporal (shared by explicit-override and policy tiling).
_HDR_ENCODE_SPATIAL_OVERLAP = 64
_HDR_TEMPORAL_TILE_FRAMES = 24
_HDR_TEMPORAL_TILE_OVERLAP = 16
# Encode tile size chosen by the runtime policy when it decides to tile.
_HDR_ENCODE_POLICY_TILE = 512
# Decode tiling defaults (always on; independent of the encode config).
_HDR_DECODE_SPATIAL_TILE = 768
_HDR_DECODE_SPATIAL_OVERLAP = 128


def _detect_vram_gib() -> int | None:
    """CUDA total VRAM as an integer GiB floor, or None if unavailable."""
    try:
        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device()
        total_bytes = cast(int, torch.cuda.get_device_properties(dev).total_memory)  # type: ignore[reportUnknownMemberType]
        return total_bytes // (1024 ** 3)
    except Exception:
        return None


def _hdr_encode_tile_policy(vram_gib: int | None, padded_frames: int) -> int:
    """Encode spatial tile size (0 = no tiling) from VRAM + padded frames.

    31GiB: ``<=121`` frames -> no tile, ``>121`` -> 512. Lower VRAM tiles earlier
    (24GiB frame cap 97; 16GiB cap 1; <16GiB always tiles). Monotonic in VRAM:
    a smaller GPU never gets a no-tile outcome a larger one already tiles.
    """
    if vram_gib is None:
        return 0  # undetectable -> default off; the OOM fallback covers real pressure
    if vram_gib >= 31:
        frame_cap = 121
    elif vram_gib >= 24:
        frame_cap = 97
    elif vram_gib >= 16:
        frame_cap = 1
    else:
        frame_cap = 0
    return 0 if padded_frames <= frame_cap else _HDR_ENCODE_POLICY_TILE


def _hdr_encode_tiling(tile_size: int) -> TilingConfig | None:
    """Build an encode ``TilingConfig`` for ``tile_size`` (``<=0`` -> ``None``)."""
    if tile_size <= 0:
        return None
    return TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=tile_size,
            tile_overlap_in_pixels=_HDR_ENCODE_SPATIAL_OVERLAP,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=_HDR_TEMPORAL_TILE_FRAMES,
            tile_overlap_in_frames=_HDR_TEMPORAL_TILE_OVERLAP,
        ),
    )


def _hdr_tiling_overrides(
    padded_num_frames: int,
) -> tuple[TilingConfig | None, int | None]:
    """Resolve HDR VAE **encode** tiling config + threshold.

    ``LTX_HDR_VAE_TILE_SIZE`` is an explicit override:

    - unset  -> runtime policy from VRAM + ``padded_num_frames`` (emits an
      ``hdr_encode_policy`` trace event with the computed tile size).
    - ``=0`` -> ``None`` (no tiling).
    - ``=N>64`` -> spatial tile N, 64px overlap, temporal 24/16.
    - ``1..64`` -> ``ValueError`` (no room for the 64px overlap).

    ``LTX_HDR_VAE_TILE_THRESHOLD=N`` overrides ``_tiled_vae_encode_threshold``
    for one call (only meaningful with an explicit tile size). Returns
    ``(encode_tiling_config_or_None, threshold_override_or_None)``.
    """
    size_env = os.environ.get(_LTX_HDR_VAE_TILE_SIZE_ENV)
    if size_env is None:
        vram_gib = _detect_vram_gib()
        tile_size = _hdr_encode_tile_policy(vram_gib, padded_num_frames)
        memory_trace.write_event(
            "hdr_encode_policy",
            "hdr_generate",
            vram_gib=(vram_gib if vram_gib is not None else -1),
            padded_frames=padded_num_frames,
            encode_tile_size=tile_size,
        )
        return _hdr_encode_tiling(tile_size), None

    try:
        size = int(size_env)
    except ValueError as exc:
        raise ValueError(
            f"{_LTX_HDR_VAE_TILE_SIZE_ENV} must be an int, got {size_env!r}"
        ) from exc
    if size < 0:
        raise ValueError(f"{_LTX_HDR_VAE_TILE_SIZE_ENV} must be >= 0, got {size}")
    if 0 < size <= _HDR_ENCODE_SPATIAL_OVERLAP:
        raise ValueError(
            f"{_LTX_HDR_VAE_TILE_SIZE_ENV}={size} must be > {_HDR_ENCODE_SPATIAL_OVERLAP} "
            f"to leave room for a {_HDR_ENCODE_SPATIAL_OVERLAP}px spatial overlap"
        )

    threshold: int | None = None
    thresh_env = os.environ.get(_LTX_HDR_VAE_TILE_THRESHOLD_ENV)
    if thresh_env is not None:
        try:
            threshold = int(thresh_env)
        except ValueError as exc:
            raise ValueError(
                f"{_LTX_HDR_VAE_TILE_THRESHOLD_ENV} must be an int, got {thresh_env!r}"
            ) from exc
        if threshold < 0:
            raise ValueError(
                f"{_LTX_HDR_VAE_TILE_THRESHOLD_ENV} must be >= 0, got {threshold}"
            )
    return _hdr_encode_tiling(size), threshold


def _hdr_decode_tiling_config() -> tuple[TilingConfig, int, int]:
    """Resolve the HDR VAE **decode** tiling config (always on; separate from encode).

    Default spatial tile 768 / overlap 128 (temporal reuses the safe 24/16). The
    default is never silently smaller than 512. ``LTX_HDR_DECODE_VAE_TILE_SIZE``
    overrides the spatial tile (unset -> 768); an explicit value must exceed the
    128px overlap or a clear ``ValueError`` is raised. Non-int values raise too.

    Returns ``(config, spatial_tile_size, spatial_overlap)`` — the ints are
    returned alongside because ``TilingConfig.spatial_config`` is typed Optional,
    so callers can log/trace them without re-derefing the Optional.
    """
    raw = os.environ.get(_LTX_HDR_DECODE_VAE_TILE_SIZE_ENV)
    if raw is None:
        size = _HDR_DECODE_SPATIAL_TILE
    else:
        try:
            size = int(raw)
        except ValueError as exc:
            raise ValueError(
                f"{_LTX_HDR_DECODE_VAE_TILE_SIZE_ENV} must be an int, got {raw!r}"
            ) from exc
        if size <= _HDR_DECODE_SPATIAL_OVERLAP:
            raise ValueError(
                f"{_LTX_HDR_DECODE_VAE_TILE_SIZE_ENV}={size} must be > "
                f"{_HDR_DECODE_SPATIAL_OVERLAP} to leave room for a "
                f"{_HDR_DECODE_SPATIAL_OVERLAP}px spatial overlap"
            )
    config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=size,
            tile_overlap_in_pixels=_HDR_DECODE_SPATIAL_OVERLAP,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=_HDR_TEMPORAL_TILE_FRAMES,
            tile_overlap_in_frames=_HDR_TEMPORAL_TILE_OVERLAP,
        ),
    )
    return config, size, _HDR_DECODE_SPATIAL_OVERLAP
    return tiling, threshold


# ── HDR temporal ContextWindows (harness/debug knob, NOT production UI policy) ──
#
# ComfyUI-style temporal context windows inside the denoising loop. On by
# default for one-stage HDR (VRAM-based defaults below); set
# ``LTX_HDR_CONTEXT_WINDOW=0`` to disable. Explicit env values override the
# defaults. NOT LoopingSampler/full-pipeline chunking: the pipeline runs once,
# and each denoise step fans the transformer out over overlapping temporal
# latent windows, fusing the denoised target tokens back into the full state.
_LTX_HDR_CONTEXT_WINDOW_ENV = "LTX_HDR_CONTEXT_WINDOW"
_LTX_HDR_CONTEXT_OVERLAP_ENV = "LTX_HDR_CONTEXT_OVERLAP"
_LTX_HDR_CONTEXT_FUSE_ENV = "LTX_HDR_CONTEXT_FUSE"


class _ContextWindowConfig(NamedTuple):
    window_latent: int
    overlap_latent: int
    fuse: str  # "pyramid" | "flat"


def _default_context_window_px(vram_gib: int | None) -> tuple[int, int]:
    """Default (window_px, overlap_px) for HDR one-stage rolling windows.

    Measured on RTX 5090 31GiB: 65-frame window with 16 overlap + 46 resident
    blocks passes full 201f Kijai; 73 fails. Smaller GPUs get tighter windows.
    """
    if vram_gib is not None and vram_gib >= 31:
        return 65, 16
    if vram_gib is not None and vram_gib >= 24:
        return 49, 16
    return 33, 8


def _build_context_config(window_px: int, overlap_px: int, fuse: str) -> _ContextWindowConfig:
    """Convert pixel window/overlap to latent frames and validate."""
    window_latent = ((window_px - 1) // 8) + 1
    overlap_latent = overlap_px // 8
    if window_latent < 2:
        raise ValueError(
            f"context window {window_px}px -> {window_latent} latent frames; "
            "need >= 2 latent frames (>= 9 pixel frames)."
        )
    if not (0 <= overlap_latent < window_latent):
        raise ValueError(
            f"context overlap {overlap_px}px -> {overlap_latent} latent frames must satisfy "
            f"0 <= overlap < window ({window_latent})."
        )
    return _ContextWindowConfig(window_latent, overlap_latent, fuse)


def _hdr_context_config(padded_frames: int | None = None) -> _ContextWindowConfig | None:
    """Resolve the HDR rolling-window config.

    ``LTX_HDR_CONTEXT_WINDOW`` set → explicit env override (0/negative disables).
    Unset → resolve the padded frame count: prefer the ``padded_frames`` argument
    (the ``generate()`` caller already knows it as a local); if that is ``None``,
    fall back to the ``LTX_HDR_PADDED_FRAMES`` env. At or below 121 padded frames
    the one-stage pass fits without rolling windows, so returns ``None``
    (disabled, with a trace event); above 121, or when no frame count is
    available/malformed, falls back to the VRAM-based default (≥31 GiB → 65/16,
    ≥24 GiB → 49/16, otherwise 33/8). Pixel frames are converted to latent frames
    via the LTX ``((px-1)//8)+1`` formula; overlap via ``px//8``. Malformed
    ``CONTEXT_WINDOW`` / overlap / fuse values raise ``ValueError``; a malformed
    ``PADDED_FRAMES`` is ignored (falls through to the VRAM default).
    """
    win_env = os.environ.get(_LTX_HDR_CONTEXT_WINDOW_ENV)
    if win_env:
        # Explicit env override — parse exactly as before.
        try:
            window_px = int(win_env)
        except ValueError as exc:
            raise ValueError(
                f"{_LTX_HDR_CONTEXT_WINDOW_ENV} must be an int, got {win_env!r}"
            ) from exc
        if window_px <= 0:
            return None  # explicit 0/negative disables
        overlap_env = os.environ.get(_LTX_HDR_CONTEXT_OVERLAP_ENV, "16")
        try:
            overlap_px = int(overlap_env)
        except ValueError as exc:
            raise ValueError(
                f"{_LTX_HDR_CONTEXT_OVERLAP_ENV} must be an int, got {overlap_env!r}"
            ) from exc
        if overlap_px < 0:
            raise ValueError(f"{_LTX_HDR_CONTEXT_OVERLAP_ENV} must be >= 0, got {overlap_px}")
        fuse = os.environ.get(_LTX_HDR_CONTEXT_FUSE_ENV, "pyramid")
        if fuse not in ("pyramid", "flat"):
            raise ValueError(
                f"{_LTX_HDR_CONTEXT_FUSE_ENV} must be 'pyramid' or 'flat', got {fuse!r}"
            )
    else:
        # Default (LTX_HDR_CONTEXT_WINDOW unset): resolve the padded frame count.
        # Prefer the explicit argument (generate() has it as a local before this
        # runs — the LTX_HDR_PADDED_FRAMES env is only set later, inside the
        # generate() try block); fall back to that env only when no argument was
        # passed. At or below 121 padded frames the one-stage pass fits without
        # rolling windows, so context windowing is disabled (None + trace event).
        # Above 121, or when no frame count is available, use the VRAM-based
        # default.
        vram_gib = _detect_vram_gib()
        if padded_frames is None:
            pf_env = os.environ.get(_LTX_HDR_PADDED_FRAMES_ENV)
            if pf_env is not None:
                try:
                    padded_frames = int(pf_env)
                except ValueError:
                    padded_frames = None  # malformed -> fall through to VRAM default
        if padded_frames is not None and padded_frames <= 121:
            memory_trace.write_event(
                "hdr_context_default",
                "hdr_context_config",
                vram_gib=(vram_gib if vram_gib is not None else -1),
                disabled=True,
                padded_frames=padded_frames,
                reason="padded_frames <= 121; context windows not needed",
            )
            return None
        window_px, overlap_px = _default_context_window_px(vram_gib)
        fuse = "pyramid"
        memory_trace.write_event(
            "hdr_context_default",
            "hdr_context_config",
            vram_gib=(vram_gib if vram_gib is not None else -1),
            window_px=window_px,
            overlap_px=overlap_px,
            fuse=fuse,
            padded_frames=(padded_frames if padded_frames is not None else -1),
        )

    return _build_context_config(window_px, overlap_px, fuse)


def _static_windows(num_frames: int, window: int, overlap: int) -> list[tuple[int, int]]:
    """Static temporal windows over ``num_frames`` latent frames.

    Stride = ``window - overlap``; the final window is shifted to end exactly at
    the last latent frame (deduped if identical to the previous window).
    """
    if window >= num_frames:
        return [(0, num_frames)]
    stride = window - overlap
    windows: list[tuple[int, int]] = []
    start = 0
    while start + window <= num_frames:
        windows.append((start, start + window))
        if start + window == num_frames:
            break
        start += stride
    last = (max(num_frames - window, 0), num_frames)
    if not windows or windows[-1] != last:
        windows.append(last)
    return windows


def _context_frame_weights(
    length: int, overlap: int, fuse: str, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Per-frame fusion weights for one window. Always >= 1 (no zeros).

    ``flat`` -> all ones. ``pyramid`` -> trapezoid ramping over ``overlap``
    frames at each edge up to a plateau (1 at the very edge, ``overlap+1`` in the
    middle), so adjacent overlapping windows crossfade smoothly.
    """
    if fuse == "flat":
        return torch.ones(length, device=device, dtype=dtype)
    idx = torch.arange(length, device=device, dtype=dtype)
    ramp = float(max(int(overlap), 1))
    left = idx + 1.0
    right = (length - idx)
    cap = torch.full((length,), ramp + 1.0, device=device, dtype=dtype)
    return torch.minimum(torch.minimum(left, right), cap)


def _slice_context_window_state(
    state: LatentState,
    f0: int,
    f1: int,
    target_hw: int,
    cond_layout: tuple[tuple[int, int, int], ...],
) -> LatentState:
    """Slice a window ``[f0:f1]`` (latent frames) out of a full target+cond state.

    Target tokens occupy the first ``target_token_count`` tokens (frame-major,
    ``target_hw`` per frame). Each conditioning appended ``F_c*H_c*W_c`` tokens
    (in order); ``cond_layout`` carries ``(offset_tokens, hw_c, f_c)`` per cond.
    ``positions`` are token-aligned on dim 2 (3 = coords, 2 = tokens, last = 2).
    ``attention_mask`` is None on this path (no masking), so nothing to slice.
    """
    t0, t1 = f0 * target_hw, f1 * target_hw
    lat_parts: list[torch.Tensor] = [state.latent[:, t0:t1]]
    mask_parts: list[torch.Tensor] = [state.denoise_mask[:, t0:t1]]
    clean_parts: list[torch.Tensor] = [state.clean_latent[:, t0:t1]]
    pos_parts: list[torch.Tensor] = [state.positions[:, :, t0:t1, :]]
    for offset, hw_c, f_c in cond_layout:
        cf0, cf1 = min(f0, f_c), min(f1, f_c)
        c0, c1 = offset + cf0 * hw_c, offset + cf1 * hw_c
        lat_parts.append(state.latent[:, c0:c1])
        mask_parts.append(state.denoise_mask[:, c0:c1])
        clean_parts.append(state.clean_latent[:, c0:c1])
        pos_parts.append(state.positions[:, :, c0:c1, :])
    return LatentState(
        latent=torch.cat(lat_parts, dim=1),
        denoise_mask=torch.cat(mask_parts, dim=1),
        positions=torch.cat(pos_parts, dim=2),
        clean_latent=torch.cat(clean_parts, dim=1),
        attention_mask=None,
    )


def _make_context_video_loop(
    cfg: _ContextWindowConfig,
    conditionings: list[ConditioningItem],
    latent_frames: int,
    latent_h: int,
    latent_w: int,
) -> Any:
    """Build a denoising ``loop`` (sampler protocol signature) that temporally
    windows the transformer call and fuses target tokens back into the full state.

    The loop receives the full ``video_state`` built by ``DiffusionStage.run``
    (target tokens followed by appended conditioning tokens). For each step it
    slices per-window target+conditioning tokens, calls the transformer directly,
    keeps only the denoised target-window tokens, fuses them by frame weight, then
    steps the full state with upstream semantics (``post_process_latent`` +
    ``stepper.step``). The conditioning region of the full denoised estimate uses
    ``clean_latent``; for HDR (conditioning strength 1.0) ``post_process_latent``
    overwrites conditioning tokens with clean regardless, matching the non-windowed
    path exactly.
    """
    target_hw = latent_h * latent_w
    target_token_count = latent_frames * target_hw

    cond_layout: list[tuple[int, int, int]] = []
    offset = target_token_count
    for cond in conditionings:
        # Only reference-video conditionings append tokens in this path; HDR
        # produces VideoConditionByReferenceLatent exclusively. Skip anything
        # else (also narrows the type so ``.latent`` is visible to the checker).
        if not isinstance(cond, VideoConditionByReferenceLatent):
            continue
        cond_shape = cond.latent.shape
        f_c = int(cond_shape[2])
        hw_c = int(cond_shape[3]) * int(cond_shape[4])
        cond_layout.append((offset, hw_c, f_c))
        offset += f_c * hw_c
    cond_layout_t = tuple(cond_layout)
    windows = tuple(_static_windows(latent_frames, cfg.window_latent, cfg.overlap_latent))

    def loop(
        sigmas: torch.Tensor,
        video_state: LatentState | None,
        audio_state: LatentState | None,
        stepper: Any,
        transformer: Any,
        denoiser: Any,
    ) -> tuple[LatentState | None, LatentState | None]:
        if video_state is None:
            return video_state, audio_state
        v_ctx: Any = getattr(denoiser, "v_context", None)
        device = video_state.latent.device
        dtype = video_state.latent.dtype
        fused = torch.zeros_like(video_state.latent[:, :target_token_count])
        frame_w = torch.zeros(latent_frames, device=device, dtype=dtype)
        for step_idx in range(len(sigmas) - 1):
            fused.zero_()
            frame_w.zero_()
            sigma = sigmas[step_idx]
            for f0, f1 in windows:
                temp = _slice_context_window_state(
                    video_state, f0, f1, target_hw, cond_layout_t
                )
                modality = modality_from_latent_state(temp, v_ctx, sigma)
                denoised_video, _audio = transformer(
                    video=modality, audio=None, perturbations=None
                )
                win_len = f1 - f0
                win_target = denoised_video[:, : win_len * target_hw]
                w_local = _context_frame_weights(
                    win_len, cfg.overlap_latent, cfg.fuse, device, dtype
                )
                w_tok = w_local.repeat_interleave(target_hw)
                fused[:, f0 * target_hw : f1 * target_hw] += win_target * w_tok[None, :, None]
                frame_w[f0:f1] += w_local
            if not bool(torch.all(frame_w > 0)):
                raise ValueError(
                    "LTX_HDR_CONTEXT_WINDOW: a latent frame got zero fused weight; "
                    "reduce LTX_HDR_CONTEXT_OVERLAP or widen LTX_HDR_CONTEXT_WINDOW."
                )
            norm = frame_w.repeat_interleave(target_hw)
            fused = fused / norm[None, :, None]
            full_denoised = torch.cat(
                [fused, video_state.clean_latent[:, target_token_count:]], dim=1
            )
            processed = post_process_latent(
                full_denoised, video_state.denoise_mask, video_state.clean_latent
            )
            video_state = replace(
                video_state,
                latent=stepper.step(video_state.latent, processed, sigmas, step_idx),
            )
        return video_state, audio_state

    return loop


_DS_CONTEXT_PATCH_SENTINEL = "_ltx_desktop_context_loop_patch_applied"


def _install_diffusion_stage_context_patch() -> None:
    """Idempotently wrap ``DiffusionStage.__call__`` to inject a context loop.

    When a stage instance carries ``_ltx_desktop_context_loop`` (a
    ``_ContextWindowConfig``) and the caller did not pass an explicit ``loop``,
    the wrapper builds a context-video loop from the call's shape + conditionings
    and injects it. Otherwise it delegates exactly. This is the only mechanism
    that lets stage-1 context-windowing reach inside upstream
    ``HDRICLoraPipeline.__call__`` (which calls ``self.stage_1(...)`` without a
    ``loop=`` argument) without editing vendored code.
    """
    original = DiffusionStage.__call__
    if getattr(original, _DS_CONTEXT_PATCH_SENTINEL, False):
        return

    @functools.wraps(original)
    def _patched(self: Any, *args: Any, **kwargs: Any) -> Any:
        cfg = getattr(self, "_ltx_desktop_context_loop", None)
        if isinstance(cfg, _ContextWindowConfig) and kwargs.get("loop") is None:
            frames = kwargs.get("frames")
            width = kwargs.get("width")
            height = kwargs.get("height")
            spec = kwargs.get("video")
            if frames is not None and width is not None and height is not None and spec is not None:
                conditionings = list(getattr(spec, "conditionings", []) or [])
                latent_frames = (int(frames) - 1) // 8 + 1
                latent_h = int(height) // 32
                latent_w = int(width) // 32
                kwargs["loop"] = _make_context_video_loop(
                    cfg, conditionings, latent_frames, latent_h, latent_w
                )
        return original(self, *args, **kwargs)

    setattr(_patched, _DS_CONTEXT_PATCH_SENTINEL, True)
    DiffusionStage.__call__ = _patched  # type: ignore[method-assign]


_install_diffusion_stage_context_patch()


def _run_stage2_phase_context(
    pipeline: Any,
    transformer: object,
    latent: torch.Tensor,
    conditionings: list[ConditioningItem],
    tiling: TileCountConfig,
    sigmas: torch.Tensor,
    v_ctx: torch.Tensor,
    frame_rate: float,
    seed: int,
    cfg: _ContextWindowConfig,
) -> torch.Tensor:
    """Context-windowed stage-2 phase (mirrors upstream ``_run_stage2_phase``).

    Identical spatial tiling via ``VideoModalityTilingHelper``, but each spatial
    tile's ``stage_2.run`` is given a temporal context-window ``loop``. Stage-2
    tiling is spatial only (single temporal tile, already clamped by the caller
    in ``__call__``), so the context loop owns temporal windowing — there is no
    double temporal tiling.
    """
    batch, n_channels, n_frames, n_height, n_width = latent.shape
    full_shape = VideoLatentShape(
        batch=batch, channels=n_channels, frames=n_frames, height=n_height, width=n_width
    )
    full_tools = VideoLatentTools(VideoLatentPatchifier(patch_size=1), full_shape, frame_rate)
    helper = VideoModalityTilingHelper(tiling, full_tools)

    ref_initial = full_tools.create_initial_state(device=pipeline.device, dtype=pipeline.dtype)
    ref_modality = modality_from_latent_state(ref_initial, v_ctx, sigmas[0])
    n_gen = full_tools.target_shape.token_count()
    blend_output = torch.zeros(batch, n_gen, n_channels, device=pipeline.device, dtype=pipeline.dtype)
    patchifier = VideoLatentPatchifier(patch_size=1)
    df = pipeline.reference_downscale_factor

    for tile_idx, tile in enumerate(helper.tiles):
        _, ctx = helper.tile_modality(ref_modality, tile, normalize_positions=True)
        frame_s, height_s, width_s = tile.in_coords
        tile_h = height_s.stop - height_s.start
        tile_w = width_s.stop - width_s.start
        tile_f = frame_s.stop - frame_s.start

        tile_conditionings: list[ConditioningItem] = [
            VideoConditionByReferenceLatent(
                latent=cond.latent[
                    :,
                    :,
                    frame_s,
                    slice(height_s.start // df, height_s.stop // df),
                    slice(width_s.start // df, width_s.stop // df),
                ].to(device=pipeline.device, dtype=pipeline.dtype),
                downscale_factor=cond.downscale_factor,
                strength=cond.strength,
            )
            for cond in conditionings
            if isinstance(cond, VideoConditionByReferenceLatent)
        ]

        tile_loop = _make_context_video_loop(cfg, tile_conditionings, tile_f, tile_h, tile_w)
        tile_video_state, _ = pipeline.stage_2.run(
            transformer=transformer,
            denoiser=SimpleDenoiser(v_ctx, None),
            sigmas=sigmas,
            noiser=GaussianNoiser(
                generator=torch.Generator(device=pipeline.device).manual_seed(seed + tile_idx)
            ),
            width=tile_w * 32,
            height=tile_h * 32,
            frames=(tile_f - 1) * 8 + 1,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_ctx,
                conditionings=tile_conditionings,
                noise_scale=sigmas[0].item(),
                initial_latent=latent[:, :, frame_s, height_s, width_s].to(
                    device=pipeline.device, dtype=pipeline.dtype
                ),
            ),
            loop=tile_loop,
        )
        assert tile_video_state is not None  # video modality is always present here
        tile_tokens = patchifier.patchify(tile_video_state.latent)
        blend_output = helper.blend(tile_tokens, tile, ctx, blend_output)

    return full_tools.unpatchify(replace(ref_initial, latent=blend_output)).latent


class LTXHdrIcLoraPipeline(HDRICLoraPipeline):
    """Thin app wrapper over the upstream HDR IC-LoRA two-stage pipeline.

    The class attribute ``pipeline_kind`` discriminates HDR pipeline state
    from other GPU-slot pipeline types. Only the conditioning-load path is
    overridden; all generation math is upstream's.
    """

    pipeline_kind: ClassVar[Literal["hdr_ic_lora"]] = "hdr_ic_lora"

    def __init__(  # noqa: PLR0913
        self,
        *,
        checkpoint_path: str | tuple[str, ...],
        upsampler_path: str,
        loras: tuple[LoraPathStrengthAndSDOps, ...],
        text_embeddings_path: str,
        device: torch.device,
        quantization: QuantizationPolicy | None,
        offload_mode: OffloadMode,
        transformer_format: TransformerFormat,
        components: ResolvedLtxComponents | None,
        hdr_lora_config: HdrLoraConfig | None,
        registry: Registry | None = None,
        # Encode-tiled decision threshold. Encode tiling is OFF by default
        # (LTX_HDR_VAE_TILE_SIZE unset -> None), so this threshold is only
        # consulted when encode tiling is explicitly enabled.
        tiled_vae_encode_pixel_threshold: int = 256 * 256,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> None:
        """Component-aware initializer that reproduces upstream block wiring.

        Does NOT call ``HDRICLoraPipeline.__init__`` (which is monolith-only).
        Builds the exact upstream block attributes the inherited
        ``__call__``/``_run_stage2_phase``/``_decode_video``/``hdr_transform``/
        ``reference_downscale_factor`` rely on (``device``, ``dtype``,
        ``_tiled_vae_encode_threshold``, ``text_embeddings``,
        ``image_conditioner``, ``stage_1``, ``stage_2``, ``upsampler``,
        ``video_decoder``, ``_hdr_config``), but with app-resolved LoRAs,
        quantization policy, and post-build component patching threaded in by
        :meth:`create`. ``_create_conditionings`` remains overridden below.
        """
        # NOTE: deliberately not calling super().__init__(); upstream's ctor is
        # monolith-only and discards components/distilled_lora_path/gemma_root.
        self.device = device
        self.dtype = torch.bfloat16
        self._tiled_vae_encode_threshold = tiled_vae_encode_pixel_threshold

        # Scene embeddings replace text prompt encoding (HDR is video-only;
        # upstream loads both ``video_context`` and ``audio_context`` tensors).
        # ``handle`` is typed Any: the installed safetensors stub types
        # ``safe_open(...)`` as a reader without ``get_tensor`` (see
        # ``hdr_scene_embeddings.py`` for the same Any-handle pattern).
        logger.info("[HDR IC-LoRA] Loading scene embeddings from %s", text_embeddings_path)
        handle: Any
        with safe_open(text_embeddings_path, framework="pt", device=str(self.device)) as handle:
            self.text_embeddings = (
                handle.get_tensor("video_context"),
                handle.get_tensor("audio_context"),
            )

        self.image_conditioner = ImageConditioner(checkpoint_path, self.dtype, self.device, registry=registry)  # type: ignore[arg-type]  # ltx_pipelines accepts tuple per split/GGUF M5 spec
        with memory_trace.phase("diffusion_stage_build:hdr_stage_1"):
            self.stage_1 = DiffusionStage(
                checkpoint_path,  # type: ignore[arg-type]  # ltx_pipelines accepts tuple per split/GGUF M5 spec
                self.dtype,
                self.device,
                loras=loras,
                quantization=quantization,
                registry=registry,
                offload_mode=offload_mode,
            )
        with memory_trace.phase("diffusion_stage_build:hdr_stage_2"):
            self.stage_2 = DiffusionStage(
                checkpoint_path,  # type: ignore[arg-type]  # ltx_pipelines accepts tuple per split/GGUF M5 spec
                self.dtype,
                self.device,
                loras=loras,
                quantization=quantization,
                registry=registry,
                offload_mode=offload_mode,
            )
        # Phase 3B: stamp the memory plan onto both DiffusionStages so the
        # block-offload build patch can read it when the transformer is built
        # lazily at generation time. ``memory_plan`` is a dynamic attribute
        # (read via ``getattr`` in the patch); not declared on DiffusionStage.
        if memory_plan is not None:
            self.stage_1.memory_plan = memory_plan  # type: ignore[attr-defined]
            self.stage_2.memory_plan = memory_plan  # type: ignore[attr-defined]
        self.upsampler = VideoUpsampler(
            checkpoint_path,  # type: ignore[arg-type]  # ltx_pipelines accepts tuple per split/GGUF M5 spec
            upsampler_path,
            self.dtype,
            self.device,
            registry=registry,
        )
        self.video_decoder = VideoDecoder(checkpoint_path, self.dtype, self.device, registry=registry)  # type: ignore[arg-type]  # ltx_pipelines accepts tuple per split/GGUF M5 spec

        # HDR LoRA metadata: explicit override, else None → inherited
        # ``hdr_transform``/``reference_downscale_factor`` properties yield the
        # upstream defaults ("logc3" / 1), matching read_hdr_lora_config absence.
        self._hdr_config = hdr_lora_config
        if self._hdr_config is not None:
            logger.info("[HDR IC-LoRA] HDR mode enabled (%s decode)", self._hdr_config.hdr_transform)

        # Post-build component patching is owned by the pipeline (it mutates
        # the block builders created above), not by the handler.
        _install_hdr_component_patches(self, checkpoint_path, transformer_format, components)

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
        memory_plan: LocalMemoryPlan | None = None,
    ) -> "LTXHdrIcLoraPipeline":
        """Construct the HDR IC-LoRA pipeline from resolved components.

        Accepts the effective builder ``checkpoint_path``:
        - monolith official distilled: a single path string;
        - split/Kijai/GGUF: a tuple from
          ``ResolvedLtxComponents.checkpoint_paths_for_filtered_builders``.

        HDR now accepts dev, distilled, split, Kijai, and GGUF component
        builds. LoRA order/strength is deterministic:
        - ``base_family == "dev"``: distilled LoRA @ 0.5 first, then HDR LoRA
          @ 1.0 (both with ``LTXV_LORA_COMFY_RENAMING_MAP``);
        - ``base_family == "distilled"``: HDR LoRA @ 1.0 only;
        - ``base_family == "unknown"``: rejected before any heavy load.

        Quantization policy (offload is owned by the memory plan — Phase 2):
        - GGUF → quantization None, GGUF loader installed post-build;
        - split safetensors with a video VAE sidecar on CUDA → Kijai FP8 policy;
        - monolith safetensors on CUDA → upstream ``build_policy``.

        Offload policy (Phase 2): when ``memory_plan`` is provided, its
        ``offload_mode`` is trusted exactly — no internal GGUF-force-NONE and no
        split-coerce-NONE/default-to-CPU (the plan owns residency strategy).
        ``memory_plan=None`` (transitional handler path) falls back to the
        caller's ``offload_mode`` (default ``NONE``). DISK offload on
        componentized split builds is still rejected as a hard guard.
        """
        # gemma_root is unused: scene embeddings replace text prompt encoding.
        del gemma_root

        if base_family == "unknown":
            raise ValueError(
                "HDR IC-LoRA requires a 'dev' or 'distilled' base family; "
                f"base_family={base_family!r} is not supported."
            )
        if base_family not in ("dev", "distilled"):
            raise ValueError(
                f"HDR IC-LoRA base_family={base_family!r} is not supported "
                "(expected 'dev' or 'distilled')."
            )
        if not scene_embeddings_path:
            raise ValueError("scene_embeddings_path is required for HDR IC-LoRA")

        is_gguf = transformer_format == "gguf"
        is_componentized_split = (
            isinstance(checkpoint_path, tuple)
            and components is not None
            and (
                components.video_vae_path is not None
                or components.audio_vae_path is not None
                or components.text_projection_path is not None
                or components.embeddings_connector_path is not None
            )
        )

        # Phase 2: the memory plan owns the offload decision. When provided,
        # trust its offload_mode exactly — no internal GGUF-force-NONE and no
        # split-coerce-NONE/default-to-CPU (the plan's strategy covers residency).
        # Transitional handler path (memory_plan=None) uses the caller's
        # offload_mode (default NONE) until the handler passes explicit plans.
        if memory_plan is not None:
            resolved_offload = memory_plan.offload_mode
        elif offload_mode is not None:
            resolved_offload = offload_mode
        else:
            resolved_offload = OffloadMode.NONE

        # Split safetensors: DISK offload is not validated for the shared
        # streaming patch; reject it. This is a hard guard, not a coercion.
        if is_componentized_split and resolved_offload == OffloadMode.DISK:
            raise ValueError(
                "HDR IC-LoRA componentized split builds do not support DISK offload; "
                "use OffloadMode.NONE or OffloadMode.CPU."
            )

        # Deterministic LoRA order/strength. Do NOT use DEFAULT_LORA_STRENGTH;
        # exact strengths are required (distilled @ 0.5, HDR @ 1.0).
        resolved_hdr_lora_path = str(Path(hdr_lora_path).resolve())
        loras: list[LoraPathStrengthAndSDOps] = []
        if base_family == "dev":
            if not distilled_lora_path:
                raise ValueError(
                    "distilled_lora_path is required for HDR IC-LoRA with a 'dev' base family."
                )
            loras.append(LoraPathStrengthAndSDOps(distilled_lora_path, 0.5, LTXV_LORA_COMFY_RENAMING_MAP))
        loras.append(LoraPathStrengthAndSDOps(resolved_hdr_lora_path, 1.0, LTXV_LORA_COMFY_RENAMING_MAP))

        # Quantization policy.
        supports_fp8 = device_supports_fp8(device)
        quantization: QuantizationPolicy | None
        if is_gguf:
            quantization = None
        elif is_componentized_split and supports_fp8:
            from services.patches.gguf_loader_fix import kijai_fp8_quantization_policy

            quantization = kijai_fp8_quantization_policy()
        elif supports_fp8:
            from ltx_core.quantization.fp8_cast import build_policy

            single = checkpoint_path[0] if isinstance(checkpoint_path, tuple) else checkpoint_path
            quantization = build_policy(single)
        else:
            quantization = None

        # Preserve upstream HDR LoRA metadata behavior: read from the LoRA
        # safetensors metadata; absence (None) falls back to the upstream
        # default transform/downscale via the inherited properties.
        hdr_lora_config = read_hdr_lora_config(resolved_hdr_lora_path)

        resolved_device = device if isinstance(device, torch.device) else torch.device(device)
        with memory_trace.phase("hdr_create"):
            return LTXHdrIcLoraPipeline(
                checkpoint_path=checkpoint_path,
                upsampler_path=upsampler_path,
                loras=tuple(loras),
                text_embeddings_path=scene_embeddings_path,
                device=resolved_device,
                quantization=quantization,
                offload_mode=resolved_offload,
                transformer_format=transformer_format,
                components=components,
                hdr_lora_config=hdr_lora_config,
                memory_plan=memory_plan,
            )

    def _create_conditionings(  # type: ignore[override]
        self,
        video_conditioning: list[tuple[str, float]],
        height: int,
        width: int,
        num_frames: int,
        video_encoder: VideoEncoder,
        tiling_config: TilingConfig | None = None,
        high_quality_hdr: bool = False,
    ) -> list[ConditioningItem]:
        """Override upstream conditioning load to use in-memory decoded frames.

        Mirrors upstream's per-frame transform exactly
        (``resize_and_reflect_pad`` for ``ResizeMode.REFLECT_PAD`` →
        ``/255`` clamp → ``LogC3().compress_ldr`` → ``to_vae_range`` →
        dtype/device), the same tiled-vs-direct encode decision, and the same
        ``VideoConditionByReferenceLatent`` downscale/strength logic. Only the
        source of frames differs: all decoded source frames in source order
        plus duplicate-final-frame padding to ``8n + 1`` — never a recompressed
        temp video.
        """
        with memory_trace.phase("hdr_conditionings"):
            if high_quality_hdr:
                raise NotImplementedError(
                    "high_quality_hdr=True is not supported by the LTX-Desktop HDR wrapper "
                    "(app invariant: use all source frames + duplicate-final-frame padding)."
                )

            scale = self.reference_downscale_factor
            if scale != 1 and (height % scale != 0 or width % scale != 0):
                raise ValueError(
                    f"Output dimensions ({height}x{width}) must be divisible by "
                    f"reference_downscale_factor ({scale})"
                )
            ref_height = height // scale
            ref_width = width // scale

            logc3 = LogC3()
            conditionings: list[ConditioningItem] = []

            # Infer the video encoder's actual input device/dtype from its own
            # parameters/buffers (preferred over self.dtype). A Kijai FP8/sidecar
            # VAE keeps float32 conv weight/bias while the pipeline dtype is bf16,
            # so blindly casting conditioning frames to self.dtype mismatches the
            # encoder conv3d (RuntimeError: Input type (BFloat16) and bias type
            # (float) should be the same). Feed the encoder what its bias expects.
            encoder_device, encoder_dtype = _module_device_dtype(
                video_encoder, fallback_device=self.device, fallback_dtype=self.dtype
            )

            # Transfer tagged non-Rec.709 SDR source video to the Rec.709 model
            # domain before the per-frame LogC3/VAE transform (untagged/bt709 is
            # an exact passthrough). frame_cap=None decodes all source frames.
            from services.exr_input import iter_video_frames_to_model_domain

            for video_path, strength in video_conditioning:
                # Decode ALL source frames (frame_cap=None). Stream decoded frames
                # straight into one preallocated CPU tensor instead of materializing
                # source_frames / padded_frames / transformed lists, so no full frame
                # list stays alive during the VAE encode (peak-memory mitigation).
                frame_iter = iter_video_frames_to_model_domain(
                    video_path, frame_cap=None, device=self.device,
                )

                def _transform(raw_frame: torch.Tensor) -> torch.Tensor:
                    # Same per-frame transform as upstream load_video_conditioning_hdr
                    # with ResizeMode.REFLECT_PAD (resize -> /255 clamp -> LogC3 ->
                    # to_vae_range). The single encoder device/dtype move happens
                    # once on the whole tensor below, not per frame.
                    resized = resize_and_reflect_pad(raw_frame.to(torch.float32), ref_height, ref_width)
                    ldr = (resized / 255.0).clamp(0.0, 1.0)
                    return to_vae_range(logc3.compress_ldr(ldr))

                # Pull the first frame to learn the per-frame shape and preallocate.
                try:
                    first_t = _transform(next(frame_iter))
                except StopIteration:
                    raise ValueError(
                        f"HDR conditioning source video decoded zero frames: {video_path!r}"
                    )

                # Each transformed frame is (1, C, 1, H, W); cat along dim=2 yields
                # (1, C, T, H, W). Preallocate the padded tensor once on CPU.
                video = torch.empty(
                    (*first_t.shape[:2], num_frames, *first_t.shape[3:]),
                    dtype=first_t.dtype,
                    device=torch.device("cpu"),
                )
                video[:, :, 0:1] = first_t
                last_t = first_t
                idx = 1
                source_count = 1
                for raw_frame in frame_iter:
                    t = _transform(raw_frame)
                    if idx < num_frames:
                        video[:, :, idx:idx + 1] = t
                    last_t = t
                    idx += 1
                    source_count += 1

                padded_count = _padded_frame_count(source_count)
                if padded_count != num_frames:
                    raise ValueError(
                        "HDR conditioning padded frame count does not match the requested "
                        f"num_frames: decoded {source_count} -> padded {padded_count}, "
                        f"requested {num_frames}."
                    )

                # Pad with duplicate copies of the genuine final transformed frame only.
                while idx < num_frames:
                    video[:, :, idx:idx + 1] = last_t
                    idx += 1

                # Phase 3A: trace the whole-tensor GPU move -> VAE encode peak.
                memory_trace.snapshot("hdr_conditionings:before_vae_encode")

                # Upstream tiled_encode accepts CPU video and moves each tile to
                # the model device internally, so the tiled path keeps the full
                # conditioning tensor off CUDA (avoids OOM materializing it). The
                # whole-tensor device/dtype move is kept only for the non-tiled path.
                use_tiled_encode = (
                    tiling_config is not None
                    and ref_height * ref_width > self._tiled_vae_encode_threshold
                )
                _enc_tile_size = 0
                if (
                    use_tiled_encode
                    and tiling_config is not None
                    and tiling_config.spatial_config is not None
                ):
                    _enc_tile_size = tiling_config.spatial_config.tile_size_in_pixels
                _enc_timing = _hdr_timing_enabled()
                _enc_t0 = time.perf_counter() if _enc_timing else 0.0
                fallback_used = False
                if use_tiled_encode:
                    encoded_video = video_encoder.tiled_encode(video, tiling_config)
                else:
                    # Direct encode. Keep ``video`` (CPU) alive for an OOM-fallback
                    # tiled retry; encode from a separate GPU-resident tensor.
                    video_enc = video.to(device=encoder_device, dtype=encoder_dtype)
                    try:
                        encoded_video = video_encoder(video_enc)
                    except torch.cuda.OutOfMemoryError:
                        # Only fall back when no explicit encode tiling was
                        # requested (LTX_HDR_VAE_TILE_SIZE unset/0). If the user
                        # set an explicit tiling policy, respect it and re-raise.
                        if tiling_config is not None:
                            raise
                        memory_trace.write_event(
                            "hdr_vae_encode_oom_fallback",
                            "hdr_conditionings",
                            ref_height=ref_height,
                            ref_width=ref_width,
                            fallback_tile=512,
                        )
                        logger.warning(
                            "[HDR IC-LoRA] VAE direct conditioning encode OOM; "
                            "retrying tiled encode (512px / 64px overlap)."
                        )
                        del video_enc
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()
                        fallback_tiling = TilingConfig(
                            spatial_config=SpatialTilingConfig(
                                tile_size_in_pixels=512,
                                tile_overlap_in_pixels=64,
                            ),
                            temporal_config=TemporalTilingConfig(
                                tile_size_in_frames=_HDR_TEMPORAL_TILE_FRAMES,
                                tile_overlap_in_frames=_HDR_TEMPORAL_TILE_OVERLAP,
                            ),
                        )
                        # tiled_encode accepts CPU video and stages each tile to
                        # the model device itself, so peak VRAM stays bounded.
                        encoded_video = video_encoder.tiled_encode(video, fallback_tiling)
                        fallback_used = True
                        _enc_tile_size = 512
                memory_trace.snapshot("hdr_conditionings:after_vae_encode")
                if _enc_timing:
                    memory_trace.write_event(
                        "hdr_timing",
                        "vae_encode",
                        encode_s=round(time.perf_counter() - _enc_t0, 3),
                        tiled=use_tiled_encode or fallback_used,
                        fallback_tiled=fallback_used,
                        tile_size=_enc_tile_size,
                    )

                conditionings.append(
                    VideoConditionByReferenceLatent(
                        latent=encoded_video,
                        downscale_factor=scale,
                        strength=strength,
                    )
                )

            if video_conditioning:
                logger.info("[HDR IC-LoRA] Added %d video conditioning(s)", len(video_conditioning))

            return conditionings

    def __call__(  # type: ignore[override]  # noqa: PLR0913
        self,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        video_conditioning: list[tuple[str, float]],
        tiling_config: TilingConfig | None = None,
        high_quality_hdr: bool = False,
        **_kwargs: Any,
    ) -> torch.Tensor:
        """One-stage HDR generation: full-res /32 latent → sample → decode.

        Overrides upstream two-stage ``__call__`` (which ran stage 1 at half-res,
        upscaled, then re-sampled at full-res). This runs a single stage-1 pass at
        the full aligned resolution, then decodes directly — no upsampler, no
        stage 2, no second conditioning encode. ``stage_2``/``upsampler`` remain
        constructed by ``__init__`` (component-patch helpers expect the attrs) but
        are never called. Returns a linear HDR float tensor ``[f, h, w, c]``.
        """
        memory_trace.write_event("hdr_one_stage", "LTXHdrIcLoraPipeline.__call__")

        if high_quality_hdr:
            gen_num_frames = 2 * num_frames - 1
        else:
            gen_num_frames = num_frames

        gen_w, gen_h, crop_w, crop_h = align_resolution(
            width, height, ResizeMode.REFLECT_PAD, divisor=ALIGNMENT_DIVISOR
        )
        if gen_h < MIN_RESOLUTION or gen_w < MIN_RESOLUTION:
            raise ValueError(
                f"Resolution ({width}x{height}) is too small after alignment "
                f"(got {gen_w}x{gen_h}, need at least {MIN_RESOLUTION}x{MIN_RESOLUTION})."
            )
        needs_crop = crop_w != gen_w or crop_h != gen_h

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        video_context, _ = self.text_embeddings

        # Single conditioning encode at full aligned resolution (upstream encoded
        # twice: once at half-res for stage 1, once at full-res for stage 2).
        conditionings = self.image_conditioner(
            lambda enc: self._create_conditionings(
                video_conditioning=video_conditioning,
                height=gen_h,
                width=gen_w,
                video_encoder=enc,
                num_frames=gen_num_frames,
                tiling_config=tiling_config,
                high_quality_hdr=high_quality_hdr,
            )
        )

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)

        # Single full-resolution stage-1 pass (upstream used gen_w//2, gen_h//2).
        video_state, _ = self.stage_1(
            denoiser=SimpleDenoiser(video_context, None),
            sigmas=stage_1_sigmas,
            noiser=noiser,
            width=gen_w,
            height=gen_h,
            frames=gen_num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=video_context,
                conditionings=conditionings,
            ),
        )
        assert video_state is not None  # HDR is video-only; audio is skipped

        # Direct decode — no upsampler, no stage-2 re-sampling.
        crop_size = (crop_w, crop_h) if needs_crop else None
        return self._decode_video(
            video_state.latent,
            tiling_config,
            generator,
            crop_size,
            high_quality_hdr=high_quality_hdr,
        )

    def _run_stage2_phase(  # type: ignore[override]
        self,
        transformer: object,
        latent: torch.Tensor,
        conditionings: list[ConditioningItem],
        tiling: TileCountConfig,
        sigmas: torch.Tensor,
        v_ctx: torch.Tensor,
        frame_rate: float,
        seed: int,
    ) -> torch.Tensor:
        """Stage-2 phase; context-windowed when ``_ltx_desktop_context_config`` is set.

        Non-context path delegates to upstream exactly (no behavior change). In
        context mode, spatial tiling is reproduced (upstream clamps the tiling to
        a single temporal tile in ``__call__``) and each spatial tile's
        ``stage_2.run`` gets a temporal context-window loop, so temporal windowing
        lives only in the loop — no double temporal tiling.
        """
        cfg = getattr(self, "_ltx_desktop_context_config", None)
        _s2_timing = _hdr_timing_enabled()
        _s2_t0 = time.perf_counter() if _s2_timing else 0.0
        if isinstance(cfg, _ContextWindowConfig):
            result = _run_stage2_phase_context(
                self, transformer, latent, conditionings, tiling, sigmas, v_ctx, frame_rate, seed, cfg
            )
        else:
            result = super()._run_stage2_phase(
                transformer, latent, conditionings, tiling, sigmas, v_ctx, frame_rate, seed
            )
        if _s2_timing:
            memory_trace.write_event(
                "hdr_timing",
                "stage2_phase",
                phase_s=round(time.perf_counter() - _s2_t0, 3),
                context_windowed=isinstance(cfg, _ContextWindowConfig),
            )
        return result

    def _decode_video(  # type: ignore[override]
        self,
        latent: torch.Tensor,
        tiling_config: TilingConfig | None,  # noqa: ARG002  # decode ignores the encode tiling config forwarded by upstream __call__
        generator: torch.Generator,
        crop_size: tuple[int, int] | None = None,
        *,
        high_quality_hdr: bool = False,
    ) -> torch.Tensor:
        """Chunked VAE decode + HDR postprocess — bounds peak VRAM.

        Upstream ``_decode_video`` tiles the VAE decode but then concatenates ALL
        decoded frames on GPU and runs ``apply_hdr_decode_postprocess`` over the
        full video, whose ``LogC3.decompress`` temporaries scale with total frame
        count (OOM on 201-frame HDR). This override decodes chunk by chunk using
        :func:`_hdr_decode_tiling_config`, postprocesses each chunk separately,
        and moves it to CPU immediately, so the HDR-postprocess temporary is
        bounded to one decode tile. The forwarded encode tiling config is ignored
        (decode always uses its own). ``generate()``'s later ``.cpu()`` is a
        harmless no-op on the already-CPU result.
        """
        from ltx_core.hdr import apply_hdr_decode_postprocess

        # Decode is always tiled with its OWN config; the forwarded encode config
        # is intentionally ignored so encode (off by default) and decode (always
        # on) stay fully independent.
        decode_tiling, dec_tile_size, dec_tile_overlap = _hdr_decode_tiling_config()
        _dec_timing = _hdr_timing_enabled()
        _dec_t0 = time.perf_counter() if _dec_timing else 0.0
        memory_trace.snapshot("hdr_decode:before_chunked_postprocess")

        cpu_chunks: list[torch.Tensor] = []
        global_offset = 0  # cumulative raw frame count, for high_quality_hdr parity
        for chunk in self.video_decoder(latent.float(), decode_tiling, generator):
            chunk = chunk.float()
            # [f,h,w,c] -> [1,c,f,h,w] (equivalent to einops "f h w c -> 1 c f h w")
            decoded = chunk.unsqueeze(0).permute(0, 4, 1, 2, 3).contiguous()
            hdr = apply_hdr_decode_postprocess(
                decoded, transform=cast(Literal["logc3"], self.hdr_transform)
            )
            del decoded
            # [c,f,h,w] -> [f,h,w,c] (equivalent to einops "c f h w -> f h w c")
            out = hdr[0].permute(1, 2, 3, 0)
            del hdr
            if crop_size is not None:
                # crop_size is (width, height) -> crop H (dim 1) then W (dim 2)
                out = out[:, : crop_size[1], : crop_size[0], :]
            chunk_frames = int(out.shape[0])
            if high_quality_hdr:
                # Keep every other GLOBAL frame (matches upstream's full-tensor
                # out[::2]); per-chunk parity depends on the cumulative offset.
                out = out[(global_offset % 2) :: 2]
            global_offset += chunk_frames
            cpu_chunks.append(out.cpu())
            del out, chunk
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        memory_trace.snapshot("hdr_decode:after_chunked_postprocess")
        if _dec_timing:
            memory_trace.write_event(
                "hdr_timing",
                "vae_decode",
                decode_s=round(time.perf_counter() - _dec_t0, 3),
                tile_size=dec_tile_size,
                tile_overlap=dec_tile_overlap,
            )
        return torch.cat(cpu_chunks, dim=0)

    @torch.inference_mode()
    def generate(
        self,
        source_video_path: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,  # noqa: ARG002  # HDR is source-driven; padded count is computed from the source.
        frame_rate: float,
        output_path: str,
        output_format: OutputFormat = OutputFormat.EXR_ZIP_HALF,
        encoder: MediaEncoder | None = None,  # noqa: ARG002  # unused: upstream writers own HDR output, not the app encoder.
        proxy_path: str | None = None,
        input_colorspace: ColorSpace | None = None,  # noqa: ARG002  # HDR source is 8-bit SDR; no colorspace detection.
        on_progress: Callable[[float], None] | None = None,
        on_phase_update: Callable[[str, str | None], None] | None = None,
    ) -> None:
        """Run the official HDR IC-LoRA two-stage flow on the source video.

        Computes the padded frame count from the decoded source video and
        delegates generation to upstream ``__call__`` (stage 1 / upsampler /
        stage 2 / decode) unchanged. Then writes the returned linear HDR
        tensor as the EXR primary sequence (no EOTF / tonemap / clamp) and,
        after the primary, the SDR proxy MP4.
        """

        def _phase(phase: str, detail: str | None = None) -> None:
            if on_phase_update is not None:
                on_phase_update(phase, detail)

        with memory_trace.phase("hdr_generate"):
            # Env-gated coarse phase timing (debug/harness knob, NOT production
            # UI policy). ~free when LTX_HDR_TIMING is unset: checkpoints and the
            # summary are skipped; only a single env read + 0.0 assignments remain.
            timing = os.environ.get(_LTX_HDR_TIMING_ENV) == "1"
            _t_total = time.perf_counter() if timing else 0.0
            _t0 = time.perf_counter() if timing else 0.0

            # Count decoded frames without holding the full source-frame list alive
            # through the heavy upstream self(...) call (peak-memory mitigation).
            source_count = sum(
                1 for _ in decode_video_by_frame(path=source_video_path, frame_cap=None, device=self.device)
            )
            if source_count == 0:
                raise ValueError(
                    f"HDR source video decoded zero frames: {source_video_path!r}"
                )
            padded_num_frames = _padded_frame_count(source_count)
            _t_decode = (time.perf_counter() - _t0) if timing else 0.0
            _t0 = time.perf_counter() if timing else _t0

            # Phase 3A: move the returned GPU video tensor to CPU before any
            # EXR/proxy writing so the large HDR frame tensor is not pinned on
            # the GPU through file I/O. Drop the GPU reference and drain the
            # cache so the resident-bytes trace dips visibly at the handoff.
            # Encode tiling policy: LTX_HDR_VAE_TILE_SIZE overrides; unset -> a
            # runtime policy from VRAM + padded_num_frames (31GiB: no tile <=121f,
            # tile512 >121f; lower VRAM tiles earlier). Decode tiling is always on
            # and resolved separately inside _decode_video.
            tiling_config, threshold_override = _hdr_tiling_overrides(padded_num_frames)
            # Context-window config: LTX_HDR_CONTEXT_WINDOW explicitly overrides;
            # unset -> disabled (None) for <=121 padded frames (the common case),
            # VRAM-based rolling window only above 121. padded_num_frames is passed
            # directly — the LTX_HDR_PADDED_FRAMES env is not set until inside the
            # try block below.
            context_cfg = _hdr_context_config(padded_num_frames)
            if context_cfg is not None:
                memory_trace.write_event(
                    "hdr_context_window",
                    "hdr_generate",
                    window_latent=context_cfg.window_latent,
                    overlap_latent=context_cfg.overlap_latent,
                    fuse=context_cfg.fuse,
                    padded_num_frames=padded_num_frames,
                )
            saved_threshold = self._tiled_vae_encode_threshold
            if threshold_override is not None:
                self._tiled_vae_encode_threshold = threshold_override
            prev_s1_loop = getattr(self.stage_1, "_ltx_desktop_context_loop", None)
            prev_ctx_cfg = getattr(self, "_ltx_desktop_context_config", None)
            if context_cfg is not None:
                # Stage 1: read by the DiffusionStage.__call__ wrapper to inject
                # the context loop. Stage 2: read by _run_stage2_phase above.
                self.stage_1._ltx_desktop_context_loop = context_cfg  # type: ignore[attr-defined]
                self._ltx_desktop_context_config = context_cfg
            # Publish the padded frame count for the resident-block policy, which
            # reads it (via LTX_HDR_PADDED_FRAMES) at transformer-build time inside
            # self(...). Restored to its prior value in the finally below.
            _prev_pf_env = os.environ.get(_LTX_HDR_PADDED_FRAMES_ENV)
            os.environ[_LTX_HDR_PADDED_FRAMES_ENV] = str(padded_num_frames)
            _phase("inference", "HDR one-stage sampling + tiled decode")
            try:
                video_gpu: torch.Tensor = self(
                    seed=seed,
                    height=height,
                    width=width,
                    num_frames=padded_num_frames,
                    frame_rate=frame_rate,
                    video_conditioning=[(source_video_path, 1.0)],
                    high_quality_hdr=False,
                    tiling_config=tiling_config,
                )
            finally:
                if _prev_pf_env is None:
                    os.environ.pop(_LTX_HDR_PADDED_FRAMES_ENV, None)
                else:
                    os.environ[_LTX_HDR_PADDED_FRAMES_ENV] = _prev_pf_env
                if threshold_override is not None:
                    self._tiled_vae_encode_threshold = saved_threshold
                if context_cfg is not None:
                    if prev_s1_loop is None:
                        try:
                            del self.stage_1._ltx_desktop_context_loop  # type: ignore[attr-defined]
                        except AttributeError:
                            pass
                    else:
                        self.stage_1._ltx_desktop_context_loop = prev_s1_loop  # type: ignore[attr-defined]
                    if prev_ctx_cfg is None:
                        try:
                            del self._ltx_desktop_context_config  # type: ignore[attr-defined]
                        except AttributeError:
                            pass
                    else:
                        self._ltx_desktop_context_config = prev_ctx_cfg  # type: ignore[attr-defined]
            _t_generate = (time.perf_counter() - _t0) if timing else 0.0
            _t0 = time.perf_counter() if timing else _t0

            memory_trace.snapshot("hdr_generate:before_cpu_handoff")
            video = video_gpu.cpu()
            del video_gpu
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            memory_trace.snapshot("hdr_generate:after_cpu_handoff")
            _t_handoff = (time.perf_counter() - _t0) if timing else 0.0
            _t0 = time.perf_counter() if timing else _t0

            # Phase 4 writer: linear HDR tensor -> EXR primary sequence
            # (linear passthrough; no EOTF / tonemap / clamp), then SDR proxy MP4
            # strictly after the primary. The proxy must never alter the primary.
            if output_format == OutputFormat.EXR_ZIP_HALF:
                half = True
            elif output_format == OutputFormat.EXR_ZIP_FLOAT:
                half = False
            else:
                raise ValueError(
                    f"HDR IC-LoRA primary output must be EXR; got output_format={output_format!r}."
                )

            out_dir = Path(output_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            _phase("exr_write", "Writing linear EXR primary")
            for idx in range(int(video.shape[0])):
                save_exr_tensor(video[idx], out_dir / f"frame_{idx:06d}.exr", half=half)
            if on_progress is not None:
                on_progress(0.9)
            _t_exr = (time.perf_counter() - _t0) if timing else 0.0
            _t0 = time.perf_counter() if timing else _t0

            if proxy_path is not None:
                _phase("proxy_encode", "Encoding SDR proxy MP4")
                encode_exr_sequence_to_mp4(out_dir, Path(proxy_path), frame_rate)
            _t_proxy = (time.perf_counter() - _t0) if timing else 0.0
            _t_total_done = (time.perf_counter() - _t_total) if timing else 0.0

            if on_progress is not None:
                on_progress(1.0)

            if timing:
                logger.info(
                    "[HDR IC-LoRA] timing: decode=%.2fs generate=%.2fs cpu_handoff=%.2fs "
                    "exr_write=%.2fs proxy=%.2fs total=%.2fs (padded_frames=%d)",
                    _t_decode, _t_generate, _t_handoff, _t_exr, _t_proxy,
                    _t_total_done, padded_num_frames,
                )
                memory_trace.write_event(
                    "hdr_timing",
                    "hdr_generate",
                    decode_s=round(_t_decode, 3),
                    generate_s=round(_t_generate, 3),
                    cpu_handoff_s=round(_t_handoff, 3),
                    exr_write_s=round(_t_exr, 3),
                    proxy_s=round(_t_proxy, 3),
                    total_s=round(_t_total_done, 3),
                    padded_frames=padded_num_frames,
                )


def _padded_frame_count(source_count: int) -> int:
    """Return the ``8n + 1`` padded frame count for ``source_count`` decoded frames.

    - ``source_count == 1`` -> ``1`` (trivially ``8*0 + 1``).
    - already ``(source_count - 1) % 8 == 0`` -> unchanged.
    - otherwise -> next value strictly greater than ``source_count`` of the
      form ``8n + 1``.

    Source frames are never trimmed.
    """
    if source_count <= 0:
        raise ValueError(f"source_count must be >= 1, got {source_count}")
    if source_count == 1:
        return 1
    if (source_count - 1) % 8 == 0:
        return source_count
    n = (source_count - 1) // 8 + 1
    return 8 * n + 1


def _module_device_dtype(
    module: torch.nn.Module,
    fallback_device: torch.device,
    fallback_dtype: torch.dtype,
) -> tuple[torch.device, torch.dtype]:
    """Infer a module's expected input device/dtype from its parameters/buffers.

    Prefers the first parameter, then the first registered buffer, so an
    encoder whose conv weight/bias is float32 (e.g. a Kijai FP8/sidecar VAE)
    is matched by a float32 input even when the pipeline dtype is bf16. Falls
    back to ``(fallback_device, fallback_dtype)`` only when the module has no
    parameters and no buffers.
    """
    for param in module.parameters():
        return param.device, param.dtype
    for buffer in module.buffers():
        return buffer.device, buffer.dtype
    return fallback_device, fallback_dtype


def _install_hdr_component_patches(
    pipeline: LTXHdrIcLoraPipeline,
    checkpoint_path: str | tuple[str, ...],
    transformer_format: TransformerFormat,
    components: ResolvedLtxComponents | None,
) -> None:
    """Install post-build component patches owned by the HDR pipeline.

    GGUF: install the GGUF transformer loader, then reroute VAE/component
    builders to their sidecar safetensors files.
    Split safetensors (Kijai-style): patch the stage transformer builders to
    read the full V2 config from the text-projection sidecar metadata, then
    reroute VAE/component builders to their sidecar files.
    Monolith (single safetensors with no sidecars): no patch.

    The patch helpers depend on the exact block attribute names
    (``stage_1``/``stage_2``/``image_conditioner``/``upsampler``/
    ``video_decoder``) set by :meth:`LTXHdrIcLoraPipeline.__init__`.
    """
    if components is None:
        return  # no-profile / monolith official-distilled path: no sidecars.

    is_gguf = transformer_format == "gguf"
    is_split = isinstance(checkpoint_path, tuple) and (
        components.video_vae_path is not None
        or components.audio_vae_path is not None
        or components.text_projection_path is not None
        or components.embeddings_connector_path is not None
    )
    if not (is_gguf or is_split):
        return  # monolith: no sidecar patch.

    from services.patches.gguf_loader_fix import (
        install_gguf_component_paths,
        install_gguf_loader,
        install_kijai_transformer_config_patch,
    )

    if is_gguf:
        install_gguf_loader(pipeline)
    else:
        # split safetensors: the full V2 transformer config lives in the
        # text-projection sidecar metadata, not the first transformer shard.
        install_kijai_transformer_config_patch(pipeline, checkpoint_path)

    install_gguf_component_paths(
        pipeline,
        checkpoint_path,
        video_vae_path=components.video_vae_path,
        audio_vae_path=components.audio_vae_path,
        mmproj_path=components.mmproj_path,
    )
