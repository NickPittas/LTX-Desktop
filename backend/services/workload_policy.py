"""IC-LoRA workload classification -> compensation plan + cache-key parts.

Pure module: maps (workflow, frame_count, width, height, VRAM) to a
:class:`LoraWorkloadPlan` describing which compensation knobs to apply for a run
(blockswap resident-block count, HDR rolling context window, HDR VAE encode
tiling). The plan's :meth:`LoraWorkloadPlan.cache_key_parts` is threaded into
the IC-LoRA / HDR pipeline cache keys so a ``normal`` run never reuses a
pipeline built for a ``large`` run and vice-versa. Cache-safety rationale: env
overrides alone are NOT enough because a pipeline cache hit skips the rebuild,
so the env would never be re-applied; the cache key must carry the plan.

Heavy imports deferred (``detect_vram_gb`` is lazy); importable without torch.
"""

from __future__ import annotations

from dataclasses import dataclass

from api_types import InpaintPipelineVersion

# >121 padded frames is the "long clip" threshold (matches the HDR context-window
# and block_offload resident-policy cutoffs). >=1080p is "large pixels".
_LARGE_FRAME_THRESHOLD = 121
_LARGE_PIXELS = 1920 * 1080


@dataclass(frozen=True)
class LoraWorkloadPlan:
    """Compensation knobs chosen for one IC-LoRA run.

    ``None``/``False`` means "no override" (use the pipeline's default policy).
    Kept intentionally flat — no capability-matrix abstraction.
    """

    label: str
    resident_blocks: int | None = None
    blockswap_prefetch: int | None = None
    hdr_context_window: int | None = None
    hdr_vae_encode_tile: bool = False
    inpaint_context_window_px: int | None = None
    inpaint_context_overlap_px: int | None = None

    def cache_key_parts(self) -> tuple[str, ...]:
        """Tuple folded into the pipeline cache key so plan changes invalidate it."""
        parts: list[str] = ["workload", self.label]
        if self.resident_blocks is not None:
            parts.append(f"resident={self.resident_blocks}")
        if self.blockswap_prefetch is not None:
            parts.append(f"prefetch={self.blockswap_prefetch}")
        if self.hdr_context_window is not None:
            parts.append(f"ctx={self.hdr_context_window}")
        if self.hdr_vae_encode_tile:
            parts.append("vae_tile=1")
        if self.inpaint_context_window_px is not None:
            parts.append(f"inpaint_ctx={self.inpaint_context_window_px}")
        if self.inpaint_context_overlap_px is not None:
            parts.append(f"inpaint_overlap={self.inpaint_context_overlap_px}")
        return tuple(parts)

    def summary(self) -> str:
        """One-line human-readable description for logs."""
        bits: list[str] = [self.label]
        if self.resident_blocks is not None:
            bits.append(f"resident_blocks={self.resident_blocks}")
        if self.blockswap_prefetch is not None:
            bits.append(f"blockswap_prefetch={self.blockswap_prefetch}")
        if self.hdr_context_window is not None:
            bits.append(f"hdr_context_window={self.hdr_context_window}")
        if self.hdr_vae_encode_tile:
            bits.append("hdr_vae_encode_tile")
        if self.inpaint_context_window_px is not None:
            bits.append(f"inpaint_context_window_px={self.inpaint_context_window_px}")
        if self.inpaint_context_overlap_px is not None:
            bits.append(f"inpaint_context_overlap_px={self.inpaint_context_overlap_px}")
        return ", ".join(bits)


def _detect_vram_gib() -> float | None:
    """Lazily detect effective CUDA VRAM tier in GiB; returns ``None`` if unavailable."""
    try:
        from services.local_memory_plan import snapshot_vram  # noqa: PLC0415
    except Exception:  # noqa: BLE001  keep the module importable without the service
        return None
    snap = snapshot_vram()
    return float(snap.effective_tier_gib) if snap.effective_tier_gib is not None else None


def classify_lora_workload(
    *,
    workflow: str,
    frame_count: int | None,
    width: int | None,
    height: int | None,
    vram_gib: float | None = None,
    inpaint_pipeline_version: InpaintPipelineVersion | None = None,
) -> LoraWorkloadPlan:
    """Classify an IC-LoRA run and pick compensation knobs.

    Intentionally simple (no capability matrix):

    - ``normal`` (not large) -> no overrides.
    - non-HDR large (``frame_count > 121`` OR ``>=1080p``) -> blockswap;
      ``37`` resident for one large axis, ``26`` for long+1080p. Below the
      ~28 GiB VRAM tier, residency drops (20 for long+1080p, 26 for one axis)
      and blockswap prefetch is disabled to fit 24 GiB cards.
    - HDR ``large_duration`` (``>121f``) -> rolling context window (65, or 49
      below 28 GiB), HDR VAE encode tile on, ``resident_blocks`` 46 at ``>=31
      GiB`` else 37.
    - HDR ``large_pixels`` only (``>=1080p``, ``<=121f``) -> HDR VAE encode tile
      on, no resident override (duration drives blockswap).
    """
    large_duration = frame_count is not None and frame_count > _LARGE_FRAME_THRESHOLD
    large_pixels = (
        width is not None and height is not None and width * height >= _LARGE_PIXELS
    )
    is_hdr = workflow == "hdr"

    if not (large_duration or large_pixels):
        return LoraWorkloadPlan(label="normal")

    if workflow == "in_outpainting" and inpaint_pipeline_version == "v2":
        # V2 receives the handler's single authoritative snapshot; probing again
        # here could classify a different effective-VRAM state.
        vram = vram_gib
        both_axes = large_duration and large_pixels
        if vram is None or vram < 12:
            resident, window, overlap, prefetch = 0, 33, 8, 0
        elif vram < 15:
            resident, window, overlap, prefetch = 0, 33, 8, 0
        elif vram < 16:
            resident, window, overlap, prefetch = 20, 33, 8, 0
        elif vram < 24:
            resident, window, overlap, prefetch = (20 if both_axes else 26), 33, 8, 0
        elif vram < 28:
            resident, window, overlap, prefetch = (20 if both_axes else 26), 49, 16, 0
        else:
            resident, window, overlap, prefetch = (26 if both_axes else 37), 65, 16, None
        return LoraWorkloadPlan(
            label=f"inpaint_v2:large:ctx{window}",
            resident_blocks=resident,
            blockswap_prefetch=prefetch,
            inpaint_context_window_px=window,
            inpaint_context_overlap_px=overlap,
        )

    if not is_hdr:
        vram = vram_gib if vram_gib is not None else _detect_vram_gib()
        # Measured (RTX 5090, Kijai fp8 + IC-LoRA, 1080p x 193f): 26 resident
        # peaks at 24.67 GiB and OOMs a 24 GiB card. Below the ~28 GiB tier drop
        # residency and disable blockswap prefetch: prefetch costs ~1.1 GiB peak
        # for ~0.6% time, and denoise is compute-bound so extra swap is
        # near-free (+4.4% at 20 resident, peak 21.43 GiB, fits 24 GiB).
        tight = vram is not None and vram < 28
        if large_duration and large_pixels:
            if tight:
                return LoraWorkloadPlan(
                    label="large:blockswap20:lowvram",
                    resident_blocks=20,
                    blockswap_prefetch=0,
                )
            return LoraWorkloadPlan(label="large:blockswap26", resident_blocks=26)
        if tight:
            # ponytail: provisional single-axis low-VRAM value (single-axis peak
            # unmeasured); 26 is lighter than the measured both-axes 26 (24.67
            # GiB) so it fits, and swap is near-free.
            return LoraWorkloadPlan(
                label="large:blockswap26:lowvram",
                resident_blocks=26,
                blockswap_prefetch=0,
            )
        return LoraWorkloadPlan(label="large:blockswap37", resident_blocks=37)

    # HDR: prefer rolling context + VAE tile; add blockswap for long clips.
    vram = vram_gib if vram_gib is not None else _detect_vram_gib()
    if large_duration:
        ctx = 49 if (vram is not None and vram < 28) else 65
        resident = 46 if (vram is not None and vram >= 31) else 37
        return LoraWorkloadPlan(
            label=f"hdr:large_duration:ctx{ctx}",
            resident_blocks=resident,
            hdr_context_window=ctx,
            hdr_vae_encode_tile=True,
        )
    # HDR large pixels only.
    return LoraWorkloadPlan(label="hdr:large_pixels", hdr_vae_encode_tile=True)
