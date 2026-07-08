"""Local memory strategy planner for LTX transformer construction.

Phase 2 Slice 1 (plan: ``docs/plans/current/...``). Pure, side-effect-free
planner that maps (transformer format, base family, split-ness, quantization
kind, VRAM, workflow, block-offload availability) to a :class:`LocalMemoryPlan`
describing how the transformer should be loaded for local generation.

Heavy imports are deferred: ``torch`` (VRAM probe) and
``ltx_pipelines.utils.types.OffloadMode`` are imported lazily inside the
functions that need them, so this module stays importable without torch/GPU —
mirroring the other pure-resolver service modules.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    # Annotation-only; the real enum is imported lazily in ``plan_for_transformer``.
    from ltx_pipelines.utils.types import OffloadMode

QuantizationKind = Literal["bf16", "official_fp8_cast", "kijai_fp8_scaled", "gguf"]
LocalMemoryStrategy = Literal[
    "upstream_streaming",
    "full_resident",
    "block_offload",
    "gguf_lazy",
    "gguf_native_streaming_later",
]
TransformerFormat = Literal["safetensors", "gguf"]
BaseFamily = Literal["dev", "distilled", "unknown"]
#: Workflow discriminator. Only HDR vs non-HDR changes memory thresholds today.
Workflow = Literal["hdr", "standard"]

#: Benchmark-only env override (harness ``--force-memory-strategy``). When set
#: to ``"block_offload"``, :func:`plan_for_transformer` forces a block-offload
#: plan regardless of VRAM/threshold so the DiffusionStage block-offload patch
#: path can be timed on hardware that would otherwise pick a resident/streaming
#: strategy. Any other non-empty value raises ``ValueError`` (fail-closed) so a
#: bad benchmark knob is loud, never silently mislabelled. Unset → normal
#: planning; normal behaviour is completely unaffected.
_FORCE_STRATEGY_ENV = "LTX_FORCE_LOCAL_MEMORY_STRATEGY"


@dataclass(frozen=True, slots=True)
class LocalMemoryPlan:
    """Result of planning local transformer loading.

    Fields are exactly the Phase 2 Slice 1 contract. ``offload_mode`` holds the
    upstream ``OffloadMode`` enum to forward to the pipeline builder.
    """

    strategy: LocalMemoryStrategy
    offload_mode: "OffloadMode"
    requires_block_offload: bool
    reason: str
    trace_labels: tuple[str, ...]
    cache_key_parts: tuple[str, ...]
    disable_compile: bool
    # Per-run effective/free VRAM tier (bucketed) — included in cache_key_parts
    # so a pipeline cached under high free VRAM is not reused under low free VRAM.
    effective_vram_tier_gib: int | None = None


def detect_vram_gb() -> int | None:
    """Floor of the current CUDA device's total VRAM in GiB, or ``None``.

    Lazy-imports torch; returns ``None`` when CUDA is unavailable or the device
    property lookup fails. Never raises.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        dev = torch.cuda.current_device()
        # torch's CUDA device-properties API is only partially typed; cast the
        # total-memory field to int (the upstream contract). Wrapped in try so a
        # probe failure returns None rather than raising.
        total_bytes = cast(int, torch.cuda.get_device_properties(dev).total_memory)  # type: ignore[reportUnknownMemberType]
    except Exception:
        return None
    return total_bytes // (1024 ** 3)


def detect_free_vram_gib() -> int | None:
    """Floor of the current CUDA device's free VRAM in GiB, or ``None``.

    Uses ``torch.cuda.mem_get_info``. Lazy-imports torch; never raises.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _total = torch.cuda.mem_get_info()  # type: ignore[reportUnknownMemberType]
    except Exception:
        return None
    return free_bytes // (1024 ** 3)


_VRAM_TIER_THRESHOLDS: tuple[int, ...] = (56, 48, 40, 31, 28, 24, 16, 15, 12)


def bucket_effective_vram_gib(effective_gib: int | None) -> int | None:
    """Bucket effective VRAM to the highest tier threshold ``<= effective_gib``.

    Returns ``effective_gib`` if below the lowest threshold (12). ``None`` if
    ``effective_gib`` is ``None``.
    """
    if effective_gib is None:
        return None
    for threshold in _VRAM_TIER_THRESHOLDS:
        if effective_gib >= threshold:
            return threshold
    return effective_gib


@dataclass(frozen=True, slots=True)
class VramSnapshot:
    """Per-run VRAM snapshot for effective/free-aware planning."""

    total_gib: int | None
    free_gib: int | None
    effective_gib: int | None
    effective_tier_gib: int | None


def snapshot_vram(reserve_gib: int = 2) -> VramSnapshot:
    """Snapshot total + free + effective VRAM at call time.

    ``effective_gib = max(0, min(total or free, free - reserve_gib))`` — the
    conservative usable estimate (free minus a safety reserve). Falls back to
    total-only if free is unavailable.

    ``effective_tier_gib`` is the bucketed tier. **Near-empty exception**: when
    both total and free are known and ``(total - free) <= reserve_gib`` (the card
    is basically empty — used VRAM is within the reserve), tiering uses ``total``
    instead of ``effective``. This prevents a fresh-boot 31 GiB card (free≈31,
    effective=29) from being downgraded to tier 28 when it should be tier 31.
    """
    total = detect_vram_gb()
    free = detect_free_vram_gib()
    if free is not None:
        upper = total if total is not None else free
        effective = max(0, min(upper, free - reserve_gib))
    elif total is not None:
        effective = total
    else:
        effective = None

    # Near-empty exception: if the card is basically empty (used VRAM within the
    # reserve), don't downgrade the tier. Tier on total (physical capacity).
    if (
        total is not None
        and free is not None
        and (total - free) <= reserve_gib
    ):
        tier_source = total
    else:
        tier_source = effective

    return VramSnapshot(
        total_gib=total,
        free_gib=free,
        effective_gib=effective,
        effective_tier_gib=bucket_effective_vram_gib(tier_source),
    )


def _block_offload_plan(
    quantization_kind: QuantizationKind,
    *,
    offload_mode: "OffloadMode",
    vram_desc: str,
    available: bool,
    extra_labels: tuple[str, ...],
) -> LocalMemoryPlan:
    """Build a plan that gates/resolves through block offload.

    ``requires_block_offload`` is always True and ``strategy`` is always
    ``block_offload``. Whether construction proceeds depends on ``available``:
    available -> proceeds; unavailable -> gated with a reason explaining it.
    ``cache_key_parts`` keeps the stable quant/strategy tokens so callers key
    pipeline caches consistently regardless of availability.
    """
    if available:
        reason = f"Block offload required ({vram_desc}); available, proceeding."
        labels: tuple[str, ...] = ("block_offload", *extra_labels)
    else:
        reason = (
            f"Block offload required ({vram_desc}) but unavailable; "
            "transformer construction gated."
        )
        labels = ("block_offload_unavailable", *extra_labels)
    return LocalMemoryPlan(
        strategy="block_offload",
        offload_mode=offload_mode,
        requires_block_offload=True,
        reason=reason,
        trace_labels=labels,
        cache_key_parts=("quant", quantization_kind, "strategy", "block_offload"),
        disable_compile=True,
    )


def _forced_block_offload_plan(
    quantization_kind: QuantizationKind,
    *,
    vram_desc: str,
    extra_labels: tuple[str, ...],
) -> LocalMemoryPlan:
    """Benchmark-only forced block-offload plan.

    Driven by ``LTX_FORCE_LOCAL_MEMORY_STRATEGY=block_offload`` (set by the live
    workflow harness ``--force-memory-strategy`` in spawn mode). Forces the
    block-offload residency strategy regardless of VRAM/threshold so the
    DiffusionStage block-offload patch path can be benchmarked on hardware that
    would otherwise pick full_resident / upstream_streaming / gguf_lazy.

    Uses ``OffloadMode.NONE`` so ``DiffusionStage._transformer_ctx`` routes
    through the patched ``_build_transformer`` (the block-offload path), NOT
    upstream layer streaming (which fires only when offload_mode != NONE). The
    cache key carries an explicit ``forced`` token and trace labels carry
    ``benchmark_forced`` so this plan never collides with, or masquerades as,
    normal threshold-driven planning.
    """
    from ltx_pipelines.utils.types import OffloadMode  # noqa: PLC0415

    return LocalMemoryPlan(
        strategy="block_offload",
        offload_mode=OffloadMode.NONE,
        requires_block_offload=True,
        reason=(
            f"BENCHMARK OVERRIDE: forcing block_offload ({vram_desc}) via "
            f"{_FORCE_STRATEGY_ENV}=block_offload; normal threshold policy bypassed."
        ),
        trace_labels=("benchmark_forced", "block_offload_forced", *extra_labels),
        cache_key_parts=(
            "quant", quantization_kind, "strategy", "block_offload", "forced", "1",
        ),
        disable_compile=True,
    )


def plan_for_transformer(
    transformer_format: TransformerFormat,
    base_family: BaseFamily,
    is_componentized_split: bool,
    quantization_kind: QuantizationKind,
    vram_gb: int | None,
    workflow: Workflow,
    block_offload_available: bool,
) -> LocalMemoryPlan:
    """Pick a local loading strategy for a transformer.

    Threshold table (plan §Phase 2 Slice 1):

    - ``bf16``: ``<15`` block-offload gate; ``15-47`` upstream streaming + CPU;
      ``>=48`` full resident + NONE; HDR streams unless ``>=56``.
    - ``official_fp8_cast``: ``<15`` block-offload gate; ``15-30`` streaming + CPU;
      ``>=31`` full resident + NONE, except HDR streams below ``40``.
    - ``kijai_fp8_scaled``: ``<40`` routes through the patched block-offload
      builder (NONE; resident count from policy — 31 GiB / 121f resolves to 48
      resident / 0 swapped, i.e. no blockswap but still the patched builder, NOT
      upstream ``full_resident`` which OOMs at 31 GiB); ``>=40`` full resident.
    - ``gguf``: ``gguf_lazy`` + NONE always; ``<24`` requires block offload
      (both standard and HDR); ``disable_compile`` True for all GGUF.
    """
    from ltx_pipelines.utils.types import OffloadMode  # noqa: PLC0415

    is_hdr = workflow == "hdr"
    common_labels: tuple[str, ...] = (
        f"quant:{quantization_kind}",
        f"fmt:{transformer_format}",
        f"family:{base_family}",
        f"workflow:{workflow}",
        f"split:{is_componentized_split}",
    )

    # Benchmark-only override (harness --force-memory-strategy). Honoured before
    # any threshold logic so block offload can be forced even on high-VRAM HW.
    # Only "block_offload" is supported; any other non-empty value fails closed.
    forced = os.environ.get(_FORCE_STRATEGY_ENV, "").strip()
    if forced:
        if forced != "block_offload":
            raise ValueError(
                f"Unsupported value {forced!r} for {_FORCE_STRATEGY_ENV}; "
                f"only 'block_offload' is supported by the benchmark override. "
                f"Unset {_FORCE_STRATEGY_ENV} to restore normal memory planning."
            )
        vram_desc = f"{vram_gb}GB" if vram_gb is not None else "VRAM unknown / CUDA unavailable"
        return _forced_block_offload_plan(
            quantization_kind, vram_desc=vram_desc, extra_labels=common_labels,
        )

    # Unknown VRAM / no CUDA: fail safe — block offload gate before construction.
    if vram_gb is None:
        return _block_offload_plan(
            quantization_kind,
            offload_mode=OffloadMode.CPU,
            vram_desc="VRAM unknown / CUDA unavailable",
            available=block_offload_available,
            extra_labels=(*common_labels, "vram:unknown"),
        )

    vram = vram_gb
    labels = (*common_labels, f"vram:{vram}")

    # ---- GGUF -----------------------------------------------------------
    if quantization_kind == "gguf":
        # GGUF loads lazily (qparam block decode) with NO streaming offload.
        # Measured: gguf_lazy holds at >=24 GiB for both HDR and non-HDR (the
        # prior HDR-only vram<40 block-offload trigger is removed); block offload
        # gates only below 24 GiB. Unknown VRAM is gated above (before this block).
        if vram < 24:
            return _block_offload_plan(
                quantization_kind,
                offload_mode=OffloadMode.NONE,
                vram_desc=f"{vram}GB < 24",
                available=block_offload_available,
                extra_labels=labels,
            )
        return LocalMemoryPlan(
            strategy="gguf_lazy",
            offload_mode=OffloadMode.NONE,
            requires_block_offload=False,
            reason=f"GGUF lazy resident load at {vram}GB.",
            trace_labels=("gguf_lazy", *labels),
            cache_key_parts=("quant", quantization_kind, "strategy", "gguf_lazy"),
            disable_compile=True,
        )

    # ---- bf16 -----------------------------------------------------------
    if quantization_kind == "bf16":
        if vram < 15:
            return _block_offload_plan(
                quantization_kind,
                offload_mode=OffloadMode.CPU,
                vram_desc=f"{vram}GB < 15",
                available=block_offload_available,
                extra_labels=labels,
            )
        full_resident_threshold = 56 if is_hdr else 48
        if vram >= full_resident_threshold:
            return LocalMemoryPlan(
                strategy="full_resident",
                offload_mode=OffloadMode.NONE,
                requires_block_offload=False,
                reason=f"bf16 full resident at {vram}GB (hdr={is_hdr}).",
                trace_labels=("full_resident", *labels),
                cache_key_parts=("quant", quantization_kind, "strategy", "full_resident"),
                disable_compile=False,
            )
        return LocalMemoryPlan(
            strategy="upstream_streaming",
            offload_mode=OffloadMode.CPU,
            requires_block_offload=False,
            reason=f"bf16 upstream streaming at {vram}GB (hdr={is_hdr}).",
            trace_labels=("upstream_streaming", *labels),
            cache_key_parts=("quant", quantization_kind, "strategy", "upstream_streaming"),
            disable_compile=False,
        )

    # ---- official_fp8_cast ----------------------------------------------
    if quantization_kind == "official_fp8_cast":
        if vram < 15:
            return _block_offload_plan(
                quantization_kind,
                offload_mode=OffloadMode.CPU,
                vram_desc=f"{vram}GB < 15",
                available=block_offload_available,
                extra_labels=labels,
            )
        if is_hdr and vram < 40:
            return LocalMemoryPlan(
                strategy="upstream_streaming",
                offload_mode=OffloadMode.CPU,
                requires_block_offload=False,
                reason=f"official FP8 HDR streaming at {vram}GB (<40).",
                trace_labels=("upstream_streaming", *labels),
                cache_key_parts=("quant", quantization_kind, "strategy", "upstream_streaming"),
                disable_compile=False,
            )
        if vram >= 31:
            return LocalMemoryPlan(
                strategy="full_resident",
                offload_mode=OffloadMode.NONE,
                requires_block_offload=False,
                reason=f"official FP8 full resident at {vram}GB.",
                trace_labels=("full_resident", *labels),
                cache_key_parts=("quant", quantization_kind, "strategy", "full_resident"),
                disable_compile=False,
            )
        # 15-30 non-HDR
        return LocalMemoryPlan(
            strategy="upstream_streaming",
            offload_mode=OffloadMode.CPU,
            requires_block_offload=False,
            reason=f"official FP8 streaming at {vram}GB (<31).",
            trace_labels=("upstream_streaming", *labels),
            cache_key_parts=("quant", quantization_kind, "strategy", "upstream_streaming"),
            disable_compile=False,
        )

    # ---- kijai_fp8_scaled -----------------------------------------------
    # Kijai FP8 scaled must route through our patched DiffusionStage builder:
    # upstream CPU offload rejects this quantization policy before the patched
    # path can run, and upstream full_resident at 31 GiB OOMs (the stage builds
    # activations that exceed VRAM even though the FP8 weights themselves fit).
    # So below 40 GiB these plans use NONE to enter the patched builder; the
    # resident-block policy (block_offload._resident_policy) then decides how
    # many of the 48 blocks stay resident vs swap. Normal 31 GiB / 121-frame
    # T2V/I2V/Ingredients/HDR runs resolve to 48 resident / 0 swapped WITHIN
    # that patched path — i.e. no blockswap, but still the patched builder, not
    # upstream full_resident. True full_resident (upstream path) is only
    # returned at >= 40 GiB where VRAM headroom makes it safe. Unified for HDR
    # and standard (the prior HDR-only vram<40 split is gone).
    if vram >= 40:
        return LocalMemoryPlan(
            strategy="full_resident",
            offload_mode=OffloadMode.NONE,
            requires_block_offload=False,
            reason=f"Kijai FP8 scaled full resident at {vram}GB (hdr={is_hdr}).",
            trace_labels=("full_resident", *labels),
            cache_key_parts=("quant", quantization_kind, "strategy", "full_resident"),
            disable_compile=False,
        )
    return _block_offload_plan(
        quantization_kind,
        offload_mode=OffloadMode.NONE,
        vram_desc=f"{vram}GB < 40 (patched builder; resident count from policy)",
        available=block_offload_available,
        extra_labels=labels,
    )
