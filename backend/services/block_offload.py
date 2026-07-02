"""Block-level CPU<->onload-device offload for LTX transformer stages.

Phase 3B Slice 1 (plan: ``docs/plans/current/...``). Core, dependency-light
service that groups a transformer's nested block list and wires per-group
forward hooks so that one group at a time is moved onto the onload device
immediately before its forward, then back to the offload device immediately
after. Pairs with :class:`services.local_memory_plan.LocalMemoryPlan`.

Heavy imports (``torch``) are deferred to call sites so this module stays
importable without torch/GPU, mirroring the other resolver/service modules.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from services.memory_trace import write_event

if TYPE_CHECKING:
    import torch

    from services.local_memory_plan import LocalMemoryPlan

logger = logging.getLogger(__name__)

#: Pipeline attributes that may carry a transformer stage (single-stage and the
#: split two-stage pipelines). ``attach_memory_plan_to_stages`` writes
#: ``memory_plan`` onto whichever of these exist.
_STAGE_ATTRS: tuple[str, ...] = ("stage", "stage_1", "stage_2")

#: Attribute names stamped onto the wrapped model so repeated
#: ``apply_block_offload`` calls are idempotent (old handles removed first).
_HANDLES_ATTR = "_ltx_block_offload_handles"
_CONFIG_ATTR = "_ltx_block_offload_config"
_GROUPS_ATTR = "_ltx_block_offload_groups"


def _cpu_device() -> "torch.device":
    import torch  # noqa: PLC0415

    return torch.device("cpu")


@dataclass(slots=True)
class BlockOffloadConfig:
    """Configuration for block offload.

    ``onload_device`` is the device a group is moved onto immediately before its
    forward and moved off of immediately after (non-default, required first).
    """

    onload_device: "torch.device"
    blocks_attr: str = "velocity_model.transformer_blocks"
    blocks_per_group: int = 1
    offload_device: "torch.device" = field(default_factory=_cpu_device)
    sync_after_group: bool = True
    #: Keep the first N transformer blocks resident on ``onload_device`` for the
    #: whole model lifetime; only blocks after N stream under the hooks. 0 = the
    #: original all-block streaming behavior.
    resident_blocks: int = 0
    #: Blockswap look-ahead: number of upcoming offloaded groups to prefetch
    #: (CPU->onload) on a side CUDA stream ahead of use. 0 = synchronous moves.
    prefetch_groups: int = 2
    #: Blockswap look-behind: number of already-run offloaded groups to keep on
    #: the onload device before evicting older ones. 0 (default) = evict each
    #: group immediately after its forward (original behavior). N>0 keeps a small
    #: trailing residency window; only groups older than ``current - N`` evict.
    lookbehind_groups: int = 0


#: Debug/tuning knob (NOT production UI policy). Overrides ``blocks_per_group``
#: at the build site. Unset -> caller default. A non-positive or non-int value is
#: ignored (clamped to the default) so a bad value never breaks generation.
_BPG_ENV = "LTX_BLOCK_OFFLOAD_BLOCKS_PER_GROUP"


def resolve_blocks_per_group(default: int = 1) -> int:
    """Resolve ``blocks_per_group`` from ``LTX_BLOCK_OFFLOAD_BLOCKS_PER_GROUP``.

    Returns ``default`` when unset or invalid (non-int / non-positive), logging a
    warning. Never raises — normal generation must not crash on a bad knob.
    """
    raw = os.environ.get(_BPG_ENV)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("block_offload: ignoring invalid %s=%r", _BPG_ENV, raw)
        return default
    if value <= 0:
        logger.warning("block_offload: ignoring non-positive %s=%d", _BPG_ENV, value)
        return default
    return value


#: Tuning knob (NOT production UI policy). Keeps the first N transformer blocks
#: permanently on the onload device; only the rest stream. Unset -> a runtime
#: resident-block curve (VRAM-driven, optionally frame-aware); an explicit
#: positive value overrides that default. Never raises.
_RESIDENT_ENV = "LTX_BLOCK_OFFLOAD_RESIDENT_BLOCKS"

#: Process-local context knob set by the HDR pipeline around generation to make
#: the padded frame count available at transformer-build time (when the resident
#: policy runs). Unset -> frame count unknown.
_PADDED_FRAMES_ENV = "LTX_HDR_PADDED_FRAMES"

#: Resident-block VRAM curve anchors: (integer GiB -> resident blocks). Monotonic
#: non-decreasing; evaluated by piecewise-linear interpolation and clamped at the
#: ends. 31GiB (RTX 5090) -> 46 of 48 blocks; 24GiB -> 37; 16GiB -> 26; <=12GiB
#: -> 0 (all-streaming). Frame-count pressure is absorbed by the HDR encode
#: tiling policy, so residency is intentionally flat in padded frames (more VRAM
#: never yields fewer blocks; more frames never yields more).
_RESIDENT_VRAM_ANCHORS: tuple[tuple[int, int], ...] = (
    (12, 0),
    (16, 26),
    (24, 37),
    (31, 46),
)


def _detect_vram_gib() -> int | None:
    """CUDA total VRAM as an integer GiB floor, or None if unavailable.

    Lazy torch import; never raises.
    """
    try:
        import torch  # noqa: PLC0415

        if not torch.cuda.is_available():
            return None
        dev = torch.cuda.current_device()
        total_bytes = cast(int, torch.cuda.get_device_properties(dev).total_memory)  # type: ignore[reportUnknownMemberType]
        return total_bytes // (1024 ** 3)
    except Exception:
        logger.debug("block_offload: CUDA VRAM probe failed", exc_info=True)
        return None


def _interp_resident_blocks(vram_gib: int) -> int:
    """Piecewise-linear resident-block count from :data:`_RESIDENT_VRAM_ANCHORS`."""
    anchors = _RESIDENT_VRAM_ANCHORS
    if vram_gib <= anchors[0][0]:
        return anchors[0][1]
    if vram_gib >= anchors[-1][0]:
        return anchors[-1][1]
    for (lo_v, lo_r), (hi_v, hi_r) in zip(anchors, anchors[1:]):
        if lo_v <= vram_gib <= hi_v:
            frac = (vram_gib - lo_v) / (hi_v - lo_v)
            return round(lo_r + frac * (hi_r - lo_r))
    return anchors[-1][1]


def _resident_policy(vram_gib: int | None, padded_frames: int | None) -> int | None:
    """Compute default resident blocks from VRAM (and, when relevant, frames).

    VRAM-driven via :func:`_interp_resident_blocks`. ``padded_frames`` is accepted
    so the policy CAN tighten residency for very long clips; today it is flat in
    frames because the HDR encode tiling policy absorbs frame-count pressure
    (more frames never increases residency). Returns ``None`` when VRAM is
    unknown so the caller falls back to its safe ``default``.
    """
    del padded_frames  # frame-count lever is HDR encode tiling; kept for future use
    if vram_gib is None:
        return None
    return _interp_resident_blocks(vram_gib)


def _default_resident_blocks(default: int) -> int:
    """Dynamic resident-block default: VRAM curve, optionally frame-aware."""
    padded_raw = os.environ.get(_PADDED_FRAMES_ENV)
    padded_frames: int | None = None
    if padded_raw:
        try:
            padded_frames = int(padded_raw)
        except (TypeError, ValueError):
            padded_frames = None
    result = _resident_policy(_detect_vram_gib(), padded_frames)
    return default if result is None else result


def resolve_resident_blocks(default: int = 0) -> int:
    """Resolve resident block count from ``LTX_BLOCK_OFFLOAD_RESIDENT_BLOCKS``.

    Resolution order:

    - Unset / empty env -> runtime resident-block curve
      (:func:`_default_resident_blocks`): VRAM-driven via piecewise-linear
      anchors (31GiB -> 46, 24GiB -> 37, 16GiB -> 26, <=12GiB -> 0), optionally
      frame-aware via ``LTX_HDR_PADDED_FRAMES``.
    - Explicit positive int -> that value (overrides the curve).
    - Explicit non-positive int (<=0) -> 0 (disable residency; all-streaming).
    - Non-int garbage -> warning + runtime curve.

    ``default`` is only the fallback when CUDA is unavailable / the VRAM probe
    fails. Never raises — generation must not crash on a bad knob or a missing GPU.
    """
    raw = os.environ.get(_RESIDENT_ENV)
    if not raw:
        return _default_resident_blocks(default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("block_offload: ignoring invalid %s=%r", _RESIDENT_ENV, raw)
        return _default_resident_blocks(default)
    if value <= 0:
        logger.debug("block_offload: %s=%d -> 0 resident blocks (disabled)", _RESIDENT_ENV, value)
        return 0
    return value


#: Debug/tuning knob (NOT production UI policy). Blockswap look-ahead: how many
#: upcoming offloaded groups to prefetch (CPU->onload) on a side CUDA stream.
#: Unset -> caller default (2); a positive value overrides; non-positive (<=0)
#: disables blockswap (synchronous moves). Never raises.
_PREFETCH_ENV = "LTX_BLOCK_OFFLOAD_PREFETCH_GROUPS"


def resolve_prefetch_groups(default: int = 2) -> int:
    """Resolve blockswap prefetch group count from ``LTX_BLOCK_OFFLOAD_PREFETCH_GROUPS``.

    - Unset / empty -> ``default`` (2).
    - Explicit positive int -> that value.
    - Explicit non-positive int (<=0) -> 0 (disable blockswap; synchronous).
    - Non-int garbage -> warning + ``default``.

    Never raises.
    """
    raw = os.environ.get(_PREFETCH_ENV)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("block_offload: ignoring invalid %s=%r", _PREFETCH_ENV, raw)
        return default
    if value <= 0:
        logger.debug("block_offload: %s=%d -> blockswap disabled", _PREFETCH_ENV, value)
        return 0
    return value


#: Debug/tuning knob (NOT production UI policy). Blockswap look-behind: how many
#: already-run offloaded groups to keep on the onload device before evicting older
#: ones. Unset -> 0 (evict immediately; original behavior). Never raises.
_LOOKBEHIND_ENV = "LTX_BLOCK_OFFLOAD_LOOKBEHIND_GROUPS"


def resolve_lookbehind_groups(default: int = 0) -> int:
    """Resolve blockswap look-behind from ``LTX_BLOCK_OFFLOAD_LOOKBEHIND_GROUPS``.

    - Unset / empty -> ``default`` (0).
    - Explicit non-negative int -> that value.
    - Negative or non-int -> warning + ``default``.

    Never raises.
    """
    raw = os.environ.get(_LOOKBEHIND_ENV)
    if not raw:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        logger.warning("block_offload: ignoring invalid %s=%r", _LOOKBEHIND_ENV, raw)
        return default
    if value < 0:
        logger.warning("block_offload: ignoring negative %s=%d", _LOOKBEHIND_ENV, value)
        return default
    return value


@dataclass(frozen=True, slots=True)
class BlockOffloadResult:
    """Outcome of :func:`apply_block_offload`."""

    model: "torch.nn.Module"
    num_blocks: int
    num_groups: int
    config: BlockOffloadConfig
    resident_blocks: int = 0
    offloaded_blocks: int = 0


def block_offload_available() -> bool:
    """Phase 3B Slice 1 ships the core service; offload is always available."""
    return True


def attach_memory_plan_to_stages(
    pipeline: object, memory_plan: "LocalMemoryPlan"
) -> None:
    """Stamp ``memory_plan`` onto every present stage attribute of ``pipeline``.

    Iterates ``("stage", "stage_1", "stage_2")``; whichever exist get
    ``stage.memory_plan = memory_plan``. Missing attributes are skipped.
    """
    for attr in _STAGE_ATTRS:
        stage: object = getattr(pipeline, attr, None)
        if stage is None:
            continue
        setattr(stage, "memory_plan", memory_plan)


def _resolve_blocks(
    model: "torch.nn.Module", blocks_attr: str
) -> "list[torch.nn.Module]":
    """Walk the dotted ``blocks_attr`` path on ``model`` and return block list.

    ``blocks_attr`` like ``"velocity_model.transformer_blocks"`` resolves to the
    nested ``ModuleList``; its children are returned as a plain list.
    """
    obj: object = model
    for part in blocks_attr.split("."):
        obj = getattr(obj, part)
    container = cast("Sequence[torch.nn.Module]", obj)
    return list(container)


def _remove_old_handles(model: "torch.nn.Module") -> None:
    """Detach any handles a previous ``apply_block_offload`` registered."""
    old: object = getattr(model, _HANDLES_ATTR, None)
    if not isinstance(old, list):
        return
    for handle in cast("list[Any]", old):
        try:
            handle.remove()
        except Exception:
            logger.warning(
                "block_offload: failed to remove stale hook handle", exc_info=True
            )


def _move_lora_attr(mod: "torch.nn.Module", device: "torch.device") -> None:
    """Move any tensors in a module's ``lora_pairs`` attribute to ``device``.

    Runtime LoRA pairs on GgufLinear/KijaiFp8ScaledLinear are plain attributes
    (not registered params/buffers), so the group param/buffer move does not
    touch them. Move them with the group so they track the group device under
    blockswap residency/eviction instead of getting stranded on a stale device.
    Expected shape: an iterable of ``(lora_A, lora_B, strength)`` tuples (A/B
    tensors, strength numeric); tensor entries are moved, non-tensor entries and
    the tuple/list container shape are preserved. No-op for modules without
    ``lora_pairs``; never raises on an unexpected shape.
    """
    pairs = getattr(mod, "lora_pairs", None)
    if pairs is None:
        return
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return

    def _move(obj: Any) -> Any:
        if isinstance(obj, torch.Tensor):
            return obj.to(device)
        if isinstance(obj, list):
            elems = cast("list[object]", obj)
            return [_move(x) for x in elems]
        if isinstance(obj, tuple):
            elems = cast("tuple[object, ...]", obj)
            return tuple(_move(x) for x in elems)
        return obj

    try:
        moved = _move(pairs)
    except Exception:
        logger.debug("block_offload: lora_pairs move skipped (unexpected shape)", exc_info=True)
        return
    if moved is not pairs:
        setattr(mod, "lora_pairs", moved)


def apply_block_offload(
    model: "torch.nn.Module", config: BlockOffloadConfig
) -> BlockOffloadResult:
    """Wire per-group on/off hooks onto ``model``'s nested block list.

    Keeps the first ``config.resident_blocks`` blocks permanently on
    ``onload_device`` (no hooks). The remaining blocks are grouped by
    ``config.blocks_per_group``; each offloaded group is moved onto
    ``onload_device`` in a forward pre-hook and back to ``offload_device`` in a
    forward post-hook on the group's last block.

    When ``config.prefetch_groups`` > 0 and ``onload_device`` is CUDA, offloaded
    groups use a sliding blockswap: a side CUDA stream prefetches the next K
    groups' CPU->GPU copy ahead of use, and the post-hook evicts the finished
    group. The pre-hook waits on the group's prefetch event before its forward
    launches, so async copies never race with execution. Falls back to plain
    synchronous moves otherwise. Repeated calls are idempotent (stale handles
    removed first).

    # ponytail: side-stream H2D prefetch overlapped with the current group's
    # compute; eviction stays on the main stream (compute is already done there,
    # race-free). Ceiling is single-stream copy bandwidth when the prefetch can't
    # outrun compute; upgrade path is a pinned allocator + D2H-on-side-stream if
    # eviction latency becomes a limit.
    """
    import torch  # noqa: PLC0415

    blocks = _resolve_blocks(model, config.blocks_attr)
    bpg = max(1, config.blocks_per_group)

    # Optional static partial residency: the first N blocks stay on the onload
    # device permanently (moved up front, never hooked, never moved back); only
    # blocks after N stream under the per-group hooks. Clamp N to the block count
    # so a too-large value simply makes every block resident (no hooks).
    resident_n = max(0, min(config.resident_blocks, len(blocks)))
    resident = blocks[:resident_n]
    offload_blocks = blocks[resident_n:]
    groups: list[list["torch.nn.Module"]] = [
        offload_blocks[i : i + bpg] for i in range(0, len(offload_blocks), bpg)
    ]
    num_groups = len(groups)

    onload = config.onload_device
    offload = config.offload_device
    do_sync = config.sync_after_group and onload.type == "cuda"
    pf_k = max(0, config.prefetch_groups)
    lookbehind = max(0, config.lookbehind_groups)

    # Blockswap (sliding look-ahead prefetch) is only meaningful on a CUDA onload
    # device with >1 offloaded group and a positive look-ahead. Otherwise the
    # original synchronous move behavior is used verbatim.
    pf_stream: Any = None
    use_blockswap = onload.type == "cuda" and pf_k > 0 and num_groups > 1
    if use_blockswap:
        try:
            pf_stream = torch.cuda.Stream(device=onload)
        except Exception:
            logger.debug(
                "block_offload: prefetch stream unavailable -> synchronous", exc_info=True
            )
            pf_stream = None
            use_blockswap = False

    def _move_group(group: list["torch.nn.Module"], device: "torch.device") -> None:
        for mod in group:
            for param in mod.parameters(recurse=True):
                param.data = param.data.to(device)
            for buf in mod.buffers(recurse=True):
                buf.data = buf.data.to(device)
            # Runtime LoRA pairs are plain attributes (not params/buffers); move
            # them with the group so they track the group device under blockswap.
            _move_lora_attr(mod, device)

    # One-shot trace proof that hooks actually fire: the first pre/post hook
    # invocation records a single event, then the flag suppresses the rest so a
    # long generation does not flood the trace. No-op unless tracing is enabled.
    hook_state: dict[str, bool] = {"pre_emitted": False, "post_emitted": False}

    # Blockswap bookkeeping (only mutated when use_blockswap). ``pf_events[gi]``
    # holds a CUDA event recorded on the prefetch stream once group ``gi``'s
    # CPU->GPU copy is queued (cleared once consumed); ``group_on_gpu[gi]`` tracks
    # residency so we never re-copy an already-resident group.
    pf_events: list[Any] = [None] * num_groups
    group_on_gpu: list[bool] = [False] * num_groups

    def _prefetch_group(gi: int) -> None:
        """Queue CPU->onload copy for group ``gi`` on the prefetch stream."""
        if pf_stream is None or group_on_gpu[gi] or pf_events[gi] is not None:
            return
        with torch.cuda.stream(pf_stream):
            _move_group(groups[gi], onload)
            ev = torch.cuda.Event()
            ev.record()
        pf_events[gi] = ev

    def _evict_behind(gi: int) -> None:
        """Evict offloaded groups older than the look-behind window back to CPU.

        After group ``gi`` finishes, evict every group ``j <= gi - lookbehind``
        that is still on GPU with no pending prefetch. ``lookbehind == 0`` evicts
        ``gi`` itself (original immediate-eviction behavior); ``lookbehind > 0``
        keeps a small trailing window. Resident blocks have no hooks and are
        never touched; future/prefetched groups (``j > gi``) are never evicted.

        Race-free: any ``j <= gi`` has already executed, so its prefetch event was
        consumed in its pre-hook (``pf_events[j] is None``); a main-stream D2H of
        ``j`` cannot race the prefetch stream (which only touches groups > gi).
        """
        evict_floor = gi - lookbehind
        if evict_floor < 0:
            return
        for j in range(0, min(evict_floor, num_groups - 1) + 1):
            if group_on_gpu[j] and pf_events[j] is None:
                _move_group(groups[j], offload)
                group_on_gpu[j] = False
                pf_events[j] = None

    def _make_pre_hook(gi: int) -> Callable[..., None]:
        group = groups[gi]

        def hook(_module: Any, _args: Any, _kwargs: Any) -> None:
            if not hook_state["pre_emitted"]:
                hook_state["pre_emitted"] = True
                write_event(
                    "block_offload_pre_hook",
                    "block_offload",
                    group_size=len(group),
                    onload_device=str(onload),
                    blockswap=use_blockswap,
                    prefetch_groups=pf_k,
                    lookbehind_groups=lookbehind,
                )
            if use_blockswap:
                # Ensure this group is on GPU before its forward launches. If a
                # prefetch is in flight on the side stream, wait (on this, the
                # main compute stream) for that copy to complete — this is the
                # only ordering point that makes the async copy safe to read.
                ev = pf_events[gi]
                if ev is not None:
                    ev.wait()
                    pf_events[gi] = None
                    group_on_gpu[gi] = True
                if not group_on_gpu[gi]:
                    _move_group(group, onload)
                    group_on_gpu[gi] = True
                # Prefetch the next K groups on the side stream, overlapping
                # their H2D copy with this group's compute.
                for j in range(gi + 1, min(gi + 1 + pf_k, num_groups)):
                    _prefetch_group(j)
            else:
                _move_group(group, onload)

        return hook

    def _make_post_hook(gi: int) -> Callable[..., None]:
        group = groups[gi]

        def hook(_module: Any, _args: Any, _kwargs: Any, _output: Any) -> None:
            if not hook_state["post_emitted"]:
                hook_state["post_emitted"] = True
                write_event(
                    "block_offload_post_hook",
                    "block_offload",
                    group_size=len(group),
                    onload_device=str(onload),
                    offload_device=str(offload),
                    blockswap=use_blockswap,
                    lookbehind_groups=lookbehind,
                )
            # Evict back to CPU. In blockswap mode, evict via the look-behind
            # window (lookbehind==0 -> evict this group now; lookbehind>0 -> keep
            # a trailing window, evict only older groups). The group's forward
            # already completed on the main stream (post-hook runs after forward),
            # so a main-stream D2H is race-free. Resident blocks have no hooks and
            # are never evicted here.
            if not use_blockswap and do_sync:
                torch.cuda.synchronize()
            if use_blockswap:
                _evict_behind(gi)
            else:
                _move_group(group, offload)
                group_on_gpu[gi] = False
                pf_events[gi] = None

        return hook

    # Idempotent re-apply: drop stale handles, then register fresh.
    _remove_old_handles(model)
    handles: list[Any] = []

    # Resident blocks -> onload device permanently (params + buffers). They get
    # NO hooks, so nothing ever moves them back to the offload device.
    for mod in resident:
        for param in mod.parameters(recurse=True):
            param.data = param.data.to(onload)
        for buf in mod.buffers(recurse=True):
            buf.data = buf.data.to(onload)

    # Offloaded blocks start on the offload device; only these carry hooks.
    for mod in offload_blocks:
        for param in mod.parameters(recurse=True):
            param.data = param.data.to(offload)
        for buf in mod.buffers(recurse=True):
            buf.data = buf.data.to(offload)

    for gi, group in enumerate(groups):
        if not group:
            continue
        pre_hook = _make_pre_hook(gi)
        for mod in group:
            handles.append(mod.register_forward_pre_hook(pre_hook, with_kwargs=True))
        # ponytail: post-hook on the LAST block only. Registering the group-off
        # post-hook on every block would evict the group after block[0], before
        # block[1] runs — correct only at group size 1. Last-only generalizes
        # correctly to any group size; at the default (blocks_per_group=1) it is
        # equivalent to "each block".
        post_hook = _make_post_hook(gi)
        handles.append(
            group[-1].register_forward_hook(post_hook, with_kwargs=True)
        )

    setattr(model, _HANDLES_ATTR, handles)
    setattr(model, _CONFIG_ATTR, config)
    setattr(model, _GROUPS_ATTR, groups)

    write_event(
        "block_offload_apply",
        "block_offload",
        num_blocks=len(blocks),
        resident_blocks=len(resident),
        offloaded_blocks=len(offload_blocks),
        num_groups=len(groups),
        blocks_per_group=bpg,
        prefetch_groups=pf_k,
        lookbehind_groups=lookbehind,
        blockswap=use_blockswap,
        onload_device=str(onload),
        offload_device=str(offload),
    )

    return BlockOffloadResult(
        model=model,
        num_blocks=len(blocks),
        num_groups=len(groups),
        config=config,
        resident_blocks=len(resident),
        offloaded_blocks=len(offload_blocks),
    )


def release_block_offload_residency(model: "torch.nn.Module") -> bool:
    """Release ALL transformer-block GPU residency from a block-offloaded model.

    Moves every block resolved by ``config.blocks_attr`` (resident blocks AND any
    prefetched/offloaded groups currently on the onload device) back to
    ``config.offload_device`` (CPU) and detaches the on/off hook handles (and
    their blockswap closure state). Intended to free GPU residency before a
    non-denoise phase that needs a large GPU temporary.

    After this the model's block hooks are gone, so the caller must NOT keep
    using it for forward passes that expect blockswap/residency. Discard the
    built model so the next denoise stage rebuilds and re-applies block offload
    afresh (re-arming via :func:`apply_block_offload` is also possible, but
    discard + rebuild is the safe choice given stale prefetch-stream state).

    Returns ``True`` if ``model`` was block-offload-configured (and released),
    ``False`` for a plain/non-configured model. Never raises.

    .. note::
       In the current HDR pipeline this has **no in-tree caller**: upstream
       ``gpu_model`` already moves the whole stage transformer to ``meta`` +
       runs ``cleanup_memory()`` (``empty_cache``) on exit of
       ``self.stage_1(...)`` / ``self.stage_2.model_context(...)``, i.e. *before*
       the next phase, so there is no live retained transformer to release at
       the available wrapper hooks. Kept as the primitive for any caller that
       does hold a live block-offloaded model.
    """
    config = getattr(model, _CONFIG_ATTR, None)
    if not isinstance(config, BlockOffloadConfig):
        return False
    _remove_old_handles(model)
    moved = 0
    try:
        blocks = _resolve_blocks(model, config.blocks_attr)
        offload = config.offload_device
        for mod in blocks:
            for param in mod.parameters(recurse=True):
                param.data = param.data.to(offload)
            for buf in mod.buffers(recurse=True):
                buf.data = buf.data.to(offload)
            # Drop runtime LoRA pairs back to CPU with the block tensors.
            _move_lora_attr(mod, offload)
        moved = len(blocks)
    except Exception:
        logger.warning("block_offload: release_residency block move failed", exc_info=True)
    # Clear stale handle/group state so a later apply_block_offload is clean.
    setattr(model, _HANDLES_ATTR, [])
    setattr(model, _GROUPS_ATTR, [])
    write_event(
        "block_offload_release",
        "block_offload",
        moved_blocks=moved,
        onload_device=str(config.onload_device),
        offload_device=str(config.offload_device),
    )
    return True


def move_non_block_modules_to_device(
    model: "torch.nn.Module", config: BlockOffloadConfig
) -> None:
    """Move every param/buffer NOT owned by a block group onto the onload device.

    Block params/buffers (resolved at ``config.blocks_attr``) are left untouched
    so they remain under offload-hook control. Identity (``id(...)``) is used to
    exclude block tensors regardless of where they sit in the module tree.
    """
    blocks = _resolve_blocks(model, config.blocks_attr)
    block_tensor_ids: set[int] = set()
    for mod in blocks:
        for param in mod.parameters(recurse=True):
            block_tensor_ids.add(id(param))
        for buf in mod.buffers(recurse=True):
            block_tensor_ids.add(id(buf))

    onload = config.onload_device
    for param in model.parameters(recurse=True):
        if id(param) not in block_tensor_ids:
            param.data = param.data.to(onload)
    for buf in model.buffers(recurse=True):
        if id(buf) not in block_tensor_ids:
            buf.data = buf.data.to(onload)
