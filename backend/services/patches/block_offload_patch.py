"""Monkey-patch: DiffusionStage block-offload build path.

Phase 3B Slice 2 (plan: ``docs/plans/current/...``). Wraps
``ltx_pipelines.utils.blocks.DiffusionStage._build_transformer`` so that a stage
carrying a ``memory_plan`` with ``requires_block_offload=True`` builds the
transformer resident on CPU, then installs per-group on/off hooks via
:mod:`services.block_offload` and moves only the non-block (static) modules onto
the target device. The whole model is **not** moved onto CUDA; transformer
blocks stream onto the onload device group-by-group under the forward hooks.

Stages without a ``memory_plan`` or with ``requires_block_offload=False``
delegate to the original builder unchanged (``device``/``kwargs`` forwarded
as-is), so all non-block-offload paths are bit-for-bit identical.

Idempotent: a sentinel on the wrapped function short-circuits re-application on
re-import. No-ops (logs info) when ``DiffusionStage`` is not importable, so the
module imports cleanly without the vendored package present.

Remove this patch once upstream ships block-level offload.

Usage::

    import services.patches.block_offload_patch  # noqa: F401
"""

from __future__ import annotations

import logging
from typing import Any

import torch

from services.block_offload import (
    BlockOffloadConfig,
    apply_block_offload,
    move_non_block_modules_to_device,
    resolve_blocks_per_group,
    resolve_lookbehind_groups,
    resolve_prefetch_groups,
    resolve_resident_blocks,
)
from services.memory_trace import write_event

logger = logging.getLogger(__name__)

#: Sentinel stored on the wrapper to short-circuit double-wrapping on re-import.
_PATCH_FLAG = "_ltx_desktop_block_offload_patch_applied"
#: Attribute on the wrapper holding the original unbound ``_build_transformer``.
_ORIGINAL_ATTR = "_ltx_desktop_block_offload_original"

try:
    from ltx_pipelines.utils.blocks import DiffusionStage as _DiffusionStage
except (ModuleNotFoundError, ImportError):
    logger.info(
        "ltx_pipelines.utils.blocks.DiffusionStage is not available; "
        "block_offload_patch will no-op."
    )
    _DiffusionStage = None  # type: ignore[assignment]


def _patched_build_transformer(
    self: Any, *, device: torch.device | None = None, **kwargs: Any
) -> Any:
    """Replacement for ``DiffusionStage._build_transformer``.

    # ponytail: synchronous build-then-hook — no prefetch/overlap with the build
    # step. The build is already a one-shot cold path; ceiling is irrelevant
    # here, the per-group hook ceiling (see block_offload.apply_block_offload)
    # is what governs steady-state throughput.
    """
    memory_plan: object = getattr(self, "memory_plan", None)
    # Resolve the original off the *installed* wrapper (reading its
    # ``_ORIGINAL_ATTR``) rather than the module-global name. A module reload
    # rebinds the global to a fresh wrapper while the installed function keeps
    # its identity — reading the class attribute survives that, and a plain
    # repeated ``import`` (the contract) never re-executes the body at all.
    installed = _DiffusionStage._build_transformer  # type: ignore[union-attr]
    original = getattr(installed, _ORIGINAL_ATTR, installed)

    # ponytail: robust attribute access — stages without a plan (every
    # non-block-offload build) fall through to the original builder unchanged.
    if memory_plan is None or not getattr(memory_plan, "requires_block_offload", False):
        return original(self, device=device, **kwargs)

    # Block offload and torch.compile are incompatible (hooks + graph capture).
    # The planner marks such plans with disable_compile=True; enforce it here so
    # a misconfigured stage fails loudly rather than silently corrupting graphs.
    if getattr(self, "_compilation_config", None) is not None:
        raise RuntimeError(
            "Block offload requires torch.compile disabled "
            "(memory_plan.disable_compile must be True), but "
            "DiffusionStage._compilation_config is set."
        )

    target: torch.device = device or getattr(self, "_device")

    # Build the full transformer resident on CPU. The original does
    # ``X0Model(...).to(target).eval()``; with target=cpu nothing lands on the
    # onload device yet — blocks stay CPU-side until a forward hook pulls them.
    model = original(self, device=torch.device("cpu"), **kwargs)

    config = BlockOffloadConfig(
        onload_device=target,
        blocks_per_group=resolve_blocks_per_group(1),
        resident_blocks=resolve_resident_blocks(0),
        prefetch_groups=resolve_prefetch_groups(2),
        lookbehind_groups=resolve_lookbehind_groups(0),
    )
    result = apply_block_offload(model, config)
    # Static (non-block) modules — embed/norm/projection/etc. — live on the
    # onload device permanently; only the offloaded transformer blocks stream.
    # Resident blocks are already on the onload device (apply_block_offload moved
    # them); move_non_block_modules_to_device leaves all block tensors untouched.
    move_non_block_modules_to_device(model, config)

    # Diagnostics marker; block handles/groups/config are already stamped by
    # apply_block_offload for idempotent re-apply and inspection.
    setattr(model, "_ltx_block_offload_built", True)

    write_event(
        "block_offload_build",
        "DiffusionStage._build_transformer",
        target_device=str(target),
        blocks_per_group=config.blocks_per_group,
        resident_blocks=result.resident_blocks,
        offloaded_blocks=result.offloaded_blocks,
        num_blocks=result.num_blocks,
        num_groups=result.num_groups,
        prefetch_groups=config.prefetch_groups,
        lookbehind_groups=config.lookbehind_groups,
    )

    # Do NOT call ``model.to(target)`` here — that would defeat the offload
    # hooks by pulling every block onto the onload device at once.
    return model.eval()


def _apply_patch() -> None:
    """Bind ``_patched_build_transformer`` onto ``DiffusionStage`` once."""
    if _DiffusionStage is None:
        return
    assert hasattr(_DiffusionStage, "_build_transformer"), (
        "DiffusionStage._build_transformer not found — patch needs updating."
    )
    original = _DiffusionStage._build_transformer
    if getattr(original, _PATCH_FLAG, False):
        # Already wrapped (re-import); leave the single existing wrapper in place.
        return
    setattr(_patched_build_transformer, _ORIGINAL_ATTR, original)
    setattr(_patched_build_transformer, _PATCH_FLAG, True)
    _DiffusionStage._build_transformer = _patched_build_transformer  # type: ignore[assignment]
    logger.info("DiffusionStage._build_transformer wrapped for block offload.")


_apply_patch()
