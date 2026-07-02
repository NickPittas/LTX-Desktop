"""Monkey-patch: replace pin-all-at-init with bounded pinned pool.

The upstream _LayerStore.__init__ pins ALL layer tensors upfront via
pin_memory(), allocating ~24 GB of page-locked host memory for a 22B model.
This wastes USS (private memory) and CUDA reserved memory since most layers
are idle at any given time.

This patch replaces _LayerStore with an on-demand pinning strategy: only the
layers about to be transferred to GPU are pinned (prefetch_count + 1 layers).
After a layer is evicted from GPU, its pinned copy is freed and the original
source data is restored.

Benefits (measured on RTX 5090, 22B distilled, spc=2, FP8):
  - Peak USS: -11 GB (33.9 -> 22.4 GB)
  - torch_peak_reserved: -4.6 GB (12.7 -> 8.1 GB)
  - Duration: -12% (278 -> 245s)

Remove this patch once the upstream ltx-core package includes the fix.

Usage:
    import services.patches.pinned_pool_fix  # noqa: F401
"""

from __future__ import annotations

import itertools
import logging
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

try:
    from ltx_core.layer_streaming import _LayerStore
except (ModuleNotFoundError, ImportError):
    logger.info(
        "ltx_core.layer_streaming is not available; pinned_pool_fix will no-op."
    )
    _LayerStore = None  # type: ignore[assignment]


def _patched_init(self: Any, layers: nn.ModuleList, target_device: torch.device) -> None:
    self.target_device = target_device
    self.num_layers = len(layers)
    self._on_gpu: set[int] = set()

    # Keep a reference to the source data for each layer so we can pin it
    # on demand and restore it after eviction.
    self._source_data: list[dict[str, torch.Tensor]] = []
    for layer in layers:
        source: dict[str, torch.Tensor] = {}
        for name, tensor in itertools.chain(layer.named_parameters(), layer.named_buffers()):
            source[name] = tensor.data
        self._source_data.append(source)

    # Hold pinned tensors alive until the H2D transfer completes.
    # Without this, the CachingHostAllocator can reclaim a pinned tensor
    # as soon as its Python reference is dropped, even if an async H2D
    # transfer is still reading from it.
    self._pinned_in_flight: dict[int, list[torch.Tensor]] = {}


def _patched_move_to_gpu(self: Any, idx: int, layer: nn.Module, *, non_blocking: bool = False) -> None:
    """Pin layer *idx* on demand, then transfer to GPU."""
    self._check_idx(idx)
    if idx in self._on_gpu:
        return
    source = self._source_data[idx]
    pinned_refs: list[torch.Tensor] = []
    for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
        pinned = source[name].pin_memory()
        param.data = pinned.to(self.target_device, non_blocking=non_blocking)
        pinned_refs.append(pinned)
    # Keep pinned tensors alive until eviction — the async H2D transfer
    # may still be reading from them.
    self._pinned_in_flight[idx] = pinned_refs
    self._on_gpu.add(idx)


def _patched_evict_to_cpu(self: Any, idx: int, layer: nn.Module) -> None:
    """Restore source data, freeing the GPU and pinned copies."""
    self._check_idx(idx)
    if idx not in self._on_gpu:
        return
    source = self._source_data[idx]
    for name, param in itertools.chain(layer.named_parameters(), layer.named_buffers()):
        param.data = source[name]
    # Release pinned tensors — the H2D transfer is complete by now.
    self._pinned_in_flight.pop(idx, None)
    self._on_gpu.discard(idx)


def _patched_cleanup(self: Any) -> None:
    """Release all source data and in-flight pinned references."""
    for source_dict in self._source_data:
        source_dict.clear()
    self._source_data.clear()
    self._pinned_in_flight.clear()


# Apply patches
if _LayerStore is not None:
    assert hasattr(_LayerStore, "__init__"), "_LayerStore.__init__ not found — patch needs updating."
    assert hasattr(_LayerStore, "move_to_gpu"), "_LayerStore.move_to_gpu not found — patch needs updating."
    assert hasattr(_LayerStore, "evict_to_cpu"), "_LayerStore.evict_to_cpu not found — patch needs updating."
    assert hasattr(_LayerStore, "cleanup"), "_LayerStore.cleanup not found — patch needs updating."

    _LayerStore.__init__ = _patched_init  # type: ignore[assignment]
    _LayerStore.move_to_gpu = _patched_move_to_gpu  # type: ignore[assignment]
    _LayerStore.evict_to_cpu = _patched_evict_to_cpu  # type: ignore[assignment]
    _LayerStore.cleanup = _patched_cleanup  # type: ignore[assignment]
