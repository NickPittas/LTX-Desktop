"""HDR scene embeddings loader/validator.

Adapted from upstream LTX-2 (commit ``780984275fd47128b02bef9b5c085404276866ee``,
file ``packages/ltx-pipelines/src/ltx_pipelines/hdr_ic_lora.py``).

The HDR workflow is video-only (an upscale/HDR mode; audio is not produced or
conditioned). It uses the pre-computed ``video_context`` scene embedding that
**replaces** the text prompt encoding. An ``audio_context`` key, if present, is
loaded/validated for metadata compatibility only and is intentionally ignored
at generation time. This loader reads and validates those tensors from a
safetensors file.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class HDRSceneEmbeddings:
    """Loaded HDR scene embeddings (video + optional audio context tensors)."""

    video_context: torch.Tensor
    audio_context: torch.Tensor | None


def load_hdr_scene_embeddings(path: str) -> HDRSceneEmbeddings:
    """Load and validate HDR scene embeddings from a safetensors file.

    Required keys:
    - ``video_context`` — video scene context tensor (must be a floating-point
      tensor with at least 2 dimensions).

    Optional keys:
    - ``audio_context`` — audio scene context tensor. Loaded and validated here
      only for metadata/compatibility; HDR generation is video-only and this
      tensor is intentionally ignored at inference time (see ``_generate_hdr``).
      The upstream HDR pipeline would be video-only when this is absent.

    Raises ``ValueError`` with a clear message if the file is missing required
    keys or the tensors have unexpected shapes/dtypes.
    """
    from typing import Any

    import safetensors.torch as _st

    # safetensors ships no type stubs; use getattr to satisfy pyright strict.
    _safe_open: Any = getattr(_st, "safe_open")

    tensors: dict[str, torch.Tensor] = {}
    f: Any = _safe_open(path, framework="pt", device="cpu")
    try:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    finally:
        del f

    if "video_context" not in tensors:
        raise ValueError(
            f"HDR scene embeddings file {path!r} is missing required key 'video_context'. "
            f"Available keys: {sorted(tensors.keys())}"
        )

    video_context = tensors["video_context"]
    if not video_context.is_floating_point():
        raise ValueError(
            f"'video_context' must be a floating-point tensor, got dtype={video_context.dtype}"
        )
    if video_context.ndim < 2:
        raise ValueError(
            f"'video_context' must have at least 2 dimensions, got shape={tuple(video_context.shape)}"
        )

    audio_context: torch.Tensor | None = tensors.get("audio_context")
    if audio_context is not None and not audio_context.is_floating_point():
        raise ValueError(
            f"'audio_context' must be a floating-point tensor, got dtype={audio_context.dtype}"
        )

    return HDRSceneEmbeddings(
        video_context=video_context,
        audio_context=audio_context,
    )


__all__ = ["HDRSceneEmbeddings", "load_hdr_scene_embeddings"]
