"""Pure resolver: turns ModelProfilePayload into typed ResolvedLtxComponents bundle.

No heavy imports (no torch, no ltx_core). Fully testable without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from api_types import ModelProfilePayload

CheckpointPath = str | tuple[str, ...]
TransformerFormat = Literal["safetensors", "gguf"]


@dataclass(frozen=True, slots=True)
class ResolvedLtxComponents:
    profile_id: str
    transformer_format: TransformerFormat
    transformer_path: str
    checkpoint_paths_for_filtered_builders: tuple[str, ...]
    upsampler_path: str | None
    gemma_root: str | None
    text_projection_path: str | None
    embeddings_connector_path: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    cache_key: tuple[str, ...]


def resolve_components(profile: ModelProfilePayload) -> ResolvedLtxComponents:
    """Turn a model profile's component paths into a typed bundle.

    Tuple ordering for split/gguf: transformer first, then text projection,
    embeddings connector, video VAE, audio VAE.
    """
    c = profile.components
    fmt: TransformerFormat = "gguf" if c.transformer_format == "gguf" else "safetensors"

    if c.transformer_format == "official_safetensors":
        # Monolithic checkpoint: single path.
        builder_paths: tuple[str, ...] = (c.transformer,) if c.transformer else ()
    else:
        # split_safetensors or gguf: transformer + component files.
        builder_paths = tuple(
            p
            for p in (
                c.transformer,
                c.text_projection,
                c.embeddings_connector,
                c.video_vae,
                c.audio_vae,
            )
            if p
        )

    cache_key = (
        profile.id,
        fmt,
        c.transformer or "",
        *(builder_paths),
        c.upsampler or "",
        c.text_encoder_root or "",
    )

    gemma_root = c.text_encoder_root if c.text_encoder_format in ("hf_folder", "gguf", "safetensors") else None

    return ResolvedLtxComponents(
        profile_id=profile.id,
        transformer_format=fmt,
        transformer_path=c.transformer or "",
        checkpoint_paths_for_filtered_builders=builder_paths,
        upsampler_path=c.upsampler,
        gemma_root=gemma_root,
        text_projection_path=c.text_projection,
        embeddings_connector_path=c.embeddings_connector,
        video_vae_path=c.video_vae,
        audio_vae_path=c.audio_vae,
        cache_key=cache_key,
    )


def checkpoint_path_arg(components: ResolvedLtxComponents) -> str | tuple[str, ...]:
    """Return the single path or path tuple for filtered builders."""
    paths = components.checkpoint_paths_for_filtered_builders
    return paths[0] if len(paths) == 1 else paths
