"""Pure resolver: turns ModelProfilePayload into typed ResolvedLtxComponents bundle.

No heavy imports (no torch, no ltx_core). Fully testable without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from api_types import ModelProfilePayload

CheckpointPath = str | tuple[str, ...]
TransformerFormat = Literal["safetensors", "gguf"]

# Phase 3D (plan §12): base model family routes the fast pipeline.
#   - ``distilled`` => ``ltx_pipelines.distilled.DistilledPipeline``
#   - ``dev``        => ``ltx_pipelines.ti2vid_two_stages.TI2VidTwoStagesPipeline``
#                       with the distilled LoRA + upstream ``LTX_2_3_PARAMS`` guider.
#   - ``unknown``   => actionable HTTPError before heavy load (never guess).
BaseFamily = Literal["dev", "distilled", "unknown"]

# Phase 3D: explicit distilled LoRA adapter roles in preference order (newest
# first). Matches ``OFFICIAL_LTX23_ADAPTERS`` keys + the resolver's preference.
_DISTILLED_LORA_ROLES: tuple[str, ...] = (
    "distilled_lora_384_1_1",
    "distilled_lora_384",
)


def _infer_base_family(transformer_path: str) -> BaseFamily:
    """Infer the base family from the transformer path/filename only.

    Rules (oracle strategy, plan §12 + Phase 3D):
    - ``distilled-lora`` / ``distilled_lora`` is an *adapter* filename, not a
      base model. It must NOT imply distilled base.
    - ``distilled`` substring => distilled base.
    - ``dev`` substring => dev base.
    - otherwise ``unknown`` (caller fails with an actionable HTTPError).
    """
    path = transformer_path.lower()
    if "distilled-lora" in path or "distilled_lora" in path:
        return "unknown"
    if "distilled" in path:
        return "distilled"
    if "dev" in path:
        return "dev"
    return "unknown"


def _extract_distilled_lora_path(official_adapters: dict[str, str]) -> str | None:
    """Return the explicit profile distilled-LoRA path, preferring v1.1 then v1.

    Returns ``None`` when the profile has no explicit distilled LoRA adapter
    path (callers may then try a canonical models-dir fallback).
    """
    for role in _DISTILLED_LORA_ROLES:
        path = official_adapters.get(role)
        if path:
            return path
    return None


@dataclass(frozen=True, slots=True)
class ResolvedLtxComponents:
    profile_id: str
    transformer_format: TransformerFormat
    transformer_path: str
    checkpoint_paths_for_filtered_builders: tuple[str, ...]
    upsampler_path: str | None
    gemma_root: str | None
    text_projection_path: str | None
    # Phase 3A (plan §9 Option A): explicit mmproj projection path. Carried so
    # active-profile metadata can reach the multimodal I2V path; not yet wired
    # to runtime llama.cpp. Cache key includes it so a profile change that
    # toggles mmproj invalidates the pipeline cache.
    mmproj_path: str | None
    embeddings_connector_path: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    # Phase 3D (plan §12): dev-vs-distilled pipeline routing metadata.
    # ``base_family`` routes the fast pipeline factory; ``distilled_lora_path``
    # is the *explicit* profile-side distilled LoRA path (canonical fallback
    # resolution lives in the handler, where the models dir is known).
    base_family: BaseFamily
    distilled_lora_path: str | None
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

    base_family = _infer_base_family(c.transformer or "")
    explicit_distilled_lora_path = _extract_distilled_lora_path(c.official_adapters)

    cache_key = (
        profile.id,
        fmt,
        c.transformer or "",
        *(builder_paths),
        c.upsampler or "",
        c.text_encoder_root or "",
        # mmproj participates in the cache key so toggling it invalidates the
        # pipeline cache (relevant once the multimodal path is wired).
        c.mmproj or "",
        # Phase 3D: base family routes the pipeline class; explicit distilled
        # LoRA path is part of the cache key. The handler appends the effective
        # canonical-fallback LoRA path when explicit is missing but canonical
        # exists, so cache invalidates correctly on fallback resolution.
        base_family,
        explicit_distilled_lora_path or "",
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
        mmproj_path=c.mmproj,
        embeddings_connector_path=c.embeddings_connector,
        video_vae_path=c.video_vae,
        audio_vae_path=c.audio_vae,
        base_family=base_family,
        distilled_lora_path=explicit_distilled_lora_path,
        cache_key=cache_key,
    )


def checkpoint_path_arg(components: ResolvedLtxComponents) -> str | tuple[str, ...]:
    """Return the single path or path tuple for filtered builders."""
    paths = components.checkpoint_paths_for_filtered_builders
    return paths[0] if len(paths) == 1 else paths
