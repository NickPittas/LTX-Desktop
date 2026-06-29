"""Pure resolver: turns ModelProfilePayload into typed ResolvedLtxComponents bundle.

No heavy imports (no torch, no ltx_core). Fully testable without GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from api_types import ModelProfilePayload, ModelSelectionID

CheckpointPath = str | tuple[str, ...]
TransformerFormat = Literal["safetensors", "gguf"]
# Mirrors ``base_video_model_registry.RuntimeReadiness``. Defined locally (not
# imported) to keep this pure-resolver module free of the registry dependency.
RuntimeReadiness = Literal["none", "requires_active_profile_sidecars"]

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


def resolve_components(
    profile: ModelProfilePayload,
    *,
    selected_transformer_path: str | None = None,
    selected_cp_id: ModelSelectionID | None = None,
    selected_transformer_format: TransformerFormat | None = None,
    selected_base_family: BaseFamily | None = None,
    selected_runtime_readiness: RuntimeReadiness | None = None,
) -> ResolvedLtxComponents:
    """Turn a model profile's component paths into a typed bundle.

    Tuple ordering for split/gguf: transformer first, then text projection,
    embeddings connector, video VAE, audio VAE.

    Live model selection (Step 4): when ``selected_transformer_path`` is
    provided, it overrides *only* the transformer while reusing the active
    profile's other sidecar components. The caller (registry-driven resolver)
    passes explicit ``selected_transformer_format``, ``selected_base_family``,
    and ``selected_runtime_readiness`` so NO path/filename inference happens
    for the selected format/family/readiness:

    - Builder checkpoint paths key on ``selected_runtime_readiness``, NOT on
      base family or container format. A selection that requires active
      profile sidecars (Fast-family QuantStack distilled GGUF, Kijai distilled
      FP8, official dev safetensors, all dev/full GGUFs) is a split build:
      selected transformer + profile text projection, embeddings connector,
      video VAE, audio VAE (falsey entries filtered). A true monolith
      (``runtime_readiness == "none"``, the official distilled) loads as a
      single self-contained file. When readiness is absent (legacy callers),
      the prior base-family-driven behavior is preserved.
    - Sidecar *metadata* (text projection, embeddings connector, VAEs) is
      cleared ONLY for a true monolith (``selected_runtime_readiness == "none"``,
      or the legacy inferred monolith when readiness is absent). Entries that
      require active profile sidecars preserve profile sidecars.

    When explicit selection metadata is omitted (backward-compatible callers),
    format/readiness fall back to path inference and the legacy
    safetensors-as-monolith behavior.
    """
    c = profile.components

    if selected_transformer_path is not None:
        # Live selection override.
        selected = selected_transformer_path
        effective_transformer = selected
        # Prefer explicit registry metadata; fall back to path inference only
        # when the caller did not supply it (backward compat).
        if selected_transformer_format is not None:
            fmt: TransformerFormat = selected_transformer_format
        else:
            fmt = "gguf" if selected.lower().endswith(".gguf") else "safetensors"
        if selected_base_family is not None:
            base_family: BaseFamily = selected_base_family
        else:
            base_family = _infer_base_family(selected)
        # Builder checkpoint tuple keys on ``selected_runtime_readiness``, NOT
        # on base family or container format. A selection that requires active
        # profile sidecars (Fast-family QuantStack distilled GGUF, Kijai FP8,
        # official dev safetensors, all dev/full GGUFs) is a split build: the
        # selected transformer plus the profile's text projection, embeddings
        # connector, and VAEs (falsey entries filtered). A true monolith
        # (``runtime_readiness == "none"``, the official distilled) loads as a
        # single self-contained file. When readiness is absent (legacy callers),
        # fall back to the prior base-family-driven behavior.
        if selected_runtime_readiness == "requires_active_profile_sidecars":
            builder_paths: tuple[str, ...] = tuple(
                p
                for p in (
                    selected,
                    c.text_projection,
                    c.embeddings_connector,
                    c.video_vae,
                    c.audio_vae,
                )
                if p
            )
        elif selected_runtime_readiness == "none":
            builder_paths = (selected,)
        else:
            # Legacy backward-compat (no explicit readiness): the dev route is
            # a split build, everything else loads as a single transformer file.
            if base_family == "dev":
                builder_paths = tuple(
                    p
                    for p in (
                        selected,
                        c.text_projection,
                        c.embeddings_connector,
                        c.video_vae,
                        c.audio_vae,
                    )
                    if p
                )
            else:
                builder_paths = (selected,)
    elif c.transformer_format == "official_safetensors":
        # Monolithic checkpoint: single path.
        effective_transformer = c.transformer or ""
        fmt = "safetensors"
        base_family = _infer_base_family(effective_transformer)
        builder_paths = (c.transformer,) if c.transformer else ()
    else:
        # split_safetensors or gguf: transformer + component files.
        effective_transformer = c.transformer or ""
        fmt = "gguf" if c.transformer_format == "gguf" else "safetensors"
        base_family = _infer_base_family(effective_transformer)
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

    explicit_distilled_lora_path = _extract_distilled_lora_path(c.official_adapters)

    # Sidecar metadata is cleared ONLY for a true monolith — the official
    # distilled (``runtime_readiness == "none"``) which loads as a single
    # self-contained file with no profile sidecar inputs. Kijai FP8 (distilled
    # safetensors, ``requires_active_profile_sidecars``) and the official dev
    # safetensors MUST preserve profile sidecars. Backward compat: without
    # explicit readiness, a safetensors selection is treated as a monolith
    # (legacy callers only select the distilled monolith).
    if selected_transformer_path is not None:
        if selected_runtime_readiness is not None:
            is_true_monolith = selected_runtime_readiness == "none"
        else:
            is_true_monolith = fmt == "safetensors"
    else:
        is_true_monolith = False
    if is_true_monolith:
        text_projection_path: str | None = None
        embeddings_connector_path: str | None = None
        video_vae_path: str | None = None
        audio_vae_path: str | None = None
    else:
        text_projection_path = c.text_projection
        embeddings_connector_path = c.embeddings_connector
        video_vae_path = c.video_vae
        audio_vae_path = c.audio_vae

    cache_key = (
        profile.id,
        fmt,
        effective_transformer,
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
    if selected_transformer_path is not None:
        # Live model selection marker + identity: presence + CP id + path so
        # switching the selection always invalidates the cache (and switching
        # back only reuses an exact match).
        cache_key = (
            *cache_key,
            "model_selection",
            selected_cp_id or "",
            selected_transformer_path,
        )

    gemma_root = c.text_encoder_root if c.text_encoder_format in ("hf_folder", "gguf", "safetensors") else None

    return ResolvedLtxComponents(
        profile_id=profile.id,
        transformer_format=fmt,
        transformer_path=effective_transformer,
        checkpoint_paths_for_filtered_builders=builder_paths,
        upsampler_path=c.upsampler,
        gemma_root=gemma_root,
        text_projection_path=text_projection_path,
        mmproj_path=c.mmproj,
        embeddings_connector_path=embeddings_connector_path,
        video_vae_path=video_vae_path,
        audio_vae_path=audio_vae_path,
        base_family=base_family,
        distilled_lora_path=explicit_distilled_lora_path,
        cache_key=cache_key,
    )


def checkpoint_path_arg(components: ResolvedLtxComponents) -> str | tuple[str, ...]:
    """Return the single path or path tuple for filtered builders."""
    paths = components.checkpoint_paths_for_filtered_builders
    return paths[0] if len(paths) == 1 else paths
