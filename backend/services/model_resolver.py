"""Model resolver / capability engine (Phase 2).

Pure function that resolves a :class:`~api_types.ModelProfilePayload` against a
Phase 1 scan catalog and returns structured capability metadata as frozen
dataclasses.

Phase 2 constraints (oracle-enforced):

- **No API/OpenAPI surface.** Result types are internal frozen dataclasses.
- **No filesystem probing.** The resolver is catalog-only; ``Path.exists()`` is
  never called.
- **Current runtime priority chain:** profile explicit path → catalog
  installed/duplicate → catalog wrong_folder_usable (candidate only) → missing.
- **HDR always gated** regardless of file presence.
- **Distilled LoRA** is ``candidate_unwired`` (not runtime-wired) until Phase 6.
- **Local Gemma/API-key suppression** derived from ``text_encoder_root`` +
  local ``text_encoder_format``, independent of text projection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from api_types import (
    ModelLibraryArtifact,
    ModelLibraryScanResponse,
    ModelProfilePayload,
    ModelProfileProblem,
)

# ============================================================
# Internal status literals
# ============================================================

BaseFamily: TypeAlias = Literal["dev", "distilled", "unknown"]
Quantization: TypeAlias = Literal["bf16", "fp8", "nvfp4", "gguf", "unknown"]

#: Pipeline-level capability status.
PipelineStatus: TypeAlias = Literal[
    "supported",
    "candidate_unwired",
    "missing",
    "gated",
    "unvalidated",
    "not_applicable",
]

#: HDR workflow status — always ``gated`` until Phase 9.
HDRStatus: TypeAlias = Literal["gated", "missing", "candidate_unvalidated"]

#: Per-artifact resolution status.
ArtifactItemStatus: TypeAlias = Literal[
    "available",
    "profile_unverified",
    "candidate_usable",
    "duplicate",
    "missing",
    "gated",
    "not_applicable",
]

#: How an artifact was resolved.
ArtifactItemSource: TypeAlias = Literal[
    "profile",
    "catalog_installed",
    "catalog_wrong_folder",
    "catalog_duplicate",
    "missing",
    "profile_only",
]


# ============================================================
# Frozen result dataclasses
# ============================================================


@dataclass(frozen=True, slots=True)
class ResolvedArtifactItem:
    """A single artifact resolved against profile + catalog."""

    component_role: str
    artifact_kind: str
    required: bool
    status: ArtifactItemStatus
    source: ArtifactItemSource
    preferred_path: str | None
    canonical_relative_path: str | None
    problems: list[ModelProfileProblem]


def _default_artifacts() -> list[ResolvedArtifactItem]:
    return []


def _default_problems() -> list[ModelProfileProblem]:
    return []


@dataclass(frozen=True, slots=True)
class ProfileCapabilityResult:
    """Full capability result for a profile resolved against a catalog."""

    profile_id: str | None
    profile_valid: bool
    base_family: BaseFamily
    quantization: Quantization
    fast_status: PipelineStatus
    distilled_lora_status: PipelineStatus
    normal_status: PipelineStatus
    hdr_status: HDRStatus
    retake_upscaler_status: PipelineStatus
    has_local_text_encoder: bool
    suppresses_api_key_prompt: bool
    has_text_projection: bool
    has_upscaler: bool
    artifacts: list[ResolvedArtifactItem] = field(default_factory=_default_artifacts)
    problems: list[ModelProfileProblem] = field(default_factory=_default_problems)


# ============================================================
# Role / field mapping tables
# ============================================================

#: Profile component field → catalog component role.
_PROFILE_FIELD_TO_ROLE: dict[str, str] = {
    "transformer": "base_diffusion_model",
    "upsampler": "spatial_upscaler",
    "text_encoder_root": "gemma",
    "text_projection": "text_projection",
    "ic_lora_union": "union_control",
    "ic_lora_hdr": "hdr",
    "ic_lora_hdr_scene_embeddings": "hdr_scene_embeddings",
}

#: Reverse: catalog role → profile component field name.
_ROLE_TO_FIELD: dict[str, str] = {v: k for k, v in _PROFILE_FIELD_TO_ROLE.items()}

#: Default artifact kind per role (used when no catalog artifact exists).
_ROLE_KIND: dict[str, str] = {
    "base_diffusion_model": "diffusion_model",
    "spatial_upscaler": "upscaler",
    "gemma": "text_encoder",
    "text_projection": "text_encoder",
    "union_control": "control_adapter",
    "hdr": "control_adapter",
    "hdr_scene_embeddings": "scene_embeddings",
    "distilled_lora_384": "lora",
    "distilled_lora_384_1_1": "lora",
}

#: Distilled LoRA roles in preference order (newest first).
_DISTILLED_LORA_ROLES: tuple[str, ...] = (
    "distilled_lora_384_1_1",
    "distilled_lora_384",
)

#: Roles whose workflow is always gated at the artifact level (HDR).
_GATED_ROLES: frozenset[str] = frozenset({"hdr", "hdr_scene_embeddings"})

#: text_encoder_format values that count as local (not API).
_LOCAL_TEXT_ENCODER_FORMATS: frozenset[str] = frozenset({"hf_folder", "safetensors", "gguf"})

#: Core roles to always resolve.
_CORE_ROLES: tuple[tuple[str, bool], ...] = (
    ("base_diffusion_model", True),
    ("spatial_upscaler", False),
    ("gemma", False),
    ("union_control", False),
    ("hdr", False),
    ("hdr_scene_embeddings", False),
)


# ============================================================
# Helpers
# ============================================================


def _index_catalog(
    catalog: ModelLibraryScanResponse | list[ModelLibraryArtifact],
) -> dict[str, ModelLibraryArtifact]:
    if isinstance(catalog, ModelLibraryScanResponse):
        artifacts = catalog.artifacts
    else:
        artifacts = catalog
    return {a.component_role: a for a in artifacts}


def _paths_equal(a: str, b: str) -> bool:
    """Lexical path comparison (no filesystem probing)."""
    return str(Path(a)) == str(Path(b))


def _get_profile_path(profile: ModelProfilePayload | None, role: str) -> str | None:
    """Extract the explicit profile path for a role, if any."""
    if profile is None:
        return None

    field_name = _ROLE_TO_FIELD.get(role)
    if field_name:
        value: str | None = getattr(profile.components, field_name, None)
        if value:
            return value

    adapter_path = profile.components.official_adapters.get(role)
    if adapter_path:
        return adapter_path

    return None


def _is_profile_valid(profile: ModelProfilePayload | None) -> bool:
    if profile is None:
        return False
    return profile.validation_status in ("candidate", "validated")


def _infer_base_family(profile: ModelProfilePayload | None) -> BaseFamily:
    if profile is None:
        return "unknown"

    transformer = (profile.components.transformer or "").lower()

    # "distilled-lora" / "distilled_lora" is an adapter, not a base model.
    if "distilled-lora" in transformer or "distilled_lora" in transformer:
        return "unknown"

    # Path/filename signal takes precedence — source alone must not imply family.
    if "distilled" in transformer:
        return "distilled"

    if "dev" in transformer:
        return "dev"

    return "unknown"


def _infer_quantization(profile: ModelProfilePayload | None) -> Quantization:
    if profile is None:
        return "unknown"

    tq = (profile.components.transformer_quantization or "").lower()
    transformer = (profile.components.transformer or "").lower()

    # Explicit quantization field
    if "nvfp4" in tq:
        return "nvfp4"
    if "fp8" in tq or "input_scaled" in tq or "scaled" in tq:
        return "fp8"
    if "bf16" in tq:
        return "bf16"
    if "gguf" in tq:
        return "gguf"

    # Format-based
    if profile.components.transformer_format == "gguf":
        return "gguf"

    # Filename heuristics
    if "nvfp4" in transformer:
        return "nvfp4"
    if "fp8" in transformer or "input_scaled" in transformer or "scaled" in transformer:
        return "fp8"
    if "gguf" in transformer:
        return "gguf"

    return "unknown"


def _resolve_profile_path(
    profile_path: str,
    catalog_artifact: ModelLibraryArtifact | None,
) -> tuple[ArtifactItemStatus, ArtifactItemSource, str | None]:
    """Resolve an explicit profile path against the catalog."""
    if catalog_artifact is None:
        return ("profile_unverified", "profile", profile_path)

    for abs_path in catalog_artifact.absolute_paths:
        if _paths_equal(profile_path, abs_path):
            if catalog_artifact.status == "duplicate":
                # Profile explicit path wins; duplicate problem still emitted upstream
                return ("duplicate", "profile", profile_path)
            # installed or wrong_folder_usable — profile path makes it usable
            return ("available", "profile", profile_path)

    return ("profile_unverified", "profile", profile_path)


def _resolve_from_catalog(
    catalog_artifact: ModelLibraryArtifact | None,
) -> tuple[ArtifactItemStatus, ArtifactItemSource, str | None]:
    """Resolve an artifact from catalog state only (no profile path)."""
    if catalog_artifact is None:
        return ("missing", "missing", None)

    status = catalog_artifact.status
    if status == "installed":
        return ("available", "catalog_installed", catalog_artifact.preferred_path)
    if status == "duplicate":
        return ("duplicate", "catalog_duplicate", catalog_artifact.preferred_path)
    if status == "wrong_folder_usable":
        return ("candidate_usable", "catalog_wrong_folder", catalog_artifact.preferred_path)
    return ("missing", "missing", None)


def _resolve_role(
    profile: ModelProfilePayload | None,
    role: str,
    required: bool,
    catalog_by_role: dict[str, ModelLibraryArtifact],
) -> ResolvedArtifactItem:
    """Resolve a single artifact role against profile + catalog."""
    profile_path = _get_profile_path(profile, role)
    catalog_artifact = catalog_by_role.get(role)
    kind = _ROLE_KIND.get(role, "lora")

    problems: list[ModelProfileProblem] = []

    if profile_path is not None:
        status, source, preferred = _resolve_profile_path(profile_path, catalog_artifact)
        if status == "profile_unverified":
            problems.append(ModelProfileProblem(
                code="profile_path_unverified",
                severity="warning",
                message=f"Profile path for '{role}' not found in scan catalog",
                path=profile_path,
                field=_ROLE_TO_FIELD.get(role),
            ))
    else:
        status, source, preferred = _resolve_from_catalog(catalog_artifact)

    if status == "duplicate" and catalog_artifact is not None:
        problems.append(ModelProfileProblem(
            code="duplicate_artifact",
            severity="warning",
            message=f"Artifact '{role}' found in {len(catalog_artifact.absolute_paths)} locations",
            path=preferred,
        ))

    if status == "missing" and required:
        problems.append(ModelProfileProblem(
            code="missing_required_artifact",
            severity="error",
            message=f"Required artifact '{role}' is missing",
            field=_ROLE_TO_FIELD.get(role),
        ))

    # HDR gating at artifact level: present HDR artifacts are not usable
    if role in _GATED_ROLES and status in ("available", "candidate_usable", "duplicate"):
        status = "gated"

    canonical = catalog_artifact.canonical_relative_path if catalog_artifact else None

    return ResolvedArtifactItem(
        component_role=role,
        artifact_kind=kind,
        required=required,
        status=status,
        source=source,
        preferred_path=preferred,
        canonical_relative_path=canonical,
        problems=problems,
    )


def _resolve_distilled_lora(
    profile: ModelProfilePayload | None,
    base_family: BaseFamily,
    catalog_by_role: dict[str, ModelLibraryArtifact],
) -> ResolvedArtifactItem | None:
    """Resolve the best available distilled LoRA, or None if not applicable."""
    # Distilled base and unknown base → not applicable
    if base_family in ("distilled", "unknown"):
        return None

    # Without a profile we cannot determine base family applicability
    if profile is None:
        return None

    # Dev or unknown base → distilled LoRA is a candidate
    for role in _DISTILLED_LORA_ROLES:
        profile_path = _get_profile_path(profile, role)
        catalog_artifact = catalog_by_role.get(role)
        if profile_path is not None or (catalog_artifact is not None and catalog_artifact.status != "missing"):
            return _resolve_role(profile, role, False, catalog_by_role)

    # Neither distilled LoRA found — report the newest role as missing
    return _resolve_role(profile, _DISTILLED_LORA_ROLES[0], False, catalog_by_role)


def _resolve_text_projection(profile: ModelProfilePayload | None) -> ResolvedArtifactItem:
    """Resolve text_projection — profile-only (no catalog artifact)."""
    if profile and profile.components.text_projection:
        return ResolvedArtifactItem(
            component_role="text_projection",
            artifact_kind="text_encoder",
            required=False,
            status="available",
            source="profile_only",
            preferred_path=profile.components.text_projection,
            canonical_relative_path=None,
            problems=[],
        )
    return ResolvedArtifactItem(
        component_role="text_projection",
        artifact_kind="text_encoder",
        required=False,
        status="not_applicable",
        source="missing",
        preferred_path=None,
        canonical_relative_path=None,
        problems=[],
    )


def _check_local_text_encoder(
    profile: ModelProfilePayload | None,
) -> tuple[bool, bool]:
    """Return (has_local_text_encoder, suppresses_api_key_prompt)."""
    if profile is None:
        return (False, False)
    te_root = profile.components.text_encoder_root
    te_format = profile.components.text_encoder_format
    has_local = bool(te_root) and te_format in _LOCAL_TEXT_ENCODER_FORMATS
    return (has_local, has_local)


def _is_available(item: ResolvedArtifactItem) -> bool:
    """Whether an artifact counts as present/usable for capability derivation.

    ``duplicate`` counts as available (file is present, just in multiple
    locations). ``gated`` (HDR) and ``candidate_usable`` (wrong folder) do not.
    """
    return item.status in ("available", "duplicate")


def _find_artifact(
    artifacts: list[ResolvedArtifactItem],
    role: str,
) -> ResolvedArtifactItem | None:
    return next((a for a in artifacts if a.component_role == role), None)


def _is_distilled_lora_available(artifacts: list[ResolvedArtifactItem]) -> bool:
    for role in _DISTILLED_LORA_ROLES:
        item = _find_artifact(artifacts, role)
        if item is not None and _is_available(item):
            return True
    return False


# ============================================================
# Pipeline status derivation
# ============================================================


def _derive_fast_status(
    base_family: BaseFamily,
    base_available: bool,
    distilled_lora_available: bool,
) -> PipelineStatus:
    if not base_available:
        return "missing"

    if base_family == "distilled":
        # Standalone distilled base supports fast natively
        return "supported"

    if base_family == "dev":
        # Dev base needs distilled LoRA for fast, but loading is not wired
        if distilled_lora_available:
            return "candidate_unwired"
        return "missing"

    # Unknown base family — cannot determine
    return "missing"


def _derive_distilled_lora_status(
    base_family: BaseFamily,
    distilled_lora_available: bool,
) -> PipelineStatus:
    if base_family == "dev":
        if distilled_lora_available:
            return "candidate_unwired"
        return "missing"

    # distilled and unknown → not applicable (unknown must not candidate)
    return "not_applicable"


def _derive_normal_status(base_available: bool) -> PipelineStatus:
    return "supported" if base_available else "missing"


def _derive_retake_upscaler_status(upscaler_available: bool) -> PipelineStatus:
    # Upscaler present but not wired into retake until Phase 6
    return "candidate_unwired" if upscaler_available else "missing"


# ============================================================
# Public API
# ============================================================


def resolve_profile_capabilities(
    profile: ModelProfilePayload | None,
    catalog: ModelLibraryScanResponse | list[ModelLibraryArtifact],
) -> ProfileCapabilityResult:
    """Resolve *profile* against *catalog* and return typed capability metadata.

    Pure function — no filesystem probing, no side effects, no API surface.
    """
    catalog_by_role = _index_catalog(catalog)

    profile_id = profile.id if profile else None
    profile_valid = _is_profile_valid(profile)
    base_family = _infer_base_family(profile)
    quantization = _infer_quantization(profile)

    # Collect deprecated-profile problem
    all_problems: list[ModelProfileProblem] = []
    if profile is not None and profile.validation_status == "deprecated":
        all_problems.append(ModelProfileProblem(
            code="deprecated_profile",
            severity="warning",
            message="Profile is deprecated",
            field="validation_status",
        ))

    # Resolve core roles
    artifacts: list[ResolvedArtifactItem] = []
    for role, required in _CORE_ROLES:
        item = _resolve_role(profile, role, required, catalog_by_role)
        artifacts.append(item)
        all_problems.extend(item.problems)

    # Resolve distilled LoRA (role depends on base family)
    distilled_item = _resolve_distilled_lora(profile, base_family, catalog_by_role)
    if distilled_item is not None:
        artifacts.append(distilled_item)
        all_problems.extend(distilled_item.problems)

    # Resolve text_projection (profile-only)
    tp_item = _resolve_text_projection(profile)
    artifacts.append(tp_item)

    # Derive availability flags
    base_item = _find_artifact(artifacts, "base_diffusion_model")
    base_available = base_item is not None and _is_available(base_item)

    distilled_lora_available = _is_distilled_lora_available(artifacts)

    upscaler_item = _find_artifact(artifacts, "spatial_upscaler")
    upscaler_available = upscaler_item is not None and _is_available(upscaler_item)

    # Derive pipeline statuses
    fast_status = _derive_fast_status(base_family, base_available, distilled_lora_available)
    distilled_lora_status: PipelineStatus = (
        "not_applicable"
        if profile is None
        else _derive_distilled_lora_status(base_family, distilled_lora_available)
    )
    normal_status = _derive_normal_status(base_available)
    retake_upscaler_status = _derive_retake_upscaler_status(upscaler_available)

    # Local text encoder / API-key suppression
    has_local_te, suppresses_api_key = _check_local_text_encoder(profile)
    has_tp = bool(profile and profile.components.text_projection)

    return ProfileCapabilityResult(
        profile_id=profile_id,
        profile_valid=profile_valid,
        base_family=base_family,
        quantization=quantization,
        fast_status=fast_status,
        distilled_lora_status=distilled_lora_status,
        normal_status=normal_status,
        hdr_status="gated",
        retake_upscaler_status=retake_upscaler_status,
        has_local_text_encoder=has_local_te,
        suppresses_api_key_prompt=suppresses_api_key,
        has_text_projection=has_tp,
        has_upscaler=upscaler_available,
        artifacts=artifacts,
        problems=all_problems,
    )
