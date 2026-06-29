"""Unified base-video model registry (source of truth for live model selection).

The registry is the single source of truth for **generation-selectable base
video transformer variants**. It owns the selectable ``ModelSelectionID`` values
and the metadata used by the model-options endpoint, the scanner, the request
resolver, and the frontend model-selection popover.

Design rules (plan: docs/plans/current/03-live-model-selection.md §
"Required source-of-truth fix"):

- ``ModelSelectionID`` is a runtime string; ``GET /api/models/model-options`` is
  the authoritative runtime source. Unknown ids are rejected by callers with
  ``UNSUPPORTED_MODEL_SELECTION`` (this module raises ``KeyError``; handlers
  translate to the HTTP error — services layer never imports routes).
- Downloadable checkpoint specs stay in
  :mod:`runtime_config.model_download_specs`; registry entries link to them via
  ``download_cp_id`` instead of duplicating downloader state. Scanner-only
  artifacts (Kijai FP8, QuantStack distilled GGUF, official dev safetensors)
  have ``download_cp_id=None``.
- The registry performs its own read-only filesystem evidence walk (it must not
  mutate, download, move, or create anything). Evidence semantics mirror the
  scanner: ``installed``/``wrong_folder_usable``/``duplicate`` ⇒ installed with
  a usable ``preferred_path``; ``missing`` ⇒ not installed, no runtime path.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from api_types import (
    ArtifactKind,
    CatalogSection,
    LTXVideoGenPipelineFamily,
    ModelCheckpointID,
    ModelSelectionID,
    ScanArtifactStatus,
)

#: Directories excluded from the registry's evidence walk (mirror the scanner).
_SKIP_DIRS: frozenset[str] = frozenset({
    ".downloading",
    "cache",
    "tmp",
    "temp",
    ".cache",
    "__pycache__",
})

#: Semantic grouping label shared by every base video transformer candidate.
BASE_VIDEO_MODEL_GROUP: str = "Base video model"

TransformerFormat = Literal["safetensors", "gguf"]
BaseFamily = Literal["distilled", "dev"]
RuntimeReadiness = Literal["none", "requires_active_profile_sidecars"]


@dataclass(frozen=True, slots=True)
class BaseVideoRegistryStaticEntry:
    """Static registry metadata for a selectable base video transformer.

    None of these fields depend on the filesystem; they are the authoritative
    catalog metadata consumed by the scanner (which derives its base-video
    canonical artifacts from this table) and by the resolved entry builder.
    """

    id: ModelSelectionID
    label: str
    pipeline_family: LTXVideoGenPipelineFamily
    section: CatalogSection
    variant_group: str
    repo_id: str
    canonical_relative_path: str
    downloadable: bool
    download_cp_id: ModelCheckpointID | None
    expected_size_bytes: int
    remote_filename: str | None
    artifact_kind: ArtifactKind
    component_role: str
    transformer_format: TransformerFormat
    base_family: BaseFamily
    runtime_readiness: RuntimeReadiness


# ------------------------------------------------------------------
# Static registry table
# ------------------------------------------------------------------
#
# Exact metadata from plan §"Exact non-derived registry metadata" / §"Selectable
# IDs and families". ``source_url`` is always ``https://huggingface.co/{repo_id}``;
# ``group`` is always ``Base video model``; the derived fields
# (expected_absolute_path, scanner_status, installed, preferred_path,
# transformer_path) are computed from ``models_dir`` + filesystem evidence.

_REGISTRY: tuple[BaseVideoRegistryStaticEntry, ...] = (
    # ---- Fast family (pipeline_family="fast") ----
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled",
        label="LTX-2.3 22B distilled (full precision)",
        pipeline_family="fast",
        section="full",
        variant_group="ltx-2.3-distilled",
        repo_id="Lightricks/LTX-2.3",
        canonical_relative_path="diffusion_models/ltx-2.3-22b-distilled.safetensors",
        downloadable=True,
        download_cp_id="ltx-2.3-22b-distilled",
        expected_size_bytes=43_000_000_000,
        remote_filename=None,
        artifact_kind="diffusion_model",
        component_role="base_diffusion_model",
        transformer_format="safetensors",
        base_family="distilled",
        runtime_readiness="none",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-fp8-kijai-v3",
        label="LTX-2.3 22B distilled FP8 (Kijai v3)",
        pipeline_family="fast",
        section="kijai",
        variant_group="ltx-2.3-distilled-fp8",
        repo_id="Kijai/LTX2.3_comfy",
        canonical_relative_path="diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=0,
        remote_filename=None,
        artifact_kind="diffusion_model",
        component_role="base_diffusion_model_fp8",
        transformer_format="safetensors",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q2-k",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q2_K (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q2_K.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=12_408_656_544,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q3-k-s",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q3_K_S (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=13_959_437_984,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q3-k-m",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q3_K_M (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=14_702_550_688,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q4-k-s",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q4_K_S (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=16_706_378_400,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q4_K_M (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=17_763_015_328,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q5-k-s",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q5_K_S (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q5_K_S.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=18_542_680_736,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-distilled-gguf-quantstack-q5-k-m",
        label="LTX-2.3 22B distilled 1.1 GGUF — Q5_K_M (QuantStack)",
        pipeline_family="fast",
        section="gguf",
        variant_group="ltx-2.3-distilled-gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q5_K_M.gguf",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=19_388_448_416,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="distilled",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    # ---- Full family (pipeline_family="full") ----
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-dev",
        label="LTX-2.3 22B dev (full precision)",
        pipeline_family="full",
        section="full",
        variant_group="ltx-2.3-dev",
        repo_id="Lightricks/LTX-2.3",
        canonical_relative_path="diffusion_models/ltx-2.3-22b-dev.safetensors",
        downloadable=False,
        download_cp_id=None,
        expected_size_bytes=43_000_000_000,
        remote_filename=None,
        artifact_kind="diffusion_model",
        component_role="base_diffusion_model",
        transformer_format="safetensors",
        base_family="dev",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-dev-gguf-q4-k-m",
        label="LTX-2.3 22B dev GGUF — Q4_K_M",
        pipeline_family="full",
        section="gguf",
        variant_group="ltx-2.3-dev-gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        canonical_relative_path="diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-Q4_K_M.gguf",
        downloadable=True,
        download_cp_id="ltx-2.3-22b-dev-gguf-q4-k-m",
        expected_size_bytes=14_326_856_736,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="dev",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-dev-gguf-ud-q4-k-m",
        label="LTX-2.3 22B dev GGUF — UD Q4_K_M",
        pipeline_family="full",
        section="gguf",
        variant_group="ltx-2.3-dev-gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        canonical_relative_path="diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-UD-Q4_K_M.gguf",
        downloadable=True,
        download_cp_id="ltx-2.3-22b-dev-gguf-ud-q4-k-m",
        expected_size_bytes=16_506_438_688,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="dev",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-dev-gguf-q6-k",
        label="LTX-2.3 22B dev GGUF — Q6_K",
        pipeline_family="full",
        section="gguf",
        variant_group="ltx-2.3-dev-gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        canonical_relative_path="diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-Q6_K.gguf",
        downloadable=True,
        download_cp_id="ltx-2.3-22b-dev-gguf-q6-k",
        expected_size_bytes=17_774_906_400,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="dev",
        runtime_readiness="requires_active_profile_sidecars",
    ),
    BaseVideoRegistryStaticEntry(
        id="ltx-2.3-22b-dev-gguf-ud-q5-k-m",
        label="LTX-2.3 22B dev GGUF — UD Q5_K_M",
        pipeline_family="full",
        section="gguf",
        variant_group="ltx-2.3-dev-gguf",
        repo_id="unsloth/LTX-2.3-GGUF",
        canonical_relative_path="diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-UD-Q5_K_M.gguf",
        downloadable=True,
        download_cp_id="ltx-2.3-22b-dev-gguf-ud-q5-k-m",
        expected_size_bytes=18_274_719_776,
        remote_filename=None,
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        transformer_format="gguf",
        base_family="dev",
        runtime_readiness="requires_active_profile_sidecars",
    ),
)


def iter_base_video_registry_static_entries() -> tuple[BaseVideoRegistryStaticEntry, ...]:
    """Return the static (filesystem-independent) registry entries in catalog order."""
    return _REGISTRY


def _find_static(selection_id: str) -> BaseVideoRegistryStaticEntry | None:
    return next((e for e in _REGISTRY if e.id == selection_id), None)


def is_known_base_video_selection(selection_id: str) -> bool:
    """True if ``selection_id`` is a registered base video selection id."""
    return _find_static(selection_id) is not None


def get_base_video_selection_family(
    selection_id: str,
) -> LTXVideoGenPipelineFamily | None:
    """Return the static pipeline family for a known selection id, else ``None``.

    Used by request validation (family-mismatch guard) without touching the
    filesystem. Unknown ids return ``None`` so the caller can fall through to
    the resolver which rejects with ``UNSUPPORTED_MODEL_SELECTION``.
    """
    static = _find_static(selection_id)
    return static.pipeline_family if static is not None else None


@dataclass(frozen=True, slots=True)
class BaseVideoModelRegistryEntry:
    """A resolved registry entry carrying filesystem evidence for ``models_dir``.

    All 23 fields from the plan's ``BaseVideoModelRegistryEntry`` contract:
    static catalog metadata + derived (``expected_absolute_path``,
    ``scanner_status``, ``installed``, ``preferred_path``, ``transformer_path``).
    """

    # Static catalog metadata
    id: ModelSelectionID
    label: str
    group: str
    pipeline_family: LTXVideoGenPipelineFamily
    section: CatalogSection
    variant_group: str
    repo_id: str
    source_url: str
    canonical_relative_path: str
    expected_absolute_path: str
    downloadable: bool
    download_cp_id: ModelCheckpointID | None
    expected_size_bytes: int
    remote_filename: str | None
    artifact_kind: ArtifactKind
    component_role: str
    transformer_format: TransformerFormat
    base_family: BaseFamily
    runtime_readiness: RuntimeReadiness
    # Derived from models_dir + filesystem evidence
    scanner_status: ScanArtifactStatus
    installed: bool
    preferred_path: str | None
    transformer_path: str | None


def _discover_file_locations(models_dir: Path, filename: str) -> list[Path]:
    """Read-only walk: all occurrences of ``filename`` under ``models_dir``.

    Skips download/cache/VCS directories (mirrors the scanner). Never mutates.
    """
    if not models_dir.exists() or not models_dir.is_dir():
        return []
    found: list[Path] = []
    for root, dirs, files in os.walk(models_dir):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        if filename in files:
            found.append(Path(root) / filename)
    return sorted(found)


def _resolve_entry(
    static: BaseVideoRegistryStaticEntry, models_dir: Path
) -> BaseVideoModelRegistryEntry:
    """Build a resolved entry with filesystem evidence for a single artifact."""
    filename = Path(static.canonical_relative_path).name
    canonical_abs = models_dir / static.canonical_relative_path
    paths = _discover_file_locations(models_dir, filename)

    source_url = f"https://huggingface.co/{static.repo_id}"

    if not paths:
        return BaseVideoModelRegistryEntry(
            id=static.id,
            label=static.label,
            group=BASE_VIDEO_MODEL_GROUP,
            pipeline_family=static.pipeline_family,
            section=static.section,
            variant_group=static.variant_group,
            repo_id=static.repo_id,
            source_url=source_url,
            canonical_relative_path=static.canonical_relative_path,
            expected_absolute_path=str(canonical_abs),
            downloadable=static.downloadable,
            download_cp_id=static.download_cp_id,
            expected_size_bytes=static.expected_size_bytes,
            remote_filename=static.remote_filename,
            artifact_kind=static.artifact_kind,
            component_role=static.component_role,
            transformer_format=static.transformer_format,
            base_family=static.base_family,
            runtime_readiness=static.runtime_readiness,
            scanner_status="missing",
            installed=False,
            preferred_path=None,
            transformer_path=None,
        )

    canonical_match = next((p for p in paths if p == canonical_abs), None)
    preferred = canonical_match if canonical_match is not None else paths[0]

    if len(paths) == 1:
        status: ScanArtifactStatus = (
            "installed" if paths[0] == canonical_abs else "wrong_folder_usable"
        )
    else:
        status = "duplicate"

    return BaseVideoModelRegistryEntry(
        id=static.id,
        label=static.label,
        group=BASE_VIDEO_MODEL_GROUP,
        pipeline_family=static.pipeline_family,
        section=static.section,
        variant_group=static.variant_group,
        repo_id=static.repo_id,
        source_url=source_url,
        canonical_relative_path=static.canonical_relative_path,
        expected_absolute_path=str(canonical_abs),
        downloadable=static.downloadable,
        download_cp_id=static.download_cp_id,
        expected_size_bytes=static.expected_size_bytes,
        remote_filename=static.remote_filename,
        artifact_kind=static.artifact_kind,
        component_role=static.component_role,
        transformer_format=static.transformer_format,
        base_family=static.base_family,
        runtime_readiness=static.runtime_readiness,
        scanner_status=status,
        installed=True,
        preferred_path=str(preferred),
        transformer_path=str(preferred),
    )


def iter_base_video_model_entries(models_dir: Path) -> list[BaseVideoModelRegistryEntry]:
    """Enumerate every registered base video entry with filesystem evidence.

    Read-only; never mutates, downloads, or creates anything. Order matches the
    static registry catalog order (Fast family first, then Full family).
    """
    return [_resolve_entry(static, models_dir) for static in _REGISTRY]


def resolve_base_video_model_selection(
    models_dir: Path, selection_id: ModelSelectionID
) -> BaseVideoModelRegistryEntry:
    """Resolve a single selection id to a registry entry with filesystem evidence.

    Raises ``KeyError`` for unknown selection ids; handlers translate this to
    ``UNSUPPORTED_MODEL_SELECTION``. A missing-but-known selection returns an
    entry with ``installed=False`` and ``transformer_path=None``; handlers raise
    ``MODEL_SELECTION_NOT_INSTALLED`` using ``expected_absolute_path``.
    """
    static = _find_static(selection_id)
    if static is None:
        raise KeyError(f"Unknown base video model selection: {selection_id!r}")
    return _resolve_entry(static, models_dir)
