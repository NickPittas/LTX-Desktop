"""Read-only recursive model library scanner (Phase 1).

Produces a catalog of **known** artifacts with status metadata, plus separate
lists of unknown and partial files. The scanner is a pure function: it accepts a
:class:`~pathlib.Path` and returns a typed result without mutating the
filesystem — no moves, deletes, downloads, or folder creation.

Canonical expectations are **subfolder-only** — no known artifact is ever
canonical at the models root:

- **Checkpoint specs**: canonical path = ``models_dir / spec.relative_path``
  (mirrors :func:`resolve_model_path`); every spec now carries a subfolder.
- **Adapters**: canonical path = ``models_dir / adapters / <filename>``.
- **Scanner-only known files** (VAE, text projection, alternate transformer
  builds, GGUF text encoder): recognized with their own subfolder canonicals
  but have no download CP spec.

Files discovered at the models root or in a non-canonical subfolder are reported
as ``wrong_folder_usable`` — the scanner never silently "fixes" them.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from api_types import (
    AdapterID,
    AdapterKind,
    ArtifactKind,
    CatalogSection,
    ModelCheckpointID,
    ModelLibraryArtifact,
    ModelLibraryScanResponse,
    PartialFile,
    ScannerConfidence,
    ScanArtifactStatus,
    UnknownFile,
)
from runtime_config.model_download_specs import (
    ADAPTER_TO_CP_ID,
    OFFICIAL_LTX23_ADAPTERS,
    get_model_cp_spec,
)
from services.base_video_model_registry import (
    iter_base_video_registry_static_entries,
)

#: Directories that are never scanned (download temp, caches, VCS).
_SKIP_DIRS: frozenset[str] = frozenset({
    ".downloading",
    "cache",
    "tmp",
    "temp",
    ".cache",
    "__pycache__",
})

#: File suffixes treated as partial downloads — never installed.
_PARTIAL_SUFFIXES: tuple[str, ...] = (".part", ".tmp")

#: Canonical subfolder for adapter (IC-LoRA / distilled-LoRA / scene-embedding) files.
_ADAPTER_CANONICAL_SUBFOLDER = "adapters"

#: Adapter IDs whose workflow remains gated at the scanner/catalog level even
#: when installed. HDR artifacts (the HDR IC-LoRA and its scene-embedding
#: support asset) are now supported — installed copies report ``supported``.
#: Selectability of the scene-embedding asset as a standalone adapter is owned
#: by the handler, not the scanner.
_GATED_ADAPTER_IDS: frozenset[AdapterID] = frozenset()


@dataclass(frozen=True, slots=True)
class _CanonicalArtifact:
    """Scanner-local canonical expectation for a single known artifact.

    ``canonical_relative_path`` is derived from current runtime semantics
    (``spec.relative_path`` for checkpoints, bare filename for adapters) so
    that ``installed`` status aligns with what the downloader/resolver actually
    checks today.
    """

    filename: str
    artifact_kind: ArtifactKind
    component_role: str
    canonical_relative_path: str
    repo_id: str
    expected_size_bytes: int
    is_folder: bool
    gated: bool
    cp_id: ModelCheckpointID | None
    adapter_id: AdapterID | None
    # ---- Phase 2A catalog grouping metadata (plan §7) ----
    section: CatalogSection = "full"
    display_name: str = ""
    variant_group: str = ""
    downloadable: bool = True
    # Remote filename when it differs from the local basename; ``None`` ⇒ equal
    # to ``filename``.
    remote_filename: str | None = None


#: Maps non-adapter checkpoint IDs to (kind, role). Canonical subfolder is
#: derived from ``spec.relative_path`` so it stays in sync with the runtime.
_CP_KIND_MAP: dict[ModelCheckpointID, tuple[ArtifactKind, str]] = {
    "ltx-2.3-22b-distilled": ("diffusion_model", "base_diffusion_model"),
    "ltx-2.3-spatial-upscaler-x2-1.0": ("upscaler", "spatial_upscaler"),
    "gemma-3-12b-it-qat-q4_0-unquantized": ("text_encoder", "gemma"),
    "dpt-hybrid-midas": ("depth_processor", "depth_processor"),
    "yolox-l-torchscript": ("person_detector", "person_detector"),
    "dw-ll-ucoco-384-bs5": ("pose_processor", "pose_processor"),
    "z-image-turbo": ("image_gen_model", "image_gen_model"),
    # unsloth LTX-2.3 22B dev GGUF quants (Phase 2A). All share the
    # ``base_diffusion_model_gguf`` role and the ``ltx-2.3-dev-gguf`` variant
    # group (set on the spec); the scanner differentiates by filename.
    "ltx-2.3-22b-dev-gguf-q4-k-m": ("gguf", "base_diffusion_model_gguf"),
    "ltx-2.3-22b-dev-gguf-ud-q4-k-m": ("gguf", "base_diffusion_model_gguf"),
    "ltx-2.3-22b-dev-gguf-q6-k": ("gguf", "base_diffusion_model_gguf"),
    "ltx-2.3-22b-dev-gguf-ud-q5-k-m": ("gguf", "base_diffusion_model_gguf"),
    # Gemma 3 mmproj (BF16) — Phase 3A: downloadable CP-backed artifact.
    # Lives inside the gemma GGUF folder artifact; detected via
    # descent-aware folder-child scan (see _FOLDER_CHILD_FILES).
    "gemma-3-12b-it-qat-gguf-mmproj": ("gguf", "gemma_mmproj"),
}


def _adapter_kind_to_artifact_kind(kind: AdapterKind) -> ArtifactKind:
    if kind == "ic_lora":
        return "control_adapter"
    if kind == "embeddings":
        return "scene_embeddings"
    return "lora"  # lora, distilled_lora


#: Scanner-only known artifacts (no download CP spec) that must be recognized
#: when present on disk so they do not show as unknown files. Canonical paths
#: are subfolder-only, consistent with all other known artifacts.
#:
#: NOTE: Fast-family Kijai/QuantStack distilled base-video artifacts (FP8 and
#: the seven QuantStack GGUF quants) are derived from the unified base-video
#: registry in :func:`_build_registry_base_video_artifacts` so the scanner and
#: the model-options endpoint share one source of truth. They are appended in
#: :func:`_build_canonical_artifacts`.
_EXTRA_KNOWN_ARTIFACTS: list[_CanonicalArtifact] = [
    _CanonicalArtifact(
        filename="ltx-2.3_text_projection_bf16.safetensors",
        artifact_kind="text_encoder",
        component_role="text_projection_file",
        canonical_relative_path="text_encoders/ltx-2.3_text_projection_bf16.safetensors",
        repo_id="Lightricks/LTX-2.3",
        expected_size_bytes=0,
        is_folder=False,
        gated=False,
        cp_id=None,
        adapter_id=None,
        section="full",
        display_name="LTX-2.3 text projection (BF16)",
        downloadable=False,
    ),
    _CanonicalArtifact(
        filename="LTX23_video_vae_bf16.safetensors",
        artifact_kind="vae",
        component_role="video_vae",
        canonical_relative_path="vae/LTX23_video_vae_bf16.safetensors",
        repo_id="Lightricks/LTX-2.3",
        expected_size_bytes=0,
        is_folder=False,
        gated=False,
        cp_id=None,
        adapter_id=None,
        section="full",
        display_name="LTX-2.3 video VAE (BF16)",
        downloadable=False,
    ),
    _CanonicalArtifact(
        filename="LTX23_audio_vae_bf16.safetensors",
        artifact_kind="vae",
        component_role="audio_vae",
        canonical_relative_path="vae/LTX23_audio_vae_bf16.safetensors",
        repo_id="Lightricks/LTX-2.3",
        expected_size_bytes=0,
        is_folder=False,
        gated=False,
        cp_id=None,
        adapter_id=None,
        section="addons",
        display_name="LTX-2.3 audio VAE (BF16)",
        downloadable=False,
    ),
    _CanonicalArtifact(
        filename="gemma-3-12b-it-qat-GGUF",
        artifact_kind="text_encoder",
        component_role="gemma_gguf",
        canonical_relative_path="text_encoders/unsloth/gemma-3-12b-it-qat-GGUF",
        repo_id="unsloth/gemma-3-12b-it-qat-GGUF",
        expected_size_bytes=0,
        is_folder=True,
        gated=False,
        cp_id=None,
        adapter_id=None,
        section="gguf",
        display_name="Gemma 3 12B IT QAT GGUF (text encoder)",
        variant_group="gemma-3-gguf",
        downloadable=False,
    ),
    # Gemma 3 mmproj (BF16) was a scanner-only artifact in Phase 2A; Phase 3A
    # promotes it to a downloadable CP-backed artifact
    # (gemma-3-12b-it-qat-gguf-mmproj, see _CP_KIND_MAP). The scanner detects
    # it via descent-aware folder-child lookup (_FOLDER_CHILD_FILES) so it
    # reports ``installed`` when present inside the gemma GGUF folder artifact
    # without leaking arbitrary internal files as unknown.
]


def _build_registry_base_video_artifacts() -> list[_CanonicalArtifact]:
    """Scanner-only base-video artifacts derived from the unified registry.

    Per plan §"Required source-of-truth fix", the scanner derives EVERY
    non-CP base-video registry entry (those with ``download_cp_id is None``)
    from the unified base-video registry so the scanner and the model-options
    endpoint share one source of truth. This covers the Fast-family Kijai
    distilled FP8, the seven QuantStack distilled GGUF quants, AND the
    Full-family official dev safetensors (``ltx-2.3-22b-dev``).

    CP-backed base entries (official distilled, unsloth dev GGUFs) are already
    provided by ``_CP_KIND_MAP`` + ``get_model_cp_spec`` and are skipped here to
    avoid duplicate filenames.
    """
    derived: list[_CanonicalArtifact] = []
    for entry in iter_base_video_registry_static_entries():
        if entry.download_cp_id is not None:
            continue  # CP-backed — already covered by _CP_KIND_MAP
        filename = Path(entry.canonical_relative_path).name
        derived.append(_CanonicalArtifact(
            filename=filename,
            artifact_kind=entry.artifact_kind,
            component_role=entry.component_role,
            canonical_relative_path=entry.canonical_relative_path,
            repo_id=entry.repo_id,
            expected_size_bytes=entry.expected_size_bytes,
            is_folder=False,
            gated=False,
            cp_id=None,
            adapter_id=None,
            section=entry.section,
            display_name=entry.label,
            variant_group=entry.variant_group,
            downloadable=entry.downloadable,
            remote_filename=entry.remote_filename,
        ))
    return derived


def _build_canonical_artifacts() -> list[_CanonicalArtifact]:
    """Build scanner-local canonical expectations from current runtime specs.

    Checkpoint canonical paths use ``spec.relative_path`` (subfolder-only,
    matches :func:`resolve_model_path`). Adapter canonical paths use
    ``adapters/<filename>`` (never bare at root). Extra known files (VAE,
    text projection, GGUF text encoder) are appended with their own subfolder
    canonicals. Every non-CP base-video registry entry (Kijai FP8, QuantStack
    distilled GGUFs, official dev safetensors) is derived from the unified
    base-video registry.
    """
    artifacts: list[_CanonicalArtifact] = []
    seen_filenames: set[str] = set()

    for cp_id, (kind, role) in _CP_KIND_MAP.items():
        spec = get_model_cp_spec(cp_id)
        filename = spec.relative_path.name
        seen_filenames.add(filename)
        artifacts.append(_CanonicalArtifact(
            filename=filename,
            artifact_kind=kind,
            component_role=role,
            canonical_relative_path=str(spec.relative_path),
            repo_id=spec.repo_id,
            expected_size_bytes=spec.expected_size_bytes,
            is_folder=spec.is_folder,
            gated=False,
            cp_id=cp_id,
            adapter_id=None,
            section=spec.section,
            display_name=spec.display_name or spec.description or filename,
            variant_group=spec.variant_group,
            downloadable=spec.downloadable,
            remote_filename=spec.remote_filename,
        ))

    for adapter_id, adapter in OFFICIAL_LTX23_ADAPTERS.items():
        filename = adapter.filename
        if filename in seen_filenames:
            continue
        seen_filenames.add(filename)
        artifacts.append(_CanonicalArtifact(
            filename=filename,
            artifact_kind=_adapter_kind_to_artifact_kind(adapter.kind),
            component_role=adapter_id,
            canonical_relative_path=f"{_ADAPTER_CANONICAL_SUBFOLDER}/{filename}",
            repo_id=adapter.repo_id,
            expected_size_bytes=adapter.expected_size_bytes,
            is_folder=False,
            gated=adapter_id in _GATED_ADAPTER_IDS,
            cp_id=ADAPTER_TO_CP_ID.get(adapter_id),
            adapter_id=adapter_id,
            section=adapter.section,
            display_name=adapter.display_name,
            variant_group=adapter.variant_group,
            downloadable=adapter.downloadable,
            remote_filename=filename,
        ))

    # Scanner-only known artifacts (no download CP).
    for extra in _EXTRA_KNOWN_ARTIFACTS:
        if extra.filename not in seen_filenames:
            seen_filenames.add(extra.filename)
            artifacts.append(extra)

    # Fast-family Kijai/QuantStack base-video artifacts derived from the
    # unified base-video registry (single source of truth shared with the
    # model-options endpoint). Skips filenames already covered above.
    for reg in _build_registry_base_video_artifacts():
        if reg.filename not in seen_filenames:
            seen_filenames.add(reg.filename)
            artifacts.append(reg)

    return artifacts


_CANONICAL_ARTIFACTS: list[_CanonicalArtifact] = _build_canonical_artifacts()
_FILE_ARTIFACTS_BY_NAME: dict[str, _CanonicalArtifact] = {
    a.filename: a for a in _CANONICAL_ARTIFACTS if not a.is_folder
}
_FOLDER_ARTIFACTS_BY_NAME: dict[str, _CanonicalArtifact] = {
    a.filename: a for a in _CANONICAL_ARTIFACTS if a.is_folder
}


def _build_folder_child_files() -> dict[str, list[str]]:
    """Map folder-artifact filename → known child file-artifact filenames that
    live inside the folder's canonical path.

    Derived automatically from canonical paths so parent/child relationships
    stay in sync with the catalog: a file artifact is a child of a folder
    artifact when the immediate parent directory of the file's canonical path
    has the same name as the folder artifact. Only known children are matched;
    arbitrary unknown files inside a folder artifact are never leaked (the
    folder is treated as a folder artifact and descent is blocked).

    Example: ``mmproj-BF16.gguf`` whose canonical path is
    ``text_encoders/unsloth/gemma-3-12b-it-qat-GGUF/mmproj-BF16.gguf`` is
    registered as a child of the ``gemma-3-12b-it-qat-GGUF`` folder artifact.
    """
    folder_names = set(_FOLDER_ARTIFACTS_BY_NAME)
    result: dict[str, list[str]] = {}
    for file_art in _CANONICAL_ARTIFACTS:
        if file_art.is_folder:
            continue
        parent_name = Path(file_art.canonical_relative_path).parent.name
        if parent_name in folder_names:
            result.setdefault(parent_name, []).append(file_art.filename)
    return result


#: Known child file artifacts inside each matched folder artifact (plan §9).
#: Drives descent-aware child detection without exposing arbitrary internal
#: files as unknown.
_FOLDER_CHILD_FILES: dict[str, list[str]] = _build_folder_child_files()


def _source_url(repo_id: str, filename: str, is_folder: bool) -> str:
    base = f"https://huggingface.co/{repo_id}"
    if is_folder:
        return base
    return f"{base}/resolve/main/{filename}"


def _canonical_path(models_dir: Path, canonical: _CanonicalArtifact) -> Path:
    return models_dir / canonical.canonical_relative_path


def _safe_file_size(path: Path) -> int | None:
    try:
        return path.stat().st_size if path.is_file() else None
    except OSError:
        return None


def _detect_partial_suffix(lower_name: str) -> str:
    for suf in _PARTIAL_SUFFIXES:
        if lower_name.endswith(suf):
            return suf
    return ""


def _build_artifact(
    canonical: _CanonicalArtifact,
    paths: list[Path],
    canonical_path: Path,
) -> ModelLibraryArtifact:
    source_url = _source_url(canonical.repo_id, canonical.filename, canonical.is_folder)
    canonical_relative = canonical.canonical_relative_path
    support: str = "gated" if canonical.gated else "supported"

    if not paths:
        return ModelLibraryArtifact(
            filename=canonical.filename,
            artifact_kind=canonical.artifact_kind,
            component_role=canonical.component_role,
            status="missing",
            scanner_confidence="exact_catalog_match",
            canonical_relative_path=canonical_relative,
            expected_size_bytes=canonical.expected_size_bytes,
            repo_id=canonical.repo_id,
            source_url=source_url,
            is_folder=canonical.is_folder,
            absolute_paths=[],
            preferred_path=None,
            size_bytes=None,
            support_status=support,
            gated=canonical.gated,
            notes="",
            cp_id=canonical.cp_id,
            adapter_id=canonical.adapter_id,
            section=canonical.section,
            display_name=canonical.display_name,
            variant_group=canonical.variant_group,
            downloadable=canonical.downloadable,
            remote_filename=canonical.remote_filename,
        )

    absolute_paths = [str(p) for p in paths]
    canonical_match: Path | None = next((p for p in paths if p == canonical_path), None)
    preferred = canonical_match if canonical_match is not None else paths[0]
    preferred_size = _safe_file_size(preferred)

    if len(paths) == 1:
        single_path = paths[0]
        if single_path == canonical_path:
            status: ScanArtifactStatus = "installed"
            confidence: ScannerConfidence = "exact_catalog_match"
            notes = ""
        else:
            status = "wrong_folder_usable"
            confidence = "filename_match"
            notes = ""
    else:
        status = "duplicate"
        confidence = "exact_catalog_match" if canonical_match is not None else "filename_match"
        notes = f"Found in {len(paths)} locations"

    return ModelLibraryArtifact(
        filename=canonical.filename,
        artifact_kind=canonical.artifact_kind,
        component_role=canonical.component_role,
        status=status,
        scanner_confidence=confidence,
        canonical_relative_path=canonical_relative,
        expected_size_bytes=canonical.expected_size_bytes,
        repo_id=canonical.repo_id,
        source_url=source_url,
        is_folder=canonical.is_folder,
        absolute_paths=absolute_paths,
        preferred_path=str(preferred),
        size_bytes=preferred_size,
        support_status=support,
        gated=canonical.gated,
        notes=notes,
        cp_id=canonical.cp_id,
        adapter_id=canonical.adapter_id,
        section=canonical.section,
        display_name=canonical.display_name,
        variant_group=canonical.variant_group,
        downloadable=canonical.downloadable,
        remote_filename=canonical.remote_filename,
    )


def scan_models(models_dir: Path) -> ModelLibraryScanResponse:
    """Recursively scan *models_dir* (read-only) and return a typed catalog.

    The scanner never creates, moves, deletes, or downloads anything. It
    classifies known artifacts by status (``installed`` / ``missing`` /
    ``wrong_folder_usable`` / ``duplicate``) and reports unknown and partial
    files in separate lists.
    """
    discovered: dict[str, list[Path]] = {}
    unknown_files: list[UnknownFile] = []
    partial_files: list[PartialFile] = []

    if models_dir.exists() and models_dir.is_dir():
        for root, dirs, files in os.walk(models_dir):
            root_path = Path(root)

            # Detect folder artifacts and prevent descending into them so their
            # internal files are not misclassified as unknowns.
            matched_folder_names: set[str] = set()
            for dir_name in dirs:
                if dir_name not in _FOLDER_ARTIFACTS_BY_NAME:
                    continue
                full = root_path / dir_name
                try:
                    entries = list(full.iterdir()) if full.is_dir() else []
                except OSError:
                    entries = []

                if not entries:
                    # Empty or unreadable — nothing to detect, no descent block.
                    continue

                # Block descent into any non-empty matched folder artifact so
                # arbitrary internal files are not leaked as unknown.
                matched_folder_names.add(dir_name)

                # Known child file artifacts (e.g. mmproj-BF16.gguf inside the
                # gemma GGUF folder) are detected explicitly here — descent-
                # aware detection (Phase 3A, plan §9). They resolve against
                # their own canonical path, independent of the parent folder's
                # install state. Unknown children are intentionally NOT emitted.
                known_children = _FOLDER_CHILD_FILES.get(dir_name, ())
                for child_filename in known_children:
                    child_full = full / child_filename
                    try:
                        if child_full.is_file():
                            discovered.setdefault(child_filename, []).append(child_full)
                    except OSError:
                        pass

                # Parent folder evidence: at least one entry that is NOT a known
                # child file artifact. Known children alone do NOT count as
                # evidence for the parent folder artifact — otherwise a folder
                # containing only a child projection file (e.g. mmproj-BF16.gguf
                # inside the gemma GGUF folder) would wrongly report the parent
                # gemma_gguf folder artifact as installed. An actual Gemma GGUF
                # model file (or any other real content) is required.
                if any(entry.name not in known_children for entry in entries):
                    discovered.setdefault(dir_name, []).append(full)

            dirs[:] = sorted(
                d for d in dirs
                if d not in _SKIP_DIRS and d not in matched_folder_names
            )

            for file_name in sorted(files):
                full = root_path / file_name
                try:
                    rel = full.relative_to(models_dir)
                    size = full.stat().st_size
                except OSError:
                    continue

                lower_name = file_name.lower()
                partial_suffix = _detect_partial_suffix(lower_name)
                if partial_suffix:
                    partial_files.append(PartialFile(
                        absolute_path=str(full),
                        relative_path=str(rel),
                        size_bytes=size,
                        suffix=partial_suffix,
                    ))
                elif file_name in _FILE_ARTIFACTS_BY_NAME:
                    discovered.setdefault(file_name, []).append(full)
                else:
                    unknown_files.append(UnknownFile(
                        absolute_path=str(full),
                        relative_path=str(rel),
                        size_bytes=size,
                    ))

    artifacts: list[ModelLibraryArtifact] = []
    for canonical in _CANONICAL_ARTIFACTS:
        paths = sorted(discovered.get(canonical.filename, []))
        canonical_path = _canonical_path(models_dir, canonical)
        artifacts.append(_build_artifact(canonical, paths, canonical_path))

    artifacts.sort(key=lambda a: (a.component_role, a.filename))
    unknown_files.sort(key=lambda f: f.relative_path)
    partial_files.sort(key=lambda f: f.relative_path)

    return ModelLibraryScanResponse(
        models_dir=str(models_dir),
        scanned_at=datetime.now(timezone.utc).isoformat(),
        artifacts=artifacts,
        unknown_files=unknown_files,
        partial_files=partial_files,
    )
