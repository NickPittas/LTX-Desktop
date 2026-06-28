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

#: Adapters that remain workflow-gated even when installed (see plan §8).
_GATED_ADAPTER_IDS: frozenset[AdapterID] = frozenset({"hdr", "hdr_scene_embeddings"})

#: Canonical subfolder for adapter (IC-LoRA / distilled-LoRA / scene-embedding) files.
_ADAPTER_CANONICAL_SUBFOLDER = "adapters"


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
_EXTRA_KNOWN_ARTIFACTS: list[_CanonicalArtifact] = [
    _CanonicalArtifact(
        filename="ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
        artifact_kind="diffusion_model",
        component_role="base_diffusion_model_fp8",
        canonical_relative_path="diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
        repo_id="Lightricks/LTX-2.3",
        expected_size_bytes=0,
        is_folder=False,
        gated=False,
        cp_id=None,
        adapter_id=None,
    ),
    _CanonicalArtifact(
        filename="LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
        artifact_kind="gguf",
        component_role="base_diffusion_model_gguf",
        canonical_relative_path="gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
        repo_id="QuantStack/LTX-2.3-GGUF",
        expected_size_bytes=0,
        is_folder=False,
        gated=False,
        cp_id=None,
        adapter_id=None,
    ),
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
    ),
]


def _build_canonical_artifacts() -> list[_CanonicalArtifact]:
    """Build scanner-local canonical expectations from current runtime specs.

    Checkpoint canonical paths use ``spec.relative_path`` (subfolder-only,
    matches :func:`resolve_model_path`). Adapter canonical paths use
    ``adapters/<filename>`` (never bare at root). Extra known files (VAE,
    text projection, alternate transformer builds, GGUF text encoder) are
    appended with their own subfolder canonicals.
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
        ))

    # Scanner-only known artifacts (no download CP).
    for extra in _EXTRA_KNOWN_ARTIFACTS:
        if extra.filename not in seen_filenames:
            seen_filenames.add(extra.filename)
            artifacts.append(extra)

    return artifacts


_CANONICAL_ARTIFACTS: list[_CanonicalArtifact] = _build_canonical_artifacts()
_FILE_ARTIFACTS_BY_NAME: dict[str, _CanonicalArtifact] = {
    a.filename: a for a in _CANONICAL_ARTIFACTS if not a.is_folder
}
_FOLDER_ARTIFACTS_BY_NAME: dict[str, _CanonicalArtifact] = {
    a.filename: a for a in _CANONICAL_ARTIFACTS if a.is_folder
}


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
                    is_nonempty = full.is_dir() and any(full.iterdir())
                except OSError:
                    is_nonempty = False
                if is_nonempty:
                    discovered.setdefault(dir_name, []).append(full)
                    matched_folder_names.add(dir_name)

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
