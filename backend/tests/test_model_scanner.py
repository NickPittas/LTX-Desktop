"""Tests for the read-only model library scanner and catalog endpoint (Phase 1).

Canonical paths are **subfolder-only**: no known artifact is ever canonical at
the models root. Root-level placements are ``wrong_folder_usable``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.model_scanner import scan_models
from tests.conftest import TEST_ADMIN_TOKEN
from tests.http_error_assertions import assert_http_error

_ADMIN_HEADERS = {"X-Admin-Token": TEST_ADMIN_TOKEN}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _write(parent: Path, name: str, data: bytes = b"\x00model") -> Path:
    parent.mkdir(parents=True, exist_ok=True)
    path = parent / name
    path.write_bytes(data)
    return path


def _snapshot_tree(root: Path) -> dict[str, int]:
    """Deterministic snapshot of all files: relative_path → size_bytes."""
    result: dict[str, int] = {}
    if not root.exists():
        return result
    for path in sorted(root.rglob("*")):
        if path.is_file():
            result[str(path.relative_to(root))] = path.stat().st_size
    return result


def _find(artifacts: list[Any], role: str) -> Any:
    return next(a for a in artifacts if a.component_role == role)


def _find_by_filename(artifacts: list[Any], filename: str) -> Any:
    """Find a scanner artifact by filename.

    Needed because multiple base-video artifacts now share the
    ``base_diffusion_model`` role (the official distilled monolith AND the
    official dev safetensors are both ``base_diffusion_model``).
    """
    return next(a for a in artifacts if a.filename == filename)


# ------------------------------------------------------------------
# Scanner unit tests
# ------------------------------------------------------------------


class TestScannerReadonly:
    def test_scanner_creates_nothing(self, tmp_path):
        models = tmp_path / "models"
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")

        before = _snapshot_tree(models)
        scan_models(models)
        after = _snapshot_tree(models)
        assert before == after

    def test_scanner_moves_deletes_nothing_with_full_layout(self, tmp_path):
        models = tmp_path / "models"
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")
        _write(models / "adapters", "some_download.part", b"partial")
        _write(models, "random_unknown.bin")
        _write(models / "latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")
        _write(models / "diffusion_models", "ltx-2.3-22b-distilled.safetensors")
        gemma = models / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"
        _write(gemma, "model.safetensors")

        before = _snapshot_tree(models)
        scan_models(models)
        after = _snapshot_tree(models)
        assert before == after


class TestScannerStatuses:
    def test_no_canonical_path_at_root(self, tmp_path):
        """Requirement: no known artifact canonical_relative_path may lack a folder component."""
        models = tmp_path / "models"
        result = scan_models(models)
        for art in result.artifacts:
            assert "/" in art.canonical_relative_path, (
                f"{art.filename}: canonical_relative_path {art.canonical_relative_path!r}"
                f" has no subfolder component"
            )

    def test_subfolder_hdr_installed_and_supported(self, tmp_path):
        """HDR at canonical adapters/ path → installed + supported (not gated).

        HDR LoRA and scene-embedding support asset are no longer scanner-gated;
        installed copies report ``supported``. Selectability of the scene
        embeddings as a standalone adapter is owned by the handler.
        """
        models = tmp_path / "models"
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors")

        result = scan_models(models)

        hdr = _find(result.artifacts, "hdr")
        assert hdr.status == "installed"
        assert hdr.gated is False
        assert hdr.support_status == "supported"

        hdr_emb = _find(result.artifacts, "hdr_scene_embeddings")
        assert hdr_emb.status == "installed"
        assert hdr_emb.gated is False
        assert hdr_emb.support_status == "supported"

    def test_root_hdr_wrong_folder_but_supported(self, tmp_path):
        """HDR at root (non-canonical) → wrong_folder_usable, supported (not gated)."""
        models = tmp_path / "models"
        _write(models, "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")

        result = scan_models(models)
        hdr = _find(result.artifacts, "hdr")
        assert hdr.status == "wrong_folder_usable"
        assert hdr.gated is False
        assert hdr.support_status == "supported"

    def test_subfolder_non_hdr_adapter_installed(self, tmp_path):
        """Adapter at canonical adapters/ path → installed."""
        models = tmp_path / "models"
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")

        result = scan_models(models)
        art = _find(result.artifacts, "ingredients")
        assert art.status == "installed"
        assert art.gated is False
        assert art.support_status == "supported"

    def test_subfolder_base_model_installed(self, tmp_path):
        """Base model at canonical diffusion_models/ path → installed."""
        models = tmp_path / "models"
        _write(models / "diffusion_models", "ltx-2.3-22b-distilled.safetensors")

        result = scan_models(models)
        # The distilled monolith and the official dev safetensors share the
        # ``base_diffusion_model`` role; look up the distilled by filename.
        base = _find_by_filename(result.artifacts, "ltx-2.3-22b-distilled.safetensors")
        assert base.status == "installed"

    def test_root_base_model_wrong_folder(self, tmp_path):
        """Base model at root (non-canonical) → wrong_folder_usable."""
        models = tmp_path / "models"
        _write(models, "ltx-2.3-22b-distilled.safetensors")

        result = scan_models(models)
        base = _find_by_filename(result.artifacts, "ltx-2.3-22b-distilled.safetensors")
        assert base.status == "wrong_folder_usable"

    def test_subfolder_upscaler_installed(self, tmp_path):
        """Upscaler at canonical latent_upscale_models/ path → installed."""
        models = tmp_path / "models"
        _write(models / "latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

        result = scan_models(models)
        upscaler = _find(result.artifacts, "spatial_upscaler")
        assert upscaler.status == "installed"
        assert upscaler.preferred_path is not None
        assert upscaler.size_bytes is not None

    def test_root_upscaler_wrong_folder(self, tmp_path):
        """Upscaler at root (non-canonical) → wrong_folder_usable."""
        models = tmp_path / "models"
        _write(models, "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

        result = scan_models(models)
        upscaler = _find(result.artifacts, "spatial_upscaler")
        assert upscaler.status == "wrong_folder_usable"

    def test_wrong_folder_usable(self, tmp_path):
        models = tmp_path / "models"
        # ingredients adapter placed in diffusion_models/ instead of adapters/
        _write(models / "diffusion_models", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")

        result = scan_models(models)
        art = _find(result.artifacts, "ingredients")
        assert art.status == "wrong_folder_usable"
        assert art.preferred_path is not None
        assert "diffusion_models" in art.preferred_path
        assert art.scanner_confidence == "filename_match"

    def test_missing_artifact_has_expected_path_and_source(self, tmp_path):
        models = tmp_path / "models"
        result = scan_models(models)

        missing = _find(result.artifacts, "lipdub")
        assert missing.status == "missing"
        assert missing.preferred_path is None
        assert missing.absolute_paths == []
        assert missing.canonical_relative_path == "adapters/ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors"
        assert "huggingface.co" in missing.source_url
        assert "LipDub" in missing.source_url

    def test_missing_diffusion_model_expected_path(self, tmp_path):
        models = tmp_path / "models"
        result = scan_models(models)

        # The distilled monolith and the official dev safetensors share the
        # ``base_diffusion_model`` role; look up the distilled by filename.
        base = _find_by_filename(result.artifacts, "ltx-2.3-22b-distilled.safetensors")
        assert base.status == "missing"
        assert base.canonical_relative_path == "diffusion_models/ltx-2.3-22b-distilled.safetensors"

    def test_duplicate_reports_all_paths_and_canonical_preferred(self, tmp_path):
        models = tmp_path / "models"
        # Canonical = adapters/ level; wrong copy at root
        canonical = _write(models / "adapters", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")
        wrong = _write(models, "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")

        result = scan_models(models)
        art = _find(result.artifacts, "ingredients")
        assert art.status == "duplicate"
        assert len(art.absolute_paths) == 2
        # Preferred = canonical match (adapters/)
        assert art.preferred_path == str(canonical)
        # All paths reported
        path_set = set(art.absolute_paths)
        assert str(canonical) in path_set
        assert str(wrong) in path_set

    def test_duplicate_without_canonical_picks_sorted_first(self, tmp_path):
        models = tmp_path / "models"
        loc_a = _write(models / "aaa", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")
        _write(models / "zzz", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")

        result = scan_models(models)
        art = _find(result.artifacts, "ingredients")
        assert art.status == "duplicate"
        assert len(art.absolute_paths) == 2
        # No canonical match → sorted-first (aaa before zzz)
        assert art.preferred_path == str(loc_a)

    def test_unknown_files_separated(self, tmp_path):
        models = tmp_path / "models"
        _write(models, "random_unknown_file.bin")
        _write(models / "adapters", "mystery_adapter.safetensors")

        result = scan_models(models)
        assert len(result.unknown_files) >= 2
        for f in result.unknown_files:
            assert f.size_bytes > 0
            assert f.relative_path != ""

    def test_partial_files_not_installed(self, tmp_path):
        models = tmp_path / "models"
        _write(models / "adapters", "download.part", b"partial")
        _write(models / "adapters", "other.tmp", b"temp")

        result = scan_models(models)
        assert len(result.partial_files) == 2
        for f in result.partial_files:
            assert f.suffix in (".part", ".tmp")
        # Partials must NOT appear as installed artifacts
        for art in result.artifacts:
            assert ".part" not in art.filename
            assert ".tmp" not in art.filename

    def test_skips_downloading_dir(self, tmp_path):
        models = tmp_path / "models"
        _write(models / ".downloading", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")

        result = scan_models(models)
        # File in .downloading must not be discovered
        ingredients = _find(result.artifacts, "ingredients")
        assert ingredients.status == "missing"
        # And must not appear in unknown_files
        for f in result.unknown_files:
            assert ".downloading" not in f.relative_path

    def test_empty_dir_all_missing(self, tmp_path):
        result = scan_models(tmp_path / "nonexistent")
        assert len(result.artifacts) > 0
        assert all(a.status == "missing" for a in result.artifacts)
        assert result.unknown_files == []
        assert result.partial_files == []

    def test_folder_artifact_internal_files_not_unknown(self, tmp_path):
        """Folder artifact at canonical subfolder: installed, internal files not unknown."""
        models = tmp_path / "models"
        gemma = models / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"
        _write(gemma, "model.safetensors")
        _write(gemma, "tokenizer.model")

        result = scan_models(models)

        gemma_art = _find(result.artifacts, "gemma")
        assert gemma_art.is_folder is True
        # Canonical is text_encoders/gemma-...; placed there → installed
        assert gemma_art.status == "installed"

        # Internal files must not appear as unknowns
        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        assert "model.safetensors" not in unknown_names
        assert "tokenizer.model" not in unknown_names

    def test_root_folder_artifact_wrong_folder(self, tmp_path):
        """Folder artifact at root (non-canonical) → wrong_folder_usable."""
        models = tmp_path / "models"
        gemma = models / "gemma-3-12b-it-qat-q4_0-unquantized"
        _write(gemma, "model.safetensors")

        result = scan_models(models)
        gemma_art = _find(result.artifacts, "gemma")
        assert gemma_art.status == "wrong_folder_usable"

    def test_extra_known_files_recognized(self, tmp_path):
        """Scanner-only known files (VAE, text projection, FP8 transformer) are
        recognized — not reported as unknown."""
        models = tmp_path / "models"
        _write(models / "vae", "LTX23_video_vae_bf16.safetensors")
        _write(models / "vae", "LTX23_audio_vae_bf16.safetensors")
        _write(models / "text_encoders", "ltx-2.3_text_projection_bf16.safetensors")
        _write(
            models / "diffusion_models",
            "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors",
        )

        result = scan_models(models)

        video_vae = _find(result.artifacts, "video_vae")
        assert video_vae.status == "installed"

        audio_vae = _find(result.artifacts, "audio_vae")
        assert audio_vae.status == "installed"

        tp = _find(result.artifacts, "text_projection_file")
        assert tp.status == "installed"

        fp8 = _find(result.artifacts, "base_diffusion_model_fp8")
        assert fp8.status == "installed"

        # None of these should appear as unknown
        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        assert "LTX23_video_vae_bf16.safetensors" not in unknown_names
        assert "LTX23_audio_vae_bf16.safetensors" not in unknown_names

    # ------------------------------------------------------------------
    # Phase 2A — unsloth LTX-2.3 dev GGUF canonical classification (plan §4)
    # ------------------------------------------------------------------

    def test_dev_gguf_at_canonical_unsloth_path_installed(self, tmp_path):
        """The four dev GGUF quants at diffusion_models/unsloth/LTX-2.3-GGUF/
        are canonical installed — not unknown, not wrong-folder."""
        models = tmp_path / "models"
        gguf_dir = models / "diffusion_models" / "unsloth" / "LTX-2.3-GGUF"
        _write(gguf_dir, "ltx-2.3-22b-dev-Q4_K_M.gguf")
        _write(gguf_dir, "ltx-2.3-22b-dev-UD-Q4_K_M.gguf")
        _write(gguf_dir, "ltx-2.3-22b-dev-Q6_K.gguf")
        _write(gguf_dir, "ltx-2.3-22b-dev-UD-Q5_K_M.gguf")

        result = scan_models(models)

        installed = {
            "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "ltx-2.3-22b-dev-UD-Q4_K_M.gguf",
            "ltx-2.3-22b-dev-Q6_K.gguf",
            "ltx-2.3-22b-dev-UD-Q5_K_M.gguf",
        }
        for art in result.artifacts:
            if art.filename in installed:
                assert art.status == "installed", art.filename
                assert art.scanner_confidence == "exact_catalog_match", art.filename
                assert art.artifact_kind == "gguf", art.filename
                assert art.component_role == "base_diffusion_model_gguf", art.filename
                assert art.cp_id in {
                    "ltx-2.3-22b-dev-gguf-q4-k-m",
                    "ltx-2.3-22b-dev-gguf-ud-q4-k-m",
                    "ltx-2.3-22b-dev-gguf-q6-k",
                    "ltx-2.3-22b-dev-gguf-ud-q5-k-m",
                }, art.filename
        # None of the GGUF files appear as unknown.
        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        assert not (installed & unknown_names)

    def test_dev_gguf_canonical_path_matches_resolve_model_path(self, tmp_path):
        from runtime_config.model_download_specs import resolve_model_path

        models = tmp_path / "models"
        gguf_dir = models / "diffusion_models" / "unsloth" / "LTX-2.3-GGUF"
        _write(gguf_dir, "ltx-2.3-22b-dev-Q4_K_M.gguf")

        result = scan_models(models)
        art = next(a for a in result.artifacts if a.filename == "ltx-2.3-22b-dev-Q4_K_M.gguf")
        assert art.preferred_path == str(
            resolve_model_path(models, "ltx-2.3-22b-dev-gguf-q4-k-m")
        )

    def test_dev_gguf_at_wrong_folder_classified_usable(self, tmp_path):
        """A dev GGUF placed outside the canonical unsloth path is NOT installed
        (wrong_folder_usable), so it would be re-fetched to the canonical path."""
        models = tmp_path / "models"
        _write(models / "diffusion_models", "ltx-2.3-22b-dev-Q4_K_M.gguf")

        result = scan_models(models)
        art = next(a for a in result.artifacts if a.filename == "ltx-2.3-22b-dev-Q4_K_M.gguf")
        assert art.status == "wrong_folder_usable"
        assert art.scanner_confidence == "filename_match"

    # ------------------------------------------------------------------
    # Unified base-video registry — Kijai/QuantStack scanner artifacts
    # ------------------------------------------------------------------

    def test_scanner_base_video_registry_artifacts_are_selectable(self, tmp_path):
        """Scanner-recognized Fast-family Kijai/QuantStack base-video artifacts
        carry the same metadata (canonical path, repo, section, variant group,
        role) as the unified base-video registry that drives
        ``GET /api/models/model-options`` (plan: source-of-truth fix).

        Places every scanner-only Fast-family base-video file at its canonical
        registry path and verifies each is ``installed`` (not unknown) and
        carries registry-matching catalog metadata.
        """
        from services.base_video_model_registry import (
            iter_base_video_registry_static_entries,
        )

        models = tmp_path / "models"

        # Materialize every scanner-only Fast-family base-video file at its
        # canonical registry path (Kijai FP8 + seven QuantStack distilled GGUFs).
        fast_scanner_only = [
            e for e in iter_base_video_registry_static_entries()
            if e.download_cp_id is None and e.pipeline_family == "fast"
        ]
        assert {e.id for e in fast_scanner_only} == {
            "ltx-2.3-22b-distilled-fp8-kijai-v3",
            "ltx-2.3-22b-distilled-gguf-quantstack-q2-k",
            "ltx-2.3-22b-distilled-gguf-quantstack-q3-k-s",
            "ltx-2.3-22b-distilled-gguf-quantstack-q3-k-m",
            "ltx-2.3-22b-distilled-gguf-quantstack-q4-k-s",
            "ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m",
            "ltx-2.3-22b-distilled-gguf-quantstack-q5-k-s",
            "ltx-2.3-22b-distilled-gguf-quantstack-q5-k-m",
        }
        for entry in fast_scanner_only:
            path = models / entry.canonical_relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"\x00model")

        result = scan_models(models)

        # Cross-check each scanner artifact against its registry metadata.
        for entry in fast_scanner_only:
            filename = Path(entry.canonical_relative_path).name
            art = next(
                a for a in result.artifacts if a.filename == filename
            )
            assert art.status == "installed", entry.id
            assert art.canonical_relative_path == entry.canonical_relative_path
            assert art.repo_id == entry.repo_id
            assert art.section == entry.section
            assert art.variant_group == entry.variant_group
            assert art.component_role == entry.component_role
            assert art.artifact_kind == entry.artifact_kind
            assert art.downloadable is False  # scanner-only (no download CP)
            # The canonical placement path the model-options endpoint would
            # surface matches the scanner's preferred path.
            assert art.preferred_path == str(models / entry.canonical_relative_path)

        # None of the Kijai/QuantStack files leak as unknown.
        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        for entry in fast_scanner_only:
            assert Path(entry.canonical_relative_path).name not in unknown_names

        # Spot-check the Kijai FP8 and a QuantStack GGUF carry the exact
        # metadata model-options uses to identify them.
        kijai = next(
            a for a in result.artifacts if a.component_role == "base_diffusion_model_fp8"
        )
        assert kijai.filename == (
            "ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors"
        )
        assert kijai.repo_id == "Kijai/LTX2.3_comfy"
        assert kijai.section == "kijai"
        assert kijai.variant_group == "ltx-2.3-distilled-fp8"

        quantstack = next(
            a for a in result.artifacts
            if a.filename == "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf"
        )
        assert quantstack.repo_id == "QuantStack/LTX-2.3-GGUF"
        assert quantstack.section == "gguf"
        assert quantstack.variant_group == "ltx-2.3-distilled-gguf"
        assert quantstack.component_role == "base_diffusion_model_gguf"

    # ------------------------------------------------------------------
    # Phase 2A — catalog grouping metadata (plan §7)
    # ------------------------------------------------------------------

    def test_dev_gguf_artifacts_carry_section_and_variant_group(self, tmp_path):
        models = tmp_path / "models"
        result = scan_models(models)

        gguf_basenames = {
            "ltx-2.3-22b-dev-Q4_K_M.gguf",
            "ltx-2.3-22b-dev-UD-Q4_K_M.gguf",
            "ltx-2.3-22b-dev-Q6_K.gguf",
            "ltx-2.3-22b-dev-UD-Q5_K_M.gguf",
        }
        gguf_arts = [a for a in result.artifacts if a.filename in gguf_basenames]
        assert len(gguf_arts) == 4
        for art in gguf_arts:
            assert art.section == "gguf", art.filename
            assert art.variant_group == "ltx-2.3-dev-gguf", art.filename
            assert art.downloadable is True, art.filename
            assert art.display_name, art.filename

    def test_sections_assign_existing_artifacts(self, tmp_path):
        """Section grouping: full official set vs add-ons vs gguf (plan §2/§7)."""
        models = tmp_path / "models"
        result = scan_models(models)
        by_role = {a.component_role: a for a in result.artifacts}

        # Full section: base model, upscaler, gemma text encoder, video VAE,
        # text projection.
        assert by_role["base_diffusion_model"].section == "full"
        assert by_role["spatial_upscaler"].section == "full"
        assert by_role["gemma"].section == "full"
        assert by_role["video_vae"].section == "full"
        assert by_role["text_projection_file"].section == "full"

        # Add-ons & Controls: IC-LoRAs, depth/pose/person processors, image gen.
        assert by_role["ingredients"].section == "addons"
        assert by_role["depth_processor"].section == "addons"
        assert by_role["person_detector"].section == "addons"
        assert by_role["pose_processor"].section == "addons"
        assert by_role["image_gen_model"].section == "addons"

        # GGUF section: dev quants + gemma GGUF folder + mmproj + distilled GGUF.
        assert by_role["base_diffusion_model_gguf"].section == "gguf"
        assert by_role["gemma_gguf"].section == "gguf"
        assert by_role["gemma_mmproj"].section == "gguf"

    def test_gemma_mmproj_is_first_class_downloadable_artifact(self, tmp_path):
        """mmproj-BF16.gguf is a first-class downloadable CP-backed catalog
        entry (Phase 3A, plan §9 Option A).

        Phase 2A registered it scanner-only (downloadable=False) because the
        scanner could not detect a file inside the matched gemma GGUF folder
        artifact. Phase 3A promotes it to a downloadable CP
        (``gemma-3-12b-it-qat-gguf-mmproj``) and the scanner is now
        descent-aware for known folder children.
        """
        models = tmp_path / "models"
        result = scan_models(models)

        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        assert mmproj.filename == "mmproj-BF16.gguf"
        assert mmproj.artifact_kind == "gguf"
        assert mmproj.repo_id == "unsloth/gemma-3-12b-it-qat-GGUF"
        assert mmproj.expected_size_bytes == 854_200_448
        assert mmproj.section == "gguf"
        assert mmproj.display_name == "Gemma 3 mmproj BF16"
        assert mmproj.variant_group == "gemma-3-gguf"
        # Phase 3A promotion: now downloadable and CP-backed.
        assert mmproj.downloadable is True
        assert mmproj.cp_id == "gemma-3-12b-it-qat-gguf-mmproj"
        # remote_filename is None because the HF remote basename equals the
        # local basename (no remote-name promotion change needed).
        assert mmproj.remote_filename is None
        # Canonical placement is inside the gemma GGUF folder.
        assert mmproj.canonical_relative_path == (
            "text_encoders/unsloth/gemma-3-12b-it-qat-GGUF/mmproj-BF16.gguf"
        )
        # Source link points at the HF resolve URL.
        assert "mmproj-BF16.gguf" in mmproj.source_url
        # Absent on disk → missing.
        assert mmproj.status == "missing"

    def test_mmproj_installed_inside_gemma_gguf_folder_artifact(self, tmp_path):
        """Descent-aware detection (Phase 3A): when the gemma GGUF folder
        artifact is present at its canonical path and contains
        ``mmproj-BF16.gguf``, the mmproj artifact must report ``installed`` at
        the canonical path — not missing/unknown. The scanner blocks full
        descent into matched folder artifacts but checks known children
        explicitly."""
        models = tmp_path / "models"
        gguf_folder = models / "text_encoders" / "unsloth" / "gemma-3-12b-it-qat-GGUF"
        gguf_folder.mkdir(parents=True)
        (gguf_folder / "mmproj-BF16.gguf").write_bytes(b"\x00mmproj")
        # An arbitrary unknown sibling inside the folder artifact.
        (gguf_folder / "random-internal-file.bin").write_bytes(b"\x00junk")
        # The actual gemma GGUF text-encoder file.
        (gguf_folder / "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf").write_bytes(b"\x00gemma")

        result = scan_models(models)

        # Folder artifact detected as installed.
        gemma_gguf = next(a for a in result.artifacts if a.component_role == "gemma_gguf")
        assert gemma_gguf.status == "installed"
        assert gemma_gguf.is_folder is True

        # mmproj detected as installed at the canonical path (descent-aware).
        from runtime_config.model_download_specs import resolve_model_path

        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        assert mmproj.status == "installed"
        assert mmproj.scanner_confidence == "exact_catalog_match"
        assert mmproj.preferred_path == str(
            resolve_model_path(models, "gemma-3-12b-it-qat-gguf-mmproj")
        )
        assert mmproj.size_bytes is not None

    def test_mmproj_missing_when_folder_artifact_absent(self, tmp_path):
        """Without the gemma GGUF folder artifact, mmproj is missing (not
        wrongly detected from an unrelated folder)."""
        models = tmp_path / "models"
        # Place mmproj at root — no folder artifact context.
        _write(models, "mmproj-BF16.gguf")

        result = scan_models(models)
        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        # Root placement is non-canonical → wrong_folder_usable (filename match).
        assert mmproj.status == "wrong_folder_usable"
        assert mmproj.scanner_confidence == "filename_match"

    def test_no_arbitrary_unknown_child_leakage_from_folder_artifact(self, tmp_path):
        """Arbitrary unknown files inside a matched folder artifact are NOT
        emitted as unknown (the folder is intentionally treated as a folder
        artifact; only known children are matched)."""
        models = tmp_path / "models"
        gguf_folder = models / "text_encoders" / "unsloth" / "gemma-3-12b-it-qat-GGUF"
        gguf_folder.mkdir(parents=True)
        (gguf_folder / "mmproj-BF16.gguf").write_bytes(b"\x00mmproj")
        (gguf_folder / "random-internal-a.bin").write_bytes(b"\x00a")
        (gguf_folder / "random-internal-b.bin").write_bytes(b"\x00b")

        result = scan_models(models)

        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        # Known child is matched (not unknown).
        assert "mmproj-BF16.gguf" not in unknown_names
        # Arbitrary internal files are NOT leaked as unknown.
        assert "random-internal-a.bin" not in unknown_names
        assert "random-internal-b.bin" not in unknown_names

    def test_mmproj_metadata_cp_id_and_downloadable(self, tmp_path):
        """mmproj scanner artifact carries cp_id and downloadable=true metadata
        regardless of install state (catalog metadata is static)."""
        models = tmp_path / "models"
        result = scan_models(models)
        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        assert mmproj.cp_id == "gemma-3-12b-it-qat-gguf-mmproj"
        assert mmproj.downloadable is True

    # ------------------------------------------------------------------
    # Phase 3A regression: parent-folder evidence excludes known children
    # ------------------------------------------------------------------

    def test_mmproj_only_does_not_install_gemma_gguf_parent(self, tmp_path):
        """Regression: a gemma GGUF folder containing ONLY mmproj-BF16.gguf
        must report mmproj installed but the parent gemma_gguf folder artifact
        as missing. A known child projection file is not evidence that the
        Gemma text-encoder model itself is present."""
        models = tmp_path / "models"
        gguf_folder = models / "text_encoders" / "unsloth" / "gemma-3-12b-it-qat-GGUF"
        gguf_folder.mkdir(parents=True)
        (gguf_folder / "mmproj-BF16.gguf").write_bytes(b"\x00mmproj")

        result = scan_models(models)

        # mmproj is installed at its canonical path (descent-aware detection).
        from runtime_config.model_download_specs import resolve_model_path

        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        assert mmproj.status == "installed"
        assert mmproj.preferred_path == str(
            resolve_model_path(models, "gemma-3-12b-it-qat-gguf-mmproj")
        )

        # The parent gemma_gguf folder artifact is NOT installed — mmproj
        # alone is not evidence of the Gemma text-encoder model.
        gemma_gguf = next(a for a in result.artifacts if a.component_role == "gemma_gguf")
        assert gemma_gguf.status == "missing"
        assert gemma_gguf.preferred_path is None
        assert gemma_gguf.absolute_paths == []

        # No unknown leakage: the folder descent is still blocked.
        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        assert "mmproj-BF16.gguf" not in unknown_names

    def test_actual_gemma_gguf_file_installs_parent_folder(self, tmp_path):
        """A gemma GGUF folder containing an actual (non-mmproj) Gemma .gguf
        model file reports the parent gemma_gguf folder artifact as installed.
        This is the positive counterpart to the mmproj-only regression above."""
        models = tmp_path / "models"
        gguf_folder = models / "text_encoders" / "unsloth" / "gemma-3-12b-it-qat-GGUF"
        gguf_folder.mkdir(parents=True)
        (gguf_folder / "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf").write_bytes(b"\x00gemma")

        result = scan_models(models)

        # Parent folder artifact installed — real Gemma model file present.
        gemma_gguf = next(a for a in result.artifacts if a.component_role == "gemma_gguf")
        assert gemma_gguf.status == "installed"
        assert gemma_gguf.is_folder is True
        assert gemma_gguf.preferred_path is not None

        # mmproj is missing (not present in the folder).
        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        assert mmproj.status == "missing"

    def test_mmproj_plus_actual_gemma_installs_both(self, tmp_path):
        """When the folder contains both mmproj and an actual Gemma model
        file, both the parent folder and mmproj report installed."""
        models = tmp_path / "models"
        gguf_folder = models / "text_encoders" / "unsloth" / "gemma-3-12b-it-qat-GGUF"
        gguf_folder.mkdir(parents=True)
        (gguf_folder / "mmproj-BF16.gguf").write_bytes(b"\x00mmproj")
        (gguf_folder / "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf").write_bytes(b"\x00gemma")

        result = scan_models(models)

        gemma_gguf = next(a for a in result.artifacts if a.component_role == "gemma_gguf")
        assert gemma_gguf.status == "installed"

        mmproj = next(a for a in result.artifacts if a.component_role == "gemma_mmproj")
        assert mmproj.status == "installed"

    def test_canonical_paths_match_resolve_model_path(self, tmp_path):
        """Scanner canonical_relative_path for CPs must match resolve_model_path()."""
        from runtime_config.model_download_specs import (
            ALL_MODEL_CP_IDS,
            get_model_cp_spec,
            resolve_model_path,
        )

        models = tmp_path / "models"
        result = scan_models(models)

        for cp_id in ALL_MODEL_CP_IDS:
            spec = get_model_cp_spec(cp_id)
            art = next(
                (a for a in result.artifacts if a.filename == spec.relative_path.name),
                None,
            )
            assert art is not None, f"No scanner artifact for {cp_id}"
            expected_canonical = resolve_model_path(models, cp_id)
            assert str(models / art.canonical_relative_path) == str(expected_canonical), (
                f"{cp_id}: scanner canonical {art.canonical_relative_path!r} "
                f"!= resolve_model_path {expected_canonical}"
            )

    def test_scanned_at_is_iso_string(self, tmp_path):
        result = scan_models(tmp_path / "models")
        assert isinstance(result.scanned_at, str)
        assert len(result.scanned_at) > 0

    def test_response_models_dir_matches(self, tmp_path):
        models = tmp_path / "my_models"
        models.mkdir()
        result = scan_models(models)
        assert result.models_dir == str(models)


class TestScannerFullLayout:
    """Comprehensive test with a tempdir shaped like the verified live layout."""

    def test_full_layout_mimicking_live_install(self, tmp_path):
        """Live layout with subfolder-only canonicals.

        Files at their canonical subfolder paths are installed; files at root
        or non-canonical subfolders are wrong_folder_usable.
        """
        models = tmp_path / "LTX_models"

        # adapters/ — HDR pair + regular adapter + distilled LoRA (all canonical here)
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors")
        _write(models / "adapters", "ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors")
        _write(models / "adapters", "ltx-2.3-22b-distilled-lora-384.safetensors")

        # diffusion_models/ (canonical subfolder)
        _write(models / "diffusion_models", "ltx-2.3-22b-distilled.safetensors")

        # text_encoders/gemma/ (canonical subfolder)
        gemma = models / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"
        _write(gemma, "model.safetensors")
        _write(gemma, "tokenizer.model")

        # vae/ (canonical for VAE files)
        _write(models / "vae", "LTX23_video_vae_bf16.safetensors")

        # latent_upscale_models/ (canonical for upscaler)
        _write(models / "latent_upscale_models", "ltx-2.3-spatial-upscaler-x2-1.0.safetensors")

        # root-level upscaler (wrong folder now)
        _write(models, "notes.txt", b"notes")
        _write(models / "adapters", "incomplete.part", b"partial")

        result = scan_models(models)

        # Subfolder adapters at canonical path → installed (HDR now supported)
        hdr = _find(result.artifacts, "hdr")
        assert hdr.status == "installed"
        assert hdr.gated is False

        hdr_emb = _find(result.artifacts, "hdr_scene_embeddings")
        assert hdr_emb.status == "installed"
        assert hdr_emb.gated is False

        ingredients = _find(result.artifacts, "ingredients")
        assert ingredients.status == "installed"

        distilled = _find(result.artifacts, "distilled_lora_384")
        assert distilled.status == "installed"

        # Diffusion model at canonical subfolder → installed.
        # The distilled monolith and the official dev safetensors share the
        # ``base_diffusion_model`` role; look up the distilled by filename.
        base = _find_by_filename(result.artifacts, "ltx-2.3-22b-distilled.safetensors")
        assert base.status == "installed"

        # Text encoder at canonical subfolder → installed
        gemma_art = _find(result.artifacts, "gemma")
        assert gemma_art.status == "installed"
        assert gemma_art.is_folder is True

        # Upscaler at canonical subfolder → installed
        upscaler = _find(result.artifacts, "spatial_upscaler")
        assert upscaler.status == "installed"

        # Unknown file
        unknown_names = {Path(f.relative_path).name for f in result.unknown_files}
        assert "notes.txt" in unknown_names

        # Partial file
        partial_names = {Path(f.relative_path).name for f in result.partial_files}
        assert "incomplete.part" in partial_names

        # Read-only: tree unchanged
        assert _snapshot_tree(models) == _snapshot_tree(models)


# ------------------------------------------------------------------
# Endpoint tests
# ------------------------------------------------------------------


class TestModelCatalogEndpoint:
    def test_catalog_success(self, client, test_state):
        models_dir: Path = test_state.config.default_models_dir
        # Place at canonical adapters/ path → installed
        _write(models_dir / "adapters", "ltx-2.3-22b-ic-lora-hdr-0.9.safetensors")

        response = client.get("/api/models/catalog", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert data["models_dir"] == str(models_dir)
        assert isinstance(data["scanned_at"], str)
        assert len(data["artifacts"]) > 0
        # HDR at canonical adapters/ → installed + supported (not gated)
        hdr = next(a for a in data["artifacts"] if a["component_role"] == "hdr")
        assert hdr["status"] == "installed"
        assert hdr["gated"] is False

    def test_catalog_requires_admin(self, client):
        response = client.get("/api/models/catalog")
        assert_http_error(response, status_code=403, code="HTTP_403", message="Admin token required")

    def test_catalog_no_path_query_param(self, client):
        """Phase 1 must not accept a path query parameter (scans effective models_dir only)."""
        response = client.get(
            "/api/models/catalog",
            params={"path": "/etc"},
            headers=_ADMIN_HEADERS,
        )
        # The endpoint ignores any path param — it always scans the effective models_dir.
        assert response.status_code == 200

    def test_catalog_empty_models_dir(self, client):
        response = client.get("/api/models/catalog", headers=_ADMIN_HEADERS)
        assert response.status_code == 200
        data = response.json()
        assert all(a["status"] == "missing" for a in data["artifacts"])


class TestCatalogOpenAPISchema:
    """Verify the FastAPI OpenAPI document registers the catalog endpoint and types."""

    def test_openapi_includes_catalog_endpoint_and_types(self, test_state):
        from app_factory import create_app

        schema = create_app(handler=test_state).openapi()

        # Endpoint registered as GET
        catalog_path = schema["paths"].get("/api/models/catalog")
        assert catalog_path is not None, "GET /api/models/catalog not in OpenAPI paths"
        assert "get" in catalog_path

        # Response references ModelLibraryScanResponse
        ok_response = catalog_path["get"]["responses"]["200"]
        response_ref = ok_response["content"]["application/json"]["schema"]["$ref"]
        assert response_ref == "#/components/schemas/ModelLibraryScanResponse"

        # Component schemas present
        schemas = schema["components"]["schemas"]
        assert "ModelLibraryScanResponse" in schemas
        assert "ModelLibraryArtifact" in schemas
        assert "ModelProfileProblem" in schemas

        # ModelProfilePayload carries the Phase 1 migration fields
        profile_props = schemas["ModelProfilePayload"]["properties"]
        assert "schema_version" in profile_props
        assert "created_by" in profile_props
        assert "validation_status" in profile_props
        assert "last_scanned_at" in profile_props
        assert "problems" in profile_props
