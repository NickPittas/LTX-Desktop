"""Pure tests for the Phase 2 model resolver / capability engine."""

from __future__ import annotations

from typing import Any

import pytest
from api_types import (
    ModelLibraryArtifact,
    ModelLibraryScanResponse,
    ModelProfilePayload,
)
from services.model_resolver import (
    ProfileCapabilityResult,
    ResolvedArtifactItem,
    resolve_profile_capabilities,
)

from tests.conftest import TEST_ADMIN_TOKEN  # noqa: F401 — ensures fixture wiring

_ADMIN_HEADERS = {"X-Admin-Token": TEST_ADMIN_TOKEN}


# ============================================================
# Local builders
# ============================================================


def _artifact(
    role: str,
    status: str = "missing",
    *,
    filename: str = "",
    absolute_paths: list[str] | None = None,
    preferred_path: str | None = None,
    canonical_relative_path: str = "",
    gated: bool = False,
    artifact_kind: str = "lora",
    repo_id: str = "test/repo",
) -> ModelLibraryArtifact:
    fname = filename or f"{role}.safetensors"
    return ModelLibraryArtifact(
        filename=fname,
        artifact_kind=artifact_kind,  # type: ignore[arg-type]
        component_role=role,
        status=status,  # type: ignore[arg-type]
        scanner_confidence="exact_catalog_match",
        canonical_relative_path=canonical_relative_path or fname,
        expected_size_bytes=0,
        repo_id=repo_id,
        source_url=f"https://huggingface.co/{repo_id}",
        is_folder=False,
        absolute_paths=absolute_paths or [],
        preferred_path=preferred_path,
        size_bytes=None,
        support_status="gated" if gated else "supported",
        gated=gated,
        notes="",
        cp_id=None,
        adapter_id=None,
    )


def _catalog(*artifacts: ModelLibraryArtifact) -> list[ModelLibraryArtifact]:
    return list(artifacts)


def _profile(
    profile_id: str = "test-profile",
    *,
    transformer: str | None = None,
    transformer_format: str = "official_safetensors",
    transformer_quantization: str | None = None,
    upsampler: str | None = None,
    text_encoder_root: str | None = None,
    text_encoder_format: str = "api",
    text_projection: str | None = None,
    source: str = "official",
    validation_status: str = "candidate",
    official_adapters: dict[str, str] | None = None,
    ic_lora_union: str | None = None,
    ic_lora_hdr: str | None = None,
    ic_lora_hdr_scene_embeddings: str | None = None,
) -> ModelProfilePayload:
    from api_types import ModelComponentPaths

    return ModelProfilePayload(
        id=profile_id,
        name="Test Profile",
        source=source,  # type: ignore[arg-type]
        components=ModelComponentPaths(
            transformer=transformer,
            transformer_format=transformer_format,  # type: ignore[arg-type]
            transformer_quantization=transformer_quantization,
            upsampler=upsampler,
            text_encoder_root=text_encoder_root,
            text_encoder_format=text_encoder_format,  # type: ignore[arg-type]
            text_projection=text_projection,
            ic_lora_union=ic_lora_union,
            ic_lora_hdr=ic_lora_hdr,
            ic_lora_hdr_scene_embeddings=ic_lora_hdr_scene_embeddings,
            official_adapters=official_adapters or {},
        ),
        validation_status=validation_status,  # type: ignore[arg-type]
    )


def _resolve(
    profile: ModelProfilePayload | None,
    catalog: ModelLibraryScanResponse | list[ModelLibraryArtifact] | None,
) -> ProfileCapabilityResult:
    return resolve_profile_capabilities(profile, catalog or [])


def _find(artifacts: list[ResolvedArtifactItem], role: str) -> ResolvedArtifactItem | None:
    return next((a for a in artifacts if a.component_role == role), None)


# ============================================================
# Tests
# ============================================================


class TestNoProfile:
    def test_no_profile_empty_catalog_returns_missing_result(self):
        result = _resolve(None, [])
        assert result.profile_id is None
        assert result.profile_valid is False
        assert result.base_family == "unknown"
        assert result.quantization == "unknown"
        assert result.fast_status == "missing"
        assert result.normal_status == "missing"
        assert result.hdr_status == "missing"
        assert result.distilled_lora_status == "not_applicable"
        assert result.has_local_text_encoder is False
        assert result.suppresses_api_key_prompt is False
        assert result.has_text_projection is False
        assert result.has_upscaler is False

    def test_no_profile_catalog_has_base_model(self):
        """Even without a profile, catalog-installed base shows in artifacts."""
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed", filename="ltx-2.3-22b-distilled.safetensors",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
        )
        result = _resolve(None, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "available"
        assert base.source == "catalog_installed"
        # Without a profile, fast is still missing (can't determine base family)
        assert result.fast_status == "missing"


class TestDistilledLoRARule:
    """Phase 3D: distilled LoRA is now runtime-wired for dev base.

    Dev base + available distilled LoRA => ``supported`` (was candidate_unwired
    pre-3D). Missing distilled LoRA stays ``missing``. Distilled base does not
    need a distilled LoRA (``not_applicable``). Unknown base family never
    candidates distilled LoRA.
    """

    def test_dev_base_distilled_lora_present_is_supported(self):
        profile = _profile(
            transformer="/models/ltx-2.3-22b-dev.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-dev.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-dev.safetensors"),
            _artifact("distilled_lora_384_1_1", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.base_family == "dev"
        # Phase 3D: distilled LoRA + dev base is now runtime-wired
        assert result.distilled_lora_status == "supported"
        assert result.fast_status == "supported"

    def test_dev_base_distilled_lora_missing(self):
        profile = _profile(
            transformer="/models/ltx-2.3-22b-dev.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-dev.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-dev.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.base_family == "dev"
        assert result.distilled_lora_status == "missing"
        assert result.fast_status == "missing"

    def test_standalone_distilled_does_not_need_distilled_lora(self):
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.base_family == "distilled"
        assert result.distilled_lora_status == "not_applicable"
        # Standalone distilled supports fast natively
        assert result.fast_status == "supported"
        # No distilled LoRA artifact items emitted
        assert _find(result.artifacts, "distilled_lora_384_1_1") is None
        assert _find(result.artifacts, "distilled_lora_384") is None

    def test_kijai_distilled_does_not_need_distilled_lora(self):
        profile = _profile(
            source="kijai",
            transformer="/models/ltx-distilled-kijai.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-distilled-kijai.safetensors"],
                      preferred_path="/models/ltx-distilled-kijai.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.base_family == "distilled"
        assert result.distilled_lora_status == "not_applicable"
        assert result.fast_status == "supported"

    def test_quantstack_distilled_does_not_need_distilled_lora(self):
        profile = _profile(
            source="quantstack",
            transformer="/models/ltx-distilled-quantstack.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-distilled-quantstack.safetensors"],
                      preferred_path="/models/ltx-distilled-quantstack.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.base_family == "distilled"
        assert result.distilled_lora_status == "not_applicable"

    # ── Blocker 1: source alone must not imply family ──

    def test_kijai_path_with_dev_is_dev(self):
        """Kijai source + dev path signal => dev, not distilled."""
        profile = _profile(source="kijai", transformer="/models/ltx-2.3-22b-dev.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "dev"

    def test_quantstack_path_with_dev_is_dev(self):
        """QuantStack source + dev path signal => dev, not distilled."""
        profile = _profile(source="quantstack", transformer="/models/ltx-dev.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "dev"

    def test_kijai_no_path_signal_is_unknown(self):
        """Kijai source with no distilled/dev path signal => unknown."""
        profile = _profile(source="kijai", transformer="/models/custom-model.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "unknown"

    def test_quantstack_no_path_signal_is_unknown(self):
        """QuantStack source with no distilled/dev path signal => unknown."""
        profile = _profile(source="quantstack", transformer="/models/custom.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "unknown"

    # ── Blocker 2: unknown base must not candidate distilled LoRA ──

    def test_unknown_base_distilled_lora_not_applicable(self):
        """Unknown base family => distilled_lora_status not_applicable (not candidate)."""
        profile = _profile(transformer="/models/custom-model.safetensors")
        catalog = _catalog(
            _artifact("distilled_lora_384_1_1", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.base_family == "unknown"
        assert result.distilled_lora_status == "not_applicable"
        # No distilled LoRA artifact items emitted for unknown base
        assert _find(result.artifacts, "distilled_lora_384_1_1") is None


class TestDevGGUFProfile:
    def test_dev_gguf_quantization_detected(self):
        profile = _profile(
            transformer="/models/ltx-2.3-22b-dev.gguf",
            transformer_format="gguf",
        )
        result = _resolve(profile, [])
        assert result.quantization == "gguf"
        assert result.base_family == "dev"

    def test_dev_gguf_distilled_lora_supported(self):
        profile = _profile(
            transformer="/models/ltx-2.3-22b-dev.gguf",
            transformer_format="gguf",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-dev.gguf"],
                      preferred_path="/models/ltx-2.3-22b-dev.gguf"),
            _artifact("distilled_lora_384_1_1", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
        )
        result = _resolve(profile, catalog)
        # Phase 3D: dev + distilled LoRA is now runtime-wired (GGUF or otherwise)
        assert result.fast_status == "supported"
        assert result.distilled_lora_status == "supported"


class TestProfilePathPriority:
    def test_profile_explicit_path_in_catalog_wins(self):
        """Profile path matching catalog absolute_paths gets source=profile."""
        path = "/models/ltx-2.3-22b-distilled.safetensors"
        profile = _profile(
            transformer=path,
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=[path],
                      preferred_path=path),
        )
        result = _resolve(profile, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "available"
        assert base.source == "profile"
        assert base.preferred_path == path

    def test_profile_explicit_path_outside_catalog_is_unverified(self):
        """Profile path not in catalog → profile_unverified + problem, no FS probe."""
        profile_path = "/other/location/model.safetensors"
        catalog_path = "/models/model.safetensors"
        profile = _profile(transformer=profile_path)
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=[catalog_path],
                      preferred_path=catalog_path),
        )
        result = _resolve(profile, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "profile_unverified"
        assert base.source == "profile"
        assert base.preferred_path == profile_path
        # Problem emitted
        codes = [p.code for p in base.problems]
        assert "profile_path_unverified" in codes
        # Also in top-level problems
        top_codes = [p.code for p in result.problems]
        assert "profile_path_unverified" in top_codes

    def test_profile_path_outside_catalog_no_filesystem_probe(self, tmp_path):
        """Profile unverified path does NOT need to exist on disk."""
        # Path that definitely doesn't exist
        fake_path = str(tmp_path / "nonexistent" / "model.safetensors")
        profile = _profile(transformer=fake_path)
        result = _resolve(profile, [])
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "profile_unverified"
        # No crash — resolver didn't try to stat the file

    def test_duplicate_profile_path_wins_over_catalog_preferred(self):
        """Blocker 3: explicit profile path wins; preferred != catalog preferred."""
        profile_path = "/models/subfolder/model.safetensors"
        catalog_preferred = "/models/model.safetensors"
        profile = _profile(transformer=profile_path)
        catalog = _catalog(
            _artifact("base_diffusion_model", "duplicate",
                      absolute_paths=[catalog_preferred, profile_path],
                      preferred_path=catalog_preferred),
        )
        result = _resolve(profile, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "duplicate"
        assert base.source == "profile"
        # Profile path wins, not catalog preferred
        assert base.preferred_path == profile_path
        assert base.preferred_path != catalog_preferred
        # Duplicate problem still emitted
        codes = [p.code for p in result.problems]
        assert "duplicate_artifact" in codes


class TestWrongFolderAndDuplicate:
    def test_wrong_folder_is_candidate_usable_not_available(self):
        """Catalog wrong_folder_usable → candidate_usable, not current-runtime available."""
        profile = _profile()  # no explicit paths
        catalog = _catalog(
            _artifact("base_diffusion_model", "wrong_folder_usable",
                      absolute_paths=["/models/diffusion_models/model.safetensors"],
                      preferred_path="/models/diffusion_models/model.safetensors"),
        )
        result = _resolve(profile, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "candidate_usable"
        assert base.source == "catalog_wrong_folder"
        # Not available at current runtime → fast missing
        assert result.fast_status == "missing"

    def test_duplicate_artifact_reports_problem_and_preferred(self):
        profile = _profile()
        preferred = "/models/model.safetensors"
        other = "/models/subfolder/model.safetensors"
        catalog = _catalog(
            _artifact("base_diffusion_model", "duplicate",
                      absolute_paths=[preferred, other],
                      preferred_path=preferred),
        )
        result = _resolve(profile, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "duplicate"
        assert base.preferred_path == preferred
        codes = [p.code for p in result.problems]
        assert "duplicate_artifact" in codes

    def test_duplicate_base_distilled_yields_supported_fast(self):
        """Duplicate base counts as available for capability derivation."""
        path_a = "/models/ltx-2.3-22b-distilled.safetensors"
        path_b = "/models/sub/ltx-2.3-22b-distilled.safetensors"
        profile = _profile(transformer=path_a)
        catalog = _catalog(
            _artifact("base_diffusion_model", "duplicate",
                      absolute_paths=[path_a, path_b],
                      preferred_path=path_a),
        )
        result = _resolve(profile, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "duplicate"
        assert result.fast_status == "supported"
        assert result.normal_status == "supported"
        # Duplicate problem retained
        codes = [p.code for p in result.problems]
        assert "duplicate_artifact" in codes

    def test_duplicate_upscaler_yields_has_upscaler_and_candidate_retake(self):
        """Duplicate upscaler counts as available; retake still candidate_unwired."""
        path_a = "/models/upscaler.safetensors"
        path_b = "/models/sub/upscaler.safetensors"
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
            upsampler=path_a,
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
            _artifact("spatial_upscaler", "duplicate",
                      absolute_paths=[path_a, path_b],
                      preferred_path=path_a),
        )
        result = _resolve(profile, catalog)
        assert result.has_upscaler is True
        assert result.retake_upscaler_status == "candidate_unwired"

    def test_wrong_folder_still_does_not_count_as_available(self):
        """candidate_usable (wrong folder) must not count as available."""
        profile = _profile()
        catalog = _catalog(
            _artifact("base_diffusion_model", "wrong_folder_usable",
                      absolute_paths=["/models/sub/model.safetensors"],
                      preferred_path="/models/sub/model.safetensors"),
            _artifact("spatial_upscaler", "wrong_folder_usable",
                      absolute_paths=["/models/sub/upscaler.safetensors"],
                      preferred_path="/models/sub/upscaler.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.fast_status == "missing"
        assert result.normal_status == "missing"
        assert result.has_upscaler is False


class TestHDRStatus:
    """HDR workflow status is derived from artifact availability: ``supported``
    only when base + HDR LoRA + scene-embedding support asset are all
    available; ``missing`` otherwise. HDR artifacts are no longer gated at the
    resolver artifact level — present artifacts report ``available``.

    The scene embeddings remain a required *support* asset (they participate in
    the HDR capability check) but are never an independently selectable
    adapter — selectability is owned by the handler.
    """

    def test_hdr_present_with_all_components_is_supported(self):
        """Base + HDR LoRA + scene embeddings installed → hdr_status supported."""
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
            _artifact("hdr", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"),
            _artifact("hdr_scene_embeddings", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.hdr_status == "supported"

        # HDR artifacts are available at the artifact level (no longer gated).
        hdr = _find(result.artifacts, "hdr")
        assert hdr is not None
        assert hdr.status == "available"

        hdr_emb = _find(result.artifacts, "hdr_scene_embeddings")
        assert hdr_emb is not None
        assert hdr_emb.status == "available"

    def test_hdr_missing_is_missing(self):
        """No artifacts → hdr_status missing (no longer hardcoded gated)."""
        result = _resolve(None, [])
        assert result.hdr_status == "missing"

    def test_hdr_artifacts_without_base_is_missing(self):
        """HDR LoRA + scene embeddings present but base absent → missing.

        Base diffusion model availability is required for the HDR workflow.
        """
        profile = _profile()  # no transformer / base
        catalog = _catalog(
            _artifact("hdr", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"),
            _artifact("hdr_scene_embeddings", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.hdr_status == "missing"

    def test_hdr_lora_without_scene_embeddings_is_missing(self):
        """Base + HDR LoRA present but scene embeddings absent → missing.

        Scene embeddings are a required support asset for the HDR workflow.
        """
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
            _artifact("hdr", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-ic-lora-hdr-0.9.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.hdr_status == "missing"

    def test_hdr_with_profile_paths_supported_when_all_present(self):
        """Explicit profile paths for HDR + base resolve available; workflow
        supported when base + HDR LoRA + scene embeddings are all present."""
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
            ic_lora_hdr="/models/hdr-lora.safetensors",
            ic_lora_hdr_scene_embeddings="/models/hdr-scene-emb.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
            _artifact("hdr", "installed",
                      absolute_paths=["/models/hdr-lora.safetensors"],
                      preferred_path="/models/hdr-lora.safetensors"),
            _artifact("hdr_scene_embeddings", "installed",
                      absolute_paths=["/models/hdr-scene-emb.safetensors"],
                      preferred_path="/models/hdr-scene-emb.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.hdr_status == "supported"

        # HDR artifact status is available via profile path (no longer gated).
        hdr = _find(result.artifacts, "hdr")
        assert hdr is not None
        assert hdr.status == "available"
        assert hdr.source == "profile"

        hdr_emb = _find(result.artifacts, "hdr_scene_embeddings")
        assert hdr_emb is not None
        assert hdr_emb.status == "available"

    def test_hdr_scene_embeddings_support_asset_not_selectable_via_resolver(self):
        """Scene embeddings resolve as a support artifact (available when
        present) — they are not gated, but selectability as a standalone
        workflow is owned by the handler, not the resolver."""
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
            _artifact("hdr_scene_embeddings", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors"),
        )
        result = _resolve(profile, catalog)
        # Without the HDR LoRA, the workflow is still missing — the scene
        # embeddings support asset alone does not constitute an HDR workflow.
        assert result.hdr_status == "missing"
        # The support asset itself reports available (not gated/hidden).
        hdr_emb = _find(result.artifacts, "hdr_scene_embeddings")
        assert hdr_emb is not None
        assert hdr_emb.status == "available"


class TestLocalTextEncoder:
    def test_local_gemma_reports_has_local_text_encoder(self):
        profile = _profile(
            text_encoder_root="/models/gemma-3-12b-it-qat-q4_0-unquantized",
            text_encoder_format="hf_folder",
        )
        result = _resolve(profile, [])
        assert result.has_local_text_encoder is True
        assert result.suppresses_api_key_prompt is True

    def test_api_text_encoder_does_not_suppress_api_key(self):
        profile = _profile(
            text_encoder_format="api",
        )
        result = _resolve(profile, [])
        assert result.has_local_text_encoder is False
        assert result.suppresses_api_key_prompt is False

    def test_text_projection_independent_of_text_encoder(self):
        """Text projection is separate from local text encoder capability."""
        profile = _profile(
            text_encoder_format="api",
            text_projection="/models/projection.safetensors",
        )
        result = _resolve(profile, [])
        assert result.has_local_text_encoder is False
        assert result.suppresses_api_key_prompt is False
        assert result.has_text_projection is True

    def test_local_gemma_with_text_projection_both_reported(self):
        profile = _profile(
            text_encoder_root="/models/gemma",
            text_encoder_format="hf_folder",
            text_projection="/models/projection.safetensors",
        )
        result = _resolve(profile, [])
        assert result.has_local_text_encoder is True
        assert result.suppresses_api_key_prompt is True
        assert result.has_text_projection is True

    def test_gguf_text_encoder_is_local(self):
        profile = _profile(
            text_encoder_root="/models/gemma.gguf",
            text_encoder_format="gguf",
        )
        result = _resolve(profile, [])
        assert result.has_local_text_encoder is True


class TestUpscaler:
    def test_upscaler_present_retake_is_candidate_unwired(self):
        """Upscaler available but retake upscaler not wired → candidate_unwired."""
        profile = _profile(
            transformer="/models/ltx-2.3-22b-distilled.safetensors",
            upsampler="/models/upscaler.safetensors",
        )
        catalog = _catalog(
            _artifact("base_diffusion_model", "installed",
                      absolute_paths=["/models/ltx-2.3-22b-distilled.safetensors"],
                      preferred_path="/models/ltx-2.3-22b-distilled.safetensors"),
            _artifact("spatial_upscaler", "installed",
                      absolute_paths=["/models/upscaler.safetensors"],
                      preferred_path="/models/upscaler.safetensors"),
        )
        result = _resolve(profile, catalog)
        assert result.has_upscaler is True
        assert result.retake_upscaler_status == "candidate_unwired"

    def test_upscaler_missing_retake_is_missing(self):
        result = _resolve(None, [])
        assert result.has_upscaler is False
        assert result.retake_upscaler_status == "missing"


class TestBaseFamilyAndQuantization:
    def test_distilled_in_path_detected(self):
        profile = _profile(transformer="/models/ltx-2.3-22b-distilled.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "distilled"

    def test_dev_in_path_detected(self):
        profile = _profile(transformer="/models/ltx-2.3-22b-dev.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "dev"

    def test_distilled_lora_in_transformer_field_is_unknown(self):
        """distilled-lora in transformer field is an adapter, not a base."""
        profile = _profile(transformer="/models/ltx-2.3-22b-distilled-lora-384.safetensors")
        result = _resolve(profile, [])
        assert result.base_family == "unknown"

    def test_fp8_quantization_from_field(self):
        profile = _profile(
            transformer="/models/model.safetensors",
            transformer_quantization="fp8_input_scaled",
        )
        result = _resolve(profile, [])
        assert result.quantization == "fp8"

    def test_nvfp4_quantization_from_filename(self):
        profile = _profile(transformer="/models/model-nvfp4.safetensors")
        result = _resolve(profile, [])
        assert result.quantization == "nvfp4"


class TestProfileValidity:
    def test_candidate_profile_is_valid(self):
        profile = _profile(validation_status="candidate")
        result = _resolve(profile, [])
        assert result.profile_valid is True

    def test_validated_profile_is_valid(self):
        profile = _profile(validation_status="validated")
        result = _resolve(profile, [])
        assert result.profile_valid is True

    def test_deprecated_profile_is_invalid_with_problem(self):
        profile = _profile(validation_status="deprecated")
        result = _resolve(profile, [])
        assert result.profile_valid is False
        codes = [p.code for p in result.problems]
        assert "deprecated_profile" in codes


class TestModelLibraryScanResponseInput:
    """Resolver accepts both list and ModelLibraryScanResponse."""

    def test_accepts_scan_response_object(self):
        catalog = ModelLibraryScanResponse(
            models_dir="/models",
            scanned_at="2025-01-01T00:00:00Z",
            artifacts=[
                _artifact("base_diffusion_model", "installed",
                          absolute_paths=["/models/model.safetensors"],
                          preferred_path="/models/model.safetensors"),
            ],
        )
        result = _resolve(None, catalog)
        base = _find(result.artifacts, "base_diffusion_model")
        assert base is not None
        assert base.status == "available"


class TestScopeCreep:
    """Oracle: no route/OpenAPI changes in Phase 2."""

    def test_resolver_types_not_in_openapi(self, test_state):
        from app_factory import create_app

        schema = create_app(handler=test_state).openapi()
        schemas: dict[str, Any] = schema["components"]["schemas"]
        # Frozen dataclasses must NOT appear as Pydantic/OpenAPI schemas
        assert "ProfileCapabilityResult" not in schemas
        assert "ResolvedArtifactItem" not in schemas

    def test_no_resolver_endpoint_in_openapi(self, test_state):
        from app_factory import create_app

        schema = create_app(handler=test_state).openapi()
        paths: dict[str, Any] = schema["paths"]
        assert "/api/models/resolve" not in paths
        assert "/api/models/capabilities" not in paths
        assert "/api/profile/capabilities" not in paths

    def test_no_mocks_in_test_file(self):
        """Guardrail: this test file uses no mocking libraries."""
        from pathlib import Path
        import re

        content = Path(__file__).read_text(encoding="utf-8")
        forbidden = (
            r"\bMagicMock\b",
            r"\bunittest\.mock\b",
            r"\bfrom\s+unittest\.mock\s+import\b",
            r"\bimport\s+unittest\.mock\b",
            r"(?<!\w)patch\(",
        )
        for pattern in forbidden:
            assert re.search(pattern, content) is None, f"Forbidden mock pattern: {pattern}"
