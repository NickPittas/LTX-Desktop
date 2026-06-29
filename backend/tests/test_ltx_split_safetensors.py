"""Integration tests: tuple checkpoint paths flow through handler to pipeline fakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from _routes._errors import HTTPError
from api_types import ModelComponentPaths, ModelProfilePayload
from runtime_config.model_download_specs import UPSAMPLER_CP_ID, resolve_model_path


def _activate_split_profile(test_state, tmp_path: Path):
    """Create a Kijai split profile directly in state and activate it.

    Transformer filename carries the ``distilled`` base-family signal so the
    Phase 3D base-family router resolves it as a distilled fast pipeline.
    """
    d = tmp_path / "kijai"
    d.mkdir()
    files = {
        "ltx-2.3-22b-distilled-transformer.safetensors": b"x",
        "tp.safetensors": b"x",
        "ec.safetensors": b"x",
        "vvae.safetensors": b"x",
        "avae.safetensors": b"x",
        "ups.safetensors": b"x",
    }
    paths = {}
    for name, content in files.items():
        p = d / name
        p.write_bytes(content)
        key = "transformer" if name.startswith("ltx-") else name.rsplit(".", 1)[0]
        paths[key] = str(p)

    profile = ModelProfilePayload(
        id="kijai-split",
        name="Kijai Split",
        source="kijai",
        components=ModelComponentPaths(
            transformer=paths["transformer"],
            transformer_format="split_safetensors",
            text_projection=paths["tp"],
            embeddings_connector=paths["ec"],
            video_vae=paths["vvae"],
            audio_vae=paths["avae"],
            upsampler=paths["ups"],
            text_encoder_format="api",
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "kijai-split"
    return profile


def _activate_official_profile(test_state, tmp_path: Path):
    """Create an official monolith profile and activate it.

    Uses the canonical official LTX-2.3 distilled transformer filename so the
    Phase 3D base-family router resolves it as a distilled fast pipeline.
    """
    d = tmp_path / "official"
    d.mkdir()
    transformer = d / "ltx-2.3-22b-distilled.safetensors"
    transformer.write_bytes(b"x")
    upsampler = d / "upsampler.safetensors"
    upsampler.write_bytes(b"x")
    profile = ModelProfilePayload(
        id="official-test",
        name="Official Test",
        source="official",
        components=ModelComponentPaths(
            transformer=str(transformer),
            upsampler=str(upsampler),
            text_encoder_format="api",
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "official-test"
    return profile


class TestLtxSplitSafetensorsIntegration:
    def test_fast_pipeline_receives_tuple_for_split_profile(
        self, test_state, tmp_path, fake_services
    ):
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")
        cp = fake_services.fast_video_pipeline.last_checkpoint_path
        assert isinstance(cp, tuple), f"Expected tuple, got {type(cp)}"
        assert len(cp) == 5, f"Expected 5 paths, got {len(cp)}"
        for p in cp:
            assert isinstance(p, str) and p.endswith(".safetensors")

    def test_fast_pipeline_receives_str_for_official_profile(
        self, test_state, tmp_path, fake_services
    ):
        _activate_official_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")
        cp = fake_services.fast_video_pipeline.last_checkpoint_path
        assert isinstance(cp, str), f"Expected str, got {type(cp)}"
        assert cp.endswith(".safetensors")

    def test_split_profile_components_carry_vae_paths(
        self, test_state, tmp_path, fake_services
    ):
        """Split safetensors profile resolves with video/audio VAE paths."""
        profile = _activate_split_profile(test_state, tmp_path)
        from services.ltx_components import resolve_components
        r = resolve_components(profile)
        assert r.video_vae_path is not None
        assert r.video_vae_path.endswith("vvae.safetensors")
        assert r.audio_vae_path is not None
        assert r.audio_vae_path.endswith("avae.safetensors")


def _activate_gguf_profile(test_state, tmp_path: Path):
    """Create a GGUF transformer profile directly in state and activate it.

    Uses a distilled GGUF filename so the Phase 3D base-family router resolves
    it as a distilled base — these tests exercise format/streaming/compile
    behavior, not the dev distilled-LoRA requirement (covered separately by
    ``TestDevDistilledLoraRouting``).
    """
    d = tmp_path / "gguf"
    d.mkdir()
    files = {
        "ltx-2.3-22b-distilled-Q4_K_M.gguf": b"GGUF",
        "tp.safetensors": b"x",
        "ec.safetensors": b"x",
        "vvae.safetensors": b"x",
        "avae.safetensors": b"x",
        "ups.safetensors": b"x",
    }
    paths = {}
    for name, content in files.items():
        p = d / name
        p.write_bytes(content)
        key = "transformer" if name.endswith(".gguf") else name.rsplit(".", 1)[0]
        paths[key] = str(p)

    profile = ModelProfilePayload(
        id="gguf-profile",
        name="GGUF Transformer",
        source="kijai",
        components=ModelComponentPaths(
            transformer=paths["transformer"],
            transformer_format="gguf",
            text_projection=paths["tp"],
            embeddings_connector=paths["ec"],
            video_vae=paths["vvae"],
            audio_vae=paths["avae"],
            upsampler=paths["ups"],
            text_encoder_format="api",
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "gguf-profile"
    return profile


class TestLtxGgufFormatIntegration:
    def test_fast_pipeline_receives_gguf_format_for_gguf_profile(
        self, test_state, tmp_path, fake_services
    ):
        _activate_gguf_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")

        fake = fake_services.fast_video_pipeline
        assert fake.last_transformer_format == "gguf"

        cp = fake.last_checkpoint_path
        assert isinstance(cp, tuple), f"Expected tuple, got {type(cp)}"
        assert len(cp) == 5, f"Expected 5 paths, got {len(cp)}"
        assert cp[0].endswith(".gguf")
        assert all(p.endswith(".safetensors") for p in cp[1:])

    def test_fast_pipeline_receives_safetensors_format_for_official_profile(
        self, test_state, tmp_path, fake_services
    ):
        _activate_official_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")

        assert fake_services.fast_video_pipeline.last_transformer_format == "safetensors"

    def test_gguf_profile_with_torch_compile_skips_compile_silently(
        self, test_state, tmp_path, fake_services
    ):
        """GGUF transformer + use_torch_compile=True must not call compile_transformer.

        The handler checks supports_torch_compile() before invoking
        compile_transformer(), so the GGUF compile path is skipped without
        raising/warning (no RuntimeError traceback). Regression for the
        "Failed to compile transformer: GGUF transformer compile is not
        supported yet" warning traceback.
        """
        _activate_gguf_profile(test_state, tmp_path)
        test_state.state.app_settings.use_torch_compile = True
        test_state.pipelines.load_gpu_pipeline("fast")

        assert fake_services.fast_video_pipeline.last_transformer_format == "gguf"
        assert fake_services.fast_video_pipeline.compile_calls == 0

    def test_safetensors_profile_with_torch_compile_still_compiles(
        self, test_state, tmp_path, fake_services
    ):
        """Non-GGUF (safetensors) profile with compile enabled still compiles.

        Proves the GGUF skip is GGUF-specific and does not regress the normal
        compile path.
        """
        _activate_official_profile(test_state, tmp_path)
        test_state.state.app_settings.use_torch_compile = True
        test_state.pipelines.load_gpu_pipeline("fast")

        assert fake_services.fast_video_pipeline.compile_calls == 1


class TestStreamingPrefetchCountGuard:
    """Kijai split-safetensors 22B does not fit full residency on 32GB.

    Guard forces streaming_prefetch_count=2 when None is passed for split
    safetensors; explicit values preserved; GGUF/monolith unaffected.
    """

    def test_split_none_uses_two(self, test_state, tmp_path, fake_services):
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 2

    def test_split_none_uses_two_retake(self, test_state, tmp_path, fake_services):
        """Retake guard: split safetensors None→2.

        Skipped when CUDA/GPU unavailable (retake loads ltx_core which
        imports GPU-specific modules).
        """
        _activate_split_profile(test_state, tmp_path)
        try:
            test_state.pipelines.load_retake_pipeline()
        except ImportError:
            import pytest
            pytest.skip("ltx_core GPU imports not available")
        assert fake_services.retake_pipeline.last_streaming_prefetch_count == 2

    def test_split_explicit_one_preserved(self, test_state, tmp_path, fake_services):
        """Explicit streaming_prefetch_count=1 preserved (not overridden to 2)."""
        _activate_split_profile(test_state, tmp_path)
        # Load pipeline first to get resolved components
        test_state.pipelines.load_gpu_pipeline("fast")
        from tests.fakes.services import FakeFastVideoPipeline
        FakeFastVideoPipeline.create(
            checkpoint_path=("t", "tp", "ec", "vv", "av"),
            gemma_root=None,
            upsampler_path="ups",
            device="cpu",
            streaming_prefetch_count=1,
            components=fake_services.fast_video_pipeline.last_components,
            transformer_format="safetensors",
        )
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 1

    def test_split_explicit_three_preserved(self, test_state, tmp_path, fake_services):
        """Explicit streaming_prefetch_count=3 preserved (not overridden to 2)."""
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")
        from tests.fakes.services import FakeFastVideoPipeline
        FakeFastVideoPipeline.create(
            checkpoint_path=("t", "tp", "ec", "vv", "av"),
            gemma_root=None,
            upsampler_path="ups",
            device="cpu",
            streaming_prefetch_count=3,
            components=fake_services.fast_video_pipeline.last_components,
            transformer_format="safetensors",
        )
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count == 3

    def test_gguf_none_remains_none(self, test_state, tmp_path, fake_services):
        _activate_gguf_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count is None

    def test_official_none_remains_none(self, test_state, tmp_path, fake_services):
        _activate_official_profile(test_state, tmp_path)
        test_state.pipelines.load_gpu_pipeline("fast")
        assert fake_services.fast_video_pipeline.last_streaming_prefetch_count is None

    # --- IC-LoRA pipeline guard ---

    def test_ic_lora_split_none_uses_two(self, test_state, tmp_path, fake_services):
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_ic_lora(lora_paths=[], depth_model_path=None)
        assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 2

    def test_ic_lora_split_explicit_one_preserved(self, test_state, tmp_path, fake_services):
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_ic_lora(lora_paths=[], depth_model_path=None)
        from tests.fakes.services import FakeIcLoraPipeline
        FakeIcLoraPipeline.create(
            checkpoint_path=("t", "tp", "ec", "vv", "av"),
            gemma_root=None,
            upsampler_path="ups",
            lora_paths=[],
            device="cpu",
            streaming_prefetch_count=1,
            components=fake_services.ic_lora_pipeline.last_components,
        )
        assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count == 1

    def test_ic_lora_gguf_none_remains_none(self, test_state, tmp_path, fake_services):
        _activate_gguf_profile(test_state, tmp_path)
        test_state.pipelines.load_ic_lora(lora_paths=[], depth_model_path=None)
        assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count is None

    def test_ic_lora_official_none_remains_none(self, test_state, tmp_path, fake_services):
        _activate_official_profile(test_state, tmp_path)
        test_state.pipelines.load_ic_lora(lora_paths=[], depth_model_path=None)
        assert fake_services.ic_lora_pipeline.last_streaming_prefetch_count is None

    # --- A2V pipeline guard ---

    def test_a2v_split_none_uses_two(self, test_state, tmp_path, fake_services):
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_a2v_pipeline()
        assert fake_services.a2v_pipeline.last_streaming_prefetch_count == 2

    def test_a2v_split_explicit_one_preserved(self, test_state, tmp_path, fake_services):
        _activate_split_profile(test_state, tmp_path)
        test_state.pipelines.load_a2v_pipeline()
        from tests.fakes.services import FakeA2VPipeline
        FakeA2VPipeline.create(
            checkpoint_path=("t", "tp", "ec", "vv", "av"),
            gemma_root=None,
            upsampler_path="ups",
            device="cpu",
            streaming_prefetch_count=1,
            components=fake_services.a2v_pipeline.last_components,
        )
        assert fake_services.a2v_pipeline.last_streaming_prefetch_count == 1

    def test_a2v_gguf_none_remains_none(self, test_state, tmp_path, fake_services):
        _activate_gguf_profile(test_state, tmp_path)
        test_state.pipelines.load_a2v_pipeline()
        assert fake_services.a2v_pipeline.last_streaming_prefetch_count is None

    def test_a2v_official_none_remains_none(self, test_state, tmp_path, fake_services):
        _activate_official_profile(test_state, tmp_path)
        test_state.pipelines.load_a2v_pipeline()
        assert fake_services.a2v_pipeline.last_streaming_prefetch_count is None


def _activate_profile_with_upsampler(test_state, tmp_path: Path, upsampler_path: str):
    """Create + activate an official profile pointing at ``upsampler_path``.

    The transformer is irrelevant to these tests; we only care about upsampler
    resolution. We write a placeholder transformer on disk so other resolver
    code paths don't blow up, but the assertions only check upsampler behavior.
    The filename carries the ``distilled`` signal so the Phase 3D base-family
    router does not reject the profile as unknown.
    """
    transformer = tmp_path / "ltx-2.3-22b-distilled.safetensors"
    transformer.write_bytes(b"x")
    profile = ModelProfilePayload(
        id="upsampler-resolution",
        name="Upsampler Resolution",
        source="official",
        components=ModelComponentPaths(
            transformer=str(transformer),
            transformer_format="official_safetensors",
            upsampler=upsampler_path,
            text_encoder_format="api",
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "upsampler-resolution"
    return profile


class TestUpsamplerRuntimeResolution:
    """Phase 1: harden ``_resolve_checkpoint_paths`` against stale upsampler paths.

    Regression: an active profile stored a stale root-level upsampler path that
    did not exist on disk; the upstream fast video pipeline then failed deep
    inside inference with FileNotFoundError. The handler now prefers an
    existing explicit path, falls back to canonical when the explicit path is
    stale/missing, and fails fast with HTTP 409 when no usable upscaler exists.
    """

    def test_stale_missing_plus_canonical_exists_uses_canonical(
        self, test_state, tmp_path, fake_services
    ):
        models_dir: Path = test_state.config.default_models_dir
        canonical = resolve_model_path(models_dir, UPSAMPLER_CP_ID)
        canonical.parent.mkdir(parents=True, exist_ok=True)
        canonical.write_bytes(b"upsampler")

        # Stale root-level path that does NOT exist on disk.
        stale = models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        assert not stale.exists()

        _activate_profile_with_upsampler(test_state, tmp_path, str(stale))
        test_state.pipelines.load_gpu_pipeline("fast")

        assert fake_services.fast_video_pipeline.last_upsampler_path == str(canonical)

    def test_stale_exists_plus_canonical_missing_uses_explicit(
        self, test_state, tmp_path, fake_services
    ):
        models_dir: Path = test_state.config.default_models_dir
        explicit = models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        explicit.parent.mkdir(parents=True, exist_ok=True)
        explicit.write_bytes(b"upsampler")

        # Canonical must NOT exist for this test.
        canonical = resolve_model_path(models_dir, UPSAMPLER_CP_ID)
        assert not canonical.exists()

        _activate_profile_with_upsampler(test_state, tmp_path, str(explicit))
        test_state.pipelines.load_gpu_pipeline("fast")

        assert fake_services.fast_video_pipeline.last_upsampler_path == str(explicit)

    def test_both_missing_raises_409_before_upstream(self, test_state, tmp_path, fake_services):
        models_dir: Path = test_state.config.default_models_dir
        # Neither the explicit nor canonical upscaler exist.
        explicit = models_dir / "ltx-2.3-spatial-upscaler-x2-1.0.safetensors"
        canonical = resolve_model_path(models_dir, UPSAMPLER_CP_ID)
        assert not explicit.exists()
        assert not canonical.exists()

        _activate_profile_with_upsampler(test_state, tmp_path, str(explicit))

        with pytest.raises(HTTPError) as exc_info:
            test_state.pipelines.load_gpu_pipeline("fast")

        assert exc_info.value.status_code == 409
        assert exc_info.value.code == "UPSCALER_REQUIRED"
        # Pipeline construction never happened.
        assert fake_services.fast_video_pipeline.last_upsampler_path is None


# ============================================================
# Phase 3D: dev-vs-distilled fast pipeline routing
# ============================================================


def _write_distilled_lora_files(test_state, which: str = "1_1") -> str:
    """Write a canonical distilled LoRA under ``<models_dir>/adapters/`` and
    return its absolute path.

    ``which`` selects v1.1 (``"1_1"``) or v1.0 (``"1_0"``).
    """
    from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS

    role = "distilled_lora_384_1_1" if which == "1_1" else "distilled_lora_384"
    adapter = OFFICIAL_LTX23_ADAPTERS[role]  # type: ignore[index]
    models_dir: Path = test_state.config.default_models_dir
    path = models_dir / "adapters" / adapter.filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"distilled-lora")
    return str(path)


def _activate_dev_profile(
    test_state,
    tmp_path: Path,
    *,
    official_adapters: dict[str, str] | None = None,
):
    """Create + activate a dev base profile (official monolith transformer)."""
    d = tmp_path / "dev"
    d.mkdir()
    transformer = d / "ltx-2.3-22b-dev.safetensors"
    transformer.write_bytes(b"x")
    upsampler = d / "upsampler.safetensors"
    upsampler.write_bytes(b"x")
    profile = ModelProfilePayload(
        id="dev-profile",
        name="Dev",
        source="official",
        components=ModelComponentPaths(
            transformer=str(transformer),
            transformer_format="official_safetensors",
            upsampler=str(upsampler),
            text_encoder_format="api",
            official_adapters=official_adapters or {},
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "dev-profile"
    return profile


def _activate_unknown_profile(test_state, tmp_path: Path):
    """Create + activate a profile whose transformer has no family signal."""
    d = tmp_path / "unknown"
    d.mkdir()
    transformer = d / "custom-model.safetensors"
    transformer.write_bytes(b"x")
    upsampler = d / "upsampler.safetensors"
    upsampler.write_bytes(b"x")
    profile = ModelProfilePayload(
        id="unknown-profile",
        name="Unknown",
        source="custom",
        components=ModelComponentPaths(
            transformer=str(transformer),
            transformer_format="official_safetensors",
            upsampler=str(upsampler),
            text_encoder_format="api",
        ),
    )
    test_state.state.model_profiles = [profile]
    test_state.state.active_model_profile_id = "unknown-profile"
    return profile


class TestDevDistilledLoraRouting:
    """Phase 3D (plan §12): dev base routes to TI2VidTwoStagesPipeline and
    requires a distilled LoRA. Distilled base keeps the existing DistilledPipeline
    route. Unknown base fails with an actionable HTTPError.
    """

    def test_dev_profile_with_explicit_lora_passes_path(
        self, test_state, tmp_path, fake_services
    ):
        explicit = tmp_path / "explicit-distilled-lora.safetensors"
        explicit.write_bytes(b"lora")
        _activate_dev_profile(
            test_state, tmp_path,
            official_adapters={"distilled_lora_384_1_1": str(explicit)},
        )

        test_state.pipelines.load_gpu_pipeline("fast")

        fake = fake_services.fast_video_pipeline
        assert fake.last_base_family == "dev"
        assert fake.last_distilled_lora_path == str(explicit)

    def test_dev_profile_with_canonical_fallback_lora(
        self, test_state, tmp_path, fake_services
    ):
        # No explicit adapter; canonical v1.1 file exists on disk.
        canonical = _write_distilled_lora_files(test_state, which="1_1")
        _activate_dev_profile(test_state, tmp_path)

        test_state.pipelines.load_gpu_pipeline("fast")

        fake = fake_services.fast_video_pipeline
        assert fake.last_base_family == "dev"
        assert fake.last_distilled_lora_path == canonical

    def test_dev_profile_falls_back_to_v1_when_v1_1_missing(
        self, test_state, tmp_path, fake_services
    ):
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS

        canonical_v1 = _write_distilled_lora_files(test_state, which="1_0")
        _activate_dev_profile(test_state, tmp_path)

        test_state.pipelines.load_gpu_pipeline("fast")

        fake = fake_services.fast_video_pipeline
        assert fake.last_base_family == "dev"
        assert fake.last_distilled_lora_path == canonical_v1
        # v1.1 canonical must NOT exist on disk for this test.
        models_dir: Path = test_state.config.default_models_dir
        v1_1 = models_dir / "adapters" / OFFICIAL_LTX23_ADAPTERS["distilled_lora_384_1_1"].filename  # type: ignore[index]
        assert not v1_1.exists()

    def test_dev_profile_missing_lora_raises_409_with_placement_path(
        self, test_state, tmp_path, fake_services
    ):
        # No explicit adapter; no canonical files on disk.
        _activate_dev_profile(test_state, tmp_path)

        with pytest.raises(HTTPError) as exc_info:
            test_state.pipelines.load_gpu_pipeline("fast")

        assert exc_info.value.status_code == 409
        assert exc_info.value.code == "DISTILLED_LORA_REQUIRED"
        # Message must include the canonical placement path(s).
        from runtime_config.model_download_specs import OFFICIAL_LTX23_ADAPTERS

        models_dir: Path = test_state.config.default_models_dir
        expected_v1_1 = str(
            models_dir / "adapters" / OFFICIAL_LTX23_ADAPTERS["distilled_lora_384_1_1"].filename  # type: ignore[index]
        )
        assert expected_v1_1 in exc_info.value.detail
        # Pipeline construction never happened.
        assert fake_services.fast_video_pipeline.last_distilled_lora_path is None

    def test_distilled_profile_passes_no_lora_and_remains_distilled(
        self, test_state, tmp_path, fake_services
    ):
        _activate_official_profile(test_state, tmp_path)  # distilled monolith
        test_state.pipelines.load_gpu_pipeline("fast")

        fake = fake_services.fast_video_pipeline
        assert fake.last_base_family == "distilled"
        # Distilled route never constructs a dev LoRA.
        assert fake.last_distilled_lora_path is None

    def test_unknown_family_raises_409_before_heavy_load(
        self, test_state, tmp_path, fake_services
    ):
        _activate_unknown_profile(test_state, tmp_path)

        with pytest.raises(HTTPError) as exc_info:
            test_state.pipelines.load_gpu_pipeline("fast")

        assert exc_info.value.status_code == 409
        assert exc_info.value.code == "UNSUPPORTED_MODEL_BASE_FAMILY"
        # Pipeline construction never happened.
        assert fake_services.fast_video_pipeline.last_base_family is None

    def test_dev_route_skips_torch_compile(
        self, test_state, tmp_path, fake_services
    ):
        """Dev route is not torch.compile-enabled in the initial wiring.

        The handler checks supports_torch_compile() before invoking
        compile_transformer(); the dev route must skip compile silently
        (no compile call) consistent with GGUF skip behavior.
        """
        _write_distilled_lora_files(test_state, which="1_1")
        _activate_dev_profile(test_state, tmp_path)
        test_state.state.app_settings.use_torch_compile = True

        test_state.pipelines.load_gpu_pipeline("fast")

        fake = fake_services.fast_video_pipeline
        assert fake.last_base_family == "dev"
        assert fake.compile_calls == 0

    def test_dev_canonical_fallback_path_joins_cache_key(
        self, test_state, tmp_path, fake_services
    ):
        """When the dev route falls back to a canonical LoRA, the cache key
        must include the effective fallback path so toggling the on-disk
        canonical file invalidates the pipeline cache."""
        canonical = _write_distilled_lora_files(test_state, which="1_1")
        _activate_dev_profile(test_state, tmp_path)

        state = test_state.pipelines.load_gpu_pipeline("fast")
        # Cache key must reflect the effective fallback LoRA path.
        assert canonical in state.cache_key
