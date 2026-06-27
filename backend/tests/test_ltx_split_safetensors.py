"""Integration tests: tuple checkpoint paths flow through handler to pipeline fakes."""

from __future__ import annotations

from pathlib import Path

from api_types import ModelComponentPaths, ModelProfilePayload


def _activate_split_profile(test_state, tmp_path: Path):
    """Create a Kijai split profile directly in state and activate it."""
    d = tmp_path / "kijai"
    d.mkdir()
    files = {
        "transformer.safetensors": b"x",
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
        paths[name.rsplit(".", 1)[0]] = str(p)

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
    """Create an official monolith profile and activate it."""
    d = tmp_path / "official"
    d.mkdir()
    transformer = d / "model.safetensors"
    transformer.write_bytes(b"x")
    profile = ModelProfilePayload(
        id="official-test",
        name="Official Test",
        source="official",
        components=ModelComponentPaths(
            transformer=str(transformer),
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
    """Create a GGUF transformer profile directly in state and activate it."""
    d = tmp_path / "gguf"
    d.mkdir()
    files = {
        "transformer.gguf": b"GGUF",
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
        paths[name.rsplit(".", 1)[0]] = str(p)

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
