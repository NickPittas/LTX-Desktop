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
