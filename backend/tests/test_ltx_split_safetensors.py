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
