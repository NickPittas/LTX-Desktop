"""Unit tests for component bundle resolver — pure data, no GPU needed."""

from __future__ import annotations

from api_types import ModelComponentPaths, ModelProfilePayload
from services.ltx_components import (
    ResolvedLtxComponents,
    checkpoint_path_arg,
    resolve_components,
)


def _profile(
    components: dict | None = None,
    profile_id: str = "test",
    source: str = "official",
) -> ModelProfilePayload:
    return ModelProfilePayload(
        id=profile_id,
        name="Test",
        source=source,
        components=ModelComponentPaths(**(components or {})),
    )


def test_gemma_root_from_gguf_format():
    profile = _profile(
        components={
            "transformer": "/models/ltx.gguf",
            "transformer_format": "gguf",
            "text_encoder_root": "/models/gemma-gguf",
            "text_encoder_format": "gguf",
        }
    )
    resolved = resolve_components(profile)
    assert resolved.gemma_root == "/models/gemma-gguf"


def test_gemma_root_from_safetensors_format():
    profile = _profile(
        components={
            "transformer": "/models/ltx.safetensors",
            "transformer_format": "official_safetensors",
            "text_encoder_root": "/models/gemma-st",
            "text_encoder_format": "safetensors",
        }
    )
    resolved = resolve_components(profile)
    assert resolved.gemma_root == "/models/gemma-st"


class TestResolveComponents:
    def test_official_monolith_single_checkpoint_path(self):
        profile = _profile(
            components={
                "transformer": "/models/ltx.safetensors",
                "transformer_format": "official_safetensors",
                "upsampler": "/models/upsampler.safetensors",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.transformer_format == "safetensors"
        assert resolved.checkpoint_paths_for_filtered_builders == (
            "/models/ltx.safetensors",
        )
        assert checkpoint_path_arg(resolved) == "/models/ltx.safetensors"

    def test_split_safetensors_tuple_ordering(self):
        profile = _profile(
            components={
                "transformer": "/m/transformer.safetensors",
                "transformer_format": "split_safetensors",
                "text_projection": "/m/tp.safetensors",
                "embeddings_connector": "/m/ec.safetensors",
                "video_vae": "/m/vvae.safetensors",
                "audio_vae": "/m/avae.safetensors",
                "upsampler": "/m/ups.safetensors",
            },
            source="kijai",
        )
        resolved = resolve_components(profile)
        assert resolved.checkpoint_paths_for_filtered_builders == (
            "/m/transformer.safetensors",
            "/m/tp.safetensors",
            "/m/ec.safetensors",
            "/m/vvae.safetensors",
            "/m/avae.safetensors",
        )
        assert checkpoint_path_arg(resolved) == resolved.checkpoint_paths_for_filtered_builders

    def test_gguf_format_detected(self):
        profile = _profile(
            components={
                "transformer": "/m/model.gguf",
                "transformer_format": "gguf",
            },
            source="quantstack",
        )
        resolved = resolve_components(profile)
        assert resolved.transformer_format == "gguf"

    def test_upsampler_path_carried(self):
        profile = _profile(
            components={
                "transformer": "/models/ltx.safetensors",
                "transformer_format": "official_safetensors",
                "upsampler": "/models/upsampler.safetensors",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.upsampler_path == "/models/upsampler.safetensors"

    def test_gemma_root_from_hf_folder(self):
        profile = _profile(
            components={
                "transformer": "/models/ltx.safetensors",
                "transformer_format": "official_safetensors",
                "text_encoder_root": "/models/gemma",
                "text_encoder_format": "hf_folder",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.gemma_root == "/models/gemma"

    def test_gemma_root_none_for_non_hf(self):
        profile = _profile(
            components={
                "transformer": "/models/ltx.safetensors",
                "transformer_format": "official_safetensors",
                "text_encoder_root": "/models/gemma",
                "text_encoder_format": "api",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.gemma_root is None

    def test_empty_builder_paths_when_no_transformer(self):
        profile = _profile(
            components={
                "transformer": None,
                "transformer_format": "official_safetensors",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.checkpoint_paths_for_filtered_builders == ()

    def test_none_fields_in_split(self):
        profile = _profile(
            components={
                "transformer": "/m/transformer.safetensors",
                "transformer_format": "split_safetensors",
                "text_projection": None,
                "embeddings_connector": "/m/ec.safetensors",
                "video_vae": None,
                "audio_vae": None,
            }
        )
        resolved = resolve_components(profile)
        assert resolved.checkpoint_paths_for_filtered_builders == (
            "/m/transformer.safetensors",
            "/m/ec.safetensors",
        )

    def test_cache_key_includes_profile_id_and_all_paths(self):
        profile = _profile(
            components={
                "transformer": "/m/transformer.safetensors",
                "transformer_format": "split_safetensors",
                "text_projection": "/m/tp.safetensors",
                "embeddings_connector": "/m/ec.safetensors",
                "video_vae": "/m/vvae.safetensors",
                "audio_vae": "/m/avae.safetensors",
                "upsampler": "/m/ups.safetensors",
                "text_encoder_root": "/m/gemma",
                "text_encoder_format": "hf_folder",
            },
            profile_id="my-profile",
        )
        resolved = resolve_components(profile)
        assert resolved.cache_key[0] == "my-profile"
        assert resolved.cache_key[1] == "safetensors"
        assert "/m/transformer.safetensors" in resolved.cache_key
        assert "/m/ups.safetensors" in resolved.cache_key

    def test_cache_key_different_profiles_differ(self):
        a = _profile(
            components={
                "transformer": "/m/a.safetensors",
                "transformer_format": "official_safetensors",
            },
            profile_id="profile-a",
        )
        b = _profile(
            components={
                "transformer": "/m/b.safetensors",
                "transformer_format": "official_safetensors",
            },
            profile_id="profile-b",
        )
        assert resolve_components(a).cache_key != resolve_components(b).cache_key

    def test_no_transformer_in_official_empty_cache_key_suffix(self):
        profile = _profile(
            components={
                "transformer": None,
                "transformer_format": "official_safetensors",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.transformer_path == ""
        assert resolved.checkpoint_paths_for_filtered_builders == ()

    def test_no_upsampler_in_profile(self):
        profile = _profile(
            components={
                "transformer": "/m/transformer.safetensors",
                "transformer_format": "official_safetensors",
                "upsampler": None,
            }
        )
        resolved = resolve_components(profile)
        assert resolved.upsampler_path is None
