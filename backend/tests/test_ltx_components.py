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

    # ------------------------------------------------------------------
    # Phase 3A (plan §9 Option A): mmproj projection path plumbing
    # ------------------------------------------------------------------

    def test_mmproj_path_none_by_default(self):
        """Profiles without mmproj resolve mmproj_path=None."""
        profile = _profile(
            components={
                "transformer": "/m/transformer.safetensors",
                "transformer_format": "official_safetensors",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.mmproj_path is None

    def test_mmproj_path_carried_through(self):
        """An explicit mmproj path is plumbed through resolve_components()."""
        profile = _profile(
            components={
                "transformer": "/m/ltx.gguf",
                "transformer_format": "gguf",
                "text_encoder_root": "/m/gemma-gguf",
                "text_encoder_format": "gguf",
                "mmproj": "/m/gemma-3-12b-it-qat-GGUF/mmproj-BF16.gguf",
            }
        )
        resolved = resolve_components(profile)
        assert resolved.mmproj_path == "/m/gemma-3-12b-it-qat-GGUF/mmproj-BF16.gguf"

    def test_cache_key_includes_mmproj(self):
        """components.mmproj participates in the cache key so toggling it
        invalidates the pipeline cache (relevant once the multimodal path
        is wired)."""
        base_components = {
            "transformer": "/m/transformer.safetensors",
            "transformer_format": "official_safetensors",
        }
        without_mmproj = _profile(components={**base_components}, profile_id="p1")
        with_mmproj = _profile(
            components={
                **base_components,
                "mmproj": "/m/mmproj-BF16.gguf",
            },
            profile_id="p1",  # same id — only mmproj differs
        )
        key_without = resolve_components(without_mmproj).cache_key
        key_with = resolve_components(with_mmproj).cache_key
        assert key_without != key_with
        assert "/m/mmproj-BF16.gguf" in key_with
        # Empty string sentinel for absent mmproj.
        assert "" in key_without

    # ------------------------------------------------------------------
    # Phase 3D (plan §12): base_family inference + distilled LoRA plumbing
    # ------------------------------------------------------------------

    def test_base_family_distilled_inferred_from_path(self):
        profile = _profile(
            components={
                "transformer": "/m/ltx-2.3-22b-distilled.safetensors",
                "transformer_format": "official_safetensors",
            }
        )
        assert resolve_components(profile).base_family == "distilled"

    def test_base_family_dev_inferred_from_path(self):
        profile = _profile(
            components={
                "transformer": "/m/ltx-2.3-22b-dev.safetensors",
                "transformer_format": "official_safetensors",
            }
        )
        assert resolve_components(profile).base_family == "dev"

    def test_base_family_dev_inferred_from_gguf_path(self):
        profile = _profile(
            components={
                "transformer": "/m/ltx-2.3-22b-dev-Q4_K_M.gguf",
                "transformer_format": "gguf",
            }
        )
        assert resolve_components(profile).base_family == "dev"

    def test_base_family_distilled_lora_adapter_does_not_imply_distilled_base(self):
        """``distilled-lora`` / ``distilled_lora`` is an adapter, not a base."""
        for path in (
            "/m/ltx-2.3-22b-distilled-lora-384.safetensors",
            "/m/ltx-2.3-22b-distilled_lora-384-1.1.safetensors",
            "/m/adapters/ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
        ):
            profile = _profile(
                components={
                    "transformer": path,
                    "transformer_format": "official_safetensors",
                }
            )
            assert resolve_components(profile).base_family == "unknown", path

    def test_base_family_unknown_for_generic_filename(self):
        profile = _profile(
            components={
                "transformer": "/m/custom-model.safetensors",
                "transformer_format": "official_safetensors",
            }
        )
        assert resolve_components(profile).base_family == "unknown"

    def test_distilled_lora_path_none_by_default(self):
        profile = _profile(
            components={
                "transformer": "/m/ltx-2.3-22b-dev.safetensors",
                "transformer_format": "official_safetensors",
            }
        )
        assert resolve_components(profile).distilled_lora_path is None

    def test_distilled_lora_path_extracts_explicit_v1_1(self):
        profile = _profile(
            components={
                "transformer": "/m/ltx-2.3-22b-dev.safetensors",
                "transformer_format": "official_safetensors",
                "official_adapters": {
                    "distilled_lora_384": "/m/old.safetensors",
                    "distilled_lora_384_1_1": "/m/new.safetensors",
                },
            }
        )
        # v1.1 wins over v1.0
        assert resolve_components(profile).distilled_lora_path == "/m/new.safetensors"

    def test_distilled_lora_path_extracts_explicit_v1_when_v1_1_absent(self):
        profile = _profile(
            components={
                "transformer": "/m/ltx-2.3-22b-dev.safetensors",
                "transformer_format": "official_safetensors",
                "official_adapters": {
                    "distilled_lora_384": "/m/old.safetensors",
                },
            }
        )
        assert resolve_components(profile).distilled_lora_path == "/m/old.safetensors"

    def test_cache_key_includes_base_family(self):
        base_components = {
            "transformer_format": "official_safetensors",
        }
        dev_profile = _profile(
            components={**base_components, "transformer": "/m/ltx-2.3-22b-dev.safetensors"},
            profile_id="p1",
        )
        distilled_profile = _profile(
            components={**base_components, "transformer": "/m/ltx-2.3-22b-distilled.safetensors"},
            profile_id="p1",  # same id, different family
        )
        # Different transformer path already differentiates, but assert the
        # family token is present in the key.
        dev_key = resolve_components(dev_profile).cache_key
        distilled_key = resolve_components(distilled_profile).cache_key
        assert "dev" in dev_key
        assert "distilled" in distilled_key
        assert dev_key != distilled_key

    def test_cache_key_includes_explicit_distilled_lora_path(self):
        base = {
            "transformer": "/m/ltx-2.3-22b-dev.safetensors",
            "transformer_format": "official_safetensors",
        }
        without_lora = _profile(components={**base}, profile_id="p1")
        with_lora = _profile(
            components={
                **base,
                "official_adapters": {"distilled_lora_384_1_1": "/m/distilled-lora.safetensors"},
            },
            profile_id="p1",  # same id — only LoRA differs
        )
        key_without = resolve_components(without_lora).cache_key
        key_with = resolve_components(with_lora).cache_key
        assert key_without != key_with
        assert "/m/distilled-lora.safetensors" in key_with
