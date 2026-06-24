"""Tests for checkpoint specs and pure path helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from api_types import AdapterID, ModelCheckpointID
from runtime_config.model_download_specs import (
    ALL_MODEL_CP_IDS,
    ALL_LTX_LOCAL_MODEL_IDS,
    OFFICIAL_LTX23_ADAPTERS,
    ModelCheckpointSpec,
    get_ic_loras_cp_ids,
    get_latest_ltx_model_id,
    get_ltx_cps,
    get_ltx_model_cp_ids,
    get_ltx_model_spec,
    get_model_cp_spec,
    resolve_downloading_dir,
    resolve_downloading_path,
    resolve_downloading_target_path,
    resolve_model_path,
)


def test_specs_cover_all_checkpoint_ids():
    assert set(ALL_MODEL_CP_IDS) == {cp_id for cp_id in ALL_MODEL_CP_IDS}


def test_primary_ltx_checkpoints_map_1_to_1_with_ltx_models():
    assert len(get_ltx_cps()) == len(ALL_LTX_LOCAL_MODEL_IDS)


def test_latest_ltx_model_is_relevant():
    latest = get_latest_ltx_model_id()
    spec = get_ltx_model_spec(latest)
    assert spec.model_cp in get_ltx_cps()


def test_ic_lora_cp_ids_are_deduped():
    spec = get_ltx_model_spec(get_latest_ltx_model_id())
    assert get_ic_loras_cp_ids(spec.ic_loras_spec) == ("ltx-2.3-22b-ic-lora-union-control-ref0.5",)


def test_ltx_model_cp_ids_include_deduped_ic_loras():
    spec = get_ltx_model_spec(get_latest_ltx_model_id())
    assert get_ltx_model_cp_ids(get_latest_ltx_model_id()) == (
        spec.model_cp,
        spec.upscale_cp,
        spec.text_encoder_cp,
        "ltx-2.3-22b-ic-lora-union-control-ref0.5",
    )


def test_official_ltx23_adapter_registry_is_complete():
    expected_ids: set[AdapterID] = {
        "distilled_lora_384",
        "distilled_lora_384_1_1",
        "union_control",
        "motion_track_control",
        "ingredients",
        "water_simulation",
        "decompression",
        "deblur",
        "colorization",
        "day_to_night",
        "in_outpainting",
        "instant_shave",
        "cross_eyed",
        "hdr",
        "hdr_scene_embeddings",
        "lipdub",
    }

    assert set(OFFICIAL_LTX23_ADAPTERS) == expected_ids
    for adapter_id, adapter in OFFICIAL_LTX23_ADAPTERS.items():
        assert adapter.id == adapter_id
        assert adapter.repo_id.startswith("Lightricks/")
        assert adapter.filename.endswith(".safetensors")
        assert adapter.expected_size_bytes > 0
        assert adapter.required_for or adapter.optional_for


def test_official_ltx23_hdr_requires_embedding_pair():
    assert OFFICIAL_LTX23_ADAPTERS["hdr"].required_for == ("hdr",)
    assert OFFICIAL_LTX23_ADAPTERS["hdr_scene_embeddings"].required_for == ("hdr",)


def test_model_path_resolves_from_relative_path(tmp_path):
    cp_id: ModelCheckpointID = "gemma-3-12b-it-qat-q4_0-unquantized"
    spec = get_model_cp_spec(cp_id)
    assert resolve_model_path(tmp_path, cp_id) == tmp_path / spec.relative_path


def test_downloading_path_is_derived_from_spec():
    models_dir = Path("/tmp/models")
    downloading_dir = resolve_downloading_dir(models_dir)

    assert resolve_downloading_path(models_dir, "ltx-2.3-22b-distilled") == downloading_dir
    assert (
        resolve_downloading_path(models_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
        == downloading_dir / "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert resolve_downloading_target_path(models_dir, "ltx-2.3-22b-distilled") == downloading_dir / "ltx-2.3-22b-distilled.safetensors"


def test_relative_paths_are_unique():
    relative_paths = {get_model_cp_spec(cp_id).relative_path for cp_id in ALL_MODEL_CP_IDS}
    assert len(relative_paths) == len(ALL_MODEL_CP_IDS)


def test_model_path_rejects_parent_traversal(monkeypatch, tmp_path):
    bad_spec = ModelCheckpointSpec(
        relative_path=Path("../escape.safetensors"),
        expected_size_bytes=1,
        is_folder=False,
        repo_id="test/repo",
        description="bad",
    )

    monkeypatch.setattr(
        "runtime_config.model_download_specs.get_model_cp_spec",
        lambda cp_id: bad_spec,
    )

    with pytest.raises(ValueError):
        resolve_model_path(tmp_path, "ltx-2.3-22b-distilled")
