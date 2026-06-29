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

    # File CPs: downloading_path includes the parent subfolder from the spec.
    assert (
        resolve_downloading_path(models_dir, "ltx-2.3-22b-distilled")
        == downloading_dir / "diffusion_models"
    )
    # Folder CPs: downloading_path includes the full relative path.
    assert (
        resolve_downloading_path(models_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
        == downloading_dir / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"
    )
    assert (
        resolve_downloading_target_path(models_dir, "ltx-2.3-22b-distilled")
        == downloading_dir / "diffusion_models" / "ltx-2.3-22b-distilled.safetensors"
    )


def test_official_ic_lora_adapter_cp_specs_match_registry():
    """Each IC-LoRA adapter with a CP entry maps to the correct repo_id and filename."""
    from runtime_config.model_download_specs import ADAPTER_TO_CP_ID, OFFICIAL_LTX23_ADAPTERS

    for adapter_id, cp_id in ADAPTER_TO_CP_ID.items():
        adapter = OFFICIAL_LTX23_ADAPTERS[adapter_id]
        spec = get_model_cp_spec(cp_id)
        expected_filename = spec.relative_path.name
        assert adapter.filename == expected_filename, (
            f"{adapter_id}: registry filename {adapter.filename!r} != spec filename {expected_filename!r}"
        )
        assert adapter.repo_id == spec.repo_id, (
            f"{adapter_id}: registry repo {adapter.repo_id!r} != spec repo {spec.repo_id!r}"
        )


def test_adapter_to_cp_id_covers_every_non_distilled_adapter():
    """Every non-distilled adapter has a CP entry and vice-versa."""
    from runtime_config.model_download_specs import ADAPTER_TO_CP_ID, OFFICIAL_LTX23_ADAPTERS

    assert set(ADAPTER_TO_CP_ID) == {
        id
        for id, adapter in OFFICIAL_LTX23_ADAPTERS.items()
        if adapter.kind != "distilled_lora"
    }


def test_official_ic_lora_adapter_cp_specs_reject_distilled_only():
    """Only IC-LoRA (non-distilled) adapters get CP entries."""
    from runtime_config.model_download_specs import ADAPTER_TO_CP_ID

    assert "distilled_lora_384" not in ADAPTER_TO_CP_ID
    assert "distilled_lora_384_1_1" not in ADAPTER_TO_CP_ID


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


# ------------------------------------------------------------------
# Phase 2A — unsloth LTX-2.3 dev GGUF catalog entries (plan §4/§5/§7)
# ------------------------------------------------------------------


# Local basename, repo, byte size, section, variant group for each dev GGUF.
_DEV_GGUF_EXPECTATIONS: dict[ModelCheckpointID, tuple[str, str, int]] = {
    "ltx-2.3-22b-dev-gguf-q4-k-m": (
        "ltx-2.3-22b-dev-Q4_K_M.gguf",
        "unsloth/LTX-2.3-GGUF",
        14_326_856_736,
    ),
    "ltx-2.3-22b-dev-gguf-ud-q4-k-m": (
        "ltx-2.3-22b-dev-UD-Q4_K_M.gguf",
        "unsloth/LTX-2.3-GGUF",
        16_506_438_688,
    ),
    "ltx-2.3-22b-dev-gguf-q6-k": (
        "ltx-2.3-22b-dev-Q6_K.gguf",
        "unsloth/LTX-2.3-GGUF",
        17_774_906_400,
    ),
    "ltx-2.3-22b-dev-gguf-ud-q5-k-m": (
        "ltx-2.3-22b-dev-UD-Q5_K_M.gguf",
        "unsloth/LTX-2.3-GGUF",
        18_274_719_776,
    ),
}


def test_dev_gguf_specs_have_canonical_paths_sizes_and_repo():
    for cp_id, (basename, repo_id, size) in _DEV_GGUF_EXPECTATIONS.items():
        spec = get_model_cp_spec(cp_id)
        assert spec.relative_path == Path(
            f"diffusion_models/unsloth/LTX-2.3-GGUF/{basename}"
        ), cp_id
        assert spec.relative_path.name == basename, cp_id
        assert spec.expected_size_bytes == size, cp_id
        assert spec.is_folder is False, cp_id
        assert spec.repo_id == repo_id, cp_id


def test_dev_gguf_specs_carry_catalog_grouping_metadata():
    for cp_id in _DEV_GGUF_EXPECTATIONS:
        spec = get_model_cp_spec(cp_id)
        assert spec.section == "gguf", cp_id
        assert spec.variant_group == "ltx-2.3-dev-gguf", cp_id
        assert spec.downloadable is True, cp_id
        assert spec.display_name, f"display_name missing for {cp_id}"
        # Remote filename equals the local basename (no override needed).
        assert spec.remote_filename is None, cp_id
        assert spec.remote_name == spec.name, cp_id


def test_dev_gguf_specs_resolve_under_models_root():
    models_dir = Path("/tmp/models")
    for cp_id, (basename, _repo, _size) in _DEV_GGUF_EXPECTATIONS.items():
        resolved = resolve_model_path(models_dir, cp_id)
        assert resolved == models_dir / "diffusion_models" / "unsloth" / "LTX-2.3-GGUF" / basename, cp_id
        # No parent traversal, no absolute path.
        assert not resolved.is_absolute() or str(resolved).startswith(str(models_dir)), cp_id


def test_q6_ud_dev_gguf_is_not_present():
    """Q6 UD does not exist upstream (plan §5) — must never be a catalog entry."""
    assert "ltx-2.3-22b-dev-gguf-ud-q6-k" not in set(ALL_MODEL_CP_IDS)
    # No CP id should mention both UD and Q6 for the dev GGUF.
    for cp_id in ALL_MODEL_CP_IDS:
        if cp_id.startswith("ltx-2.3-22b-dev-gguf-"):
            lower = cp_id.lower()
            assert not ("ud" in lower and "q6" in lower), cp_id


def test_dev_gguf_paths_are_unique_against_existing_specs():
    """Adding GGUF entries must not collide with any existing relative path."""
    relative_paths = {get_model_cp_spec(cp_id).relative_path for cp_id in ALL_MODEL_CP_IDS}
    assert len(relative_paths) == len(ALL_MODEL_CP_IDS)


def test_dev_gguf_entries_do_not_change_latest_ltx_model_baseline():
    """Adding GGUF entries must not shift the default/relevant LTX model (plan §15)."""
    assert get_latest_ltx_model_id() == "ltx-2.3-22b-distilled"
    # GGUF ids are not LTX-local model ids.
    assert not any(
        cp_id in ALL_LTX_LOCAL_MODEL_IDS for cp_id in _DEV_GGUF_EXPECTATIONS
    )


def test_remote_name_property_defaults_to_local_basename():
    """remote_name falls back to relative_path.name when remote_filename is None."""
    spec = get_model_cp_spec("ltx-2.3-22b-distilled")
    assert spec.remote_filename is None
    assert spec.remote_name == spec.name


def test_remote_name_property_uses_explicit_override():
    """When remote_filename is set, remote_name returns it (HF name differs from local)."""
    spec = ModelCheckpointSpec(
        relative_path=Path("diffusion_models/local-name.gguf"),
        expected_size_bytes=1,
        is_folder=False,
        repo_id="test/repo",
        description="override",
        remote_filename="remote-name.gguf",
    )
    assert spec.name == "local-name.gguf"
    assert spec.remote_name == "remote-name.gguf"


# ------------------------------------------------------------------
# Phase 3A — Gemma 3 mmproj downloadable CP (plan §9 Option A)
# ------------------------------------------------------------------


def test_mmproj_spec_has_canonical_path_size_and_repo():
    """mmproj CP: canonical placement inside the gemma GGUF folder, BF16 size."""
    cp_id: ModelCheckpointID = "gemma-3-12b-it-qat-gguf-mmproj"
    spec = get_model_cp_spec(cp_id)
    assert spec.relative_path == Path(
        "text_encoders/unsloth/gemma-3-12b-it-qat-GGUF/mmproj-BF16.gguf"
    )
    assert spec.relative_path.name == "mmproj-BF16.gguf"
    assert spec.expected_size_bytes == 854_200_448
    assert spec.is_folder is False
    assert spec.repo_id == "unsloth/gemma-3-12b-it-qat-GGUF"


def test_mmproj_spec_carry_catalog_grouping_metadata():
    spec = get_model_cp_spec("gemma-3-12b-it-qat-gguf-mmproj")
    assert spec.section == "gguf"
    assert spec.variant_group == "gemma-3-gguf"
    assert spec.downloadable is True
    assert spec.display_name == "Gemma 3 mmproj BF16"
    assert spec.description


def test_mmproj_spec_remote_name_equals_local_basename():
    """remote_filename is None because the HF remote basename equals the local
    basename (mmproj-BF16.gguf). No remote-name promotion change is needed."""
    spec = get_model_cp_spec("gemma-3-12b-it-qat-gguf-mmproj")
    assert spec.remote_filename is None
    assert spec.remote_name == spec.name == "mmproj-BF16.gguf"


def test_mmproj_spec_path_does_not_collide_with_gemma_folder_artifact():
    """The mmproj file lives inside the gemma GGUF folder artifact's canonical
    path but must not collide with any other CP relative path."""
    relative_paths = {get_model_cp_spec(cp_id).relative_path for cp_id in ALL_MODEL_CP_IDS}
    assert len(relative_paths) == len(ALL_MODEL_CP_IDS)
    # The gemma GGUF text-encoder folder CP is a distinct path.
    gemma_cp = get_model_cp_spec("gemma-3-12b-it-qat-q4_0-unquantized").relative_path
    mmproj_cp = get_model_cp_spec("gemma-3-12b-it-qat-gguf-mmproj").relative_path
    assert gemma_cp != mmproj_cp
    # mmproj path is nested under the unsloth GGUF folder name but is a file,
    # not the folder itself.
    assert mmproj_cp.parent.name == "gemma-3-12b-it-qat-GGUF"


def test_mmproj_spec_resolve_under_models_root():
    models_dir = Path("/tmp/models")
    resolved = resolve_model_path(models_dir, "gemma-3-12b-it-qat-gguf-mmproj")
    assert resolved == (
        models_dir
        / "text_encoders"
        / "unsloth"
        / "gemma-3-12b-it-qat-GGUF"
        / "mmproj-BF16.gguf"
    )


def test_mmproj_entries_do_not_change_latest_ltx_model_baseline():
    """Adding mmproj must not shift the default/relevant LTX model (plan §15)."""
    assert get_latest_ltx_model_id() == "ltx-2.3-22b-distilled"
    assert "gemma-3-12b-it-qat-gguf-mmproj" not in set(ALL_LTX_LOCAL_MODEL_IDS)
