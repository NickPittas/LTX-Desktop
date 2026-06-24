"""Unit tests for the GGUF state-dict loader + install helper (slice 1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.patches.gguf_loader_fix import (
    GgufNativeSDOps,
    GgufStateDictLoader,
    install_gguf_loader,
)

# Native/diffusers-style tensor names confirmed in real QuantStack LTX 2.3 GGUF:
# no `model.diffusion_model.` Comfy prefix (0 of ~4.4k tensors carry it).
_NATIVE_GGUF_KEYS = (
    "transformer_blocks.0.attn1.to_q.weight",
    "adaln_single.emb.timestep_embedder.linear_1.weight",
    "video_embeddings_connector.connector_in.weight",
)


def _write_tiny_gguf(path: str, *, with_config: bool, tensor_name: str = "x.weight") -> None:
    """Write a minimal GGUF file with an embedded config (optional) and one F32 tensor."""
    import gguf

    writer = gguf.GGUFWriter(path, arch="ltxv")
    if with_config:
        writer.add_string("general.config", json.dumps({"transformer": {"num_layers": 7}}))
    tensor = np.arange(2 * 3, dtype=np.float32).reshape(3, 2)
    writer.add_tensor(tensor_name, tensor, raw_shape=(3, 2))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def _make_pipe_with_builder(builder: object) -> SimpleNamespace:
    return SimpleNamespace(stage=SimpleNamespace(_transformer_builder=builder))


# ---------------------------------------------------------------------------
# install_gguf_loader
# ---------------------------------------------------------------------------


def test_install_gguf_loader_replaces_model_loader_and_sd_ops() -> None:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_path="/fake/transformer.gguf",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pipe = _make_pipe_with_builder(builder)

    install_gguf_loader(pipe)

    replaced = pipe.stage._transformer_builder
    assert replaced is not builder
    # loader + sd_ops both replaced with GGUF-native versions.
    assert isinstance(replaced.model_loader, GgufStateDictLoader)
    assert isinstance(replaced.model_sd_ops, GgufNativeSDOps)
    # original builder fields preserved through dataclasses.replace.
    assert replaced.model_path == "/fake/transformer.gguf"
    assert replaced.model_class_configurator is LTXModelConfigurator
    assert replaced.loras == ()
    assert replaced.registry is builder.registry
    assert replaced.lora_load_device == builder.lora_load_device


def test_install_gguf_loader_updates_existing_loader_with_wrong_sd_ops() -> None:
    """A GGUF loader already present but with the Comfy renaming map must be repaired."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_path="/fake/t.gguf",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        model_loader=GgufStateDictLoader(),
    )
    pipe = _make_pipe_with_builder(builder)

    install_gguf_loader(pipe)

    replaced = pipe.stage._transformer_builder
    assert replaced is not builder
    assert isinstance(replaced.model_loader, GgufStateDictLoader)
    assert isinstance(replaced.model_sd_ops, GgufNativeSDOps)


def test_install_gguf_loader_is_idempotent() -> None:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_path="/fake/t.gguf",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pipe = _make_pipe_with_builder(builder)

    install_gguf_loader(pipe)
    after_first = pipe.stage._transformer_builder
    install_gguf_loader(pipe)

    assert pipe.stage._transformer_builder is after_first


def test_install_gguf_loader_raises_on_wrong_pipeline_shape() -> None:
    with pytest.raises(RuntimeError):
        install_gguf_loader(object())

    with pytest.raises(RuntimeError):
        install_gguf_loader(SimpleNamespace(stage=SimpleNamespace()))


# ---------------------------------------------------------------------------
# GgufStateDictLoader.load
# ---------------------------------------------------------------------------


def test_gguf_native_sd_ops_keeps_native_tensor_names() -> None:
    """Identity sd_ops must pass native GGUF keys/values through unchanged."""
    sd_ops = GgufNativeSDOps()
    assert sd_ops.name == "gguf_native"
    value = torch.zeros(2, 3)
    for key in _NATIVE_GGUF_KEYS:
        assert sd_ops.apply_to_key(key) == key
        results = sd_ops.apply_to_key_value(key, value)
        assert len(results) == 1
        assert results[0].new_key == key
        assert torch.equal(results[0].new_value, value)


def test_gguf_loader_load_raises_when_no_gguf_path() -> None:
    with pytest.raises(RuntimeError):
        GgufStateDictLoader().load(["/tmp/a.safetensors", "/tmp/b.safetensors"])


def test_gguf_loader_load_skips_safetensors_in_tuple(tmp_path: Path) -> None:
    gguf_path = str(tmp_path / "transformer.gguf")
    _write_tiny_gguf(gguf_path, with_config=False, tensor_name="x.weight")
    bogus_safetensors = str(tmp_path / "does_not_exist.safetensors")

    # Native sd_ops lets the GGUF tensor through (Comfy map would filter it).
    state_dict = GgufStateDictLoader().load([bogus_safetensors, gguf_path], sd_ops=GgufNativeSDOps())

    # Loaded only the .gguf tensor; bogus safetensors path was skipped (never opened).
    assert "x.weight" in state_dict.sd
    assert state_dict.size > 0
    assert len(state_dict.dtype) > 0


def test_gguf_loader_load_raises_when_sd_ops_filters_everything(tmp_path: Path) -> None:
    """If sd_ops drops every tensor key, load() must raise rather than return an empty state dict.

    Models the confirmed real-world mismatch: native-named LTX GGUF tensors carry
    no Comfy ``model.diffusion_model.`` prefix, so a renaming map built for Comfy
    safetensors filters them all (validated against QuantStack/LTX-2.3-GGUF:
    0 of 4444 tensors carry the prefix).
    """
    gguf_path = str(tmp_path / "transformer.gguf")
    _write_tiny_gguf(gguf_path, with_config=False, tensor_name="model.diffusion_model.foo.weight")

    class _FiltersEveryKey:
        def apply_to_key(self, key: str) -> str | None:
            return None

    with pytest.raises(RuntimeError):
        GgufStateDictLoader().load(gguf_path, sd_ops=_FiltersEveryKey())


# ---------------------------------------------------------------------------
# GgufStateDictLoader.metadata
# ---------------------------------------------------------------------------


def test_gguf_loader_metadata_reads_embedded_config(tmp_path: Path) -> None:
    gguf_path = str(tmp_path / "transformer.gguf")
    _write_tiny_gguf(gguf_path, with_config=True)

    config = GgufStateDictLoader().metadata(gguf_path)

    assert config == {"transformer": {"num_layers": 7}}


def test_gguf_loader_metadata_raises_when_no_config(tmp_path: Path) -> None:
    gguf_path = str(tmp_path / "transformer.gguf")
    _write_tiny_gguf(gguf_path, with_config=False)

    with pytest.raises(RuntimeError):
        GgufStateDictLoader().metadata(gguf_path)
