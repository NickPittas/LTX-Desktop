"""Unit tests for the GGUF state-dict loader + install helper (slice 1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import gguf

from services.patches.gguf_loader_fix import (
    GGUF_DEQUANT_LINEAR_OP,
    GGUF_GEMMA_DEQUANT_LINEAR_OP,
    GgufGemmaSDOps,
    GgufLinear,
    GgufNativeSDOps,
    GgufStateDictLoader,
    KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER,
    KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER,
    KijaiFp8ScaledLinear,
    QParam,
    _amend_forward_with_gguf,
    _find_gemma_gguf,
    _find_v2_embeddings_config,
    _is_gemma_linear_name,
    _is_quantized_type,
    _resolve_gemma_tokenizer_root,
    install_gguf_component_paths,
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


def _nonquant_raw_samples() -> list[tuple[object, np.ndarray]]:
    """Quantized-raw bytes (logical shape (3,2)) for F32/F16/BF16 GGUF types.

    BF16 has no numpy dtype, so its bytes are produced by ``gguf.quants.quantize``
    (same helper the existing quantized GGUF writer uses) which lays out the
    uint8 storage the GGUF reader expects; ``raw_shape`` is the block shape it
    returns.
    """
    import gguf
    from gguf import quants as gquants

    src = np.arange(6, dtype=np.float32).reshape(3, 2)
    samples: list[tuple[object, np.ndarray]] = []
    for gguf_type in (
        gguf.GGMLQuantizationType.F32,
        gguf.GGMLQuantizationType.F16,
        gguf.GGMLQuantizationType.BF16,
    ):
        qblock = gquants.quantize(src, qtype=gguf_type)
        samples.append((gguf_type, qblock))
    return samples


def _make_pipe_with_builder(builder: object) -> SimpleNamespace:
    return SimpleNamespace(stage=SimpleNamespace(_transformer_builder=builder))


def test_kijai_fp8_scaled_linear_applies_weight_scale() -> None:
    layer = torch.nn.Linear(2, 1, bias=False)
    layer.__class__ = KijaiFp8ScaledLinear
    layer.load_state_dict(
        {
            "weight": torch.tensor([[2.0, -4.0]], dtype=torch.float8_e4m3fn),
            "weight_scale": torch.tensor(0.5),
        },
        strict=False,
        assign=True,
    )

    out = layer(torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16))

    assert torch.allclose(out.float(), torch.tensor([[-1.0]]), atol=0.05)


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
    assert any(op.name == GGUF_DEQUANT_LINEAR_OP.name for op in replaced.module_ops)


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


def test_install_gguf_component_paths_uses_explicit_paths() -> None:
    """Explicit video_vae_path/audio_vae_path override heuristic filename matching."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXModelConfigurator

    fake_vae_path = "/explicit/path/vae.safetensors"
    dummy_checkpoint = ("/profiles/no_match_1.safetensors", "/profiles/no_match_2.safetensors")

    # Builders with initial (wrong) paths — explicit paths should replace these
    enc_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    dec_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    voc_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )

    pipe = SimpleNamespace(
        image_conditioner=SimpleNamespace(_encoder_builder=enc_builder),
        upsampler=SimpleNamespace(_encoder_builder=enc_builder),
        video_decoder=SimpleNamespace(_decoder_builder=dec_builder),
        audio_decoder=SimpleNamespace(
            _decoder_builder=dec_builder,
            _vocoder_builder=voc_builder,
        ),
    )

    # Use same explicit path for both video and audio in this test
    install_gguf_component_paths(
        pipe, dummy_checkpoint, video_vae_path=fake_vae_path, audio_vae_path=fake_vae_path
    )

    assert pipe.image_conditioner._encoder_builder.model_path == fake_vae_path
    assert pipe.upsampler._encoder_builder.model_path == fake_vae_path
    assert pipe.video_decoder._decoder_builder.model_path == fake_vae_path
    assert pipe.audio_decoder._decoder_builder.model_path == fake_vae_path
    assert pipe.audio_decoder._vocoder_builder.model_path == fake_vae_path


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


def test_gguf_loader_metadata_reads_safetensors_config(tmp_path: Path) -> None:
    """include_safetensors=True, metadata() on .safetensors must not call GGUFReader."""
    import torch
    from safetensors.torch import save_file

    safetensors_path = str(tmp_path / "checkpoint.safetensors")
    save_file(
        {"dummy": torch.zeros(1)},
        safetensors_path,
        metadata={"config": json.dumps({"transformer": {"num_layers": 7}})},
    )

    loader = GgufStateDictLoader(include_safetensors=True)
    config = loader.metadata(safetensors_path)

    assert config == {"transformer": {"num_layers": 7}}


# ---------------------------------------------------------------------------
# Lazy dequant: QParam / GgufLinear / module op / load wrapping
# ---------------------------------------------------------------------------


def _write_quantized_gguf(path: str, tensor_name: str = "x.weight") -> np.ndarray:
    """Write a GGUF file with one Q4_0 tensor (4x256) and return the source fp32 array."""
    import gguf
    import gguf.quants as gquants

    source = np.random.RandomState(0).rand(4, 256).astype(np.float32)
    qblock = gquants.quantize(source, gguf.GGMLQuantizationType.Q4_0)
    writer = gguf.GGUFWriter(path, arch="ltxv")
    writer.add_tensor(tensor_name, qblock, raw_shape=qblock.shape, raw_dtype=gguf.GGMLQuantizationType.Q4_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return source


def test_is_quantized_type() -> None:
    import gguf

    assert _is_quantized_type(gguf.GGMLQuantizationType.F32) is False
    assert _is_quantized_type(gguf.GGMLQuantizationType.F16) is False
    assert _is_quantized_type(gguf.GGMLQuantizationType.BF16) is False
    assert _is_quantized_type(gguf.GGMLQuantizationType.Q4_K) is True
    assert _is_quantized_type(gguf.GGMLQuantizationType.Q5_K) is True
    assert _is_quantized_type(gguf.GGMLQuantizationType.Q6_K) is True
    assert _is_quantized_type(gguf.GGMLQuantizationType.Q4_0) is True


# ---------------------------------------------------------------------------
# _is_gemma_linear_name
# ---------------------------------------------------------------------------


def test_is_gemma_linear_name_matches_linear_suffixes() -> None:
    assert _is_gemma_linear_name("blk.0.attn_q.weight") is True
    assert _is_gemma_linear_name("blk.12.ffn_down.weight") is True
    assert _is_gemma_linear_name("blk.3.ffn_gate.weight") is True


def test_is_gemma_linear_name_rejects_norms_and_embeddings() -> None:
    assert _is_gemma_linear_name("blk.0.attn_norm.weight") is False
    assert _is_gemma_linear_name("blk.0.attn_q_norm.weight") is False
    assert _is_gemma_linear_name("token_embd.weight") is False
    assert _is_gemma_linear_name("output_norm.weight") is False


def test_is_gemma_linear_name_rejects_unmatched_prefix() -> None:
    assert _is_gemma_linear_name("ffn_down.weight") is False
    assert _is_gemma_linear_name("") is False


def test_qparam_survives_load_state_dict_assign_and_dequants() -> None:
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)  # weight shape (out=2, in=3)
    qparam = QParam(raw, gguf.GGMLQuantizationType.F32, name="x.weight")

    linear = torch.nn.Linear(3, 2, bias=False, device="meta")
    linear.__class__ = GgufLinear
    linear.load_state_dict({"weight": qparam}, assign=True)

    weight = linear.weight
    assert isinstance(weight, QParam)
    assert weight.gguf_name == "x.weight"
    # Placeholder data stays tiny: no full fp32 weight materialized.
    assert weight.numel() == 0
    assert weight.quantized_nbytes == raw.nbytes

    inp = torch.randn(1, 3)
    out = linear(inp)
    expected = torch.nn.functional.linear(inp, torch.from_numpy(raw))
    assert torch.allclose(out, expected)


def _build_qparam_linear(raw: np.ndarray, *, bias: bool = False) -> GgufLinear:
    import gguf

    qparam = QParam(raw, gguf.GGMLQuantizationType.F32, name="x.weight")
    linear = torch.nn.Linear(raw.shape[1], raw.shape[0], bias=bias, device="meta")
    linear.__class__ = GgufLinear
    linear.load_state_dict({"weight": qparam}, assign=True)
    return linear


def test_qparam_raw_survives_module_to_and_stays_cpu() -> None:
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    container = torch.nn.Module()
    container.lin = _build_qparam_linear(raw)

    container.to(device=torch.device("cpu"))  # simulates model.to(device)

    weight = container.lin.weight
    assert isinstance(weight, QParam)
    assert str(weight._raw.device) == "cpu"  # raw quantized bytes never moved
    assert torch.equal(weight._raw, torch.from_numpy(raw))

    # forward still works after the move.
    inp = torch.randn(1, 3)
    out = container.lin(inp)
    expected = torch.nn.functional.linear(inp, torch.from_numpy(raw))
    assert torch.allclose(out, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qparam_raw_survives_module_to_dtype_bfloat16_and_optional_cuda() -> None:
    """QParam must survive module.to(dtype=bf16) and, when CUDA is present,
    module.to('cuda'); the raw quantized bytes must stay where the loader left
    them (no premature materialization, no attr loss)."""
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    container = torch.nn.Module()
    container.lin = _build_qparam_linear(raw)

    container.to(dtype=torch.bfloat16)
    weight = container.lin.weight
    assert isinstance(weight, QParam)
    # forward works with bf16 input after the dtype move.
    out = container.lin(torch.randn(1, 3, dtype=torch.bfloat16))
    assert out.dtype == torch.bfloat16

    if torch.cuda.is_available():
        container.to(device=torch.device("cuda"))
        weight = container.lin.weight
        assert isinstance(weight, QParam)
        inp = torch.randn(1, 3, dtype=torch.bfloat16, device="cuda")
        out = container.lin(inp)
        assert out.device.type == "cuda"
        assert out.dtype == torch.bfloat16


def test_qparam_to_dtype_preserves_qparam() -> None:
    """Direct QParam.to(dtype=...) preserves QParam subclass, gguf_name, _raw, _tensor_type.

    This is the root cause fix for SingleGPUModelBuilder.build's
    ``sd = {key: value.to(dtype=dtype) ...}`` which runs before
    ``load_state_dict``.
    """
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    qp = QParam(raw, gguf.GGMLQuantizationType.F32, name="test.weight")

    qp2 = qp.to(dtype=torch.bfloat16)

    assert isinstance(qp2, QParam)
    assert qp2.gguf_name == "test.weight"
    assert qp2.numel() == 0  # placeholder still tiny
    assert torch.equal(qp2._raw, torch.from_numpy(raw))
    assert qp2._tensor_type == gguf.GGMLQuantizationType.F32


def test_qparam_to_device_preserves_qparam() -> None:
    """QParam.to(device=...) preserves attrs (module.to(device) path)."""
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    qp = QParam(raw, gguf.GGMLQuantizationType.F32, name="test.weight")

    qp2 = qp.to(device=torch.device("cpu"))

    assert isinstance(qp2, QParam)
    assert qp2.gguf_name == "test.weight"
    assert qp2.numel() == 0
    assert torch.equal(qp2._raw, torch.from_numpy(raw))


def test_qparam_to_identity_preserves_qparam() -> None:
    """QParam.to() with no args returns self."""
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    qp = QParam(raw, gguf.GGMLQuantizationType.F32, name="test.weight")

    qp2 = qp.to()
    assert qp2 is qp


# ---------------------------------------------------------------------------
# GGUF-comfy-like residency: device forwarding + env gates
# ---------------------------------------------------------------------------


def test_qparam_raw_device_with_explicit_device_arg() -> None:
    """QParam._raw placed on requested device when device kwarg is passed."""
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    qp = QParam(raw, gguf.GGMLQuantizationType.F32, name="test.weight", device=device)
    assert qp._raw.device.type == device.type
    assert isinstance(qp, QParam)
    assert qp.gguf_name == "test.weight"
    assert qp.numel() == 0


def test_gguf_loader_qparam_raw_on_cpu_with_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """LTX_GGUF_KEEP_RAW_ON_CPU=1 forces raw CPU even when device=torch.device('cuda')."""
    gguf_path = str(tmp_path / "quant.gguf")
    _write_quantized_gguf(gguf_path)
    monkeypatch.setenv("LTX_GGUF_KEEP_RAW_ON_CPU", "1")
    requested = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sd = GgufStateDictLoader().load(gguf_path, sd_ops=GgufNativeSDOps(), device=requested).sd
    weight = sd["x.weight"]
    assert isinstance(weight, QParam)
    assert weight._raw.device.type == "cpu"


def test_gguf_loader_qparam_raw_device_follows_device_arg(tmp_path: Path) -> None:
    """QParam raw tensor placed on device passed to load() (CUDA if available, CPU fallback)."""
    gguf_path = str(tmp_path / "quant.gguf")
    _write_quantized_gguf(gguf_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    sd = GgufStateDictLoader().load(gguf_path, sd_ops=GgufNativeSDOps(), device=device).sd
    weight = sd["x.weight"]
    assert isinstance(weight, QParam)
    assert weight._raw.device.type == device.type


def test_gguf_linear_forward_no_unconditional_empty_cache() -> None:
    """GgufLinear.forward must only call torch.cuda.empty_cache() within an
    LTX_GGUF_EMPTY_CACHE_EACH_FORWARD env gate — no unconditional call."""
    import inspect
    source = inspect.getsource(GgufLinear.forward)
    # The env gate must be present somewhere in forward
    assert "LTX_GGUF_EMPTY_CACHE_EACH_FORWARD" in source, \
        "forward missing env gate for empty_cache"
    # The env gate string must appear before the empty_cache call in source
    # (they're on separate lines: if-condition line contains gate, call is on next)
    env_pos = source.index("LTX_GGUF_EMPTY_CACHE_EACH_FORWARD")
    empty_cache_pos = source.index("torch.cuda.empty_cache()")
    assert env_pos < empty_cache_pos, \
        "torch.cuda.empty_cache() appears before env gate"


def test_gguf_loader_forward_old_behavior_gated_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Setting LTX_GGUF_EMPTY_CACHE_EACH_FORWARD=1 restores old aggressive empty_cache.

    Verifies the env gate is present by checking the source code contains the
    correct conditional. Since empty_cache is a CUDA-only call, we check the
    source contract rather than requiring a CUDA device.
    """
    import inspect
    source = inspect.getsource(GgufLinear.forward)
    # Both the env gate and the empty_cache call must be present
    assert "os.environ.get(\"LTX_GGUF_EMPTY_CACHE_EACH_FORWARD\")" in source


# ---------------------------------------------------------------------------
# Runtime LoRA on GgufLinear
# ---------------------------------------------------------------------------


def test_gguf_linear_load_after_dtype_conversion() -> None:
    """GgufLinear load_state_dict must still claim QParam after it has gone
    through ``.to(dtype=...)`` — simulating what
    SingleGPUModelBuilder.build does before load_state_dict."""
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    qp = QParam(raw, gguf.GGMLQuantizationType.F32, name="x.weight")

    # Simulate SingleGPUModelBuilder.build dtype conversion
    sd = {"weight": qp.to(dtype=torch.bfloat16)}
    assert isinstance(sd["weight"], QParam)

    linear = torch.nn.Linear(3, 2, bias=False, device="meta")
    linear.__class__ = GgufLinear
    linear.load_state_dict(sd, assign=True)

    weight = linear.weight
    assert isinstance(weight, QParam)
    assert weight.gguf_name == "x.weight"
    assert weight.numel() == 0
    assert weight.quantized_nbytes == raw.nbytes

    # Forward still works
    inp = torch.randn(1, 3)
    out = linear(inp)
    expected = torch.nn.functional.linear(inp, torch.from_numpy(raw))
    assert torch.allclose(out, expected)


def test_gguf_dequant_linear_module_op_rewrites_linear_layers() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
    out = _amend_forward_with_gguf(model)

    assert isinstance(out[0], GgufLinear)
    assert type(out[1]).__name__ == "ReLU"  # non-linear untouched
    assert isinstance(out[2], GgufLinear)


def test_install_gguf_loader_repairs_missing_module_op() -> None:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_class_configurator=LTXModelConfigurator,
        model_path="/fake/t.gguf",
        model_loader=GgufStateDictLoader(),
        model_sd_ops=GgufNativeSDOps(),
        module_ops=(),  # GGUF loader/native sd_ops present but no module op
    )
    pipe = _make_pipe_with_builder(builder)

    install_gguf_loader(pipe)

    replaced = pipe.stage._transformer_builder
    assert isinstance(replaced.model_loader, GgufStateDictLoader)
    assert isinstance(replaced.model_sd_ops, GgufNativeSDOps)
    assert any(op.name == GGUF_DEQUANT_LINEAR_OP.name for op in replaced.module_ops)


# ---------------------------------------------------------------------------
# Kijai video VAE SDOps filters
# ---------------------------------------------------------------------------


def test_kijai_video_vae_encoder_filter() -> None:
    assert KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER.apply_to_key("encoder.conv_in.conv.weight") == "conv_in.conv.weight"
    assert KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER.apply_to_key("per_channel_statistics.mean") == "per_channel_statistics.mean"
    assert KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER.apply_to_key("decoder.conv_in.conv.weight") is None


def test_kijai_video_vae_decoder_filter() -> None:
    assert KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER.apply_to_key("decoder.conv_in.conv.weight") == "conv_in.conv.weight"
    assert KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER.apply_to_key("per_channel_statistics.std") == "per_channel_statistics.std"
    assert KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER.apply_to_key("encoder.conv_in.conv.weight") is None


def test_gguf_loader_load_wraps_quantized_tensor_as_qparam(tmp_path: Path) -> None:
    import gguf
    import gguf.quants as gquants

    gguf_path = str(tmp_path / "quant.gguf")
    source = _write_quantized_gguf(gguf_path)

    sd = GgufStateDictLoader().load(gguf_path, sd_ops=GgufNativeSDOps()).sd
    assert "x.weight" in sd
    weight = sd["x.weight"]
    assert isinstance(weight, QParam)
    # Placeholder stays tiny; raw quantized bytes stored, not the fp32 weight.
    assert weight.numel() == 0
    assert weight.gguf_name == "x.weight"

    # Dequant round-trips against the Q4_0 reference.
    dequant = weight.dequant(device=torch.device("cpu"), dtype=torch.float32)
    reference = torch.from_numpy(np.ascontiguousarray(gquants.dequantize(weight._raw.numpy(), gguf.GGMLQuantizationType.Q4_0)).copy())
    assert dequant.shape == torch.Size([4, 256])
    assert torch.allclose(dequant, reference, atol=1e-5)
    # And is close to the original source (Q4_0 is lossy within tolerance).
    assert torch.allclose(dequant, torch.from_numpy(source), atol=0.1)


def test_gguf_loader_load_keeps_nonquantized_tensor_as_normal(tmp_path: Path) -> None:
    """Non-quantized (F32) tensors still load as plain tensors (not QParam),
    coerced to bf16 to match DistilledPipeline activations."""
    gguf_path = str(tmp_path / "f32.gguf")
    _write_tiny_gguf(gguf_path, with_config=False, tensor_name="x.weight")

    sd = GgufStateDictLoader().load(gguf_path, sd_ops=GgufNativeSDOps()).sd
    weight = sd["x.weight"]
    assert not isinstance(weight, QParam)
    assert isinstance(weight, torch.Tensor)
    assert weight.shape == torch.Size([3, 2])
    assert weight.dtype == torch.bfloat16


def test_gguf_loader_load_coerces_nonquantized_float_tensors_to_bfloat16(
    tmp_path: Path,
) -> None:
    """F32/F16/BF16 GGUF support tensors must load as bf16, not stored dtype.

    Models the review blocker: DistilledPipeline runs bf16 activations and feeds
    bf16 latents; an F32/F16 support tensor (norm weight, bias, scale_shift)
    left at stored dtype would make the first F.linear/RMSNorm raise a dtype
    mismatch.
    """
    import gguf

    for gguf_type, raw_bytes in _nonquant_raw_samples():
        gguf_path = str(tmp_path / f"t_{gguf_type.name}.gguf")
        writer = gguf.GGUFWriter(gguf_path, arch="ltxv")
        writer.add_tensor(
            "x.weight", raw_bytes, raw_shape=raw_bytes.shape, raw_dtype=gguf_type
        )
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()

        state = GgufStateDictLoader().load(gguf_path, sd_ops=GgufNativeSDOps())
        weight = state.sd["x.weight"]
        assert not isinstance(weight, QParam), gguf_type
        assert isinstance(weight, torch.Tensor), gguf_type
        assert weight.dtype == torch.bfloat16, gguf_type
        assert weight.shape == torch.Size([3, 2]), gguf_type


def test_gguf_loader_load_lazy_quantized_filter_matches_linear(tmp_path: Path) -> None:
    """With lazy_quantized_filter, a matched Linear tensor becomes QParam."""
    gguf_path = str(tmp_path / "gemma_linear.gguf")
    _write_quantized_gguf(gguf_path, tensor_name="blk.0.attn_q.weight")

    sd = GgufStateDictLoader(
        lazy_quantized=True,
        lazy_quantized_filter=_is_gemma_linear_name,
    ).load(gguf_path, sd_ops=GgufNativeSDOps()).sd

    assert isinstance(sd["blk.0.attn_q.weight"], QParam)


def test_gguf_loader_load_lazy_quantized_filter_rejects_norm(tmp_path: Path) -> None:
    """With lazy_quantized_filter, a non-matched quantized tensor dequants eagerly."""
    gguf_path = str(tmp_path / "gemma_norm.gguf")
    _write_quantized_gguf(gguf_path, tensor_name="blk.0.attn_norm.weight")

    sd = GgufStateDictLoader(
        lazy_quantized=True,
        lazy_quantized_filter=_is_gemma_linear_name,
    ).load(gguf_path, sd_ops=GgufNativeSDOps()).sd

    weight = sd["blk.0.attn_norm.weight"]
    assert not isinstance(weight, QParam)
    assert weight.shape == torch.Size([4, 256])
    assert weight.dtype == torch.bfloat16


def test_gguf_loader_load_lazy_quantized_filter_default_noop(tmp_path: Path) -> None:
    """Default lazy_quantized_filter=None keeps existing lazy behavior for non-Gemma GGUFs."""
    gguf_path = str(tmp_path / "native.gguf")
    _write_quantized_gguf(gguf_path, tensor_name="x.weight")

    sd = GgufStateDictLoader(
        lazy_quantized=True,
    ).load(gguf_path, sd_ops=GgufNativeSDOps()).sd

    assert isinstance(sd["x.weight"], QParam)


def test_gguf_linear_forward_casts_nonqparam_floating_bias_to_input_dtype() -> None:
    """A non-QParam bias at a wrong dtype/device must be cast to the compute dtype,
    matching the bf16 input (backstop for support-tensor dtype coherence)."""
    import gguf

    raw = np.arange(6, dtype=np.float32).reshape(2, 3)  # weight (out=2, in=3)
    qparam = QParam(raw, gguf.GGMLQuantizationType.F32, name="x.weight")
    linear = torch.nn.Linear(3, 2, bias=True, device="meta")
    linear.__class__ = GgufLinear
    # Bias deliberately left as a plain F32 tensor (dtype mismatch with bf16 input).
    f32_bias = torch.arange(2, dtype=torch.float32)
    linear.load_state_dict({"weight": qparam, "bias": f32_bias}, assign=True)

    inp = torch.randn(1, 3, dtype=torch.bfloat16)
    out = linear(inp)
    assert out.dtype == torch.bfloat16
    # Result equals F.linear with the bias upcast to bf16.
    expected_weight = torch.from_numpy(raw).to(torch.bfloat16)
    expected_bias = f32_bias.to(torch.bfloat16)
    assert torch.allclose(out, torch.nn.functional.linear(inp, expected_weight, expected_bias))


# ---------------------------------------------------------------------------
# torch dequant: CPU parity for Q4_K/Q5_K/Q6_K, CUDA route
# ---------------------------------------------------------------------------


def test_dequantize_gguf_tensor_torch_unsupported_returns_none() -> None:
    from services.patches.gguf_torch_dequant import dequantize_gguf_tensor_torch

    raw = torch.arange(16, dtype=torch.uint8).reshape(2, 8)
    result = dequantize_gguf_tensor_torch(
        raw, gguf.GGMLQuantizationType.F32, device=torch.device("cpu"), dtype=torch.float32
    )
    assert result is None


@pytest.mark.parametrize(
    "qtype",
    [
        gguf.GGMLQuantizationType.Q4_K,
        gguf.GGMLQuantizationType.Q5_K,
        gguf.GGMLQuantizationType.Q6_K,
    ],
)
def test_dequantize_gguf_tensor_torch_cpu_parity(qtype: object) -> None:
    from services.patches.gguf_torch_dequant import dequantize_gguf_tensor_torch

    _, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = torch.arange(type_size * 2, dtype=torch.uint8).reshape(2, type_size)

    expected_np = gguf.quants.dequantize(raw.numpy(), qtype)
    actual = dequantize_gguf_tensor_torch(
        raw, qtype, device=torch.device("cpu"), dtype=torch.float32
    )

    assert actual is not None
    assert actual.shape == expected_np.shape
    assert actual.dtype == torch.float32
    assert torch.allclose(actual, torch.from_numpy(expected_np.copy()), atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qparam_dequant_cuda() -> None:
    qtype = gguf.GGMLQuantizationType.Q4_K
    _, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = torch.arange(type_size * 2, dtype=torch.uint8).reshape(2, type_size)

    expected_np = gguf.quants.dequantize(raw.numpy(), qtype)
    qp = QParam(raw.numpy(), qtype, name="test.weight")
    out = qp.dequant(device=torch.device("cuda"), dtype=torch.bfloat16)

    assert out.device.type == "cuda"
    assert out.dtype == torch.bfloat16
    assert out.shape == expected_np.shape


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_qparam_dequant_cuda_fallback_to_numpy_on_unsupported_qtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When torch dequant returns None (unsupported qtype), numpy fallback must
    handle CUDA raw tensor by copying it to CPU before dequantizing.

    Regression test for: can't convert cuda:0 device type tensor to numpy.
    """
    from services.patches.gguf_torch_dequant import dequantize_gguf_tensor_torch

    qtype = gguf.GGMLQuantizationType.Q4_0
    _, type_size = gguf.GGML_QUANT_SIZES[qtype]
    raw = torch.arange(type_size * 2, dtype=torch.uint8).reshape(2, type_size)

    expected_np = gguf.quants.dequantize(raw.numpy(), qtype)
    # Place raw on CUDA as the loader does when device="cuda"
    qp = QParam(raw.numpy(), qtype, name="test.weight", device=torch.device("cuda"))
    assert qp._raw.device.type == "cuda", "raw must be on CUDA for this test"

    # Monkeypatch torch dequant to return None, forcing the numpy fallback
    original_fn = dequantize_gguf_tensor_torch

    def _return_none(*args: object, **kwargs: object) -> None:
        return None

    monkeypatch.setattr(
        "services.patches.gguf_torch_dequant.dequantize_gguf_tensor_torch",
        _return_none,
    )
    try:
        out = qp.dequant(device=torch.device("cuda"), dtype=torch.bfloat16)
    finally:
        monkeypatch.setattr(
            "services.patches.gguf_torch_dequant.dequantize_gguf_tensor_torch",
            original_fn,
        )

    assert out.device.type == "cuda"
    assert out.dtype == torch.bfloat16
    assert out.shape == expected_np.shape
    assert torch.allclose(
        out.float(), torch.from_numpy(expected_np.copy()).cuda().float(), atol=0.1
    )


# ---------------------------------------------------------------------------
# install_gguf_loader: multi-stage support (IC-LoRA: stage_1, stage_2)
# ---------------------------------------------------------------------------


def test_install_gguf_loader_patches_stage_1_and_stage_2() -> None:
    """Patches stage_1 and stage_2 when no stage exists (IC-LoRA upstream)."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    builder_1 = SingleGPUModelBuilder(
        model_path="/fake/s1.gguf",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    builder_2 = SingleGPUModelBuilder(
        model_path="/fake/s2.gguf",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pipe = SimpleNamespace(
        stage_1=SimpleNamespace(_transformer_builder=builder_1),
        stage_2=SimpleNamespace(_transformer_builder=builder_2),
    )

    install_gguf_loader(pipe)

    s1 = pipe.stage_1._transformer_builder
    s2 = pipe.stage_2._transformer_builder
    assert isinstance(s1.model_loader, GgufStateDictLoader)
    assert isinstance(s2.model_loader, GgufStateDictLoader)
    assert s1 is not builder_1
    assert s2 is not builder_2


def test_install_gguf_loader_skips_missing_stage_and_stage_1() -> None:
    """Only patches the stage that exists with a _transformer_builder."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_path="/fake/t.gguf",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    pipe = SimpleNamespace(
        stage=SimpleNamespace(_transformer_builder=builder),
        stage_1=SimpleNamespace(),  # exists but no _transformer_builder
        stage_2=None,  # missing entirely
    )

    install_gguf_loader(pipe)

    assert isinstance(pipe.stage._transformer_builder.model_loader, GgufStateDictLoader)
    assert not hasattr(pipe.stage_1, "_transformer_builder")


# ---------------------------------------------------------------------------
# install_gguf_component_paths: optional components
# ---------------------------------------------------------------------------


def test_install_gguf_component_paths_handles_optional_components() -> None:
    """Does not require upsampler/audio_decoder to exist (A2V, retake)."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXModelConfigurator

    vae_path = "/explicit/vae.safetensors"
    enc_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    dec_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    # Pipeline with image_conditioner + video_decoder only (no upsampler, no audio components)
    pipe = SimpleNamespace(
        image_conditioner=SimpleNamespace(_encoder_builder=enc_builder),
        video_decoder=SimpleNamespace(_decoder_builder=dec_builder),
    )

    install_gguf_component_paths(pipe, ("/checkpoint.safetensors",), video_vae_path=vae_path)

    assert pipe.image_conditioner._encoder_builder.model_path == vae_path
    assert pipe.video_decoder._decoder_builder.model_path == vae_path


def test_install_gguf_component_paths_patches_audio_conditioner() -> None:
    """Patches audio_conditioner._encoder_builder with audio VAE path only (no filter)."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXModelConfigurator

    video_vae = "/videos/vae.safetensors"
    audio_vae = "/audio/vae.safetensors"
    enc_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    dec_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    voc_builder = SingleGPUModelBuilder(
        model_path="/initial/wrong.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    pipe = SimpleNamespace(
        image_conditioner=SimpleNamespace(_encoder_builder=enc_builder),
        audio_conditioner=SimpleNamespace(_encoder_builder=enc_builder),
        audio_decoder=SimpleNamespace(
            _decoder_builder=dec_builder,
            _vocoder_builder=voc_builder,
        ),
    )

    install_gguf_component_paths(
        pipe, ("/checkpoint.safetensors",),
        video_vae_path=video_vae,
        audio_vae_path=audio_vae,
    )

    assert pipe.image_conditioner._encoder_builder.model_path == video_vae
    # audio_conditioner gets audio VAE path without encoder filter
    assert pipe.audio_conditioner._encoder_builder.model_path == audio_vae
    assert pipe.audio_decoder._decoder_builder.model_path == audio_vae
    assert pipe.audio_decoder._vocoder_builder.model_path == audio_vae


def test_install_gguf_component_paths_raises_only_when_expected_component_missing() -> None:
    """Raises when video components exist but no video VAE."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_path="/dummy.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )

    with pytest.raises(RuntimeError, match="missing video VAE"):
        install_gguf_component_paths(
            SimpleNamespace(
                image_conditioner=SimpleNamespace(_encoder_builder=builder),
            ),
            ("/checkpoint.safetensors",),
            video_vae_path=None,
        )

    # No audio components → no audio VAE required
    pipe = SimpleNamespace(
        image_conditioner=SimpleNamespace(_encoder_builder=builder),
    )
    install_gguf_component_paths(
        pipe,
        ("/checkpoint.safetensors",),
        video_vae_path="/videos/vae.safetensors",
        audio_vae_path=None,
    )
    assert pipe.image_conditioner._encoder_builder.model_path == "/videos/vae.safetensors"


def test_install_gguf_component_paths_applies_kijai_vae_filters() -> None:
    """Applies Kijai encoder/decoder key filters regardless of checkpoint format."""
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXModelConfigurator
    from services.patches.gguf_loader_fix import (
        KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER,
        KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER,
    )

    video_vae = "/videos/vae.safetensors"
    enc_builder = SingleGPUModelBuilder(
        model_path="/initial.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )
    dec_builder = SingleGPUModelBuilder(
        model_path="/initial.safetensors",
        model_class_configurator=LTXModelConfigurator,
    )

    pipe = SimpleNamespace(
        image_conditioner=SimpleNamespace(_encoder_builder=enc_builder),
        video_decoder=SimpleNamespace(_decoder_builder=dec_builder),
    )

    # Call with non-GGUF path (as split safetensors would)
    install_gguf_component_paths(
        pipe, ("/checkpoint.safetensors",),
        video_vae_path=video_vae,
    )

    assert pipe.image_conditioner._encoder_builder.model_path == video_vae
    assert (
        pipe.image_conditioner._encoder_builder.model_sd_ops
        == KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER
    )
    assert (
        pipe.video_decoder._decoder_builder.model_sd_ops
        == KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER
    )


# ---------------------------------------------------------------------------
# Gemma GGUF sd_ops
# ---------------------------------------------------------------------------


def test_gemma_sdops_converts_norm_weights_to_hf_form() -> None:
    value = torch.tensor([1.5, 2.0])
    result = GgufGemmaSDOps().apply_to_key_value("model.model.language_model.norm.weight", value)

    assert torch.equal(result[0].new_value, torch.tensor([0.5, 1.0]))


def test_gguf_gemma_dequant_linear_op_has_correct_name_and_mutator() -> None:
    """GGUF_GEMMA_DEQUANT_LINEAR_OP name and mutator are wired correctly."""
    assert GGUF_GEMMA_DEQUANT_LINEAR_OP.name == "gguf_gemma_dequant_linear"
    assert GGUF_GEMMA_DEQUANT_LINEAR_OP.mutator is not None
    assert GGUF_GEMMA_DEQUANT_LINEAR_OP.mutator.__name__ == "_amend_forward_with_gguf"


# ---------------------------------------------------------------------------
# _find_gemma_gguf
# ---------------------------------------------------------------------------


def test_find_gemma_gguf_raises_on_none():
    with pytest.raises(ValueError, match="Gemma GGUF path"):
        _find_gemma_gguf(None)


def test_find_gemma_gguf_raises_on_empty():
    with pytest.raises(ValueError, match="Gemma GGUF path"):
        _find_gemma_gguf("")


def test_find_gemma_gguf_accepts_direct_gguf_file(tmp_path: Path) -> None:
    gguf_path = tmp_path / "gemma-2-2b-it-q4_0.gguf"
    gguf_path.write_text("dummy")
    result = _find_gemma_gguf(str(gguf_path))
    assert result == gguf_path


# ---------------------------------------------------------------------------
# _resolve_gemma_tokenizer_root
# ---------------------------------------------------------------------------


def test_resolve_gemma_tokenizer_root_gguf_file_resolves_to_parent(tmp_path: Path) -> None:
    """A .gguf file path resolves to its parent directory."""
    gguf_path = tmp_path / "gemma-3-12b-it.gguf"
    gguf_path.write_text("dummy")
    result = _resolve_gemma_tokenizer_root(str(gguf_path))
    assert result == str(tmp_path)


def test_resolve_gemma_tokenizer_root_directory_passes_through(tmp_path: Path) -> None:
    """A directory path passes through unchanged."""
    result = _resolve_gemma_tokenizer_root(str(tmp_path))
    assert result == str(tmp_path)


def test_patched_init_passes_tokenizer_root_not_gguf_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patched_init must pass the tokenizer root directory (not the .gguf file path)
    to module_ops_from_gemma_root."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

    gguf_path = tmp_path / "gemma-3-12b-it.gguf"
    gguf_path.write_text("dummy")

    captured_root: list[str] = []

    class _FakeEncoder:
        pass

    def fake_module_ops(root: str) -> list:
        captured_root.append(root)
        return []

    # Patch the source module so the import inside install_gguf_prompt_encoder_patch
    # picks up the fake.
    monkeypatch.setattr(
        "ltx_core.text_encoders.gemma.module_ops_from_gemma_root",
        fake_module_ops,
    )

    # Must be called after the monkeypatch, because the function re-imports
    # module_ops_from_gemma_root from the source module.
    install_gguf_prompt_encoder_patch()

    blocks.PromptEncoder.__init__(
        _FakeEncoder(),
        checkpoint_path=str(tmp_path / "checkpoint.safetensors"),
        gemma_root=str(gguf_path),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )

    assert len(captured_root) == 1
    assert captured_root[0] == str(tmp_path)


def test_patched_init_embeddings_builder_uses_upstream_ops_for_safetensors_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patched_init uses EMBEDDINGS_PROCESSOR_KEY_OPS (not GgufEmbeddingsProcessorSDOps)
    when checkpoint_path contains only .safetensors paths (Kijai split or official)."""
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        module_ops_from_gemma_root,
    )
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

    gguf_path = tmp_path / "gemma-3-12b-it.gguf"
    gguf_path.write_text("dummy")
    safetensors_path = tmp_path / "model.diffusion_model.safetensors"
    safetensors_path.write_bytes(b"x")

    monkeypatch.setattr(
        "ltx_core.text_encoders.gemma.module_ops_from_gemma_root",
        lambda root: [],
    )

    install_gguf_prompt_encoder_patch()

    class _FakeEncoder1:
        pass
    encoder1 = _FakeEncoder1()
    blocks.PromptEncoder.__init__(
        encoder1,
        checkpoint_path=(
            str(tmp_path / "transformer.safetensors"),
            str(tmp_path / "tp.safetensors"),
        ),
        gemma_root=str(gguf_path),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    sd_ops = encoder1._embeddings_processor_builder.model_sd_ops  # type: ignore[attr-defined]
    assert sd_ops.name == EMBEDDINGS_PROCESSOR_KEY_OPS.name, (
        f"Expected upstream ops, got {sd_ops.name}"
    )

    # --- single safetensors path (official) ---
    encoder2 = _FakeEncoder1()
    blocks.PromptEncoder.__init__(
        encoder2,
        checkpoint_path=str(tmp_path / "model.safetensors"),
        gemma_root=str(gguf_path),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    sd_ops2 = encoder2._embeddings_processor_builder.model_sd_ops  # type: ignore[attr-defined]
    assert sd_ops2.name == EMBEDDINGS_PROCESSOR_KEY_OPS.name


def test_patched_init_embeddings_builder_uses_gguf_ops_for_gguf_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patched_init uses GgufEmbeddingsProcessorSDOps when any path is .gguf."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import (
        GgufEmbeddingsProcessorSDOps,
        install_gguf_prompt_encoder_patch,
    )

    gguf_path = tmp_path / "gemma-3-12b-it.gguf"
    gguf_path.write_text("dummy")

    monkeypatch.setattr(
        "ltx_core.text_encoders.gemma.module_ops_from_gemma_root",
        lambda root: [],
    )

    install_gguf_prompt_encoder_patch()

    class _FakeEncoder2:
        pass
    encoder = _FakeEncoder2()
    blocks.PromptEncoder.__init__(
        encoder,
        checkpoint_path=(
            str(tmp_path / "transformer.gguf"),
            str(tmp_path / "tp.safetensors"),
        ),
        gemma_root=str(gguf_path),
        dtype=torch.bfloat16,
        device=torch.device("cpu"),
    )
    sd_ops = encoder._embeddings_processor_builder.model_sd_ops  # type: ignore[attr-defined]
    assert isinstance(sd_ops, GgufEmbeddingsProcessorSDOps), (
        f"Expected GgufEmbeddingsProcessorSDOps, got {type(sd_ops).__name__}"
    )


# ---------------------------------------------------------------------------
# Runtime LoRA on GgufLinear
# ---------------------------------------------------------------------------


def test_gguf_linear_runtime_lora_matches_manual_math() -> None:
    """Runtime LoRA delta in GgufLinear.forward matches manual fused math."""
    import gguf

    in_f, out_f, rank = 8, 4, 2
    rng = np.random.RandomState(42)
    raw = rng.randn(out_f, in_f).astype(np.float32)
    lora_A = torch.randn(rank, in_f)
    lora_B = torch.randn(out_f, rank)
    strength = 0.5

    qparam = QParam(raw, gguf.GGMLQuantizationType.F32, name="test.weight")
    linear = torch.nn.Linear(in_f, out_f, bias=False, device="meta")
    linear.__class__ = GgufLinear
    linear.load_state_dict({"weight": qparam}, assign=True)

    linear.lora_pairs = ((lora_A.cpu(), lora_B.cpu(), strength),)

    inp = torch.randn(2, in_f)

    # Path A: runtime GgufLinear.forward
    out_runtime = linear(inp)

    # Path B: manual dequant weight + fused delta
    base_weight = torch.from_numpy(raw)
    delta = torch.nn.functional.linear(
        torch.nn.functional.linear(inp, lora_A), lora_B
    ) * strength
    out_manual = torch.nn.functional.linear(inp, base_weight) + delta

    assert torch.allclose(out_runtime, out_manual, atol=1e-5)


# ---------------------------------------------------------------------------
# _find_v2_embeddings_config
# ---------------------------------------------------------------------------


@pytest.fixture
def _v2_config() -> dict[str, object]:
    return {
        "transformer": {
            "caption_proj_before_connector": True,
            "caption_projection_first_linear": False,
            "caption_proj_input_norm": False,
            "caption_projection_second_linear": False,
            "num_layers": 7,
        }
    }


@pytest.fixture
def _v1_config() -> dict[str, object]:
    return {"transformer": {"num_layers": 7, "attention_head_dim": 128}}


def _make_safetensors(path: Path, config: dict[str, object], tensors: dict[str, torch.Tensor] | None = None) -> None:
    from safetensors.torch import save_file
    save_file(tensors or {"dummy": torch.zeros(1)}, str(path), metadata={"config": json.dumps(config)})


def test_find_v2_embeddings_config_prefers_text_projection(
    tmp_path: Path,
    _v1_config: dict[str, object],
    _v2_config: dict[str, object],
) -> None:
    """When transformer has V1 config and text_projection has V2 config,
    _find_v2_embeddings_config returns the V2 config."""
    transformer = tmp_path / "transformer.safetensors"
    text_proj = tmp_path / "text_projection.safetensors"
    _make_safetensors(transformer, _v1_config)
    _make_safetensors(text_proj, _v2_config)

    result = _find_v2_embeddings_config([str(transformer), str(text_proj)])
    assert result is not None
    assert result["transformer"]["caption_proj_before_connector"] is True


def test_find_v2_embeddings_config_no_v2_returns_none(
    tmp_path: Path,
    _v1_config: dict[str, object],
) -> None:
    """When no safetensors has V2 config keys, returns None."""
    paths: list[str] = []
    for name in ("transformer.safetensors", "other.safetensors"):
        p = tmp_path / name
        _make_safetensors(p, _v1_config)
        paths.append(str(p))

    assert _find_v2_embeddings_config(paths) is None


def test_find_v2_embeddings_config_no_safetensors_returns_none() -> None:
    """When no .safetensors paths exist, returns None without error."""
    assert _find_v2_embeddings_config(["/fake/model.gguf", "/fake/tp.pt"]) is None


def test_find_v2_embeddings_config_empty_paths_returns_none() -> None:
    """Empty path list returns None without error."""
    assert _find_v2_embeddings_config([]) is None


# ---------------------------------------------------------------------------
# Safetensors test helpers
# ---------------------------------------------------------------------------


_V1_CONFIG: dict[str, object] = {
    "transformer": {
        "connector_num_layers": 2,
        "connector_attention_head_dim": 128,
        "connector_num_attention_heads": 32,
        "audio_connector_attention_head_dim": 128,
        "audio_connector_num_attention_heads": 16,
        "audio_connector_num_layers": 2,
    }
}

_V2_CONFIG: dict[str, object] = {
    "transformer": {
        "activation_fn": "gelu-approximate",
        "apply_gated_attention": True,
        "attention_bias": True,
        "attention_head_dim": 128,
        "attention_type": "default",
        "audio_attention_head_dim": 64,
        "audio_connector_attention_head_dim": 64,
        "audio_connector_num_attention_heads": 32,
        "audio_cross_attention_dim": 2048,
        "audio_num_attention_heads": 32,
        "audio_out_channels": 128,
        "audio_positional_embedding_max_pos": [20],
        "av_ca_timestep_scale_multiplier": 1000.0,
        "av_cross_ada_norm": True,
        "caption_channels": 3840,
        "caption_proj_before_connector": True,
        "caption_proj_input_norm": False,
        "caption_projection_first_linear": False,
        "caption_projection_second_linear": False,
        "causal_temporal_positioning": True,
        "connector_attention_head_dim": 128,
        "connector_apply_gated_attention": True,
        "connector_num_attention_heads": 32,
        "connector_num_layers": 8,
        "connector_num_learnable_registers": 128,
        "connector_positional_embedding_max_pos": [4096],
        "cross_attention_adaln": True,
        "cross_attention_dim": 4096,
        "cross_attention_norm": True,
        "double_self_attention": False,
        "dropout": 0.0,
        "frequencies_precision": "float64",
        "in_channels": 128,
        "norm_elementwise_affine": False,
        "norm_eps": 1e-06,
        "norm_num_groups": 32,
        "num_attention_heads": 32,
        "num_embeds_ada_norm": 1000,
        "num_layers": 7,
        "num_vector_embeds": None,
        "only_cross_attention": False,
        "out_channels": 128,
        "positional_embedding_max_pos": [20, 2048, 2048],
        "qk_norm": "rms_norm",
        "use_audio_video_cross_attention": True,
    }
}


@pytest.fixture
def _v1_config() -> dict[str, object]:
    return dict(_V1_CONFIG)


@pytest.fixture
def _v2_config() -> dict[str, object]:
    return dict(_V2_CONFIG)


def _make_safetensors(path: Path, meta: dict[str, object]) -> None:
    """Write a minimal .safetensors file with __metadata__.

    Uses safe_open's expected format: 8-byte header length + JSON header + tensor data.
    """
    import json

    header: dict[str, object] = {"__metadata__": {"config": json.dumps(meta)}}
    encoded = json.dumps(header).encode()
    length = len(encoded).to_bytes(8, "little")
    path.write_bytes(length + encoded)


# ---------------------------------------------------------------------------
# install_kijai_transformer_config_patch
# ---------------------------------------------------------------------------


def _make_pipe_single_stage() -> SimpleNamespace:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    builder = SingleGPUModelBuilder(
        model_path="/fake/transformer.safetensors",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    return SimpleNamespace(stage=SimpleNamespace(_transformer_builder=builder))


def _make_pipe_two_stage() -> SimpleNamespace:
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
    from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

    b1 = SingleGPUModelBuilder(
        model_path="/fake/stage_1.safetensors",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    b2 = SingleGPUModelBuilder(
        model_path="/fake/stage_2.safetensors",
        model_class_configurator=LTXModelConfigurator,
        model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
    )
    return SimpleNamespace(
        stage_1=SimpleNamespace(_transformer_builder=b1),
        stage_2=SimpleNamespace(_transformer_builder=b2),
    )


class TestInstallKijaiTransformerConfigPatch:
    """Tests for install_kijai_transformer_config_patch."""

    def test_patches_builder_with_v2_config(
        self,
        tmp_path: Path,
        _v1_config: dict[str, object],
        _v2_config: dict[str, object],
    ) -> None:
        """Safetensors-only tuple: builder metadata returns V2 config after patch."""
        transformer = tmp_path / "transformer.safetensors"
        text_proj = tmp_path / "text_projection.safetensors"
        _make_safetensors(transformer, _v1_config)
        _make_safetensors(text_proj, _v2_config)

        checkpoint_path = (str(transformer), str(text_proj))
        # Build pipe with real paths so model_config() works
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

        builder = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        pipe = SimpleNamespace(stage=SimpleNamespace(_transformer_builder=builder))

        original_meta = pipe.stage._transformer_builder.model_config()
        # original config is from transformer.safetensors (V1 — no caption_proj_before_connector)
        assert original_meta.get("transformer", {}).get("caption_proj_before_connector") is not True

        from services.patches.gguf_loader_fix import install_kijai_transformer_config_patch

        install_kijai_transformer_config_patch(pipe, checkpoint_path)

        patched_meta = pipe.stage._transformer_builder.model_config()
        assert patched_meta["transformer"]["caption_proj_before_connector"] is True
        assert patched_meta["transformer"]["caption_channels"] == 3840
        assert patched_meta["transformer"]["num_layers"] == 7

    def test_skips_gguf_tuple(
        self,
        tmp_path: Path,
        _v2_config: dict[str, object],
    ) -> None:
        """GGUF-containing tuple: builder unchanged, loader not swapped."""
        transformer = tmp_path / "transformer.gguf"
        transformer.write_bytes(b"GGUF")
        text_proj = tmp_path / "text_projection.safetensors"
        _make_safetensors(text_proj, _v2_config)

        checkpoint_path = (str(transformer), str(text_proj))
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

        builder = SingleGPUModelBuilder(
            model_path=(str(transformer),),
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        pipe = SimpleNamespace(stage=SimpleNamespace(_transformer_builder=builder))

        original_loader = pipe.stage._transformer_builder.model_loader

        from services.patches.gguf_loader_fix import install_kijai_transformer_config_patch

        install_kijai_transformer_config_patch(pipe, checkpoint_path)

        from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader
        assert isinstance(pipe.stage._transformer_builder.model_loader, SafetensorsModelStateDictLoader)
        assert pipe.stage._transformer_builder.model_loader is original_loader

    def test_skips_single_string_path(self) -> None:
        """Single string path: no-op, builder unchanged."""
        pipe = _make_pipe_single_stage()
        original_loader = pipe.stage._transformer_builder.model_loader

        from services.patches.gguf_loader_fix import install_kijai_transformer_config_patch

        install_kijai_transformer_config_patch(pipe, "/fake/model.safetensors")

        assert pipe.stage._transformer_builder.model_loader is original_loader

    def test_patches_both_stages_two_stage(
        self,
        tmp_path: Path,
        _v1_config: dict[str, object],
        _v2_config: dict[str, object],
    ) -> None:
        """Two-stage pipeline (IC-LoRA / A2V): both stage_1 and stage_2 get V2 config."""
        transformer = tmp_path / "transformer.safetensors"
        text_proj = tmp_path / "text_projection.safetensors"
        _make_safetensors(transformer, _v1_config)
        _make_safetensors(text_proj, _v2_config)

        checkpoint_path = (str(transformer), str(text_proj))
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

        b1 = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        b2 = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        pipe = SimpleNamespace(
            stage_1=SimpleNamespace(_transformer_builder=b1),
            stage_2=SimpleNamespace(_transformer_builder=b2),
        )

        from services.patches.gguf_loader_fix import install_kijai_transformer_config_patch

        install_kijai_transformer_config_patch(pipe, checkpoint_path)

        meta_1 = pipe.stage_1._transformer_builder.model_config()
        assert meta_1["transformer"]["caption_proj_before_connector"] is True
        assert meta_1["transformer"]["caption_channels"] == 3840

        meta_2 = pipe.stage_2._transformer_builder.model_config()
        assert meta_2["transformer"]["caption_proj_before_connector"] is True
        assert meta_2["transformer"]["caption_channels"] == 3840

    def test_no_v2_config_skips_gracefully(
        self,
        tmp_path: Path,
        _v1_config: dict[str, object],
    ) -> None:
        """When no V2 config found, builder unchanged, no error."""
        transformer = tmp_path / "transformer.safetensors"
        other = tmp_path / "other.safetensors"
        _make_safetensors(transformer, _v1_config)
        _make_safetensors(other, _v1_config)

        checkpoint_path = (str(transformer), str(other))
        from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
        from ltx_core.model.transformer import LTXV_MODEL_COMFY_RENAMING_MAP, LTXModelConfigurator

        builder = SingleGPUModelBuilder(
            model_path=checkpoint_path,
            model_class_configurator=LTXModelConfigurator,
            model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
        )
        pipe = SimpleNamespace(stage=SimpleNamespace(_transformer_builder=builder))

        original_loader = pipe.stage._transformer_builder.model_loader

        from services.patches.gguf_loader_fix import install_kijai_transformer_config_patch

        install_kijai_transformer_config_patch(pipe, checkpoint_path)

        assert pipe.stage._transformer_builder.model_loader is original_loader


# ---------------------------------------------------------------------------
# llama.cpp standalone enhancement helper + call patch
# ---------------------------------------------------------------------------


def test_llama_cpp_standalone_helper_loads_system_prompt() -> None:
    """_load_gemma_t2v_system_prompt reads the prompt file correctly."""
    from services.patches.gguf_loader_fix import _load_gemma_t2v_system_prompt

    prompt = _load_gemma_t2v_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 50
    # ponytail: exact file content not validated; assertion will fail if file is missing
    assert "Creative Assistant" in prompt


def test_llama_cpp_call_patch_is_idempotent() -> None:
    """Installing the __call__ patch twice does not wrap again."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import _install_llama_cpp_prompt_encoder_call_patch

    _install_llama_cpp_prompt_encoder_call_patch()
    first_ref = blocks.PromptEncoder.__call__
    _install_llama_cpp_prompt_encoder_call_patch()
    second_ref = blocks.PromptEncoder.__call__

    assert first_ref is second_ref


# ---------------------------------------------------------------------------
# llama.cpp __call__ timing patch behavior
# ---------------------------------------------------------------------------


def test_gguf_call_patch_gguf_path_calls_llama_cpp_then_original(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GGUF model_path + enhance_first_prompt=True: llama helper called, original with enhance=False."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import _install_llama_cpp_prompt_encoder_call_patch

    # Replace __call__ with recording wrapper BEFORE patch captures it as original_call.
    recorded: dict[str, object] = {}

    def recording(
        self: object,
        prompts: list[str],
        **kwargs: object,
    ) -> list[object]:
        recorded["prompts"] = list(prompts)
        recorded["kwargs"] = kwargs
        return [prompts[0]]

    monkeypatch.setattr(blocks.PromptEncoder, "__call__", recording)
    _install_llama_cpp_prompt_encoder_call_patch()

    llama_called: list[tuple[object, ...]] = []

    def fake_enhance(
        model_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        llama_called.append((model_path, prompt, max_new_tokens, seed))
        return f"ENHANCED:{prompt}"

    monkeypatch.setattr(
        "services.patches.gguf_loader_fix._enhance_prompt_with_llama_cpp",
        fake_enhance,
    )

    encoder = SimpleNamespace(_ltx_desktop_llama_cpp_model_path="/fake/gemma.gguf")

    result = blocks.PromptEncoder.__call__(
        encoder,
        ["hello world"],
        enhance_first_prompt=True,
        enhance_prompt_seed=42,
    )

    assert len(llama_called) == 1
    assert llama_called[0][0] == "/fake/gemma.gguf"
    assert llama_called[0][1] == "hello world"
    assert recorded["kwargs"].get("enhance_first_prompt") is False
    assert "ENHANCED:hello world" in recorded["prompts"][0]
    assert result == ["ENHANCED:hello world"]


def test_gguf_call_patch_non_gguf_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No model_path: original called with original enhance flag, llama helper not called."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import _install_llama_cpp_prompt_encoder_call_patch

    recorded: dict[str, object] = {}

    def recording(
        self: object,
        prompts: list[str],
        **kwargs: object,
    ) -> list[object]:
        recorded["prompts"] = list(prompts)
        recorded["kwargs"] = kwargs
        return [prompts[0]]

    monkeypatch.setattr(blocks.PromptEncoder, "__call__", recording)
    _install_llama_cpp_prompt_encoder_call_patch()

    llama_called: list[tuple[object, ...]] = []

    def fake_enhance(
        model_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        llama_called.append((model_path, prompt, max_new_tokens, seed))
        return f"ENHANCED:{prompt}"

    monkeypatch.setattr(
        "services.patches.gguf_loader_fix._enhance_prompt_with_llama_cpp",
        fake_enhance,
    )

    encoder = SimpleNamespace()  # no _ltx_desktop_llama_cpp_model_path

    result = blocks.PromptEncoder.__call__(
        encoder,
        ["hello world"],
        enhance_first_prompt=True,
        enhance_prompt_seed=42,
    )

    assert len(llama_called) == 0
    assert recorded["kwargs"].get("enhance_first_prompt") is True
    assert recorded["prompts"][0] == "hello world"
    assert result == ["hello world"]


def test_gguf_call_patch_image_enhance_passthrough(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """enhance_prompt_image set: original called with original flags, llama helper not called."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import _install_llama_cpp_prompt_encoder_call_patch

    recorded: dict[str, object] = {}

    def recording(
        self: object,
        prompts: list[str],
        **kwargs: object,
    ) -> list[object]:
        recorded["prompts"] = list(prompts)
        recorded["kwargs"] = kwargs
        return [prompts[0]]

    monkeypatch.setattr(blocks.PromptEncoder, "__call__", recording)
    _install_llama_cpp_prompt_encoder_call_patch()

    llama_called: list[tuple[object, ...]] = []

    def fake_enhance(
        model_path: str,
        prompt: str,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        llama_called.append((model_path, prompt, max_new_tokens, seed))
        return f"ENHANCED:{prompt}"

    monkeypatch.setattr(
        "services.patches.gguf_loader_fix._enhance_prompt_with_llama_cpp",
        fake_enhance,
    )

    encoder = SimpleNamespace(_ltx_desktop_llama_cpp_model_path="/fake/gemma.gguf")

    result = blocks.PromptEncoder.__call__(
        encoder,
        ["hello world"],
        enhance_first_prompt=True,
        enhance_prompt_image="/path/to/image.png",
        enhance_prompt_seed=42,
    )

    assert len(llama_called) == 0
    assert recorded["kwargs"].get("enhance_first_prompt") is True
    assert recorded["kwargs"].get("enhance_prompt_image") == "/path/to/image.png"
    assert recorded["prompts"][0] == "hello world"
    assert result == ["hello world"]


def test_patched_init_empty_gemma_root_skips_find_gemma_gguf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty gemma_root in API mode: patched_init delegates to original_init, not _find_gemma_gguf."""
    from ltx_pipelines.utils import blocks
    from services.patches.gguf_loader_fix import install_gguf_prompt_encoder_patch

    saved_init = blocks.PromptEncoder.__init__

    find_called: list[object] = []

    def tracking_find(gemma_root: str | None) -> None:
        find_called.append(gemma_root)
        raise RuntimeError("_find_gemma_gguf was called")

    monkeypatch.setattr(
        "services.patches.gguf_loader_fix._find_gemma_gguf",
        tracking_find,
    )

    install_gguf_prompt_encoder_patch()

    patched_init = blocks.PromptEncoder.__init__
    assert getattr(patched_init, "_ltx_desktop_gguf_patch", False)

    # Empty/None gemma_root → patched_init delegates to original_init (which
    # also fails — no real files). The guard means _find_gemma_gguf is NEVER
    # called; if it were, tracking_find raises RuntimeError.
    for root in ("", None):
        find_called.clear()
        try:
            patched_init(SimpleNamespace(), "ckpt", root, torch.float32, torch.device("cpu"))  # type: ignore[arg-type]
        except RuntimeError:
            pytest.fail(f"_find_gemma_gguf called for gemma_root={root!r}")
        except Exception:
            pass  # expected: original_init can't build without real model files
        assert len(find_called) == 0, f"_find_gemma_gguf called for gemma_root={root!r}"

    blocks.PromptEncoder.__init__ = saved_init


# ---------------------------------------------------------------------------
# install_gguf_t2v_conditioning_patch: ImageConditioner patched_call
# ---------------------------------------------------------------------------


def _fake_original_call(self: object, fn: object) -> object:
    """Simulate the real ImageConditioner.__call__: calls fn with a real encoder sentinel."""
    return fn("REAL_ENCODER")


class TestT2VConditioningPatch:
    """Tests for install_gguf_t2v_conditioning_patch patched_call logic."""

    def _apply_patch_against(self, monkeypatch: pytest.MonkeyPatch) -> object:
        from ltx_pipelines.utils import blocks
        from services.patches.gguf_loader_fix import install_gguf_t2v_conditioning_patch

        monkeypatch.setattr(blocks.ImageConditioner, "__call__", _fake_original_call)
        install_gguf_t2v_conditioning_patch()
        return blocks.ImageConditioner.__call__

    def test_empty_images_no_video_conditioning_returns_fn_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty images=[] with no video_conditioning in closure: fn(None) shortcut."""
        patched = self._apply_patch_against(monkeypatch)

        images: list = []  # noqa: F841 — must be referenced by lambda to be in closure
        video_conditioning: list | None = None  # noqa: F841
        result = patched(
            object(),
            lambda enc: f"{enc}_{len(images)}_{video_conditioning}",  # type: ignore[operator]
        )

        # shortcut: fn(None) → enc is None
        assert result == "None_0_None"

    def test_empty_images_with_video_conditioning_calls_original(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty images=[] but video_conditioning=[(...)]: original path, gets real encoder."""
        patched = self._apply_patch_against(monkeypatch)

        images: list = []  # noqa: F841
        video_conditioning = [("/fake/video.mp4", 1.0)]  # noqa: F841
        result = patched(
            object(),
            lambda enc: f"{enc}_{len(images)}_{len(video_conditioning)}",  # type: ignore[operator]
        )

        # original: fn("REAL_ENCODER") → enc is the sentinel
        assert result == "REAL_ENCODER_0_1"
