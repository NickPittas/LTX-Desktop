"""Unit tests for the GGUF state-dict loader + install helper (slice 1)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from services.patches.gguf_loader_fix import (
    GGUF_DEQUANT_LINEAR_OP,
    GgufLinear,
    GgufNativeSDOps,
    GgufStateDictLoader,
    QParam,
    _amend_forward_with_gguf,
    _is_quantized_type,
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
