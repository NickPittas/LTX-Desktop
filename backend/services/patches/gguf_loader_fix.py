"""GGUF state-dict loader + injection helper for the fast T2V pipeline.

Implements the ``ltx_core.loader.primitives.StateDictLoader`` protocol by reading
transformer weights and the embedded config from GGUF checkpoints via the ``gguf``
library. Installed into a ``DistilledPipeline``'s ``DiffusionStage`` transformer
builder (see :func:`install_gguf_loader`) so a GGUF transformer replaces the
default safetensors loader.

No silent fallback: when GGUF is configured, GGUF must load or we raise.
``load`` only processes ``.gguf`` paths and skips the safetensors component
paths that share the same builder tuple (each builder filters keys via
``model_sd_ops``).
"""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import numpy as np
import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult
from ltx_core.model.transformer.model import LTXModel

# Metadata keys searched, in priority order, for the embedded JSON config.
_CONFIG_KEYS = ("config", "ltx.config", "general.config", "ltx_config")

# GGUF tensor types that store unquantized fp values; loaded eagerly as normal
# tensors (small support tensors: norms, biases, scale_shift). Everything else
# is quantized (Q4_K/Q5_K/Q6_K/...) and kept lazy via :class:`QParam`.
_NON_QUANTIZED_TYPE_NAMES = frozenset({"F32", "F16", "BF16"})


def _is_quantized_type(tensor_type: object) -> bool:
    """True for GGUF quantized types (Q4_K, Q5_K, Q6_K, IQ4_XS, ...), False for F32/F16/BF16."""
    name = getattr(tensor_type, "name", str(tensor_type))
    return name not in _NON_QUANTIZED_TYPE_NAMES


class QParam(torch.nn.Parameter):
    """A Parameter that stores raw GGUF quantized bytes and dequantizes lazily in forward.

    The Parameter's own data is a tiny float placeholder (``empty(0)``) so the model
    stays cheap to move across devices (``.to(device)`` never copies the quantized
    bytes to GPU) and no full fp32 weight is materialized at load time. The real
    quantized block lives in ``_raw`` (CPU, owned) and is dequantized on demand by
    :meth:`dequant` inside ``GgufLinear.forward``.
    """

    _raw: torch.Tensor
    _tensor_type: object
    _gguf_name: str

    def __new__(
        cls, raw: "np.ndarray | torch.Tensor", tensor_type: object, *, name: str
    ) -> "QParam":
        placeholder = torch.empty(0, dtype=torch.float32)
        obj = super().__new__(cls, placeholder, requires_grad=False)  # type: ignore[call-overload]
        if isinstance(raw, np.ndarray):
            # Copy (decoupled from the GGUFReader memmap, which is released after load).
            obj._raw = torch.from_numpy(np.ascontiguousarray(raw).copy())
        else:
            obj._raw = raw.contiguous().clone()
        obj._tensor_type = tensor_type
        obj._gguf_name = name
        return obj

    @property
    def gguf_name(self) -> str:
        return self._gguf_name

    @property
    def quantized_nbytes(self) -> int:
        return int(self._raw.nbytes)

    def dequant(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        import gguf

        try:
            array = gguf.quants.dequantize(self._raw.numpy(), self._tensor_type)
        except NotImplementedError as exc:
            raise RuntimeError(
                f"GGUF tensor '{self._gguf_name}' uses unsupported quant type "
                f"{self._tensor_type}; cannot dequantize"
            ) from exc
        target_dtype = dtype if dtype.is_floating_point else torch.float32
        return torch.from_numpy(np.ascontiguousarray(array).copy()).to(device=device, dtype=target_dtype)


class GgufLinear(torch.nn.Linear):
    """nn.Linear that dequantizes a :class:`QParam` weight/bias per forward.

    Installed via ``__class__`` reassignment (see :func:`_amend_forward_with_gguf`),
    mirroring ``ltx_core.quantization.fp8_cast.Fp8CastLinear``. A class-level
    forward + ``_load_from_state_dict`` override is required for correctness:
    ``load_state_dict(assign=True)`` still enforces a shape check, but a QParam's
    placeholder data is shape ``(0,)`` while the dequantized weight is full-sized,
    so the override assigns QParams directly, bypassing that check. Non-QParam
    tensors (e.g. F32 biases) fall through to the default path unchanged.
    """

    def _load_from_state_dict(  # type: ignore[override]
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata: dict[str, object],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        claimed: list[str] = []
        for param_name in ("weight", "bias"):
            key = prefix + param_name
            if key in state_dict and isinstance(state_dict[key], QParam):
                setattr(self, param_name, state_dict[key])
                del state_dict[key]
                claimed.append(key)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        # QParams were assigned directly above; reclaim them from the missing-key
        # accounting that the default loader recorded for the now-absent keys.
        for key in claimed:
            if key in missing_keys:
                missing_keys.remove(key)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = self.weight
        if isinstance(weight, QParam):
            compute_dtype = input.dtype if input.is_floating_point() else torch.float32
            weight = weight.dequant(device=input.device, dtype=compute_dtype)
        bias = self.bias
        if isinstance(bias, QParam):
            bias = bias.dequant(device=input.device, dtype=weight.dtype)
        elif isinstance(bias, torch.Tensor) and bias.is_floating_point():
            # Backstop: a non-QParam bias (e.g. loaded F32/BF16) must match the
            # dequanted/input dtype+device or F.linear raises a dtype mismatch.
            bias = bias.to(device=input.device, dtype=weight.dtype)
        return torch.nn.functional.linear(input, weight, bias)


def _amend_forward_with_gguf(model: torch.nn.Module) -> torch.nn.Module:
    """Swap every ``nn.Linear`` to :class:`GgufLinear` in place."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.__class__ = GgufLinear
    return model


GGUF_DEQUANT_LINEAR_OP = ModuleOps(
    name="gguf_dequant_linear",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=_amend_forward_with_gguf,
)


class GgufStateDictLoader:
    """Loads an LTX transformer from a GGUF checkpoint.

    Duck-types ``StateDictLoader``. ``metadata`` reads the embedded config;
    ``load`` dequantizes tensors via ``gguf.quants`` and applies ``SDOps`` the
    same way the safetensors loader does.
    """

    def metadata(self, path: str) -> dict[str, object]:
        import gguf

        reader = gguf.GGUFReader(path)
        for key in _CONFIG_KEYS:
            field = reader.get_field(key)
            if field is None:
                continue
            text = _coerce_config_text(field.contents())
            if text is None:
                continue
            try:
                parsed = json.loads(text)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(parsed, dict) and "transformer" in parsed:
                return parsed

        raise RuntimeError(
            f"GGUF transformer config not found in metadata of {path}; "
            "expected config/ltx.config/general.config/ltx_config JSON with a transformer object"
        )

    def load(
        self,
        path: str | list[str],
        sd_ops: object | None = None,
        device: torch.device | None = None,
    ) -> StateDict:
        import gguf

        model_paths = list(path) if isinstance(path, (list, tuple)) else [path]
        gguf_paths = [p for p in model_paths if _is_gguf_path(p)]
        if not gguf_paths:
            raise RuntimeError("GGUF loader received no .gguf path; this is a pipeline wiring bug")

        target_device = device or torch.device("cpu")
        sd: dict[str, torch.Tensor] = {}
        size = 0
        dtype: set[torch.dtype] = set()

        for gguf_path in gguf_paths:
            reader = gguf.GGUFReader(gguf_path)
            for tensor in reader.tensors:
                if _is_quantized_type(tensor.tensor_type):
                    # Lazy: keep raw quantized bytes as a QParam; dequant happens
                    # per-forward inside GgufLinear. No full fp32 materialized here.
                    tensor_t: torch.Tensor = QParam(tensor.data, tensor.tensor_type, name=tensor.name)
                else:
                    # Non-quantized (F32/F16/BF16): small support tensors (norms,
                    # biases, scale_shift). Dequantize eagerly to a normal tensor.
                    try:
                        array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                    except NotImplementedError as exc:
                        raise RuntimeError(
                            f"GGUF tensor '{tensor.name}' uses unsupported quant type "
                            f"{tensor.tensor_type.name}; cannot dequantize"
                        ) from exc
                    arr = np.ascontiguousarray(array)
                    if not arr.flags.writeable:
                        arr = arr.copy()
                    tensor_t = torch.from_numpy(arr).to(device=target_device)
                    # ponytail: GGUF support tensors (norms/biases/scale_shift)
                    # are coerced to bf16 to match DistilledPipeline bf16
                    # activations; make dtype configurable only if another pipeline
                    # requires a different activation dtype.
                    if tensor_t.is_floating_point():
                        tensor_t = tensor_t.to(dtype=torch.bfloat16)
                for key, value in _apply_sd_ops(tensor.name, tensor_t, sd_ops):
                    if isinstance(value, QParam):
                        size += value.quantized_nbytes
                        dtype.add(torch.float32)
                    else:
                        size += value.nbytes
                        dtype.add(value.dtype)
                    sd[key] = value

        # ponytail: no-silent-garbage guard. A native-named LTX GGUF carries no
        # Comfy `model.diffusion_model.` prefix, so an sd_ops renaming map built
        # for Comfy safetensors filters every tensor; without this guard the
        # pipeline would load an empty transformer and emit garbage silently.
        if not sd:
            raise RuntimeError(
                f"GGUF loader produced an empty state dict from {gguf_paths}; "
                "tensor names did not survive sd_ops filtering or the checkpoint has no tensors"
            )

        return StateDict(sd=sd, device=target_device, size=size, dtype=dtype)


class GgufNativeSDOps:
    """Identity SDOps: native/diffusers-style GGUF tensor names need no remapping.

    Real QuantStack LTX 2.3 GGUF checkpoints already carry native/diffusers-style
    tensor names (``transformer_blocks``, ``adaln_single``, connectors) with no
    Comfy ``model.diffusion_model.`` prefix. Applying the Comfy renaming map
    filters every tensor, so this duck-types ``ltx_core.loader.sd_ops.SDOps`` and
    passes keys/values through unchanged.
    """

    name = "gguf_native"

    def apply_to_key(self, key: str) -> str:
        return key

    def apply_to_key_value(self, key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return [KeyValueOperationResult(key, value)]


def install_gguf_loader(pipeline: object) -> None:
    """Install GGUF-native loader, sd_ops, and the lazy dequant module op on a DistilledPipeline.

    Replaces the builder's ``model_loader`` with :class:`GgufStateDictLoader`,
    ``model_sd_ops`` with :class:`GgufNativeSDOps`, and adds
    :data:`GGUF_DEQUANT_LINEAR_OP` to ``module_ops`` so every transformer
    ``nn.Linear`` dequantizes its QParam weights lazily in forward. Idempotent
    once all three are in place; repairs a partially-installed builder (e.g. a
    GGUF loader/native sd_ops missing the module op). Raises ``RuntimeError``
    if the pipeline shape does not match expectations.
    """
    stage = getattr(pipeline, "stage", None)
    if stage is None:
        raise RuntimeError("install_gguf_loader: pipeline has no .stage (expected DistilledPipeline)")
    builder = getattr(stage, "_transformer_builder", None)
    if builder is None:
        raise RuntimeError("install_gguf_loader: stage has no _transformer_builder; Lightricks API changed")
    has_gguf_op = any(op.name == GGUF_DEQUANT_LINEAR_OP.name for op in builder.module_ops)
    already_installed = (
        isinstance(builder.model_loader, GgufStateDictLoader)
        and isinstance(builder.model_sd_ops, GgufNativeSDOps)
        and has_gguf_op
    )
    if already_installed:
        return
    # Preserve any existing module ops, dropping a stale GGUF op (no duplicates).
    module_ops = tuple(op for op in builder.module_ops if op.name != GGUF_DEQUANT_LINEAR_OP.name)
    module_ops = (*module_ops, GGUF_DEQUANT_LINEAR_OP)
    stage._transformer_builder = replace(  # type: ignore[arg-type]
        builder,
        model_loader=GgufStateDictLoader(),
        model_sd_ops=GgufNativeSDOps(),
        module_ops=module_ops,
    )


# --- helpers ---


def _is_gguf_path(path: object) -> bool:
    return str(path).lower().endswith(".gguf")


def _coerce_config_text(value: Any) -> str | None:
    """Normalize a GGUF field value (str/bytes/single-element list) to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_config_text(value[0])
    return None


def _apply_sd_ops(name: str, value: torch.Tensor, sd_ops: object | None) -> list[tuple[str, torch.Tensor]]:
    if sd_ops is None:
        return [(name, value)]
    expected_name = sd_ops.apply_to_key(name)  # type: ignore[attr-defined]
    if expected_name is None:
        return []
    return [(r.new_key, r.new_value) for r in sd_ops.apply_to_key_value(expected_name, value)]  # type: ignore[attr-defined]
