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

import functools
import gc
import inspect
import json
import logging
import os
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.model.transformer.model import LTXModel
from ltx_core.quantization import QuantizationPolicy
from collections.abc import Callable

from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder

logger = logging.getLogger(__name__)

# Metadata keys searched, in priority order, for the embedded JSON config.
_CONFIG_KEYS = ("config", "ltx.config", "general.config", "ltx_config")
_V2_EMBEDDINGS_CONFIG_KEYS = frozenset(
    (
        "caption_proj_before_connector",
        "caption_projection_first_linear",
        "caption_proj_input_norm",
        "caption_projection_second_linear",
    )
)

# GGUF tensor types that store unquantized fp values; loaded eagerly as normal
# tensors (small support tensors: norms, biases, scale_shift). Everything else
# is quantized (Q4_K/Q5_K/Q6_K/...) and kept lazy via :class:`QParam`.
_NON_QUANTIZED_TYPE_NAMES = frozenset({"F32", "F16", "BF16"})


def _is_quantized_type(tensor_type: object) -> bool:
    """True for GGUF quantized types (Q4_K, Q5_K, Q6_K, IQ4_XS, ...), False for F32/F16/BF16."""
    name = getattr(tensor_type, "name", str(tensor_type))
    return name not in _NON_QUANTIZED_TYPE_NAMES


# Gemma GGUF Linear weight suffixes — only these get lazy QParam treatment.
# All other Gemma tensors (norms, embeddings) use small non-quantized types
# and dequant eagerly regardless, but filter protects against hypothetical
# quantized norm types that would fail QParam shape mismatch.
_GEMMA_LINEAR_WHITELIST: frozenset[str] = frozenset({
    "attn_q.weight",
    "attn_k.weight",
    "attn_v.weight",
    "attn_output.weight",
    "ffn_gate.weight",
    "ffn_up.weight",
    "ffn_down.weight",
})


def _is_gemma_linear_name(name: str) -> bool:
    """True if `name` is a Gemma transformer Linear weight (blk.N.<suffix>)."""
    match = re.match(r"^blk\.\d+\.(.+)$", name)
    return match is not None and match.group(1) in _GEMMA_LINEAR_WHITELIST


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
        cls, raw: "np.ndarray | torch.Tensor", tensor_type: object, *,
        name: str, device: torch.device | None = None,
    ) -> "QParam":
        placeholder = torch.empty(0, dtype=torch.float32)
        obj = super().__new__(cls, placeholder, requires_grad=False)  # type: ignore[call-overload]
        if isinstance(raw, np.ndarray):
            # Copy (decoupled from the GGUFReader memmap, which is released after load).
            obj._raw = torch.from_numpy(np.ascontiguousarray(raw).copy())
        else:
            obj._raw = raw.contiguous().clone()
        if device is not None:
            obj._raw = obj._raw.to(device=device)
        obj._tensor_type = tensor_type
        obj._gguf_name = name
        return obj

    @property
    def gguf_name(self) -> str:
        return self._gguf_name

    @property
    def quantized_nbytes(self) -> int:
        return int(self._raw.nbytes)

    def detach(self) -> "QParam":
        # load_state_dict() detaches state_dict tensors before child modules see
        # them. Preserve attrs so GgufLinear._load_from_state_dict can intercept.
        return self

    def to(self, *args: object, **kwargs: object) -> "QParam":
        # SingleGPUModelBuilder.build converts state-dict dtypes via
        # ``value.to(dtype=dtype)`` before ``load_state_dict``. The QParam's
        # placeholder data (``empty(0)``) is never used in computation — the
        # real weight is dequantized in GgufLinear.forward. Returning self
        # preserves the QParam subclass, gguf_name, _raw, and _tensor_type
        # through dtype/device conversion.
        # ponytail: raw quantized bytes residency chosen by loader target_device;
        # add device forwarding to _raw if model parallelism requires it.
        return self

    def dequant(self, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        import gguf

        if device.type == "cuda":
            from services.patches.gguf_torch_dequant import dequantize_gguf_tensor_torch

            tensor = dequantize_gguf_tensor_torch(
                self._raw, self._tensor_type, device=device, dtype=dtype
            )
            if tensor is not None:
                return tensor

        # ponytail: gguf.quants.dequantize needs CPU numpy; copy CUDA raw to CPU first
        raw_dev = self._raw.device
        raw_for_numpy = self._raw if raw_dev.type == "cpu" else self._raw.cpu()
        try:
            array = gguf.quants.dequantize(raw_for_numpy.numpy(), self._tensor_type)
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
        result = torch.nn.functional.linear(input, weight, bias)

        # ponytail: explicit dealloc after dequant; cache cleanup gated by
        # env (off by default — Comfy-like: dequantized weight is temporary
        # per forward, quantized blocks live on GPU, no per-layer empty_cache).
        # Set LTX_GGUF_EMPTY_CACHE_EACH_FORWARD=1 to restore old aggressive
        # clearing that prevented CUDA allocator fragmentation during ~432
        # dequant cycles in Gemma encode (at higher per-forward cost).
        had_qparam = isinstance(self.weight, QParam) or isinstance(self.bias, QParam)
        if isinstance(self.weight, QParam):
            del weight
        if isinstance(self.bias, QParam):
            del bias
        if had_qparam and input.device.type == "cuda" and os.environ.get("LTX_GGUF_EMPTY_CACHE_EACH_FORWARD") == "1":
            torch.cuda.empty_cache()

        # Runtime LoRA delta (IC-LoRA / multi-LoRA).
        compute_dtype = input.dtype if input.is_floating_point() else torch.float32
        for lora_A, lora_B, strength in getattr(self, "lora_pairs", ()):
            a = lora_A.to(device=input.device, dtype=compute_dtype)
            b = lora_B.to(device=input.device, dtype=compute_dtype)
            result = result + torch.nn.functional.linear(torch.nn.functional.linear(input, a), b) * strength
        return result


def _amend_forward_with_gguf(model: torch.nn.Module) -> torch.nn.Module:
    """Swap every ``nn.Linear`` to :class:`GgufLinear` in place."""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.__class__ = GgufLinear
    return model


# ── Runtime LoRA globals (GGUF) ──────────────────────────────────────────
_GGUF_BUILD_PATCHED: bool = False


GGUF_DEQUANT_LINEAR_OP = ModuleOps(
    name="gguf_dequant_linear",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=_amend_forward_with_gguf,
)

GGUF_EMBEDDINGS_DEQUANT_LINEAR_OP = ModuleOps(
    name="gguf_embeddings_dequant_linear",
    matcher=lambda model: isinstance(model, EmbeddingsProcessor),
    mutator=_amend_forward_with_gguf,
)


def _strip_gemma_vision_tower(module: GemmaTextEncoder) -> GemmaTextEncoder:
    # ponytail: LTX uses Gemma text hidden states only; strip unloaded vision tower to avoid meta tensors.
    module.model.model.vision_tower = torch.nn.Identity()
    module.model.model.multi_modal_projector = torch.nn.Identity()
    return module


GGUF_GEMMA_TEXT_ONLY_OP = ModuleOps(
    name="gguf_gemma_text_only",
    matcher=lambda model: isinstance(model, GemmaTextEncoder),
    mutator=_strip_gemma_vision_tower,
)

GGUF_GEMMA_DEQUANT_LINEAR_OP = ModuleOps(
    name="gguf_gemma_dequant_linear",
    matcher=lambda model: isinstance(model, GemmaTextEncoder),
    mutator=_amend_forward_with_gguf,
)


def _llama_cpp_prompt_enhancer_op(model_path: Path) -> ModuleOps:
    def attach(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module._ltx_desktop_llama_cpp_model_path = str(model_path)  # type: ignore[attr-defined]
        return module

    return ModuleOps(
        name="llama_cpp_prompt_enhancer",
        matcher=lambda model: isinstance(model, GemmaTextEncoder),
        mutator=attach,
    )


def _install_llama_cpp_enhance_patch() -> None:
    if getattr(GemmaTextEncoder._enhance, "_ltx_desktop_llama_cpp_patch", False):
        return

    original_enhance = GemmaTextEncoder._enhance

    def patched_enhance(
        self: GemmaTextEncoder,
        messages: list[dict[str, Any]],
        image: torch.Tensor | None = None,
        max_new_tokens: int = 512,
        seed: int = 10,
    ) -> str:
        model_path = getattr(self, "_ltx_desktop_llama_cpp_model_path", None)
        if model_path is None or image is not None:
            return original_enhance(self, messages, image=image, max_new_tokens=max_new_tokens, seed=seed)
        try:
            from llama_cpp import Llama, llama_supports_gpu_offload
        except ImportError as exc:
            raise RuntimeError(
                "GGUF prompt enhancement requires llama-cpp-python built with CUDA support"
            ) from exc
        if callable(llama_supports_gpu_offload) and not llama_supports_gpu_offload():
            raise RuntimeError("llama-cpp-python was installed without GPU offload support")

        llm: Any | None = None
        try:
            llm = Llama(
                model_path=str(model_path),
                n_gpu_layers=-1,
                n_ctx=8192,
                n_batch=512,
                seed=seed,
                verbose=False,
            )
            result = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=0.7,
            )
            choice = result["choices"][0]
            content = choice.get("message", {}).get("content") or choice.get("text", "")
            return str(content).strip()
        finally:
            if llm is not None:
                close = getattr(llm, "close", None)
                if callable(close):
                    close()
                del llm
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    patched_enhance._ltx_desktop_llama_cpp_patch = True  # type: ignore[attr-defined]
    GemmaTextEncoder._enhance = patched_enhance  # type: ignore[method-assign]


# --------------------------------------------------------------------------
# GGUF llama.cpp standalone enhancement helper (runs before PyTorch Gemma build)
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=2)
def _load_gemma_t2v_system_prompt() -> str:
    """Read gemma_t2v_system_prompt.txt from installed ltx_core package."""
    import ltx_core.text_encoders.gemma.encoders.base_encoder as _be
    return (Path(_be.__file__).parent / "prompts" / "gemma_t2v_system_prompt.txt").read_text()


def _enhance_prompt_with_llama_cpp(
    model_path: str,
    prompt: str,
    max_new_tokens: int = 512,
    seed: int = 10,
) -> str:
    """Run llama.cpp prompt enhancement standalone, free GPU memory, return enhanced text.

    Constructs the same system prompt + user message format as
    ``GemmaTextEncoder.enhance_t2v`` but uses llama.cpp directly so PyTorch
    Gemma is not loaded unnecessarily.
    """
    try:
        from llama_cpp import Llama, llama_supports_gpu_offload
    except ImportError as exc:
        raise RuntimeError(
            "GGUF prompt enhancement requires llama-cpp-python built with CUDA support"
        ) from exc
    if callable(llama_supports_gpu_offload) and not llama_supports_gpu_offload():
        raise RuntimeError("llama-cpp-python was installed without GPU offload support")

    system_prompt = _load_gemma_t2v_system_prompt()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"user prompt: {prompt}"},
    ]

    llm: Any | None = None
    try:
        llm = Llama(
            model_path=str(model_path),
            n_gpu_layers=-1,
            n_ctx=8192,
            n_batch=512,
            seed=seed,
            verbose=False,
        )
        result = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.7,
        )
        choice = result["choices"][0]
        content = choice.get("message", {}).get("content") or choice.get("text", "")
        return str(content).strip()
    finally:
        if llm is not None:
            close = getattr(llm, "close", None)
            if callable(close):
                close()
            del llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _install_llama_cpp_prompt_encoder_call_patch() -> None:
    """Patch ``PromptEncoder.__call__`` to run llama.cpp before PyTorch Gemma.

    When ``enhance_first_prompt=True`` and GGUF is configured:
      1. Run llama.cpp text enhancement standalone (regardless of whether an
         ``enhance_prompt_image`` was supplied).
      2. Free llama.cpp GPU memory.
      3. Call original ``__call__`` with ``enhance_first_prompt=False`` and
         ``enhance_prompt_image=None`` so the dense PyTorch
         ``GemmaTextEncoder._enhance`` never builds and image-conditioned
         enhancement cannot hit the GGUF-stripped Gemma vision tower.

    Non-GGUF calls pass through unchanged.
    """
    from ltx_pipelines.utils import blocks
    from ltx_pipelines.utils.helpers import clean_response

    if getattr(blocks.PromptEncoder.__call__, "_ltx_desktop_gguf_call_patch", False):
        return

    original_call = blocks.PromptEncoder.__call__

    def patched_call(
        self: object,
        prompts: list[str],
        *,
        enhance_first_prompt: bool = False,
        enhance_prompt_image: str | None = None,
        enhance_prompt_seed: int = 42,
        streaming_prefetch_count: int | None = None,
    ) -> list[object]:
        gguf_model_path = getattr(self, "_ltx_desktop_llama_cpp_model_path", None)
        if enhance_first_prompt and gguf_model_path is not None:
            # Standalone llama.cpp text enhancement before _text_encoder_ctx
            # builds PyTorch Gemma. We intentionally drop enhance_prompt_image
            # so upstream image-conditioned enhancement never runs against the
            # GGUF-stripped (Identity) Gemma vision tower.
            prompts = list(prompts)
            prompts[0] = clean_response(
                _enhance_prompt_with_llama_cpp(
                    gguf_model_path,
                    prompts[0],
                    max_new_tokens=512,
                    seed=enhance_prompt_seed,
                )
            )
            # Enter _text_encoder_ctx with enhance=False / image=None to prevent
            # PyTorch Gemma enhance (text and image-conditioned).
            return original_call(
                self,
                prompts,
                enhance_first_prompt=False,
                enhance_prompt_image=None,
                enhance_prompt_seed=enhance_prompt_seed,
                streaming_prefetch_count=streaming_prefetch_count,
            )
        return original_call(
            self,
            prompts,
            enhance_first_prompt=enhance_first_prompt,
            enhance_prompt_image=enhance_prompt_image,
            enhance_prompt_seed=enhance_prompt_seed,
            streaming_prefetch_count=streaming_prefetch_count,
        )

    patched_call._ltx_desktop_gguf_call_patch = True  # type: ignore[attr-defined]
    blocks.PromptEncoder.__call__ = patched_call  # type: ignore[method-assign]


class KijaiFp8ScaledLinear(torch.nn.Linear):
    """Linear for Kijai `fp8_input_scaled` weights.

    Kijai stores standard-layout FP8 weights plus `*.weight_scale`. Upstream
    Fp8CastLinear only upcasts raw FP8 and drops the scale, producing black video.
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
        scale_key = prefix + "weight_scale"
        if scale_key in state_dict:
            self.register_buffer("weight_scale", state_dict.pop(scale_key))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        weight = self.weight
        if weight.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            weight = weight.to(input.dtype)
            weight_scale = getattr(self, "weight_scale", None)
            if isinstance(weight_scale, torch.Tensor):
                weight = weight * weight_scale.to(device=input.device, dtype=input.dtype)
        bias = self.bias
        if isinstance(bias, torch.Tensor) and bias.is_floating_point():
            bias = bias.to(device=input.device, dtype=input.dtype)
        return torch.nn.functional.linear(input, weight, bias)


def _amend_forward_with_kijai_fp8_scaled(model: torch.nn.Module) -> torch.nn.Module:
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            module.__class__ = KijaiFp8ScaledLinear
    return model


KIJAI_FP8_SCALED_LINEAR_OP = ModuleOps(
    name="kijai_fp8_scaled_linear",
    matcher=lambda model: isinstance(model, LTXModel),
    mutator=_amend_forward_with_kijai_fp8_scaled,
)


def kijai_fp8_quantization_policy() -> QuantizationPolicy:
    return QuantizationPolicy(sd_ops=SDOps(name="identity"), module_ops=(KIJAI_FP8_SCALED_LINEAR_OP,))


class GgufStateDictLoader:
    """Loads an LTX transformer from a GGUF checkpoint.

    Duck-types ``StateDictLoader``. ``metadata`` reads the embedded config;
    ``load`` dequantizes tensors via ``gguf.quants`` and applies ``SDOps`` the
    same way the safetensors loader does.
    """

    def __init__(
        self,
        *,
        require_transformer_config: bool = True,
        include_safetensors: bool = False,
        lazy_quantized: bool = True,
        lazy_quantized_filter: Callable[[str], bool] | None = None,
        allow_safetensors_only: bool = False,
    ) -> None:
        self._require_transformer_config = require_transformer_config
        self._include_safetensors = include_safetensors
        self._lazy_quantized = lazy_quantized
        self._lazy_quantized_filter = lazy_quantized_filter
        self._allow_safetensors_only = allow_safetensors_only

    def metadata(self, path: str) -> dict[str, object]:
        if not str(path).lower().endswith(".gguf"):
            # Safetensors path: read config from header metadata.
            from safetensors import safe_open

            with safe_open(path, framework="pt", device="cpu") as f:
                meta = f.metadata() or {}
            for key in _CONFIG_KEYS:
                raw = meta.get(key)
                if raw is None:
                    continue
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, dict) and "transformer" in parsed:
                    return parsed
            if not self._require_transformer_config:
                return {}
            raise RuntimeError(
                f"Transformer config not found in safetensors metadata of {path}; "
                "expected config/ltx.config/general.config/ltx_config JSON with a transformer object"
            )

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

        if not self._require_transformer_config:
            return {}

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
        safetensors_paths = [p for p in model_paths if str(p).lower().endswith(".safetensors")]
        if not gguf_paths and not (
            (self._include_safetensors or self._allow_safetensors_only) and safetensors_paths
        ):
            raise RuntimeError("GGUF loader received no .gguf path; this is a pipeline wiring bug")

        # ponytail: Comfy-like behavior — quantized raw bytes can live on
        # active/offload device; dequantized weight remains temporary per forward.
        # Set LTX_GGUF_KEEP_RAW_ON_CPU=1 to force CPU residency unconditionally.
        if device is not None and os.environ.get("LTX_GGUF_KEEP_RAW_ON_CPU") != "1":
            target_device = torch.device(device)
        else:
            target_device = torch.device("cpu")
        sd: dict[str, torch.Tensor] = {}
        size = 0
        dtype: set[torch.dtype] = set()

        def add_value(name: str, tensor_t: torch.Tensor) -> None:
            nonlocal size
            for key, value in _apply_sd_ops(name, tensor_t, sd_ops):
                if isinstance(value, QParam):
                    size += value.quantized_nbytes
                    dtype.add(torch.float32)
                else:
                    size += value.nbytes
                    dtype.add(value.dtype)
                sd[key] = value

        for gguf_path in gguf_paths:
            reader = gguf.GGUFReader(gguf_path)
            for tensor in reader.tensors:
                if _is_quantized_type(tensor.tensor_type) and self._lazy_quantized:
                    if self._lazy_quantized_filter is None or self._lazy_quantized_filter(tensor.name):
                        # Lazy: keep raw quantized bytes as a QParam; dequant happens
                        # per-forward inside GgufLinear. Raw bytes placed on
                        # target_device (GPU for full-load, CPU for streaming/env override).
                        tensor_t: torch.Tensor = QParam(
                            tensor.data, tensor.tensor_type, name=tensor.name, device=target_device
                        )
                    else:
                        # Filter mismatch: eagerly dequant even though lazy mode is on.
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
                        if tensor_t.is_floating_point():
                            tensor_t = tensor_t.to(dtype=torch.bfloat16)
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
                add_value(tensor.name, tensor_t)

        if self._include_safetensors or (self._allow_safetensors_only and not gguf_paths):
            from safetensors import safe_open

            for safetensors_path in safetensors_paths:
                with safe_open(safetensors_path, framework="pt", device=str(target_device)) as handle:
                    for key in handle.keys():
                        tensor_t = handle.get_tensor(key)
                        if tensor_t.is_floating_point():
                            tensor_t = tensor_t.to(dtype=torch.bfloat16)
                        add_value(key, tensor_t)

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


class GgufGemmaSDOps:
    """Map llama.cpp Gemma3 GGUF names to HF Gemma3 state-dict names."""

    name = "gguf_gemma"
    _layer_re = re.compile(r"^blk\.(\d+)\.(.+)$")
    _layer_suffixes = {
        "attn_q.weight": "self_attn.q_proj.weight",
        "attn_k.weight": "self_attn.k_proj.weight",
        "attn_v.weight": "self_attn.v_proj.weight",
        "attn_output.weight": "self_attn.o_proj.weight",
        "attn_q_norm.weight": "self_attn.q_norm.weight",
        "attn_k_norm.weight": "self_attn.k_norm.weight",
        "ffn_gate.weight": "mlp.gate_proj.weight",
        "ffn_up.weight": "mlp.up_proj.weight",
        "ffn_down.weight": "mlp.down_proj.weight",
        "attn_norm.weight": "input_layernorm.weight",
        "post_attention_norm.weight": "post_attention_layernorm.weight",
        "ffn_norm.weight": "pre_feedforward_layernorm.weight",
        "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    }

    def apply_to_key(self, key: str) -> str | None:
        if key == "token_embd.weight":
            return "model.model.language_model.embed_tokens.weight"
        if key == "output_norm.weight":
            return "model.model.language_model.norm.weight"
        match = self._layer_re.match(key)
        if match is None:
            return None
        layer_index, suffix = match.groups()
        mapped_suffix = self._layer_suffixes.get(suffix)
        if mapped_suffix is None:
            return None
        return f"model.model.language_model.layers.{layer_index}.{mapped_suffix}"

    def apply_to_key_value(self, key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        # Gemma GGUF stores norm weights in llama.cpp form; HF Gemma expects weight - 1.
        # Matches transformers.modeling_gguf_pytorch_utils.Gemma2TensorProcessor.
        if "norm.weight" in key:
            value = value - 1
        results = [KeyValueOperationResult(key, value)]
        if key == "model.model.language_model.embed_tokens.weight":
            results.append(KeyValueOperationResult("model.lm_head.weight", value))
        return results


class GgufEmbeddingsProcessorSDOps:
    """Map LTX 2.3 GGUF connector + safetensors projection names."""

    name = "gguf_embeddings_processor"

    def apply_to_key(self, key: str) -> str | None:
        replacements = (
            ("text_embedding_projection.video_aggregate_embed.", "feature_extractor.video_aggregate_embed."),
            ("text_embedding_projection.audio_aggregate_embed.", "feature_extractor.audio_aggregate_embed."),
            ("video_embeddings_connector.", "video_connector."),
            ("audio_embeddings_connector.", "audio_connector."),
        )
        for prefix, replacement in replacements:
            if key.startswith(prefix):
                return replacement + key.removeprefix(prefix)
        return None

    def apply_to_key_value(self, key: str, value: torch.Tensor) -> list[KeyValueOperationResult]:
        return [KeyValueOperationResult(key, value)]


KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER = (
    SDOps("KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER")
    .with_matching(prefix="encoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("encoder.", "")
)
KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER = (
    SDOps("KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER")
    .with_matching(prefix="decoder.")
    .with_matching(prefix="per_channel_statistics.")
    .with_replacement("decoder.", "")
)


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


def install_gguf_prompt_encoder_patch() -> None:
    """Patch PromptEncoder so Gemma GGUF folders work.

    Upstream PromptEncoder requires ``model*.safetensors``. ComfyUI LTX 2.3 GGUF
    workflows use ``gemma-*.gguf`` plus tokenizer/processor files in the text
    encoder folder. This patch preserves upstream behavior for safetensors roots.
    """

    from ltx_core.loader import SafetensorsModelStateDictLoader
    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        GEMMA_MODEL_OPS,
        EmbeddingsProcessorConfigurator,
        GemmaTextEncoderConfigurator,
    )
    from ltx_pipelines.utils import blocks

    if getattr(blocks.PromptEncoder.__init__, "_ltx_desktop_gguf_patch", False):
        return

    _install_gemma_encode_patch()
    _install_llama_cpp_enhance_patch()
    original_init = blocks.PromptEncoder.__init__

    def patched_init(
        self: object,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: object | None = None,
    ) -> None:
        # ponytail: empty/None gemma_root = API mode, delegate to original instead of _find_gemma_gguf raising.
        if not gemma_root:
            original_init(self, checkpoint_path, gemma_root, dtype, device, registry)
            return
        gguf_path = _find_gemma_gguf(gemma_root)
        if gguf_path is None:
            logger.info("PromptEncoder GGUF patch fallback gemma_root=%s checkpoint=%s", gemma_root, checkpoint_path)
            original_init(self, checkpoint_path, gemma_root, dtype, device, registry)
            return

        logger.info("PromptEncoder GGUF init gemma=%s checkpoint=%s", gguf_path, checkpoint_path)
        self._dtype = dtype
        self._device = device
        module_ops = _module_ops_from_gemma_root_slow_processor(_resolve_gemma_tokenizer_root(gemma_root))
        registry_obj = registry or DummyRegistry()
        self._text_encoder_builder = Builder(
            model_path=str(gguf_path),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GgufGemmaSDOps(),
            # ponytail: lazy dequant Gemma Linear weights; norms/embeddings dequant eagerly.
            model_loader=GgufStateDictLoader(
                require_transformer_config=False,
                lazy_quantized=True,
                lazy_quantized_filter=_is_gemma_linear_name,
            ),
            module_ops=(GEMMA_MODEL_OPS, GGUF_GEMMA_TEXT_ONLY_OP, GGUF_GEMMA_DEQUANT_LINEAR_OP, _llama_cpp_prompt_enhancer_op(gguf_path), *module_ops),
            registry=registry_obj,
        )
        # Determine whether transformer checkpoint is GGUF or safetensors.
        # Kijai split-safetensors carry Comfy-style keys
        # (model.diffusion_model.video_embeddings_connector.*) that need the
        # upstream EMBEDDINGS_PROCESSOR_KEY_OPS renaming. All-GGUF profiles use
        # GgufEmbeddingsProcessorSDOps for native GGUF key names.
        _cp_paths = [checkpoint_path] if isinstance(checkpoint_path, (str, bytes)) else list(checkpoint_path or [])
        if any(str(p).lower().endswith(".gguf") for p in _cp_paths):
            # ponytail: GGUF transformer profile — embeddings processor uses
            # safetensors from checkpoint_path tuple with native GGUF key names.
            self._embeddings_processor_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=EmbeddingsProcessorConfigurator,
                model_sd_ops=GgufEmbeddingsProcessorSDOps(),
                model_loader=GgufStateDictLoader(include_safetensors=True),
                module_ops=(GGUF_EMBEDDINGS_DEQUANT_LINEAR_OP,),
                registry=registry_obj,
            )
        else:
            # Safetensors-only: Kijai split safetensors or official monolith.
            # Use upstream EMBEDDINGS_PROCESSOR_KEY_OPS for Comfy-style keys.
            loader = SafetensorsModelStateDictLoader()
            v2_config = _find_v2_embeddings_config(_cp_paths)
            if v2_config is not None:
                # ponytail: Kijai transformer shard has V1 metadata while the
                # text-projection shard has V2 weights+config. Feed builder the
                # V2 config; remove when upstream checkpoint metadata is fixed.
                class _V2ConfigLoader(SafetensorsModelStateDictLoader):
                    def metadata(self, path: str) -> dict:  # noqa: ARG002
                        return v2_config

                loader = _V2ConfigLoader()

            self._embeddings_processor_builder = Builder(
                model_path=checkpoint_path,
                model_class_configurator=EmbeddingsProcessorConfigurator,
                model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
                model_loader=loader,
                module_ops=(),
                registry=registry_obj,
            )

        # Store GGUF model path on PromptEncoder for standalone llama.cpp enhancement
        # before _text_encoder_ctx builds PyTorch Gemma.
        self._ltx_desktop_llama_cpp_model_path = str(gguf_path)  # type: ignore[attr-defined]

    patched_init._ltx_desktop_gguf_patch = True  # type: ignore[attr-defined]
    blocks.PromptEncoder.__init__ = patched_init

    # Patch PromptEncoder.__call__ to run llama.cpp enhancement standalone before PyTorch Gemma.
    _install_llama_cpp_prompt_encoder_call_patch()


def _install_gemma_encode_patch() -> None:
    if getattr(GemmaTextEncoder.encode, "_ltx_desktop_gguf_patch", False):
        return

    @torch.inference_mode()
    def patched_encode(
        self: GemmaTextEncoder,
        text: str,
        padding_side: str = "left",  # noqa: ARG001
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        token_pairs = self.tokenizer.tokenize_with_weights(text)["gemma"]
        language_model = self.model.model.language_model
        device = language_model.embed_tokens.weight.device
        if device.type == "meta":
            for param in language_model.parameters():
                if param.device.type != "meta":
                    device = param.device
                    break
        if device.type == "cpu" and torch.cuda.is_available():
            device = torch.device("cuda")
            language_model.to(device)  # ponytail: builder short-circuits on unused meta vision params, only language_model moves
        input_ids = torch.tensor([[t[0] for t in token_pairs]], device=device)
        attention_mask = torch.tensor([[w[1] for w in token_pairs]], device=device)
        outputs = self.model.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        del outputs
        return hidden_states, attention_mask

    patched_encode._ltx_desktop_gguf_patch = True  # type: ignore[attr-defined]
    GemmaTextEncoder.encode = patched_encode  # type: ignore[method-assign]


def install_gguf_t2v_conditioning_patch() -> None:
    from ltx_pipelines.utils import blocks

    if getattr(blocks.ImageConditioner.__call__, "_ltx_desktop_gguf_t2v_patch", False):
        return

    original_call = blocks.ImageConditioner.__call__

    def patched_call(self: object, fn: object) -> object:
        if callable(fn):
            closure = inspect.getclosurevars(fn).nonlocals
            closure_images = closure.get("images")
            closure_video_conditioning = closure.get("video_conditioning")
            if closure_images == [] and not closure_video_conditioning:
                return fn(None)
        return original_call(self, fn)

    patched_call._ltx_desktop_gguf_t2v_patch = True  # type: ignore[attr-defined]
    blocks.ImageConditioner.__call__ = patched_call


def install_gguf_component_paths(
    pipeline: object,
    checkpoint_path: object,
    *,
    video_vae_path: str | None = None,
    audio_vae_path: str | None = None,
) -> None:
    """Route non-transformer profile component builders to their safetensors files.

    ponytail: not GGUF-specific — works for any profile with explicit VAE paths.
    Name is historical; called for both GGUF and split safetensors profiles.

    Only patches components that exist on the pipeline — allows the same helper
    to work for T2V, A2V, IC-LoRA, and retake without wrapper changes.

    Parameters
    ----------
    pipeline
        The distilled pipeline instance.
    checkpoint_path
        Profile paths (tuple of paths).
    video_vae_path
        Explicit video VAE safetensors path. When given, skips heuristic
        filename matching. Default None = use heuristic.
    audio_vae_path
        Explicit audio VAE safetensors path. Same override behavior.
    """

    paths = [str(p) for p in checkpoint_path] if isinstance(checkpoint_path, (list, tuple)) else [str(checkpoint_path)]
    video_vae = video_vae_path or _pick_component_path(paths, ("video_vae", "video-vae", "video"), ".safetensors")
    audio_vae = audio_vae_path or _pick_component_path(paths, ("audio_vae", "audio-vae", "audio"), ".safetensors")

    image_conditioner = getattr(pipeline, "image_conditioner", None)
    upsampler = getattr(pipeline, "upsampler", None)
    video_decoder = getattr(pipeline, "video_decoder", None)
    audio_conditioner = getattr(pipeline, "audio_conditioner", None)
    audio_decoder = getattr(pipeline, "audio_decoder", None)

    has_video_components = image_conditioner is not None or upsampler is not None or video_decoder is not None
    has_audio_components = audio_conditioner is not None or audio_decoder is not None

    if has_video_components and video_vae is None:
        raise RuntimeError("Profile missing video VAE safetensors path")
    if has_audio_components and audio_vae is None:
        raise RuntimeError("Profile missing audio VAE safetensors path")

    if image_conditioner is not None and video_vae is not None:
        _replace_builder_model_path(image_conditioner, "_encoder_builder", video_vae, model_sd_ops=KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER)
    if upsampler is not None and video_vae is not None:
        _replace_builder_model_path(upsampler, "_encoder_builder", video_vae, model_sd_ops=KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER)
    if video_decoder is not None and video_vae is not None:
        _replace_builder_model_path(video_decoder, "_decoder_builder", video_vae, model_sd_ops=KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER)
    if audio_conditioner is not None and audio_vae is not None:
        _replace_builder_model_path(audio_conditioner, "_encoder_builder", audio_vae)
    if audio_decoder is not None and audio_vae is not None:
        _replace_builder_model_path(audio_decoder, "_decoder_builder", audio_vae)
        _replace_builder_model_path(audio_decoder, "_vocoder_builder", audio_vae)


# ── GGUF runtime LoRA helpers ─────────────────────────────────────────────


def _preload_gguf_loras(
    builder: object,
) -> tuple[tuple[dict[str, torch.Tensor], float], ...]:
    """Pre-load LoRA safetensors through the builder's GGUF loader.

    Returns ``((lora_sd_dict, strength), ...)`` with tensors on the builder's
    ``lora_load_device`` (typically CPU).
    """
    data: list[tuple[dict[str, torch.Tensor], float]] = []
    for lora in builder.loras:  # type: ignore[attr-defined]
        sd = builder.load_sd(  # type: ignore[attr-defined]
            [lora.path],
            sd_ops=lora.sd_ops,
            registry=builder.registry,  # type: ignore[attr-defined]
            device=builder.lora_load_device,  # type: ignore[attr-defined]
        )
        data.append((sd.sd, lora.strength))
    return tuple(data)


def _attach_gguf_loras_to_model(
    model: torch.nn.Module,
    lora_data: tuple[tuple[dict[str, torch.Tensor], float], ...],
) -> None:
    """Attach LoRA A/B weights to GgufLinear modules by matching module names.

    Sets ``module.lora_pairs`` to CPU-stored ``(A, B, strength)`` tuples.
    """
    name_to_module: dict[str, torch.nn.Module] = {
        name: mod for name, mod in model.named_modules() if isinstance(mod, GgufLinear)
    }
    module_pairs: dict[str, list[tuple[torch.Tensor, torch.Tensor, float]]] = {}
    for lora_sd, strength in lora_data:
        for key in lora_sd:
            if key.endswith(".lora_A.weight"):
                base = key[: -len(".lora_A.weight")]
                b_key = f"{base}.lora_B.weight"
                if base in name_to_module and b_key in lora_sd:
                    module_pairs.setdefault(base, []).append(
                        (lora_sd[key].contiguous().cpu(), lora_sd[b_key].contiguous().cpu(), strength)
                    )
    for base, pairs in module_pairs.items():
        name_to_module[base].lora_pairs = tuple(pairs)


def _patch_gguf_lora_build() -> None:
    """Patch SingleGPUModelBuilder.build to attach runtime LoRAs post-load.

    One-shot (idempotent via module-level flag). GGUF builders with LoRAs load
    adapters separately, clear them for the upstream build, then attach runtime
    LoRA pairs after ``load_state_dict``.
    """
    global _GGUF_BUILD_PATCHED  # noqa: PLW0602
    if _GGUF_BUILD_PATCHED:
        return
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder

    original = SingleGPUModelBuilder.build

    @functools.wraps(original)
    def _patched(self: object, *args: object, **kwargs: object) -> object:
        loras = getattr(self, "loras", ())
        is_gguf = isinstance(getattr(self, "model_loader", None), GgufStateDictLoader)
        if is_gguf and loras and any(l.strength for l in loras):
            lora_data = _preload_gguf_loras(self)
            model = original(replace(self, loras=()), *args, **kwargs)
            _attach_gguf_loras_to_model(model, lora_data)
            return model
        return original(self, *args, **kwargs)

    SingleGPUModelBuilder.build = _patched
    _GGUF_BUILD_PATCHED = True


def install_gguf_loader(pipeline: object) -> None:
    """Install GGUF-native loader, sd_ops, and the lazy dequant module op on all present stages.

    Patches every stage that exists (``stage``, ``stage_1``, ``stage_2``) — supports
    T2V, A2V, IC-LoRA, and retake without wrapper changes. Idempotent per stage;
    raises ``RuntimeError`` only if none of the expected stages carry a ``_transformer_builder``.
    """
    stage_names = ("stage", "stage_1", "stage_2")
    found = False
    for name in stage_names:
        stage = getattr(pipeline, name, None)
        if stage is None:
            continue
        builder = getattr(stage, "_transformer_builder", None)
        if builder is None:
            continue
        found = True
        has_gguf_op = any(op.name == GGUF_DEQUANT_LINEAR_OP.name for op in builder.module_ops)
        already_installed = (
            isinstance(builder.model_loader, GgufStateDictLoader)
            and isinstance(builder.model_sd_ops, GgufNativeSDOps)
            and has_gguf_op
        )
        if already_installed:
            continue
        # Preserve any existing module ops, dropping a stale GGUF op (no duplicates).
        module_ops = tuple(op for op in builder.module_ops if op.name != GGUF_DEQUANT_LINEAR_OP.name)
        module_ops = (*module_ops, GGUF_DEQUANT_LINEAR_OP)
        replaced = replace(
            builder,
            model_loader=GgufStateDictLoader(allow_safetensors_only=True),
            model_sd_ops=GgufNativeSDOps(),
            module_ops=module_ops,
        )

        if replaced.loras and any(l.strength for l in replaced.loras):  # type: ignore[union-attr]
            _patch_gguf_lora_build()

        stage._transformer_builder = replaced  # type: ignore[arg-type]
    if not found:
        raise RuntimeError(
            "install_gguf_loader: pipeline has no stage/stage_1/stage_2 "
            "with a _transformer_builder (expected DistilledPipeline)"
        )


# --- helpers ---


def _is_gguf_path(path: object) -> bool:
    return str(path).lower().endswith(".gguf")


def _replace_builder_model_path(owner: object, attr: str, path: str, *, model_sd_ops: object | None = None) -> None:
    builder = getattr(owner, attr, None)
    if builder is None:
        raise RuntimeError(f"Component path patch failed: missing {owner}.{attr}")
    if model_sd_ops is not None:
        setattr(owner, attr, replace(builder, model_path=path, model_sd_ops=model_sd_ops))
    else:
        setattr(owner, attr, replace(builder, model_path=path))


def _pick_component_path(paths: list[str], needles: tuple[str, ...], suffix: str) -> str | None:
    for path in paths:
        lower = Path(path).name.lower()
        if lower.endswith(suffix) and all(needle in lower for needle in needles[:1]) and any(
            needle in lower for needle in needles
        ):
            return path
    for path in paths:
        lower = Path(path).name.lower()
        if lower.endswith(suffix) and any(needle in lower for needle in needles):
            return path
    return None


def _find_v2_embeddings_config(paths: list[object]) -> dict[str, object] | None:
    """Scan safetensors metadata for V2 embeddings config.

    Prefers paths whose filename suggests text projection (``text_projection``,
    ``tp.safetensors``) but falls back to any safetensors path.

    ponytail: metadata-only scan, no tensor loads. Returns first V2-capable
    config. Add per-path scoring if multiple text_projection paths carry
    different V2 config values.
    """
    from safetensors import safe_open

    safetensors_paths = [str(p) for p in paths if str(p).lower().endswith(".safetensors")]

    def _scan(path_strs: list[str]) -> dict[str, object] | None:
        for sp in path_strs:
            meta: dict[str, str] = {}
            try:
                with safe_open(sp, framework="pt", device="cpu") as f:
                    meta = f.metadata() or {}
            except Exception:
                continue
            for key in _CONFIG_KEYS:
                raw = meta.get(key)
                if raw is None:
                    continue
                try:
                    parsed = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if isinstance(parsed, dict):
                    tr = parsed.get("transformer", {})
                    if isinstance(tr, dict) and tr.keys() & _V2_EMBEDDINGS_CONFIG_KEYS:
                        return parsed
        return None

    # Scan text_projection paths first.
    tp = [
        sp for sp in safetensors_paths
        if "text_projection" in Path(sp).name.lower() or Path(sp).name.lower() == "tp.safetensors"
    ]
    return _scan(tp) or _scan([sp for sp in safetensors_paths if sp not in tp])


def _find_gemma_gguf(gemma_root: str | None) -> Path | None:
    if not gemma_root:
        raise ValueError(
            "Gemma GGUF path is empty/None; the text encoder profile is missing a text_encoder_root path"
        )
    root = Path(gemma_root)
    if root.is_file() and root.suffix.lower() == ".gguf":
        return root
    if not root.is_dir():
        return None
    candidates = sorted(p for p in root.glob("*.gguf") if "mmproj" not in p.name.lower())
    return candidates[0] if candidates else None


def _resolve_gemma_tokenizer_root(gemma_root: str) -> str:
    """Return tokenizer root directory for module_ops_from_gemma_root.

    If gemma_root is a .gguf file, return its parent directory.
    If gemma_root is a directory, return it as-is.
    """
    root = Path(gemma_root)
    if root.is_file() and root.suffix.lower() == ".gguf":
        return str(root.parent)
    return gemma_root


def _module_ops_from_gemma_root_slow_processor(gemma_root: str) -> tuple[Any, ...]:
    """Tokenizer/processor ModuleOps mirroring upstream but with ``use_fast=False``.

    Upstream ``ltx_core.text_encoders.gemma.module_ops_from_gemma_root`` calls
    ``AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)``
    without an explicit ``use_fast``, so transformers emits a slow-processor
    fallback warning when the Gemma image processor has no fast implementation.
    We preserve the existing slow-processor behavior but pass ``use_fast=False``
    explicitly to opt in (silencing the warning). Tokenizer/processor semantics
    are otherwise identical to upstream.
    """
    from transformers import AutoImageProcessor, Gemma3Processor

    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder
    from ltx_core.text_encoders.gemma.tokenizer import LTXVGemmaTokenizer
    from ltx_core.utils import find_matching_file

    tokenizer_root = str(find_matching_file(gemma_root, "tokenizer.model").parent)
    processor_root = str(find_matching_file(gemma_root, "preprocessor_config.json").parent)

    def load_tokenizer(module: GemmaTextEncoder) -> GemmaTextEncoder:
        module.tokenizer = LTXVGemmaTokenizer(tokenizer_root, 1024)
        return module

    def load_processor(module: GemmaTextEncoder) -> GemmaTextEncoder:
        image_processor = AutoImageProcessor.from_pretrained(
            processor_root, local_files_only=True, use_fast=False
        )
        if not module.tokenizer:
            raise ValueError("Tokenizer model operation must be performed before processor model operation")
        module.processor = Gemma3Processor(image_processor=image_processor, tokenizer=module.tokenizer.tokenizer)
        return module

    tokenizer_load_ops = ModuleOps(
        "TokenizerLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.tokenizer is None,
        mutator=load_tokenizer,
    )
    processor_load_ops = ModuleOps(
        "ProcessorLoad",
        matcher=lambda module: isinstance(module, GemmaTextEncoder) and module.processor is None,
        mutator=load_processor,
    )
    return (tokenizer_load_ops, processor_load_ops)


def _coerce_config_text(value: Any) -> str | None:
    """Normalize a GGUF field value (str/bytes/single-element list) to text."""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _coerce_config_text(value[0])
    return None


def install_kijai_transformer_config_patch(
    pipeline: object,
    checkpoint_path: object,
) -> None:
    """Patch stage transformer builders to use V2 config from text_projection metadata.

    Kijai split-safetensors checkpoints carry the full V2 config (57 keys) in
    text_projection.safetensors metadata, but the transformer builder reads from
    the first shard (transformer.safetensors) which has a minimal 6-key config.
    This causes ``KeyError: 'caption_channels'`` in ``_build_caption_projections``.

    Idempotent for safetensors-only tuples. Skips GGUF paths and single-string
    (official monolith) checkpoint paths entirely.
    """
    paths = [str(p) for p in checkpoint_path] if isinstance(checkpoint_path, (list, tuple)) else []
    if not paths:
        return  # single string path = official monolith
    if any(str(p).lower().endswith(".gguf") for p in paths):
        return  # GGUF transformer path has its own loader
    if not any(str(p).lower().endswith(".safetensors") for p in paths):
        return

    v2_config = _find_v2_embeddings_config(paths)
    if v2_config is None:
        return  # no V2 config found in any shard

    from dataclasses import replace

    from ltx_core.loader.sft_loader import SafetensorsModelStateDictLoader

    class _V2TransformerConfigLoader(SafetensorsModelStateDictLoader):
        """Loader that returns V2 config from text_projection metadata instead of first shard."""

        def metadata(self, path: str) -> dict:  # noqa: ARG002
            return v2_config

    loader = _V2TransformerConfigLoader()
    stage_names = ("stage", "stage_1", "stage_2")
    found = False
    for name in stage_names:
        stage = getattr(pipeline, name, None)
        if stage is None:
            continue
        builder = getattr(stage, "_transformer_builder", None)
        if builder is None:
            continue
        found = True
        stage._transformer_builder = replace(builder, model_loader=loader)  # type: ignore[arg-type]
    if not found:
        import logging

        logging.getLogger(__name__).warning(
            "install_kijai_transformer_config_patch: no stage/stage_1/stage_2 "
            "with _transformer_builder found"
        )


def _apply_sd_ops(name: str, value: torch.Tensor, sd_ops: object | None) -> list[tuple[str, torch.Tensor]]:
    if sd_ops is None:
        return [(name, value)]
    expected_name = sd_ops.apply_to_key(name)  # type: ignore[attr-defined]
    if expected_name is None:
        return []
    return [(r.new_key, r.new_value) for r in sd_ops.apply_to_key_value(expected_name, value)]  # type: ignore[attr-defined]
