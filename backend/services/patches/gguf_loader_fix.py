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

import inspect
import json
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
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessor
from ltx_core.text_encoders.gemma.encoders.base_encoder import GemmaTextEncoder

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

    def detach(self) -> "QParam":
        # load_state_dict() detaches state_dict tensors before child modules see
        # them. Preserve attrs so GgufLinear._load_from_state_dict can intercept.
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

GGUF_EMBEDDINGS_DEQUANT_LINEAR_OP = ModuleOps(
    name="gguf_embeddings_dequant_linear",
    matcher=lambda model: isinstance(model, EmbeddingsProcessor),
    mutator=_amend_forward_with_gguf,
)


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
        allow_safetensors_only: bool = False,
    ) -> None:
        self._require_transformer_config = require_transformer_config
        self._include_safetensors = include_safetensors
        self._lazy_quantized = lazy_quantized
        self._allow_safetensors_only = allow_safetensors_only

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

        target_device = torch.device("cpu")
        # ponytail: state-dict registry is long-lived RAM cache; GPU residency
        # belongs to model contexts only (builder moves model to GPU later).
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

    from ltx_core.loader.registry import DummyRegistry
    from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
    from ltx_core.text_encoders.gemma import (
        EMBEDDINGS_PROCESSOR_KEY_OPS,
        GEMMA_MODEL_OPS,
        EmbeddingsProcessorConfigurator,
        GemmaTextEncoderConfigurator,
        module_ops_from_gemma_root,
    )
    from ltx_pipelines.utils import blocks

    if getattr(blocks.PromptEncoder.__init__, "_ltx_desktop_gguf_patch", False):
        return

    _install_gemma_encode_patch()
    original_init = blocks.PromptEncoder.__init__

    def patched_init(
        self: object,
        checkpoint_path: str,
        gemma_root: str,
        dtype: torch.dtype,
        device: torch.device,
        registry: object | None = None,
    ) -> None:
        gguf_path = _find_gemma_gguf(gemma_root)
        if gguf_path is None:
            original_init(self, checkpoint_path, gemma_root, dtype, device, registry)
            return

        self._dtype = dtype
        self._device = device
        module_ops = module_ops_from_gemma_root(gemma_root)
        registry_obj = registry or DummyRegistry()
        self._text_encoder_builder = Builder(
            model_path=str(gguf_path),
            model_class_configurator=GemmaTextEncoderConfigurator,
            model_sd_ops=GgufGemmaSDOps(),
            # ponytail: Gemma GGUF is eagerly bf16-dequanted on CPU so existing
            # layer streaming can page dense layers; revisit lazy Gemma after
            # QParam+LayerStreaming support exists.
            model_loader=GgufStateDictLoader(require_transformer_config=False, lazy_quantized=False),
            module_ops=(GEMMA_MODEL_OPS, *module_ops),
            registry=registry_obj,
        )
        self._embeddings_processor_builder = Builder(
            model_path=checkpoint_path,
            model_class_configurator=EmbeddingsProcessorConfigurator,
            model_sd_ops=GgufEmbeddingsProcessorSDOps(),
            model_loader=GgufStateDictLoader(include_safetensors=True),
            module_ops=(GGUF_EMBEDDINGS_DEQUANT_LINEAR_OP,),
            registry=registry_obj,
        )

    patched_init._ltx_desktop_gguf_patch = True  # type: ignore[attr-defined]
    blocks.PromptEncoder.__init__ = patched_init


def _install_gemma_encode_patch() -> None:
    if getattr(GemmaTextEncoder.encode, "_ltx_desktop_gguf_patch", False):
        return

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
        closure_images = inspect.getclosurevars(fn).nonlocals.get("images") if callable(fn) else None
        if closure_images == []:
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
    """Route non-transformer GGUF profile builders to their component files.

    Only patches components that exist on the pipeline — allows the same helper
    to work for T2V, A2V, IC-LoRA, and retake without wrapper changes.

    Parameters
    ----------
    pipeline
        The distilled pipeline instance.
    checkpoint_path
        GGUF profile paths (tuple of paths).
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
        raise RuntimeError("GGUF profile missing video VAE safetensors path")
    if has_audio_components and audio_vae is None:
        raise RuntimeError("GGUF profile missing audio VAE safetensors path")

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
        stage._transformer_builder = replace(  # type: ignore[arg-type]
            builder,
            model_loader=GgufStateDictLoader(allow_safetensors_only=True),
            model_sd_ops=GgufNativeSDOps(),
            module_ops=module_ops,
        )
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
        raise RuntimeError(f"GGUF component path patch failed: missing {owner}.{attr}")
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


def _find_gemma_gguf(gemma_root: str) -> Path | None:
    root = Path(gemma_root)
    if root.is_file() and root.suffix.lower() == ".gguf":
        return root
    if not root.is_dir():
        return None
    candidates = sorted(p for p in root.glob("*.gguf") if "mmproj" not in p.name.lower())
    return candidates[0] if candidates else None


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
