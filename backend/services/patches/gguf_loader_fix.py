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

from ltx_core.loader.primitives import StateDict
from ltx_core.loader.sd_ops import KeyValueOperationResult

# Metadata keys searched, in priority order, for the embedded JSON config.
_CONFIG_KEYS = ("config", "ltx.config", "general.config", "ltx_config")


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
                try:
                    array = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
                except NotImplementedError as exc:
                    raise RuntimeError(
                        f"GGUF tensor '{tensor.name}' uses unsupported quant type "
                        f"{tensor.tensor_type.name}; cannot dequantize"
                    ) from exc
                # dequantize yields the final element-shaped array; ensure a
                # writable, contiguous copy (F32 stays a read-only view into the
                # memmap) so torch.from_numpy owns its storage.
                arr = np.ascontiguousarray(array)
                if not arr.flags.writeable:
                    arr = arr.copy()
                tensor_t = torch.from_numpy(arr).to(device=target_device)
                for key, value in _apply_sd_ops(tensor.name, tensor_t, sd_ops):
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
    """Replace a ``DistilledPipeline.stage`` transformer builder's loader + sd_ops with GGUF-native ones.

    Installs both :class:`GgufStateDictLoader` and :class:`GgufNativeSDOps` so the
    builder no longer applies the Comfy renaming map (which filters every native
    GGUF tensor). Idempotent once both are in place; if a GGUF loader is already
    present with the wrong sd_ops, the sd_ops is replaced to repair it.
    Raises ``RuntimeError`` if the pipeline shape does not match expectations.
    """
    stage = getattr(pipeline, "stage", None)
    if stage is None:
        raise RuntimeError("install_gguf_loader: pipeline has no .stage (expected DistilledPipeline)")
    builder = getattr(stage, "_transformer_builder", None)
    if builder is None:
        raise RuntimeError("install_gguf_loader: stage has no _transformer_builder; Lightricks API changed")
    if isinstance(builder.model_loader, GgufStateDictLoader) and isinstance(
        builder.model_sd_ops, GgufNativeSDOps
    ):
        return
    stage._transformer_builder = replace(  # type: ignore[arg-type]
        builder, model_loader=GgufStateDictLoader(), model_sd_ops=GgufNativeSDOps()
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
