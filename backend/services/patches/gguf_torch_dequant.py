"""Torch-device GGUF dequant for LTX GGUF quantized qtypes only.

Supports Q4_K, Q5_K, Q6_K. Returns None for all other qtypes.
Replaces slow per-forward CPU numpy dequant in the future (not wired yet).

Qtype math adapted from ComfyUI-GGUF (Apache-2.0) dequant.py formulas.
"""

from __future__ import annotations

import gguf
import torch

QK_K = 256
K_SCALE_SIZE = 12


def _split_block_dims(blocks: torch.Tensor, *args: int) -> tuple[torch.Tensor, ...]:
    """Split blocks along dim=1 at cumulative positions, remainder auto-computed."""
    n_max = blocks.shape[1]
    dims = list(args) + [n_max - sum(args)]
    return torch.split(blocks, dims, dim=1)


def _get_scale_min(scales: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack interleaved scale/min 6-bit values from 12-byte scale block."""
    n_blocks = scales.shape[0]
    scales = scales.view(torch.uint8)
    scales = scales.reshape((n_blocks, 3, 4))
    d, m, m_d = torch.split(scales, scales.shape[-2] // 3, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    min = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n_blocks, 8)), min.reshape((n_blocks, 8))


def _dequantize_blocks_Q4_K(
    blocks: torch.Tensor,
    _block_size: int,
    _type_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    qs = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qs = (qs & 0x0F).reshape((n_blocks, -1, 32))
    return (d * qs - dm).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q5_K(
    blocks: torch.Tensor,
    _block_size: int,
    _type_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    d, dmin, scales, qh, qs = _split_block_dims(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)
    sc, m = _get_scale_min(scales)
    d = (d * sc).reshape((n_blocks, -1, 1))
    dm = (dmin * m).reshape((n_blocks, -1, 1))
    ql = qs.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [i for i in range(8)], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 8, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = (qh & 0x01).reshape((n_blocks, -1, 32))
    q = ql | (qh << 4)
    return (d * q - dm).reshape((n_blocks, QK_K))


def _dequantize_blocks_Q6_K(
    blocks: torch.Tensor,
    _block_size: int,
    _type_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    n_blocks = blocks.shape[0]
    ql, qh, scales, d = _split_block_dims(blocks, QK_K // 2, QK_K // 4, QK_K // 16)
    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n_blocks, QK_K // 16, 1))
    ql = ql.reshape((n_blocks, -1, 1, 64)) >> torch.tensor(
        [0, 4], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 2, 1))
    ql = (ql & 0x0F).reshape((n_blocks, -1, 32))
    qh = qh.reshape((n_blocks, -1, 1, 32)) >> torch.tensor(
        [0, 2, 4, 6], device=d.device, dtype=torch.uint8
    ).reshape((1, 1, 4, 1))
    qh = (qh & 0x03).reshape((n_blocks, -1, 32))
    q = (ql | (qh << 4)).to(torch.int8) - 32
    q = q.reshape((n_blocks, QK_K // 16, -1))
    return (d * q).reshape((n_blocks, QK_K))


_DEQUANTIZE_DISPATCH: dict[object, object] = {
    gguf.GGMLQuantizationType.Q4_K: _dequantize_blocks_Q4_K,
    gguf.GGMLQuantizationType.Q5_K: _dequantize_blocks_Q5_K,
    gguf.GGMLQuantizationType.Q6_K: _dequantize_blocks_Q6_K,
}


def dequantize_gguf_tensor_torch(
    raw: torch.Tensor,
    tensor_type: object,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Dequantize a raw GGUF quantized tensor on ``device`` using torch ops.

    Supports only Q4_K / Q5_K / Q6_K. Returns ``None`` for all other qtypes.
    Final values match ``gguf.quants.dequantize(raw.numpy(), tensor_type)``
    for the supported qtypes.
    """
    dequant_fn = _DEQUANTIZE_DISPATCH.get(tensor_type)
    if dequant_fn is None:
        return None

    block_size, type_size = gguf.GGML_QUANT_SIZES[tensor_type]  # type: ignore[index]
    target_dtype = dtype if dtype.is_floating_point else torch.float32

    rows = raw.to(device).reshape((-1, raw.shape[-1])).contiguous()

    rows = rows.view(torch.uint8)
    n_blocks = rows.numel() // type_size
    blocks = rows.reshape((n_blocks, type_size))
    result = dequant_fn(blocks, block_size, type_size, target_dtype)
    out_shape = gguf.quants.quant_shape_from_byte_shape(raw.shape, tensor_type)
    return result.reshape(out_shape).to(device=device)
