"""Official LTX 2.3 IC-LoRA inpaint utilities.

Implements the exact two-stage inpaint workflow from
`LTX-2.3_ICLoRA_Inpaint_Two_Stage_Distilled.json`:

  Preprocess (green composite + mask dilation)
  → Stage 1 denoising (half res, full sigma schedule)
  → Decode → Laplacian blend with green guide
  → Resize 2× → VAE encode tiled
  → Stage 2 denoising (full res, 3-step sigma schedule)
  → Decode → Laplacian blend with green guide
  → Output

Laplacian pyramid uses kornia 0.8.3 matching ComfyUI-LTXVideo pyramid_blending.py.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from kornia.geometry.transform.pyramid import (
    PyrUp,
    build_laplacian_pyramid,
    build_pyramid,
    find_next_powerof_two,
    is_powerof_two,
)


# ---------------------------------------------------------------------------
# Green composite preprocessing (LTXVInpaintPreprocess equivalent)
# ---------------------------------------------------------------------------

BG_COLOR_RGB = (102, 255, 0)  # #66FF00
_bg_tensor: torch.Tensor | None = None


def _get_bg(device: torch.device) -> torch.Tensor:
    global _bg_tensor
    if _bg_tensor is None or _bg_tensor.device != device:
        _bg_tensor = torch.tensor(BG_COLOR_RGB, dtype=torch.float32, device=device).div_(255.0)
    return _bg_tensor


def green_composite_preprocess(
    images: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Composite images onto #66FF00 background where mask is active.

    Args:
        images: (B, C, F, H, W) tensor in [-1, 1] range (ltx_video_preprocess output).
        mask: (F, H, W) or (1, F, H, W) binary mask in [0, 1]. White = inpaint region.
              Single-frame masks are broadcast to video length.
    Returns:
        (B, C, F, H, W) green composite, same range as input.
    """
    _, _, f, _, _ = images.shape
    # Normalise [-1, 1] → [0, 1] for compositing
    images_01 = (images + 1.0) / 2.0  # (B, C, F, H, W)

    if mask.ndim == 4:
        mask = mask[:, :, :, 0]  # (1, F, H, W) → (1, F, H, W) still has 1 channel
    # Ensure mask is (F, H, W) or broadcastable
    if mask.ndim == 3:
        if mask.shape[0] == 1 and f > 1:
            mask = mask.expand(f, -1, -1)
        if mask.shape[0] != f:
            # Trim to shortest
            min_f = min(mask.shape[0], f)
            mask = mask[:min_f]
            images_01 = images_01[:, :, :min_f]
            f = min_f

    bg = _get_bg(images.device)  # (3,)

    # mask shape: (F, H, W) → (1, 1, F, H, W) for broadcasting against (B, C, F, H, W)
    mask_5d = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, F, H, W)
    bg_543 = bg.view(1, 3, 1, 1, 1)  # (1, C, 1, 1, 1)

    result_01 = images_01 * (1.0 - mask_5d) + bg_543 * mask_5d
    # Normalise back to [-1, 1]
    return result_01 * 2.0 - 1.0


# ---------------------------------------------------------------------------
# Mask dilation (LTXVDilateVideoMask equivalent)
# ---------------------------------------------------------------------------


def dilate_video_mask(
    mask: torch.Tensor,
    spatial_radius: int = 5,
    temporal_radius: int = 0,
) -> torch.Tensor:
    """Dilate a video mask spatially and/or temporally using separable max-pooling.

    Args:
        mask: (F, H, W) in [0, 1]. Single-frame (1, H, W) is accepted.
        spatial_radius: Half-size of 2D spatial kernel (kernel = 2*radius+1).
        temporal_radius: Half-size of 1D temporal kernel.
    Returns:
        Thresholded binary mask (F, H, W) with values in {0.0, 1.0}.
    """
    if mask.ndim not in (2, 3):
        raise ValueError(f"dilate_video_mask expects (F, H, W) or (H, W), got {mask.shape}")

    if mask.ndim == 2:
        mask = mask.unsqueeze(0)  # (1, H, W)

    f, h, w = mask.shape
    s_kernel = spatial_radius * 2 + 1
    t_kernel = temporal_radius * 2 + 1

    # Spatial dilation
    if s_kernel > 1:
        mask_4d = mask.unsqueeze(1)  # (F, 1, H, W)
        mask_4d = F.max_pool2d(mask_4d, kernel_size=s_kernel, stride=1, padding=spatial_radius)
        mask = mask_4d.squeeze(1)  # (F, H, W)

    # Temporal dilation
    if t_kernel > 1 and f > 1:
        # Reshape for 1D pool over time: (H*W, 1, F)
        mask_t = mask.permute(1, 2, 0).reshape(h * w, 1, f)
        mask_t = F.max_pool1d(mask_t, kernel_size=t_kernel, stride=1, padding=temporal_radius)
        mask = mask_t.reshape(h, w, f).permute(2, 0, 1)

    return (mask > 0.5).float()


# ---------------------------------------------------------------------------
# Laplacian pyramid blend (pyramid_blending.py equivalent, pure torch)
# ---------------------------------------------------------------------------

_CHUNK_SIZE = 8
_MASK_LOW_RES_LONG_SIDE = 64


def _pad_for_laplacian(image: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
    """Pad H, W to powers of two (kornia requirement for Laplacian pyramid)."""
    h, w = image.shape[-2], image.shape[-1]
    pad_right = 0
    pad_down = 0
    if not (is_powerof_two(w) and is_powerof_two(h)):
        pad_right = find_next_powerof_two(w) - w
        pad_down = find_next_powerof_two(h) - h
        image = F.pad(image, (0, pad_right, 0, pad_down), mode="reflect")
    return image, (pad_right, pad_down)


def _gaussian_pyramid(
    images: torch.Tensor,
    max_level: int,
    border_type: str = "reflect",
    align_corners: bool = False,
) -> list[torch.Tensor]:
    """Build a Gaussian pyramid using kornia's gaussian blur + downscale (build_pyramid).

    Args:
        images: (B, C, H, W) tensor.
        max_level: Number of pyramid levels.
    Returns:
        List of tensors from finest to coarsest.
    """
    h, w = images.shape[-2], images.shape[-1]
    if not (is_powerof_two(w) and is_powerof_two(h)):
        padding = (0, find_next_powerof_two(w) - w, 0, find_next_powerof_two(h) - h)
        images = F.pad(images, padding, mode=border_type)

    # ponytail: .float() guards against uint8/bool inputs; kornia gaussian blur
    # requires float. Handled once here.
    images = images.float()

    return build_pyramid(images, max_level, border_type, align_corners)


def _resize_preserving_aspect_ratio(
    images: torch.Tensor, long_side: int, mode: str = "bilinear"
) -> torch.Tensor:
    """Resize preserving aspect ratio so the long side matches `long_side`."""
    h, w = images.shape[-2:]
    current_long = max(h, w)
    if current_long == long_side:
        return images
    scale = long_side / current_long
    rh = max(1, int(round(h * scale)))
    rw = max(1, int(round(w * scale)))
    if mode == "nearest":
        return F.interpolate(images, size=(rh, rw), mode=mode)
    return F.interpolate(images, size=(rh, rw), mode=mode, align_corners=False)


def _apply_low_res_mask_dilation(
    mask: torch.Tensor,
    spatial_radius: int,
    long_side: int = _MASK_LOW_RES_LONG_SIDE,
) -> torch.Tensor:
    """Downscale mask → dilate spatially → upscale back, to soften blend boundary."""
    if spatial_radius <= 0:
        return mask
    original_size = mask.shape[-2:]
    mask_low_res = _resize_preserving_aspect_ratio(mask.float(), long_side, mode="bilinear")
    mask_low_res = F.max_pool2d(
        mask_low_res,
        kernel_size=spatial_radius * 2 + 1,
        stride=1,
        padding=spatial_radius,
    )
    return F.interpolate(mask_low_res, size=original_size, mode="bilinear", align_corners=False)


def _pyramid_blend_chunk(
    image1: torch.Tensor,
    image2: torch.Tensor,
    mask: torch.Tensor,
    max_level: int = 7,
) -> torch.Tensor:
    """Blend a single batch chunk (already padded, already on device).

    Uses kornia build_laplacian_pyramid (gaussian blur + bilinear) matching
    ComfyUI-LTXVideo pyramid_blending.py.

    Args:
        image1: (B, C, H_pad, W_pad)
        image2: (B, C, H_pad, W_pad)
        mask: (B, 1, H_pad, W_pad) — white=image1, black=image2
    """
    pyr1 = build_laplacian_pyramid(image1, max_level=max_level)
    pyr2 = build_laplacian_pyramid(image2, max_level=max_level)
    pyr_mask = _gaussian_pyramid(mask, max_level=max_level)
    pyr_up = PyrUp()

    # Coarsest level blend
    output = pyr1[-1] * pyr_mask[-1] + pyr2[-1] * (1.0 - pyr_mask[-1])

    # Reconstruct from coarse to fine (matching Comfy order)
    for i in range(len(pyr1) - 2, -1, -1):
        residual = pyr1[i] * pyr_mask[i] + pyr2[i] * (1.0 - pyr_mask[i])
        output = pyr_up(output) + residual

    return output


def _normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    """Normalise uint8/bool/out-of-range float tensors to float [0, 1].

    uint8 → float / 255.0
    bool → float (0/1 already correct)
    float w/ max > 1 + ε → / 255.0
    Already-normalised float → unchanged.
    """
    if t.dtype == torch.uint8:
        return t.float().div_(255.0)
    if t.dtype == torch.bool:
        return t.float()
    if t.is_floating_point() and t.numel() > 0 and t.max() > 1.0 + 1e-5:
        return t / 255.0
    return t


def laplacian_pyramid_blend(
    image_a: torch.Tensor,
    image_b: torch.Tensor,
    mask: torch.Tensor,
    max_level: int = 7,
    mask_low_res_dilation: int = 5,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Blend two image/video tensors using Laplacian pyramid.

    Args:
        image_a: (F, H, W, 3) or (B, F, H, W, 3) pixel frames in [0, 1].
        image_b: Same shape as image_a. Second source.
        mask: (F, H, W) or (B, F, H, W) — white=image_a, black=image_b.
        max_level: Number of pyramid levels. Default 7.
        mask_low_res_dilation: Dilate mask at low res before blend. 0 to disable.
        device: Computation device.
    Returns:
        Blended result, same shape as input.
    """
    # Normalise uint8/bool/out-of-range inputs to float [0, 1]
    image_a = _normalize_tensor(image_a)
    image_b = _normalize_tensor(image_b)
    mask = _normalize_tensor(mask)

    # Normalise to 4D (B, C, H, W)
    orig_ndim = image_a.ndim
    # Track original grid shape for 5D case
    b_grid = 0
    f_grid = 0
    h = 0
    w = 0
    if orig_ndim == 4:  # (F, H, W, 3)
        f_grid = image_a.shape[0]
        image_a = image_a.permute(0, 3, 1, 2)  # (F, 3, H, W)
        image_b = image_b.permute(0, 3, 1, 2)
        if mask.ndim == 4:
            mask = mask[:, :, :, 0]  # (F, H, W) from (F, H, W, 1) or similar
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)  # (F, 1, H, W)
    elif orig_ndim == 5:  # (B, F, H, W, 3)
        b_grid, f_grid, h, w, c = image_a.shape
        image_a = image_a.view(b_grid * f_grid, c, h, w)
        image_b = image_b.view(b_grid * f_grid, c, h, w)
        if mask.ndim == 4:  # (B, F, H, W)
            mask = mask.view(b_grid * f_grid, 1, h, w)
        elif mask.ndim == 3:  # (F, H, W)
            mask = mask.unsqueeze(0).expand(b_grid, -1, -1, -1).reshape(b_grid * f_grid, 1, h, w)
    else:
        raise ValueError(f"Unsupported image ndim: {orig_ndim}")

    if image_a.shape != image_b.shape:
        raise ValueError(f"image_a {image_a.shape} != image_b {image_b.shape}")
    if image_a.shape[0] != mask.shape[0]:
        raise ValueError(f"Batch mismatch: {image_a.shape[0]} != {mask.shape[0]}")
    if image_a.shape[-2:] != mask.shape[-2:]:
        raise ValueError(f"Spatial mismatch: {image_a.shape[-2:]} != {mask.shape[-2:]}")

    # Apply low-res mask dilation (matches official LTXVLaplacianPyramidBlend)
    if mask_low_res_dilation > 0:
        mask = _apply_low_res_mask_dilation(mask, mask_low_res_dilation)

    # Determine max_level based on padded size
    _, padding = _pad_for_laplacian(image_a[:1])
    orig_h, orig_w = image_a.shape[-2], image_a.shape[-1]
    padded_min = min(orig_h + padding[1], orig_w + padding[0])
    max_level = min(max_level, int(math.log2(padded_min)))

    # Pad mask too if needed
    if any(padding):
        mask = F.pad(mask.float(), (0, padding[0], 0, padding[1]), mode="reflect")

    b_total = image_a.shape[0]
    results: list[torch.Tensor] = []

    for start in range(0, b_total, _CHUNK_SIZE):
        end = min(start + _CHUNK_SIZE, b_total)
        img1_chunk, _ = _pad_for_laplacian(image_a[start:end])
        img2_chunk, _ = _pad_for_laplacian(image_b[start:end])
        mask_chunk = mask[start:end]

        if device is not None:
            img1_chunk = img1_chunk.to(device)
            img2_chunk = img2_chunk.to(device)
            mask_chunk = mask_chunk.to(device)

        blended = _pyramid_blend_chunk(img1_chunk, img2_chunk, mask_chunk, max_level=max_level)
        cropped = blended[..., :orig_h, :orig_w].clamp(0, 1)
        # ponytail: keep result on compute device — callers own .cpu() if needed
        results.append(cropped)

    result = torch.cat(results, dim=0)

    # Restore original shape
    if orig_ndim == 4:
        result = result.permute(0, 2, 3, 1)  # (F, H, W, 3)
    elif orig_ndim == 5 and b_grid > 0:
        result = result.view(b_grid, f_grid, 3, h, w).permute(0, 1, 3, 4, 2)  # (B, F, H, W, 3)

    return result
