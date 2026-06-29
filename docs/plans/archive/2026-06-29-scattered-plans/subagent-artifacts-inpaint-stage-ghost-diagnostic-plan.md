# IC-LoRA Inpaint Edge Ghost / Extra Scene — Diagnostic Plan

## Hypothesis

The ghost/extra scene beyond the mask edge originates at one of five pipeline stages. Each stage transition is an opportunity for the alternate generated content to bleed past the intended boundary.

## Pipeline Flow (exact line refs)

```
Input load (432-463)
→ Mask dilation (474-484)       mask_stage1_half, mask_stage2_full
→ Green composite (485-503)     green_half
→ Prompt encode (502)
→ Stage 1 denoising (544-559)   video_state_s1 (latent)
→ Video decoder (576-578)       decoded_s1_frames     ← INSTRUMENT POINT A
→ Laplacian blend 1 (581-591)   blend_stage1           ← INSTRUMENT POINT B
→ Upscale 2× + VAE encode       blend_full → encoded_blend (594-614)
→ Stage 2 denoising (618-682)   video_state_s2 (latent)
→ Video decoder (689-692)       decoded_s2_frames     ← INSTRUMENT POINT C
→ Laplacian blend 2 (703-711)   blend_stage2           ← INSTRUMENT POINT D
→ Raw mask guard (714-718)       final clamp            ← INSTRUMENT POINT E (post-guard)
→ Output encode (722-734)
```

## Instrumentation Points

### Point A — Stage 1 Raw Decode (line 578)
**Location:** `ltx_ic_lora_pipeline.py:578`
**Tensor shape:** `(F, H_half, W_half, 3)` in [0, 1]
**What it reveals:** Raw generated content before any blending. If ghost exists here, the model hallucinated outside the mask context during stage 1 denoising. If ghost does NOT exist here, the stage 1 decode is clean and the leak comes from blend or later.
**Dump:** Save first/last/middle frames as PNG, plus short video clip.

### Point B — Stage 1 Laplacian Blend (line 591)
**Location:** `ltx_ic_lora_pipeline.py:591`
**Tensor shape:** `(F, H_half, W_half, 3)` in [0, 1]
**What it reveals:** The Laplacian pyramid blend of decoded_s1 onto the original half-res video. If ghost first appears here, the blend is leaking — either `mask_low_res_dilation=5` is too aggressive or the pyramid levels allow content outside the mask to be generated and blended.
**Dump:** Same format.

### Point C — Stage 2 Raw Decode (line 692)
**Location:** `ltx_ic_lora_pipeline.py:692`
**Tensor shape:** `(F, H_full, W_full, 3)` in [0, 1]
**What it reveals:** Raw stage 2 decoded output. If ghost appears here, the stage 2 denoising (operating on the upscaled+VaeEncoded blend from stage 1) regenerated content outside the mask area. This can happen if the VAE re-encode of stage 1 blend introduces artifacts, or if stage 2 conditioning allows the model to "reimagine" outside the green guide.
**Dump:** Full-res PNG frames + short video clip.

### Point D — Stage 2 Laplacian Blend (line 711)
**Location:** `ltx_ic_lora_pipeline.py:711`
**Tensor shape:** `(F, H_full, W_full, 3)` in [0, 1]
**What it reveals:** The second Laplacian pyramid blend (decoded_s2 onto original full-res video). If ghost appears here but NOT at point C, the stage 2 Laplacian blend is the culprit — the mask dilation for blend (`laplacian_blend_grow`, default 6) is too wide, or the pyramid reconstruction leaks content.
**Dump:** Full-res PNG frames + short video clip.

### Point E — Final Raw Mask Guard (post line 718)
**Location:** `ltx_ic_lora_pipeline.py:718`
**Tensor shape:** `(F, H_full, W_full, 3)` in [0, 1]
**What it reveals:** The guard clamps anything outside the *raw* (undilated) user mask back to original. If ghost persists here, the raw mask itself is the wrong shape or has edge artifacts. The guard is a linear interpolation: `blend * mask + original * (1-mask)`, so anti-aliased mask edges naturally produce a half-strength ghost. If the user's mask is soft-edged, this is expected behavior — solution is a threshold or tighter mask.
**Dump:** Full-res final output + overlay of raw user mask boundaries.

## How to dump (existing utilities, zero new dependencies)

Available: `imageio` + `imageio-ffmpeg`, `opencv-python-headless`, PIL (transitive).

### Per-frame PNG dump (simplest)
```python
import cv2
import numpy as np
import os

def dump_frames(tensor: torch.Tensor, label: str, output_dir: str, max_frames: int = 8):
    """tensor: (F, H, W, 3) in [0, 1]. Saves first max_frames as PNG."""
    os.makedirs(output_dir, exist_ok=True)
    f = min(tensor.shape[0], max_frames)
    for i in range(f):
        arr = (tensor[i].cpu().numpy() * 255).astype(np.uint8)  # (H, W, 3)
        cv2.imwrite(f"{output_dir}/{label}_frame_{i:04d}.png", cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
```

### Short video dump (for visual inspection)
```python
import imageio

def dump_video(tensor: torch.Tensor, label: str, output_dir: str, fps: int = 8):
    """tensor: (F, H, W, 3) in [0, 1]."""
    os.makedirs(output_dir, exist_ok=True)
    arr = (tensor.cpu().numpy() * 255).astype(np.uint8)
    imageio.mimsave(f"{output_dir}/{label}.mp4", arr, fps=fps, codec="libx264")
```

### Contact sheet (single image grid)
```python
import math
import torch.nn.functional as F
import cv2
import numpy as np

def dump_contact_sheet(tensor: torch.Tensor, label: str, output_dir: str, cols: int = 4):
    """tensor: (F, H, W, 3) in [0, 1]. Arrange frames in a grid."""
    os.makedirs(output_dir, exist_ok=True)
    f, h, w, c = tensor.shape
    rows = math.ceil(f / cols)
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i in range(f):
        r, c_idx = divmod(i, cols)
        arr = (tensor[i].cpu().numpy() * 255).astype(np.uint8)
        grid[r*h:(r+1)*h, c_idx*w:(c_idx+1)*w] = arr
    cv2.imwrite(f"{output_dir}/{label}_contact_sheet.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
```

### Mask overlay (binary boundary on video frame)
```python
def dump_mask_overlay(video: torch.Tensor, mask: torch.Tensor, label: str, output_dir: str):
    """video: (F, H, W, 3) in [0,1], mask: (F, H, W) in [0,1]."""
    os.makedirs(output_dir, exist_ok=True)
    f = min(video.shape[0], 4)
    for i in range(f):
        frame = (video[i].cpu().numpy() * 255).astype(np.uint8).copy()
        m = (mask[i].cpu().numpy() > 0.5).astype(np.uint8) * 255
        # Red outline at mask boundary
        contour = cv2.Canny(m, 100, 200)
        frame[contour > 0] = [255, 0, 0]
        cv2.imwrite(f"{output_dir}/{label}_mask_overlay_{i:04d}.png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
```

## Diagnostic Approach

### Minimal run (single generation with env gate)

Gate the debug dump behind an env var so it's harmless in production:
```python
DEBUG_INPAINT = os.environ.get("LTX_DEBUG_INPAINT", "").lower() in ("1", "true", "yes")
```

When `DEBUG_INPAINT=true`, dump all 5 points to a timestamped output directory alongside the normal output.

### Comparison matrix

| Point | Ghost present? | Diagnosis |
|---|---|---|
| A | No | Stage 1 decode clean → not stage 1 generation |
| A | Yes | Stage 1 model generated outside mask — check conditioning/guidance |
| B | No (A clean), B has ghost | Stage 1 Laplacian blend leaking — reduce `INPAINT_BLEND1_LOW_RES_DILATION` (5) |
| C | No (B clean), C has ghost | Stage 2 denoising regenerated outside — check VAE re-encode quality or stage 2 conditioning |
| D | No (C clean), D has ghost | Stage 2 Laplacian blend leaking — reduce `laplacian_blend_grow` (6) |
| E | Ghost persists | Raw mask edge is soft/anti-aliased — threshold or morphological clean |

### If all raw decodes (A, C) are clean but blends (B, D, E) show ghost
The Laplacian pyramid blend itself is the origin. Blame path:
1. `mask_low_res_dilation` (5 → try 3 or 2) — the low-res dilation softens the mask boundary too much
2. `max_level=7` (try 5) — too many pyramid levels allow low-frequency content to bleed
3. The `_apply_low_res_mask_dilation` function in `official_inpaint.py:246-258` downsamples mask to 64px long side, dilates, upscales — this blurring may soften the mask enough that green-conditioned content half a pixel away bleeds in

## Insertion spots (file:line)

One block at the end of `generate_inpaint()` before the output encode (line 720), gated by env var:

```
# In ltx_ic_lora_pipeline.py, around line 718-720, insert:
#
# if DEBUG_INPAINT:
#     _dump_inpaint_stages(...)
```

The dump function itself goes as a free function in the same file or in `official_inpaint.py`:
```python
def _dump_inpaint_stages(
    output_dir: str,
    decoded_s1: Tensor,      # point A
    blend_s1: Tensor,        # point B
    decoded_s2: Tensor,      # point C
    blend_s2: Tensor,        # point D
    final: Tensor,           # point E
    mask_s1: Tensor,
    mask_s2: Tensor,
    raw_mask: Tensor,
    original_full: Tensor,
    original_half: Tensor,
) -> None:
    ...
```

Do NOT gate individual points — dump all 5 synchronously at the end. This avoids needing intermediate hook points (avoids threading/yield complexity) and gives a complete comparison matrix in a single run.

## Line Reference Summary

| Component | File | Line(s) |
|---|---|---|
| derive_stage_radii | `ltx_ic_lora_pipeline.py` | 27-45 |
| INPAINT_BLEND1_LOW_RES_DILATION=5 | `ltx_ic_lora_pipeline.py` | 22 |
| INPAINT_BLEND2_LOW_RES_DILATION=6 | `ltx_ic_lora_pipeline.py` | 23 |
| generate_inpaint signature | `ltx_ic_lora_pipeline.py` | 387-402 |
| Input load | `ltx_ic_lora_pipeline.py` | 432-463 |
| Mask dilation | `ltx_ic_lora_pipeline.py` | 474-484 |
| Green composite | `ltx_ic_lora_pipeline.py` | 485-503 |
| Stage 1 denoising | `ltx_ic_lora_pipeline.py` | 544-559 |
| Stage 1 decode (POINT A) | `ltx_ic_lora_pipeline.py` | 576-578 |
| Stage 1 Laplacian (POINT B) | `ltx_ic_lora_pipeline.py` | 581-591 |
| Upscale + VAE encode | `ltx_ic_lora_pipeline.py` | 594-614 |
| Stage 2 denoising | `ltx_ic_lora_pipeline.py` | 618-682 |
| Stage 2 decode (POINT C) | `ltx_ic_lora_pipeline.py` | 689-692 |
| Stage 2 Laplacian (POINT D) | `ltx_ic_lora_pipeline.py` | 703-711 |
| Raw mask guard (POINT E) | `ltx_ic_lora_pipeline.py` | 714-718 |
| Output encode | `ltx_ic_lora_pipeline.py` | 722-734 |
| _apply_raw_mask_guard | `ltx_ic_lora_pipeline.py` | 300-310 |
| _collect_frames | `ltx_ic_lora_pipeline.py` | 767-772 |
| laplacian_pyramid_blend | `official_inpaint.py` | 242-363 |
| _pyramid_blend_chunk | `official_inpaint.py` | 205-228 |
| _apply_low_res_mask_dilation | `official_inpaint.py` | 246-258 |
| green_composite_preprocess | `official_inpaint.py` | 35-79 |
| dilate_video_mask | `official_inpaint.py` | 86-128 |
| _pad_for_laplacian | `official_inpaint.py` | 142-148 |
| encode_video_output | `ltx_pipeline_common.py` | 35-41 |
| imageio (available) | `pyproject.toml` | dep |
| opencv-python-headless (available) | `pyproject.toml` | dep |

## Task Packet

```
Files to edit:
  1. backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py — add env gate + dump call at line ~718
  2. backend/services/ic_lora_pipeline/official_inpaint.py — add _dump_inpaint_stages helper (or same file)

Edit scope:
  Add ~50 lines total. One env-gated debug dump function + one call site.
  No new dependencies. No test changes. No file creation beyond dump output dir.

Proof:
  LATENCY_DEBUG_INPAINT=true pnpm backend:test -- -k "inpaint" will produce
  debug_dumps/<timestamp>/ directory with 5 videos + contact sheets per stage.
  Production path: unchanged when env var absent.
```
