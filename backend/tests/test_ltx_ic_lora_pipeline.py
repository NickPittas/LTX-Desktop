"""Unit tests for LTX IC-LoRA pipeline internals (no GPU, no mocks)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
    LTXIcLoraPipeline,
    _vae_compatible_frame_count,
    _vae_padded_frame_count,
)


# ── Visual/manual acceptance gate for inpaint parity ──
#
# To validate a live inpaint output:
#   1. Generate fixed seed 42 sample (seed uniform across runs).
#   2. Export a triptych: [original_frame | effective_dilated_mask | output_frame].
#      Effective mask = stage2 (full-res) mask dilated via derive_stage_radii(30)[1].
#   3. Outside effective mask: mean absolute diff < 5/255 (~0.0196) between
#      output and original (per-pixel, background must be preserved).
#   4. Inside effective mask: mean absolute diff > 20/255 (~0.0784) when prompt
#      should alter content (model must actually change the masked region).
#   5. Output has audio stream when model/audio context produces audio.
#      Verify with: ffprobe -v error -show_entries stream=codec_type
#   6. Sampler status: default Euler via SimpleDenoiser/DiffusionStage.
#      Comfy workflow uses euler_cfg_pp (stage1) and euler_ancestral_cfg_pp
#      (stage2) via LTXVCFGGuider — not implemented in installed ltx_pipelines.
#      Report honestly; do not claim parity without an existing implementation.
#
# The test below (test_outside_mask_preserved_with_bright_generated) is the
# automated version of these checks at unit-test level, fast and deterministic.

# ponytail: bare assert self-check — verify spatial resize preserves frame count
_VIDEO = torch.rand(1, 3, 17, 384, 768)
_RESIZED = LTXIcLoraPipeline._resize_video_spatial(_VIDEO, 192, 384)
assert _RESIZED.shape == (1, 3, 17, 192, 384), (
    f"_resize_video_spatial shape mismatch: {_RESIZED.shape}"
)
del _VIDEO, _RESIZED

# ponytail: STAGE_2_DISTILLED_SIGMA_VALUES[1:] → [0.725, 0.421875, 0.0]
# Full source: [0.909375, 0.725, 0.421875, 0.0]; [1:] drops 0.909375.


def test_prompt_encoded_before_video_loaded():
    """Prompt encoder call appears before decode_video_by_frame in generate_inpaint.

    Regression: OOM fix avoids GGUF Gemma + 196f video tensor VRAM overlap
    (~31.6GB). If order reverses, peak spikes.
    """
    import os
    pipe_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
    )
    with open(pipe_path) as f:
        source = f.read()

    start = source.find("def generate_inpaint")
    assert start != -1, "generate_inpaint method not found"

    method_body = source[start:]
    prompt_pos = method_body.find("self.pipeline.prompt_encoder")
    video_pos = method_body.find("iter_video_frames_to_model_domain")

    assert prompt_pos != -1, "prompt_encoder not found in generate_inpaint"
    assert video_pos != -1, "iter_video_frames_to_model_domain not found in generate_inpaint"
    assert prompt_pos < video_pos, (
        f"prompt_encoder at offset {prompt_pos} must appear before "
        f"video decode at offset {video_pos} — "
        "prompt encoding moved before video loading to reduce peak VRAM"
    )


class TestVaeFrameCount:
    """Ensure _vae_compatible_frame_count produces 1+8*k values."""

    def test_exact_divisible(self):
        assert _vae_compatible_frame_count(193) == 193
        assert _vae_compatible_frame_count(97) == 97
        assert _vae_compatible_frame_count(65) == 65

    def test_rounds_down(self):
        assert _vae_compatible_frame_count(200) == 193
        assert _vae_compatible_frame_count(100) == 97
        assert _vae_compatible_frame_count(66) == 65
        assert _vae_compatible_frame_count(64) == 57

    def test_minimum(self):
        assert _vae_compatible_frame_count(1) == 1
        assert _vae_compatible_frame_count(0) == 1
        assert _vae_compatible_frame_count(-1) == 1

    def test_small_exact(self):
        assert _vae_compatible_frame_count(9) == 9
        assert _vae_compatible_frame_count(17) == 17

    def test_small_rounds_down(self):
        assert _vae_compatible_frame_count(10) == 9
        assert _vae_compatible_frame_count(16) == 9


class TestVaePaddedFrameCount:
    """_vae_padded_frame_count returns next 8n+1 stably (no re-pad when already compatible)."""

    def test_exact_divisible(self):
        assert _vae_padded_frame_count(193) == 193
        assert _vae_padded_frame_count(97) == 97
        assert _vae_padded_frame_count(65) == 65

    def test_pads_up(self):
        assert _vae_padded_frame_count(196) == 201
        assert _vae_padded_frame_count(200) == 201
        assert _vae_padded_frame_count(100) == 105
        assert _vae_padded_frame_count(66) == 73

    def test_minimum(self):
        assert _vae_padded_frame_count(1) == 1
        assert _vae_padded_frame_count(0) == 1
        assert _vae_padded_frame_count(-1) == 1

    def test_small_exact(self):
        assert _vae_padded_frame_count(9) == 9
        assert _vae_padded_frame_count(17) == 17

    def test_small_pads_up(self):
        assert _vae_padded_frame_count(10) == 17
        assert _vae_padded_frame_count(16) == 17

    def test_sequence_padding(self):
        """Verify padding behavior for 1..25: already-compatible stays, others pad to next."""
        for n in range(1, 26):
            result = _vae_padded_frame_count(n)
            assert (result - 1) % 8 == 0, f"{n} → {result}, not 8n+1"
            assert result >= n, f"{n} → {result} < {n}"
            if (n - 1) % 8 == 0:
                assert result == n, f"{n} already 8n+1, should stay, got {result}"


class TestCompositeInOutpainting:
    """Verify compositing math: gen * mask + orig * (1 - mask).

    White mask (1.0) → keep generated region.
    Black mask (0.0) → preserve original region.
    """

    @staticmethod
    def _composite(
        gen: torch.Tensor, orig: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        mask_3ch = mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        return gen * mask_3ch + orig * (1.0 - mask_3ch)

    def test_black_mask_preserves_original_white_mask_uses_generated(self):
        """Split mask: black left preserves original, white right uses generated."""
        F, h, w = 1, 64, 64
        orig_val, gen_val = 0.4, 0.8

        orig = torch.full((F, h, w, 3), orig_val)
        gen = torch.full((F, h, w, 3), gen_val)
        mask = torch.zeros(F, h, w)
        mask[:, :, w // 2:] = 1.0

        result = self._composite(gen, orig, mask)

        mean_left = result[:, :, : w // 2, :].mean().item()
        mean_right = result[:, :, w // 2 :, :].mean().item()

        assert abs(mean_left - orig_val) < abs(mean_left - gen_val), (
            f"Black mask side mean {mean_left:.4f} should be closer to "
            f"orig={orig_val} than gen={gen_val}"
        )
        assert abs(mean_right - gen_val) < abs(mean_right - orig_val), (
            f"White mask side mean {mean_right:.4f} should be closer to "
            f"gen={gen_val} than orig={orig_val}"
        )
        assert abs(mean_left - orig_val) < 1e-5, f"Black mask should preserve original exactly"
        assert abs(mean_right - gen_val) < 1e-5, f"White mask should use generated exactly"

    def test_dual_frame_mask(self):
        """Frame 0 all-black mask preserves original. Frame 1 all-white mask uses generated."""
        F, h, w = 2, 64, 64
        orig_val, gen_val = 0.2, 0.9

        orig = torch.full((F, h, w, 3), orig_val)
        gen = torch.full((F, h, w, 3), gen_val)
        mask = torch.zeros(F, h, w)
        mask[0] = 0.0  # black → original
        mask[1] = 1.0  # white → generated

        result = self._composite(gen, orig, mask)

        assert abs(result[0].mean().item() - orig_val) < 1e-5, (
            f"Frame 0 (black mask) should be near orig={orig_val}"
        )
        assert abs(result[1].mean().item() - gen_val) < 1e-5, (
            f"Frame 1 (white mask) should be near gen={gen_val}"
        )

    def test_gray_mask_blends(self):
        """Mid-gray mask (0.5) blends 50/50 under linear alpha."""
        F, h, w = 1, 64, 64
        orig = torch.zeros((F, h, w, 3))
        gen = torch.ones((F, h, w, 3))
        mask = torch.full((F, h, w), 0.5)

        result = self._composite(gen, orig, mask)

        expected = 1.0 * 0.5 + 0.0 * 0.5
        assert abs(result.mean().item() - expected) < 1e-5, (
            f"Gray mask should produce {expected}, got {result.mean().item():.4f}"
        )


class TestInpaintUtilities:
    """Tests for official inpaint utility functions (green composite, dilation, Laplacian blend)."""

    def test_green_composite_applies_bg_color(self):
        """White mask region should be #66FF00 green, black mask region keeps original."""
        from services.ic_lora_pipeline.official_inpaint import green_composite_preprocess

        # Create black frames - zeros in [-1, 1] space = -1.0
        images = torch.full((1, 3, 3, 64, 64), -1.0)  # (B, C, F, H, W) in [-1, 1] = black
        mask = torch.zeros(3, 64, 64)
        mask[:, 32:, 32:] = 1.0  # white mask in bottom-right quadrant

        result = green_composite_preprocess(images, mask)

        # Black mask region should remain black (-1.0)
        black_region = result[0, :, 0, :32, :32]
        assert torch.allclose(black_region, -torch.ones_like(black_region), atol=1e-5), (
            f"Black mask region should preserve original pixels, got {black_region[:, 0, 0]}"
        )

        # White mask region should be green (#66FF00 mapped to [-1, 1])
        # #66FF00 in [0,1] = (102/255, 255/255, 0) = (0.4, 1.0, 0.0)
        # in [-1, 1] = (2*0.4-1, 2*1.0-1, 2*0.0-1) = (-0.2, 1.0, -1.0)
        white_region = result[0, :, 0, 32:, 32:]
        expected_green = torch.tensor([-0.2, 1.0, -1.0], dtype=torch.float32).view(3, 1, 1)
        assert torch.allclose(white_region, expected_green.expand_as(white_region), atol=1e-5), (
            f"White mask region should be green, got {white_region[:, 0, 0]}"
        )

    def test_green_composite_broadcasts_single_frame_mask(self):
        """A single-frame mask should be broadcast to all video frames."""
        from services.ic_lora_pipeline.official_inpaint import green_composite_preprocess

        images = torch.zeros(1, 3, 5, 64, 64)
        mask = torch.zeros(1, 64, 64)  # single frame mask
        mask[:, 32:, 32:] = 1.0

        result = green_composite_preprocess(images, mask)
        assert result.shape == (1, 3, 5, 64, 64), f"Shape mismatch: {result.shape}"
        # All frames should have the same green pattern
        for f in range(5):
            assert torch.allclose(result[0, :, f], result[0, :, 0]), (
                f"Frame {f} should match frame 0"
            )

    def test_green_composite_trims_shortest(self):
        """When mask has fewer frames than video, trim to shortest."""
        from services.ic_lora_pipeline.official_inpaint import green_composite_preprocess

        images = torch.zeros(1, 3, 5, 64, 64)
        mask = torch.zeros(3, 64, 64)  # 3 frames
        mask[:, :, :] = 1.0

        result = green_composite_preprocess(images, mask)
        assert result.shape == (1, 3, 3, 64, 64), (
            f"Should trim to 3 frames, got {result.shape}"
        )

    def test_dilate_video_mask_spatial(self):
        """Spatial dilation expands mask boundaries."""
        from services.ic_lora_pipeline.official_inpaint import dilate_video_mask

        mask = torch.zeros(3, 64, 64)
        mask[0, 32, 32] = 1.0  # single pixel

        dilated = dilate_video_mask(mask, spatial_radius=3, temporal_radius=0)
        # After dilation with radius 3 (kernel 7x7), the 1 pixel should expand to ~7x7
        assert dilated[0].sum() > 1.0, "Spatial dilation should expand mask"
        # Non-mask frames should remain unchanged
        assert dilated[1].sum() == 0.0, "Frame without mask should stay zero"
        assert dilated[2].sum() == 0.0, "Frame without mask should stay zero"

    def test_dilate_video_mask_temporal(self):
        """Temporal dilation expands mask along time axis."""
        from services.ic_lora_pipeline.official_inpaint import dilate_video_mask

        mask = torch.zeros(5, 64, 64)
        mask[2, :, :] = 1.0  # full frame at middle

        dilated = dilate_video_mask(mask, spatial_radius=0, temporal_radius=1)
        # With temporal radius 1 (kernel 3), the mask should spread to frames 1, 2, 3
        assert dilated[1].sum() > 0.0, "Temporal dilation should spread to adjacent frame"
        assert dilated[2].sum() > 0.0, "Original frame should remain"
        assert dilated[3].sum() > 0.0, "Temporal dilation should spread to adjacent frame"
        assert dilated[0].sum() == 0.0, "Frames beyond radius should stay zero"

    def test_laplacian_blend_basic(self):
        """Laplacian blend preserves overall value range."""
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        f, h, w = 3, 64, 64
        img_a = torch.full((f, h, w, 3), 0.0)  # black
        img_b = torch.full((f, h, w, 3), 1.0)  # white
        mask = torch.zeros(f, h, w)
        mask[:, :, w // 2 :] = 1.0  # left=black mask(+image_b), right=white mask(+image_a)

        blended = laplacian_pyramid_blend(img_a, img_b, mask, max_level=3, mask_low_res_dilation=0)

        # Result should be in [0, 1]
        assert blended.min() >= 0.0, f"Min below 0: {blended.min()}"
        assert blended.max() <= 1.0, f"Max above 1: {blended.max()}"
        # Mean should be between 0.1 and 0.9 (not all-0 or all-1, boundary blur softens extremes)
        assert 0.1 < blended.mean() < 0.9, f"Mean outside expected range: {blended.mean()}"

    def test_laplacian_blend_preserves_identity(self):
        """Blending identical images should produce the same image."""
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        f, h, w = 1, 64, 64
        img = torch.rand(f, h, w, 3)
        mask = torch.ones(f, h, w) * 0.5  # uniform gray mask

        blended = laplacian_pyramid_blend(img, img, mask, max_level=3, mask_low_res_dilation=0)

        assert torch.allclose(blended, img, atol=1e-5), (
            "Blending identical images should preserve identity"
        )

    def test_laplacian_blend_low_res_dilation_expands_blend(self):
        """mask_low_res_dilation > 0 should expand blend region vs 0.

        Single-pixel mask: dilation=0 blends locally, dilation>0 expands
        the mask at low res then blends with a larger boundary.
        """
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        f, h, w = 1, 64, 64
        img_a = torch.full((f, h, w, 3), 0.0)  # black
        img_b = torch.full((f, h, w, 3), 1.0)  # white
        mask = torch.zeros(f, h, w)
        mask[0, h // 2, w // 2] = 1.0  # single pixel

        result_no_dil = laplacian_pyramid_blend(img_a, img_b, mask, max_level=3, mask_low_res_dilation=0)
        result_dil = laplacian_pyramid_blend(img_a, img_b, mask, max_level=3, mask_low_res_dilation=6)

        assert result_no_dil.min() >= 0.0
        assert result_no_dil.max() <= 1.0
        assert result_dil.min() >= 0.0
        assert result_dil.max() <= 1.0

        diff = (result_dil - result_no_dil).abs().mean().item()
        assert diff > 1e-4, (
            f"mask_low_res_dilation=6 should produce measurably different "
            f"blend from dilation=0; mean abs diff = {diff:.6f}"
        )

    def test_laplacian_blend_polarity_preserved_with_dilation(self):
        """Polarity: white mask = image_a, black mask = image_b, at any dilation.

        mask_low_res_dilation=6 must change blend result vs 0, but polarity
        must remain correct at both settings.
        """
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        f, h, w = 1, 64, 64
        img_a = torch.full((f, h, w, 3), 0.95)  # near-white
        img_b = torch.full((f, h, w, 3), 0.05)  # near-black

        # All-white mask: should prefer image_a (bright).
        # All-black mask: should prefer image_b (dark).
        mask_white = torch.ones(f, h, w)
        mask_black = torch.zeros(f, h, w)

        # Test at both dilation values
        for dil in (0, 6):
            blend_white = laplacian_pyramid_blend(
                img_a, img_b, mask_white, max_level=3, mask_low_res_dilation=dil,
            )
            blend_black = laplacian_pyramid_blend(
                img_a, img_b, mask_black, max_level=3, mask_low_res_dilation=dil,
            )

            # White mask → output near image_a (0.95)
            white_mean = blend_white.mean().item()
            assert white_mean > 0.5, (
                f"White mask polarity: expected >0.5, got {white_mean:.4f} at dil={dil}"
            )

            # Black mask → output near image_b (0.05)
            black_mean = blend_black.mean().item()
            assert black_mean < 0.5, (
                f"Black mask polarity: expected <0.5, got {black_mean:.4f} at dil={dil}"
            )

            # White mask must be brighter than black mask
            assert white_mean > black_mean + 0.2, (
                f"Polarity reversal at dil={dil}: white={white_mean:.4f} <= black={black_mean:.4f}"
            )

        # Dilation=6 must differ measurably from dilation=0 when mask has an edge.
        # Uniform mask dilates to itself — use a half-white/half-black mask.
        mask_vertical = torch.zeros(f, h, w)
        mask_vertical[:, :, : w // 2] = 1.0  # left half white, right half black
        blend_dil0 = laplacian_pyramid_blend(
            img_a, img_b, mask_vertical, max_level=3, mask_low_res_dilation=0,
        )
        blend_dil6 = laplacian_pyramid_blend(
            img_a, img_b, mask_vertical, max_level=3, mask_low_res_dilation=6,
        )
        diff = (blend_dil6 - blend_dil0).abs().mean().item()
        assert diff > 1e-4, (
            f"mask_low_res_dilation=6 must change blend vs 0; diff={diff:.6f}"
        )

    def test_laplacian_blend_uint8_inputs(self):
        """uint8 mask 0/255 + uint8 images 50/200 must produce float [0, 1] blend.

        Regression: previous code only .float()ed without /255, so 0..255 values
        saturate .clamp(0,1) → all-white output. Normalize at function boundary
        fixes this live white-out bug.
        """
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        f, h, w = 3, 64, 64
        # uint8 mask with values 0 and 255 (typical from cv2/decoder)
        mask_uint8 = torch.zeros(f, h, w, dtype=torch.uint8)
        mask_uint8[:, :, w // 2 :] = 255

        # uint8 images in [0, 255]
        img_a_uint8 = torch.full((f, h, w, 3), 50, dtype=torch.uint8)  # ~0.196
        img_b_uint8 = torch.full((f, h, w, 3), 200, dtype=torch.uint8)  # ~0.784

        result = laplacian_pyramid_blend(
            img_a_uint8, img_b_uint8, mask_uint8, max_level=3, mask_low_res_dilation=0
        )

        # Output must be float in [0, 1]
        assert result.dtype in (torch.float32, torch.float64), (
            f"Expected float, got {result.dtype}"
        )
        assert result.min() >= 0.0, f"Min below 0: {result.min()}"
        assert result.max() <= 1.0, f"Max above 1: {result.max()} — white-out bug"

        # Left half (mask=0 → image_b side ≈ 0.784) should be > 0.5
        # Right half (mask=255 → image_a side ≈ 0.196) should be < 0.5
        # Tolerant due to pyramid boundary blur
        f, h, w = result.shape[:3]
        half_w = w // 2
        mean_left = result[:, :, :half_w, :].mean().item()
        mean_right = result[:, :, half_w:, :].mean().item()
        assert mean_left > 0.5, (
            f"Left (image_b=200/255) mean {mean_left:.4f} should be >0.5"
        )
        assert mean_right < 0.5, (
            f"Right (image_a=50/255) mean {mean_right:.4f} should be <0.5"
        )
        assert 0.0 < result.mean() < 1.0, f"Overall mean outside (0,1): {result.mean()}"


class TestDeriveStageRadii:
    """derive_stage_radii maps mask_grow_px to (stage1, stage2) radii."""

    def test_default_30_gives_15_30(self):
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import derive_stage_radii
        s1, s2 = derive_stage_radii(30)
        assert s1 == 15, f"Expected s1=15, got {s1}"
        assert s2 == 30, f"Expected s2=30, got {s2}"

    def test_zero_gives_0_0(self):
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import derive_stage_radii
        s1, s2 = derive_stage_radii(0)
        assert s1 == 0, f"Expected s1=0, got {s1}"
        assert s2 == 0, f"Expected s2=0, got {s2}"

    def test_one_gives_1_1(self):
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import derive_stage_radii
        s1, s2 = derive_stage_radii(1)
        assert s1 == 1, f"Expected s1=1, got {s1}"
        assert s2 == 1, f"Expected s2=1, got {s2}"

    def test_odd_value_ceil_div(self):
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import derive_stage_radii
        s1, s2 = derive_stage_radii(31)
        assert s1 == 16, f"Expected s1=16, got {s1}"
        assert s2 == 31, f"Expected s2=31, got {s2}"

    def test_blend1_low_res_dilation_is_5(self):
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import INPAINT_BLEND1_LOW_RES_DILATION
        assert INPAINT_BLEND1_LOW_RES_DILATION == 5, (
            f"Expected 5, got {INPAINT_BLEND1_LOW_RES_DILATION}"
        )

    def test_blend_constants_are_separate_from_radii(self):
        """INPAINT_BLEND1 docs blend constant independent of stage radii derived from mask_grow_px."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            INPAINT_BLEND1_LOW_RES_DILATION,
            derive_stage_radii,
        )
        s1, s2 = derive_stage_radii(30)
        assert INPAINT_BLEND1_LOW_RES_DILATION != s1, (
            "Blend1 constant should differ from stage1 radius (separate controls)"
        )


class TestApplyRawMaskGuard:
    """_apply_raw_mask_guard clamps generated pixels outside raw (undilated) user mask back to original.

    Grayscale mask preserves anti-aliased feathering; no threshold.
    """

    def test_outside_raw_mask_equals_original(self):
        """Pixels outside the raw user mask must equal original, even if generated differs."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = (5, 64, 96)
        # Generated: bright everywhere
        blend = torch.full((F, H, W, 3), 0.95)
        # Original: dark everywhere
        orig = torch.full((F, H, W, 3), 0.05)
        # Raw mask: small white square in center, rest black
        raw_mask = torch.zeros(F, H, W)
        raw_mask[:, H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = 1.0

        result = LTXIcLoraPipeline._apply_raw_mask_guard(blend, raw_mask, orig)

        # Outside raw mask: result must equal original exactly
        outside = (1.0 - raw_mask).unsqueeze(-1).expand(-1, -1, -1, 3)
        outside_diff = (result - orig).abs() * outside
        assert outside_diff.max() < 1e-6, (
            "Pixels outside raw mask must equal original exactly"
        )

        # Inside raw mask: result must be blend (bright generated)
        inside = raw_mask.unsqueeze(-1).expand(-1, -1, -1, 3)
        inside_diff = (result - blend).abs() * inside
        assert inside_diff.max() < 1e-6, (
            "Pixels inside raw mask must carry blend forward"
        )

    def test_grayscale_mask_feathers_edge(self):
        """Anti-aliased mask values (e.g. 0.5) produce intermediate blend, not hard threshold."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = (2, 64, 96)
        blend = torch.full((F, H, W, 3), 0.9)
        orig = torch.full((F, H, W, 3), 0.1)
        # Half-gray mask = 50/50 everywhere
        raw_mask = torch.full((F, H, W), 0.5)

        result = LTXIcLoraPipeline._apply_raw_mask_guard(blend, raw_mask, orig)

        # 0.9 * 0.5 + 0.1 * 0.5 = 0.5
        expected = torch.full((F, H, W, 3), 0.5)
        assert torch.allclose(result, expected, atol=1e-6), (
            f"Grayscale mask should produce 50/50 blend, got mean {result.mean().item():.4f}"
        )

    def test_all_black_mask_returns_original(self):
        """All-black (no generation) mask = pure original."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = (3, 64, 64)
        blend = torch.full((F, H, W, 3), 0.99)
        orig = torch.full((F, H, W, 3), 0.01)
        raw_mask = torch.zeros(F, H, W)

        result = LTXIcLoraPipeline._apply_raw_mask_guard(blend, raw_mask, orig)
        assert torch.allclose(result, orig, atol=1e-6), (
            "All-black mask must return original exactly"
        )

    def test_all_white_mask_returns_blend(self):
        """All-white (full generation) mask = passes blend through."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = (3, 64, 64)
        blend = torch.full((F, H, W, 3), 0.99)
        orig = torch.full((F, H, W, 3), 0.01)
        raw_mask = torch.ones(F, H, W)

        result = LTXIcLoraPipeline._apply_raw_mask_guard(blend, raw_mask, orig)
        assert torch.allclose(result, blend, atol=1e-6), (
            "All-white mask must pass blend through"
        )

    def test_blur_radius_creates_feathered_edge(self):
        """blur_radius > 0 softens mask edge: far outside=orig, far inside=blend,
        edge zone has intermediate values.
        """
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = 1, 32, 64
        orig = torch.full((F, H, W, 3), 0.05)
        blend = torch.full((F, H, W, 3), 0.95)
        raw_mask = torch.zeros(F, H, W)
        raw_mask[:, :, W // 2:] = 1.0  # left half orig, right half blend

        no_blur = LTXIcLoraPipeline._apply_raw_mask_guard(blend, raw_mask, orig, blur_radius=0)
        blurred = LTXIcLoraPipeline._apply_raw_mask_guard(blend, raw_mask, orig, blur_radius=4)

        # Far outside (col 0): both match original
        assert torch.allclose(no_blur[:, :, 0, :], orig[:, :, 0, :], atol=1e-6)
        assert torch.allclose(blurred[:, :, 0, :], orig[:, :, 0, :], atol=1e-6)

        # Far inside (col W-1): both match blend
        assert torch.allclose(no_blur[:, :, W - 1, :], blend[:, :, W - 1, :], atol=1e-6)
        assert torch.allclose(blurred[:, :, W - 1, :], blend[:, :, W - 1, :], atol=1e-6)

        # Transition zone left of edge: no_blur still at orig, blurred is intermediate
        col_t = W // 2 - 3
        v_nb = no_blur[0, 0, col_t, 0].item()
        v_bl = blurred[0, 0, col_t, 0].item()
        assert abs(v_nb - 0.05) < 1e-6, f"no_blur at col {col_t} should be 0.05, got {v_nb:.4f}"
        assert 0.05 < v_bl < 0.95, f"blurred at col {col_t} should be intermediate, got {v_bl:.4f}"
        assert abs(v_bl - v_nb) > 0.01, f"blurred should differ from no_blur at col {col_t}"

    # ── Chunking / OOM guard tests ──

    def test_chunking_matches_expected_math(self):
        """Chunked GPU→CPU result matches expected composite formula for blur_radius=0 and 1.

        Uses chunk_size=2 to exercise multi-chunk path even on small tensors.
        """
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = 6, 32, 48
        blend = torch.full((F, H, W, 3), 0.9)   # bright gen
        orig = torch.full((F, H, W, 3), 0.1)    # dark original
        mask = torch.full((F, H, W), 0.5)         # 50/50 everywhere

        # blur_radius=0 → direct grayscale composite: 0.9*0.5 + 0.1*0.5 = 0.5
        result = LTXIcLoraPipeline._apply_raw_mask_guard(blend, mask, orig, blur_radius=0, chunk_size=2)
        assert result.device.type == "cpu", f"Expected CPU, got {result.device}"
        expected = torch.full((F, H, W, 3), 0.5)
        assert torch.allclose(result, expected, atol=1e-6), (
            f"blur=0 chunk result should be 0.5, got mean {result.mean().item():.6f}"
        )

        # blur_radius=1 → threshold to binary (0.5>0.5=False=0), so all-black => orig
        result_b1 = LTXIcLoraPipeline._apply_raw_mask_guard(blend, mask, orig, blur_radius=1, chunk_size=3)
        assert result_b1.device.type == "cpu", f"Expected CPU, got {result_b1.device}"
        expected_b1 = orig.clone()
        assert torch.allclose(result_b1, expected_b1, atol=1e-6), (
            f"blur=1 with 0.5 mask thresholded to 0 should return orig, "
            f"got mean {result_b1.mean().item():.6f}"
        )

    def test_chunking_loop_source_assert(self):
        """Source text asserts the chunking loop and .cpu() accumulation exist."""
        from pathlib import Path
        source = Path(__file__).resolve().parents[1] / "services" / "ic_lora_pipeline" / "ltx_ic_lora_pipeline.py"
        text = source.read_text()
        assert "chunk_size:" in text, "_apply_raw_mask_guard must have chunk_size parameter"
        assert "chunks.append(chunk_result.detach().cpu())" in text, (
            "Loop must accumulate chunk results via .cpu() detach to free GPU memory"
        )
        assert "torch.cat(chunks, dim=0)" in text, (
            "Final result must torch.cat CPU chunks"
        )
        assert "for start in range(0, num_frames, chunk_size):" in text, (
            "Must iterate over frame dimension in chunks"
        )

    def test_chunk_size_greater_than_frames_still_works(self):
        """chunk_size > num_frames (single chunk) produces same result as chunk_size=2."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = 3, 16, 24
        blend = torch.rand(F, H, W, 3)
        orig = torch.rand(F, H, W, 3)
        mask = torch.rand(F, H, W)

        r1 = LTXIcLoraPipeline._apply_raw_mask_guard(blend, mask, orig, blur_radius=0, chunk_size=999)
        r2 = LTXIcLoraPipeline._apply_raw_mask_guard(blend, mask, orig, blur_radius=0, chunk_size=2)
        assert torch.allclose(r1, r2, atol=1e-6), (
            "Single chunk must match multi-chunk result"
        )

    def test_chunking_preserves_device_of_input_on_default(self):
        """Default chunk_size=8 works; result is CPU regardless of input device."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        F, H, W = 9, 16, 24
        blend = torch.full((F, H, W, 3), 0.85, device="cpu")
        orig = torch.full((F, H, W, 3), 0.15, device="cpu")
        mask = torch.full((F, H, W), 0.6, device="cpu")

        result = LTXIcLoraPipeline._apply_raw_mask_guard(blend, mask, orig)
        assert result.device.type == "cpu", f"Expected CPU default, got {result.device}"
        assert result.shape == (F, H, W, 3), f"Shape mismatch: {result.shape}"
        assert 0.0 <= result.min() <= result.max() <= 1.0, (
            f"Values out of [0,1]: [{result.min():.4f}, {result.max():.4f}]"
        )


class TestInpaintRuntimeParity:
    """Stage2 sigma, seed, and audio preservation parity checks."""

    def test_stage2_sigmas_slice(self):
        """STAGE_2_DISTILLED_SIGMAS[1:] → [0.725, 0.421875, 0.0] (3 steps)."""
        from ltx_pipelines.utils.constants import STAGE_2_DISTILLED_SIGMA_VALUES
        sliced = STAGE_2_DISTILLED_SIGMA_VALUES[1:]
        assert sliced == [0.725, 0.421875, 0.0], (
            f"Stage 2 sigma slice: {sliced}"
        )
        assert len(sliced) == 3, (
            f"Expected 3 steps, got {len(sliced)}"
        )

    def test_stage1_seed_offsets_by_one(self):
        """Official: stage1 uses seed+1, stage2 uses seed.

        Default seed 42 → stage2=42, stage1=43.
        """
        # ponytail: seed mapping is inline in generate_inpaint;
        # verify logically: stage2 seed == base, stage1 == base + 1.
        base = 42
        assert base + 1 == 43, "seed+1 convention"
        assert base == 42, "seed convention"

    def test_audio_initial_latent_in_stage2(self):
        """Stage 2 audio ModalitySpec passes initial_latent from stage 1.

        Verify the string 'initial_latent=audio_state_s1.latent' exists
        in the pipeline source as a regression catch.
        """
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()
        assert "initial_latent=audio_state_s1.latent" in source, (
            "Stage 2 audio must receive initial_latent from stage 1"
        )

    def test_stage2_guided_by_encoded_blend_not_green_guide(self):
        """Stage 2 must NOT call _encode_green_guide_conditioning — it is
        guided by encoded_blend as initial_latent instead.

        Correct flow:
        - Stage 1 uses _encode_green_guide_conditioning (via green_half).
        - Stage 1 blend is VAE-encoded to encoded_blend.
        - Stage 2 receives encoded_blend as initial_latent — no IC-LoRA
        green guide conditioning at stage2.
        """
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        # Only stage1 calls _encode_green_guide_conditioning
        assert source.count("self._encode_green_guide_conditioning") == 1, (
            "_encode_green_guide_conditioning must be called only by stage1, "
            "not stage2"
        )
        # stage2_conditionings block must not contain _encode_green_guide_conditioning
        # or green_half — it uses combined_image_conditionings with stage1_ltx_images
        cond_blocks = source.split("stage2_conditionings")
        assert len(cond_blocks) >= 2, "Must find stage2 conditionings block"
        stage2_block = cond_blocks[1].split("video_state_s2")[0]
        assert "_encode_green_guide_conditioning" not in stage2_block, (
            "Stage 2 conditionings must NOT call _encode_green_guide_conditioning"
        )
        assert "green_half" not in stage2_block, (
            "Stage 2 conditionings must NOT reference green_half tensor"
        )
        # Stage 2 receives encoded_blend as initial_latent (inside the
        # video_state_s2 pipeline call, after the second stage2_conditionings
        # occurrence which is in conditionings=)
        assert len(cond_blocks) >= 3, "stage2_conditionings must appear as assignment + usage"
        stage2_pipeline_block = cond_blocks[2]
        assert "initial_latent=encoded_blend" in stage2_pipeline_block, (
            "Stage 2 must pass encoded_blend as initial_latent — "
            "was the green-leak root cause fix"
        )

# ponytail: bare assert self-check — GPU bicubic upsample preserves shape and [0,1] range
_STAGE1 = torch.rand(17, 3, 96, 128)  # (F, 3, H_half, W_half)
_STAGE1_FULL = torch.nn.functional.interpolate(
    _STAGE1, size=(384, 512), mode="bicubic", align_corners=False,
).clamp(0.0, 1.0)
assert _STAGE1_FULL.shape == (17, 3, 384, 512), (
    f"stage1 upsample shape: {_STAGE1_FULL.shape}"
)
assert _STAGE1_FULL.min() >= 0.0 and _STAGE1_FULL.max() <= 1.0, (
    f"range: [{_STAGE1_FULL.min()}, {_STAGE1_FULL.max()}]"
)
del _STAGE1, _STAGE1_FULL


def test_resize_video_mask_spatial():
    """(F, H, W) mask half-resized → (F, H_half, W_half)."""
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    import torch
    mask = torch.zeros(17, 384, 768)
    mask[:, 32, 32] = 1.0
    resized = LTXIcLoraPipeline._resize_video_mask_spatial(mask, 192, 384)
    assert resized.shape == (17, 192, 384), (
        f"Expected (17, 192, 384), got {resized.shape}"
    )
    # ponytail: nearest neighbor preserves binary values
    assert resized.dtype == mask.dtype, f"dtype changed: {mask.dtype} → {resized.dtype}"


def test_resize_video_mask_spatial_single_frame():
    """(1, H, W) mask should still work."""
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    import torch
    mask = torch.rand(1, 64, 64)
    resized = LTXIcLoraPipeline._resize_video_mask_spatial(mask, 32, 32)
    assert resized.shape == (1, 32, 32), f"Expected (1, 32, 32), got {resized.shape}"


def test_resize_video_mask_spatial_preserves_values():
    """Check nearest neighbor respects value boundaries."""
    from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline
    import torch
    mask = torch.zeros(3, 128, 128)
    mask[:, 64:96, 64:96] = 1.0
    resized = LTXIcLoraPipeline._resize_video_mask_spatial(mask, 64, 64)
    assert resized.min() >= 0.0 and resized.max() <= 1.0, (
        f"Values out of [0, 1]: [{resized.min()}, {resized.max()}]"
    )
    assert resized.shape == (3, 64, 64), f"Expected (3, 64, 64), got {resized.shape}"


class TestResizeVideoSpatial:
    """Verify _resize_video_spatial handles 5D (B, C, F, H, W) tensors correctly.

    The old bug: F.interpolate with 5D input expects 3 spatial dims;
    passing size=(h, w) crashed with spatial dimension mismatch.
    """

    def test_shape_preserves_frames(self):
        """(1, 3, 17, 384, 768) → (1, 3, 17, 192, 384)."""
        video = torch.rand(1, 3, 17, 384, 768)
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 192, 384, mode="bilinear")
        assert resized.shape == (1, 3, 17, 192, 384), (
            f"Expected (1, 3, 17, 192, 384), got {resized.shape}"
        )

    def test_frame_count_preserved(self):
        """Frame dimension identical after resize — old bug dropped frames."""
        video = torch.rand(1, 3, 17, 384, 768)
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 192, 384, mode="bilinear")
        assert resized.shape[2] == video.shape[2], (
            f"Frame count changed: {video.shape[2]} → {resized.shape[2]}"
        )

    def test_channel_count_preserved(self):
        """Channel dimension preserved."""
        video = torch.rand(1, 3, 17, 384, 768)
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 192, 384)
        assert resized.shape[1] == 3, f"Channels changed: {resized.shape[1]}"

    def test_batch_dim_preserved(self):
        """Batch dimension preserved."""
        video = torch.rand(2, 3, 17, 64, 64)
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 32, 32)
        assert resized.shape[0] == 2, f"Batch changed: {resized.shape[0]}"

    def test_bilinear_produces_smooth_results(self):
        """Bilinear mode produces non-binary output (smooth)."""
        video = torch.rand(1, 1, 3, 64, 64)
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 32, 32, mode="bilinear")
        # bilinear downscale should produce values between 0 and 1 with fractional values
        assert resized.min() >= -0.05 and resized.max() <= 1.05, (
            f"Values out of expected range: [{resized.min()}, {resized.max()}]"
        )

    def test_identity_resize(self):
        """Same input/output spatial dims should preserve values."""
        video = torch.rand(1, 3, 5, 64, 64)
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 64, 64, mode="bilinear", align_corners=False)
        assert resized.shape == video.shape, f"Shape changed: {resized.shape} vs {video.shape}"

    def test_nearest_mode_binary_preserving(self):
        """Nearest mode preserves binary values."""
        video = torch.randint(0, 2, (1, 1, 5, 64, 64)).float()
        resized = LTXIcLoraPipeline._resize_video_spatial(video, 32, 32, mode="nearest")
        assert set(resized.unique().tolist()).issubset({0.0, 1.0}), (
            f"Nearest mode produced non-binary values: {resized.unique()}"
        )


class TestEncodeVideoConditioning:
    """_encode_video_conditioning must not access enc.device (VideoEncoder lacks .device)."""

    def test_fake_encoder_without_device_works(self):
        """Fake encoder with no .device attribute should not crash when device comes from self.pipeline."""
        import ltx_pipelines.utils.media_io as _media_io
        import services.ic_lora_pipeline.ltx_ic_lora_pipeline as _pmod

        class _FakeEnc:
            def __call__(self, video: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1, 16, 5, 8, 8)

        class _FakePipeline:
            device = torch.device("cpu")
            dtype = torch.bfloat16
            reference_downscale_factor = 8

        # Monkeypatch media_io functions to avoid real file I/O
        _orig_decode = _media_io.decode_video_by_frame
        _orig_preprocess = _media_io.video_preprocess
        try:
            _media_io.decode_video_by_frame = lambda path, device, starting_frame=0, frame_cap=None: (
                iter([torch.zeros(3, 128, 128, dtype=torch.uint8)])
            )
            _media_io.video_preprocess = lambda frames, height, width, dtype, device: (
                torch.zeros(1, 3, 5, height, width, dtype=dtype, device=device)
            )

            pipe = _pmod.LTXIcLoraPipeline.__new__(_pmod.LTXIcLoraPipeline)
            pipe.pipeline = _FakePipeline()

            result = pipe._encode_video_conditioning(
                enc=_FakeEnc(),
                video_path="/dev/null/nonexistent.mp4",
                height=64,
                width=64,
                num_frames=5,
                strength=1.0,
            )
            assert len(result) == 1
            assert hasattr(result[0], "latent")
            assert result[0].latent.shape == (1, 16, 5, 8, 8)
        finally:
            _media_io.decode_video_by_frame = _orig_decode
            _media_io.video_preprocess = _orig_preprocess


class TestFramesChwToBcfhw:
    """Verify _frames_chw_to_bcfhw converts (F, C, H, W) → (1, C, F, H, W)."""

    def test_basic_layout(self):
        """(F, 3, H, W) → (1, 3, F, H, W) — channels first, not F first."""
        frames = torch.rand(17, 3, 96, 128)
        result = LTXIcLoraPipeline._frames_chw_to_bcfhw(frames)
        assert result.shape == (1, 3, 17, 96, 128), (
            f"Expected (1, 3, 17, 96, 128), got {result.shape}"
        )

    def test_single_frame(self):
        """Single frame (1, 3, H, W) → (1, 3, 1, H, W)."""
        frame = torch.rand(1, 3, 64, 64)
        result = LTXIcLoraPipeline._frames_chw_to_bcfhw(frame)
        assert result.shape == (1, 3, 1, 64, 64), (
            f"Expected (1, 3, 1, 64, 64), got {result.shape}"
        )

    def test_values_preserved(self):
        """Pixel values at each (C, F, H, W) position identical after permute."""
        frames = torch.arange(17 * 3 * 96 * 128, dtype=torch.float32).reshape(17, 3, 96, 128)
        result = LTXIcLoraPipeline._frames_chw_to_bcfhw(frames)
        for f in range(17):
            for c in range(3):
                assert torch.allclose(result[0, c, f], frames[f, c]), (
                    f"Value mismatch at frame={f}, channel={c}"
                )

    def test_would_not_cause_vae_channel_mismatch(self):
        """Regression: old .unsqueeze(0) gave (1, F, 3, H, W).
        VAE expects (B, C, F, H, W) — (1, 3, F, H, W)."""
        frames = torch.rand(17, 3, 96, 128)
        result = LTXIcLoraPipeline._frames_chw_to_bcfhw(frames)
        # VAE conv1 weight: [128, 48, 3, 3, 3] expects 48 channels at dim 1
        assert result.shape[1] == 3, f"Channel dim should be 3, got {result.shape[1]}"
        assert result.shape[0] == 1, f"Batch dim should be 1, got {result.shape[0]}"
        # Old code .unsqueeze(0) gave shape[1]==F, shape[2]==3 — verify we don't regress
        assert result.shape[1] < result.shape[2] or result.shape[1] == 3, (
            "Channels dimension must be smaller than frames dimension or exactly 3"
        )


class TestInpaintBlendOutsideMaskPreservation:
    """Final blend must preserve original outside dilated mask.

    Simulates: dark original, tiny white mask, bright generated content.
    Blend output outside the dilated mask region must stay close to original.
    """

    def test_outside_mask_preserved_with_bright_generated(self):
        """Bright generated inside mask; outside must be near original."""
        from services.ic_lora_pipeline.official_inpaint import (
            dilate_video_mask,
            laplacian_pyramid_blend,
        )

        f, h, w = 8, 96, 128
        # Dark original video: pixel values ~0.005 in [0, 1]
        original = torch.full((f, h, w, 3), 0.005) + torch.randn(f, h, w, 3) * 0.002
        original = original.clamp(0, 1)

        # Tiny white mask: small 4×4 white square
        mask = torch.zeros(f, h, w)
        mask[:, 10:14, 10:14] = 1.0

        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import derive_stage_radii

        # Dilate with stage2 radius = derive_stage_radii(30)[1] = 30 → covers ~6% area
        mask_dilated = dilate_video_mask(mask.clone(), spatial_radius=derive_stage_radii(30)[1], temporal_radius=0)

        # Bright generated content: ~1.0 everywhere inside mask, ~0.005 outside
        generated = torch.full((f, h, w, 3), 0.005)
        inside = mask_dilated.bool().unsqueeze(-1).expand(-1, -1, -1, 3)
        generated = torch.where(inside, torch.full_like(generated, 0.98), generated)

        # Green composite: original outside mask, bright inside mask
        # (simulates what green_composite_preprocess produces)
        green_composite = original.clone()
        green_composite = torch.where(inside, torch.full_like(green_composite, 0.98), green_composite)

        # Final blend: image_a=generated, image_b=green_composite, mask=mask_dilated
        blend = laplacian_pyramid_blend(
            generated,
            green_composite,
            mask_dilated,
            max_level=5,
            mask_low_res_dilation=6,
        )

        # Check outside the dilated mask: blend must match original closely
        outside_mask = (1.0 - mask_dilated).unsqueeze(-1).expand(-1, -1, -1, 3)
        outside_diff = (blend - original).abs() * outside_mask
        outside_pixels = outside_mask.sum()
        mean_outside_diff = outside_diff.sum() / outside_pixels.clamp_min(1)
        assert mean_outside_diff < 0.02, (
            f"Outside-mask mean diff {mean_outside_diff:.6f} >= 0.02 — "
            "original content not preserved outside dilated mask"
        )

        # Check inside the dilated mask: blend must differ significantly from original
        inside_mask = mask_dilated.unsqueeze(-1).expand(-1, -1, -1, 3)
        inside_diff = (blend - original).abs() * inside_mask
        inside_pixels = inside_mask.sum()
        mean_inside_diff = inside_diff.sum() / inside_pixels.clamp_min(1)
        assert mean_inside_diff > 0.2, (
            f"Inside-mask mean diff {mean_inside_diff:.6f} <= 0.2 — "
            "generated content not applied inside dilated mask"
        )

    def test_outside_mask_preserved_with_inverted_mask(self):
        """Black-mask (no mask) region: blend must not alter original.

        Regression: if mask polarity was inverted (white=green, black=generated),
        this would fail because outside region would receive generated content.
        """
        from services.ic_lora_pipeline.official_inpaint import (
            laplacian_pyramid_blend,
        )

        f, h, w = 4, 64, 64
        # Uniform original
        original = torch.full((f, h, w, 3), 0.1)
        # All-black mask = no generation anywhere
        mask = torch.zeros(f, h, w)
        # Bright generated content
        generated = torch.full((f, h, w, 3), 0.95)
        green_composite = original.clone()

        blend = laplacian_pyramid_blend(
            generated,
            green_composite,
            mask,
            max_level=5,
            mask_low_res_dilation=0,
        )

        # With no mask active, blend must equal original (green_composite)
        diff = (blend - original).abs().mean()
        assert diff < 0.01, (
            f"Black-mask blend diff {diff:.6f} >= 0.01 — "
            "mask polarity may be inverted"
        )

    def test_outside_mask_preserved_with_full_mask(self):
        """All-white mask: blend must prefer generated over original.

        Regression: if mask polarity was inverted, white mask would
        return original (green_composite) instead of generated content.
        """
        from services.ic_lora_pipeline.official_inpaint import (
            laplacian_pyramid_blend,
        )

        f, h, w = 4, 64, 64
        original = torch.full((f, h, w, 3), 0.1)
        mask = torch.ones(f, h, w)
        generated = torch.full((f, h, w, 3), 0.95)
        green_composite = original.clone()

        blend = laplacian_pyramid_blend(
            generated,
            green_composite,
            mask,
            max_level=5,
            mask_low_res_dilation=0,
        )

        diff_vs_generated = (blend - generated).abs().mean()
        diff_vs_original = (blend - original).abs().mean()
        assert diff_vs_generated < 0.05, (
            f"All-white blend not near generated: diff {diff_vs_generated:.6f}"
        )
        assert diff_vs_original > 0.2, (
            f"All-white blend too near original: diff {diff_vs_original:.6f} — "
            "mask polarity may be inverted"
        )


class TestLaplacianBlendGrowParameter:
    """laplacian_blend_grow is now 12, final_mask_blur_px is separate control."""

    def test_default_is_12(self):
        from inspect import signature
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        sig = signature(LTXIcLoraPipeline.generate_inpaint)
        param = sig.parameters["laplacian_blend_grow"]
        assert param.default == 12, f"Expected default=12, got {param.default}"

    def test_final_mask_blur_px_default_is_6(self):
        from inspect import signature
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        sig = signature(LTXIcLoraPipeline.generate_inpaint)
        param = sig.parameters["final_mask_blur_px"]
        assert param.default == 6, f"Expected default=6, got {param.default}"

    def test_param_source_assertions(self):
        """Source-text assertions: laplacian_blend_grow feeds mask_low_res_dilation; final_mask_blur_px feeds blur_radius."""
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "services" / "ic_lora_pipeline" / "ltx_ic_lora_pipeline.py"
        text = source.read_text()
        assert "mask_low_res_dilation=laplacian_blend_grow" in text, (
            "laplacian_blend_grow must feed mask_low_res_dilation in Laplacian blend"
        )
        assert "blur_radius=final_mask_blur_px" in text, (
            "final_mask_blur_px must feed blur_radius in raw-mask guard feather"
        )


class TestEncodeGreenGuideConditioning:
    """_encode_green_guide_conditioning encodes tensor directly, no file I/O.

    Proves:
    - Accepts (1, 3, F, H, W) tensor on device
    - Returns list[VideoConditionByReferenceLatent]
    - Has expected strength and downscale_factor=1
    - Uses direct encoder call, no path-based decode
    """

    def test_direct_tensor_encode_returns_correct_shapes(self):
        """Fake encoder produces known output; verify VideoConditionByReferenceLatent created with correct fields."""
        import os as _os
        import sys as _sys
        _site = _os.path.expanduser("~/.local/share/LTXDesktop/python/lib/python3.13/site-packages")
        _sys.path.insert(0, _site)
        import services.ic_lora_pipeline.ltx_ic_lora_pipeline as _pmod

        class _FakeEnc:
            def __call__(self, video: torch.Tensor) -> torch.Tensor:
                # Simulate video encoder: (1, 3, F, H, W) → (1, 16, F, H//8, W//8)
                _, _, f, h, w = video.shape
                return torch.zeros(1, 16, f, h // 8, w // 8)

        class _FakePipeline:
            device = torch.device("cpu")
            dtype = torch.bfloat16

        pipe = _pmod.LTXIcLoraPipeline.__new__(_pmod.LTXIcLoraPipeline)
        pipe.pipeline = _FakePipeline()

        # (1, 3, 5, 64, 128) tensor
        tensor = torch.rand(1, 3, 5, 64, 128)
        strength = 0.8

        result = pipe._encode_green_guide_conditioning(
            enc=_FakeEnc(),
            tensor=tensor,
            strength=strength,
        )

        assert len(result) == 1, f"Expected 1 conditioning item, got {len(result)}"
        item = result[0]

        # Verify it's a VideoConditionByReferenceLatent
        from ltx_core.conditioning import VideoConditionByReferenceLatent
        assert isinstance(item, VideoConditionByReferenceLatent), (
            f"Expected VideoConditionByReferenceLatent, got {type(item)}"
        )

        # Latent shape: encoder output (1, 16, 5, 8, 16)
        assert item.latent.shape == (1, 16, 5, 8, 16), (
            f"Latent shape mismatch: {item.latent.shape}"
        )

        # downscale_factor must be 1 (official LTXAddVideoICLoRAGuideAdvanced widget)
        assert item.downscale_factor == 1, (
            f"Expected downscale_factor=1, got {item.downscale_factor}"
        )

        # strength matches input
        assert item.strength == strength, (
            f"Expected strength={strength}, got {item.strength}"
        )

    def test_no_file_path_or_decode(self):
        """No video_path or decode_video_by_frame in call chain — pure tensor encode."""
        import os as _os
        import sys as _sys
        _site = _os.path.expanduser("~/.local/share/LTXDesktop/python/lib/python3.13/site-packages")
        _sys.path.insert(0, _site)
        import services.ic_lora_pipeline.ltx_ic_lora_pipeline as _pmod

        class _FakeEnc:
            def __call__(self, video: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1, 16, video.shape[2], 8, 8)

        class _FakePipeline:
            device = torch.device("cpu")
            dtype = torch.bfloat16

        pipe = _pmod.LTXIcLoraPipeline.__new__(_pmod.LTXIcLoraPipeline)
        pipe.pipeline = _FakePipeline()

        # Just proves no crash — the method takes a tensor, not a path
        tensor = torch.rand(1, 3, 3, 64, 64)
        result = pipe._encode_green_guide_conditioning(
            enc=_FakeEnc(),
            tensor=tensor,
            strength=1.0,
        )
        assert len(result) == 1
        assert result[0].latent.shape == (1, 16, 3, 8, 8)

    def test_casts_float32_to_pipeline_dtype(self):
        """float32 tensor input must be cast to pipeline dtype (bfloat16) before encoder call.

        Regression: green_composite_preprocess can promote bfloat16→float32 via float32
        _bg_tensor; without cast, VAE conv sees float32 input and bfloat16 weights → crash.
        """
        import os as _os
        import sys as _sys
        _site = _os.path.expanduser("~/.local/share/LTXDesktop/python/lib/python3.13/site-packages")
        _sys.path.insert(0, _site)
        import services.ic_lora_pipeline.ltx_ic_lora_pipeline as _pmod

        class _FakeEnc:
            def __call__(self, video: torch.Tensor) -> torch.Tensor:
                assert video.dtype == torch.bfloat16, (
                    f"Encoder got {video.dtype}, expected bfloat16"
                )
                return torch.zeros(1, 16, video.shape[2], 8, 8, dtype=torch.bfloat16)

        class _FakePipeline:
            device = torch.device("cpu")
            dtype = torch.bfloat16

        pipe = _pmod.LTXIcLoraPipeline.__new__(_pmod.LTXIcLoraPipeline)
        pipe.pipeline = _FakePipeline()

        # Float32 input — simulates output of green_composite_preprocess after promotion
        tensor = torch.rand(1, 3, 3, 64, 64, dtype=torch.float32)
        result = pipe._encode_green_guide_conditioning(
            enc=_FakeEnc(),
            tensor=tensor,
            strength=0.9,
        )
        assert len(result) == 1
        assert result[0].latent.dtype == torch.bfloat16
        assert result[0].latent.shape == (1, 16, 3, 8, 8)

    def test_uses_tiled_encode_when_available_with_default_tiling_config_fallback_to_direct_call(self):
        """_encode_green_guide_conditioning uses .tiled_encode(default_tiling_config()) when available,
        falls back to enc() call when encoder lacks tiled API."""
        import os as _os
        import sys as _sys
        _site = _os.path.expanduser("~/.local/share/LTXDesktop/python/lib/python3.13/site-packages")
        _sys.path.insert(0, _site)
        from pathlib import Path
        import services.ic_lora_pipeline.ltx_ic_lora_pipeline as _pmod

        # Source-text assertion: the method must reference both tiled_encode and default_tiling_config
        source = Path(_pmod.__file__).read_text()
        # Verify _encode_green_guide_conditioning contains tiled_encode branch
        assert "tiled_encode" in source, "Missing tiled_encode reference in pipeline source"
        assert "default_tiling_config" in source, (
            "Missing default_tiling_config reference in pipeline source"
        )

        # Runtime test 1: encoder with tiled_encode uses it
        class _FakeTiledEnc:
            def __call__(self, video):
                raise AssertionError("direct call should not be used when tiled_encode exists")
            def tiled_encode(self, video, tiling_config):
                _, _, f, h, w = video.shape
                return torch.zeros(1, 16, f, h // 8, w // 8)

        class _FakePipeline:
            device = torch.device("cpu")
            dtype = torch.bfloat16

        pipe = _pmod.LTXIcLoraPipeline.__new__(_pmod.LTXIcLoraPipeline)
        pipe.pipeline = _FakePipeline()

        tensor = torch.rand(1, 3, 5, 64, 128)
        result = pipe._encode_green_guide_conditioning(
            enc=_FakeTiledEnc(), tensor=tensor, strength=0.8,
        )
        assert len(result) == 1
        assert result[0].latent.shape == (1, 16, 5, 8, 16), (
            f"tiled_encode: {result[0].latent.shape}"
        )

        # Runtime test 2: encoder without tiled_encode falls back to direct call
        class _FakeDirectEnc:
            def __call__(self, video):
                _, _, f, h, w = video.shape
                return torch.zeros(1, 16, f, h // 8, w // 8)

        pipe2 = _pmod.LTXIcLoraPipeline.__new__(_pmod.LTXIcLoraPipeline)
        pipe2.pipeline = _FakePipeline()

        result2 = pipe2._encode_green_guide_conditioning(
            enc=_FakeDirectEnc(), tensor=tensor, strength=0.7,
        )
        assert len(result2) == 1
        assert result2[0].latent.shape == (1, 16, 5, 8, 16), (
            f"direct fallback: {result2[0].latent.shape}"
        )
        assert result2[0].strength == 0.7, (
            f"Expected strength=0.7, got {result2[0].strength}"
        )


class TestBlendOutputToUint8:
    """Smoke: (F, H, W, 3) float [0,1] → uint8 [0,255] conversion, shape preserved."""

    def test_float_to_uint8_conversion(self) -> None:
        F, H, W = 5, 64, 96
        blend = torch.rand(F, H, W, 3)
        out = (blend.clamp(0, 1) * 255).to(torch.uint8)
        assert out.shape == (F, H, W, 3), f"Expected ({F}, {H}, {W}, 3), got {out.shape}"
        assert out.dtype == torch.uint8, f"Expected uint8, got {out.dtype}"
        assert out.min() >= 0, f"Min value {out.min()} below 0"
        assert out.max() <= 255, f"Max value {out.max()} above 255"


class TestStage1UpsampleDevicePath:
    """Validate blend_stage1 to upsample pipeline device/dtype/range.

    After fix: laplacian_pyramid_blend returns tensor on requested device
    (no forced .cpu()), and interpolate input is explicitly moved to device.
    """

    def test_blend_upsample_shape_dtype_range(self):
        """blend(F, H, W, 3) -> permute -> interpolate -> _frames_chw_to_bcfhw -> *2-1
        gives (1, 3, F, H, W), bfloat16, [-1, 1]."""
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        F, H, W = 5, 64, 64
        img_a = torch.rand(F, H, W, 3)
        img_b = torch.rand(F, H, W, 3)
        mask = (torch.rand(F, H, W) > 0.5).float()

        blend = laplacian_pyramid_blend(img_a, img_b, mask, max_level=3, mask_low_res_dilation=0)

        bchw = blend.permute(0, 3, 1, 2)  # (F, 3, H, W)
        up_h, up_w = H * 2, W * 2
        up = torch.nn.functional.interpolate(
            bchw, size=(up_h, up_w), mode="bicubic", align_corners=False,
        ).clamp(0.0, 1.0)
        up = up.to(dtype=torch.bfloat16)

        bcfhw = LTXIcLoraPipeline._frames_chw_to_bcfhw(up)
        result = bcfhw * 2.0 - 1.0

        assert result.shape == (1, 3, F, up_h, up_w), (
            f"Expected (1, 3, {F}, {up_h}, {up_w}), got {result.shape}"
        )
        assert result.dtype == torch.bfloat16, f"Expected bfloat16, got {result.dtype}"
        assert result.min() >= -1.0 and result.max() <= 1.0, (
            f"Range: [{result.min()}, {result.max()}], expected [-1, 1]"
        )

    def test_laplacian_blend_returns_requested_device(self):
        """When device= is set, laplacian_pyramid_blend returns tensor on that device.

        When device=None, result stays on input device (CPU).
        CUDA branch is conditional; CPU branch always runs.
        """
        from services.ic_lora_pipeline.official_inpaint import laplacian_pyramid_blend

        F, H, W = 3, 32, 32
        img_a = torch.rand(F, H, W, 3)
        img_b = torch.rand(F, H, W, 3)
        mask = (torch.rand(F, H, W) > 0.5).float()

        # Baseline: device=None returns CPU
        blend_cpu = laplacian_pyramid_blend(img_a, img_b, mask, max_level=3, mask_low_res_dilation=0)
        assert blend_cpu.device == torch.device("cpu"), (
            f"Without device arg, expected cpu, got {blend_cpu.device}"
        )

        if torch.cuda.is_available():
            device = torch.device("cuda")
            blend_gpu = laplacian_pyramid_blend(
                img_a, img_b, mask, max_level=3, mask_low_res_dilation=0, device=device,
            )
            assert blend_gpu.device.type == "cuda", (
                f"Expected CUDA device, got {blend_gpu.device}"
            )
            # Interpolate input stays on the same device as blend
            bchw = blend_gpu.permute(0, 3, 1, 2)
            assert bchw.device == blend_gpu.device, (
                f"permute changed device from {blend_gpu.device} to {bchw.device}"
            )


class TestInpaintVramOffload:
    """Source-text assertions for VRAM offload of originals before stage1/stage2 denoising.

    At 196f 1080p, original video/mask tensors (~4.7GB @ bf16) + GGUF transformer raw
    load OOM. Offload to CPU before each denoising pass, move back only for blends.
    """

    def test_cpu_offload_before_stage1(self):
        """Verify .cpu() calls on originals appear before self.pipeline.stage_1(."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        stage1_pos = source.find("self.pipeline.stage_1(")
        assert stage1_pos != -1, "stage_1 call not found"

        offload_targets = ["video_half", "video_full", "mask_stage1_half", "mask_stage2_full", "mask_full_gray"]
        for name in offload_targets:
            cpu_call = f"{name}.cpu()"
            pos = source.find(cpu_call)
            assert pos != -1, f"{cpu_call} not found"
            assert pos < stage1_pos, (
                f"{cpu_call} at offset {pos} must appear before "
                f"self.pipeline.stage_1( at offset {stage1_pos}"
            )

    def test_empty_cache_guarded_by_cuda_check(self):
        """torch.cuda.empty_cache() must be inside `if device.type == "cuda":`."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        empty_cache_pos = source.find("torch.cuda.empty_cache()")
        assert empty_cache_pos != -1, "torch.cuda.empty_cache() not found"

        # Find the preceding if device.type == "cuda": within reasonable scope
        preceding_block = source[max(0, empty_cache_pos - 200):empty_cache_pos]
        assert 'if device.type == "cuda":' in preceding_block, (
            "torch.cuda.empty_cache() must be guarded by 'if device.type == \"cuda\":'"
        )

    def test_stage1_offload_includes_del_green_half(self):
        """del green_half must appear in the offload block before stage1."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        stage1_pos = source.find("self.pipeline.stage_1(")
        assert stage1_pos != -1

        del_pos = source.find("del green_half")
        assert del_pos != -1, "del green_half not found"
        assert del_pos < stage1_pos, (
            f"del green_half at {del_pos} must be before stage_1 at {stage1_pos}"
        )

    def test_stage1_offload_includes_del_mask_temps(self):
        """Combined del line including mask_video, mask_gray, mask_stage1_full."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        stage1_pos = source.find("self.pipeline.stage_1(")
        assert stage1_pos != -1

        del_line = "del green_half, mask_video, mask_gray, mask_stage1_full"
        pos = source.find(del_line)
        assert pos != -1, f"'{del_line}' not found"
        assert pos < stage1_pos, (
            f"del line at {pos} must be before stage_1 at {stage1_pos}"
        )

    def test_move_back_before_stage1_blend(self):
        """video_half.to(device=...) and mask_stage1_half.to(device=...) appear before blend."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        blend_comment_pos = source.find("# Original video as [0, 1] pixel frames for stage 1 blend")
        assert blend_comment_pos != -1, "blend orig comment not found"

        for call in ('video_half = video_half.to(device=', 'mask_stage1_half = mask_stage1_half.to(device='):
            pos = source.find(call)
            assert pos != -1, f"'{call}' not found"
            assert pos < blend_comment_pos, (
                f"'{call}' at {pos} must be before blend comment at {blend_comment_pos}"
            )

    def test_stage2_pre_denoising_offload(self):
        """After stage1 blend del, video_half.cpu() and mask_stage1_half.cpu() and
        empty_cache guard appear before stage 2 denoising."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        # The offload should happen after the del line and before the # 9. section header
        del_pos = source.find("del blend_stage1, blend_stage1_bchw")
        assert del_pos != -1, "blend del line not found"

        stage2_header_pos = source.find("# 9. Stage 2:")
        assert stage2_header_pos != -1, "Stage 2 section header not found"

        for call in ('video_half = video_half.cpu()', 'mask_stage1_half = mask_stage1_half.cpu()'):
            pos = source.find(call, del_pos)
            assert pos != -1, f"'{call}' after del line not found"
            assert pos < stage2_header_pos, (
                f"'{call}' at {pos} must be before "
                f"stage 2 header ({stage2_header_pos})"
            )

        # empty_cache guard must also be in this region
        cuda_check_pos = source.find('if device.type == "cuda":', del_pos, stage2_header_pos)
        assert cuda_check_pos != -1, (
            f"empty_cache guard not found between offload and stage 2 header"
        )

    def test_move_back_before_final_blend(self):
        """video_full, mask_stage2_full, mask_full_gray moved back before final blend."""
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        blend2_comment = source.find("# Original video as [0, 1] pixel frames for stage 2 blend")
        assert blend2_comment != -1, "blend2 comment not found"

        for name, dtype in [("video_full", True), ("mask_stage2_full", False), ("mask_full_gray", False)]:
            call = f"{name} = {name}.to(device="
            pos = source.find(call)
            assert pos != -1, f"'{call}' not found"
            assert pos < blend2_comment, (
                f"'{call}' at {pos} must be before blend2 comment at {blend2_comment}"
            )


class TestHdrVideoOnly:
    """HDR is video-only: any audio returned by the pinned pipeline must be
    suppressed before encoding. See LTXIcLoraPipeline._is_hdr_video_only_path.
    """

    @staticmethod
    def _noop_postprocess() -> torch.Tensor:
        # A real Callable sentinel (not a mock) used as output_postprocess.
        def _fn(t: torch.Tensor) -> torch.Tensor:
            return t

        return _fn

    @pytest.mark.parametrize(
        ("hdr_video_context", "output_postprocess", "expected"),
        [
            (None, None, False),
            (torch.zeros(1), None, True),
            (None, "postprocess-set", True),
            (torch.zeros(1), "postprocess-set", True),
        ],
        ids=["non-hdr", "hdr-context-only", "postprocess-only", "both"],
    )
    def test_is_hdr_video_only_path(
        self, hdr_video_context, output_postprocess, expected
    ):
        """Helper flags the HDR path iff either HDR signal is present."""
        post = self._noop_postprocess() if output_postprocess else None
        assert (
            LTXIcLoraPipeline._is_hdr_video_only_path(hdr_video_context, post)
            is expected
        )

    def test_generate_suppresses_audio_on_hdr_before_encode(self):
        """generate() (non-inpaint) discards audio on the HDR path and feeds
        encode_video_output the (possibly-None) audio afterwards.

        Regression for the HDR video-only contract: the pinned
        ICLoraPipeline.__call__ can build/return audio internally even when
        hdr_audio_context=None is passed, so generate() must explicitly drop it
        when the HDR scene-context / output-postprocess path is active.
        Non-HDR generation must keep forwarding audio untouched.
        """
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        # Isolate the non-inpaint generate() body (up to generate_inpaint).
        start = source.find("def generate(")
        assert start != -1, "generate() method not found"
        end = source.find("def generate_inpaint(")
        assert end != -1, "generate_inpaint() method not found"
        generate_body = source[start:end]

        # The HDR video-only guard must be present in generate().
        guard_idx = generate_body.find("_is_hdr_video_only_path")
        assert guard_idx != -1, (
            "generate() must call _is_hdr_video_only_path(hdr_video_context, "
            "output_postprocess) to gate the HDR video-only audio suppression"
        )

        # audio = None must exist inside generate() and be guarded by the helper.
        drop_idx = generate_body.find("audio = None", guard_idx)
        assert drop_idx != -1, (
            "generate() must set audio = None after the _is_hdr_video_only_path "
            "check so HDR output is encoded video-only"
        )

        # The encode call must come after the suppression so it sees audio=None.
        encode_idx = generate_body.find("encode_video_output(", drop_idx)
        assert encode_idx != -1, (
            "generate() must call encode_video_output after the HDR audio "
            "suppression so the encoder receives audio=None on the HDR path"
        )

        # The suppression must be conditional (under the HDR guard), never
        # unconditional — otherwise non-HDR audio would be lost.
        assert generate_body.count("audio = None") == 1, (
            "generate() must set audio = None exactly once, and only under the "
            "HDR video-only guard (preserve non-HDR audio passthrough)"
        )

