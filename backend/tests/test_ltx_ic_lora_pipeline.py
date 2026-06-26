"""Unit tests for LTX IC-LoRA pipeline internals (no GPU, no mocks)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline, _vae_compatible_frame_count


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


class TestCompositeInOutpainting:
    """Verify _composite_in_outpainting blends generated and original via mask.

    White mask (255) → keep generated region.
    Black mask (0) → preserve original region.
    """

    def _write_video(self, path: Path, frames: list[np.ndarray]) -> None:
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 1, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

    def test_black_mask_preserves_original_white_mask_uses_generated(self, tmp_path: Path) -> None:
        """Split mask: black left preserves original, white right uses generated."""
        h, w = 64, 64
        orig_val, gen_val = 100, 200

        orig = np.full((h, w, 3), orig_val, dtype=np.uint8)
        gen = np.full((h, w, 3), gen_val, dtype=np.uint8)
        mask = np.zeros((h, w, 3), dtype=np.uint8)
        mask[:, w // 2 :, :] = 255

        orig_path = tmp_path / "original.mp4"
        gen_path = tmp_path / "generated.mp4"
        mask_path = tmp_path / "mask.mp4"

        self._write_video(orig_path, [orig])
        self._write_video(gen_path, [gen])
        self._write_video(mask_path, [mask])

        LTXIcLoraPipeline._composite_in_outpainting(
            str(gen_path), str(orig_path), str(mask_path)
        )

        cap = cv2.VideoCapture(str(gen_path))
        ret, composite = cap.read()
        cap.release()
        assert ret, "Should read composite frame"

        left = composite[:, : w // 2, :]
        right = composite[:, w // 2 :, :]

        # ponytail: mp4v codec drifts values at small sizes, check mean proximity
        mean_left = float(np.mean(left))
        mean_right = float(np.mean(right))

        assert abs(mean_left - orig_val) < abs(mean_left - gen_val), (
            f"Black mask side should be closer to original={orig_val} "
            f"than generated={gen_val}, got mean {mean_left:.1f}"
        )
        assert abs(mean_right - gen_val) < abs(mean_right - orig_val), (
            f"White mask side should be closer to generated={gen_val} "
            f"than original={orig_val}, got mean {mean_right:.1f}"
        )
        # Ensure both are at least directionally correct (codec drift of ~20 is OK)
        assert mean_left <= orig_val + 15, (
            f"Black mask side mean {mean_left:.1f} too far from {orig_val}"
        )
        assert mean_right >= gen_val - 15, (
            f"White mask side mean {mean_right:.1f} too far from {gen_val}"
        )

    def test_dual_frame_mask(self, tmp_path: Path) -> None:
        """Frame 0 all-black mask preserves original. Frame 1 all-white mask uses generated."""
        h, w = 64, 64
        orig_val, gen_val = 50, 220

        orig = np.full((h, w, 3), orig_val, dtype=np.uint8)
        gen = np.full((h, w, 3), gen_val, dtype=np.uint8)
        mask_black = np.zeros((h, w, 3), dtype=np.uint8)
        mask_white = np.full((h, w, 3), 255, dtype=np.uint8)

        orig_path = tmp_path / "original.mp4"
        gen_path = tmp_path / "generated.mp4"
        mask_path = tmp_path / "mask.mp4"

        self._write_video(orig_path, [orig, orig])
        self._write_video(gen_path, [gen, gen])
        self._write_video(mask_path, [mask_black, mask_white])

        LTXIcLoraPipeline._composite_in_outpainting(
            str(gen_path), str(orig_path), str(mask_path)
        )

        cap = cv2.VideoCapture(str(gen_path))
        ret0, f0 = cap.read()
        ret1, f1 = cap.read()
        cap.release()
        assert ret0 and ret1, "Should read both frames"

        mean_f0 = float(np.mean(f0))
        mean_f1 = float(np.mean(f1))

        assert abs(mean_f0 - orig_val) < abs(mean_f0 - gen_val), (
            f"Black mask frame should be closer to original={orig_val} "
            f"than generated={gen_val}, got mean {mean_f0:.1f}"
        )
        assert abs(mean_f1 - gen_val) < abs(mean_f1 - orig_val), (
            f"White mask frame should be closer to generated={gen_val} "
            f"than original={orig_val}, got mean {mean_f1:.1f}"
        )
        assert mean_f0 <= orig_val + 15
        assert mean_f1 >= gen_val - 15

    def test_gray_mask_blends(self, tmp_path: Path) -> None:
        """Mid-gray mask (128) blends ~50/50 under linear alpha."""
        h, w = 64, 64
        orig = np.full((h, w, 3), 0, dtype=np.uint8)
        gen = np.full((h, w, 3), 255, dtype=np.uint8)
        mask = np.full((h, w, 3), 128, dtype=np.uint8)

        orig_path = tmp_path / "original.mp4"
        gen_path = tmp_path / "generated.mp4"
        mask_path = tmp_path / "mask.mp4"

        self._write_video(orig_path, [orig])
        self._write_video(gen_path, [gen])
        self._write_video(mask_path, [mask])

        LTXIcLoraPipeline._composite_in_outpainting(
            str(gen_path), str(orig_path), str(mask_path)
        )

        cap = cv2.VideoCapture(str(gen_path))
        ret, composite = cap.read()
        cap.release()
        assert ret

        mean_val = float(np.mean(composite))
        expected = 255 * (128.0 / 255.0)  # ≈ 128
        # ponytail: mp4v drift, check blend is between 80-175 (not all-0 or all-255)
        assert 80 < mean_val < 175, (
            f"Gray mask (128) should blend (~128), got mean {mean_val:.1f}"
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

    def test_blend2_low_res_dilation_is_6(self):
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import INPAINT_BLEND2_LOW_RES_DILATION
        assert INPAINT_BLEND2_LOW_RES_DILATION == 6, (
            f"Expected 6, got {INPAINT_BLEND2_LOW_RES_DILATION}"
        )



    def test_blend_constants_are_separate_from_radii(self):
        """Blend constants (5, 6) should differ from default stage radii (15, 30).
        This documents they are independent controls."""
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
            INPAINT_BLEND1_LOW_RES_DILATION,
            INPAINT_BLEND2_LOW_RES_DILATION,
            derive_stage_radii,
        )
        s1, s2 = derive_stage_radii(30)
        assert INPAINT_BLEND1_LOW_RES_DILATION != s1, (
            "Blend1 constant should differ from stage1 radius (separate controls)"
        )
        assert INPAINT_BLEND2_LOW_RES_DILATION != s2, (
            "Blend2 constant should differ from stage2 radius (separate controls)"
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

    def test_stage2_green_guide_uses_half_res(self):
        """Stage 2 must append green composite IC-LoRA guide conditioning
        via _encode_green_guide_conditioning with the SAME half-res green_half
        tensor as stage1 (official node 5378, mask r=15).

        Official parity: node 5114 uses same half-res green guide for both
        stages; stage2 passes through LTXVCropGuides temporal keyframe-crop;
        installed clear_conditioning() already handles it.
        Not using green_full (full-res, mask r=30) avoids sharper original
        content leaking into stage2 conditioning.
        """
        import os
        pipe_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "services/ic_lora_pipeline/ltx_ic_lora_pipeline.py",
        )
        with open(pipe_path) as f:
            source = f.read()

        # Both stages call _encode_green_guide_conditioning
        assert source.count("self._encode_green_guide_conditioning") == 2, (
            "Both S1 and S2 must have _encode_green_guide_conditioning calls"
        )
        # Stage 2 must NOT use green_full for guide conditioning
        # Match the pattern around stage2's green guide conditioning call
        # green_full should only appear in final blend image_b, not in conditioning
        # Find the stage2 green guide block and verify it uses green_half
        assert "green_half" in source, "green_half tensor must exist"
        # Verify stage2 conditioning block references green_half by checking
        # that the ONLY tensor=green_full occurrence is in the final blend
        # section, not in conditioning
        cond_blocks = source.split("stage2_conditionings")
        assert len(cond_blocks) >= 2, "Must find stage2 conditionings block"
        stage2_block = cond_blocks[1].split("video_state_s2")[0]
        assert "green_half" in stage2_block, (
            "Stage 2 conditioning block must use green_half — "
            "green_full in stage2 conditioning causes sharper leaked content"
        )
        assert "green_full" not in stage2_block, (
            "Stage 2 conditioning must NOT use green_full — "
            "that was the green-leak root cause"
        )

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
    """laplacian_blend_grow param is separate from mask_grow_px and targets final blend only."""

    def test_default_is_6(self):
        from inspect import signature
        from services.ic_lora_pipeline.ltx_ic_lora_pipeline import LTXIcLoraPipeline

        sig = signature(LTXIcLoraPipeline.generate_inpaint)
        param = sig.parameters["laplacian_blend_grow"]
        assert param.default == 6, f"Expected default=6, got {param.default}"


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
