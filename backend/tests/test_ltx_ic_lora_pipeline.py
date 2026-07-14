"""Unit tests for LTX IC-LoRA pipeline internals (no GPU, no mocks)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import pytest
import torch

from services.ic_lora_pipeline.ltx_ic_lora_pipeline import (
    LTXIcLoraPipeline,
    _vae_compatible_frame_count,
)
from services.ic_lora_pipeline.latent_inpaint import (
    OFFICIAL_NEGATIVE_PROMPT,
    InpaintTargetGuideAttention,
    _stage1_cfg_pp_ancestral_loop,
    _windows,
    blend_half_latents,
    build_inpaint_context_config,
    mask_to_latent_denoise_mask,
)
import services.ic_lora_pipeline.latent_inpaint as latent_inpaint


@dataclass
class _State:
    latent: torch.Tensor
    denoise_mask: torch.Tensor | None = None
    clean_latent: torch.Tensor | None = None
    positions: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None


class _InpaintHarness:
    """Small recording pipeline that runs the real latent-inpaint function on CPU."""
    def __init__(self, monkeypatch: pytest.MonkeyPatch, preview: bool = False, context: bool = False, *, prompt: str = "p", images: list[Any] | None = None, mask_active: bool = True) -> None:
        from ltx_core import conditioning
        from ltx_core.components import noisers
        from ltx_pipelines.utils import constants, denoisers, helpers, media_io, types
        import services.exr_input as exr_input
        self.calls: list[tuple[str, Any]] = []; self.encoded: list[dict[str, Any]] = []
        self.device, self.dtype, self.preview, self.context = torch.device("cpu"), torch.float32, preview, context
        self.prompt, self.images, self.mask_active, self.blend_masks = prompt, images or [], mask_active, []
        self.video_encoding = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]])
        self.audio_encoding = object()
        self.prompt_batches: list[list[str]] = []
        def prompt_encoder(prompts: list[str], **_kwargs: Any) -> tuple[Any, ...]:
            self.prompt_batches.append(prompts)
            return tuple(type("Prompt", (), {
                "video_encoding": self.video_encoding + index,
                "attention_mask": torch.tensor([[1, 1, 0, 0]]),
                "audio_encoding": self.audio_encoding,
            })() for index, _ in enumerate(prompts))
        self.prompt_encoder = prompt_encoder
        self.stage_1 = lambda **kwargs: self._stage1(**kwargs)
        self.stage_2 = lambda **kwargs: self._stage2(**kwargs)
        self.upsampler, self.video_decoder = self._upsample, self._decode
        monkeypatch.setattr(helpers, "assert_resolution", lambda **_: None)
        monkeypatch.setattr(helpers, "combined_image_conditionings", lambda **_: ["anchor"])
        monkeypatch.setattr(exr_input, "iter_video_frames_to_model_domain", lambda *_, **__: iter(["source"]))
        monkeypatch.setattr(media_io, "decode_video_by_frame", lambda **_: iter(["mask"]))
        def video_preprocess(frames: Any, h: int, w: int, *_: Any) -> torch.Tensor:
            kind = next(frames)
            if kind == "source":
                return torch.full((1, 3, 3, h, w), -1.0)
            return torch.full((1, 3, 9, h, w), 1.0 if self.mask_active else -1.0)

        original_blend_half_latents = blend_half_latents

        def record_blend_masks(*args: Any, **kwargs: Any) -> torch.Tensor:
            self.blend_masks.append(args[2])
            return original_blend_half_latents(*args, **kwargs)

        monkeypatch.setattr(media_io, "video_preprocess", video_preprocess)
        monkeypatch.setattr(latent_inpaint, "blend_half_latents", record_blend_masks)
        monkeypatch.setattr(constants, "DISTILLED_SIGMAS", torch.tensor([1.0, 0.0])); monkeypatch.setattr(constants, "STAGE_2_DISTILLED_SIGMAS", torch.tensor([0.909375, 0.725, 0.421875, 0.0]))
        monkeypatch.setattr(denoisers, "SimpleDenoiser", lambda *args: args); monkeypatch.setattr(noisers, "GaussianNoiser", lambda **kwargs: kwargs)
        monkeypatch.setattr(types, "ModalitySpec", lambda **kwargs: type("Spec", (), kwargs)())
        monkeypatch.setattr(conditioning, "VideoConditionByKeyframeIndex", lambda **kwargs: ("keyframe", kwargs))
        monkeypatch.setattr(conditioning, "VideoConditionByMask", lambda **kwargs: ("preserve", kwargs))
        monkeypatch.setattr(latent_inpaint, "make_ltx_image_conditioning_input", lambda *args: args)
        monkeypatch.setattr(latent_inpaint, "encode_video_output", lambda **kwargs: self.encoded.append(kwargs))

    def image_conditioner(self, callback: Any) -> list[Any]:
        outer = self
        class Encoder:
            def tiled_encode(self, video: torch.Tensor, tiling: Any) -> torch.Tensor:
                outer.calls.append(("encode", video.clone()))
                return torch.full((1, 2, 2, 1, 1), 3.0 if len([n for n, _ in outer.calls if n == "encode"]) == 1 else 5.0)
        return callback(Encoder())

    def _stage1(self, **kwargs: Any) -> tuple[_State, None]: self.calls.append(("stage1", kwargs)); return _State(torch.full((1, 2, 2, 1, 1), 7.0)), None
    def _upsample(self, latent: torch.Tensor) -> torch.Tensor:
        self.calls.append(("upsample", latent.clone())); self.stage2_initial = latent.clone(); return self.stage2_initial
    def _stage2(self, **kwargs: Any) -> tuple[_State, None]: self.calls.append(("stage2", kwargs)); return _State(kwargs["video"].initial_latent.clone()), None
    def _decode(self, latent: torch.Tensor, *_: Any) -> list[torch.Tensor]: self.calls.append(("decode", latent.clone())); return [torch.zeros(9, 2, 2, 3)]
    def run(self) -> None:
        latent_inpaint.generate_inpaint(type("Outer", (), {"pipeline": self})(), prompt=self.prompt, seed=4, height=64, width=64, num_frames=9, frame_rate=24, images=self.images, video_path="source", mask_path="mask", output_path="out.mp4", save_stage_1_preview=self.preview, inpaint_context_window_px=33 if self.context else None, inpaint_context_overlap_px=8 if self.context else None)


class TestInpaintMaskConversion:
    def test_all_black(self): assert torch.equal(mask_to_latent_denoise_mask(torch.zeros(9, 32, 32), torch.zeros(1, 2, 2, 1, 1)), torch.zeros(1, 1, 2, 1, 1))
    def test_all_white(self): assert torch.equal(mask_to_latent_denoise_mask(torch.ones(9, 32, 32), torch.zeros(1, 2, 2, 1, 1)), torch.ones(1, 1, 2, 1, 1))
    def test_mask_to_latent_denoise_mask_thresholds_compressed_values_to_binary(self):
        mask = torch.tensor([[[0.49, 0.5, 0.51]]] + [[[0.0, 0.0, 0.0]]] * 4 + [[[1.0, 0.0, 0.0]]] + [[[0.0, 0.0, 0.0]]] * 3)
        result = mask_to_latent_denoise_mask(mask, torch.zeros(1, 2, 2, 1, 3))
        assert result.shape == (1, 1, 2, 1, 3)
        assert torch.equal(result[0, 0, 0, 0], torch.tensor([0.0, 1.0, 1.0]))
        assert result[0, 0, 1, 0, 0] == 1
        assert set(result.unique().tolist()) <= {0.0, 1.0}
    def test_frame_zero_causal_mapping(self): assert mask_to_latent_denoise_mask(torch.cat([torch.ones(1, 2, 2), torch.zeros(8, 2, 2)]), torch.zeros(1, 2, 2, 1, 1))[0, 0, 0, 0, 0] == 1
    def test_later_group_uses_temporal_max(self):
        mask = torch.zeros(9, 2, 2); mask[5] = 1; assert mask_to_latent_denoise_mask(mask, torch.zeros(1, 2, 2, 1, 1))[0, 0, 1, 0, 0] == 1
    def test_exact_8n_plus_1_shape(self): assert mask_to_latent_denoise_mask(torch.zeros(17, 2, 2), torch.zeros(1, 2, 3, 1, 1)).shape == (1, 1, 3, 1, 1)
    def test_half_and_full_mask_direction(self): assert torch.all(blend_half_latents(torch.ones(1, 2, 1, 1, 1), torch.zeros(1, 2, 1, 1, 1), torch.ones(1, 1, 1, 1, 1)) == 1)


class TestInpaintLatentBridge:
    def test_stage1_receives_anchor_and_keyframe_guide(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); s = next(v for n, v in h.calls if n == "stage1")
        guide = s["video"].conditionings[-1]
        assert s["video"].conditionings[0] == "anchor"
        assert isinstance(guide, InpaintTargetGuideAttention)
        inner = guide.inner
        assert inner[0] == "keyframe"
        assert inner[1]["frame_idx"] == 0 and inner[1]["num_pixel_frames"] == 9
        assert inner[1]["strength"] == 1.0
    def test_half_latent_composition(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); assert torch.equal(next(v for n, v in h.calls if n == "upsample"), torch.full((1, 2, 2, 1, 1), 7.0))
    def test_upsampler_called_once_with_exact_blend(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); assert [n for n, _ in h.calls].count("upsample") == 1
    def test_no_intermediate_decode_or_reencode(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); names = [n for n, _ in h.calls]; assert names.index("upsample") < names.index("decode") and names.count("encode") == 2
    def test_stage2_uses_only_the_preserve_conditioning(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run()
        conditionings = next(v for n, v in h.calls if n == "stage2")["video"].conditionings
        assert len(conditionings) == 1 and conditionings[0][0] == "preserve"
        assert conditionings[0][1]["latent"] is h.stage2_initial
        assert conditionings[0][1]["strength"] == 1.0
    def test_green_guide_gate_bypasses_blank_prompt_images_and_empty_mask(self, monkeypatch: pytest.MonkeyPatch):
        wrapped = _InpaintHarness(monkeypatch, mask_active=True); wrapped.run()
        guide = next(v for n, v in wrapped.calls if n == "stage1")["video"].conditionings[-1]
        assert isinstance(guide, InpaintTargetGuideAttention) and guide.target_mask is wrapped.blend_masks[0]
        wrapped_stage1 = next(v for n, v in wrapped.calls if n == "stage1")
        wrapped_stage2 = next(v for n, v in wrapped.calls if n == "stage2")
        ancestral = next(cell.cell_contents for cell in wrapped_stage1["loop"].__closure__ if isinstance(cell.cell_contents, torch.Generator))
        assert wrapped.prompt_batches == [["p", OFFICIAL_NEGATIVE_PROMPT]] and callable(wrapped_stage1["loop"]) and wrapped_stage2["loop"] is None and wrapped_stage1["noiser"]["generator"] is not ancestral
        for prompt, images, active in [("", [], True), ("  ", [], True), ("p", [SimpleNamespace(path="x", frame_idx=0, strength=1.0)], True), ("p", [], False)]:
            harness = _InpaintHarness(monkeypatch, prompt=prompt, images=images, mask_active=active); harness.run()
            guide = next(v for n, v in harness.calls if n == "stage1")["video"].conditionings[-1]
            stage1 = next(v for n, v in harness.calls if n == "stage1")
            stage2 = next(v for n, v in harness.calls if n == "stage2")
            assert guide[0] == "keyframe" and harness.prompt_batches == [[prompt]] and "loop" not in stage1 and stage2["loop"] is None


class TestInpaintTargetGuideAttention:
    class _Tools:
        class target_shape:
            @staticmethod
            def token_count() -> int: return 8
        class patchifier:
            @staticmethod
            def patchify(mask: torch.Tensor) -> torch.Tensor:
                return mask.flatten()[torch.tensor([5, 0, 7, 2, 4, 1, 6, 3])].view(1, 8, 1)

    class _Inner:
        def __init__(self, guide_count: int = 8) -> None: self.guide_count = guide_count
        def apply_to(self, state: _State, _tools: Any) -> _State:
            return _State(torch.cat((state.latent, torch.zeros(1, self.guide_count, 1)), dim=1))

    def test_asymmetric_mask_local_target_to_guide_isolation_in_patchifier_order(self):
        mask = torch.tensor([[[[[0., 1.], [0., 0.]], [[0., 1.], [0., 1.]]]]])
        result = InpaintTargetGuideAttention(self._Inner(), mask).apply_to(_State(torch.arange(8.).view(1, 8, 1)), self._Tools())
        assert result.attention_mask is not None and result.attention_mask.shape == (1, 16, 16)
        masked = torch.tensor([True, False, True, False, False, True, False, False])
        assert torch.all(~result.attention_mask[0, :8, 8:][masked])
        assert torch.all(result.attention_mask[0, 8:, :8])
        assert torch.all(result.attention_mask[0, :8, :8]) and torch.all(result.attention_mask[0, 8:, 8:]) and torch.all(result.attention_mask[0, :8, 8:][~masked])

    @pytest.mark.parametrize("state,mask,inner", [
        (_State(torch.zeros(2, 8, 1)), torch.zeros(1, 1, 2, 2, 2), _Inner()),
        (_State(torch.zeros(1, 8, 1)), torch.zeros(2, 1, 2, 2, 2), _Inner()),
        (_State(torch.zeros(1, 7, 1)), torch.zeros(1, 1, 2, 2, 2), _Inner()),
        (_State(torch.zeros(1, 8, 1), attention_mask=torch.ones(1, 8, 8)), torch.zeros(1, 1, 2, 2, 2), _Inner()),
        (_State(torch.zeros(1, 8, 1)), torch.zeros(1, 1, 1, 1, 1), _Inner()),
        (_State(torch.zeros(1, 8, 1)), torch.zeros(1, 1, 2, 2, 2), _Inner(7)),
    ])
    def test_rejects_invalid_batch_shape_existing_mask_original_or_guide_count(self, state: _State, mask: torch.Tensor, inner: Any):
        with pytest.raises(ValueError): InpaintTargetGuideAttention(inner, mask).apply_to(state, self._Tools())


class TestInpaintStage2:
    def test_accepted_stage2_sigmas_and_noise_scale(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); s = next(v for n, v in h.calls if n == "stage2")
        assert torch.allclose(s["sigmas"], torch.tensor([0.55, 0.55 * 0.421875 / 0.725, 0.0]))
        assert s["video"].noise_scale == pytest.approx(0.55) and s["video"].initial_latent is h.stage2_initial
    def test_full_prompt_context_is_shared_by_video_only_stages(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run()
        stage1 = next(v for n, v in h.calls if n == "stage1")
        stage2 = next(v for n, v in h.calls if n == "stage2")
        expected = h.video_encoding
        assert torch.equal(stage1["denoiser"][0], expected)
        assert torch.equal(stage1["video"].context, expected)
        assert torch.equal(stage2["denoiser"][0], expected)
        assert torch.equal(stage2["video"].context, expected)
        assert stage1["denoiser"][1] is None and stage1["audio"] is None
        assert stage2["denoiser"][1] is None and stage2["audio"] is None
        assert all(encoded["audio"] is None for encoded in h.encoded)
    def test_ordinary_euler_preserves_clean_tokens(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); assert next(v for n, v in h.calls if n == "stage2")["loop"] is None
        context = _InpaintHarness(monkeypatch, context=True); context.run(); assert callable(next(v for n, v in context.calls if n == "stage2")["loop"])
    def test_context_video_pyramid_fusion(self, monkeypatch: pytest.MonkeyPatch):
        import ltx_pipelines.utils.samplers as samplers
        monkeypatch.setattr(samplers, "post_process_latent", lambda value, *_: value + 100)
        loop, windows = latent_inpaint._context_loop(5, 2, 9, 1, 1), ((0, 5), (3, 8), (4, 9)); latent = torch.arange(9.).view(1, 9, 1)
        state = _State(latent, torch.zeros_like(latent), torch.zeros_like(latent), torch.zeros(1, 1, 9, 1)); calls: list[torch.Tensor] = []; steps: list[torch.Tensor] = []
        def denoiser(_t: Any, sliced: _State, audio: Any, _s: Any, _i: int) -> tuple[Any, None]: calls.append(sliced.latent); assert audio is None; return type("R", (), {"denoised": torch.full_like(sliced.latent, 10 + sliced.latent[0, 0, 0] // 3 * 10)})(), None
        stepper = type("Stepper", (), {"step": lambda _, _before, after, *_args: steps.append(after) or after})()
        result, audio = loop(torch.tensor([1., .5, 0.]), state, None, stepper, object(), denoiser)
        expected = torch.tensor([110, 110, 110, 113.3333, 117.5, 120, 120, 120, 120]).view(1, 9, 1)
        assert latent_inpaint._windows(9, 5, 2) == windows and len(calls) == 6 and len(steps) == 2 and audio is None and torch.allclose(steps[0], expected, atol=.001) and torch.equal(result.latent, steps[-1]) and all(torch.equal(call, source[:, start:end]) for source, group in ((latent, calls[:3]), (steps[0], calls[3:])) for call, (start, end) in zip(group, windows))
    def test_context_rejects_appended_tokens_or_attention_mask(self):
        state = _State(torch.zeros(1, 10, 1), torch.zeros(1, 10, 1), torch.zeros(1, 10, 1), torch.zeros(1, 1, 10, 1))
        with pytest.raises(ValueError, match="target-only"): latent_inpaint._context_loop(5, 2, 9, 1, 1)(torch.tensor([1., 0.]), state, None, None, None, None)
        attention = _State(torch.zeros(1, 9, 1), torch.zeros(1, 9, 1), torch.zeros(1, 9, 1), torch.zeros(1, 1, 9, 1), torch.ones(1, 9))
        with pytest.raises(ValueError, match="target-only"): latent_inpaint._context_loop(5, 2, 9, 1, 1)(torch.tensor([1., 0.]), attention, None, None, None, None)
    def test_preview_does_not_mutate_handoff(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch, preview=True); h.run(); decoded = next(v for n, v in h.calls if n == "decode"); upsampled = next(v for n, v in h.calls if n == "upsample"); assert torch.equal(decoded, torch.full_like(decoded, 7.0)) and torch.equal(upsampled, torch.full_like(upsampled, 7.0))
    def test_final_decode_crop_and_encode_are_video_only(self, monkeypatch: pytest.MonkeyPatch):
        h = _InpaintHarness(monkeypatch); h.run(); assert h.encoded[-1]["audio"] is None and h.encoded[-1]["total_frames"] == 3 and h.encoded[-1]["video"].shape[0] == 3


class TestStage1CfgPpAncestralLoop:
    @staticmethod
    def _install_fakes(monkeypatch: pytest.MonkeyPatch, records: dict[str, list[Any]]) -> None:
        from ltx_pipelines.utils import denoisers, samplers

        class _Denoiser:
            def __init__(self, context: torch.Tensor, _audio: None) -> None: self.context = context
            def __call__(self, _transformer: Any, state: _State, audio: None, sigmas: torch.Tensor, step: int) -> tuple[Any, None]:
                records["denoise"].append((self.context, state, audio, sigmas, step))
                return SimpleNamespace(denoised=state.latent + self.context), None

        class _Stepper:
            def __init__(self, **_kwargs: Any) -> None: pass
            def step(self, **kwargs: Any) -> torch.Tensor:
                records["step"].append(kwargs)
                return kwargs["denoised_sample"] + kwargs["uncond_denoised"] + (kwargs["noise"] if kwargs["noise"] is not None else 0)

        monkeypatch.setattr(denoisers, "SimpleDenoiser", _Denoiser)
        monkeypatch.setattr(samplers, "EulerCfgPpDiffusionStep", _Stepper)
        monkeypatch.setattr(samplers, "post_process_latent", lambda value, mask, clean: records["post"].append((value, mask, clean)) or value * mask + clean * (1 - mask))

    def test_positive_x0_is_processed_negative_is_raw_and_terminal_uses_positive_once(self, monkeypatch: pytest.MonkeyPatch):
        records: dict[str, list[Any]] = {"denoise": [], "post": [], "step": []}; self._install_fakes(monkeypatch, records)
        state = _State(torch.zeros(1, 2, 1), torch.ones(1, 2, 1), torch.full((1, 2, 1), 3.0))
        result, audio = _stage1_cfg_pp_ancestral_loop(torch.tensor(1.0), torch.tensor(2.0), torch.Generator().manual_seed(5))(
            sigmas=torch.tensor([1.0, 0.5, 0.0]), video_state=state, audio_state=None,
            stepper=object(), transformer=object(), denoiser=object(),
        )
        assert audio is None and len(records["denoise"]) == 4 and len(records["post"]) == 3 and len(records["step"]) == 1
        for index in range(0, 4, 2):
            positive, negative = records["denoise"][index:index + 2]
            assert positive[1] is negative[1] and positive[2] is negative[2] is None and positive[3] is negative[3] and positive[4] == negative[4]
        for index, step in enumerate(records["step"]):
            positive_post, x_next_post = records["post"][2 * index:2 * index + 2]
            negative_state = records["denoise"][2 * index + 1][1]
            assert step["denoised_sample"] is not step["uncond_denoised"]
            assert torch.equal(step["denoised_sample"], positive_post[0])
            assert torch.equal(step["uncond_denoised"], negative_state.latent + 2)
            assert torch.equal(x_next_post[0], step["denoised_sample"] + step["uncond_denoised"] + (step["noise"] if step["noise"] is not None else 0))
        second_step_state = records["denoise"][2][1]
        terminal_positive = records["post"][-1]
        assert torch.equal(terminal_positive[0], second_step_state.latent + 1)
        assert torch.equal(result.latent, terminal_positive[0] * terminal_positive[1] + terminal_positive[2] * (1 - terminal_positive[1]))
        assert result is not state and second_step_state is not state and result is not second_step_state and records["denoise"][0][1] is state

    def test_terminal_fractional_mask_blends_clean_tokens_once(self, monkeypatch: pytest.MonkeyPatch):
        records: dict[str, list[Any]] = {"denoise": [], "post": [], "step": []}; self._install_fakes(monkeypatch, records)
        state = _State(torch.zeros(1, 1, 1), torch.full((1, 1, 1), 0.5), torch.full((1, 1, 1), 10.0))
        result, _ = _stage1_cfg_pp_ancestral_loop(torch.tensor(4.0), torch.tensor(2.0), torch.Generator().manual_seed(5))(
            torch.tensor([1.0, 0.5, 0.0]), state, None, object(), object(), object(),
        )
        terminal_state = records["denoise"][-2][1]
        expected = (terminal_state.latent + 4.0) * 0.5 + 5.0
        assert len(records["step"]) == 1 and len(records["post"]) == 3
        assert torch.equal(result.latent, expected)
        assert not torch.equal(result.latent, result.latent * 0.5 + 5.0)

    def test_post_step_restore_preserves_clean_tokens_for_next_step(self, monkeypatch: pytest.MonkeyPatch):
        records: dict[str, list[Any]] = {"denoise": [], "post": [], "step": []}; self._install_fakes(monkeypatch, records)
        state = _State(torch.zeros(1, 2, 1), torch.tensor([[[1.0], [0.0]]]), torch.tensor([[[3.0], [7.0]]]))
        result, _ = _stage1_cfg_pp_ancestral_loop(torch.tensor(1.0), torch.tensor(2.0), torch.Generator().manual_seed(5))(
            torch.tensor([1.0, 0.5, 0.0]), state, None, object(), object(), object(),
        )
        assert records["post"][1][0][0, 1, 0] != state.clean_latent[0, 1, 0]
        assert records["denoise"][2][1].latent[0, 1, 0] == state.clean_latent[0, 1, 0]
        assert result.latent[0, 1, 0] == state.clean_latent[0, 1, 0]

    def test_rng_is_seeded_separately_and_advances_between_ancestral_steps(self, monkeypatch: pytest.MonkeyPatch):
        def run(seed: int) -> tuple[torch.Tensor, list[torch.Tensor]]:
            records: dict[str, list[Any]] = {"denoise": [], "post": [], "step": []}; self._install_fakes(monkeypatch, records)
            initial = torch.Generator().manual_seed(seed + 1)
            ancestral = torch.Generator().manual_seed(seed + 1)
            result, _ = _stage1_cfg_pp_ancestral_loop(torch.tensor(1.0), torch.tensor(2.0), ancestral)(
                torch.tensor([1.0, 0.75, 0.5, 0.0]), _State(torch.zeros(1, 2, 1), torch.ones(1, 2, 1), torch.zeros(1, 2, 1)), None, object(), object(), object(),
            )
            noises = [step["noise"] for step in records["step"] if step["noise"] is not None]
            assert initial is not ancestral
            return result.latent, noises

        first, first_noises = run(4); second, second_noises = run(4)
        assert torch.equal(first, second) and len(first_noises) == 2 and all(torch.equal(a, b) for a, b in zip(first_noises, second_noises))
        assert not torch.equal(first_noises[0], first_noises[1])


class TestIcLoraStageConstruction:
    @pytest.mark.parametrize(
        "transformer_format",
        ["gguf", "safetensors"],
    )
    def test_upstream_stage2_builders_have_loras_before_installers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        transformer_format: str,
    ) -> None:
        import ltx_pipelines.ic_lora as upstream
        import services.ic_lora_pipeline.ltx_ic_lora_pipeline as pipeline_module
        from services.patches import gguf_loader_fix

        events: list[str] = []

        class _Builder:
            def __init__(self, stage_name: str, builder_name: str, loras: tuple[Any, ...] = ()) -> None:
                self.stage_name = stage_name
                self.builder_name = builder_name
                self.loras = loras

            def with_loras(self, loras: tuple[Any, ...]) -> "_Builder":
                events.append(f"builder:{self.stage_name}:{self.builder_name}")
                return _Builder(self.stage_name, self.builder_name, loras)

        class _Stage:
            def __init__(self, name: str, *, has_streaming_builder: bool = False) -> None:
                self._transformer_builder = _Builder(name, "main")
                if has_streaming_builder:
                    self._streaming_builder = _Builder(name, "streaming")

        class _UpstreamPipeline:
            def __init__(self, **kwargs: Any) -> None:
                self.dtype = torch.float32
                self.device = kwargs["device"]
                self.stage_1 = _Stage("stage_1")
                self.stage_2 = _Stage("stage_2", has_streaming_builder=True)
                self.loras = kwargs["loras"]

        def _memory_plan(*_args: Any, **_kwargs: Any) -> None:
            events.append("memory_plan")
            assert _args[0].stage_2 is not _args[0].stage_1

        def _installer(*_args: Any, **_kwargs: Any) -> None:
            events.append("installer")
            assert _args[0].stage_2 is not _args[0].stage_1

        monkeypatch.setattr(upstream, "ICLoraPipeline", _UpstreamPipeline)
        monkeypatch.setattr(pipeline_module, "device_supports_fp8", lambda _: False)
        monkeypatch.setattr(pipeline_module, "attach_memory_plan_to_stages", _memory_plan)
        monkeypatch.setattr(gguf_loader_fix, "install_gguf_loader", _installer)
        monkeypatch.setattr(gguf_loader_fix, "install_gguf_component_paths", _installer)
        monkeypatch.setattr(gguf_loader_fix, "install_kijai_transformer_config_patch", _installer)

        components = SimpleNamespace(
            transformer_format=transformer_format,
            video_vae_path="video" if transformer_format == "safetensors" else None,
            audio_vae_path="audio",
            gemma_root=None,
        )
        result = LTXIcLoraPipeline(
            checkpoint_path="checkpoint",
            gemma_root=None,
            upsampler_path="upsampler",
            lora_paths=["lora-a", "lora-b"],
            device=torch.device("cpu"),
            offload_mode="caller",  # type: ignore[arg-type]
            components=components,
            memory_plan=SimpleNamespace(offload_mode="planned"),
        )

        assert result.pipeline.stage_2 is not result.pipeline.stage_1
        expected_loras = tuple(result.pipeline.loras)
        assert result.pipeline.stage_1._transformer_builder.loras == ()
        assert result.pipeline.stage_2._transformer_builder.loras == expected_loras
        assert result.pipeline.stage_2._streaming_builder.loras == expected_loras
        assert events[:2] == ["builder:stage_2:main", "builder:stage_2:streaming"]
        assert "memory_plan" in events
        assert "installer" in events
        assert events.index("builder:stage_2:main") < events.index("memory_plan")
        assert events.index("builder:stage_2:main") < events.index("installer")


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
    """Tests for the green-composite and mask-dilation utilities used by latent inpaint."""

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


def test_dilate_video_mask_chunked_spatial_matches_square_pool() -> None:
    """Chunked separable spatial dilation must equal one unchunked square pool."""
    from services.ic_lora_pipeline.official_inpaint import dilate_video_mask

    mask = torch.zeros(9, 64, 64, dtype=torch.float32)
    mask[0, 0, 0] = 1.0  # corner pixel, first chunk
    mask[3, 31, 31] = 1.0  # interior, last frame of first chunk
    mask[4, 63, 63] = 1.0  # corner pixel, first frame of second chunk
    mask[8, 5, 60] = 1.0  # last frame, final partial chunk

    actual = dilate_video_mask(mask.clone(), spatial_radius=3, temporal_radius=0)

    expected = torch.nn.functional.max_pool2d(
        mask.unsqueeze(1), kernel_size=7, stride=1, padding=3
    ).squeeze(1)
    expected = (expected > 0.5).float()

    assert torch.equal(actual, expected)


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
