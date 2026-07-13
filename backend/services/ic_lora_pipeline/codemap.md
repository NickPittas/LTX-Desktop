# backend/services/ic_lora_pipeline/

## Responsibility

IC-LoRA (Image-Conditioned LoRA) video generation: produces video conditioned on reference images and optional video reference latents, with a LoRA stack applied to the diffusion transformer. Provides three paths — general `generate` (T2V/I2V with optional video conditioning and outpaint compositing), V1 `generate_inpaint` (RGB two-stage bridge), and experimental video-only latent `generate_inpaint_v2` (latent blend → learned upsample → masked Stage 2). Exposes `IcLoraPipeline` Protocol and `LTXIcLoraPipeline` concrete wrapper.

Files:
- `ic_lora_pipeline.py` — `IcLoraPipeline` Protocol (`create` with `lora_paths`, `lora_strength`; `generate`, `generate_inpaint`, `generate_inpaint_v2`).
- `ltx_ic_lora_pipeline.py` — `LTXIcLoraPipeline` implementation (wraps `ltx_pipelines.ic_lora.ICLoraPipeline`); its V2 method is delegation-only.
- `inpaint_v2.py` — isolated video-only latent bridge: half-source/Stage-1 latent blend, exactly one learned upsample, full-resolution mask preserving the learned-upsampled blended latent, ordinary or call-scoped context Stage 2.
- `official_inpaint.py` — inpaint helpers: `green_composite_preprocess` (#66FF00), `dilate_video_mask` (separable max-pool), `laplacian_pyramid_blend` (kornia-based, chunked).

## Design Patterns

- **Protocol + concrete wrapper**: handler layer depends on `IcLoraPipeline`; `LTXIcLoraPipeline.create(...)` is the only constructor.
- **Single lora_strength applies to whole stack**: in `__init__` (lines 168–171) `lora_paths` is mapped one-to-one to `LoraPathStrengthAndSDOps(path=lp, strength=lora_strength, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)`. One `lora_strength` value is reused for every LoRA in the stack; there is no per-LoRA strength.
- **Format/quantization branching** identical to other pipelines: GGUF → None; split + cuda → the shared `kijai_fp8_quantization_policy()` from `services.patches.gguf_loader_fix` (Kijai scaled-FP8, including the policy-level `FuseRule`); else `build_policy()` (fp8_cast) if cuda else None. **Offload/quantization decisions are supplied by `LocalMemoryPlan`** when the handler passes one (`memory_plan.offload_mode` overrides the caller's); the transitional `memory_plan=None` path uses the caller's `offload_mode` as-is — no internal split/GGUF offload coercion remains.
- **VAE frame-count helpers**: `_vae_compatible_frame_count` (largest 1+8k ≤ n, for mask trim) and `_vae_padded_frame_count` (next 1+8k ≥ n, for inpaint padding; output cropped back).
- **Mask-radius derivation**: `derive_stage_radii(mask_grow_px)` → `(stage1=(n+1)//2, stage2=n)` mapping the user-facing `mask_grow_px` to per-stage dilation radii.
- **Green-guide direct tensor encode** (`_encode_green_guide_conditioning`): replaces a temp-mp4 roundtrip; builds a `VideoConditionByReferenceLatent(latent=encoded, downscale_factor=1, strength=strength)` from a direct tiled VAE encode of the green composite tensor.
- **Streaming prefetch tuning for inpaint** (`_inpaint_streaming_prefetch_count`): returns explicit override unchanged; else `LTX_INPAINT_STREAM_PREFETCH` env var if frames ≥ 97; else default 2 for long (≥97f), None for short.
- **Large inpaint stage-2 context window**: for long clips (>121f) or ≥1080p, `generate_inpaint` stamps the HDR-installed `_ltx_desktop_context_loop` config on `stage_2` before the full-res diffusion call, then clears it for non-large cached-stage reuse.
- **`@torch.inference_mode()`** on `generate` and `generate_inpaint`.

## Data & Control Flow

### `generate` — frame production (a)

`generate(prompt, seed, height, width, num_frames, frame_rate, images, video_conditioning, output_path, mask_path=None, conditioning_strength=1.0, original_video_path=None)` (lines 389–436):

1. `tiling_config = default_tiling_config()`.
2. `result = self._run_inference(...)` → `self.pipeline(...)` (`ICLoraPipeline`). Inside `_run_inference`, if `mask_path is not None`, loads mask via `ic_lora_module._load_mask_video(mask_path, height//2, width//2, num_frames_vae)` and passes it as `conditioning_attention_mask` with `conditioning_attention_strength`. Returns `(video, audio)` where `video` is `torch.Tensor | Iterator[torch.Tensor]`.
3. **Optional outpaint compositing** (lines 422–433): if both `original_video_path` and `mask_path` are set and `video` is an Iterator, it is materialized via `torch.cat(list(video), dim=0)`, then `_composite_in_outpainting(...)` blends generated frames with the decoded original using the decoded mask (`gen*mask + orig*(1-mask)` in [0,1], uint8 out).
4. `chunks = video_chunks_number(num_frames, tiling_config)`.

### `generate` — encode call site (b)

Line 436 — single encode call at the frame→encode boundary:

```python
encode_video_output(
    video=video,        # Tensor or Iterator of decoded/composited frames
    audio=audio,        # AudioOrNone
    fps=int(frame_rate),
    output_path=output_path,
    video_chunks_number_value=chunks,
)
```

Imported from `services.ltx_pipeline_common.encode_video_output`; forwards unchanged to `ltx_pipelines.utils.media_io.encode_video(video=, fps=, audio=, output_path=, video_chunks_number=)`.

### `generate_inpaint` — frame production (a)

`generate_inpaint(...)` (lines 442–837) runs the official LTX-2.3 two-stage IC-LoRA inpaint flow entirely inside this method (does not delegate the stages to `ICLoraPipeline`; only uses `self.pipeline.{prompt_encoder, image_conditioner, stage_1, stage_2, video_decoder, audio_decoder, device, dtype, reference_downscale_factor}`):

1. `assert_resolution(height, width, is_two_stage=True)`; `derive_stage_radii(mask_grow_px)` → `(stage1_radius, stage2_radius)`; `num_frames_vae_padded = _vae_padded_frame_count(num_frames)`.
2. Encode prompt → `(video_context, audio_context)`.
3. Load video (`decode_video_by_frame` + `video_preprocess`) at full + half res; pad to `num_frames_vae_padded` by repeating last frame. Load mask → grayscale `[0,1]`.
4. Dilate masks: `dilate_video_mask(..., spatial_radius=stage1_radius|stage2_radius)` at full res; downscale stage1 mask to half res.
5. `green_half = green_composite_preprocess(video_half, mask_stage1_half)`.
6. **Stage 1** (half res, `DISTILLED_SIGMAS`, `SimpleDenoiser`): conditionings = `combined_image_conditionings(...)` + `_encode_green_guide_conditioning(enc, green_half, conditioning_strength)`; `self.pipeline.stage_1(...)` → `video_state_s1`.
7. Decode stage 1 (`self.pipeline.video_decoder(video_state_s1.latent, tiling_config, generator)`) → `_collect_frames` → `(F, H_half, W_half, 3)` in [0,1]. `laplacian_pyramid_blend(decoded_s1, video_half_frames_01, mask_s1_blend, max_level=7, mask_low_res_dilation=INPAINT_BLEND1_LOW_RES_DILATION=5)` → `blend_stage1`.
8. Upscale `blend_stage1` 2× (bicubic), convert `(F,3,H,W)→(1,3,F,H,W)` in [-1,1], trim to `_vae_compatible_frame_count`, tiled VAE encode via `image_conditioner(lambda enc: enc.tiled_encode(..., tiling_config))` → `encoded_blend`.
9. **Stage 2** (full res, `STAGE_2_DISTILLED_SIGMAS[1:]` scaled so first sigma ≈ 0.55, `noise_scale=0.55`, `initial_latent=encoded_blend`; audio `noise_scale=stage2_sigmas[0].item()`, `initial_latent=audio_state_s1.latent`): large workloads stamp a rolling context-window config on `stage_2`; `self.pipeline.stage_2(...)` → `video_state_s2`, `audio_state_s2`. Optional: if `save_stage_1_preview`, encodes a half-res `<primary_stem>_stage1_preview.mp4` after the first-stage Laplacian blend; `generate_inpaint` returns the preview path (`str | None`).
10. Decode stage 2 → `_collect_frames`. Stage 2 output used **directly** as `blend_stage2 = decoded_s2_frames` — the second `laplacian_pyramid_blend` and `_apply_raw_mask_guard` are **skipped** per the requested inpaint change (full-res decoded frames are the final output, no blend back to original video).
11. (removed — raw-mask guard no longer applied)
12. Decode audio: `decoded_audio = self.pipeline.audio_decoder(audio_state_s2.latent)`. Crop `blend_stage2 = blend_stage2[:num_actual_frames]`; `chunks = video_chunks_number(num_actual_frames, tiling_config)`.

### `generate_inpaint` — encode call site (b)

Lines 830–836 — single encode call:

```python
encode_video_output(
    video=blend_stage2.clamp(0, 1),                              # (F,H,W,3) float32 CPU tensor in [0,1]
    audio=decoded_audio,                                       # Audio from audio_decoder
    fps=int(frame_rate),
    output_path=output_path,
    video_chunks_number_value=chunks,                          # from num_actual_frames
)
```

Note: inpaint passes a single concrete float32 [0,1] tensor (not an Iterator — passing uint8 crashed upstream color conversion via `avg_pool2d` on Byte) and `num_actual_frames`-derived chunks (cropped back from the padded count).

### Output path hardcoding (c)

`output_path` is caller-supplied `str` for both `generate` and `generate_inpaint`; the wrapper does not rewrite the extension. Callers (backend handlers) currently pass `.mp4` paths and the downstream `ltx_pipelines.utils.media_io.encode_video` hardcodes H.264 / yuv420p. To add MOV ProRes / EXR as primary output, the routing points are the **two** `encode_video_output(...)` call sites: line 436 (`generate`) and lines 830–836 (`generate_inpaint`). The inpaint call passes a float32 [0,1] CPU tensor, so an EXR path is a natural fit (no uint8→float conversion needed); it must still explicitly drop the audio mux (`audio=None`).

### T2V no-video path

The Ingredients T2V no-video path is `generate(..., video_conditioning=[], mask_path=None, original_video_path=None)` — `_run_inference` still runs (no mask load, `conditioning_attention_mask=None`), no compositing branch is taken, and the decoded `(video, audio)` is encoded directly via `encode_video_output` at line 436.

## Integration Points

- **`services.ltx_pipeline_common`**: `default_tiling_config`, `video_chunks_number`, shared `encode_video_output` (single encode chokepoint; two call sites here).
- **`services.ltx_components`**: `CheckpointPath`, `ResolvedLtxComponents`.
- **`services.services_utils`**: `AudioOrNone`, `TilingConfigType`, `device_supports_fp8`.
- **`services.patches.gguf_loader_fix`**: GGUF/Kijai loader + component-path/transformer-config patches; provides the shared `kijai_fp8_quantization_policy()` (Kijai scaled-FP8 + policy-level `FuseRule`) used by the split-cuda branch.
- **`services.local_memory_plan`**: `LocalMemoryPlan` — when the handler passes `memory_plan=`, its `offload_mode` drives quantization/offload (Phase 2).
- **`api_types.ImageConditioningInput`**: remapped to `ltx_pipelines.utils.args.ImageConditioningInput`.
- **`ltx_pipelines.ic_lora.ICLoraPipeline`**: underlying IC-LoRA pipeline (used for `generate` entirely; for `generate_inpaint` only its blocks are reused).
- **`ltx_core.loader.primitives.LoraPathStrengthAndSDOps`** + **`ltx_core.loader.sd_ops.LTXV_LORA_COMFY_RENAMING_MAP`**: LoRA stack construction (single `lora_strength` applied to every entry).
- **`ltx_core.conditioning.VideoConditionByReferenceLatent`**: reference-latent conditioning wrapper (green guide, `_encode_video_conditioning`).
- **`ltx_pipelines.utils.media_io`**: `decode_video_by_frame`, `video_preprocess` (input decode for compositing/inpaint), and `encode_video` (ultimate encoder reached via `encode_video_output`).
- **`ltx_pipelines.utils.helpers`**: `assert_resolution`, `combined_image_conditionings`.
- **`ltx_pipelines.utils.constants`**: `DISTILLED_SIGMAS`, `STAGE_2_DISTILLED_SIGMAS`.
- **`kornia.geometry.transform.pyramid`**: `build_laplacian_pyramid`, `build_pyramid`, `PyrUp`, `find_next_powerof_two`, `is_powerof_two` (used by `official_inpaint.laplacian_pyramid_blend`).
- **Depth/pose preprocessor pipelines**: when Union Control is enabled, `backend/services/depth_processor_pipeline` and `.../pose_processor_pipeline` produce conditioning frames consumed upstream of `video_conditioning` (see those codemaps).
- Handler layer constructs via `LTXIcLoraPipeline.create(...)` and calls `.generate(...)` / `.generate_inpaint(...)`; see the relevant handler codemap for caller-supplied `output_path` resolution.
