# backend/services/ic_lora_pipeline/

## Responsibility

IC-LoRA (Image-Conditioned LoRA) video generation: produces video conditioned on reference images and optional video reference latents, with a LoRA stack applied to the diffusion transformer. Provides the general `generate` path (T2V/I2V with optional video conditioning and outpaint compositing) and one canonical latent-only `generate_inpaint` path. Exposes `IcLoraPipeline` Protocol and `LTXIcLoraPipeline` concrete wrapper.

Files:
- `ic_lora_pipeline.py` — `IcLoraPipeline` Protocol (`create` with `lora_paths`, `lora_strength`; `generate`, `generate_inpaint`).
- `ltx_ic_lora_pipeline.py` — `LTXIcLoraPipeline` implementation (wraps `ltx_pipelines.ic_lora.ICLoraPipeline`).
- `latent_inpaint.py` — canonical video-only latent in/outpaint flow: green guide Stage 1 conditioning, latent blend, one learned upsample, binary-mask Stage 2, and optional context loop/preview.
- `official_inpaint.py` — inpaint helpers: `green_composite_preprocess` (#66FF00), `dilate_video_mask` (separable max-pool), `laplacian_pyramid_blend` (kornia-based, chunked).

## Design Patterns

- **Protocol + concrete wrapper**: handler layer depends on `IcLoraPipeline`; `LTXIcLoraPipeline.create(...)` is the only constructor.
- **Single lora_strength applies to whole stack**: in `__init__` (lines 168–171) `lora_paths` is mapped one-to-one to `LoraPathStrengthAndSDOps(path=lp, strength=lora_strength, sd_ops=LTXV_LORA_COMFY_RENAMING_MAP)`. One `lora_strength` value is reused for every LoRA in the stack; there is no per-LoRA strength.
- **Format/quantization branching** identical to other pipelines: GGUF → None; split + cuda → the shared `kijai_fp8_quantization_policy()` from `services.patches.gguf_loader_fix` (Kijai scaled-FP8, including the policy-level `FuseRule`); else `build_policy()` (fp8_cast) if cuda else None. **Offload/quantization decisions are supplied by `LocalMemoryPlan`** when the handler passes one (`memory_plan.offload_mode` overrides the caller's); the transitional `memory_plan=None` path uses the caller's `offload_mode` as-is — no internal split/GGUF offload coercion remains.
- **VAE frame-count helpers**: `_vae_compatible_frame_count` (largest 1+8k ≤ n, for mask trim) and `_vae_padded_frame_count` (next 1+8k ≥ n, for inpaint padding; output cropped back).
- **Mask-radius derivation**: `derive_stage_radii(mask_grow_px)` → `(stage1=(n+1)//2, stage2=n)` mapping the user-facing `mask_grow_px` to per-stage dilation radii.
- **Stage-1 green guide gate**: the final full-video keyframe guide is wrapped only for a nonblank prompt, no optional images, and a nonempty half latent mask. The wrapper uses patchifier token order to install a persistent `(1,2N,2N)` boolean matrix blocking only masked target-query → green-guide-key attention; guide → target remains open. That exact gated Stage 1 uses CFG=1 official CFG++ ancestral two-pass sampling with the positive prompt and the module-local official negative prompt: it post-processes the positive x0 prediction, passes the negative prediction raw, then restores clean tokens on the stochastic x_next. Empty/whitespace prompts, image requests, or empty masks use the raw guide and ordinary Euler; Stage 2 remains ordinary Euler preserve-conditioning-only and the source-latent half-mask is reused unchanged for its Stage-1 blend.
- **Canonical latent mask and Stage 2**: `mask_to_latent_denoise_mask` thresholds source masks at 0.5, nearest-resizes spatially, and causally amax-groups frames into exact float32 binary latent values. Stage 2 uses `STAGE_2_DISTILLED_SIGMAS[1:]` scaled to start at 0.55.
- **Shared LoRA stages**: Stage 1 and Stage 2 are distinct `DiffusionStage` instances configured with the same LoRA stack. Stage 2 is replaced before memory/GGUF/Kijai patch installation.
- **Streaming prefetch tuning for inpaint** (`_inpaint_streaming_prefetch_count`): returns explicit override unchanged; else `LTX_INPAINT_STREAM_PREFETCH` env var if frames ≥ 97; else default 2 for long (≥97f), None for short.
- **Large inpaint stage-2 context window**: for long clips (>121f) or ≥1080p, `generate_inpaint` stamps the HDR-installed `_ltx_desktop_context_loop` config on `stage_2` before the full-res diffusion call, then clears it for non-large cached-stage reuse.
- **`@torch.inference_mode()`** on `generate` and `generate_inpaint`.

## Data & Control Flow

### `generate` — frame production (a)

`generate(prompt, seed, height, width, num_frames, frame_rate, images, video_conditioning, output_path, mask_path=None, conditioning_strength=1.0, original_video_path=None)` (lines 389–436):

1. `tiling_config = default_tiling_config()`.
2. `result = self._run_inference(...)` → `self.pipeline(...)` (`ICLoraPipeline`). Inside `_run_inference`, if `mask_path is not None`, loads mask via `ic_lora_module._load_mask_video(mask_path, height//2, width//2, num_frames_vae)` and passes it as `conditioning_attention_mask` with `conditioning_attention_strength`. Returns `(video, audio)` where `video` is `torch.Tensor | Iterator[torch.Tensor]`.
3. **Optional outpaint compositing** (lines 422–433): if both `original_video_path` and `mask_path` are set and `video` is an Iterator, it is materialized via `torch.cat(list(video), dim=0)`, then `_composite_in_outpainting(...)` blends generated frames with the decoded original using the decoded mask (`gen*mask + orig*(1-mask)` in [0,1], uint8 out).
4. `chunks = video_chunks_number(num_frames, tiling_config)`. Shared tiling uses the app-owned 768/256 spatial seam mitigation (80/24 temporal); retake and HDR remain excluded under their separate policies.

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

`generate_inpaint(...)` is the sole in/outpaint implementation and delegates to `latent_inpaint.py`. It runs the two-stage latent flow using the IC-LoRA pipeline's prompt/image conditioning, diffusion stages, video decoder, device, dtype, and reference downscale factor:

1. `assert_resolution(height, width, is_two_stage=True)`; `derive_stage_radii(mask_grow_px)` → `(stage1_radius, stage2_radius)`; `num_frames_vae_padded = _vae_padded_frame_count(num_frames)`.
2. Encode prompt and load/pad the source video and grayscale mask. Dilate per-stage masks and make the Stage 1 green guide (`#66FF00`) at half resolution.
3. Canonical inpaint shares the full prompt video context across both stages; it never uses transient or generated audio.
4. Blend the Stage 1 latent with the source latent, then apply exactly one learned upsample.
5. **Stage 2** uses `VideoConditionByMask` with the binary latent mask and the 0.55 schedule. Large-workload context policy may install a context loop; a Stage 1 preview remains optional.
6. Decode and crop the Stage 2 video frames, then encode video only; Stage 2 and output encoding discard audio, so no audio muxing is used.

### `generate_inpaint` — encode call site (b)

Lines 830–836 — single encode call:

```python
encode_video_output(
    video=blend_stage2.clamp(0, 1),                              # (F,H,W,3) float32 CPU tensor in [0,1]
    audio=None,                                                # canonical inpaint is video-only
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
- Handler layer constructs via `LTXIcLoraPipeline.create(...)` and calls `.generate(...)` / canonical `.generate_inpaint(...)`; see the relevant handler codemap for caller-supplied `output_path` resolution.
