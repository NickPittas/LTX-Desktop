# backend/services/retake_pipeline/

## Responsibility

Video retake / V2V regeneration: re-denoises a sub-region (or all) of an existing video's video and audio modalities, driven by a temporal region mask (`TemporalRegionMask` over `[start_time, end_time]`). Forks orchestration from `ltx_pipelines.retake` with three intentional divergences (documented in the module docstring): `@torch.no_grad()` instead of `@torch.inference_mode()` (custom autograd functions in the checkpoint), tiled source-video encoding via `video_latent_from_file(..., tiling_config)`, and tiled video decoding via `VideoDecoder(..., tiling_config)`. Exposes `RetakePipeline` Protocol and `LTXRetakePipeline` concrete wrapper.

Files:
- `retake_pipeline.py` — `RetakePipeline` Protocol (`create` with `loras`, `quantization`; `generate`).
- `ltx_retake_pipeline.py` — `LTXRetakePipeline` implementation (owns `PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoDecoder`, `AudioDecoder` directly; orchestrates encode/decode/denoise/blend itself rather than delegating to a single `ltx_pipelines.*` pipeline object).

## Design Patterns

- **Protocol + concrete wrapper**: handler depends on `RetakePipeline`; `LTXRetakePipeline.create(...)` is the only constructor. Note `create(...)` is annotated to return `RetakePipeline` (the Protocol) while actually returning `LTXRetakePipeline`.
- **Self-assembled block lifecycle**: unlike fast/a2v/ic_lora wrappers (which construct a single `ltx_pipelines.*.Pipeline`), this wrapper instantiates each `ltx_pipelines.utils.blocks` primitive (`PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoDecoder`, `AudioDecoder`) on `self` and orchestrates them inside `_run`.
- **`loras: list[LoraPathStrengthAndSDOps]` and `quantization` are constructor args** (not generate args) — passed straight into `DiffusionStage(..., loras=tuple(loras), quantization=stage_quantization)`. Caller (handler) builds the LoRA entries with per-entry strengths before construction.
- **Format/quantization branching** in `__init__`: `is_gguf` → `stage_quantization=None`; `is_split and quantization is not None` → `kijai_fp8_quantization_policy()`; else `stage_quantization = quantization` (passed through). GGUF prompt-encoder patch installed when `components.gemma_root is not None`. GGUF loader + component paths, or Kijai transformer config patch + component paths, installed per branch (note: `install_gguf_loader(self)` / `install_gguf_component_paths(self, ...)` — patch targets the wrapper instance itself, since the wrapper owns the blocks). Split 22B defaults `streaming_prefetch_count=2`.
- **Two tiling configs**: `tiling = TilingConfig.default()` (decode) and a tighter `encoding_tiling` (`SpatialTilingConfig(tile_size_in_pixels=256, tile_overlap_in_pixels=64)`, `TemporalTilingConfig(tile_size_in_frames=24, tile_overlap_in_frames=16)`) for source-video VAE encoding to cap encoder VRAM.
- **Source frame snapping**: `output_shape.frames` snapped down to nearest `8n+1` via `SpatioTemporalScaleFactors.default().time` when `(frames-1) % time != 0`.
- **Dual denoiser modes**: `distilled=True` → `DISTILLED_SIGMA_VALUES` + `SimpleDenoiser`; `distilled=False` → `LTX2Scheduler().execute(steps=num_inference_steps)` + `GuidedDenoiser` (requires `video_guider_params` and `audio_guider_params`, encodes both `[prompt]` and `[negative_prompt]`).
- **`TemporalRegionMask(start_time, end_time, fps=output_shape.fps)`** conditioning on each modality, gated by `regenerate_video` / `regenerate_audio` flags and the presence of `initial_audio_latent`. `frozen=not regenerate_*`.
- **Decode ordering**: audio decoded eagerly (`self.audio_decoder(audio_state.latent)`), video decoded lazily (`self.video_decoder(video_state.latent, tiling, generator)` returns `Iterator[torch.Tensor]`).
- **`@torch.no_grad()`** on `_run` and `generate` (NOT `inference_mode` — see module docstring).

## Data & Control Flow

### Frame production (a)

`generate(*, video_path, prompt, start_time, end_time, seed, output_path, negative_prompt="", num_inference_steps=40, video_guider_params=None, audio_guider_params=None, regenerate_video=True, regenerate_audio=True, enhance_prompt=False, distilled=True)` (lines 322–367):

1. `meta = get_videostream_metadata(video_path)` → `fps, num_frames = meta.fps, meta.frames`.
2. `video_iter, audio = self._run(...)` (lines 164–319):
   - Validates `start_time < end_time`; resolves `effective_seed` (random if `seed < 0`).
   - `get_videostream_metadata(video_path)` → `output_shape`; snaps frames to `8n+1`.
   - **Source video encode (tiled)**: `self.image_conditioner(lambda enc: video_latent_from_file(video_encoder=enc, file_path=video_path, output_shape=output_shape, dtype=dtype, device=self.device, tiling_config=encoding_tiling))` → `initial_video_latent`.
   - **Source audio encode**: `self.audio_conditioner(lambda enc: audio_latent_from_file(audio_encoder=enc, file_path=video_path, output_shape=output_shape, dtype=dtype, device=self.device))` → `initial_audio_latent`.
   - **Text encode**: `self.prompt_encoder([prompt] if distilled else [prompt, negative_prompt], enhance_first_prompt=enhance_prompt, enhance_prompt_seed=effective_seed, streaming_prefetch_count=streaming_prefetch_count)`.
   - Build `video_modality_spec` / `audio_modality_spec` with `TemporalRegionMask` conditionings + `initial_latent` + `frozen`.
   - Build denoiser (`SimpleDenoiser` or `GuidedDenoiser`).
   - `video_state, audio_state = self.stage(denoiser=, sigmas=, noiser=, width=output_shape.width, height=output_shape.height, frames=output_shape.frames, fps=output_shape.fps, video=video_modality_spec, audio=audio_modality_spec, streaming_prefetch_count=streaming_prefetch_count)`.
   - `decoded_audio = self.audio_decoder(audio_state.latent)` (eager).
   - `decoded_video = self.video_decoder(video_state.latent, tiling, generator)` (lazy Iterator).
   - Returns `(decoded_video: Iterator[torch.Tensor], decoded_audio: Audio)`.
3. `tiling_config = TilingConfig.default()`; `video_chunks = get_video_chunks_number(num_frames, tiling_config)` (note: uses module-level `get_video_chunks_number` import, NOT `services.ltx_pipeline_common.video_chunks_number`).

### Encode call site (b) — DIRECT `encode_video`, BYPASSES common wrapper

**Lines 25 and 361–367 — this pipeline imports `encode_video` directly from `ltx_pipelines.utils.media_io` and calls it directly, NOT via `services.ltx_pipeline_common.encode_video_output`.** This is the only video-generation wrapper in `backend/services/` that bypasses the shared encode wrapper.

Line 25 import:
```python
from ltx_pipelines.utils.media_io import encode_video, get_videostream_metadata
```

Lines 361–367 call:
```python
encode_video(
    video=video_iter,            # Iterator[torch.Tensor] from self.video_decoder
    fps=int(fps),                # from get_videostream_metadata(video_path).fps
    audio=audio_out,             # Audio | None (= decoded_audio)
    output_path=output_path,
    video_chunks_number=video_chunks,
)
```

**CRITICAL FOR MOV/EXR WORK**: when introducing the new encoder service, this pipeline MUST be routed through the new encoder explicitly. Unlike the other pipelines (fast/a2v/ic_lora) which all funnel through `services.ltx_pipeline_common.encode_video_output` and would be fixed by editing that one wrapper, this pipeline has its own direct import and call site that must be changed separately. The call signature is otherwise identical to the common wrapper's forwarded arguments.

### Output path hardcoding (c)

`output_path` is caller-supplied `str`; the wrapper does not rewrite the extension. Callers (backend handlers) currently pass `.mp4` paths and `ltx_pipelines.utils.media_io.encode_video` hardcodes H.264 / yuv420p regardless of extension. To add MOV ProRes / EXR as primary output, the routing point is the **direct** `encode_video(...)` call at lines 361–367 (delete the line 25 direct import and route through the new encoder service / common wrapper to match the other pipelines). `fps` is sourced from `get_videostream_metadata(video_path).fps` (the source video, not a generation param).

## Integration Points

- **`ltx_pipelines.utils.media_io`**: `encode_video` (DIRECT import + call — bypasses `services.ltx_pipeline_common.encode_video_output`), `get_videostream_metadata` (source video shape/fps).
- **`services.ltx_pipeline_common`**: this wrapper does **not** import `encode_video_output` (intentional divergence to flag during encoder refactor). It also does not use `default_tiling_config` / `video_chunks_number` from the common module — uses `TilingConfig.default()` and `get_video_chunks_number` directly.
- **`services.ltx_components`**: `CheckpointPath`, `ResolvedLtxComponents`.
- **`services.patches.gguf_loader_fix`**: GGUF/Kijai loader and FP8 patches (patch target is `self`, not a wrapped pipeline object).
- **`ltx_pipelines.utils.blocks`**: `PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoDecoder`, `AudioDecoder` (instantiated on `self`).
- **`ltx_pipelines.utils.helpers`**: `video_latent_from_file` (tiled source encode), `audio_latent_from_file` (source audio encode).
- **`ltx_pipelines.utils.denoisers`**: `SimpleDenoiser` (distilled), `GuidedDenoiser` (non-distilled).
- **`ltx_pipelines.utils.constants`**: `DISTILLED_SIGMA_VALUES`.
- **`ltx_core.components.guiders`**: `MultiModalGuider`, `MultiModalGuiderParams`.
- **`ltx_core.components.schedulers`**: `LTX2Scheduler`.
- **`ltx_core.components.noisers`**: `GaussianNoiser`.
- **`ltx_core.conditioning.types.noise_mask_cond`**: `TemporalRegionMask`.
- **`ltx_core.loader` / `ltx_core.loader.primitives`**: `LoraPathStrengthAndSDOps` (constructor arg, caller-built).
- **`ltx_core.quantization`**: `QuantizationPolicy` (constructor arg).
- **`ltx_core.model.video_vae`**: `TilingConfig`, `SpatialTilingConfig`, `TemporalTilingConfig`, `get_video_chunks_number`, `SpatioTemporalScaleFactors`.
- Handler layer constructs via `LTXRetakePipeline.create(...)` (passing caller-built `loras` and `quantization`) and calls `.generate(...)`; see the relevant handler codemap for caller-supplied `output_path` resolution.
