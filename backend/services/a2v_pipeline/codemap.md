# backend/services/a2v_pipeline/

## Responsibility

Audio-to-Video generation: produces a video whose frames are synthesized to match a supplied audio track. Wraps a two-stage distilled pipeline (`DistilledA2VPipeline`) that runs half-resolution generation with frozen audio conditioning, 2× upsamples, then refines at full resolution, returning the original (not VAE-decoded) audio for maximum fidelity. Exposes `A2VPipeline` Protocol and `LTXa2vPipeline` concrete wrapper plus the in-folder `DistilledA2VPipeline` implementation.

Files:
- `a2v_pipeline.py` — `A2VPipeline` Protocol (`create`, `generate`).
- `ltx_a2v_pipeline.py` — `LTXa2vPipeline` wrapper (model lifecycle, GGUF/Kijai patches, FP8 selection, encode call).
- `distilled_a2v_pipeline.py` — `DistilledA2VPipeline` two-stage distilled A2V implementation (own block-based model lifecycle, no LoRA swap between stages).

## Design Patterns

- **Protocol + wrapper + in-folder distilled impl**: `LTXa2vPipeline` owns construction/patches/encode; `DistilledA2VPipeline` owns the denoising stages and is constructed via `ltx_pipelines.utils.blocks` primitives (`PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoUpsampler`, `VideoDecoder`).
- **Deferred imports** of all `ltx_pipelines.*` / `ltx_core.*` / `services.patches.*` symbols inside methods.
- **Format/quantization branching** (in `LTXa2vPipeline.__init__`): `is_gguf` (components.transformer_format == "gguf") → quantization None; `is_split` (safetensors + video_vae_path) + cuda → `kijai_fp8_quantization_policy()`; else `QuantizationPolicy.fp8_cast()` if cuda else None. GGUF prompt-encoder patch installed when `components.gemma_root is not None`. GGUF loader + component paths, or Kijai transformer config patch + component paths, installed per branch. Split 22B defaults `streaming_prefetch_count=2`.
- **Two-stage distilled schedule** (in `DistilledA2VPipeline.__call__`): Stage 1 half-res with `DISTILLED_SIGMA_VALUES`, audio `ModalitySpec(frozen=True, noise_scale=0.0, initial_latent=encoded_audio_latent)`; `VideoUpsampler` 2× on `video_state.latent[:1]`; Stage 2 full-res with `STAGE_2_DISTILLED_SIGMA_VALUES`, video `noise_scale=stage_2_sigmas[0].item()`, audio still frozen.
- **`assert_resolution(height, width, is_two_stage=True)`** enforces resolution compatible with the 2-stage halving.
- **Audio fidelity rule**: `decoded_audio = decode_audio_from_file(...)` is encoded into the latent for conditioning, but the **original** waveform (trimmed via `AudioLatentShape.from_duration` + `round(num_frames/frame_rate * sampling_rate)`) is returned — not the VAE-decoded audio.
- **`@torch.inference_mode()`** on both `DistilledA2VPipeline.__call__` and `LTXa2vPipeline.generate`.

## Data & Control Flow

### Frame production (a)

`LTXa2vPipeline.generate(prompt, negative_prompt, seed, height, width, num_frames, frame_rate, num_inference_steps, images, audio_path, audio_start_time, audio_max_duration, output_path)`:

1. `tiling_config = default_tiling_config()`.
2. `video, audio = self._run_inference(...)` → `self.pipeline(...)` (`DistilledA2VPipeline.__call__`).
   - Inside `DistilledA2VPipeline.__call__`: text encode (positive only) → `decode_audio_from_file(audio_path, device, audio_start_time, audio_max_duration)` → `audio_conditioner(lambda enc: vae_encode_audio(...))` → pad/trim latent to `AudioLatentShape.from_duration(...).frames` → Stage 1 (half res, frozen audio) → `upsampler(video_state.latent[:1])` → Stage 2 (full res) → `decoded_video = self.video_decoder(video_state.latent, tiling_config, generator)` (returns `Iterator[torch.Tensor]`) → trim original audio waveform to `num_frames/frame_rate` → return `(decoded_video, original_audio)`.
   - `negative_prompt` and `num_inference_steps` are accepted by the wrapper signature for Protocol conformance but the distilled path does not consume them (distilled sigmas are fixed).
3. `chunks = video_chunks_number(num_frames, tiling_config)`.

### Encode call site (b)

Line 172 — single encode call at the frame→encode boundary:

```python
encode_video_output(
    video=video,        # Iterator[torch.Tensor] of decoded frames from video_decoder
    audio=audio,        # original (non-VAE-decoded) Audio, trimmed to video duration
    fps=int(frame_rate),
    output_path=output_path,
    video_chunks_number_value=chunks,
)
```

Imported from `services.ltx_pipeline_common.encode_video_output`, which forwards unchanged to `ltx_pipelines.utils.media_io.encode_video(video=, fps=, audio=, output_path=, video_chunks_number=)`. No other encode path exists for this pipeline; no warmup method.

### Output path hardcoding (c)

`output_path` is a caller-supplied `str`; the wrapper does not rewrite the extension. Callers (backend handlers) currently pass a `.mp4` path and `ltx_pipelines.utils.media_io.encode_video` hardcodes H.264 / yuv420p regardless of extension. To add MOV ProRes / EXR as primary output, the routing point is this single `encode_video_output(...)` call (line 172). Note the audio argument is the original waveform, so an EXR (video-only) path must explicitly pass `audio=None` / drop muxing.

## Integration Points

- **`services.ltx_pipeline_common`**: `default_tiling_config`, `video_chunks_number`, shared `encode_video_output` wrapper (single encode chokepoint).
- **`services.ltx_components`**: `CheckpointPath`, `ResolvedLtxComponents`.
- **`services.services_utils`**: `AudioOrNone`, `TilingConfigType`, `device_supports_fp8`.
- **`services.patches.gguf_loader_fix`**: GGUF/Kijai loader and FP8 patches.
- **`api_types.ImageConditioningInput`**: remapped to `(path, frame_idx, strength)` tuples for `DistilledA2VPipeline` (note: tuple form, not the `ltx_pipelines.utils.args.ImageConditioningInput` dataclass used elsewhere).
- **`ltx_pipelines.utils.blocks`**: `PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoUpsampler`, `VideoDecoder`.
- **`ltx_pipelines.utils.media_io.decode_audio_from_file`**: source audio decode for conditioning.
- **`ltx_pipelines.utils.constants`**: `DISTILLED_SIGMA_VALUES`, `STAGE_2_DISTILLED_SIGMA_VALUES`.
- **`ltx_core.model.audio_vae.encode_audio`** (via `vae_encode_audio`): audio VAE encode for latent conditioning.
- **`ltx_pipelines.utils.media_io.encode_video`**: ultimate encoder (H.264/yuv420p hardcoded) reached via `encode_video_output`. See `backend/services/ltx_pipeline_common/codemap.md`.
- Handler layer constructs via `LTXa2vPipeline.create(...)` and calls `.generate(...)`; see the relevant handler codemap for caller-supplied `output_path` resolution.
