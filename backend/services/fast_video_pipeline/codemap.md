# backend/services/fast_video_pipeline/

## Responsibility

Wraps `ltx_pipelines.distilled.DistilledPipeline` to provide the "fast" distilled text/image-to-video generation path (single-stage distilled sigma schedule, optional spatial upsampler, optional audio). Exposes a thin Protocol (`FastVideoPipeline`) and the concrete `LTXFastVideoPipeline` implementation that owns model lifecycle, GGUF/split-safetensors patch installation, FP8 quantization policy selection, and frame-to-file output encoding.

Files:
- `fast_video_pipeline.py` — `FastVideoPipeline` Protocol (`pipeline_kind: ClassVar[Literal["fast"]]`, `create`, `generate`, `warmup`, `compile_transformer`).
- `ltx_fast_video_pipeline.py` — `LTXFastVideoPipeline` implementation.

## Design Patterns

- **Protocol + concrete wrapper**: handler layer depends on `FastVideoPipeline`; `LTXFastVideoPipeline.create(...)` is the only constructor.
- **Deferred imports**: all `ltx_pipelines.*`, `ltx_core.*`, and `services.patches.*` imports happen inside `__init__`/methods (lazy load on first construction), keeping module import cheap.
- **Format-aware construction**: `transformer_format` (`"safetensors"` default, or `"gguf"`) plus `ResolvedLtxComponents` drives three branches: GGUF loader path, split-safetensors (Kijai) path, and the plain single-file path. `is_split` = safetensors + `components.video_vae_path is not None`.
- **Quantization policy selection**: GGUF → `None`; split + cuda → `kijai_fp8_quantization_policy()`; otherwise `QuantizationPolicy.fp8_cast()` if `device_supports_fp8(device)` else `None`.
- **Streaming prefetch defaulting**: split 22B on ≤32 GB GPUs defaults `streaming_prefetch_count` to 2 layers unless caller overrides.
- **Patch installation** (via `services.patches.gguf_loader_fix`): `install_gguf_t2v_conditioning_patch()` (always), `install_gguf_prompt_encoder_patch()` (gguf or gemma_root), `install_gguf_loader` + `install_gguf_component_paths` (gguf), `install_kijai_transformer_config_patch` + `install_gguf_component_paths` (split).
- **`@torch.inference_mode()`** on `generate` and `warmup`.

## Data & Control Flow

### Frame production (a)

`generate(prompt, seed, height, width, num_frames, frame_rate, images, output_path, enhance_prompt=False)`:

1. `tiling_config = default_tiling_config()` (from `services.ltx_pipeline_common`, returns `TilingConfig.default()`).
2. `video, audio = self._run_inference(...)` — delegates to `self.pipeline(...)` (the wrapped `DistilledPipeline`). Returns `tuple[torch.Tensor | Iterator[torch.Tensor], AudioOrNone]`. Frames come back already VAE-decoded by the underlying pipeline; the wrapper does no decoding itself.
3. `chunks = video_chunks_number(num_frames, tiling_config)` (calls `ltx_core.model.video_vae.get_video_chunks_number`).

### Encode call site (b)

Line 180 — single encode call at the frame→encode boundary:

```python
encode_video_output(
    video=video,                                   # torch.Tensor | Iterator[torch.Tensor], decoded frames
    audio=audio,                                   # AudioOrNone
    fps=int(frame_rate),
    output_path=output_path,
    video_chunks_number_value=chunks,
)
```

`encode_video_output` is imported from `services.ltx_pipeline_common` and forwards unchanged to `ltx_pipelines.utils.media_io.encode_video(video=, fps=, audio=, output_path=, video_chunks_number=)`.

`warmup(output_path)` (line 182) runs the same path with fixed 256×384×9f@8fps dummy inputs, calls `encode_video_output` at line 200, then `os.unlink(output_path)` in the `finally` block — the warmup artifact is never kept.

### Output path hardcoding (c)

`output_path` is a caller-supplied `str`; the wrapper does not enforce or rewrite the extension. Callers (backend handlers) currently always pass a `.mp4` path, and the downstream `ltx_pipelines.utils.media_io.encode_video` hardcodes H.264 / yuv420p regardless of the extension. To add MOV ProRes / EXR as primary output, the routing point is this single `encode_video_output(...)` call (line 180) — swap the encoder service while keeping `video`/`audio`/`fps`/`output_path`/`chunks` arguments identical. `warmup` (line 200) needs the same swap if warmup should exercise the new encoder.

## Integration Points

- **`services.ltx_pipeline_common`**: imports `default_tiling_config`, `video_chunks_number`, and the shared `encode_video_output` wrapper (the single encode chokepoint for this pipeline).
- **`services.ltx_components`**: `CheckpointPath`, `ResolvedLtxComponents` (carries `transformer_format`, `video_vae_path`, `audio_vae_path`, `gemma_root`).
- **`services.services_utils`**: `AudioOrNone`, `TilingConfigType`, `device_supports_fp8`.
- **`services.patches.gguf_loader_fix`**: GGUF/Kijai loader and FP8 patches.
- **`api_types.ImageConditioningInput`**: input image DTO (`.path`, `.frame_idx`, `.strength`), remapped to `ltx_pipelines.utils.args.ImageConditioningInput` before being passed downstream.
- **`ltx_pipelines.distilled.DistilledPipeline`**: the underlying distilled model pipeline.
- **`ltx_pipelines.utils.media_io.encode_video`**: ultimate encoder (H.264/yuv420p hardcoded) reached via `encode_video_output`. See `backend/services/ltx_pipeline_common/codemap.md`.
- Handler layer constructs via `LTXFastVideoPipeline.create(...)` and calls `.generate(...)`; see the relevant handler codemap for caller-supplied `output_path` resolution.
