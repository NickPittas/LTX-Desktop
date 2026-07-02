# backend/services/image_generation_pipeline/

## Responsibility

Standalone still-image generation (no video). Wraps `diffusers.pipelines.auto_pipeline.ZImagePipeline` (Z-Image-Turbo) to produce one or more PIL images from a text prompt. Used for image-only generation flows (e.g., reference frame generation, storyboarding). Exposes `ImageGenerationPipeline` Protocol and `ZitImageGenerationPipeline` concrete wrapper.

Files:
- `image_generation_pipeline.py` — `ImageGenerationPipeline` Protocol (`@runtime_checkable`; `create`, `generate`, `to`).
- `zit_image_generation_pipeline.py` — `ZitImageGenerationPipeline` implementation + private `_ZImageOutput` dataclass.

## Design Patterns

- **Protocol + concrete wrapper**: handler depends on `ImageGenerationPipeline` (runtime-checkable); `ZitImageGenerationPipeline.create(...)` is the only constructor.
- **diffusers delegation**: `ZImagePipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)` owns the entire model lifecycle; the wrapper adds device routing and output normalization only.
- **CPU offload strategy** (`to(device)`): on `cuda` or `mps` → `self.pipeline.enable_model_cpu_offload()` + `_cpu_offload_active=True`; otherwise → `self.pipeline.to(runtime_device)` + `_cpu_offload_active=False`. Generator device is resolved by `_resolve_generator_device()` (returns `"cuda"` if cpu-offload active, else `_device`, else `_execution_device` from the pipeline).
- **Output normalization** (`_normalize_output`): validates `output.images` is a `Sequence` of `PIL.Image.Image` and returns a `_ZImageOutput(images=validated_images)`; raises `RuntimeError` on shape mismatch. Bridges diffusers' untyped output to `ImagePipelineOutputLike`.
- **`guidance_scale` is dropped**: ZIT ignores it; the param is accepted for Protocol conformance, `_ = guidance_scale`, and `guidance_scale=0.0` is passed downstream.
- **`@torch.inference_mode()`** on `generate`.
- **`@dataclass(slots=True)`** for `_ZImageOutput`.

## Data & Control Flow

### Frame production (a)

`generate(prompt, height, width, guidance_scale, num_inference_steps, seed)` (lines 64–87):

1. Discards `guidance_scale` (`_ = guidance_scale`).
2. `generator = torch.Generator(device=self._resolve_generator_device()).manual_seed(seed)`.
3. `output = self.pipeline(prompt=prompt, height=height, width=width, guidance_scale=0.0, num_inference_steps=num_inference_steps, generator=generator, output_type="pil", return_dict=True)`.
4. `return self._normalize_output(output)` → `_ZImageOutput(images: Sequence[PILImageType])`.

### Encode call site (b)

**None.** This pipeline does not produce video and does not call `encode_video_output` / `encode_video` / any muxer. Output is in-memory PIL images only. The caller (handler / Electron layer) is responsible for any persistence (typically PNG/JPEG save, handled outside this service).

### Output path hardcoding (c)

**Not applicable.** No `output_path` parameter is accepted by `generate`; no file is written by this pipeline. This pipeline is explicitly out of scope for the MOV ProRes / EXR primary-output work (which targets decoded-frame video pipelines).

## Integration Points

- **`services.services_utils`**: `ImagePipelineOutputLike` (return type contract; has `.images: Sequence[PILImageType]`), `PILImageType`, `get_device_type`.
- **`diffusers.pipelines.auto_pipeline.ZImagePipeline`**: underlying diffusers pipeline (`from_pretrained`, `__call__`, `enable_model_cpu_offload`, `to`, `_execution_device`).
- **`PIL.Image.Image`**: output element type.
- **`torch`**: `torch.bfloat16` dtype, `torch.Generator`, `torch.inference_mode`.
- No dependency on `services.ltx_pipeline_common`, `ltx_pipelines.*`, or `ltx_core.*` — this is a pure-diffusers island.
- Handler layer constructs via `ZitImageGenerationPipeline.create(model_path, device=...)`, optionally calls `.to(device)` for device migration, then `.generate(...)`; image persistence is the caller's responsibility.
