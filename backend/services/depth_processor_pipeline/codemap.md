# backend/services/depth_processor_pipeline/

## Responsibility

Frame-level monocular depth estimation used as **IC-LoRA conditioning preprocessing under Union Control**. Per `AGENTS.md`: depth runs ONLY when Union Control is explicitly enabled; non-Union-LoRA flows never invoke this pipeline. Loads a HuggingFace `DPTForDepthEstimation` (MiDaS DPT-Hybrid) model + `DPTImageProcessor`, runs one frame at a time, and returns a colorized (Inferno colormap) depth map with the same H×W as the input frame. Exposes `DepthProcessorPipeline` Protocol and `MidasDPTPipeline` concrete wrapper.

Files:
- `depth_processor_pipeline.py` — `DepthProcessorPipeline` Protocol (`create(model_path, device)`, `apply(frame) -> FrameArray`).
- `midas_dpt_pipeline.py` — `MidasDPTPipeline` implementation (HF Transformers DPT).

## Design Patterns

- **Protocol + concrete wrapper**: caller depends on `DepthProcessorPipeline`; `MidasDPTPipeline.create(...)` is the only constructor.
- **Stateless frame transform**: single-frame API `apply(frame: FrameArray) -> FrameArray`. No batch dimension, no temporal context, no video container awareness — the caller iterates frames and (if needed) reassembles the depth video outside this service.
- **HF Transformers delegation**: `DPTImageProcessor.from_pretrained(model_path)` for preprocessing, `DPTForDepthEstimation.from_pretrained(model_path, torch_dtype=dtype, device_map=device, low_cpu_mem_usage=True)` for inference; `model.eval()`.
- **Dtype policy**: `torch.float16` when `device.type == "cuda"`, else `torch.float32`.
- **Min/max normalization + colormap**: linear depth → normalized to `[0,1]` via per-frame min/max (zero-map guard when `max-min ≤ 1e-6`) → `uint8` → `cv2.applyColorMap(..., cv2.COLORMAP_INFERNO)` → BGR `uint8` FrameArray.
- **`@torch.inference_mode()`** on `apply`.
- **Deferred imports** of `cv2`, `numpy`, `PIL.Image`, and HF Transformers classes inside `__init__` / `apply`.

## Data & Control Flow

### Frame production (a)

`apply(frame: FrameArray)` (lines 45–77) — input `frame` is a BGR `uint8` ndarray (H, W, 3):

1. `h, w = frame.shape[:2]`.
2. Convert BGR → RGB → `PIL.Image.Image`.
3. `inputs = self._image_processor(images=image, return_tensors="pt")`; cast all values to `(self._device, self._dtype)`.
4. `predicted_depth = self._model(**inputs).predicted_depth`.
5. `torch.nn.functional.interpolate(predicted_depth.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False)[0, 0]` → back to original H×W.
6. `depth_np = depth.detach().float().cpu().numpy()`.
7. Per-frame min/max normalize → `[0,1]` → `uint8` (zeros if degenerate range).
8. `colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)` → BGR `uint8`.
9. Return `cast(FrameArray, colored)` (same H×W×3 BGR uint8 as input).

### Encode call site (b)

**None.** This pipeline produces single-frame ndarrays only. It does not call `encode_video_output`, `encode_video`, or any video muxer; it has no `output_path` parameter. Persistence of the depth-map stream (if any) is the caller's responsibility — the caller iterates frames, calls `apply` per frame, and assembles/writes the output elsewhere.

### Output path hardcoding (c)

**Not applicable directly.** No `output_path` is accepted here. However, callers that assemble depth-map frames into a conditioning video currently write `.mp4` (H.264/yuv420p) via the same `ltx_pipelines.utils.media_io.encode_video` chokepoint when building the Union Control conditioning stream — that caller-side encode is in scope for the MOV/EXR work, not this service.

## Integration Points

- **`services.services_utils`**: `FrameArray` (BGR `uint8` ndarray alias; `NDArray[np.uint8]` under TYPE_CHECKING, `object` at runtime).
- **`transformers.DPTForDepthEstimation`** + **`transformers.DPTImageProcessor`**: underlying MiDaS DPT-Hybrid model.
- **`cv2` (OpenCV)**: BGR↔RGB conversion, `applyColorMap(..., COLORMAP_INFERNO)`.
- **`PIL.Image`**: intermediate RGB image for the HF processor.
- **`torch`**: model loader, interpolate, dtype policy.
- **Union Control / IC-LoRA**: consumed (frame-by-frame) by the IC-LoRA flow when Union Control is enabled; see `backend/services/ic_lora_pipeline/codemap.md`. Per `AGENTS.md`: Union Control loads/applies first, then the selected LoRA; depth conditioning never runs for non-Union LoRAs.
- No dependency on `services.ltx_pipeline_common`, `ltx_pipelines.*`, or `ltx_core.*`.
- Handler layer constructs via `MidasDPTPipeline.create(model_path, device)` and calls `.apply(frame)` per frame; see the relevant handler codemap for the iteration/assembly loop.
