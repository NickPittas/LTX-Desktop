# backend/services/video_processor/

## Responsibility
Thin cv2 (OpenCV) wrapper service for all video decode / frame processing used by
the IC-LoRA flow: opening source videos, reading their metadata (fps, frame count,
dimensions), reading/seeking frames, applying edge (Canny) / depth / pose
conditioning, JPEG-encoding conditioning frames, and creating/releasing video
writers for output. Central to deriving the `fps`/`width`/`height`/`frame_count`
inputs that drive V2V IC-LoRA generation.

## Design Patterns
- **Protocol-first service**: `video_processor.py` declares the `VideoProcessor`
  Protocol and the `VideoInfoPayload` TypedDict (`fps`, `frame_count`, `width`,
  `height`); `video_processor_impl.py` provides `VideoProcessorImpl` — the only
  production implementation. Re-exported via `services/interfaces.py` /
  `services/__init__.py`.
- **Lazy cv2/numpy import**: every method does `import cv2` (and `import numpy as np`
  in `apply_canny`) inside the function body, so importing this module never
  requires cv2 at module load.
- **Structural typing over concrete types**: parameters are typed as
  `VideoCaptureLike` / `VideoWriterLike` / `FrameArray` (from
  `services/services_utils.py`) and results are `cast(...)` back — keeps cv2 out of
  the type graph.
- **Delegation to sibling pipelines**: `apply_depth` and `apply_pose` do no image
  work themselves — they forward to `DepthProcessorPipeline.apply(frame)` and
  `PoseProcessorPipeline.apply(frame)` respectively, keeping this service a pure
  cv2/coordination layer.
- **Test seam**: handler tests substitute the service via the Protocol; cv2 itself
  is never invoked in the unit suite.

## Data & Control Flow
**Metadata extraction (drives V2V IC-LoRA sizing)**:
1. `open_video(path)` → `cv2.VideoCapture(path)` cast to `VideoCaptureLike`.
2. `get_video_info(cap)` → reads `CAP_PROP_FPS` (default `24.0` when missing),
   `CAP_PROP_FRAME_COUNT`, `CAP_PROP_FRAME_WIDTH`, `CAP_PROP_FRAME_HEIGHT`; returns
   them as a `VideoInfoPayload`.

**Frame reading / seeking**:
- `read_frame(cap, frame_idx=None)` — if `frame_idx` given, first
  `cap.set(CAP_PROP_POS_FRAMES, frame_idx)`, then `cap.read()`; returns `None` on
  `ret=False`.

**Conditioning generation**:
- `apply_canny(frame)` — copies frame, pads H/W up to multiples of 64 (edge mode,
  for parity with training), `cv2.cvtColor(..., COLOR_BGR2GRAY)`,
  `cv2.Canny(gray, 100, 200)`, strips the padding, expands the single-channel edges
  to 3-channel (HWC3). Returns `FrameArray`.
- `apply_depth(frame, depth_pipeline)` → `depth_pipeline.apply(frame)`.
- `apply_pose(frame, pose_pipeline)` → `pose_pipeline.apply(frame)`.
- `encode_frame_jpeg(frame, quality=85)` → `cv2.imencode(".jpg", frame,
  [IMWRITE_JPEG_QUALITY, quality])`; raises `RuntimeError` on failure; returns
  `bytes`.

**Output writing**:
- `create_writer(path, fourcc, fps, size)` →
  `cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*fourcc), fps, size)`.
- `release(cap_or_writer)` — calls `.release()` inside try/except (warns on
  failure), so a double-release or bad handle never crashes a run.

## Integration Points
- **`services/services_utils.py`**: `FrameArray`, `VideoCaptureLike`,
  `VideoWriterLike` type aliases/Protocols used throughout the interface.
- **`handlers/ic_lora_handler.py`** (primary consumer):
  - Conditioning-frame path (~lines 249-269): `open_video` → `get_video_info` →
    `read_frame(cap, frame_idx=target_frame)` → `release(cap)`; then
    `apply_canny`/`apply_depth`/`apply_pose` on the frame; finally
    `encode_frame_jpeg(..., quality=85)` for both the conditioning result and the
    original frame.
  - V2V path (~lines 503-555): `open_video` → `get_video_info` (derives fps/dims/
    frame_count for the generation request) → frame loop with `read_frame` and
    per-frame conditioning → `create_writer` → final `release` of both cap and
    writer.
- **`services/depth_processor_pipeline/`** and
  **`services/pose_processor_pipeline/`**: `DepthProcessorPipeline` /
  `PoseProcessorPipeline` collaborators passed in by the caller (IC-LoRA handler).
- **`app_handler.py`**: constructs `VideoProcessorImpl` and injects it through the
  service bundle.
- **Canny/depth guard**: per repo policy, `apply_canny`/`apply_depth` are only
  invoked when Union Control is enabled; other LoRAs never touch this preprocessing
  path.
