# backend/services/pose_processor_pipeline/

## Responsibility

Frame-level 2D human pose estimation used as **IC-LoRA conditioning preprocessing under Union Control**. Per `AGENTS.md`: pose runs ONLY when Union Control is explicitly enabled; non-Union-LoRA flows never invoke this pipeline. Loads two TorchScript models (a YOLOX person detector and a DW-Pose SimCC keypoint estimator), runs one BGR frame at a time, and returns an OpenPose-style body+hand+face skeleton rendering on a black canvas with the same H×W×3 as the input frame. Exposes `PoseProcessorPipeline` Protocol and `DWPosePipeline` concrete wrapper.

Files:
- `pose_processor_pipeline.py` — `PoseProcessorPipeline` Protocol (`create(pose_model_path, person_detector_model_path, device)`, `apply(frame) -> FrameArray`).
- `dw_pose_pipeline.py` — `DWPosePipeline` implementation (TorchScript YOLOX + SimCC pose).

## Design Patterns

- **Protocol + concrete wrapper**: caller depends on `PoseProcessorPipeline`; `DWPosePipeline.create(...)` is the only constructor.
- **Stateless frame transform**: single-frame API `apply(frame: FrameArray) -> FrameArray`. No batch dimension over time, no video container awareness — caller iterates frames and reassembles the pose video outside this service.
- **Two-stage top-down pose estimation**: (1) YOLOX person detection (multi-class NMS, person class only) → bounding boxes; (2) per-box top-down affine warp → DW-Pose SimCC inference → keypoint decode → unwarp → render. Mirrors the standard MMPose top-down pipeline.
- **TorchScript model loading**: both `_pose_model` and `_detector_model` loaded via `torch.jit.load(path, map_location=device)`; `.eval()`. Device/dtype inferred from a model parameter via `_module_device_dtype`.
- **Pure-numpy postprocessing** (CPU): grid decode (`_detector_postprocess`), NMS (`_nms`, `_multiclass_nms`), SimCC argmax (`_simcc_maximum`), keypoint rescaling, instance reformatting, OpenCV drawing.
- **Constants**: `_DETECTOR_INPUT_SIZE=(640,640)`, `_POSE_INPUT_SIZE=(288,384)` (W,H), `_POSE_BATCH_SIZE=5`, `_SIMCC_SPLIT_RATIO=2.0`, `_CONFIDENCE_THRESHOLD=0.3`, `_EPS=0.01`.
- **`@torch.inference_mode()`** on `apply`.
- **Deferred imports** of `cv2`, `numpy` inside each method that needs them.

## Data & Control Flow

### Frame production (a)

`apply(frame: FrameArray)` (lines 584–599) — input `frame` is a BGR `uint8` ndarray (H, W, 3):

1. `boxes = self._detect_person_boxes(frame)`; if empty → return `np.zeros(frame.shape, uint8)` (black canvas).
   - `_detector_preprocess`: letterbox-pad to 640×640, transpose to CHW float32.
   - `self._detector_model(input_tensor)` → raw output → `_detector_postprocess` (grid + stride decode) → xyxy boxes + per-class scores → `_multiclass_nms(nms_threshold=0.45, score_threshold=0.1)` → keep person class (0) with score > `_CONFIDENCE_THRESHOLD` (0.3).
2. `images, centers, scales = self._preprocess_pose(frame, boxes)` — per box: `_bbox_xyxy_to_center_scale(padding=1.25)` → `_top_down_affine` (warp to 288×384 via `_warp_matrix`) → ImageNet normalize (`mean=[123.675,116.28,103.53]`, `std=[58.395,57.12,57.375]`).
3. `keypoints, scores = self._infer_pose_model(images)`:
   - Pad to multiple of `_POSE_BATCH_SIZE=5`; stack + transpose to (N, C, H, W); batched inference (5 at a time); concatenate simcc_x/simcc_y outputs.
   - `_decode_pose_outputs` → `_simcc_maximum` (argmax x/y per keypoint) → divide by `_SIMCC_SPLIT_RATIO=2.0`.
4. If empty → return black canvas.
5. `rescaled_keypoints = self._rescale_keypoints(keypoints, centers, scales)` (unwarp to original frame coords).
6. `instances = self._format_instances(rescaled_keypoints, scores)` — insert synthetic neck keypoint (mean of shoulders), remap MMPose→OpenPose indices; slice body[:18], face[24:92]+body eyes/nose, left_hand[92:113], right_hand[113:134].
7. `return self._render_instances(instances, canvas_shape=frame.shape)` — black `uint8` canvas; `_draw_body_pose` (ellipses between joints, 18-color palette), `_draw_hand_pose` (21-edge HSV-gradient lines per hand), `_draw_face_pose` (white dots). Points below `_CONFIDENCE_THRESHOLD` are dropped (`_to_optional_point` returns `None`); points near (0,0) (`≤ _EPS`) are skipped during hand/face drawing.

### Encode call site (b)

**None.** This pipeline produces single-frame ndarrays only. It does not call `encode_video_output`, `encode_video`, or any video muxer; it has no `output_path` parameter. Persistence of the pose-stream (if any) is the caller's responsibility — the caller iterates frames, calls `apply` per frame, and assembles/writes the output elsewhere.

### Output path hardcoding (c)

**Not applicable directly.** No `output_path` is accepted here. However, callers that assemble pose frames into a conditioning video currently write `.mp4` (H.264/yuv420p) via the same `ltx_pipelines.utils.media_io.encode_video` chokepoint when building the Union Control conditioning stream — that caller-side encode is in scope for the MOV/EXR work, not this service.

## Integration Points

- **`services.services_utils`**: `FrameArray` (BGR `uint8` ndarray alias).
- **`torch.jit` / `torch.jit.ScriptModule`**: TorchScript model loading for both detector and pose estimator.
- **`cv2` (OpenCV)**: resize (`INTER_LINEAR`), `warpAffine`, `getAffineTransform`, `ellipse2Poly`, `fillConvexPoly`, `circle`, `line`, `cvtColor` (HSV→BGR for hand edges).
- **`numpy`**: all postprocessing (NMS, grid decode, SimCC argmax, keypoint remap).
- **Union Control / IC-LoRA**: consumed (frame-by-frame) by the IC-LoRA flow when Union Control is enabled; see `backend/services/ic_lora_pipeline/codemap.md`. Per `AGENTS.md`: Union Control loads/applies first, then the selected LoRA; pose conditioning never runs for non-Union LoRAs.
- No dependency on `services.ltx_pipeline_common`, `ltx_pipelines.*`, or `ltx_core.*`.
- Handler layer constructs via `DWPosePipeline.create(pose_model_path, person_detector_model_path, device)` and calls `.apply(frame)` per frame; see the relevant handler codemap for the iteration/assembly loop.
