# Primary Output Formats Architecture Plan

## Status
ready-for-review — architecture plan only, no implementation packet

## User Goal
Add MOV ProRes and EXR sequence as primary generation output formats from decoded frames, not from MP4 transcode. Existing MP4/H.264 remains default.

## Sources Used
- `AGENTS.md`: repo workflow, backend/frontend architecture, validation commands.
- `subagent-artifacts/primary-output-formats-locator.md`: high-confidence trace for encode bottleneck, handlers, frontend playback, project model, tests, and risks.
- Windowed/source checks around locator anchors: `backend/services/ltx_pipeline_common.py`, generation pipeline callers, `backend/api_types.py`, handlers, `backend/.venv/lib/python3.13/site-packages/ltx_pipelines/utils/media_io.py`.

## Core Decision
Do not generate MP4 then transcode it to ProRes/EXR. Route decoded frame tensors (`torch.Tensor | Iterator[torch.Tensor]`) plus decoded audio into a backend-owned primary output encoder. MP4 proxy files may be created for playback/timeline only; they are sidecars and must not replace the primary output.

## Non-Goals
- No timeline EXR frame-accurate editing in first implementation; use proxy playback.
- No direct edit of `backend/.venv/.../ltx_pipelines/utils/media_io.py` unless no repo-level wrapper can satisfy retake/generation paths.
- No frontend player custom codec implementation.
- No generated model re-downloads or model pipeline behavior changes.

---

## Proposed API Shape

### Backend type alias
Add in `backend/api_types.py`:

- `GenerationOutputFormat = Literal[`
  - `"mp4_h264"`,
  - `"mov_prores_proxy"`,
  - `"mov_prores_lt"`,
  - `"mov_prores_422"`,
  - `"mov_prores_422_hq"`,
  - `"mov_prores_4444"`,
  - `"mov_prores_4444_xq"`,
  - `"exr_zip_half"`,
  - `"exr_zip_float"`,
  - `]`

### Request fields
Match existing per-endpoint naming style:

- `GenerateVideoRequest.outputFormat: GenerationOutputFormat = "mp4_h264"`
- `IcLoraGenerateRequest.output_format: GenerationOutputFormat = "mp4_h264"`
- `RetakeRequest.output_format: GenerationOutputFormat = "mp4_h264"`

Forced/API-backed generation and API retake cannot satisfy non-MP4 primary output because returned bytes are already encoded video, not decoded frames. For non-`mp4_h264` requests on API paths: return `400` with clear message, or route to local-only generation if policy allows. Do not transcode API MP4 into ProRes and call it primary.

### Response fields
Keep existing `video_path` for compatibility, but define it as primary output path even when extension is `.mov` or `.zip`.

Add:

- `output_format: GenerationOutputFormat`
- `preview_video_path: str | None = None` — MP4/H.264 proxy for non-browser-safe formats.
- `is_playable_in_browser: bool` — true for default MP4 and maybe MOV on macOS only if not using proxy; false for EXR primary.

Longer-term rename `video_path` to `asset_path`, but do not force that in first change.

---

## Backend Encode Abstraction

### New code area
Create `backend/services/output_encoder.py` (or similarly named service) as repo-owned wrapper. Purpose: single primary output path for decoded frames.

Minimal API:

- `extension_for_output_format(format: GenerationOutputFormat) -> str`
- `encode_primary_output(video, audio, fps, output_path, video_chunks_number, output_format) -> EncodeResult`
- `EncodeResult(primary_path: str, output_format: str, preview_video_path: str | None, manifest_path: str | None)`

### Why repo-owned wrapper
Locator shows installed `ltx_pipelines.utils.media_io.encode_video()` is the true bottleneck and hardcodes `libx264`. Editing `.venv` is fragile. Copy the small existing encode loop into a repo service and parameterize it. Then route all generation wrappers through this service.

### Files / exact anchors
- `backend/services/ltx_pipeline_common.py`
  - symbol: `encode_video_output()` lines 35–51
  - change: call repo output encoder instead of `ltx_pipelines.utils.media_io.encode_video`; add `output_format` parameter.
- `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
  - anchors: calls `encode_video_output(...)` around lines ~180 and warmup ~200
  - change: generation call accepts and passes `output_format`; warmup stays `mp4_h264`.
- `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
  - anchor: `encode_video_output(...)` around ~172
  - change: accept/pass `output_format`.
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  - anchors: `encode_video_output(...)` around ~431 and ~825
  - change: accept/pass `output_format` for final primary output; keep temp/control videos MP4.
- `backend/services/retake_pipeline/ltx_retake_pipeline.py`
  - anchor: direct import `from ltx_pipelines.utils.media_io import encode_video` and direct call around ~361
  - change: import/use repo output encoder wrapper; accept/pass `output_format`.

### Format mapping

| Output format | Extension | Video codec/path | Pixel format | Audio |
|---|---:|---|---|---|
| `mp4_h264` | `.mp4` | `libx264` | `yuv420p` | AAC |
| `mov_prores_proxy` | `.mov` | `prores_ks`, profile `0` | `yuv422p10le` | PCM s16le |
| `mov_prores_lt` | `.mov` | `prores_ks`, profile `1` | `yuv422p10le` | PCM s16le |
| `mov_prores_422` | `.mov` | `prores_ks`, profile `2` | `yuv422p10le` | PCM s16le |
| `mov_prores_422_hq` | `.mov` | `prores_ks`, profile `3` | `yuv422p10le` | PCM s16le |
| `mov_prores_4444` | `.mov` | `prores_ks`, profile `4` | `yuva444p10le` or validated `yuv444p10le` fallback | PCM s16le |
| `mov_prores_4444_xq` | `.mov` | `prores_ks`, profile `5` | `yuva444p12le` if available, else reject | PCM s16le |
| `exr_zip_half` | `.zip` | OpenEXR frames in ZIP | RGB half float | `audio.wav` sidecar in ZIP if audio exists |
| `exr_zip_float` | `.zip` | OpenEXR frames in ZIP | RGB float32 | `audio.wav` sidecar in ZIP if audio exists |

Implementation note: generated frames are RGB tensors. The encoder owns normalization and dtype conversion once: support current `uint8` RGB frames and float RGB frames. For ProRes, feed `av.VideoFrame.from_ndarray(..., format="rgb24")` then let PyAV/FFmpeg convert to stream `pix_fmt`. For EXR, write linear-ish decoded RGB values as channel arrays; include `manifest.json` with fps, width, height, frame count, output format, source app version if available, and audio sidecar name.

### EXR ZIP layout

```
manifest.json
frames/frame_000000.exr
frames/frame_000001.exr
...
audio.wav   # only when decoded audio exists
```

Use ZIP compression for archive; OpenEXR frame compression should be ZIP too. Clean temp dir on success/failure.

---

## Handler / Path Plan

### `backend/handlers/video_generation_handler.py`
- `_make_output_path()` around line 397 becomes `_make_output_path(output_format)` and uses `extension_for_output_format()`.
- Local fast generation and A2V pass request format through pipeline.
- Forced/API generation rejects non-`mp4_h264` because bytes are not decoded frames.
- Responses include `output_format` and optional `preview_video_path`.

### `backend/handlers/ic_lora_handler.py`
- Final output around line 457 uses `extension_for_output_format(req.output_format)`.
- Pass `req.output_format` into `generate()` / `generate_inpaint()`.
- Leave conditioning cache/control files around line 406 as internal `.mp4`; not primary output.

### `backend/handlers/retake_handler.py`
- Local output around line 149 uses selected format extension.
- API result bytes around line 114 remain MP4-only; reject non-`mp4_h264` before API request.
- Pass format into local retake pipeline.

---

## Frontend / Player / Timeline Plan

### Format selector
Add generation output selector in `frontend/views/GenSpace.tsx` near existing resolution/fps/duration controls. Default `mp4_h264`. Show short warnings:

- ProRes: large files; proxy used for playback where needed.
- EXR ZIP: primary output is image sequence archive; MP4 proxy used for preview/timeline.

IC-LoRA and Retake panels need the same field in submit payloads if they expose output choice. If UX should avoid clutter, first implementation can centralize selector in GenSpace and pass it to all generation modes.

### Project model
Locator shows `frontend/types/project-model.ts` has `asset.path: z.string()` only. Extend asset metadata without breaking old projects:

- `outputFormat?: GenerationOutputFormat`
- `previewPath?: string | null`
- `primaryPath?: string` optional alias if keeping `path` as primary is too risky
- `isSequence?: boolean`

Lazy compatibility default: existing assets with no `outputFormat` are `mp4_h264`, `previewPath = null`, `path` remains primary.

### Copy/import/proxy strategy
Critical risk: `electron/ipc/file-handlers.ts:transcodeVideoInPlace()` currently transcodes project video copies to H.264/AAC for browser playback. That would destroy ProRes if it runs on the only project copy.

Required change:

1. Preserve primary generated file untouched in project storage.
2. Generate or copy MP4 proxy separately for non-`mp4_h264` outputs.
3. Store primary path and proxy path distinctly.
4. UI playback uses `previewPath ?? path`; download/reveal/source-of-truth uses primary `path`.

MOV ProRes preview can use backend/electron proxy creation for all platforms, even if macOS can play ProRes directly. Simpler UX, fewer codec branches.

EXR preview must use MP4 proxy because `<video>` cannot play ZIP/EXR. Timeline can use proxy clip initially; primary EXR ZIP remains downloadable/export-source only until image-sequence timeline support exists.

### Playback call sites
Update consumers currently using `pathToFileUrl(asset.path)` for videos to use a helper like `assetPlaybackPath(asset)`:

- `frontend/views/GenSpace.tsx` hover/detail video cards.
- `frontend/components/ICLoraPanel.tsx` output preview.
- `frontend/components/RetakePanel.tsx` selected input video remains video-only; EXR ZIP should not be valid retake input until sequence import exists.
- `frontend/views/editor/ProgramMonitor.tsx` timeline playback.
- `frontend/views/editor/VideoEditorSourceMonitor.tsx` source monitor.

Do not make EXR ZIP selectable as retake/IC-LoRA driving video unless using its MP4 proxy explicitly.

---

## Phased Delivery Plan

### Phase 0 — Decisions / gates
- Choose EXR dependency: prefer `OpenEXR` Python package if it supports Python 3.13 in target env; fallback `imageio[freeimage]` or block EXR until dependency verified.
- Confirm EXR with audio policy: plan default is `audio.wav` inside ZIP plus MP4 proxy with audio.
- Confirm API-backed generation behavior: reject non-MP4 primary output.

### Phase 1 — Backend contracts and path extensions
- Add output format literal/type in `backend/api_types.py`.
- Add request/response fields for generation, IC-LoRA, retake.
- Add output path extension mapping.
- Handler validation rejects unsupported API paths.
- Tests: path extension + invalid enum + API non-MP4 rejection.

### Phase 2 — Backend primary encoder
- Add repo-owned `backend/services/output_encoder.py`.
- Move/copy existing H.264 loop from `media_io.py` into wrapper.
- Add ProRes profiles via `prores_ks` and ffprobe/PyAV validation tests.
- Add EXR ZIP writer with temp-dir cleanup and manifest.
- Route `ltx_pipeline_common.encode_video_output()` and retake direct encode through wrapper.
- Tests: tiny 2-frame tensor encode for H.264, ProRes, EXR half/float; EXR value tolerance check.

### Phase 3 — Pipeline propagation
- Thread `output_format` through fast, A2V, IC-LoRA, IC-LoRA inpaint, and retake final output calls.
- Warmup remains MP4.
- Temp/control conditioning files remain MP4.
- Tests: handler/pipeline fake or small integration proves selected extension reaches encoder without full model execution.

### Phase 4 — Proxy and project persistence
- Add proxy path in backend response or generate proxy during Electron project copy.
- Change `addVisualAssetToProject` flow so primary ProRes/EXR is never overwritten by H.264 transcode.
- Extend project asset metadata with output format + preview path.
- Add playback path helper.
- Tests/typecheck: project schema defaults old assets; IPC copy preserves primary and creates proxy path.

### Phase 5 — Frontend UX
- Add format selector in GenSpace and include in normal, IC-LoRA, retake submit payloads.
- Display primary/proxy badges in asset card/details where useful.
- Hide/disable EXR ZIP as direct retake/IC-LoRA input unless proxy selected.
- Regenerate OpenAPI/generated TS types after backend API changes.

### Phase 6 — Validation / release checks
- `rtk pnpm typecheck`
- `rtk pnpm backend:test -- tests/test_output_encoder.py`
- `rtk pnpm backend:test -- tests/test_generation_output_formats.py`
- `rtk pnpm build:frontend`
- Manual QA on at least one target OS: generate MP4, ProRes 422 HQ MOV, EXR half ZIP; import to project; verify primary file extension/codec, proxy playback, timeline playback, download returns primary.

---

## Test Plan

Backend:
- `test_output_format_extension_map`: every enum maps to expected extension and codec spec.
- `test_encode_h264_from_decoded_frames`: tiny RGB tensor → `.mp4`, readable by PyAV.
- `test_encode_prores_422_hq_from_decoded_frames`: tiny RGB tensor → `.mov`, stream codec `prores`, profile/pix_fmt expected where ffmpeg exposes it.
- `test_encode_exr_zip_half`: tiny float tensor → ZIP with manifest + EXR frames; read back within float16 tolerance.
- `test_encode_exr_zip_float`: same with float32 tolerance.
- `test_exr_zip_includes_audio_sidecar_when_audio_present` if audio fixture is cheap; otherwise unit-test WAV sidecar writer.
- `test_handlers_choose_extension_from_output_format` for video, IC-LoRA, retake.
- `test_api_paths_reject_non_mp4_primary_output` for forced/API generation.

Frontend/Electron:
- Project schema migration/default: old asset without `outputFormat` still parses.
- Playback helper returns `previewPath` for ProRes/EXR assets, `path` for MP4.
- `addVisualAssetToProject` preserves primary path and does not transcode primary ProRes in place.
- Typecheck generated API usage.

Manual:
- MOV ProRes opens in Resolve/QuickTime/ffprobe.
- EXR ZIP opens in external EXR-aware tool; manifest fps/frame count correct.
- Timeline preview uses MP4 proxy and does not claim EXR is natively editable.

---

## Risks / Mitigations

- FFmpeg/PyAV ProRes support varies by platform. Mitigation: use `prores_ks`, add startup/test-time codec availability check, reject unsupported ProRes flavors with clear error.
- ProRes 4444/XQ pixel formats may fail depending on encoder build. Mitigation: validate at encode start with tiny frame or feature-probe; keep 422 HQ as recommended default.
- EXR dependency may not support Python 3.13 or packaging. Mitigation: verify dependency before implementation; make EXR phase gated.
- EXR cannot carry audio as video container. Mitigation: include `audio.wav` in ZIP + MP4 proxy with audio.
- Existing project copy transcode can destroy primary ProRes. Mitigation: preserve primary and create proxy sidecar; never run in-place transcode on primary.
- Large files and temp dirs. Mitigation: stream chunks, clean temp dirs on failure, surface estimated sizes.
- Generated OpenAPI/types may drift. Mitigation: regenerate frontend API types in same phase as `api_types.py` changes.

## Reviewer Gate
Required before implementation packets.

## Planner Self-Check
- locator evidence sufficient: yes — locator is high confidence and source windows confirm encode bottleneck/callers.
- allowed edit files minimal and explicit: not applicable — architecture plan only; no implementation packet yet.
- read-only context minimal: yes — only `AGENTS.md`, locator, and anchor windows/searches around locator evidence.
- anchors/lines included: yes — key backend/frontend anchors included from locator.
- validation concrete: yes — commands and test targets listed.
- parallelization decision explicit and safe: yes — phases are sequential because API shape, generated types, project schema, and encoder propagation share files/state.
- non-goals and stop conditions sufficient: yes — no MP4 transcode as primary, no EXR native timeline promise, no `.venv` edit unless unavoidable.
- reviewer findings addressed, if revision: not applicable — no reviewer findings supplied.
