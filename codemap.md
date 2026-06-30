# Repository Atlas: LTX-Desktop

> Master entry point. Read this before working on any task. Each sub-folder has its own
> `codemap.md` with deeper detail — links in the Directory Map below.

## Project Responsibility
LTX-Desktop is an Electron desktop application for **local/offline AI video generation using LTX-2.x models** (text-to-video, image-to-video, audio-to-video, IC-LoRA workflows, retake, and image generation). It wraps a Python FastAPI ML backend that orchestrates Lightricks `ltx_pipelines` + HuggingFace GGUF/Kijai/local transformer profiles, behind a React renderer and an Electron main process that supervises the backend lifecycle, IPC, and ffmpeg-based timeline export.

Three-layer architecture:
- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind renderer.
- **Electron** (`electron/`): Main process — app lifecycle, window, IPC bridge, Python-backend supervision, ffmpeg export.
- **Backend** (`backend/`): Python 3.13 FastAPI server (dev port `41954`, documented `8000`) handling ML model orchestration and generation.

## System Entry Points
| Entry | Role |
|---|---|
| `package.json` | Workspace root; pnpm scripts (`dev`, `dev:debug`, `typecheck`, `backend:test`, `build:frontend`, `openapi:generate/check`). `packageManager: pnpm@10.30.3`. |
| `electron/main.ts` | Electron app lifecycle; creates window, spawns Python backend, registers IPC, sets CSP + protocols. Compiled to `dist-electron/`. |
| `electron/preload.ts` | CommonJS preload exposing the typed `window.electronAPI` surface (contract in `shared/electron-api-schema.ts`). |
| `backend/ltx2_server.py` | FastAPI app bootstrap; uvicorn launch, SageAttention/faulthandler init, calls `app_factory.create_app()`. |
| `backend/app_factory.py` | Builds the FastAPI app: middleware, exception handlers, router registration, boundary-owned traceback logging. |
| `backend/app_handler.py` | `AppHandler` — single composition root owning all sub-handlers, `AppState`, shared `RLock`, thread pool, `ServiceBundle`. |
| `frontend/main.tsx` → `App.tsx` | Renderer mount; provider composition, boot/gating waterfall, view routing. |
| `vite.config.ts` / `tsconfig.json` | Vite + `vite-plugin-electron` build; path alias `@/*` → `frontend/*`. |
| `electron-builder.yml` | Packaging config. |
| `settings.json` | Default app-settings schema shipped with the app. |

## Core Architectural Conventions
- **Backend request flow**: `_routes/* (thin plumbing) → AppHandler → handlers/* (logic) → services/* (side effects) + state/* (mutations)`.
- **State**: centralized `AppState` with **discriminated-union** state machines (`GenerationRunning | Complete | Error | Cancelled`, plus HF-auth, download, GPU-slot pipeline unions).
- **Concurrency**: thread pool + shared `RLock`. Pattern: `lock → read/validate → unlock → heavy work → lock → write`. **Never hold the lock during heavy compute/IO.**
- **Services**: Protocol interfaces with real + `Fake*` test implementations (`ServiceBundle`). The test boundary for GPU/network side effects.
- **Exception handling**: boundary-owned traceback policy. Handlers raise `HTTPError(..., )` with `from exc` chaining; `app_factory.py` owns logging. No `logger.exception()` then rethrow.
- **Naming**: `*Payload` = DTO/TypedDict, `*Like` = structural wrapper, `Fake*` = test impl.
- **Frontend state**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`, `ViewContext`). `zustand` is in deps but used **only inside the editor store** (`frontend/views/editor/editor-store.tsx`), not for global app state.
- **Frontend→backend HTTP**: always `backendFetch` from `frontend/lib/backend.ts` (attaches auth/session). Never raw `fetch` to app backend.
- **Frontend→Electron**: always `window.electronAPI` (typed via `shared/electron-api-schema.ts` → `InvokeAPI` mapped type).
- **Testing**: backend integration-first via Starlette `TestClient` against the real FastAPI app; **no mocks** (`test_no_mock_usage.py` enforces), swap services via `ServiceBundle` fakes. Pyright strict mode is also a test (`test_pyright.py`). No frontend tests exist.
- **OpenAPI**: backend schema is the source of truth; `openapi:generate` exports JSON + regenerates `frontend/generated/backend-openapi.{json,ts}`; `openapi:check` asserts no drift in CI.

## Directory Map (Aggregated)
Deep-dive codemap for each folder lives at the linked path. Responsibility summaries below.

### Backend (`backend/`)
| Directory | Responsibility Summary | Detailed Map |
|---|---|---|
| `backend/` | FastAPI bootstrap, app factory (middleware/exception logging), `AppHandler` composition root, request/response DTOs (`api_types`), capability/validator matrices (`api_model_specs`), logging policy, OpenAPI tooling. | [backend/codemap.md](backend/codemap.md) |
| `backend/_routes/` | Thin route modules — HTTP method+path → `AppHandler` method delegation. Zero business logic. Admin-guarded routes, 404/409 mappings, localhost-only shutdown. | [backend/_routes/codemap.md](backend/_routes/codemap.md) |
| `backend/handlers/` | 18 domain handlers: lock-aware state transitions, generation/T2V resolution maps + frame-count logic, IC-LoRA workflow dispatch (Ingredients T2V no-video path; HDR V2V branch — official-distilled-safetensors-only gating, ignores prompt/audio/`conditioning_type`/`images`, no frame-count rejection, forwards `scene_embeddings_path`, EXR primary + SDR proxy; `_UNAVAILABLE_WORKFLOWS` for still-gated adapters like `motion_track_control`/`lipdub`; `lora_strength` forwarding), pipeline cache-key matching, `HTTPError` chaining convention. Non-HDR handlers hardcode `.mp4`; HDR emits an EXR dir + proxy MP4. | [backend/handlers/codemap.md](backend/handlers/codemap.md) |
| `backend/state/` | `AppState` + all discriminated-union state machines, `ICLoraState.lora_strength=1.0`, pydantic `AppSettings`/patch/response, `ConditioningCache`, AppHandler singleton in `deps.py`. | [backend/state/codemap.md](backend/state/codemap.md) |
| `backend/runtime_config/` | `RuntimeConfig` (+ derived flags), `LocalGenerationMode` policy (CUDA VRAM 15/31 thresholds, prefetch), `model_download_specs` catalog (`OFFICIAL_LTX23_ADAPTERS`, `ADAPTER_TO_CP_ID`), port constant `41954`. | [backend/runtime_config/codemap.md](backend/runtime_config/codemap.md) |
| `backend/server_utils/` | Media validation (image/audio magic-byte sniffing, size caps → HTTPError 400), legacy model-layout migration. | [backend/server_utils/codemap.md](backend/server_utils/codemap.md) |
| `backend/services/` | Service root: Protocol interfaces, `ltx_pipeline_common.py::encode_video_output()` **(central encode wrapper — EXR/MOV bottleneck)**, `ltx_components`, `base_video_model_registry` (**unified source of truth for generation-selectable base video variants** — drives model-options, scanner, resolver, family-mismatch guard), `services_utils`. | [backend/services/codemap.md](backend/services/codemap.md) |
| `backend/services/patches/` | Runtime monkey-patches: GGUF loader/torch-dequant, pinned-pool, record-stream, safetensors loader/metadata fixes. | [backend/services/patches/codemap.md](backend/services/patches/codemap.md) |
| `backend/services/http_client/` | `HTTPClient` Protocol + `requests`-backed impl, `HttpTimeoutError` translation. | [backend/services/http_client/codemap.md](backend/services/http_client/codemap.md) |
| `backend/services/gpu_info/` | `GpuInfo` Protocol + impl (pynvml/torch/sysctl cascade). | [backend/services/gpu_info/codemap.md](backend/services/gpu_info/codemap.md) |
| `backend/services/gpu_cleaner/` | `GpuCleaner` Protocol + `TorchCleaner` (`empty_cache` + `gc.collect`). | [backend/services/gpu_cleaner/codemap.md](backend/services/gpu_cleaner/codemap.md) |
| `backend/services/model_downloader/` | `ModelDownloader` Protocol + `HuggingFaceDownloader` (tqdm + http/xet monkey-patch progress). | [backend/services/model_downloader/codemap.md](backend/services/model_downloader/codemap.md) |
| `backend/services/task_runner/` | `TaskRunner` Protocol + `ThreadingRunner.run_background` (daemon thread, guarded error sink). | [backend/services/task_runner/codemap.md](backend/services/task_runner/codemap.md) |
| `backend/services/text_encoder/` | `LTXTextEncoder` idempotent monkey-patches; `/v1/prompt-embedding` pickle/split flow; local-profile API-key-suppression path. | [backend/services/text_encoder/codemap.md](backend/services/text_encoder/codemap.md) |
| `backend/services/video_processor/` | `VideoProcessor` Protocol + cv2 impl; derives fps/dims/frame count for V2V IC-LoRA; 64-pad Canny; depth/pose delegation. | [backend/services/video_processor/codemap.md](backend/services/video_processor/codemap.md) |
| `backend/services/ltx_api_client/` | `LTXAPIClient` Protocol, Pydantic parsers, 4 generation endpoints + 2-step upload, camera-motion mapping. | [backend/services/ltx_api_client/codemap.md](backend/services/ltx_api_client/codemap.md) |
| `backend/services/zit_api_client/` | `ZitAPIClient` (FAL `z-image/turbo`) submit-then-download, `Key` auth. | [backend/services/zit_api_client/codemap.md](backend/services/zit_api_client/codemap.md) |
| `backend/services/fast_video_pipeline/` | `LTXFastVideoPipeline.generate`; encode via shared `encode_video_output` (line ~180). | [backend/services/fast_video_pipeline/codemap.md](backend/services/fast_video_pipeline/codemap.md) |
| `backend/services/a2v_pipeline/` | `LTXa2vPipeline.generate`; two-stage distilled A2V with frozen-audio conditioning; encode at ~172. | [backend/services/a2v_pipeline/codemap.md](backend/services/a2v_pipeline/codemap.md) |
| `backend/services/ic_lora_pipeline/` | IC-LoRA Protocol + `ltx_ic_lora_pipeline` impl; two encode sites; single `lora_strength` over whole `LoraPathStrengthAndSDOps` stack; T2V no-video path via empty `video_conditioning`. | [backend/services/ic_lora_pipeline/codemap.md](backend/services/ic_lora_pipeline/codemap.md) |
| `backend/services/hdr_ic_lora_pipeline/` | HDR Protocol + `LTXHdrIcLoraPipeline` — **thin subclass of upstream `HDRICLoraPipeline`**; overrides only `_create_conditionings` (decode all source frames + in-memory duplicate-final-frame pad to `8n+1`, no temp/recompressed video); writes linear EXR primary via `save_exr_tensor` + SDR proxy via `encode_exr_sequence_to_mp4` (**bypasses `encode_video_output`**). | — |
| `backend/services/retake_pipeline/` | `ltx_retake_pipeline`; **calls `encode_video` DIRECTLY (import line 25, call 361–367), bypassing the common wrapper — must be re-routed for EXR/MOV.** | [backend/services/retake_pipeline/codemap.md](backend/services/retake_pipeline/codemap.md) |
| `backend/services/image_generation_pipeline/` | PIL-image-only output (no video encode path). | [backend/services/image_generation_pipeline/codemap.md](backend/services/image_generation_pipeline/codemap.md) |
| `backend/services/depth_processor_pipeline/` | Per-frame depth maps (Inferno); Union-Control-only; no encode. | [backend/services/depth_processor_pipeline/codemap.md](backend/services/depth_processor_pipeline/codemap.md) |
| `backend/services/pose_processor_pipeline/` | Per-frame OpenPose skeleton render; Union-Control-only; no encode. | [backend/services/pose_processor_pipeline/codemap.md](backend/services/pose_processor_pipeline/codemap.md) |

### Frontend (`frontend/`)
| Directory | Responsibility Summary | Detailed Map |
|---|---|---|
| `frontend/` | `App.tsx` provider composition, boot/gating waterfall, view routing (`home`/`project`), DOM CustomEvent command bus; `main.tsx` mount. | [frontend/codemap.md](frontend/codemap.md) |
| `frontend/contexts/` | `AppSettingsContext`, `KeyboardShortcutsContext`, `ProjectContext`, `ViewContext` — state machines, persistence, consumer wiring. | [frontend/contexts/codemap.md](frontend/contexts/codemap.md) |
| `frontend/types/` | Zod schema-first `project-model.ts` (V1→V2 migration), static catalogs in `project.ts`, OpenAPI façade in `model-profile.ts`. | [frontend/types/codemap.md](frontend/types/codemap.md) |
| `frontend/hooks/` | 10 hooks: backend-data, generation-flow, HF-gating, migration families; `use-ic-lora` conditional request body fields; `onComplete`-before-`setState` pattern. | [frontend/hooks/codemap.md](frontend/hooks/codemap.md) |
| `frontend/lib/` | `backend.ts`/`api-client.ts` (sole typed egress + two transports), project storage + migrations, `timeline-import.ts` (flags `transcodeVideoInPlace` in-place-destroy risk for ProRes/EXR plan). | [frontend/lib/codemap.md](frontend/lib/codemap.md) |
| `frontend/views/` | `GenSpace` (embedded `PromptBar`, Ingredients free-numeric FPS/duration/res/aspect vs V2V reuse source video, `canSubmit`⇔`handleGenerate` no-op fix), `Home`, `Project`, `VideoEditor` (largest file). | [frontend/views/codemap.md](frontend/views/codemap.md) |
| `frontend/views/editor/` | Store/selectors/actions split (Zustand vanilla + `setStateWithHistory`/`WithoutHistory`), undo/redo snapshot model (MAX 50), 164 action names, playback/drag/keyboard/regeneration hooks, 19 components. | [frontend/views/editor/codemap.md](frontend/views/editor/codemap.md) |
| `frontend/components/` | `ICLoraPanel` (Ingredients `videoPath:null`+`conditioningType:null`, no FPS controls), `RetakePanel` (VAE 1+8k frame snapping), model-download/HF-auth recipe, `GenerationError` discriminated union, `SettingsPanel.GenerationSettings`. | [frontend/components/codemap.md](frontend/components/codemap.md) |
| `frontend/components/ui/` | CVA primitives (button/progress/select/textarea/tooltip), `cn` (clsx+tailwind-merge), semantic Tailwind tokens. | [frontend/components/ui/codemap.md](frontend/components/ui/codemap.md) |
| `frontend/generated/` | Auto-generated OpenAPI JSON + TS types. **Excluded from codemap (regenerated by `openapi:generate`).** | — |

### Electron (`electron/`) & Shared (`shared/`)
| Directory | Responsibility Summary | Detailed Map |
|---|---|---|
| `electron/` | `main.ts` lifecycle/window/protocol, `preload.ts` API surface, config/paths/state/csp/gpu/logger, `python-backend.ts` supervision (managed/adopted ownership, liveness monitor, crash debounce), `python-setup.ts` download/stage/promote, `updater.ts`. | [electron/codemap.md](electron/codemap.md) |
| `electron/ipc/` | `typed-handle.ts` schema-typed wrapper, `app-handlers.ts` (backend admin proxy), `file-handlers.ts` (`transcodeVideoInPlace` H.264 proxy — ProRes/EXR proxy precedent), `image-utils.ts` (Pillow), `video-processing-handlers.ts`. | [electron/ipc/codemap.md](electron/ipc/codemap.md) |
| `electron/export/` | `export-handler.ts` three-pass export (video→audio→mux) with h264/prores/vp9 blocks. **Export-only, NOT primary generation.** `ffmpeg-utils.ts` binary resolver + cancellation, `timeline.ts` NLE flatten+merge, `video-filter.ts`, `audio-mix.ts`. | [electron/export/codemap.md](electron/export/codemap.md) |
| `shared/` | `electron-api-schema.ts` (Zod schemas, `ipcResult`, `InvokeAPI` mapped type, `ElectronAPI` contract + full channel inventory), `feature-flags.ts` (HF gating isolation). | [shared/codemap.md](shared/codemap.md) |

> Excluded from codemap (intentionally): `backend/tests/`, `backend/typings/` (third-party stubs), `backend/_services/` (empty), `frontend/generated/` (regenerated), `node_modules/`, `dist*/`, `build*/`, `release/`, `.venv/`, caches, `.git`, all `.md` docs/locator artifacts, `subagent-artifacts/`, `.task-reports/`, and IDE/tooling dirs.

## Key Data & Control Flow (end-to-end generation)
1. User configures a generation in the renderer (e.g. `GenSpace.tsx` → `PromptBar`; `ICLoraPanel.tsx` for IC-LoRA).
2. A hook (e.g. `use-ic-lora.ts`) builds a typed request and POSTs via `backendFetch` → app backend.
3. `backend/_routes/<domain>.py` parses input, calls the matching `AppHandler` method.
4. `AppHandler` (under `RLock`) validates + transitions `AppState`, then dispatches to a domain handler (`ic_lora_handler`, `video_generation_handler`, …).
5. Handler selects a pipeline service (`ServiceBundle`), runs heavy GPU work off-lock, then re-locks to write terminal state.
6. Pipeline decodes frames and encodes output via `services/ltx_pipeline_common.py::encode_video_output()` → `ltx_pipelines.utils.media_io.encode_video` (**H.264/yuv420p hardcoded**, `.mp4` path). `retake_pipeline` bypasses this with a direct call.
7. Output path returned to renderer; Electron serves media via a custom protocol; optional ffmpeg timeline export via `electron/export/`.

## Integration Points Worth Knowing
- **Encode chokepoint**: `backend/services/ltx_pipeline_common.py::encode_video_output()` is the single shared encode path for all video pipelines except retake. Any primary MOV-ProRes/EXR work must thread codec/profile/pixel-format/dims through here AND fix `retake_pipeline`'s direct call.
- **Hardcoded `.mp4`**: `video_generation_handler.py`, `ic_lora_handler.py`, `retake_handler.py` construct output paths as `.mp4`.
- **Proxy precedent**: `electron/ipc/file-handlers.ts::transcodeVideoInPlace` and `electron/export/export-handler.ts` ProRes blocks inform a keep-primary + proxy-sidecar design (frontend `<video>` cannot play EXR; ProRes playback is OS-dependent).
- **HDR (implemented — official-upstream-backed)**: `LTXHdrIcLoraPipeline` subclasses upstream `HDRICLoraPipeline` and overrides only `_create_conditionings` (in-memory all-frames + duplicate-final-frame pad to `8n+1`). HDR is a supported V2V workflow: `ic_lora_handler` ignores prompt/audio/`conditioning_type`/`images`, requires a source video, never rejects by frame count, gates to the **official distilled safetensors** base only (rejects dev/full/GGUF/split/Kijai/QuantStack/non-safetensors), forwards `scene_embeddings_path`, and returns `video_path` (EXR dir) + `proxy_path` (SDR MP4). The frontend allows an empty HDR prompt and keeps EXR `videoPath` as the primary asset with `proxyPath` as the preview sidecar. **Remaining-gated**: real-model numeric parity / non-flat EXR / no-clamp linear-HDR validation requires a separate exact harness (Phase 6b), not yet defined.
- **State machines**: `backend/state/app_state_types.py` — extend discriminated unions when adding new generation/output kinds.

Current planning lives under `docs/plans/current/`; root `plan.md` is only a pointer.
