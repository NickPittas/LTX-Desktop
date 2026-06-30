# frontend/hooks
## Responsibility
Reusable renderer data hooks. Two families: (1) backend-data hooks wrapping `ApiClient` with a `data`/`isLoading`/`errorMessage`/`refresh` state machine; (2) generation-flow hooks (`use-generation`, `use-ic-lora`, `use-retake`) owning an abortable request + result/error state. Plus `use-backend` (process-liveness subscription), the HF gating pair, `use-video-generation-model-specs` (one-shot fetch), and the one-shot `useProjectReferencesMigration`.

## Design Patterns
- **Backend-data hook template** (`use-model-profiles`, `use-official-adapters`): `useState<{data|null, isLoading, errorMessage}>` + `refresh()` calling the `ApiClient` method, `enabled`-gated initial `useEffect`, mutation helpers that `await ApiClient.*` then `await refresh()`, and a local `requireOk<T>` that throws on `{ok:false}`.
- **Generation-flow hook template** (`use-generation`, `use-ic-lora`, `use-retake`): `abortControllerRef` for cancel, a single `setState`-reduced state object, and an `onCompleteRef` (`use-ic-lora`, `use-retake`) **fired before `setState`** so `ProjectContext` mutations still run if the GenSpace source unmounts (documented as "Bug A fix").
- **HF gating short-circuit** (`use-hf-auth`, `use-hf-model-access`): read `window.electronAPI.hfGatingEnabled` once; when `false`, return `authenticated`/`allAuthorized: true` and `NOOP` callbacks (stable identity) instead of subscribing.
- **Poll-while-waiting** (`use-hf-auth` 2 s while `hfAuthPolling`; `use-hf-model-access` 5 s while any model `not_authorized`).
- **Idempotent one-shot migration** (`useProjectReferencesMigration`): `inFlightRef` dedupes, `yieldToUi()` (`setTimeout(resolve, 0)`) between items, status union `{needed}|{inProgress,ratio}|{completed}`.
- **Endpoint-result consumption:** all hooks branch on `result.ok`; failures go to `logger.error` + `errorMessage`/`error` (generation hooks wrap via `createLocalGenerationError`).
- **Typed request bodies via `ApiRequestBodyOf`/`ApiSuccessOf`** (`use-ic-lora`, `use-hf-model-access`, `use-generation`).

## Data & Control Flow
- **`use-backend.ts`** → subscribes `window.electronAPI.onBackendHealthStatus` + `getBackendHealthStatus()` snapshot; `toBackendHealthStatus` validates; on `'alive'` calls `resetBackendCredentials()` and clears `isLoading`. Returns `{ processStatus, connected: processStatus === 'alive', isLoading }`.
- **`use-generation.ts`**
  - `generate(prompt, imagePath, settings, audioPath?)`: builds body `{ prompt, model, duration, resolution: settings.videoResolution, fps, audio, cameraMotion, negativePrompt, aspectRatio }` plus conditional `imagePath`/`audioPath`; starts 500 ms `pollProgress` (`ApiClient.getGenerationProgress`) with time-based interpolation during `inference` (`estimatedInferenceTime` 120 s for `'pro'`, else 45 s; maps 15–95 %); `await ApiClient.generateVideo(body, { signal })` (synchronous, resolves when backend done); `complete` → `videoPath`, `cancelled` → status, `AbortError` → "Cancelled".
  - `generateImage(prompt, settings)`: if `forceApiGenerations`, checks `hasFalApiKey` via `ApiClient.getSettings` (fallback to `appSettings`); if missing dispatches `'open-api-gateway'` (`requiredKeys: ['fal']`) and returns; computes dims via `IMAGE_SHORT_SIDE_BY_RESOLUTION` × `IMAGE_ASPECT_RATIO_VALUE` (`getImageDimensions`); polls progress; `await ApiClient.generateImage({ prompt, width, height, numSteps: settings.imageSteps||4, numImages: settings.variations||1 })`.
  - `cancel()`: `abortControllerRef.current?.abort()` + `void ApiClient.cancelGeneration()`.
- **`use-ic-lora.ts`** — `submitIcLora(params, onComplete?)`: builds snake_case body (`conditioning_strength`, `prompt` passed through as-is — an empty string is fine for HDR, no fake-prompt substitution, `frame_rate` default 24); conditional fields added only when present (`video_path` if `videoPath`, `conditioning_type` if not null, `adapter_id`, `mask_path`, `images`, `model_selection` for HDR); defaults `mask_grow_px 30`, `laplacian_blend_grow 12`, `final_mask_blur_px 6`; optional `lora_strength`/`width`/`height`/`num_frames`. Never sends `scene_embeddings_path`/`hdr_lora_path` (backend-resolved internal variables). `ApiClient.generateIcLora`; on `complete` calls `onCompleteRef` (with `{ videoPath, proxyPath }` — EXR dir primary + SDR proxy sidecar) **before** `setState`.
- **`use-retake.ts`** — `submitRetake(params, onComplete?)`: `ApiClient.retake({ video_path, start_time, duration, prompt, mode })` where `RetakeMode = 'replace_audio_and_video'|'replace_video'|'replace_audio'`; same onComplete-before-setState pattern.
- **`use-model-profiles.ts`** — `refresh` = `ApiClient.getModelProfiles()`; `createProfile`/`patchProfile`/`deleteProfile`/`activateProfile` each `requireOk(await ApiClient.*)` then `await refresh()`; `validateProfile` does not refresh.
- **`use-official-adapters.ts`** — `refresh` = `ApiClient.getAdapterStatus(pipeline ? {pipeline} : undefined)`; `getRecommendation(nextPipeline)` = `requireOk(ApiClient.getAdapterRecommendation({pipeline}))`.
- **`use-video-generation-model-specs.ts`** — one-shot `ApiClient.getGenerateVideoModelSpecs(undefined, { signal })` in a `useEffect` with `AbortController` cleanup; returns `{modelSpecs, isLoading, errorMessage}`.
- **`use-hf-auth.ts`** — one-time `ApiClient.getHuggingFaceAuthStatus()` when `enabled`; 2 s poll while `hfAuthPolling`; `startHuggingFaceLogin` = `ApiClient.startHuggingFaceLogin()` then `window.electronAPI.openHuggingFaceAuth({client_id, redirect_uri, scope, state, code_challenge, code_challenge_method})`; `handleHuggingFaceLogout` = `ApiClient.huggingFaceLogout()`.
- **`use-hf-model-access.ts`** — `doCheck` = `ApiClient.checkModelAccess({cp_ids: [...modelTypes]})`; initial check + 5 s poll while any status `!== 'authorized'`; `allAuthorized` derived.
- **`useProjectReferencesMigration.ts`** — pure helpers operate on `localStorage['ltx-projects']` (legacy) vs per-id `ltx-project-<id>` (current); `migrateProjects` writes each `writeRawProject`, `writeProjectIds`, `deleteLegacyProjectsEntry`, then `reloadProjectIds()`.

## Integration Points
- **`lib/api-client`:** `ApiClient`, `ApiRequestBodyOf`, `ApiSuccessOf`, `EndpointResult` (every hook).
- **`lib/generation-errors`:** `createLocalGenerationError`, `GenerationError` (`use-generation`).
- **`lib/logger`:** error/warn reporting (`use-backend`, `use-generation`, `use-hf-auth`, `use-hf-model-access`, `use-ic-lora`, `use-retake`, `useProjectReferencesMigration`).
- **`lib/project-storage`:** `PROJECT_IDS_STORAGE_KEY`, `PROJECT_STORAGE_KEY_PREFIX`, `getProjectStorageKey`, `readProject`, `readProjectIds`, `writeProjectIds` (`useProjectReferencesMigration`).
- **`lib/backend`:** `resetBackendCredentials` (`use-backend`).
- **`lib/video-generation-model-specs`:** `VideoGenerationModelSpecsResponse` (`use-video-generation-model-specs`).
- **`types/project-model`** (`Project`, `projectSchema`, `projectReferenceSchema`) and **`types/model-profile`** (`ModelProfilesResponse`, `ModelProfilePayload`, `ModelProfilePatchPayload`, `ModelProfileValidationResponse`, `ModelProfileActivateResponse`, `AdapterPipeline`, `AdapterStatusResponse`, `AdapterRecommendationResponse`).
- **`contexts/AppSettingsContext`:** `useAppSettings` (`use-generation` reads `settings`, `forceApiGenerations`, `refreshSettings`).
- **`contexts/ProjectContext`:** `useProjects().reloadProjectIds` (`useProjectReferencesMigration`); `use-ic-lora`/`use-retake` results flow back to `ProjectContext` via caller-supplied `onComplete`.
- **`components/SettingsPanel`:** `use-generation` imports the `GenerationSettings` type from `components/SettingsPanel` (cross-folder type coupling).
- **`window.electronAPI`:** `onBackendHealthStatus`, `getBackendHealthStatus` (`use-backend`); `hfGatingEnabled`, `openHuggingFaceAuth` (`use-hf-auth`); `hfGatingEnabled` (`use-hf-model-access`).
