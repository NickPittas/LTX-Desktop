# electron/ipc/
## Responsibility

The **`ipcMain.handle` dispatcher layer**. Wires every channel exposed under `window.electronAPI` (defined by `shared/electron-api-schema.ts`) to a concrete implementation. Covers: app/backend info, GPU probe, first-run setup, license, model-dir management, file dialogs & saves, project-asset import (keep-primary + proxy copy when a `proxyPath` is supplied; legacy in-place video transcode otherwise; `transcodeVideoForPreview` for non-mutating preview MP4; thumbnail/dimension generation; progress streamed on `asset:importProgress`), log read/write, and video-frame extraction. Each `register*Handlers()` function is called exactly once from `main.ts` at startup.

## Design Patterns

- **Schema-typed `handle()` wrapper** (`typed-handle.ts`). A single generic `handle<K>(key, handler)` re-exports `ipcMain.handle(key, (_event, input) => handler(input ?? {}))` while inferring the handler's input and output types from `electronAPISchemas[K]`. New channels cannot drift from the shared contract without a TS error.
- **One registrar per domain.** `registerAppHandlers` (backend/app/system), `registerFileHandlers` (dialogs/IO/asset import), `registerLogHandlers`, `registerVideoProcessingHandlers`, `registerExportHandlers` (the last lives in `electron/export/`, see that folder's codemap). All called from `main.ts` before `app.whenReady()`.
- **Path allowlist before any filesystem write.** Every handler that writes or resolves a path calls `validatePath(filePath, getAllowedRoots())` first; `readLocalFile`, `saveFile`, `saveBinaryFile`, `openParentFolderOfFile`, and `exportNative` all enforce this. Native save/open dialog results are pushed into the approved set via `approvePath()` so subsequent writes to the user-chosen path succeed.
- **Result envelope for fallible operations.** Asset/save/dialog handlers return `ipcResult(...)` shapes — `{ success: true, ...payload }` or `{ success: false, error }` — matching the discriminated-union helper in `shared/electron-api-schema.ts`. Non-recoverable errors (e.g. `readLocalFile` failures) are rethrown so the renderer's `ipcRenderer.invoke` rejects.
- **Pillow-via-Python for image work** (`image-utils.ts`). Thumbnails and image dimensions are computed by `spawnSync(getPythonPath(), ['-c', <inline PIL script>, ...])` — there is no native image dependency. EXIF orientation is applied via `ImageOps.exif_transpose`. Thumbnail generation uses `LANCZOS` resampling capped at `DEFAULT_THUMBNAIL_MAX_DIMENSION = 400`.
- **Lazy `import('electron')` of `shell`** in several file handlers — keeps the `dialog` import narrow at the top and defers shell loading to the handler body.

## Data & Control Flow

### `app-handlers.ts` (`registerAppHandlers`)
- **Backend surface:** `getBackend` → `{ url: getBackendUrl() ?? '', token: getAuthToken() ?? '' }`. `backendAdminRequest({ path, method, headers, body })` proxies to the Python backend with an **allowlist**: target origin must equal `getBackendUrl()` origin AND the pathname must be `/api/model-profiles`, `/api/model-profiles/*`, `/api/models/adapters/status`, or `/api/models/adapters/recommendation`. Injects `Authorization: Bearer <authToken>` and `X-Admin-Token: <adminToken>`. Returns `{ status, statusText, ok, body }` (503 if backend not ready, 403 if path disallowed).
- **App/system:** `getModelsPath` (ensures `userData/models` exists), `checkGpu` (delegates to `gpu.ts`), `getAppInfo` (`{ version, isPackaged, modelsPath, userDataPath }`), `getDownloadsPath`, `getResourcePath` (`process.resourcesPath` if packaged, else `null`).
- **First-run/license:** `checkFirstRun` reads `userData/app_state.json`; if absent, falls back to inspecting `model_profiles.json` for an `isActive` profile (returns `{ needsSetup: false, needsLicense: true }`) — otherwise `{ needsSetup: true, needsLicense: true }`. `acceptLicense` / `completeSetup` write `{ licenseAccepted, setupComplete, licenseAcceptedDate, setupDate }` via `writeSettingsFile`. `fetchLicenseText` GETs the LTX-2.3 LICENSE from HuggingFace. `getNoticesText` reads `NOTICES.md` from `app.getAppPath()`.
- **Python lifecycle:** `checkPythonReady` → `isPythonReady()`, `startPythonSetup` → `downloadPythonEmbed(progress => webContents.send('python-setup-progress', progress))`, `startPythonBackend` → `startPythonBackend()`, `getBackendHealthStatus` → cached snapshot from `python-backend.ts`.
- **Analytics:** `getAnalyticsState`, `setAnalyticsEnabled`, `sendAnalyticsEvent` pass straight through to `analytics.ts`.
- **`openModelsDirChangeDialog`:** shows a directory picker, then POSTs `{ modelsDir: newDir }` to `${url}/api/settings` with Bearer + `X-Admin-Token` headers. Returns `{ success, path }` or `{ success: false, error }`.

### `file-handlers.ts` (`registerFileHandlers`)
- **External links:** `openLtxApiKeyPage`, `openLtxBillingPage`, `openFalApiKeyPage`, `openHuggingFaceRepo({ repoId })`, `openHuggingFaceAuth({ clientId, redirectUri, scope, state, codeChallenge, codeChallengeMethod })` — the last builds a full `https://huggingface.co/oauth/authorize?...` URL with PKCE params and opens it via `shell.openExternal`.
- **Filesystem reveal:** `openParentFolderOfFile` (validates path, `shell.openPath(path.dirname(...))`), `showItemInFolder`.
- **Read:** `readLocalFile({ filePath })` → `readLocalFileAsBase64` returns `{ data: base64, mimeType }` (MIME table covers png/jpg/jpeg/webp/gif/mp3/wav/ogg/aac/flac/m4a/mp4/webm/mkv/mov). `searchDirectoryForFiles({ directory, filenames })` walks the tree (case-insensitive, skips dot-dirs, depth-capped at 10) returning `Record<lowercasedName, fullPath>`.
- **Dialogs:** `showSaveDialog`, `showOpenDirectoryDialog`, `showOpenFileDialog` (honors `properties: ['multiSelections']`); every accepted result is pushed through `approvePath()`. `checkFilesExist({ filePaths })` returns `Record<path, boolean>`.
- **Saves:** `saveFile({ filePath, data, encoding })` (base64 or utf-8), `saveBinaryFile({ filePath, data: ArrayBuffer })`. Both `validatePath` first and return `ipcResult({ path })`.
- **Project-asset import (keep-primary + proxy / in-place transcode):** progress for every path is streamed on `asset:importProgress` (`{jobId, percent, label, done?}`), weighted across phases via an import-job tracker.
  - `addVisualAssetToProject({ srcPath, projectId, type: 'video' | 'image', proxyPath?, jobId? })`:
    1. `resolveLocalSourcePath` — reject empty/relative/missing/non-file.
    2. `copyVisualAssetWithProgress` — primary copied to `getProjectAssetsPath()/projectId/<name>` (collision-renamed via `getUniqueDestinationPath`); an EXR dir is copied recursively.
    3. **If `proxyPath` is supplied (ProRes/EXR primary):** the primary is preserved verbatim (never transcoded/destroyed) and the proxy MP4 is copied alongside → `projectProxyPath`; thumbnails/dimensions are generated from the proxy copy. **Else if `type==='video'` (legacy/no-proxy path):** `transcodeVideoInPlace(destPath)` — ffmpeg `-c:v libx264 -pix_fmt yuv420p -preset veryfast -crf 18 -c:a aac -b:a 192k -movflags +faststart` into `*.tmp_transcode.mp4`, then atomically replaces the project copy (in-place transcode — the project copy is **not** preserved).
    4. Thumbnails: video → `extractVideoFrameToFile({ seekTime: 0 })` as the big thumbnail; image → big thumbnail = the asset itself. Small thumbnail always via `createDownsampledThumbnail`.
    5. Dimensions via `getVideoDimensions` (video) or `getImageDimensions` (image).
    Returns `ipcResult({ path, proxyPath?, bigThumbnailPath, smallThumbnailPath, width, height })`.
  - `transcodeVideoForPreview({ srcPath, jobId? })` — `transcodeVideoForPreviewImpl` transcodes the source to a browser-playable H.264/AAC/yuv420p MP4 under `os.tmpdir()/ltx-preview/` and returns `ipcResult({ path })`. The source is **never** mutated (no delete/rename/overwrite); progress is streamed on `asset:importProgress`. Used to preview a source video without importing/mutating it.
  - `addGenericAssetToProject({ srcPath, projectId })` — copy only, returns `ipcResult({ path })`.
  - `makeThumbnailsForProjectAsset` / `makeDimensionsForProjectAsset` regenerate the same outputs for an already-imported asset.
  - `getProjectAssetsPath`, `openProjectAssetsPathChangeDialog` (dialog → `setProjectAssetsPath` + `approvePath`).

### `image-utils.ts`
Pure helpers (no IPC): `getThumbnailPaths(assetPath)` → `{ bigThumbnailPath: <name>_big_thumbnail.png, smallThumbnailPath: <name>_small_thumbnail.png }` (siblings of the asset). `createDownsampledThumbnail(source, output, maxDimension=400)` and `getImageDimensions(source)` both spawn the bundled Python with an inline PIL script.

### `log-handlers.ts` (`registerLogHandlers`)
- `writeLog({ level, message })` — validates `level ∈ {INFO, WARNING, ERROR, DEBUG}`, forwards to `writeLog(level, 'Renderer', message)` so renderer logs land in the same session file as Electron/Backend.
- `getLogs()` — returns `{ logPath, lines: <last 200 trimmed lines> }` from `getCurrentLogFilename()`.
- `getLogPath()` → `{ logPath, logDir }`.
- `openLogFolder()` → `shell.openPath(getLogDir())` if it exists.

### `video-processing-handlers.ts` (`registerVideoProcessingHandlers`)
- `extractVideoFrame({ videoPath, seekTime, width, quality })` → `{ path }`. Delegates to `extractVideoFrameToFile` (default `quality: 2`, `timeoutMs: 10000`) from `electron/export/ffmpeg-utils.ts`.

## Integration Points

- **Shared contract:** `typed-handle.ts` imports `electronAPISchemas` from `shared/electron-api-schema.ts` — every handler signature is checked against that schema at compile time. New channels require a schema entry + a `handle(...)` here.
- **`electron/` core:** handlers depend on `gpu.ts` (`checkGPU`), `python-backend.ts` (`getBackendUrl/getAuthToken/getAdminToken/startPythonBackend/getBackendHealthStatus`), `python-setup.ts` (`isPythonReady/downloadPythonEmbed`), `analytics.ts`, `app-state.ts`, `config.ts` (`getAllowedRoots`), `path-validation.ts`, `logging-management.ts`, `logger.ts`, `window.ts` (`getMainWindow` for dialogs + `webContents.send`).
- **`electron/export/`:** `file-handlers.ts` imports `extractVideoFrameToFile`, `findFfmpegPath`, `getVideoDimensions`, `runFfmpeg` from `export/ffmpeg-utils.ts`; `video-processing-handlers.ts` imports `extractVideoFrameToFile`. The same ffmpeg resolver and `runFfmpeg` runner back both the import-transcode path and the export pipeline.
- **Renderer (indirectly):** every channel is consumed through `window.electronAPI.<channel>` (see `electron/preload.ts` and `shared/codemap.md`).
- **ProRes/EXR primary-output (implemented):** the keep-primary + proxy-sidecar design is live — when the renderer passes a `proxyPath`, `addVisualAssetToProject` preserves the primary verbatim and copies the proxy alongside (the `proxyPath` field is part of the `ipcResult` schema), and `transcodeVideoForPreview` mints a non-mutating preview MP4. The legacy `transcodeVideoInPlace` step remains the precedent for the no-proxy path ("make a browser-playable proxy from a non-playable source" via `runFfmpeg` + atomic tmp-rename). EXR frames cannot be played by `<video>` at all and ProRes playback is OS-dependent, so a proxy is **required** (not optional) for renderer preview of those outputs; preview/transcode `runFfmpeg` calls run with `isolated: true` so the global export-cancel cannot kill an in-flight import/preview transcode.
