# shared/
## Responsibility

The **cross-process contract layer**. Holds the single source of truth for the IPC schema that both the Electron main process (via `electron/ipc/typed-handle.ts`) and the renderer (via `electron/preload.ts` + `frontend/`) import. Also holds cross-process feature flags that must compile under both the main-process `tsconfig` and the renderer's bundler without pulling in Electron APIs.

Two files only:
- `electron-api-schema.ts` — Zod-defined `electronAPISchemas` map + derived `ElectronAPI` type consumed by `window.electronAPI`.
- `feature-flags.ts` — compile-time boolean `HF_GATING_ENABLED`, imported by `python-backend.ts` (env passthrough to backend) and `preload.ts` (renderer-visible mirror).

## Design Patterns

- **Single schema, dual inference.** `electronAPISchemas` is a plain object literal keyed by channel name; each entry is `{ input: ZodObject, output: ZodType }`. The preload bridge uses only `Object.keys(electronAPISchemas)` to enumerate channels; `typed-handle.ts` uses `electronAPISchemas[K]['input' | 'output']` for handler typing; the frontend imports `ElectronAPI` (via `z.infer`) so `window.electronAPI.<channel>` is fully typed on the renderer side too. **Adding/removing a channel requires editing only this file + one `handle(...)` site.**
- **Discriminated-union result envelope.** `ipcResult<T>(valueShape)` returns `z.discriminatedUnion('success', [{ success: true, ...valueShape }, { success: false, error: string }])`. `emptyResult = ipcResult({})` is reused for `exportNative`/`exportCancel`. All fallible operations that want to return structured errors (rather than rejecting the promise) flow through this helper.
- **Renderer-invokable helper typing via mapped type.** `InvokeAPI` maps each schema key to either `() => Promise<output>` (when input extends `Record<string, never>`) or `(input) => Promise<output>`. `ElectronAPI` extends `InvokeAPI` with the seven non-invoke members (`onPythonSetupProgress`, `removePythonSetupProgress`, `onBackendHealthStatus`, `onAssetImportProgress`, `getPathForFile`, `platform`, `hfGatingEnabled`) that the preload adds manually.
- **Flag isolation.** `feature-flags.ts` is explicitly documented as Electron-API-free so it is safe to import from any process config (main, preload, renderer). When `HF_GATING_ENABLED` is `false`, HF downloads proceed anonymously and the HF auth UI is hidden; when `true`, OAuth + per-repo access checks are enforced before downloads.
- **Reused domain object schemas.** `exportClip`, `exportSubtitle`, `logsResponse`, `backendHealthStatus`, `backendAdminRequest` are extracted as named Zod objects at module scope so they can be referenced from multiple channel definitions and mirrored as TypeScript interfaces in `frontend/types/`.

## Data & Control Flow

This folder defines types only — no runtime control flow. The data flow it enables:

1. **Renderer call:** `await window.electronAPI.exportNative({ clips, outputPath, codec, ... })` — typed against `ElectronAPI['exportNative']` whose input/output are `z.infer`'d from `electronAPISchemas.exportNative`.
2. **Preload bridge:** `preload.ts` binds `api['exportNative'] = (input) => ipcRenderer.invoke('exportNative', input)` for every key.
3. **Main dispatcher:** `ipcMain.handle('exportNative', (_event, input) => handler(input ?? {}))` (from `typed-handle.ts`) routes to the `handle('exportNative', ...)` registered in `electron/export/export-handler.ts`.
4. **Validation boundary:** schemas are currently used for **type inference only** — inputs are not re-parsed with `.parse()` at the IPC boundary in the current code (the `handle()` wrapper does not invoke the input schema). The contract is enforced at compile time on both sides instead.

### `electronAPISchemas` channel inventory (by domain)
- **Backend/app/system:** `getBackend` (`{url, token}`), `backendAdminRequest` (proxied with allowlist in `app-handlers.ts`), `getModelsPath`, `readLocalFile`, `checkGpu`, `getAppInfo`.
- **First-run/license:** `checkFirstRun` (`{needsSetup, needsLicense}`), `acceptLicense`, `completeSetup`, `fetchLicenseText`, `getNoticesText`.
- **External links:** `openLtxApiKeyPage`, `openLtxBillingPage`, `openFalApiKeyPage`, `openHuggingFaceRepo({repoId})`, `openHuggingFaceAuth({clientId, redirectUri, scope, state, codeChallenge, codeChallengeMethod})`, `openParentFolderOfFile({filePath})`, `showItemInFolder({filePath})`.
- **Logs:** `getLogs`, `getLogPath` (`{logPath, logDir}`), `openLogFolder`.
- **Paths:** `getResourcePath`, `getDownloadsPath`.
- **Project assets:** `addVisualAssetToProject` (optional `proxyPath` + `jobId`; returns optional `proxyPath` — when `proxyPath` is supplied the primary is preserved and the proxy copied), `addGenericAssetToProject`, `makeThumbnailsForProjectAsset`, `makeDimensionsForProjectAsset`, `getProjectAssetsPath`, `openProjectAssetsPathChangeDialog` (all `ipcResult`-shaped), `transcodeVideoForPreview` (`{srcPath, jobId?}` → `{path}` preview MP4; source is never mutated, progress streamed on `asset:importProgress`).
- **File dialogs/save:** `showSaveDialog`, `saveFile`, `saveBinaryFile` (`ArrayBuffer`), `showOpenDirectoryDialog`, `searchDirectoryForFiles` (`Record<filename, path>`), `checkFilesExist`, `showOpenFileDialog`.
- **Export:** `exportNative` (clips + codec + letterbox + subtitles), `exportCancel` (`{sessionId}` — accepted but unused by current impl).
- **Python lifecycle:** `checkPythonReady`, `startPythonSetup`, `startPythonBackend`, `getBackendHealthStatus` (`BackendHealthStatus | null`).
- **Video processing:** `extractVideoFrame` (`{videoPath, seekTime, width?, quality?}` → `{path}`).
- **Logging:** `writeLog` (`{level, message}`).
- **Models:** `openModelsDirChangeDialog`.
- **Analytics:** `getAnalyticsState`, `setAnalyticsEnabled`, `sendAnalyticsEvent`.

### Non-invoke `ElectronAPI` members (added manually in preload)
- `onPythonSetupProgress(cb: (data: unknown) => void)` — subscribes to `'python-setup-progress'` channel.
- `removePythonSetupProgress()` — removes all listeners on that channel.
- `onBackendHealthStatus(cb: (data: BackendHealthStatus) => void) => () => void` — subscribes to `'backend-health-status'`; returns an unsubscribe.
- `onAssetImportProgress(cb: (data: AssetImportProgressEvent) => void) => () => void` — subscribes to `'asset:importProgress'` (asset import / preview-transcode progress `{jobId, percent, label, done?}`); returns an unsubscribe.
- `getPathForFile(file: File) => string` — wraps `webUtils.getPathForFile`.
- `platform: string` — `process.platform` mirror.
- `hfGatingEnabled: boolean` — `HF_GATING_ENABLED` mirror.

### Exported types
- `ElectronAPI` (mapped type), `IpcResult<T>`, `BackendHealthStatus` (`z.enum(['alive','restarting','dead'])` + optional nullable `exitCode`). These are the types the frontend imports to type `window.electronAPI`.

## Integration Points

- **`electron/preload.ts`:** iterates `Object.keys(electronAPISchemas)` to auto-build the invoke surface; imports `BackendHealthStatus` for the typed `onBackendHealthStatus` subscription; imports `HF_GATING_ENABLED` from `feature-flags.ts` to expose `window.electronAPI.hfGatingEnabled`.
- **`electron/ipc/typed-handle.ts`:** imports `electronAPISchemas` to type the generic `handle<K>()` wrapper; every `register*Handlers` in `electron/ipc/*` and `electron/export/export-handler.ts` goes through it.
- **`electron/python-backend.ts`:** imports `HF_GATING_ENABLED` and forwards it to the backend via the `LTX_HF_GATING_ENABLED` env var (so the Python process and the renderer agree on gating state).
- **`electron/export/export-handler.ts`:** its `ExportClip`/`ExportSubtitle` source interfaces must stay structurally identical to the `exportClip`/`exportSubtitle` Zod objects here — they are the wire format the renderer sends.
- **Frontend (renderer):** `ElectronAPI` and the individual input/output types are imported to type all `window.electronAPI.*` calls; per AGENTS.md, the renderer uses `backendFetch` (not raw `fetch`) for backend HTTP, and Electron IPC for everything that crosses the main/renderer boundary.
- **Primary-output (forward-looking):** when MOV/ProRes/EXR primary generation lands, new schema entries will be needed here — e.g. an output-location channel, a proxy-path field added to relevant `ipcResult` outputs, and likely a codec/profile enum shared with the export codec block in `electron/export/export-handler.ts`. Because schemas drive typing in three places (preload auto-bind, typed-handle, renderer `ElectronAPI`), this is the single edit point to extend the IPC surface for that feature.
