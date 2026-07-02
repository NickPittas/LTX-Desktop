# electron/
## Responsibility

The Electron **main process** layer. Owns app lifecycle, window creation, the privileged `ltx-file://` media protocol, CSP enforcement, the typed IPC bridge surface exposed to the renderer, the session logger, the auto-updater, anonymous analytics, and — most importantly — spawning and supervising the Python FastAPI backend (`backend/ltx2_server.py`). It also owns the bundled-Python (`python-embed`) download / staging / promotion pipeline used for first-run setup and update pre-fetches.

This folder is compiled to `dist-electron/`. `preload.ts` is **CommonJS** (`require('electron')`) and is the sole bridge into the renderer.

## Design Patterns

- **Schema-driven IPC, single registration site.** `preload.ts` iterates `electronAPISchemas` (from `shared/electron-api-schema.ts`) and binds every key to `ipcRenderer.invoke(key, input)`. Main-process handlers register through `ipc/typed-handle.ts`, which infers input/output types from the same Zod schemas. Adding an IPC method requires only adding a schema entry + a `handle(...)` implementation; preload never needs editing.
- **Side-effectful import for path bootstrapping.** `app-paths.ts` is the first import in `main.ts` (`import './app-paths'`) so `app.setPath('userData', resolveUserDataPath())` runs before anything else. The userData path is platform-specific (`LOCALAPPDATA/LTXDesktop` on Windows, `~/Library/Application Support/LTXDesktop` on macOS, `XDG_DATA_HOME/LTXDesktop` on Linux).
- **Single-instance lock with graceful second-instance handling.** `app.requestSingleInstanceLock()`; on `second-instance`, focus the existing window or recreate it.
- **Privileged custom protocol declared before ready.** `protocol.registerSchemesAsPrivileged([{ scheme: 'ltx-file', privileges: { bypassCSP: false, stream: true, supportFetchAPI: true } }])` runs at module load (before `app.whenReady()`); the handler is registered inside `whenReady()`.
- **Self-healing managed child process with ownership model.** `python-backend.ts` tracks `backendOwnership: 'managed' | 'adopted' | null` — "managed" = this process spawned and supervises it; "adopted" = latched onto an already-running backend (port conflict path) and tries to reclaim ownership asynchronously via `startOwnershipTakeover()`.
- **No-restart crash loop protection.** `CRASH_DEBOUNCE_MS = 10_000` prevents restart thrash; after the debounce window a crash flips health to `dead` instead of `restarting`.
- **Typed event channels.** Main → renderer pushes use fixed channel names: `'backend-health-status'`, `'python-setup-progress'`, `'python-update-progress'`.
- **Path allowlist enforcement.** `path-validation.ts` `validatePath()` resolves against `getAllowedRoots()` (cwd, userData, downloads, tmpdir, resourcesPath in prod, project-assets path) plus a runtime-approved set populated by `approvePath()` after native save/open dialogs.
- **Log file rotation in main process.** `logging-management.ts` creates one `session_<ISO>_<githash>.log` per launch, retains the newest 30 files (session_ and backend_ prefixed), and pushes the path into the logger via `setLogFilePath()`. Backend stderr/stdout is tee'd into the same file with source `'Backend'`.
- **Fire-and-forget analytics with retry.** `sendAnalyticsEvent` is a no-op in dev and when opted out; otherwise retries 429/5xx up to 3 times with `[1s, 3s, 10s]` backoff and a 5s per-request timeout.

## Data & Control Flow

### Boot sequence (`main.ts`)
1. `import './app-paths'` → rewrites `userData` to the platform LTXDesktop path.
2. `gotLock = app.requestSingleInstanceLock()`; quit if not the singleton.
3. `initSessionLog()` → mkdir log dir, git-hash filename, rotate, `setLogFilePath`.
4. `logAppVersion()`.
5. `registerAppHandlers()` / `registerFileHandlers()` / `registerLogHandlers()` / `registerExportHandlers()` / `registerVideoProcessingHandlers()` — wire all `ipcMain.handle` routes.
6. `app.whenReady().then(...)`:
   - `protocol.handle('ltx-file', ...)` — stream files with HTTP Range support (206 partial content for `bytes=start-end`, 200 full otherwise). Decodes the path from `request.url.slice('ltx-file://'.length)`. Returns 404 on any error. MIME lookup by extension.
   - `setupCSP()` — attach CSP via `onHeadersReceived`. Dev CSP permits `'unsafe-inline'` scripts/styles + localhost connects; prod CSP is locked down. Both allow `media-src`/`img-src` from `blob: file: ltx-file:`.
   - `createWindow()`.
   - `initAutoUpdater()` — check 5s after boot, then every 4h.
   - `void sendAnalyticsEvent('ltxdesktop_app_launched')`.

### Lifecycle hooks
- `second-instance` → restore/show/focus existing window, or recreate.
- `window-all-closed` → on non-macOS: `stopPythonBackend()` + `app.quit()`.
- `activate` → recreate window on macOS if null.
- `before-quit` → `stopExportProcess()` + `stopPythonBackend()`.

### Window (`window.ts`)
- `createWindow()` builds a `BrowserWindow` (1400×900, min 1200×700), preload = `dist-electron/preload.js`, `contextIsolation: true`, `nodeIntegration: false`, `webSecurity: !isDev`. Dev loads Vite (`http://localhost:5173`), prod loads packaged `dist/index.html`. `ready-to-show` defers visibility; `closed` nulls the module-level `mainWindow`. `getMainWindow()` is the accessor used everywhere else.

### preload.ts exposed surface (`window.electronAPI`)
- **All `electronAPISchemas` keys** as `(input?) => Promise<output>` (auto-bound at load).
- **Event subscriptions:** `onPythonSetupProgress(cb)`, `removePythonSetupProgress()`, `onBackendHealthStatus(cb)` (returns an unsubscribe function).
- **Non-IPC helpers:** `getPathForFile(file)` (wraps `webUtils.getPathForFile` for `<input type=file>` File objects), `platform` (`process.platform`), `hfGatingEnabled` (from `shared/feature-flags.ts`).

### Backend supervision (`python-backend.ts`)
`startPythonBackend()` is invoked from the renderer via the `startPythonBackend` IPC after `checkPythonReady()` returns `ready: true`. Internals:
- `getPythonPath()` resolves in priority order: bundled `python-embed` (prod), backend `.venv` Python, dev fallback (`python3`/`python`).
- `spawn(pythonPath, args, { cwd: backendPath, env })` with args tailored per platform: Windows bundled uses `-c "import sys; sys.path.insert(0, backendPath); import runpy; runpy.run_path(mainPy, run_name='__main__')"` (works around embedded-Python `._pth` quirks); dev adds `-Xfrozen_modules=off`; otherwise `['-u', mainPy]`.
- Generates per-session `authToken` and `adminToken` (`crypto.randomBytes(32).toString('base64url')`), passed via env `LTX_AUTH_TOKEN` / `LTX_ADMIN_TOKEN`. Also sets `LTX_LOG_FILE`, `LTX_APP_DATA_DIR`, `LTX_DEV_MODE`, `LTX_HF_GATING_ENABLED`, `PYTORCH_ENABLE_MPS_FALLBACK=1`, and on macOS prod `PYTHONHOME=getPythonDir()`.
- `checkStarted(output)` parses stdout for `/Server running on (http:\/\/\S+)/` → captures `backendUrl` → `gateAliveOnProbe()` HTTP-probes `${url}/health` (Bearer token) for up to `STARTUP_PROBE_TIMEOUT_MS=30s` at 500ms cadence. On success: `backendOwnership='managed'`, publish `'alive'`, resolve, `startLivenessMonitor()`.
- **Liveness monitor** polls `/health` every `LIVENESS_POLL_INTERVAL_MS=10s`; after `LIVENESS_FAILURE_THRESHOLD=3` consecutive failures it `SIGTERM`s the process and lets the `exit` handler restart it.
- **Exit handler** classifies: startup failure (`!started`) → if port conflict + healthy existing backend on `LTX_PORT`, adopt it (`backendOwnership='adopted'`, `startOwnershipTakeover()`); otherwise publish `'dead'`. Crash while running + intentional-shutdown flag → just clear ownership. Crash while running + not intentional → publish `'restarting'` (unless within `CRASH_DEBOUNCE_MS`) and re-call `startPythonBackend()`.
- `stopPythonBackend()` sets `isIntentionalShutdown=true`, `SIGTERM`, escalates to `SIGKILL` after 5s.
- Health broadcast: `publishBackendHealthStatus()` caches `latestBackendHealthStatus` and `webContents.send('backend-health-status', status)`.
- 5-minute hard timeout kills the process if startup never settled.

### Python setup pipeline (`python-setup.ts`)
- `isPythonReady()` short-circuits to `ready: true` on macOS (bundled) and in dev. Otherwise compares `getBundledHashPath()` (`resources/python-deps-hash.txt` prod, `./python-deps-hash.txt` dev) against `getInstalledHashPath()` (`userData/python/deps-hash.txt`). If a staged `python-next/` exists with matching hash, it is **promoted** (atomic rename over `python/`).
- `downloadPythonEmbed(onProgress)` is called from the `startPythonSetup` IPC. Source resolution in `getArchiveBase()`: prod → `github.com/Lightricks/ltx-desktop/releases/download/v<version>`; dev → `process.env.LTX_PYTHON_URL` (local-path injection blocked in prod). On primary failure → CDN fallback `https://storage.googleapis.com/ltx-desktop-artifacts/<prefix>/<hash>/<prefix>.tar.gz`. Archive acquired via manifest-driven multi-part (`acquirePartsRemote`) or single-file (`downloadFileWithGlobalProgress`), concatenated, `tar -xzf`'d into `userData/python/`, and the bundled hash is copied in. Progress is reported back through `onProgress` (`PythonSetupProgress`).
- `preDownloadPythonForUpdate(newVersion, onProgress)` is invoked by `updater.ts` after `update-downloaded` on Windows/Linux: fetches the new version's hash, skips if unchanged, otherwise stages into `userData/python-next/` so the next launch promotes it instantly. Never blocks `autoInstallOnAppQuit`.
- GitHub auth (`getAuthHeaders()`) only attaches `GH_TOKEN`/`GITHUB_TOKEN` when `private: true` is set in `app-update.yml`/`dev-app-update.yml` (mirrors electron-updater).

### Updater (`updater.ts`)
`initAutoUpdater(channel='latest')` configures `autoUpdater` (beta/alpha enable `allowPrerelease`), registers `update-downloaded` → `preDownloadPythonForUpdate()` with progress streamed to `'python-update-progress'`, and schedules `checkForUpdatesAndNotify()` at +5s and every 4h.

## Integration Points

- **Renderer:** every IPC call lands in `window.electronAPI[<schemaKey>]`. See `shared/codemap.md` for the contract and `electron/ipc/codemap.md` for the dispatcher implementations.
- **Renderer event streams:** `onBackendHealthStatus`, `onPythonSetupProgress` (and the main-only `python-update-progress`).
- **Renderer media playback:** the `ltx-file://` protocol is how `<video>`/`<img>` elements load local files in prod; it bypasses Chromium `file://` restrictions while still respecting CSP.
- **Python backend:** spawned `ChildProcess`; main process is the only thing that knows the per-session `authToken`/`adminToken`. The `getBackend` IPC hands `{ url, token }` to the renderer so it can call `backendFetch` (`frontend/lib/backend.ts`). The `backendAdminRequest` IPC (`ipc/app-handlers.ts`) is a tightly-allowlisted proxy (`/api/model-profiles*`, `/api/models/adapters/status`, `/api/models/adapters/recommendation`) that injects both the Bearer token and the `X-Admin-Token` header — the renderer never sees the admin token.
- **Export pipeline:** `main.ts` imports `registerExportHandlers` and `stopExportProcess` from `electron/export/` — see `electron/export/codemap.md`. Note: that ProRes path is **timeline/export-only**, not primary generation output.
- **File-import proxying:** `ipc/file-handlers.ts` transcodes imported videos to H.264/AAC in place (via the same ffmpeg binary the export layer resolves) so the renderer's `<video>` can play them — directly relevant to any future MOV/ProRes/EXR primary-output proxy plan, since the same "make a browser-playable copy" pattern will be reused.
- **Shared contract:** `shared/feature-flags.ts` (`HF_GATING_ENABLED`) is imported by both `python-backend.ts` (env passthrough) and `preload.ts` (renderer-visible flag), keeping it free of main-process APIs.
