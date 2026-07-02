# frontend/
## Responsibility
React 18 + TypeScript renderer entry layer. `main.tsx` mounts the app and installs dev-only storage debug tools; `App.tsx` composes the four React contexts, drives the boot/gating waterfall (Python → backend → license → setup → required-models), routes between the two top-level views, and hosts the global modals/overlays. Despite `zustand` being listed in `package.json`, it is never imported here — all state is React contexts.

## Design Patterns
- **Provider composition (outer → inner):** default export `App` wraps children as `ProjectProvider` → `ViewProvider` → `KeyboardShortcutsProvider` → `AppSettingsProvider`, then renders `<AppContent />` + `<KeyboardShortcutsModal />` inside `AppSettingsProvider` so `AppContent` can consume all four contexts.
- **Context-based view routing (no react-router):** `renderView()` switches on `useView().currentView` — `'home'` → `<Home />`, `'project'` → `<Project />`, default → `<Home />`.
- **Gating waterfall as early-return state machine:** sequential gates each short-circuit render — `pythonReady` (`null`/`false` → `<PythonSetup>`), backend `processStatus === 'dead'` → crash screen with embedded `<LogViewer>` + Restart, `setupState.needsLicense`/`needsSetup` → `<LaunchGate>`, `requiredModelsGate === 'missing'` → `<LaunchGate>`, otherwise `renderView()`.
- **DOM CustomEvent command bus:** listens for `'open-settings'` (sets `settingsInitialTab` + opens `SettingsModal`) and `'open-api-gateway'` (builds `ApiGatewayRequest` → opens `ApiGatewayModal`); `use-generation` and others dispatch these events to request keys.
- **In-flight dedup refs:** `setupCompletionInFlightRef`, `localSetupCompletedRef`, `pendingLocalModelsSetupRef`, `forcedApiGatewayRequestRef` prevent double finalize / re-entry.
- **Backend liveness → credentials reset:** on `processStatus === 'alive'` the app (via `useBackend` and `AppSettingsContext`) calls `resetBackendCredentials()` so the next `backendFetch` re-reads port/token.

## Data & Control Flow
1. `main.tsx`: `installProjectStorageDevtools()` (DEV-only, exposes `window.__ltxProjectStorageDebug`) → `ReactDOM.createRoot(...).render(<React.StrictMode><App /></React.StrictMode>)`; imports `index.css`.
2. Provider nest (see above); `AppContent` runs the boot sequence.
3. `checkPythonReady()` IPC → if not ready, `<PythonSetup onReady>`; else `startPythonBackend()` IPC (guarded by `backendStarted`).
4. `useBackend()` subscribes `onBackendHealthStatus` + `getBackendHealthStatus` snapshot → drives `connected`/`processStatus`/`isLoading`. `AppSettingsContext` independently mirrors status → loads runtime policy (`ApiClient.getRuntimePolicy` → `forceApiGenerations`, fail-closed `true`) then settings (`ApiClient.getSettings`, retry loop).
5. `checkFirstRun()` IPC → `setupState` (`'loading' | { needsSetup; needsLicense }`); render `<LaunchGate>` with `handleAcceptLicense` / `handleFirstRunComplete` (`completeSetup` IPC) / `onLocalModelsComplete`.
6. `areModelsReady()`: `ApiClient.getModelProfiles()` (active profile with `components.transformer` ⇒ ready), else `Promise.all([getLtxRecommendation, getImgGenRecommendation])` checks `status !== 'download'` and `cp_to_download === null` → sets `requiredModelsGate` (`'checking'|'missing'|'ready'`).
7. On `connected && requiredModelsGate === 'ready' && !forceApiGenerations`: `ApiClient.getLtxRecommendation()` → if `status === 'upgrade'` and not dismissed, render `<LtxUpgradePrompt>` (`handleDismissLtxUpgradePrompt` / `handleCompleteLtxUpgradePrompt`).
8. Forced-API path: if `forceApiGenerations && !settings.hasLtxApiKey`, auto-opens blocking `ApiGatewayModal` (`requiredKeys: ['ltx']`); `shouldAutoFinalizeForcedFirstRun` calls `handleFirstRunComplete` once an LTX key exists.

## Integration Points
- **Contexts (`frontend/contexts/`):** `useView` (ViewContext), `useBackend` mirrors status consumed here via `useAppSettings`; `ProjectProvider`/`KeyboardShortcutsProvider`/`AppSettingsProvider` wrap `AppContent`.
- **Hooks (`frontend/hooks/`):** `useBackend` (process status), indirectly `use-generation`/others dispatch `'open-api-gateway'`.
- **Lib (`frontend/lib/`):** `ApiClient` + `ApiSuccessOf` (`lib/api-client`), `resetBackendCredentials` (`lib/backend`), `logger` (`lib/logger`), `installProjectStorageDevtools` (`lib/project-storage-devtools`).
- **Views:** `views/Home`, `views/Project`.
- **Components:** `FirstRunSetup` (`LaunchGate`), `LtxUpgradePrompt`, `PythonSetup`, `SettingsModal` (`SettingsTabId`), `LogViewer`, `ApiGatewayModal` (`ApiGatewaySection`), `KeyboardShortcutsModal`, `ui/button`.
- **Electron IPC (`window.electronAPI`, defined in `electron/preload.ts`):** `checkPythonReady`, `startPythonBackend`, `checkFirstRun`, `completeSetup`, `acceptLicense`, `openLtxApiKeyPage`, `openFalApiKeyPage`, `onBackendHealthStatus`, `getBackendHealthStatus`.
- **Backend HTTP:** all via `ApiClient` → `backendFetch`/`backendAdminFetch` (never raw `fetch`).
