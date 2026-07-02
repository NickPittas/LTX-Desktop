# frontend/contexts
## Responsibility
The four React contexts that hold all cross-cutting renderer state (no Redux/Zustand in app code). `AppSettingsContext` mirrors backend process liveness into settings/runtime-policy state and exposes API-key savers; `KeyboardShortcutsContext` persists the active keymap; `ProjectContext` is the localStorage-backed project domain store + GenSpace hand-off state; `ViewContext` is a thin view router over `ProjectContext`.

## Design Patterns
- **Uniform context shape:** every file uses `createContext<T | null>(null)` + a `*Provider` + a `use*()` hook that throws `'must be used within *Provider'` when the value is null.
- **Memoized provider values** (e.g. `AppSettingsContext`, `KeyboardShortcutsContext`) to bound consumer re-renders.
- **`ProjectContext` is the domain store; `ViewContext` is a façade** that depends on `useProjects()` (`activeProject`, `activateProject`, `clearActiveProject`, `setCurrentTab`) rather than owning data.
- **IPC push + snapshot subscription** (`AppSettingsContext`, mirrored independently in `hooks/use-backend.ts`): subscribe `onBackendHealthStatus` and seed via `getBackendHealthStatus()`, then `resetBackendCredentials()` on `'alive'`.
- **localStorage-backed context** (`KeyboardShortcutsContext`, `ProjectContext`): load once into a ref/state, persist on change via effect or mutation helper.
- **Stale-closure ref** (`KeyboardShortcutsContext.activeLayoutRef`, `ProjectContext.projectRevision`) so callbacks read the latest layout / force `useCallback` identity refresh.
- **Fail-closed policy default** (`AppSettingsContext`): if `getRuntimePolicy` fails or shape is wrong, `forceApiGenerations` becomes `true`.

## Data & Control Flow
**`AppSettingsContext.tsx`**
- State: `settings: AppSettings` (initialized to `DEFAULT_APP_SETTINGS`), `isLoaded`, `runtimePolicyLoaded`, `forceApiGenerations` (default `true`), `backendProcessStatus`.
- Effect 1 (on `backendProcessStatus === 'alive'`): `ApiClient.getRuntimePolicy()` → sets `forceApiGenerations` from `payload.force_api_generations` + `runtimePolicyLoaded=true` (fail-closed `true` on error/non-boolean).
- Effect 2: subscribes `window.electronAPI.onBackendHealthStatus` + `getBackendHealthStatus()` snapshot → `toBackendProcessStatus` → on `'alive'` calls `resetBackendCredentials()`.
- Effect 3 (when alive & not loaded): `refreshSettings()` → `ApiClient.getSettings()` → `normalizeAppSettings`; retries every 1000 ms on failure.
- Effect 4 (auto-sync, 150 ms debounce when alive & loaded): `ApiClient.updateSettings(syncPayload)` with `hasLtxApiKey`/`hasFalApiKey`/`hasGeminiApiKey`/`modelsDir` stripped.
- `saveLtxApiKey`/`saveFalApiKey`/`saveGeminiApiKey`: `ApiClient.updateSettings({ ltxApiKey | falApiKey | geminiApiKey })` then `refreshSettings()`.
- Derived: `shouldVideoGenerateWithLtxApi = forceApiGenerations || (settings.userPrefersLtxApiVideoGenerations && settings.hasLtxApiKey)`.

**`KeyboardShortcutsContext.tsx`**
- `loadFromStorage()`/`saveToStorage()` against `localStorage['ltx-keyboard-shortcuts']` (`PersistedState = { activePresetId; customLayout?; customPresets? }`).
- `activeLayout = customLayout || [...BUILT_IN_PRESETS, ...customPresets].find(id)?.layout || LTX_DEFAULT_LAYOUT`; `activeLayoutRef` mirrors it for `updateBinding`.
- Persist effect writes on every `activePresetId`/`customLayout`/`customPresets` change.
- `switchPreset`/`resetToPreset` clear `customLayout`; `saveAsCustomPreset` clones current layout into a `custom-<ts>` preset; `deleteCustomPreset` falls back to `'ltx-default'`.

**`ProjectContext.tsx`**
- `loadInitialProjectIds()`: returns `[]` while `hasLegacyProjectsEntry()` (defers to migration hook).
- Core mutator `persistProject(id, project)` = `writeProject(id, normalizeProject({...project, id}))` → update `activeProject` if match → `bumpProjectRevision()`. `mutateProject(id, updater)` = `readProject` → `updater` → `persistProject`.
- Project CRUD: `createProject` (generates `project-<ts>-<rand>` id, `createDefaultTimeline('Timeline 1')`, prepends to `projectIds`), `deleteProject`, `renameProject`, `activateProject`/`clearActiveProject`/`reloadProjectIds`.
- Asset ops (`addAsset`/`deleteAsset`/`updateAsset`/`addTakeToAsset`/`deleteTakeFromAsset`/`setAssetActiveTake`/`toggleFavorite`) all go through `mutateProject`; take ops sync the asset's primary `path`/`bigThumbnailPath`/`smallThumbnailPath`/`width`/`height` to the active/selected take, seeding `takes` from the primary fields when absent.
- GenSpace hand-off state: `genSpaceEditImagePath`/`genSpaceEditMode` (`'image'|'video'|null`)/`genSpaceAudioPath`/`genSpaceRetakeSource`+`pendingRetakeUpdate`/`genSpaceIcLoraSource`+`pendingIcLoraUpdate`.

**`ViewContext.tsx`**
- `openProject(id)` = `activateProject(id)` + `setCurrentTab('gen-space')` + `setCurrentView('project')`.
- `goHome()` = `clearActiveProject()` + `setCurrentView('home')`.
- Guard effect: if `currentView === 'project' && !activeProject` → force `'home'`.

## Integration Points
- **`types/project-model`:** `Project`, `Asset`, `AssetTake`, `ProjectTab`, `ViewType`, `createDefaultTimeline`, `normalizeProject` (ProjectContext, ViewContext).
- **`lib/project-storage`:** `readProject`/`writeProject`/`readProjectIds`/`writeProjectIds`/`deleteProjectEntry`/`getProjectStorageKey`/`PROJECT_IDS_STORAGE_KEY`/`PROJECT_STORAGE_KEY_PREFIX` (ProjectContext).
- **`hooks/useProjectReferencesMigration`:** `hasLegacyProjectsEntry` (ProjectContext initial load).
- **`lib/keyboard-shortcuts`:** `KeyboardLayout`/`KeyboardPreset`/`BUILT_IN_PRESETS`/`LTX_DEFAULT_LAYOUT`/`cloneLayout`/`ActionId` (KeyboardShortcutsContext).
- **`lib/api-client` (`ApiClient`)** + **`lib/backend` (`resetBackendCredentials`)** (AppSettingsContext).
- **`window.electronAPI`:** `onBackendHealthStatus`, `getBackendHealthStatus` (AppSettingsContext).
- **Consumers:** `App.tsx` (all four); `use-generation`/`use-hf-auth`/etc. (`useAppSettings`); `VideoEditor`/`GenSpace`/views (`useProjects`, `useView`); `KeyboardShortcutsModal` (`useKeyboardShortcuts`).
