# frontend/views

## Responsibility

Top-level routed screens rendered inside the app shell. Four files:

- **`Home.tsx`** — Project gallery / landing screen. Lists projects, creates/renames/deletes them, and triggers the one-time project-references migration on mount.
- **`Project.tsx`** — Active project container. Hosts the `gen-space` / `video-editor` tab switcher, runs the visual-asset-metadata migration pass for the active project, and routes pending Retake/IC-LoRA clip updates into the editor.
- **`GenSpace.tsx`** — The generation UI. Renders the asset gallery + an embedded `PromptBar` (declared in the same file) and drives all four generation modes (image, video, retake, ic-lora) plus result persistence into the active project.
- **`VideoEditor.tsx`** — The non-linear editor host. Creates the editor store, wires up playback/keyboard/regeneration/import hooks, composes the resizable `react-resizable-panels` layout, and mounts the panels implemented under `frontend/views/editor/`.

## Design Patterns

- **Context-only state.** All screens read/mutate app state through `useProjects` (`ProjectContext`), `useAppSettings` (`AppSettingsContext`), `useView` (`ViewContext`), and `useKeyboardShortcuts` (`KeyboardShortcutsContext`). No Redux/Zustand at this layer.
- **Composition over inheritance.** `Project` renders either `<GenSpace />` or `<VideoEditor key={activeProject.id} ... />` based on `currentTab`. The `key` forces a fresh editor instance per project.
- **Embedded sub-components in `GenSpace.tsx`.** `AssetCard`, `SettingsDropdown`, `PromptBar`, and several icon components (`LightricksIcon`, `ZitIcon`, `AspectIcon`, grid-size icons) are private to the file; only `GenSpace` is exported.
- **Generation delegated to hooks.** `GenSpace` owns no fetch logic — it uses `useGeneration`, `useRetake`, `useIcLora`, and `useVideoGenerationModelSpecs` and consumes their `{isGenerating, progress, statusMessage, videoPath/imagePaths, error, reset}` tuples.
- **Editor store isolation.** `VideoEditor` is split into a thin outer `VideoEditor` (owns the `EditorStoreApi` ref + pending-take application) and an inner `VideoEditorWithStore` wrapped in `EditorStoreProvider`. This lets the store survive re-renders while keeping selector consumers under the provider.
- **Bridging via refs.** The editor communicates playback time, zoom/fit, focus, and gap-generation actions to panels through `useRef` handles (`playbackTimeRef`, `fitToViewRef`, `matchFrameRef`, `selectedGapRef`, etc.) rather than prop drilling state on every frame.

## Data & Control Flow

### GenSpace generation modes

`PromptBar` receives `mode: 'image' | 'video' | 'retake' | 'ic-lora'` and a shared `settings` object (`DEFAULT_VIDEO_SETTINGS`: `model:'fast'`, `duration:5`, `videoResolution:'540p'`, `fps:24`, `aspectRatio:'16:9'`, `imageResolution:'1080p'`, `variations:1`, `audio:true`). The generate button is gated by a single `canSubmit` flag computed once:

- **retake**: `retakeInput.ready && !!retakeInput.videoPath && !isRetaking`
- **ic-lora**: `(isInOutpainting || isHdr || !!prompt.trim()) && icLoraInput.ready && (!!icLoraInput.videoPath || isIngredients) && !isIcLoraGenerating`
- **image/video**: `!!prompt.trim() && hasCompatibleVideoSettings`

`handleGenerate` mirrors `canSubmit` exactly with early `return`s — retake requires `videoPath + duration>=2`; ic-lora requires `ready`, a video unless `adapterId==='ingredients'`, and a prompt unless `adapterId==='in_outpainting'` or `adapterId==='hdr'` (HDR is source-video-driven and ignores the prompt). This alignment is the Generate no-op fix (button enabled ⇔ handler runs). For HDR the IC-LoRA submit sends `conditioningType: null` + `images: []`; the response's EXR `videoPath` is persisted as the primary asset and `proxyPath` as the in-app preview sidecar.

### Ingredients vs V2V LoRA output settings

`PromptBar` derives `isIngredients = mode==='ic-lora' && icLoraInput.adapterId==='ingredients'`. `showIcLoraOutputSettings={isIngredients}` is passed down:

- **Ingredients** (`showIcLoraOutputSettings` true): the bar renders free-numeric **FPS** `<input type="number">` (1–120, calls `onFpsChange`), plus **DURATION**, **RESOLUTION** (`settings.videoResolution`), and **ASPECT RATIO** dropdowns resolved via `resolveVideoGenerationOptions({settings, modelSpecs, hasAudio:false})`. On generate, `handleGenerate` derives output geometry locally: a `RES_MAP` (`540p/720p/1080p` → `{width,height}`), swaps width/height for `9:16`, and computes `numFrames = max(9, 1 + 8*floor((duration*fps)/8))`, passing `{width,height,numFrames,frameRate}` only for Ingredients.
- **V2V LoRAs** (standard_video / in_outpainting): the output-settings block is hidden — these reuse the source driving video's fps/dimensions, so no FPS/duration/resolution controls are shown.

### Result persistence

Both video and image completion are handled in `useEffect`s keyed on `videoPath`/`imagePaths`. `persistedVideoKeyRef` guards against double-persisting the same `videoPath`. Results are copied into project storage via `addVisualAssetToProject` then registered with `addAsset`/`addTakeToAsset`. Retake and IC-LoRA results persist inside the hook's async completion callback (survives GenSpace unmount) and, when launched from the editor, publish `setPendingRetakeUpdate`/`setPendingIcLoraUpdate` so the editor can re-point linked clips to the new take.

### Incoming hand-offs into GenSpace

`useEffect`s consume context fields set by the editor: `genSpaceEditImagePath` (→ video mode I2V), `genSpaceAudioPath` (→ A2V), `genSpaceRetakeSource` (→ retake, pre-fills prompt from source asset), `genSpaceIcLoraSource` (→ ic-lora, blocked when `forceApiGenerations`). Each effect clears its trigger after consuming it.

### VideoEditor lifecycle

1. Outer component lazily creates the store once: `getEditorModel(currentProject)` → `applyPendingClipTakeUpdate` for both retake & ic-lora pending updates → `createEditorStore(createInitialEditorState(model, loadLayout()))`.
2. A follow-up effect re-applies pending updates arriving after mount via `store.getState().setStateWithoutHistory`, then clears the pending flags (Bug B fix).
3. Inner `VideoEditorWithStore` subscribes to selectors (`editorModel`, `currentTime`, `selectedClipIds`, `layout`, `showSourceMonitor`, …), instantiates `useEditorKeyboard`, `usePlaybackEngine`, `usePlaybackAudioSync`, `useRegeneration`, `useEditorMediaImport`, `useSubtitleImportExport`, `useTimelineXmlExport`, and `useBuildMenuDefinitions`.
4. Autosave debounces (`AUTOSAVE_DELAY=500ms`) a `saveProject(updatedProject(currentProject, editorModel))` call and also flushes on unmount.

### Project metadata migration

`Project.useEffect` streams `runVisualAssetMetadataMigration(activeProjectAssets, electronAPI)` events, applying `updateAsset` patches and showing a progress screen while `hasVisualAssetMetadataForMigration` is true. `Home` runs `useProjectReferencesMigration` once.

## Integration Points

- **`frontend/contexts`** — `ProjectContext` (assets, takes, `pendingRetakeUpdate`/`pendingIcLoraUpdate`, gen-space hand-off fields), `AppSettingsContext` (`shouldVideoGenerateWithLtxApi`, `forceApiGenerations`, `hasLtxApiKey`), `ViewContext` (`openProject`/`goHome`), `KeyboardShortcutsContext` (`activeLayout`, `isEditorOpen`).
- **`frontend/hooks`** — `use-generation`, `use-retake`, `use-ic-lora`, `use-video-generation-model-specs`, `useProjectReferencesMigration`.
- **`frontend/lib`** — `video-generation-model-specs` (`resolveVideoGenerationOptions`, `sanitizeVideoGenerationSettings`, `areVideoGenerationSettingsEquivalent`, `getVideoGenerationModelSpecs`), `asset-copy` (`addVisualAssetToProject`), `file-url` (`pathToFileUrl`), `generation-errors` (`createLocalGenerationError`), `project-asset-metadata-migration`, `logger`.
- **`frontend/components`** — `RetakePanel`, `ICLoraPanel`, `FreeApiKeyBubble`, `GenerationErrorDialog`, `ExportModal`, `ImportTimelineModal`, `MenuBar`, `Tooltip`.
- **`frontend/views/editor/`** — `VideoEditor` imports the store (`editor-store`), state (`editor-state`), selectors (`editor-selectors`), actions (`editor-actions`), bridging (`editor-project-bridging`), utils (`video-editor-utils`), hooks (`useEditorKeyboard`, `usePlaybackEngine`, `usePlaybackAudioSync`, `useRegeneration`, `useEditorMediaImport`, `useSubtitleImportExport`, `useTimelineXmlExport`, `useBuildMenuDefinitions`), and all editor panels. See `frontend/views/editor/codemap.md` for the full store/selectors/actions/undo-redo model.
- **Electron bridge** — `window.electronAPI.extractVideoFrame`, `getPathForFile`, `showOpenFileDialog`, `getResourcePath`, plus file `file://` URLs for media playback.
