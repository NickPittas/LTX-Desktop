# frontend/views/editor

## Responsibility

The non-linear video editor. 30 files implementing the editor state machine, the timeline/program/source/assets UI, and the supporting hooks. The folder is organized as a classic **store / selectors / actions** split plus presentational components and controller hooks.

- **State core** — `editor-state.ts` (types + `createInitialEditorState` + undo snapshot helpers), `editor-store.tsx` (Zustand vanilla store + React binding + undo/redo recording), `editor-selectors.ts` (pure derivations), `editor-actions.ts` (163 reducer-style mutators), `editor-project-bridging.ts` (`Project` ⇔ `EditorModel` conversion).
- **Shared utils** — `video-editor-utils.ts` (tools, color labels, layout persistence, active clip color-correction/transform CSS, time formatting, overlap resolution, migrations).
- **Hooks** — `useEditorKeyboard`, `usePlaybackEngine`, `usePlaybackAudioSync`, `useRegeneration`, `useEditorMediaImport`, `useSubtitleImportExport`, `useTimelineXmlExport`, `useTimelineDrag`, `useBuildMenuDefinitions`.
- **Components** — 17 panels/widgets (see below).

## Design Patterns

### Store / selectors / actions split

- **`editor-state.ts`** defines `EditorState` as four slices: `editorModel` (`{assets, bins, timelines, activeTimelineId}` — the persisted/document data), `session` (`selection`, `transport`, `tools`, `ui`, `regeneration`, `clipboard` — ephemeral UI state), `history` (`undoStack`/`redoStack` of `EditorUndoSnapshot`), and `projectSync` (`dirty` flag). `createInitialEditorState(model, layout)` seeds defaults (zoom 1, snap on, tool `select`, focus `timeline`).
- **`editor-store.tsx`** builds a Zustand *vanilla* store (`createStore`) wrapping `{state, setStateWithHistory, setStateWithoutHistory}`. `EditorStoreProvider` injects the `StoreApi` via context. `useEditorStore(selector, equalityFn)` is the only subscription hook (uses `useStoreWithEqualityFn`). `useEditorGetState()` returns a stable `() => state` for non-reactive reads (used heavily in keyboard/drag hot loops). `useEditorActions()` reflects every exported function in `editor-actions.ts` into a callback that applies the reducer through `setStateWithHistory` — **except `undo`/`redo`**, which use `setStateWithoutHistory`.
- **`editor-selectors.ts`** holds ~60 pure `(state) => T` functions. Many are composed (`selectClips` → `selectActiveTimeline` → `getActiveTimelineFromEditorModel`). Reusable module-level empty constants (`EMPTY_CLIPS`, `DEFAULT_TIMELINE_TRACKS`) keep selector outputs referentially stable. Asset/clip derivations come in `*FromAssets` variants so playback hot loops can pass a captured `assets` array.
- **`editor-actions.ts`** reducers are pure `(state, ...args) => state`. Internal helpers `updateEditorModel` (marks `projectSync.dirty`) and `updateSession` (no dirty mark) centralize slice updates; `withActiveTimeline` scopes edits to the active timeline. Action IDs flow to callers through the typed `EditorActions` map in the store.

### Undo / redo model

Only the `editorModel` slice (`assets`, `bins`, `timelines`) participates in history. `getUndoSnapshot`/`applyUndoSnapshot`/`equalUndoSnapshot` (in `editor-state.ts`) define what is checkpointed; `session`, `history`, and `projectSync` are never snapshotted. `recordHistoryStep` (in the store) pushes the *previous* snapshot onto `undoStack` (capped at `MAX_UNDO_HISTORY=50`), clears `redoStack`, and short-circuits when `equalUndoSnapshot` says nothing changed — so transport/selection/tool changes never pollute history. `undo`/`redo` are the only actions dispatched without history and both carefully maintain the opposite stack, collapsing no-op snapshots.

### Component conventions

- **Store-connected components** subscribe via `useEditorStore(selectX)` and dispatch via `useEditorActions()` (e.g. `ProgramMonitor`, `ClipPropertiesPanel`, `SubtitlePropertiesPanel`, `VideoEditorTimelineControlPanel`, `SubtitleTrackStyleEditor`).
- **Props-driven presentational components** receive callbacks and data from a parent (e.g. `ClipContextMenu`, `AssetContextMenu`, `TakeContextMenu`, `GapGenerationModal`, `TimelineToolbar`, `VideoEditorLayoutMenu`). This keeps them testable and avoids nested store subscriptions.
- **Imperative handles** expose escape-hatch APIs to the host: `ProgramMonitorHandle.toggleFullscreen`, `VideoEditorAssetsPanelHandle.{revealAsset, deleteAsset}`, `VideoEditorSourceMonitorHandle.{openAsset, pause, dispatchKeyboardAction}`.

## Data & Control Flow

### Project ⇔ editor bridging (`editor-project-bridging.ts`)

`getEditorModel(project)` normalizes every timeline (defaults tracks to `DEFAULT_TRACKS`, runs `migrateTracks`/`migrateClip`, ensures `subtitles`). `updatedProject(project, editorModel)` projects the model back onto the `Project` (bumping `updatedAt`). `applyPendingClipTakeUpdate(model, pending)` re-points clips matching `{assetId, clipIds}` to a new `takeIndex` — used at mount and on runtime pending Retake/IC-LoRA updates.

### Selection

`selectSelectedClipForProperties` resolves the clip shown in the properties panel: single selection returns that clip; multi-selection returns a representative only if every selected clip belongs to one linked group, picking the video/image clip. `selectSelectedLinkedGroup` does BFS over `linkedClipIds` to expand a selection.

### Tracks & gaps

`selectOrderedTracks` reorders raw tracks for display (subtitles → video reversed → audio) and assigns `displayRow`. `selectTimelineGaps` finds empty ranges per track (>0.05s tolerance). `selectCutPoints` detects adjacent-clip edit points within `CUT_POINT_TOLERANCE` and flags dissolves.

### Clip derivations

`selectLiveAssetForClip`/`selectClipPathFromAssets` resolve the current take path (`clip.takeIndex ?? asset.activeTakeIndex`). `selectClipMaxDurationFromAssets` computes the usable timeline duration accounting for trim + speed. `selectClipResolutionFromAssets` labels 4K/1080p/720p/other with colors. `selectClipCapabilities` derives booleans (`canRegenerate`, `canRetake`, `canUseIcLora`, `canCreateVideoFromImage/Audio`, …) consumed by context menus.

### Export projections

`selectExportModalModel` aggregates `selectExportClipData` (filters disabled tracks, maps flip/speed/volume/opacity), `selectExportSubtitleData` (merges track + subtitle styles), and `selectExportLetterbox` (picks the longest enabled adjustment-layer letterbox, resolving aspect via `LETTERBOX_RATIO_MAP`).

### Playback

- **`usePlaybackEngine`** owns the rAF loop. While `isPlaying`, it advances `playbackTimeRef.current` by `deltaSec * effectiveSpeed` (shuttle speed or 1), loops within `[inPoint,outPoint]` when `playingInOut`, stops at 0/`totalDuration`, and commits `setCurrentTime` at most every `STATE_UPDATE_INTERVAL_MS=250ms` (plus on stop/unmount via a `useLayoutEffect` flush). `playbackTimeRef` is authoritative during playback; React state syncs only when paused.
- **`usePlaybackAudioSync`** mirrors the loop for audio `<audio>` elements pooled per clip id. It preloads a look-ahead window, activates/deactivates elements crossing the playhead, applies trim/reverse/speed/mute/volume/solo, and drift-corrects (>1.5s) or re-seeks. A separate effect handles paused scrubbing.

### Editing interactions (`useTimelineDrag.ts`)

A 1121-line controller hook instantiated by `VideoEditorTimelineEditingPanel`. It owns `draggingClip`/`resizingClip`/`slipSlideClip`/`lassoRect` state and exposes handlers: `handleRulerMouseDown` (scrub), `handleClipMouseDown` (move/select/lasso), `handleMouseMove`/`handleMouseUp` (move commit with `resolveOverlaps`), `handleResizeStart`/`handleResizeMove` (ripple/roll/regular trim with linked-audio and adjacent-clip coordination), `handleSlipSlideMove`/`handleSlipSlideUp` (slip/slide), `handleTrackDrop` (asset drag from the panel). It receives delegated setters (`setClips`, `setSelectedClipIds`, `splitClipAtPlayhead`, `addClipToTimeline`, …) so it mutates state through the host's action wrappers. Snap points (clip edges, playhead, in/out, cut points) are honored when `snapEnabled`.

### Keyboard (`useEditorKeyboard.ts`)

A single stable `useEffect` (no deps — reads latest state via refs) resolves the pressed combo through `resolveAction(kbLayout, event)` and switches on `ActionId`. It is **panel-aware**: transport/marking actions route to the source monitor when `focusArea==='source'` (via `sourceDispatchRef`) and to the timeline otherwise. JKL shuttle uses `FORWARD_SPEEDS`/`REVERSE_SPEEDS` ladders; `kHeldRef` turns shuttle into frame stepping. Editing actions (`edit.delete`, `edit.insertEdit`, `edit.matchFrame`, `nav.prevEdit`/`nav.nextEdit`) dispatch editor actions or invoke ref-bound handlers.

### Regeneration (`useRegeneration.ts`)

`handleRegenerate(assetId, clipId?)` calls `startClipRegeneration`, then either reuses `asset.generationParams` or synthesizes one for imported assets by extracting a frame and calling `ApiClient.suggestGapPrompt` (persisted back via `updateAsset`). It rejects retake/ic-lora modes. On `regenVideoPath`/`regenImagePath`, `persistGeneratedTake` copies the output and calls `applyGeneratedTake` (adds a take + sets `clip.takeIndex`). Failures route to `failClipRegeneration`/`cancelClipRegeneration`.

### Menus (`buildMenuDefinitions.ts`)

`useBuildMenuDefinitions(deps)` memoizes six `MenuDefinition`s (File, Edit, Clip, Sequence, Tools, View, Help) wiring shortcuts via `getShortcutLabel` and gating items on `selectCanUndo/Redo/UseClipboard/InsertEdit/OverwriteEdit` and `selectMenuState`.

### Layout persistence (`video-editor-utils.ts`)

`EditorLayout` (`leftPanelWidth`, `rightPanelWidth`, `timelineHeight`, `assetsHeight`) is clamped by `LAYOUT_LIMITS`, persisted to `localStorage` under `LAYOUT_STORAGE_KEY`, and reloaded via `loadLayout`. Presets live under `LAYOUT_PRESETS_KEY` (`loadLayoutPresets`/`saveLayoutPresets`). The host applies sizes to `react-resizable-panels` `PanelImperativeHandle.resize`.

## Component reference

- **`ProgramMonitor.tsx`** (forwardRef, `ProgramMonitorHandle`) — timeline preview. Composites a video pool + cross-dissolve incoming clip, image layers, text overlays, subtitles, letterbox, and audio-only clips using a `buildFrameRenderCache` + `deriveFrameRenderState` pipeline. Manages preview zoom/pan, playback resolution, and fullscreen.
- **`VideoEditorTimelineEditingPanel.tsx`** (3306 lines) — the timeline orchestration hub. Renders ruler, track headers, clips, gap selection, `TimelineToolbar`, `ClipContextMenu`, `GapGenerationModal`. Instantiates `useTimelineDrag` and owns gap-generation suggestion flow (extracting before/after frames, `ApiClient.suggestGapPrompt`, abortable).
- **`VideoEditorAssetsPanel.tsx`** (forwardRef, `VideoEditorAssetsPanelHandle`) — left-panel media library. Grid/list views, filtering/sorting (`selectVisibleAssets`), bins, lasso selection, takes view, and the asset/take/bin context menus. Bridges to `AssetContextMenu`, `TakeContextMenu`, `VideoThumbnailCard`.
- **`VideoEditorSourceMonitor.tsx`** (forwardRef, `VideoEditorSourceMonitorHandle`; exports `SourceKeyboardAction`) — clip viewer with In/Out marks, loop playback, and insert/overwrite edit dispatch (`insertSourceEdit`/`overwriteSourceEdit`).
- **`ClipPropertiesPanel.tsx`** — right panel; Properties/Metadata tabs. Flip, transitions, color correction, letterbox, text style, takes, speed, opacity, audio levels (`selectSelectedClipAudioControls`).
- **`ClipContextMenu.tsx`** (props-driven; `ClipContextMenuState` = `background | clip`) — copy/cut/paste, duplicate, split, delete, color, takes, regenerate, reveal, create-video-from-image/audio, retake, IC-LoRA, capture frame, speed, flip, mute, link/unlink.
- **`AssetContextMenu.tsx`** (props-driven) — add to timeline, regenerate, color labels, bins, takes, delete.
- **`TakeContextMenu.tsx`** (props-driven) — set active take, add to timeline, create asset from take, delete take.
- **`GapGenerationModal.tsx`** (props-driven) — generate media into a timeline gap (text-to-video / image-to-video / text-to-image), with prompt suggestion, before/after frames, and `SettingsPanel`.
- **`SubtitlePropertiesPanel.tsx`**, **`SubtitleTrackStyleEditor.tsx`** — per-subtitle and per-track subtitle style editors.
- **`TimelineToolbar.tsx`**, **`VideoEditorLayoutMenu.tsx`** (preset save/apply/reset), **`VideoEditorTimelineControlPanel.tsx`** (timeline tabs + create/import/rename/delete), **`VideoThumbnailCard.tsx`** (hover-scrub canvas preview). Persisted project effect schemas remain supported for round-trip compatibility, but effect runtime/UI helpers are inactive.

## Integration Points

- **Host** — `frontend/views/VideoEditor.tsx` creates the store, applies pending updates, wires all hooks, and mounts these panels inside `react-resizable-panels`. See `frontend/views/codemap.md`.
- **Types** — `frontend/types/project-model` (`Asset`, `Timeline`, `TimelineClip`, `Track`, `SubtitleClip`, `SubtitleStyle`, `LetterboxSettings`, `TextOverlayStyle`, `DEFAULT_TRACKS`, `DEFAULT_SUBTITLE_STYLE`, `DEFAULT_COLOR_CORRECTION`), `frontend/types/project` (`TEXT_PRESETS`).
- **Lib** — `frontend/lib/keyboard-shortcuts` (`resolveAction`, `ACTION_REGISTRY`, `formatKeyCombo`, `findConflicts`), `frontend/lib/asset-copy`, `frontend/lib/file-url`, `frontend/lib/srt` (`parseSrt`/`exportSrt`), `frontend/lib/timeline-import` (`parseTimelineXml`, `exportFcp7Xml`), `frontend/lib/api-client`, `frontend/lib/logger`.
- **Hooks** — `use-generation`, `useAppSettings` (`shouldVideoGenerateWithLtxApi`, `forceApiGenerations`).
- **Store reuse outside the editor** — `frontend/components/ExportModal.tsx` and `frontend/components/ImportTimelineModal.tsx` import `editor-selectors`/`editor-store` directly, so they must be rendered within the `EditorStoreProvider` (i.e. inside `VideoEditor`).
- **Zustand** — `zustand/vanilla` (`createStore`), `zustand/traditional` (`useStoreWithEqualityFn`), `zustand/vanilla/shallow` (`shallow` equality in selectors).
