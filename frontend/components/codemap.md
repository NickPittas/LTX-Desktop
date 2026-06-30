# frontend/components

## Responsibility

Reusable, mostly self-contained UI building blocks shared across screens: dialogs/modals, setup gates, generation sub-panels, branding, inputs, and the app menu. None of these own routing; they are mounted by `Home`, `Project`/`GenSpace`, `VideoEditor`, or the root app shell. Several consume context (`AppSettingsContext`, `KeyboardShortcutsContext`) or the editor store directly; most are presentational.

## Design Patterns

- **Controlled components.** Modals receive `isOpen`/`onClose` (or `isOpen` derived from a store selector) and emit completion via callbacks. API-key inputs (`LtxApiKeyInput`, `ApiGatewayModal`) never persist themselves — they call `onSave`/`onChange` handlers supplied by `AppSettingsContext`.
- **Discriminated union error model.** `GenerationErrorDialog` switches exhaustively on `error.status` (`402 | 4XX | 5XX | default`) and `error.error.code`, mapping each to a human message, technical JSON detail, and optional primary action (`assertNever` guards completeness). `getGenericHumanMessage` substring-matches for OOM/409/network/model-load cases.
- **Model download gating, repeated as a recipe.** `FirstRunSetup.LaunchGate`, `LtxUpgradePrompt`, `ICLoraPanel`, and `SettingsModal` (text-encoder/models tabs) all follow the same flow: `useHfAuth` → `useHfModelAccess(cpIds, hfAuthStatus)` → `ApiClient.startModelDownload({cp_ids})` → poll `ApiClient.getModelDownloadProgress({sessionId})` until `complete`/`error`. Each renders HF sign-in, license-accept prompts, per-file progress bars, and a refresh action.
- **Reusable settings primitives.** `SettingsPanel` exports the canonical `GenerationSettings`/`GenerationMode` types consumed across hooks (`use-generation`, `useRegeneration`, `GapGenerationModal`) and renders video or image controls driven by `video-generation-model-specs` (`resolveVideoGenerationOptions`/`sanitizeVideoGenerationSettings`).
- **Editor-store-aware components.** `ExportModal` and `ImportTimelineModal` import `editor-selectors`/`editor-store` directly and therefore must live inside `EditorStoreProvider` (i.e. within `VideoEditor`).
- **Imperative/event bridges.** `FreeApiKeyBubble`, `SettingsModal` (via the `open-settings` CustomEvent), and `LtxLogo` use `window.dispatchEvent(new CustomEvent('open-settings', {detail:{tab}}))` to request deep-linked settings tabs from the app shell.

## Data & Control Flow

### Generation sub-panels

- **`ICLoraPanel.tsx`** — exports `ICLoraConditioningType` (`'canny' | 'depth' | null`), `CONDITIONING_TYPES` (None/Canny/Depth), `IC_LORA_ADAPTERS` (flat registry tagged by `AdapterWorkflow`: `standard_video`, `ingredients`, `in_outpainting`, `hdr`, `unavailable`), and the panel. A `useEffect` computes readiness and emits `onChange({videoPath, conditioningType, conditioningStrength, adapterId, maskPath, images, ready, maskGrowPx, laplacianBlendGrow, finalMaskBlurPx})`.
  - **HDR mode** (`adapterId === 'hdr'`, workflow `hdr`): hides conditioning-type/strength controls, drops the prompt-required expectation (prompt is optional), and renders grounded copy — HDR uses the source video/sequence and writes a linear EXR primary with an SDR proxy preview. `GenSpace` preserves the EXR `videoPath` as the primary asset and routes `proxyPath` to the preview player.
  - **Ingredients workflow**: `needsVideo = selectedWorkflow !== 'ingredients'` is false, so the input column renders a static "no driving video needed" placeholder and the panel forces `videoPath: null` and `conditioningType: null` in every `onChange`. The right column becomes an ingredient-image picker (`ingredientPaths` → `images`). There are **no FPS/duration/resolution controls inside the panel** — those live in `GenSpace`'s embedded `PromptBar` (Ingredients branch).
  - **in_outpainting workflow**: requires `maskPath`; exposes mask-grow / Laplacian-blend-grow / final-blur sliders.
  - **standard_video workflow**: driving video + optional canny/depth conditioning (preview via `ApiClient.extractIcLoraConditioning`, throttled 300ms).
  - `checkIcLoraAvailability` merges `getLtxIcLoraRecommendation` + `getAdapterRecommendation` into `requiredIcLoraCpIds`; the download gate UI renders until `icLoraReady`.
- **`RetakePanel.tsx`** — video trim selector. Snaps selections to VAE-compatible frame counts via `snapFramesDown` (`1 + 8*floor((n-1)/8)` at `RETAKE_FPS=24`). Generates a 20-thumbnail filmstrip through `window.electronAPI.extractVideoFrame`. Emits `onChange({videoPath, startTime, duration, videoDuration, ready})`. `MIN_DURATION=2` enforces the lower bound.
- **`FreeApiKeyBubble.tsx`** — surfaces only after 2.5s of generation while `!forceApiGenerations && !hasLtxApiKey`. A module-level `dismissedThisSession` flag persists dismissal until reload. "Get a free LTX API key" dispatches `open-settings` with `tab:'apiKeys'`.

### Setup / gating

- **`FirstRunSetup.tsx`** exports `LaunchGate` (steps `license → source → location → installing → complete`). Builds download steps from `getLtxRecommendation` + `getImgGenRecommendation` (`buildDownloadSteps`), gates on HF auth/access (`buildAccessCheckpointIds`), and calls `onComplete`/`onAcceptLicense`/`onLocalModelsComplete`.
- **`PythonSetup.tsx`** — listens to `onPythonSetupProgress` IPC for the bundled Python runtime download (`downloading/extracting/complete/error`), calls `onReady` on `complete`, plays a splash video.
- **`LtxUpgradePrompt.tsx`** — handles `getLtxRecommendation` `status:'upgrade'`. Phases `idle → starting → downloading → finishing`; only closeable in `idle`.
- **`ApiGatewayModal.tsx`** — generic sectioned API-key collector (`ApiGatewaySection[]` keyed by `ApiKeyType = 'ltx' | 'fal'`). Auto-closes when all required sections become configured (unless `blocking`); Escape closes when non-blocking.

### Settings

- **`SettingsModal.tsx`** (1357 lines) — tabbed modal (`general | apiKeys | promptEnhancer | models | about`). Uses `useAppSettings` (`settings`, `updateSettings`, `saveLtxApiKey/saveFalApiKey/saveGeminiApiKey`, `forceApiGenerations`), `useHfAuth`, `useHfModelAccess`, `useModelProfiles`, `useOfficialAdapters`, and embeds `ModelProfileWizard`. Fetches text-encoder recommendation, analytics state, project-assets path, app version, notices, and license text on demand. Re-exports `AppSettings` and `SettingsTabId`.
- **`SettingsPanel.tsx`** — see pattern above. Image mode shows aspect ratio + quality (steps 4/8/12); video mode shows model/duration/resolution/FPS/aspect/audio/camera-motion with FPS shown only when `fpsOptions.length > 1`.
- **`ModelProfileWizard.tsx`** — multi-step local-model profile creator. `COMPONENT_FIELDS` lists 19 component slots (transformer, text_encoder_root, video_vae, upsampler, vocoder, audio_vae, person_detector, depth/pose processor, embeddings_connector, text_projection, 7 IC-LoRA slots, transformer_quantization). `PREFILL_CANDIDATES`/`OFFICIAL_ADAPTER_FILENAMES` pre-fill the known QuantStack/Kijai layout. Validates via `ApiClient` (`ModelProfileValidationResponse`) and calls `onCreated(profileId)`.

### Editor-adjacent

- **`ExportModal.tsx`** — codec (`h264 | prores | vp9` via `CODEC_INFO`), resolution (4K/1080p/720p), frame rate (24/25/30/60), ProRes profile. Reads the active timeline through editor selectors; `generateFCPXML` produces a Premiere/DaVinci interchange; closing dispatches `closeExportModal`. Subtitles burn-in via `DEFAULT_SUBTITLE_STYLE`; letterbox via `LETTERBOX_RATIO_MAP`.
- **`ImportTimelineModal.tsx`** — `select → parsing → relink → error` flow. `parseTimelineXml` produces a `ParsedTimeline`; media refs are checked (`ApiClient.checkMediaAvailability`), copied into project storage (`addVisualAssetToProject`/`addGenericAssetToProject`), and finally committed via `importParsedTimeline`.

### Supporting widgets

- **`KeyboardShortcutsModal.tsx`** — visual QWERTY editor (`KB_ROWS`), search, conflict detection (`findConflicts`), rebind capture, reset/save via `useKeyboardShortcuts`.
- **`MenuBar.tsx`** — generic dropdown menu bar + command-palette search over `MenuDefinition[]`; exports `MenuItem`/`MenuDefinition` types consumed by `useBuildMenuDefinitions`.
- **`AudioWaveform.tsx`** — decodes audio to amplitude peaks via WebAudio (`computeWaveform`, cached in module-level `waveformCache`), draws on canvas with a playhead. Supports `file://` via `electronAPI.readLocalFile`.
- **`LogViewer.tsx`** — polls `electronAPI.getLogs` every 2s, color-codes levels, download/open-folder/refresh/auto-scroll.
- **`GenerationErrorDialog.tsx`** — see error model above.
- **`LtxLogo.tsx`** — currentColor SVG wordmark.
- **`LtxApiKeyInput.tsx`** — password input + `ApiKeyHelperRow`/`LtxApiKeyHelperRow` (opens `electronAPI.openLtxApiKeyPage`).
- **`ModelComponentPicker.tsx`** — text input + folder/file browser using `showOpenFileDialog`/`showOpenDirectoryDialog`; exports `MODEL_FILE_FILTERS` (safetensors/gguf/all).

## Integration Points

- **`frontend/contexts`** — `AppSettingsContext` (`SettingsModal`, `ICLoraPanel`, `FreeApiKeyBubble`), `KeyboardShortcutsContext` (`KeyboardShortcutsModal`).
- **`frontend/views/editor`** — `editor-store`/`editor-selectors` used by `ExportModal`, `ImportTimelineModal`; `GapGenerationModal` (in editor folder) imports `SettingsPanel`.
- **`frontend/lib`** — `api-client` (`ApiClient` + `ApiRequestBodyOf`/`ApiSuccessOf` typed helpers), `video-generation-model-specs`, `asset-copy`, `file-url`, `timeline-import`, `srt`, `keyboard-shortcuts`, `logger`, `generation-errors`.
- **`frontend/hooks`** — `use-hf-auth`, `use-hf-model-access`, `use-model-profiles`, `use-official-adapters`.
- **`frontend/types`** — `project-model`, `project` (`TEXT_PRESETS`), `model-profile`.
- **`frontend/components/ui`** — `Button`, `Select`, `Tooltip`, `Progress` used throughout.
- **Electron bridge** — `showOpenFileDialog`, `showOpenDirectoryDialog`, `showSaveDialog`, `saveFile`, `extractVideoFrame`, `readLocalFile`, `getLogs`, `openLogFolder`, `openLtxApiKeyPage`, `openLtxBillingPage`, `openHuggingFaceRepo`, `getAppInfo`, `getAnalyticsState`, `getProjectAssetsPath`, `getResourcePath`, `onPythonSetupProgress`.
- **App shell events** — `open-settings` CustomEvent (with `detail.tab`) consumed by the root to open `SettingsModal` at a specific tab.
