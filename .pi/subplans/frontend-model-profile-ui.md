# Planner Report

## Status
split-required

## Why Split / Parallelize
Slice C touches shared frontend API types, launch gating, settings UI, profile forms, adapter UI, and Electron dialogs. Keep it sequential around shared files (`frontend/lib/api-client.ts`, `frontend/App.tsx`, `frontend/components/SettingsModal.tsx`) to avoid merge conflicts; component-only work can be prepared after API types land.

## Interference Check
- parallel safe: partial
- shared files or generated outputs: `frontend/lib/api-client.ts`, `frontend/generated/backend-openapi.*`, `frontend/App.tsx`, `frontend/components/SettingsModal.tsx`
- shared validation state: `pnpm typecheck:ts`, `pnpm build:frontend`; OpenAPI generation if backend slices not landed
- worktree isolation required: no for sequential; yes if component tasks run in parallel
- rationale: existing Electron file/folder dialogs are sufficient; no new IPC unless a picker requirement appears that `showOpenFileDialog` / `showOpenDirectoryDialog` cannot satisfy.

## Proposed Task Sequence Or Parallel Batch
1. Task name: API client + UI model types/hooks
   - purpose: expose backend model-profile and adapter registry APIs to React, using generated OpenAPI types.
   - allowed files:
     - `frontend/lib/api-client.ts`
     - `frontend/types/model-profile.ts` (new)
     - `frontend/hooks/use-model-profiles.ts` (new)
     - `frontend/hooks/use-official-adapters.ts` (new)
     - `frontend/generated/backend-openapi.json` (generated only, if backend slice already changed)
     - `frontend/generated/backend-openapi.ts` (generated only, if backend slice already changed)
   - validation: `pnpm openapi:check` if backend slice present; `pnpm typecheck:ts`
   - can run in parallel with: none
2. Task name: Component file pickers + profile wizard
   - purpose: build reusable picker rows and profile create/edit/validate/activate UI.
   - allowed files:
     - `frontend/components/ModelComponentPicker.tsx` (new)
     - `frontend/components/ModelProfileWizard.tsx` (new)
     - `frontend/types/model-profile.ts`
     - `frontend/hooks/use-model-profiles.ts`
   - validation: `pnpm typecheck:ts`, `pnpm build:frontend`
   - can run in parallel with: adapter checklist after task 1, with worktree isolation
3. Task name: Official adapters checklist
   - purpose: show official adapter status and browse/download actions.
   - allowed files:
     - `frontend/components/AdapterChecklist.tsx` (new)
     - `frontend/hooks/use-official-adapters.ts`
     - `frontend/components/ModelComponentPicker.tsx`
     - `frontend/types/model-profile.ts`
   - validation: `pnpm typecheck:ts`, `pnpm build:frontend`
   - can run in parallel with: profile wizard after task 1, with worktree isolation
4. Task name: First-run choose model source
   - purpose: replace download-only first-run path with source choices: existing local profile, download official bundle, API-only.
   - allowed files:
     - `frontend/App.tsx`
     - `frontend/components/FirstRunSetup.tsx`
     - `frontend/components/FirstRunSetup.css`
     - `frontend/components/ModelProfileWizard.tsx`
     - `frontend/hooks/use-model-profiles.ts`
   - validation: `pnpm typecheck:ts`, `pnpm build:frontend`; manual first-run smoke
   - can run in parallel with: none
5. Task name: Settings Models tab
   - purpose: add persistent Models tab for profile management and adapter checklist.
   - allowed files:
     - `frontend/components/SettingsModal.tsx`
     - `frontend/components/ModelProfileWizard.tsx`
     - `frontend/components/AdapterChecklist.tsx`
     - `frontend/hooks/use-model-profiles.ts`
     - `frontend/hooks/use-official-adapters.ts`
   - validation: `pnpm typecheck:ts`, `pnpm build:frontend`; manual settings smoke
   - can run in parallel with: none

## Task Packets

### Task Packet 1 — API client + UI model types/hooks

## User Goal
Plan and prepare frontend/Electron UX for first-run model source choice, model profiles, adapter checklist, and component file pickers in `/tmp/clones/LTX-Desktop`.

## Mode
general-coding

## Relevant Locations
- file: `frontend/lib/api-client.ts`
  symbol: `ApiClient`
  approximate lines: 300-379
  stable anchor: `static getLtxRecommendation = makeEndpointClient('/api/models/ltx-recommendation', 'get')`
  reason: add model-profile and adapter endpoint clients after generated OpenAPI includes those paths.
  confidence: high
- file: `frontend/generated/backend-openapi.ts`
  symbol: `paths`, `components`
  approximate lines: 1-450
  stable anchor: `export interface paths`
  reason: source of typed endpoint schemas; current checkout has no model-profile/adapter paths.
  confidence: high
- file: `frontend/components/FirstRunSetup.tsx`
  symbol: `LaunchGate`, `Step`, `startInstallation`, `handleNext`, `isNextDisabled`
  approximate lines: 10-360, 520-860
  stable anchor: `type Step = 'license' | 'location' | 'installing' | 'complete'`
  reason: existing download-only first-run flow to extend later.
  confidence: high
- file: `frontend/App.tsx`
  symbol: `areRequiredModelsDownloaded`, `handleMissingModelsComplete`, LaunchGate render branches
  approximate lines: 186-266, 450-490
  stable anchor: `return ltxResult.data.status !== 'download' && imgGenResult.data.cp_to_download === null`
  reason: current gate blocks startup on missing official downloads; later must accept valid active profile/API-only.
  confidence: high
- file: `frontend/components/SettingsModal.tsx`
  symbol: `TabId`, `tabs`, settings content
  approximate lines: 1-260, 270-600
  stable anchor: `type TabId = 'general' | 'apiKeys' | 'promptEnhancer' | 'about'`
  reason: later add `models` tab and mount profile/adapter components.
  confidence: high
- file: `shared/electron-api-schema.ts`
  symbol: `electronAPISchemas.showOpenFileDialog`, `showOpenDirectoryDialog`, `checkFilesExist`
  approximate lines: 204-242, 313-328
  stable anchor: `showOpenFileDialog: {`
  reason: existing typed IPC supports component file/folder pickers.
  confidence: high
- file: `electron/ipc/file-handlers.ts`
  symbol: `showOpenDirectoryDialog`, `showOpenFileDialog`, `checkFilesExist`
  approximate lines: 265-394
  stable anchor: `handle('showOpenFileDialog', async ({ title, filters, properties }) => {`
  reason: reuse current Electron dialogs; no Electron source edit expected.
  confidence: high
- file: `research/06-revised-ltx-desktop-implementation-roadmap.md`
  symbol: `Milestone 1`, `Milestone 3`
  approximate lines: 1-70
  stable anchor: `Change first-run flow from “download required” to “choose model source”.`
  reason: confirms UX scope and sequence.
  confidence: high
- file: `research/05-ltx-desktop-model-profiles-and-gguf-kijai-plan.md`
  symbol: `First-Run Download Gate`, `Suggested schema`, `File dialogs`
  approximate lines: 121-135, 253-304, 413-416
  stable anchor: `Electron already has general file/directory dialogs`
  reason: confirms schema shape and dialog reuse.
  confidence: high

## Allowed Edit Files
- `frontend/lib/api-client.ts`
- `frontend/types/model-profile.ts`
- `frontend/hooks/use-model-profiles.ts`
- `frontend/hooks/use-official-adapters.ts`
- `frontend/generated/backend-openapi.json` (only via `pnpm openapi:generate`)
- `frontend/generated/backend-openapi.ts` (only via `pnpm openapi:generate`)

## Read-Only Context Files
- `frontend/App.tsx`
- `frontend/components/FirstRunSetup.tsx`
- `frontend/components/SettingsModal.tsx`
- `shared/electron-api-schema.ts`
- `electron/ipc/file-handlers.ts`
- `electron/preload.ts`
- `package.json`
- `research/05-ltx-desktop-model-profiles-and-gguf-kijai-plan.md`
- `research/06-revised-ltx-desktop-implementation-roadmap.md`

## Required Change
Add typed React-facing API surface only after backend OpenAPI contains the model profile and adapter paths.

Minimum shape:
- Add `frontend/types/model-profile.ts` with aliases/imports from `components['schemas']` for `ModelProfile`, `ModelComponentPaths`, profile validation result, official adapter entry/status, and request bodies. Keep hand-written fallback types out unless backend OpenAPI is missing only harmless display fields; prefer generated types.
- Add `ApiClient` methods for model profile endpoints from research: `GET /api/model-profiles`, `POST /api/model-profiles`, `PATCH /api/model-profiles/{id}`, `DELETE /api/model-profiles/{id}`, `POST /api/model-profiles/{id}/validate`, `POST /api/model-profiles/{id}/activate`. Use custom methods with encoded path segments for `{id}` because current `makeEndpointClient` only handles fixed paths.
- Add `ApiClient` methods for official adapter registry endpoints exactly as generated by the backend adapter slice. Do not invent names if absent.
- Add `use-model-profiles.ts`: load/list, create/update, validate, activate, delete, expose `loading`, `error`, `refresh`; no global state library.
- Add `use-official-adapters.ts`: load statuses, set local path/browse callback integration point, download action if backend endpoint exists, expose `loading`, `error`, `refresh`.
- Do not add new dependencies.

## Non-Goals
- No backend endpoint implementation.
- No GGUF loader or pipeline changes.
- No new Electron IPC unless existing dialogs cannot meet picker needs.
- No broad settings refactor or state-management library.
- No manual edits to generated OpenAPI files.

## Validation
Commands:
- `pnpm openapi:check` (only if backend slice is present and generated files are expected to match)
- `pnpm typecheck:ts`
- `pnpm build:frontend`

Expected result:
- TypeScript passes with strict `noUnusedLocals`/`noUnusedParameters`.
- Frontend build succeeds.
- If `pnpm openapi:check` runs, no diff in generated OpenAPI files.

## Stop Conditions
Stop and report if:
- target symbol is missing
- required fix exceeds allowed files
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet
- generated OpenAPI lacks model-profile paths
- generated OpenAPI lacks official adapter paths needed for checklist actions
- endpoint names or schemas differ from research and no backend API contract is supplied

## Planner Self-Check
- locator evidence sufficient: yes + frontend/Electron anchors verified in repo; backend adapter endpoint names not verified, so stop condition added.
- allowed edit files minimal and explicit: yes + first task limited to API/types/hooks/generated outputs.
- read-only context minimal: yes + only launch/settings/Electron dialog/API/research files.
- anchors/lines included: yes + relevant locations include symbols, anchors, line ranges.
- validation concrete: yes + package scripts verified in `package.json`.
- parallelization decision explicit and safe: yes + sequential around shared files; component-only work may parallel after API task with isolation.
- non-goals and stop conditions sufficient: yes + blocks backend/GGUF/new IPC creep.
- reviewer findings addressed, if revision: not applicable + no reviewer feedback supplied.

## Required Return Contract
Return only a task-focused summary. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks.

## Downstream UX Implementation Notes
- Component pickers should call existing `window.electronAPI.showOpenFileDialog` with filters: safetensors (`safetensors`), GGUF (`gguf`), any model file (`safetensors`, `gguf`, `bin`, `pt`, `ckpt`), and existing `showOpenDirectoryDialog` for Gemma/HF folders.
- First-run source choices:
  1. `Use existing local model components` → mount `ModelProfileWizard`, save + validate + activate, then complete setup.
  2. `Download official Lightricks bundle` → keep current `location`/download flow.
  3. `API-only mode` → require/save LTX API key or open existing API gateway, set video/text preferences, then complete setup without local downloads.
- Settings `Models` tab should include active profile card, create/edit profile wizard, validate/activate/delete actions, and `AdapterChecklist` below profiles.
- `App.tsx` gate should be renamed from `areRequiredModelsDownloaded` to model-readiness semantics and treat valid active profile or API-only config as startup-satisfied; keep current transient-error fail-open behavior.
- Electron edits intentionally skipped: `electron/preload.ts` auto-exposes schema keys, and `electron/ipc/file-handlers.ts` already approves selected file/folder paths.
