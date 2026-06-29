# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LTX Desktop is an Electron app for AI video generation using LTX models. Three-layer architecture:

- **Frontend** (`frontend/`): React 18 + TypeScript + Tailwind CSS renderer
- **Electron** (`electron/`): Main process managing app lifecycle, IPC, Python backend process, ffmpeg export
- **Backend** (`backend/`): Python FastAPI server (port 8000) handling ML model orchestration and generation

## Session Workflow Rules

- **Read AGENTS.md** at the start of every turn before acting.
- **Orchestrator delegates, never codes.** Orchestrator/parent reads, plans, delegates, reviews, verifies, runs commands, and reverts accidental direct edits only. No direct implementation.
- **Subagents get narrow prompts** with exact file paths, symbols, and line ranges. No broad repo search unless explicitly assigned a locator phase.
- **No parallel self-coding while subagents run.** Do not continue broad implementation while locator/worker subagents are active.
- **Reviews target exact files/ranges/changes** — specific, not open-ended.
- **Reviewers are static review only.** Orchestrator owns validation execution.
- **Targeted tests per slice** until product path is ready. No full test suite runs prematurely.
- **Never re-download existing models.** First scan/status files under the user-set models folder before any download.
- **One models folder is source of truth** for downloads, profile scanning, status, Gemma/text projection, IC-LoRAs, VAE/upscaler/audio models. Missing model UI must show source link and exact required placement path.
- **Local profile Gemma/text encoder must suppress API-key prompting** for prompt enhancement when local GGUF Gemma is configured.
- **Canny/depth disabled by default.** Runs only when Union Control is explicitly enabled. Other LoRAs never get canny/depth preprocessing.
- **Union Control first, then LoRA.** If canny/depth + another adapter enabled: load/apply Union Control first, selected LoRA second. If disabled, LoRA runs without edge/depth conditioning.
- **Honest workflow gating.** Never fake unsupported workflow support; mark unavailable/gated features clearly.
- **No push without explicit confirmation.**

## Codemap-First Search Rules

- **Use codemap before search.** Any assistant or subagent that searches, explores, or changes code must read `codemap.md` first, then the relevant folder-level `codemap.md` files before using grep/glob/global search.
- **Prefer codemap-guided lookup over broad search.** Use the codemap to choose exact directories/files/symbols; broad repo-wide searches are a last resort and must be justified by the task.
- **Keep codemap current.** If work changes architecture, directory responsibilities, API/control flow, model-discovery semantics, or other codemap-covered behavior, update/regenerate codemap artifacts as part of the planned work.

## Full-Stack Contract Rules

- **No backend-only changes for user-visible/API behavior.** Backend work that affects frontend behavior, API schemas, generated types, model visibility, generation options, settings, validation, or error handling is prohibited unless the same plan includes the required frontend changes.
- **Plan backend and frontend together.** Any backend API/schema/DTO/endpoint change must define, before implementation:
  - endpoint path/method, if new or changed;
  - request/response field names;
  - exact shared variable/type names;
  - backend files/functions touched;
  - frontend files/hooks/components/types touched;
  - generated OpenAPI/type regeneration steps;
  - targeted validation commands.
- **Preserve names end-to-end.** Variable names, API field names, endpoint names, and type names must be decided in the plan and preserved consistently across backend, generated types, frontend state, hooks, and UI.
- **No subagent invention.** Subagents must not invent API names, variable names, endpoint names, schema shape, UI behavior, or validation policy. These must be predetermined by the orchestrator plan.

## Planning Requirements

- Before implementation, the orchestrator must produce a concrete plan containing:
  - backend files/functions to touch;
  - frontend files/functions/components/hooks to touch;
  - generated files, if any;
  - exact new/changed endpoint names;
  - exact new/changed request/response fields;
  - exact variable/type names to preserve end-to-end;
  - targeted validation commands only;
  - explicit subagent boundaries and prohibited actions.
- No implementation starts until this plan exists.
- If the work reveals a required scope expansion, stop and revise the plan before editing.

## Subagent Scope Rules

- Worker subagents receive narrow implementation instructions only.
- Prompts must include exact files, symbols/functions, expected variable names, allowed commands, and prohibited commands.
- Workers must not perform broad repo discovery unless explicitly assigned a locator-only phase.
- Workers must not run full test suites unless explicitly instructed.
- Default validation is targeted tests/checks only for touched behavior.
- **Exact validation commands only.** Workers may run only the exact validation commands listed in their prompt. They must not substitute broader commands, run whole files, run "affected tests", run an entire test module, or add extra validation commands because a wrapper fails or a flag is inconvenient.
- **Validation fallback requires orchestrator approval.** If an allowed validation command fails because of tooling/wrapper issues (for example, argument forwarding), the worker must stop and report the validation blocker. The orchestrator decides any alternate command; the worker must not improvise a broader test command.
- **Test scope must be smaller than implementation scope.** If test edits or test runtime begin to dominate the implementation, the worker must stop and return a revised plan. Tests prove the changed contract; they are not the main work.
- If tests outside the assigned scope fail, workers report the failure and stop; they must not spend time broadly fixing unrelated tests.
- If the needed implementation exceeds the assigned file/function scope, workers stop and return a revised plan instead of improvising.

## Git/Data-Loss Protection

- Subagents must never run `git stash`, `git reset`, `git checkout`, `git restore`, `git clean`, or destructive filesystem commands unless the user explicitly requests that exact command.
- Subagents must never discard, overwrite, or revert dirty files they did not create.
- Before any write task, dirty files must be identified and protected in the prompt.
- No commit, amend, push, broad staging, or git config changes without explicit user confirmation.

## Model Discovery Rules

- One model registry/source of truth must drive scanner, model-options, resolver, downloader metadata, frontend generated types, and frontend model-selection UI.
- Generation model options must not be derived only from downloadable checkpoint IDs when scanner-recognized local artifacts are selectable.
- Scanner-only artifacts must either be promoted to selectable IDs or the model-selection API must explicitly support scanner artifact IDs.
- Kijai and QuantStack distilled models are Fast-family selectable base models.
- Dev/full GGUF models are Full-family selectable base models.
- Missing model UI must show source link and exact required placement path for the same selectable ID that generation would use.

## Common Commands

| Command | Purpose |
|---|---|
| `pnpm dev` | Start dev server (Vite + Electron + Python backend) |
| `pnpm dev:debug` | Dev with Electron inspector + Python debugpy |
| `pnpm typecheck` | Run TypeScript (`tsc --noEmit`) and Python (`pyright`) type checks |
| `pnpm typecheck:ts` | TypeScript only |
| `pnpm typecheck:py` | Python pyright only |
| `pnpm backend:test` | Run Python pytest tests |
| `pnpm build:frontend` | Vite frontend build only |
| `pnpm build` | Full platform build (auto-detects platform) |
| `pnpm setup:dev` | One-time dev environment setup (auto-detects platform) |

Run a single backend test file via pnpm: `pnpm backend:test -- tests/test_ic_lora.py`

## CI Checks

PRs must pass: `pnpm typecheck` + `pnpm backend:test` + frontend Vite build.

## Frontend Architecture

- **Path alias**: `@/*` maps to `frontend/*`
- **State management**: React contexts only (`ProjectContext`, `AppSettingsContext`, `KeyboardShortcutsContext`) — no Redux/Zustand
- **Routing**: View-based via `ProjectContext` with views: `home`, `project`
- **IPC bridge**: All Electron communication through `window.electronAPI` (defined in `electron/preload.ts`)
- **Backend calls**: Always use `backendFetch` from `frontend/lib/backend.ts` for app backend HTTP requests (it attaches auth/session details). Do not call `fetch` directly for backend endpoints.
- **Styling**: Tailwind with custom semantic color tokens via CSS variables; utilities from `class-variance-authority` + `clsx` + `tailwind-merge`
- **No frontend tests** currently exist

## Backend Architecture

Request flow: `_routes/* (thin) → AppHandler → handlers/* (logic) → services/* (side effects) + state/* (mutations)`

Key patterns:
- **Routes** (`_routes/`): Thin plumbing only — parse input, call handler, return typed output. No business logic.
- **AppHandler** (`app_handler.py`): Single composition root owning all sub-handlers, state, and lock
- **State** (`state/`): Centralized `AppState` using discriminated union types for state machines (e.g., `GenerationState = GenerationRunning | GenerationComplete | GenerationError | GenerationCancelled`)
- **Services** (`services/`): Protocol interfaces with real implementations and fake test implementations. The test boundary for heavy side effects (GPU, network).
- **Concurrency**: Thread pool with shared `RLock`. Pattern: lock→read/validate→unlock→heavy work→lock→write. Never hold lock during heavy compute/IO.
- **Exception handling**: Boundary-owned traceback policy. Handlers raise `HTTPError` with `from exc` chaining; `app_factory.py` owns logging. Don't `logger.exception()` then rethrow.
- **Naming**: `*Payload` for DTOs/TypedDicts, `*Like` for structural wrappers, `Fake*` for test implementations

### Backend Testing

- Integration-first using Starlette `TestClient` against real FastAPI app
- **No mocks**: `test_no_mock_usage.py` enforces no `unittest.mock`. Swap services via `ServiceBundle` fakes only.
- Fakes live in `tests/fakes/`; `conftest.py` wires fresh `AppHandler` per test
- Pyright strict mode is also enforced as a test (`test_pyright.py`)

### Adding a Backend Feature

1. Define request/response models in `api_types.py`
2. Add endpoint in `_routes/<domain>.py` delegating to handler
3. Implement logic in `handlers/<domain>_handler.py` with lock-aware state transitions
4. If new heavy side effect needed, add service in `services/` with Protocol + real + fake implementations
5. Add integration test in `tests/` using fake services

## TypeScript Config

- Strict mode with `noUnusedLocals`, `noUnusedParameters`
- Frontend: ES2020 target, React JSX
- Electron main process: ESNext, compiled to `dist-electron/`
- Preload script must be CommonJS

## Python Config

- Python 3.13+ (per `.python-version`), managed with `uv`
- Pyright strict mode (`backend/pyrightconfig.json`)
- Dependencies in `backend/pyproject.toml`

## Key File Locations

- Backend architecture doc: `backend/architecture.md`
- Default app settings schema: `settings.json`
- Electron builder config: `electron-builder.yml`
- Video editor (largest frontend file): `frontend/views/VideoEditor.tsx`
- Project types: `frontend/types/project.ts`

## Repository Map

A full hierarchical codemap is available at `codemap.md` in the project root, with
per-folder detail in `<folder>/codemap.md` files across `backend/`, `frontend/`,
`electron/`, and `shared/`.

Before working on any task, read `codemap.md` to understand:
- Project architecture, system entry points, and core architectural conventions
- Directory responsibilities and design patterns
- End-to-end data/control flow (renderer → backend routes → AppHandler → handlers → services + state)
- Integration points and the encode-output chokepoint relevant to EXR/MOV work

For deep work on a specific folder, also read that folder's `codemap.md`.
Change detection state lives in `.slim/codemap.json` (regenerate via the codemap skill).
