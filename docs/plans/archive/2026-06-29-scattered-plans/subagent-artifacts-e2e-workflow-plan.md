# Planner Report

## Status
split-required

## Why Split / Parallelize
Full local profile E2E touches profile persistence, adapter path resolution, frontend profile entry, IC-LoRA request UI, and Kijai runtime loading. These are independently understandable but not all independent: backend IC-LoRA schema/resolution must land before frontend sends `adapter_id`; Kijai VAE remap is runtime-critical and independent. Use ordered slices to keep validation localized and avoid full-suite churn.

## Interference Check
- parallel safe: partial
- shared files or generated outputs: `backend/api_types.py` and `frontend/generated/backend-openapi.*` are shared by backend IC-LoRA schema and frontend selector; avoid concurrent edits there.
- shared validation state: backend pytest files and app-local `model_profiles.json` can be shared if workers use same app data dir; isolate temp dirs for tests/manual checks.
- worktree isolation required: recommended if running Kijai fix in parallel with IC-LoRA schema/frontend work; otherwise sequential is simpler.
- rationale: Kijai slice edits pipeline wrappers only; profile wizard edits one frontend file; IC-LoRA backend/frontend slices must be sequential because request shape and generated types depend on backend schema.

## Proposed Task Sequence Or Parallel Batch
1. Task name: Profile CRUD persistence regression guard
   - purpose: lock in known worker fix: blank profile IDs become UUIDs and persisted blank IDs are repaired, so create→validate→activate→reload does not fall back to `NO_DOWNLOADED_LTX_MODEL`.
   - allowed files: `backend/tests/test_model_profiles.py`
   - validation: `pnpm backend:test -- tests/test_model_profiles.py`
   - can run in parallel with: Kijai VAE remap, frontend wizard prefill
2. Task name: Backend IC-LoRA adapter selection from active profile
   - purpose: add minimal `adapter_id` request field and resolve selected official LoRA from active profile `components.official_adapters`, with legacy typed-field and official-spec fallback.
   - allowed files: `backend/api_types.py`, `backend/handlers/ic_lora_handler.py`, `backend/tests/test_ic_lora.py`
   - validation: `pnpm backend:test -- tests/test_ic_lora.py`
   - can run in parallel with: Kijai VAE remap, frontend wizard prefill; not frontend selector/OpenAPI generation
3. Task name: Frontend profile wizard official adapter dict prefill
   - purpose: expose/prefill all official adapter paths through existing `components.official_adapters` dict; do not add seven typed fields.
   - allowed files: `frontend/components/ModelProfileWizard.tsx`
   - validation: `pnpm typecheck:ts`; supervised UI check creates profile whose payload has `official_adapters` entries for all available official LoRA files.
   - can run in parallel with: profile CRUD guard, Kijai VAE remap, backend IC-LoRA selection
4. Task name: Frontend IC-LoRA adapter selector and generated request type
   - purpose: let user select any official adapter and send `adapter_id` on generate request.
   - allowed files: `frontend/components/ICLoraPanel.tsx`, `frontend/generated/backend-openapi.ts`, `frontend/generated/backend-openapi.json`, `frontend/lib/api-client.ts`
   - validation: `pnpm api:generate` if script exists, then `pnpm typecheck:ts`; supervised UI check confirms selected adapter id reaches generate request.
   - can run in parallel with: Kijai VAE remap only; depends on backend IC-LoRA schema landing first.
5. Task name: Kijai split VAE key remap
   - purpose: apply Kijai video VAE encoder/decoder key filters to split safetensors profiles, not only GGUF profiles.
   - allowed files: `backend/services/patches/gguf_loader_fix.py`, `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`, `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`, `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`, `backend/services/retake_pipeline/ltx_retake_pipeline.py`, `backend/tests/test_gguf_loader.py`, `backend/tests/test_ltx_split_safetensors.py`
   - validation: `pnpm backend:test -- tests/test_gguf_loader.py tests/test_ltx_split_safetensors.py`
   - can run in parallel with: profile CRUD guard, backend IC-LoRA selection, frontend wizard prefill; avoid overlapping final manual E2E.
6. Task name: Supervised local E2E smoke
   - purpose: verify product live path only after slices 1–5: create, validate, save/reload, activate, run GGUF, run Kijai split, run IC-LoRA with selected official adapter.
   - allowed files: none
   - validation: manual app run only; no full test suites.
   - can run in parallel with: none; depends on all implementation slices.

## Task Packets

### Task Packet 1 — Profile CRUD persistence regression guard

## User Goal
Protect the local profile workflow: create profile with blank frontend id, validate/activate it, save/reload it on later runs, and avoid `NO_DOWNLOADED_LTX_MODEL` caused by blank or unrepaired IDs.

## Mode
general-coding

## Relevant Locations
- file: `backend/tests/test_model_profiles.py`
  symbol: existing model profile tests
  approximate lines: existing file; locator reports tests use explicit non-empty IDs only
  stable anchor: tests currently cover profile CRUD with explicit ids but not empty-id create or persisted blank-id repair
  reason: add localized regression coverage for known worker fix
  confidence: high
- file: `backend/handlers/model_profiles_handler.py`
  symbol: `create_profile`, `load_profiles`, `save_profiles`
  approximate lines: `create_profile` around 121-130 after known worker fix; load/save around 65-84 in locator
  stable anchor: backend now generates UUID for blank id and repairs persisted blank ids
  reason: source behavior under test; do not edit unless test reveals known fix missing
  confidence: high

## Allowed Edit Files
- `backend/tests/test_model_profiles.py`

## Read-Only Context Files
- `subagent-artifacts/profile-crud-locator.md`
- `backend/handlers/model_profiles_handler.py`
- `backend/handlers/pipelines_handler.py`

## Required Change
Add the smallest localized tests that prove:
1. `create_profile()` with `id=""` returns and stores a non-empty id.
2. persisted profiles with blank ids are repaired/deduplicated on load, using a temp app-data path only.
3. validate/activate by the repaired/generated id succeeds for a valid test profile.

If the current source does not contain the known worker fix, stop and report instead of re-implementing it in this slice.

## Non-Goals
- Do not change frontend wizard code.
- Do not change profile schema.
- Do not run full backend suite.
- Do not touch real user `model_profiles.json`.

## Validation
Commands:
- `pnpm backend:test -- tests/test_model_profiles.py`

Expected result:
Targeted profile tests pass, including new blank-id/reload coverage.

## Stop Conditions
Stop and report if:
- `backend/tests/test_model_profiles.py` does not exist or its fixtures require broad rewiring.
- known worker fix is missing from `model_profiles_handler.py`.
- test requires real app-data directory or real model files.
- validation cannot run.
- required fix exceeds allowed files.

## Required Return Contract
Return status, files inspected/changed, exact targeted test command/result, and any source-fix blocker. No full logs.

---

### Task Packet 2 — Backend IC-LoRA adapter selection from active profile

## User Goal
Run IC-LoRA inference with any official LoRA safetensors available under the active local profile, while keeping Kijai and GGUF paths working.

## Mode
general-coding

## Relevant Locations
- file: `backend/api_types.py`
  symbol: `IcLoraGenerateRequest`
  approximate lines: 569-597
  stable anchor: request currently has `conditioning_type` but no adapter choice field
  reason: add minimal `adapter_id: str | None = None` so frontend can select any official adapter
  confidence: high
- file: `backend/handlers/ic_lora_handler.py`
  symbol: `_require_ic_lora_model_paths`
  approximate lines: 80-96
  stable anchor: currently always uses `get_ltx_model_spec(model_id).ic_loras_spec.canny_cp/depth_cp`
  reason: route selected adapter through active profile before official fallback
  confidence: high
- file: `backend/runtime_config/model_download_specs.py`
  symbol: `OFFICIAL_LTX23_ADAPTERS`, `resolve_adapter_path`
  approximate lines: adapters 105-240; resolver via models handler around 114-132 per locator
  stable anchor: official registry contains all 19 adapter ids and filenames
  reason: validate adapter ids and fallback to installed official files without new dependency
  confidence: high
- file: `backend/tests/test_ic_lora.py`
  symbol: IC-LoRA generate tests
  approximate lines: existing file
  stable anchor: existing canny/depth generate coverage lacks profile-aware adapter routing
  reason: add localized backend coverage
  confidence: high

## Allowed Edit Files
- `backend/api_types.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/tests/test_ic_lora.py`

## Read-Only Context Files
- `subagent-artifacts/e2e-profile-lora-locator.md`
- `backend/runtime_config/model_download_specs.py`
- `backend/handlers/model_profiles_handler.py`
- `backend/services/ltx_components.py`

## Required Change
Implement adapter resolution strategy:
1. Add `adapter_id: str | None = None` to `IcLoraGenerateRequest` only. Do not change extract request.
2. Validate non-empty `adapter_id` against `OFFICIAL_LTX23_ADAPTERS`; unknown id returns existing HTTP error style with 400/409, not a silent fallback.
3. In `_require_ic_lora_model_paths(conditioning_type, adapter_id=None)`, preserve existing behavior when `adapter_id` is absent.
4. When `adapter_id` is present, resolve selected LoRA path in this order:
   - active profile `components.official_adapters[adapter_id]` if present and non-empty;
   - legacy typed component field for existing typed adapter ids only (`union_control`, `motion_track_control`, `ingredients`, `hdr`, `hdr_scene_embeddings`, `lipdub`, `in_outpainting`) if mapper exists;
   - official installed adapter fallback using existing registry/model-dir resolution pattern.
5. Keep `conditioning_type` responsible for canny/depth conditioning/control path behavior; only swap selected effect LoRA path.
6. Do not add the seven missing typed adapter fields. Use `official_adapters` as source of truth for all official adapters.
7. Add localized tests for active-profile `official_adapters` selected path, unknown `adapter_id`, and no-`adapter_id` legacy behavior.

Use direct helper functions in `ic_lora_handler.py`; no new resolver class.

## Non-Goals
- Do not add seven typed `ic_lora_*` fields.
- Do not regenerate OpenAPI in this backend slice.
- Do not change pipeline constructors.
- Do not change adapter download/status endpoints.
- Do not change `conditioning_type` enum.

## Validation
Commands:
- `pnpm backend:test -- tests/test_ic_lora.py`

Expected result:
Existing IC-LoRA tests still pass; new tests prove selected official adapter path can come from active profile `official_adapters`.

## Stop Conditions
Stop and report if:
- `_require_ic_lora_model_paths` signature/callers differ from locator evidence.
- handler cannot access active profile state without broader AppHandler/service rewiring.
- selected adapter requires changing pipeline constructor API.
- request schema change cascades beyond allowed files.
- validation cannot run.

## Required Return Contract
Return status, files inspected/changed, exact request field added, resolution order implemented, targeted test evidence, and blockers/risks. No broad logs.

---

### Task Packet 3 — Frontend profile wizard official adapter dict prefill

## User Goal
Create local profiles from frontend with all official LoRA safetensors available, without adding seven typed fields or requiring OpenAPI regeneration for profile components.

## Mode
general-coding

## Relevant Locations
- file: `frontend/components/ModelProfileWizard.tsx`
  symbol: `COMPONENT_FIELDS`
  approximate lines: 20-67
  stable anchor: current wizard lists 18 component fields and only 7 typed IC-LoRA fields
  reason: add UI/prefill path for `official_adapters` dict instead of typed fields
  confidence: high
- file: `frontend/components/ModelProfileWizard.tsx`
  symbol: `visibleFields`
  approximate lines: 110-125
  stable anchor: source-specific fields shown for Kijai mode
  reason: show official adapter dict section for local/Kijai profile creation
  confidence: high
- file: `frontend/components/ModelProfileWizard.tsx`
  symbol: payload construction
  approximate lines: 258-267
  stable anchor: builds `ModelComponentPaths` from form state
  reason: include `official_adapters` in `components` payload
  confidence: high

## Allowed Edit Files
- `frontend/components/ModelProfileWizard.tsx`

## Read-Only Context Files
- `subagent-artifacts/e2e-profile-lora-locator.md`
- `frontend/types/model-profile.ts`

## Required Change
Implement frontend prefill strategy:
1. Keep existing typed component fields; do not add seven new typed fields.
2. Add a small local constant for official adapter ids/display labels/filenames from locator evidence, covering all 19 official adapters.
3. Add an Official Adapters section backed by `components.official_adapters: Record<string, string>`.
4. Prefill adapter paths using the existing wizard path/default pattern if present; otherwise use editable paths under `/mnt/ssd1/LTX_models/adapters/<filename>` because user confirmed files exist there.
5. Allow per-adapter path edits and omit blank adapter entries from payload.
6. Capability detection must count non-empty `official_adapters` values as `ic_lora`; keep existing `ic_lora_*` detection.
7. Keep generated OpenAPI types unchanged; use the existing `official_adapters` dict type.

## Non-Goals
- Do not add backend endpoints for auto-detect.
- Do not add seven typed fields to `ModelComponentPaths`.
- Do not add dependencies or file picker libraries.
- Do not change settings modal/list behavior.
- Do not change IC-LoRA generation UI in this slice.

## Validation
Commands:
- `pnpm typecheck:ts`

Manual check:
- Open profile wizard, choose local/Kijai profile flow, confirm official adapter rows prefill to `/mnt/ssd1/LTX_models/adapters/*.safetensors`, create payload contains `components.official_adapters` with selected non-empty entries.

Expected result:
TypeScript passes; profile creation payload can include all official adapters through dict.

## Stop Conditions
Stop and report if:
- `official_adapters` is missing from generated `ModelComponentPaths` type.
- wizard state shape cannot carry nested dict without touching additional files.
- UI requires product/design decisions beyond simple editable rows.
- validation cannot run.
- required fix exceeds allowed file.

## Required Return Contract
Return status, files inspected/changed, adapter ids covered, typecheck/manual evidence, and any UX limitation. No broad logs.

---

### Task Packet 4 — Frontend IC-LoRA adapter selector and generated request type

## User Goal
Select which official LoRA adapter to use for IC-LoRA generation, so any official safetensors file in the active profile can be used.

## Mode
general-coding

## Relevant Locations
- file: `frontend/components/ICLoraPanel.tsx`
  symbol: `CONDITIONING_TYPES`
  approximate lines: 37-40
  stable anchor: panel only exposes canny/depth; no adapter selector
  reason: add official adapter selector separate from conditioning selector
  confidence: high
- file: `frontend/components/ICLoraPanel.tsx`
  symbol: generate request construction
  approximate lines: locator reports full download/generate flow around 102-170
  stable anchor: request currently sends conditioning fields only
  reason: include selected `adapter_id`
  confidence: high
- file: `frontend/views/GenSpace.tsx`
  symbol: IC-LoRA panel integration
  approximate lines: 1004-1079
  stable anchor: panel reset/initiation lives here
  reason: read-only unless selector state must be lifted; prefer keeping state inside panel
  confidence: high
- file: `frontend/generated/backend-openapi.ts`
  symbol: `IcLoraGenerateRequest`
  approximate lines: generated schema area near backend request types
  stable anchor: generated types must include `adapter_id` after backend schema change
  reason: avoid TS casts if OpenAPI regen script exists
  confidence: medium

## Allowed Edit Files
- `frontend/components/ICLoraPanel.tsx`
- `frontend/generated/backend-openapi.ts`
- `frontend/generated/backend-openapi.json`
- `frontend/lib/api-client.ts`

## Read-Only Context Files
- `subagent-artifacts/e2e-profile-lora-locator.md`
- `backend/api_types.py`
- `frontend/views/GenSpace.tsx`

## Required Change
Implement selector strategy:
1. Add official adapter selector in `ICLoraPanel.tsx`, separate from `conditioningType` (`canny`/`depth`).
2. Options must use official adapter ids from backend registry/locator, including the seven previously untyped ids: `water_simulation`, `decompression`, `deblur`, `colorization`, `day_to_night`, `instant_shave`, `cross_eyed`.
3. Default selection should preserve current behavior: no `adapter_id` sent until user chooses one, or choose the closest current default only if existing UI already has a recommendation flow. Avoid product-specific magic.
4. Send `adapter_id` only on generate request. Do not send it on extract-conditioning.
5. Regenerate OpenAPI if `pnpm api:generate` exists. If it does not exist, stop and report exact missing script rather than hand-editing generated files.
6. Edit `frontend/lib/api-client.ts` only if generated type/API wrapper requires it.

## Non-Goals
- Do not change backend resolution in this slice.
- Do not change profile wizard.
- Do not change conditioning algorithms or introduce new conditioning types.
- Do not add dependencies.
- Do not run full frontend build.

## Validation
Commands:
- `pnpm api:generate` (only if script exists)
- `pnpm typecheck:ts`

Manual check:
- In IC-LoRA panel, select `deblur` or another official adapter, generate request includes `adapter_id` and extract request does not.

Expected result:
Generated request type includes optional `adapter_id`; TypeScript passes; UI can select all official adapters.

## Stop Conditions
Stop and report if:
- backend `IcLoraGenerateRequest.adapter_id` has not landed.
- OpenAPI generation script is missing or fails for unrelated reasons.
- selector state must be lifted into `GenSpace.tsx`; stop unless orchestrator expands allowed edit files.
- API wrapper rejects the new field in a way that requires broader client generation changes.
- validation cannot run.

## Required Return Contract
Return status, files inspected/changed, generated command evidence, typecheck evidence, selector default behavior, and blockers. No broad logs.

---

### Task Packet 5 — Kijai split VAE key remap

## User Goal
Run Kijai split safetensors profiles end-to-end, not just GGUF profiles.

## Mode
general-coding

## Relevant Locations
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER`, `KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER`
  approximate lines: 399-408
  stable anchor: filters strip Kijai `encoder.`/`decoder.` prefixes
  reason: reuse existing filters for split safetensors
  confidence: high
- file: `backend/services/patches/gguf_loader_fix.py`
  symbol: `install_gguf_component_paths`
  approximate lines: 547-603
  stable anchor: filters are currently applied only inside GGUF component path installer
  reason: factor/reuse VAE path replacement for non-GGUF split profiles
  confidence: high
- file: `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
  symbol: pipeline `__init__` GGUF block
  approximate lines: 25-58
  stable anchor: only calls component path installer when `transformer_format == "gguf"`
  reason: add split-safetensors VAE remap call
  confidence: high
- file: `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
  symbol: pipeline `__init__` GGUF block
  approximate lines: 24-47
  stable anchor: same GGUF-only installer pattern
  reason: keep Kijai split path consistent across pipelines
  confidence: high
- file: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
  symbol: pipeline `__init__` GGUF block
  approximate lines: 25-49
  stable anchor: same GGUF-only installer pattern
  reason: IC-LoRA must work with Kijai split active profile
  confidence: high
- file: `backend/services/retake_pipeline/ltx_retake_pipeline.py`
  symbol: pipeline `__init__` GGUF block
  approximate lines: 39-76
  stable anchor: retake passes `self` to installer; compatible shape per locator
  reason: retake should receive same split VAE remap
  confidence: high
- file: `backend/tests/test_gguf_loader.py`
  symbol: Kijai VAE filter tests
  approximate lines: around 428-436
  stable anchor: filters already unit-tested only in isolation
  reason: add installer-level coverage
  confidence: high
- file: `backend/tests/test_ltx_split_safetensors.py`
  symbol: split safetensors integration tests
  approximate lines: around 112
  stable anchor: tests handler passes tuple but not VAE remap
  reason: add localized split pipeline wiring check if existing fakes support it
  confidence: high

## Allowed Edit Files
- `backend/services/patches/gguf_loader_fix.py`
- `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
- `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/services/retake_pipeline/ltx_retake_pipeline.py`
- `backend/tests/test_gguf_loader.py`
- `backend/tests/test_ltx_split_safetensors.py`

## Read-Only Context Files
- `subagent-artifacts/e2e-kijai-gguf-locator.md`
- `backend/services/ltx_components.py`

## Required Change
Implement Kijai VAE remap scope:
1. Reuse existing `KIJAI_VIDEO_VAE_ENCODER_KEYS_FILTER` and `KIJAI_VIDEO_VAE_DECODER_KEYS_FILTER`; do not create a second mapping table.
2. Add the smallest helper in `gguf_loader_fix.py` that applies video VAE builder path replacement and those filters for split safetensors component paths. Mark any deliberate narrowness with a `ponytail:` comment if needed.
3. Call the helper in fast, a2v, ic_lora, and retake wrappers when `components` exists, `components.transformer_format == "safetensors"`, and split component paths are present (`checkpoint_paths_for_filtered_builders` length > 1 or equivalent explicit signal already in `ResolvedLtxComponents`).
4. Do not call GGUF loader/dequant patches for non-GGUF profiles.
5. Do not change `ResolvedLtxComponents` unless wrappers cannot identify split profiles from existing fields; if required, stop and request approval for schema/dataclass change.
6. Add localized tests proving split safetensors path applies VAE filters and GGUF behavior remains unchanged.

## Non-Goals
- Do not alter GGUF quantization/dequant logic.
- Do not change text encoder `gemma_root` behavior unless test proves it blocks this slice.
- Do not add real GPU/model load tests.
- Do not change profile schema.
- Do not run full backend suite.

## Validation
Commands:
- `pnpm backend:test -- tests/test_gguf_loader.py tests/test_ltx_split_safetensors.py`

Expected result:
Targeted tests pass; new tests prove Kijai split safetensors VAE filters are installed for split profiles and not limited to GGUF.

## Stop Conditions
Stop and report if:
- wrapper architecture differs from locator evidence.
- split-vs-official cannot be detected without changing `ResolvedLtxComponents`.
- helper would affect official monolith profiles.
- validation requires real model/GPU files.
- required fix exceeds allowed files.

## Required Return Contract
Return status, files inspected/changed, exact condition used for split detection, targeted test evidence, and remaining manual E2E risk. No broad logs.

---

### Task Packet 6 — Supervised local E2E smoke

## User Goal
Verify full end-to-end local profile workflow after product live path is ready: create profile, validate, save/reload, activate/select, run inference with GGUF and Kijai split profiles, and use official LoRA safetensors.

## Mode
general-coding

## Relevant Locations
- file: `frontend/components/ModelProfileWizard.tsx`
  symbol: profile create flow
  approximate lines: 258-318
  stable anchor: payload create, validate, activate steps
  reason: manual UI path starts here
  confidence: high
- file: `frontend/components/ICLoraPanel.tsx`
  symbol: IC-LoRA generate flow
  approximate lines: 102-170
  stable anchor: availability/recommendation/generate flow
  reason: manual LoRA selection path ends here
  confidence: high
- file: `backend/handlers/pipelines_handler.py`
  symbol: `_resolve_active_components`, `_resolve_checkpoint_paths`
  approximate lines: 120-170
  stable anchor: active profile controls generation paths; fallback raises `NO_DOWNLOADED_LTX_MODEL`
  reason: manual E2E should prove active profile path, not fallback
  confidence: high

## Allowed Edit Files
- none

## Read-Only Context Files
- `subagent-artifacts/e2e-profile-lora-locator.md`
- `subagent-artifacts/e2e-kijai-gguf-locator.md`
- `subagent-artifacts/profile-crud-locator.md`

## Required Change
No code change. Run supervised manual smoke only after Task Packets 1-5 pass:
1. Start app in dev mode.
2. Create a GGUF profile, validate, activate, close/reopen app, confirm active profile persists.
3. Run the smallest practical local T2V inference with GGUF active profile; confirm output exists and backend logs show active profile path, not `NO_DOWNLOADED_LTX_MODEL` fallback.
4. Create/activate a Kijai split safetensors profile, validate, reload, run smallest practical local T2V inference; confirm output exists and no VAE meta/OOM key-filter failure.
5. In IC-LoRA panel, select an official adapter such as `deblur` or `hdr` from `/mnt/ssd1/LTX_models/adapters`, generate, and confirm backend uses selected adapter id/path.

## Non-Goals
- Do not run full test suites.
- Do not benchmark quality/performance.
- Do not change model dropdown entries; dropdown may still show one local pipeline because profiles change component paths, not pipeline type.
- Do not alter real profiles except the supervised test profiles created for this smoke.

## Validation
Commands:
- Manual supervised app run; no automated full suite.

Expected result:
GGUF, Kijai split, and IC-LoRA selected official adapter each produce an output using active profile paths.

## Stop Conditions
Stop and report if:
- any prior implementation slice did not pass localized validation.
- app attempts fallback to downloaded official model instead of active profile.
- Kijai split fails with VAE key/meta-device issue.
- selected LoRA adapter id/path is not visible in request/log evidence.
- manual run risks overwriting user data; use disposable test profile names.

## Required Return Contract
Return status, exact profiles used, selected adapter id/path, output evidence, logs proving active profile path, and blockers. No full logs or large media dumps.

## Planner Self-Check
- locator evidence sufficient: yes — all implementation files and anchors come from supplied locator reports plus known worker result.
- allowed edit files minimal and explicit: yes — each slice lists exact files; no directory edit scopes.
- read-only context minimal: yes — limited to locator files and direct files named by locators.
- anchors/lines included: yes — each task includes path, symbol/anchor, approximate lines, reason, confidence.
- validation concrete: yes — targeted pytest/typecheck/manual smoke only; no full suites before live path.
- parallelization decision explicit and safe: yes — sequential default with partial parallel options; shared generated/schema risks called out.
- non-goals and stop conditions sufficient: yes — block typed-field expansion, new deps, broad schema churn, full suites, and product judgment.
- reviewer findings addressed, if revision: not applicable — no reviewer findings supplied.
