# Application Issue Report — 2026-07-11

## Status

Report only; no issue fixes applied.

The original full backend suite produced **988 passed, 10 failed** after the over-engineering cleanup. Focused 109 tests, typechecks, and the production frontend/Electron build passed. Static review determined those failures were not caused by that cleanup. Additional user-reported application issues were also statically traced; none has been fixed by this report.

**Plain-language terms:** a **sidecar** is a small companion file required by some model weights. A **proxy** is a smaller, broadly playable copy used for viewing while the original-quality primary file is retained. **Managed files** are files the app created and owns, rather than originals imported from a user’s disk.

## Resolved product decisions

- IC-LoRA uses one grouped model list, including valid Full/dev/GGUF choices. When a choice depends on sidecars, it automatically activates its one matching valid profile. If several profiles match, the app asks which profile to use. A profile-independent choice leaves the active profile unchanged.
- Deleting every generated asset always asks whether to remove it from the project only or also send eligible files to the OS Trash. Trash may contain only unshared, app-managed files; imported originals are never sent there.
- If Trash partly fails, the asset is still removed from the project, the app reports the leftover files, and editor undo history is cleared.
- An imported MOV remains the primary project file. It always receives a persistent, project-local H.264/AAC proxy for playback. All viewers prefer that proxy; generation/backend input continues to use the primary.
- The two no-op inpaint fields are removed end-to-end. The fixed stage-1 blend remains; there is no stage-2 restore.

## Summary

| Group | Priority | Classification | Failures / impact | Recommended action |
|---|---|---:|---:|---|
| Generated-asset deletion and managed-file Trash | P0 safety prerequisite / P1 feature | Data-safety contract | State-only deletion can leave duplicate managed files | Add provenance, reference analysis, guarded Trash IPC, and mandatory choice |
| uint8 MP4 encoding | P1 | App contract bug | 3 backend-suite failures | Normalize uint8 at the app adapter |
| IC-LoRA model list hides valid Full/dev/GGUF selections | P1 | Selection and profile-activation bug | Valid options are hidden or judged against only the active profile | Add compatibility metadata and grouped, profile-aware selection |
| Imported MOV primary replaced instead of persistent proxy | P1 | Media-preservation bug | Project copy loses its original-quality primary | Preserve primary and persist a project-local proxy |
| Non-HDR IC-LoRA output ignores persistent proxy | P2 | Viewer-path bug | Non-HDR playback can bypass the proxy | Prefer `outputProxyPath ?? outputVideoPath` |
| Inactive inpaint controls | P2 | Resolved contract removal | 1 backend-suite failure and misleading controls | Remove both no-op fields end-to-end |
| GGUF raw-weight placement | P2 | Nondeterministic test debt | 1 backend-suite failure | Pin the VRAM tier in the test |
| Offload-mode assertions | P2 | Environment-sensitive test debt | 3 backend-suite failures | Pin memory inputs and assert policy |
| Generic IC-LoRA model options | P3 | Stale test debt | 2 backend-suite failures | Update expectations and unsupported reason |

## P1 — uint8 MP4 encoding

**Failing nodes**

- `tests/test_media_encoder.py::test_mp4_is_rec709_tagged`
- `tests/test_media_encoder.py::test_mp4_pixels_byte_identical_to_external`
- `tests/test_media_encoder.py::test_mp4_encode_no_progress`

**Observed behavior:** The `backend/services/media_encoder/media_encoder.py::MediaEncoder` contract accepts uint8 and float frames, but `backend/services/media_encoder/media_encoder_impl.py::MediaEncoderImpl._encode_mp4` forwards uint8 tensors to the installed `ltx_pipelines` BT.709 `avg_pool2d` path, which crashes. No affected normal-generation caller was identified.

**User/CI impact:** Internal callers meeting the advertised uint8 contract cannot encode MP4; the three backend-suite checks fail. Normal generation is not currently known to be affected.

**Evidence:** `backend/services/media_encoder/media_encoder.py::MediaEncoder` defines the contract and `backend/services/media_encoder/media_encoder_impl.py::MediaEncoderImpl._encode_mp4` forwards frame tensors; `tests/test_media_encoder.py` contains the failing contract checks. The installed `ltx_pipelines` MP4 BT.709 path requires floating-point input for `avg_pool2d`.

**Smallest safe fix:** At the app adapter, convert uint8 tensors to float in `[0, 1]` before the dependency call, including single-pass frame iterators. Preserve float inputs, audio, chunks, remuxing, and progress behavior. In `backend/tests/test_media_encoder.py::test_mp4_pixels_byte_identical_to_external`, normalize only the direct-upstream baseline input to float `[0, 1]`; keep the app-side input uint8. App-adapter normalization alone cannot make the direct upstream uint8 baseline pass.

**Do not:** Patch `ltx_pipelines`; change EXR or ProRes paths; materialize or double-consume a single-pass iterator; alter unrelated encoding behavior.

**Validation**

```sh
rtk pnpm backend:test -- tests/test_media_encoder.py::test_mp4_is_rec709_tagged tests/test_media_encoder.py::test_mp4_pixels_byte_identical_to_external tests/test_media_encoder.py::test_mp4_encode_no_progress
```

## P1 — IC-LoRA model list hides valid Full/dev/GGUF selections

**Confirmed finding:** Existing endpoints `GET /api/models/model-options?workflow=ic-lora|hdr-ic-lora` and `POST /api/model-profiles/{profile_id}/activate` already support model options and profile activation. GenSpace filters choices by the active `settings.model` family, IC-LoRA has no family selector, and sidecar-dependent readiness checks inspect only the active profile. This hides valid Full/dev/GGUF selections even when another existing profile can support them.

**Approved fix:** Add `ModelSelectionOption.compatible_profile_ids: string[]` additively from the backend. The backend evaluates existing profiles without activating them; the existing activation endpoint remains the authority that performs activation. The grouped renderer list removes the family filter only for `ic-lora` and `hdr-ic-lora`.

- One compatible profile: call `POST /api/model-profiles/{profile_id}/activate`, refresh `GET /api/models/model-options?workflow=ic-lora|hdr-ic-lora`, and commit `settings.modelSelection` only after the activated profile remains in the selected option's `compatible_profile_ids`.
- Multiple compatible profiles: show an explicit profile choice before activation, then refresh options and perform the same compatibility verification before committing `settings.modelSelection`.
- Empty compatibility list for a profile-independent option: select it without changing the active profile.
- No valid profile: keep the option disabled.

**Exact contract files/symbols:** `backend/api_types.py::ModelSelectionOption`; `backend/handlers/models_handler.py::{_dev_option_disabled_reason,_disabled_reason_for_entry,get_model_selection_options}`; `backend/handlers/model_profiles_handler.py::activate_profile`; `frontend/views/GenSpace.tsx::{PromptBar,ModelSelectionPopover}`; `frontend/hooks/use-model-selection-options.ts`; `frontend/hooks/use-model-profiles.ts::activateProfileSafe`; `frontend/lib/model-selection.ts`; generated `frontend/generated/backend-openapi.json` and `frontend/generated/backend-openapi.ts`; and focused tests under `backend/tests/test_models.py`, `backend/tests/test_model_profiles.py`, and `backend/tests/test_ic_lora.py`.

**Hard constraints:** Additive metadata only; do not activate a profile during option discovery; do not create a new activation authority; do not apply the relaxed family filtering outside `ic-lora`/`hdr-ic-lora`; do not enable a sidecar-dependent option unless a valid profile exists.

**Validation**

```sh
rtk pnpm backend:test -- tests/test_models.py tests/test_model_profiles.py tests/test_ic_lora.py
rtk pnpm openapi:check
rtk pnpm typecheck
rtk pnpm build:frontend
```

## P2 — non-HDR IC-LoRA output ignores persistent proxy

**Confirmed breakpoint:** The non-HDR output branch in `frontend/components/ICLoraPanel.tsx` uses `outputVideoPath`. HDR and persisted viewers already prefer the proxy.

**Fix:** Use `outputProxyPath ?? outputVideoPath` for the renderer source and its error logging. The primary remains the primary/backend input. This needs no API, IPC, or schema change.

**Validation:** `rtk pnpm typecheck:ts`, `rtk pnpm build:frontend`, and manual ProRes-with-proxy, MP4, and HDR playback cases.

## P0 safety prerequisite / P1 feature — generated-asset deletion and managed-file Trash

**Confirmed finding:** Current deletion removes project state only. Generated work can leave duplicate managed files in `.ltx-generations`, project copies, proxies, thumbnails, and takes.

**Proposed persisted fields (not yet approved implementation):** `origin: "generated" | "imported" | "unknown"` and `managedSourcePaths?: string[]`. Historical records migrate to `unknown` and cannot be sent to Trash. Project media fields become eligible only from a successful app-managed copy result; `managedSourcePaths` may contain only app-created `.ltx-generations` files. `origin` alone never proves ownership.

Before Trash, scan every persisted project plus the current unsaved in-memory project/editor state. Compute canonical, deduplicated references after excluding only the asset/take being removed. Enumerate every asset/take `path`, `proxyPath`, `bigThumbnailPath`, `smallThumbnailPath`, and `managedSourcePaths`; only explicit managed paths with zero references are eligible. The user is always asked whether deletion is project-only or Trash.

**Proposed IPC (not yet approved implementation):** `trashManagedProjectFiles`, accepting `projectId` and `filePaths` and returning the discriminated result exactly: success `{ success: true, trashedPaths: string[], failedPaths: Array<{ path: string; reason: string }> }`; preflight/operation failure `{ success: false, error: string }`. Electron must independently canonicalize every path; reject relative paths, symlink escapes, and paths outside the exact target project's managed directory or configured `.ltx-generations` root; and use `shell.trashItem` only. It must complete all containment/ownership preflight before the first `shell.trashItem`, then report nontransactional per-path runtime failures. It must never permanently unlink files.

On partial success, remove the asset from the project, report failed leftovers, and clear editor undo. Never Trash an imported original, shared/unknown/external path, directory bundle not individually proven safe, or any permanently unlinked file.

**Exact files:** `frontend/types/project-model.ts`, `frontend/lib/project-storage.ts`, `frontend/contexts/ProjectContext.tsx`, deletion entry points under `frontend/views/GenSpace.tsx` and `frontend/views/editor/`, `shared/electron-api-schema.ts`, and `electron/ipc/file-handlers.ts`.

**Validation:** Before implementing Trash, the implementation plan must predetermine the exact pure reference-helper check file path and exact command. Do not enable filesystem behavior until that check passes; no vague reference-count check is an executable validation.

## P1 — imported MOV primary replaced instead of persistent proxy

**Confirmed finding:** When no proxy is supplied, `addVisualAssetToProject` copies an import and then `transcodeVideoInPlace` replaces that project copy. The external original survives, but the project primary loses its original quality.

**Approved semantics:** Keep the existing IPC shape. The copied path always remains the project primary. Every imported video receives a collision-safe, project-local H.264/AAC/yuv420p proxy beside it and returns `proxyPath`, while preserving any supplied persistent proxy; derive thumbnails and metadata from the proxy. If proxy creation fails, do not report a successful video import. Persist `copied.proxyPath` at every import, take, and timeline call site. Every viewer uses `proxyPath ?? path`; backend input remains the primary. Do not persist temporary proxies.

**Exact owners:** `electron/ipc/file-handlers.ts::{transcodeVideoInPlace,registerFileHandlers}`, `shared/electron-api-schema.ts::electronAPISchemas.addVisualAssetToProject`, `frontend/types/project-model.ts::{assetSchema,assetTakeSchema}`. Persist/viewer call sites remain responsible for using the supplied persistent proxy.

**Validation:** `rtk pnpm typecheck`, `rtk pnpm build:frontend`, plus a manual import/viewer matrix covering MOV, ProRes, MP4, proxy presence, and all viewers.

## P2 — inactive inpaint controls: resolved contract removal

**Failing node**

- `tests/test_ltx_ic_lora_pipeline.py::TestLaplacianBlendGrowParameter::test_param_source_assertions`

**Confirmed finding:** `laplacian_blend_grow` and `final_mask_blur_px` are exposed by the frontend/API and forwarded into the pipeline, but neither affects processing. The only blend uses a hardcoded value of `5`; the stage-2 blend/raw-mask guard was intentionally removed.

**Resolved contract:** Remove `laplacian_blend_grow` and `final_mask_blur_px` from the `POST /api/ic-lora/generate` DTO, handler, protocol, concrete pipeline, fake, frontend state/UI/hook/request body, tests, and generated OpenAPI. Keep `mask_grow_px` and the hardcoded stage-1 blend of `5`. Do not restore stage 2 or its raw-mask guard.

**Exact files:** `backend/api_types.py`, `backend/handlers/ic_lora_handler.py`, `backend/services/ic_lora_pipeline/ic_lora_pipeline.py`, `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py::generate_inpaint`, `backend/tests/fakes/services.py`, `backend/tests/test_ic_lora.py`, `backend/tests/test_ltx_ic_lora_pipeline.py`, `frontend/components/ICLoraPanel.tsx`, `frontend/views/GenSpace.tsx`, `frontend/hooks/use-ic-lora.ts`, `frontend/generated/backend-openapi.json`, and `frontend/generated/backend-openapi.ts`. Remove or rewrite stale forwarding/default/source assertions to verify the fields are absent and the fixed blend remains.

**Validation**

```sh
rtk pnpm backend:test -- tests/test_ic_lora.py
rtk pnpm backend:test -- tests/test_ltx_ic_lora_pipeline.py
rtk pnpm openapi:check
rtk pnpm typecheck
rtk pnpm build:frontend
```

## P2 — GGUF raw-weight placement test

**Failing node**

- `tests/test_gguf_loader.py::test_install_gguf_loader_forces_none_offload_and_disables_streaming_builder`

**Observed behavior:** The test unconditionally expects `keep_raw_on_cpu=True`, but the loader policy is VRAM-tier dependent.

**User/CI impact:** CI can fail based on the host memory tier despite correct production policy.

**Evidence:** `tests/test_gguf_loader.py::test_install_gguf_loader_forces_none_offload_and_disables_streaming_builder` asserts a fixed raw-weight placement while the GGUF loader’s offload policy selects placement by available VRAM tier.

**Smallest safe fix:** Pin a low-VRAM tier in the test and assert `keep_raw_on_cpu=True`; optionally add a separate high-tier case.

**Do not:** Change production placement policy; derive the expected result from host memory.

**Validation**

```sh
rtk pnpm backend:test -- tests/test_gguf_loader.py::test_install_gguf_loader_forces_none_offload_and_disables_streaming_builder
```

## P2 — offload-mode assertions

**Failing nodes**

- `tests/test_ltx_split_safetensors.py::TestOffloadModeGuard::test_official_remains_none`
- `tests/test_ltx_split_safetensors.py::TestOffloadModeGuard::test_ic_lora_official_remains_none`
- `tests/test_ltx_split_safetensors.py::TestOffloadModeGuard::test_a2v_official_remains_none`

**Observed behavior:** These three failures are official-profile cases with fixed `NONE` expectations that vary with the environment’s available-memory inputs. Under unknown VRAM, the global fail-safe returns `OffloadMode.CPU` before quantization-specific policy runs, while the tests expect official FP8 `NONE`.

**User/CI impact:** Correct policy behavior produces host-dependent suite failures.

**Evidence:** `tests/test_ltx_split_safetensors.py::TestOffloadModeGuard` asserts fixed modes; the memory planner chooses a route from memory inputs, returns `OffloadMode.CPU` for unknown VRAM before quantization-specific policy, and routes Kijai to `NONE` below 40 GiB.

**Smallest safe fix:** Pin a known non-HDR effective tier of at least 31 GiB when asserting official FP8 `NONE`. Keep unknown-VRAM CPU and below-40-GiB Kijai `NONE` as separate production invariants.

**Do not:** Change production routing to satisfy fixed assertions; remove the unknown-VRAM CPU fail-safe; change the below-40-GiB Kijai `NONE` route.

**Validation**

```sh
rtk pnpm backend:test -- tests/test_ltx_split_safetensors.py::TestOffloadModeGuard::test_official_remains_none tests/test_ltx_split_safetensors.py::TestOffloadModeGuard::test_ic_lora_official_remains_none tests/test_ltx_split_safetensors.py::TestOffloadModeGuard::test_a2v_official_remains_none
```

## P3 — generic IC-LoRA model options

**Failing nodes**

- `tests/test_models.py::TestModelSelectionOptions::test_unsupported_workflow_disables_even_installed_options`
- `tests/test_models.py::TestModelSelectionOptions::test_generic_ic_lora_model_options_remain_unsupported`

**Observed behavior:** `test_generic_ic_lora_model_options_remain_unsupported` incorrectly treats generic IC-LoRA model options as unsupported, while production/frontend/API intentionally support them. `test_unsupported_workflow_disables_even_installed_options` correctly tests unsupported retake, but its expected reason text is stale because it omits IC-LoRA from supported workflows.

**User/CI impact:** Two stale assertions fail despite intended selectable-model behavior.

**Evidence:** `tests/test_models.py::TestModelSelectionOptions::test_generic_ic_lora_model_options_remain_unsupported` contains the obsolete generic IC-LoRA unsupported expectation. `tests/test_models.py::TestModelSelectionOptions::test_unsupported_workflow_disables_even_installed_options` correctly asserts that retake is unsupported but has stale expected reason text omitting IC-LoRA from supported workflows. Generic `ic-lora` support is intentional across the model-options API, generated types, and frontend selection UI.

**Smallest safe fix:** Rewrite `test_generic_ic_lora_model_options_remain_unsupported` for supported generic IC-LoRA readiness gating. Keep `test_unsupported_workflow_disables_even_installed_options` testing unsupported retake, and update only its stale expected reason/constant to list IC-LoRA as supported.

**Do not:** Change production behavior, generated types, or frontend code solely to satisfy obsolete tests.

**Validation**

```sh
rtk pnpm backend:test -- tests/test_models.py::TestModelSelectionOptions::test_unsupported_workflow_disables_even_installed_options tests/test_models.py::TestModelSelectionOptions::test_generic_ic_lora_model_options_remain_unsupported
```

## Recommended implementation order

1. Persistent MOV proxy core.
2. Persist/viewer audit, plus the immediate `ICLoraPanel` proxy fix.
3. Model compatibility metadata and OpenAPI.
4. Grouped model list and profile-activation UI.
5. Deletion provenance only.
6. Reference analysis, guarded IPC, and deletion confirmation.
7. Remove the inpaint controls.
8. Update codemaps.
9. Run final OpenAPI, typecheck, backend-suite, and frontend-build validation.

## Final validation

```sh
rtk pnpm openapi:check
rtk pnpm typecheck
rtk pnpm backend:test
rtk pnpm build:frontend
```

## Remaining decisions

None. Implementation still requires an exact reviewed plan before edits.
