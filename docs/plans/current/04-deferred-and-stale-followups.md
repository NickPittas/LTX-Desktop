# 04 — Deferred and Stale Follow-ups

> Step 5 of the [current plan](README.md).

Explicit backlog of items that are **deferred** (valid but not in scope right
now) or **stale/superseded** (do not reopen). Kept here so they are not
accidentally picked up by future agents as active work.

Legend:
- 🟡 **Deferred** — still valid, pull in deliberately when prioritized.
- ⛔ **Stale / superseded** — do not reopen; replaced by the current plan or
  made obsolete by shipped code.

---

## Deferred (🟡 still valid)

- **Distilled GGUF inventory.** Verify distilled GGUF filenames + byte sizes
  **before** any distilled GGUF option is added to the UI or downloader. Do not
  pre-populate speculative rows. (Distilled dev GGUF entries must be confirmed,
  not assumed.)
- **Official original live validation.** A scoped live smoke against the
  official/original (non-GGUF) model set to confirm the canonical path still
  performs and outputs correctly after the recent profile/handler changes.
- **Motion Track.** `motion_track_control` IC-LoRA remains unavailable in the
  UI. Wiring is not planned yet; do not enable until a real motion-track
  pipeline path exists.
- **LipDub.** `lipdub` remains unavailable. Requires audio workflow support
  that is not wired yet.
- **Expert OOM / quality controls.** Advanced knobs (scheduler/sigma/noise
  scale, halfres guide toggles, etc.) seen in archived worker files are not
  user-facing product features. They can be revisited as an expert panel only
  if a concrete need arises.
- **GGUF perf / lazy-dequant plans.** The lazy `QParam`/`GgufLinear` GPU
  dequant path for Gemma Linear weights (and the broader torch-GGUF dequant
  worker ideas) are perf follow-ups, not blockers. Pursue only if measured
  dequant hotspots justify the work. Audit `QParam`/`GgufLinear` in
  `backend/services/patches/gguf_loader_fix.py` before any change.
- **Inpaint parity diagnostics — only if quality issue reappears.** Inpaint /
  in_outpainting is considered working. Do **not** touch inpaint mask / green
  guide / sigma code proactively. Reopen diagnostic work **only** if a real
  quality regression is observed.
- **Retake upsample — out of scope.** The retake upsample path stays out of
  scope. Do not modify retake code as part of the current track.

## Stale / superseded (⛔ do not reopen)

- ⛔ **Old root `plan.md` body (GGUF/dev I2V + downloader four-section plan).**
  Superseded by the consolidated current plan. Preserved verbatim under
  `../archive/2026-06-29-scattered-plans/`. The four-section downloader
  grouping (Full / Kijai / GGUF / Add-ons & Controls), catalog/scanner
  contract, mmproj data-model decision, and CRF <= 18 work are **already
  implemented/committed** as backend profile hardening — do not re-plan them.
- ⛔ **`ca9d62f feat(models): add dev GGUF downloader support` — fully shipped,
  do not re-plan.** This single commit landed the entire dev-GGUF downloader
  scope. Concretely, **do not reopen** any of:
  - **Downloader sections** (Full / Kijai / GGUF / Add-ons & Controls) —
    `backend/handlers/download_handler.py`,
    `backend/runtime_config/model_download_specs.py`,
    `backend/services/model_scanner.py`, `frontend/components/ModelLibraryPanel.tsx`.
  - **Dev GGUF catalog / scanner** — the dev GGUF entries and scanner
    recognition in `model_download_specs.py` and `model_scanner.py`.
  - **`mmproj` field / detection** — the mmproj data-model field in
    `backend/api_types.py`, `frontend/types/model-library.ts`, and scanner
    detection.
  - **GGUF I2V degrade / gate removal** — GGUF I2V no longer degrades/gates;
    see `backend/services/fast_video_pipeline/`,
    `backend/services/patches/gguf_loader_fix.py`, and
    `backend/tests/test_image_conditioning_crf.py`. (The unused GGUF text
    helper cleanup in `01-finish-uncommitted-code.md` task 2 is a separate,
    narrow matter and must **not** reintroduce any GGUF I2V gate.)
  - **CRF = 18** — the image-conditioning CRF=18 behaviour and its test.
  - **Dev-vs-distilled routing** — `backend/services/ltx_components.py`,
    `backend/services/model_resolver.py`,
    `backend/handlers/pipelines_handler.py`.
  If any of these surfaces a real regression, treat it as a new bug — not a
  re-plan of the feature.
- ⛔ **Old IC-LoRA weight / Ingredients T2V / PromptBar Generate no-op fixes.**
  Already committed in `239820b fix: improve local LTX workflows`. Superseded;
  do not redo.
- ⛔ **`subagent-artifacts/hdr-iclora-plan.md` and EXR/MOV primary-output
  plan.** Folded into `02-hdr-completion-and-testing.md` (architecture portion)
  and the EXR-primary dependency note. The detailed artifacts are archived; the
  active direction is the current subplan.
- ⛔ **Fast-T2V GGUF/split guard prior slice.** Already shipped
  (`UNSUPPORTED_FAST_T2V_MODEL_PROFILE` guard). Not active work.
- ⛔ **Kijai/streaming-fallback, green-leak, mask-edge-ghosting, and other
  inpaint/ingredients worker files** in the archive. Those were
  implementation-time diagnostics for already-shipped fixes; superseded by the
  shipped code. Do not reopen unless a regression appears.

> When in doubt: the [current plan README](README.md) and its subplans are the
> source of truth. Anything in `../archive/` that conflicts with them is wrong.
