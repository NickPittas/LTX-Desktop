# Current Plan — LTX-Desktop

This is the **canonical entry point** for all current planning on LTX-Desktop.

It **supersedes** the old root and scattered plan files
(`plan.md` detailed body, `progress.md`, and the various research and sub-plans
that previously lived under `docs/`, `backend/`,
`subagent-artifacts/`, and `.task-reports/`). Those scattered files were
consolidated on 2026-06-29 under
[`../archive/2026-06-29-scattered-plans/`](../archive/2026-06-29-scattered-plans/).
Archived files may be stale; if anything there conflicts with the documents in
this folder, **the current documents here win**.

> **Traceability caveat:** the prior detailed root `plan.md` body is **not**
> duplicated in the archive folder — root `plan.md` was rewritten in place to
> the short pointer below. Its earlier content is reachable via
> `git log -p -- plan.md`, not via an archived file. The other scattered files
> (the four git-tracked ones plus untracked artifacts) **are** preserved in the
> archive.

The root [`plan.md`](../../../plan.md) is now a short pointer to this file.

---

## Progress / Status Ledger

Single source of truth for what is done and what is in flight. Update a row to
`completed` / `in-progress` / `pending` immediately after each task finishes.
Planning rows are immutable history; execution-step rows track the work below.
Dates are `YYYY-MM-DD`.

### Planning (setup — immutable)

| Task | Status |
|---|---|
| Scattered plans consolidated into `docs/plans/current/` | ✅ completed (2026-06-29) |
| Stale plans archived under `docs/plans/archive/2026-06-29-scattered-plans/` | ✅ completed (2026-06-29) |
| Oracle / planner review of consolidated plan | ✅ completed (2026-06-29) |
| Planner corrections applied | ✅ completed (2026-06-29) |
| **HDR gate/status decision (Lane D)** — option (a) temporary re-gate | ❎ OBSOLETE / wrong-turn (2026-06-29) — superseded by user clarification |
| **HDR gate/status decision (revised)** — fully finish HDR to `supported` before commit (option b); requires SDR proxy + real smoke | ✅ completed (2026-06-29) — smoke passed; HDR is `supported` |

### Execution steps

Order is fixed by the **Current Execution Order** section below; this table
records status only and does not change sequencing.

| Step | Status | Notes |
|---|---|---|
| 1 — Finish / reconcile uncommitted code (`01-finish-uncommitted-code.md`) | 🔄 in-progress | IC-LoRA/text-handler validation, GGUF helper decision (resolved: removed), Electron logging ride-along. **HDR gate/status: the Lane D "option (a) temporary re-gate" is OBSOLETE — a wrong-turn superseded by user clarification (2026-06-29).** HDR is no longer parked behind a re-gate; it is **`supported`** as of Step 2 (smoke passed). Step 1's own reconcile items (commit-staging sign-off) remain; the HDR gate is no longer a blocker. |
| 2 — HDR closeout, validation, and status cleanup (`02-hdr-completion-and-testing.md`) | ✅ completed | HDR is **`supported`**. All four implementation lanes (A backend/API, B SDR proxy encoder, C scanner/resolver flip + tests, D UI proxy preview + EXR reveal) landed. **End-to-end smoke PASSED (2026-06-29):** real HDR generation succeeded after the audio-context fix — endpoint **200**; **9 linear EXR frames** at `…/outputs/hdr_20260629_145849_645d0893_exr`; SDR proxy `…/hdr_20260629_145849_645d0893_exr_proxy.mp4` = **H.264, yuv420p, BT.709, 512×512, 9 frames**. Prior validation: **238 targeted backend tests**, `pnpm typecheck`, `pnpm build:frontend` all passed. Step 3 (commit) unblocked. |
| 3 — Commit validated current work | ⏳ pending (unblocked) | Steps 1 & 2 done and HDR is `supported`; stage enumerated files only; no push without explicit confirmation. |
| 4 — Live model selection / request-scoped profile switching (`03-live-model-selection.md`) | 🔄 in-progress (model registry split discovered) | Phases 1–4 landed at code/test/build level but are **not closed**: current `model-options` is still too CP-ID-centric and misses scanner-known Fast-family assets (Kijai distilled FP8 and QuantStack distilled GGUF). Must first fix the unified base-video model registry/source-of-truth plan in `03-live-model-selection.md` before smoke/commit. Phase 5 (A2V/IC-LoRA/retake) deferred by design. |
| 5 — Deferred and stale follow-ups (`04-deferred-and-stale-followups.md`) | ⏳ pending | Pulled in deliberately, not opportunistically. |

---

## Subplans

| # | Subplan | Scope |
|---|---|---|
| 1 | [01-finish-uncommitted-code.md](01-finish-uncommitted-code.md) | Reconcile/validate the current uncommitted working-tree changes before anything new. |
| 2 | [02-hdr-completion-and-testing.md](02-hdr-completion-and-testing.md) | HDR IC-LoRA is **implemented** and **validated** in the backend/UI path; **HDR is `supported`** — implementation lanes A–D landed and the **real-asset end-to-end smoke passed** (2026-06-29: endpoint 200; 9 linear EXR frames + H.264 yuv420p BT.709 512×512 9-frame SDR proxy). Prior validation: 238 targeted backend tests + `typecheck` + `build:frontend`. Step 3 (commit) is unblocked. |
| 3 | [03-live-model-selection.md](03-live-model-selection.md) | Live, request-scoped model selection with backend-owned options; prompt-box Model popover. **Status (2026-06-29): in-progress** — Phases 1–4 landed at code/test/build level but exposed a source-of-truth split: selectable options are derived from CP IDs and omit scanner-known Fast-family Kijai/QuantStack assets. The current required next slice is the unified base-video model registry plan in that document. |
| 4 | [04-deferred-and-stale-followups.md](04-deferred-and-stale-followups.md) | Explicit backlog of deferred / superseded items so they are not accidentally reopened. |

---

## Current Execution Order

Work proceeds top-to-bottom. Later steps depend on earlier ones.

1. **Finish / reconcile uncommitted code** — see `01-finish-uncommitted-code.md`.
   Validate the in-flight changes to IC-LoRA / text handler / Electron logging /
   the HDR UI workflow, decide what stays, resolve the unused GGUF helper, and
   reconcile the HDR gate/status mismatch.
2. **HDR closeout, validation, and status cleanup — DONE; HDR is `supported`** —
   see `02-hdr-completion-and-testing.md`. HDR generation is **implemented
   end-to-end** (backend `_generate_hdr` handler dispatch, pipeline
   scene-embedding prompt-encoder swap + LogC3→linear postprocess, EXR primary
   output, and the UI `hdr` workflow) and **validated**. **End-to-end smoke
   PASSED (2026-06-29):** real HDR generation succeeded after the audio-context
   fix — endpoint returned **200**; output was **9 linear EXR frames** at
   `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr`;
   SDR proxy `..._exr_proxy.mp4` is **H.264, yuv420p, BT.709, 512×512, 9 frames**.
   Prior validation: **238 targeted backend tests**, `pnpm typecheck`, and
   `pnpm build:frontend` all passed. Backend handler/UI, `model_scanner.py`, and
   `model_resolver.py` (+ tests) are all at `supported` (with
   `hdr_scene_embeddings` kept as a support asset). Step 2 is complete; Step 3
   (commit) is unblocked subject to explicit confirmation.
3. **Commit validated current work** — only after steps 1 and 2 land and pass
   the validation commands **and HDR is `supported`** (not before). Stage only
   intended tracked source files; never commit `HF_TOKEN`, `.env`, codemap
   artifacts, or `backend/data/`. No push without explicit user confirmation.
4. **Live model selection / request-scoped profile switching** — see
   `03-live-model-selection.md`. Build the backend contract first, then
   request-scoped resolver context, T2V/I2V first, then the frontend popover.
   **Status (2026-06-29): in-progress.** Phases 1–4 landed at code/test/build
   level but revealed the model-discovery source-of-truth split: options are
   still derived from downloadable CP IDs while Kijai distilled FP8 and
   QuantStack distilled GGUF are scanner-known but not selectable. The next
   required slice is the unified base-video model registry plan in
   `03-live-model-selection.md`; no smoke/commit closes Step 4 before that is
   fixed. Phase 5 (A2V/IC-LoRA/retake) stays deferred by design.
5. **Deferred follow-ups** — see `04-deferred-and-stale-followups.md`. Pulled in
   deliberately, not opportunistically.

---

## Parallelization Notes

**Cannot be parallelized (hard dependencies):**

- Step 1 (finish/reconcile) **blocks** step 3 (commit) — do not commit before
  reconciliation.
- Step 2 (HDR closeout) must reach **`supported`** (backend + UI + scanner +
  resolver all consistent) before HDR touches are committed. The earlier
  "re-gate to match scanner/resolver" option is **obsolete**; commit now waits
  on HDR being fully finished (SDR proxy + real smoke). Do not commit a
  half-gated state and do not commit a re-gated state.
- `03-live-model-selection.md` Phase 1 (backend contract / model-options
  endpoint) **blocks** its Phase 4 (frontend popover) — the frontend must not
  infer model support; it renders backend-owned options.
- Cache-key hardening (part of live model selection) **must land before** broad
  live switching is enabled, otherwise stale pipelines will be served.

**Can be parallelized (independent tracks):**

- The HDR backend implementation is **already done**, and as of 2026-06-29 all
  four HDR closeout implementation lanes landed (Lane A backend/API →
  `supported` path, Lane B SDR proxy encoder, Lane C scanner/resolver flip +
  tests, Lane D UI proxy preview + EXR reveal) **and the real-asset end-to-end
  smoke passed** (endpoint 200; 9 linear EXR frames + H.264 yuv420p BT.709
  512×512 9-frame SDR proxy). **HDR is `supported`; Step 2 is complete.** The
  only HDR-adjacent item left is the Step 3 commit decision (not parallelizable
  implementation). Live-model-selection (step 4) work may proceed after the
  Step 3 commit per the execution order — do **not** start it in parallel with
  the commit unless the execution order is explicitly changed. (Earlier drafts
  claimed HDR backend implementation could proceed in parallel with the
  live-model-selection backend contract; that is stale now that HDR is
  implemented and validated.)
- **Deferred follow-ups** that are clearly independent (e.g. distilled GGUF
  inventory, motion track research) can be explored in parallel with the main
  track as long as they do not edit source the main track depends on.
- **Frontend Model popover design/UX** (live model selection) can be drafted in
  parallel with the backend contract, but must not be wired until the backend
  options endpoint exists.

**Do not parallelize across the same file.** If two tracks need
`ic_lora_handler.py`, sequence them.
