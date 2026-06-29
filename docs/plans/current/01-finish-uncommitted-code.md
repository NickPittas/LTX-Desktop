# 01 — Finish / Reconcile Uncommitted Code

> Step 1 of the [current plan](README.md).

Goal: reconcile the current uncommitted working-tree changes, validate them,
decide what stays, and get the tree to a clean, intentional state **before** any
HDR or live-model-selection work continues. No new features in this step.

## Current uncommitted files (current uncommitted tracked source files for this code slice)

The files below are the **current uncommitted tracked source files for this
code slice** — not a blanket "per `git status`" snapshot. The full working tree
also contains untracked planning docs and renamed archive files; those are
addressed in the **Working-tree hygiene / staging inventory** below, not here.

| File | Nature of change |
|---|---|
| `backend/handlers/ic_lora_handler.py` | IC-LoRA prompt enhancer now consults the **I2V** enhancer setting when image conditioning is present (Ingredients + image-conditioned path), and T2V otherwise. Added comments noting IC-LoRA handler-level enhancement is API-only (local GGUF Gemma never does image-conditioned enhancement here). |
| `backend/handlers/text_handler.py` | Added `active_profile_uses_gguf_text_encoder` helper (detects GGUF Gemma text encoder from profile components). **Status: currently unused — must be wired in or removed.** |
| `frontend/components/ICLoraPanel.tsx` | Promotes `hdr` from `unavailable` to a real `hdr` workflow (no-input, prompt-only; EXR folder output messaging). **HDR is implemented end-to-end.** A temporary re-gate was applied then **superseded** — see task 3 below and `02-hdr-completion-and-testing.md`; the active plan finishes HDR to `supported` (UI previews SDR proxy + reveals EXR). |
| `electron/main.ts` | Adds `import './linux-graphics-env'` before the `electron` import (Linux + NVIDIA VAAPI env defaults). |
| `electron/python-backend.ts` | Backend stderr classifier: demotes benign `llama.cpp` init / `tqdm` progress lines to INFO so they stop flooding the session log as ERROR; real errors stay ERROR. |

## Notes on already-decided / low-priority items

- **Electron logging fix (`electron/main.ts`, `electron/python-backend.ts`):**
  the user has stated this is **already fixed** and the prior plan reference was
  **stale**. Treat it as decided and low priority for re-planning — it just
  needs to ride along with the commit; do not re-litigate the design here. (It
  also depends on the untracked `electron/linux-graphics-env.ts` file — see
  commit caution below.)

## Working-tree hygiene / staging inventory

When staging the commit for this slice, stage an **intentional, enumerated**
set — never `git add -A`/`git add .`. Concretely:

- **Untracked planning docs — stage only if intentionally committing them:**
  `plan.md` (now a pointer), `docs/plans/current/*` (README + subplans 01–04),
  and `docs/plans/archive/2026-06-29-scattered-plans/README.md`. These are
  untracked, so they will **not** be included unless explicitly added. Decide
  deliberately whether docs ride along in this slice or a separate docs commit.
- **Four old planning files are staged/renamed into the archive** via `git mv`
  (history preserved): `HANDOFF.md`,
  `docs--model-library-download-profile-plan.md`,
  `docs--ltx-offline-research--04-recommended-implementation-plan.md`,
  `docs--ltx-offline-research--05-ltx-desktop-model-profiles-and-gguf-kijai-plan.md`.
  These renames are intentional; keep them staged together.
- **`electron/main.ts` depends on the untracked `electron/linux-graphics-env.ts`.**
  Commit **both or neither** — never commit `main.ts` alone, or the tree will
  import a file that is absent from the repo.
- **Never stage** any of: `HF_TOKEN`, `.env`, `backend/data/`, codemap artifacts
  (e.g. `codemap.md`, `<folder>/codemap.md`, `.slim/codemap.json`),
  `.task-reports/`, or locator/scratch artifacts. Verify the staged set with
  `git status --short` and `git diff --cached --stat` before committing.

## Explicit tasks

1. **Validate the IC-LoRA prompt enhancer setting change.**
   - Confirm the Ingredients / image-conditioned path now correctly uses
     `prompt_enhancer_enabled_i2v`, and the pure-video path uses
     `prompt_enhancer_enabled_t2v`.
   - Confirm local GGUF profiles still never request image-conditioned
     enhancement at the handler level (the GGUF call patch remains the deep
     safety net). The added comments assert this; verify against behaviour.

2. **Resolve the unused GGUF text helper.**
   - `TextHandler.active_profile_uses_gguf_text_encoder` in
     `backend/handlers/text_handler.py` is currently **added but unused**.
   - **Do not** wire this helper to route GGUF profiles to the API prompt path,
     and **do not** reintroduce any GGUF I2V gate or API-key prompting for
     prompt enhancement. Those behaviours are intentionally gone (local GGUF
     Gemma must run without prompting for API credentials).
   - Decide between exactly two options:
     - **Remove it** as dead code (preferred if there is no real consumer), or
     - **Wire it only to a non-prompting safety path** — i.e. a guard that does
       **not** trigger API-key prompting and does **not** gate GGUF I2V (for
       example, a purely local assertion/skip that is logged, never a hard
       gate that blocks GGUF I2V or falls back to API).
   - **Tests must prove** that GGUF I2V remains **ungated** and that GGUF
     profiles **do not** prompt for API credentials, regardless of which option
     is chosen. Add/extend a backend test asserting this before the helper
     lands wired or is removed.

3. **HDR is implemented in backend/UI; resolve the gate/status mismatch before commit.**
   - The `ICLoraPanel.tsx` change promotes `hdr` from `unavailable` to a real
     `hdr` workflow, and the backend path is **already implemented** (not
     scaffolding): `_generate_hdr()` exists in
     `backend/handlers/ic_lora_handler.py`, `hdr` is **not** in
     `_UNAVAILABLE_WORKFLOWS` (only `hdr_scene_embeddings` is, as a support
     asset), the scene-embedding prompt-encoder swap + LogC3→linear postprocess
     exists in
     `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`, and HDR
     success/postprocess tests exist (`backend/tests/test_ic_lora.py`,
     `backend/tests/test_hdr_utils.py`). EXR output plumbing exists.
   - **Gate/status mismatch (must-fix before commit):** the handler/UI
     generation path is **enabled**, but `backend/services/model_scanner.py`
     (`_GATED_ADAPTER_IDS = {"hdr", "hdr_scene_embeddings"}`) and
     `backend/services/model_resolver.py` (`_GATED_ROLES`, `hdr_status="gated"`,
     plus the resolver tests asserting gated) still mark HDR **gated**. This
     half-gated state is inconsistent and must not be committed.
    - **Decision — the active direction is option (b): finish HDR to
      `supported` before commit.**
      - **(a) Re-gate the UI/backend** — ⚠️ **OBSOLETE / WRONG-TURN.** This was
        briefly chosen (2026-06-29, Lane D) and applied to the working tree,
        but it is **superseded by user clarification (2026-06-29)**. It is
        retained here only as history; do not treat the re-gate as the active
        direction.
      - **(b) Validate HDR and flip it to supported** — update `model_scanner.py`,
        `model_resolver.py`, and the resolver/scanner tests so HDR reads
        `supported`/not-gated, while keeping `hdr_scene_embeddings` as a
        **support asset** (still not selectable as a standalone adapter). This
        is now the active goal and is executed in Step 2
        (`02-hdr-completion-and-testing.md`).
    - The validation evidence and SDR proxy policy required for option (b) live
      in `02-hdr-completion-and-testing.md`. Per oracle's **revised**
      recommendation, `supported` requires both an **SDR proxy** (generated
      alongside the linear EXR) and a passing **real-asset HDR smoke**.
    - **✳️ DECISION REVISED (2026-06-29, user clarification): option (b) —
      fully finish HDR to `supported` before commit.** The earlier "option (a)
      temporary re-gate" (recorded below) was a wrong-turn and is obsolete.
      - **Obsolete / superseded — kept for history only:** "option (a)
        temporary re-gate" (2026-06-29, Lane D). Oracle's *then-current*
        rationale was that the HDR backend/UI path is implemented but lacks
        real-asset smoke validation and a defined SDR proxy tonemap policy,
        while the scanner/resolver still mark HDR `gated`; it recommended
        re-gating UI/backend to match the scanner/resolver state now and
        un-gating later. **This is no longer the plan.** The working-tree
        artifacts of that decision still need to be reverted as part of Step 2:
        - `backend/handlers/ic_lora_handler.py`: `hdr` was re-added to
          `_UNAVAILABLE_WORKFLOWS` with a temporary-unavailable message; the
          dormant `_generate_hdr()` was left in place. **(To be reverted:
          remove `hdr` from `_UNAVAILABLE_WORKFLOWS` to re-enable dispatch as
          part of reaching `supported`.)**
        - `frontend/components/ICLoraPanel.tsx`: `hdr` adapter entry was
          reverted to `workflow: 'unavailable'`. **(To be reverted: restore the
          real `hdr` workflow, with SDR-proxy preview + EXR reveal.)**
        - `backend/tests/test_ic_lora.py`: four HDR endpoint tests were flipped
          to assert the unavailable 400. **(To be reverted: restore the HDR
          success tests once smoke passes.)**
        - Scanner/resolver were intentionally left `gated`. **(To be flipped to
          `supported` as part of Step 2.)**

4. **Separate unrelated Electron changes at commit time.**
   - The Electron logging/graphics-env changes are unrelated to the IC-LoRA /
     HDR model work. If/when committing, stage them as a **separate commit**
     from any model/HDR commit, so history stays bisectable.
   - **Caution:** `electron/main.ts` imports the **untracked**
     `electron/linux-graphics-env.ts`. Do not commit `main.ts` without also
     deciding what to do with that file (stage it, or drop the import). Never
     leave the tree referencing an untracked file in a committed import.

## Validation commands

Run via the RTK-wrapped pnpm pin:

```bash
# TypeScript (frontend + electron)
rtk npx --yes pnpm@10.30.3 typecheck:ts

# Python pyright (strict)
rtk npx --yes pnpm@10.30.3 typecheck:py

# Both
rtk npx --yes pnpm@10.30.3 typecheck

# IC-LoRA tests (the handler path touched here)
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_ic_lora.py

# OpenAPI consistency (only if any api_types / response shape changed)
rtk npx --yes pnpm@10.30.3 openapi:check

# Frontend build
rtk npx --yes pnpm@10.30.3 build:frontend
```

CI gate (must pass before commit): `pnpm typecheck` + `pnpm backend:test` +
frontend Vite build.

## Parallel lanes

This slice is small and mostly independent validation; where helpful, run it as
three parallel lanes (do **not** parallelize edits within the same file):

- **Lane A — Python IC-LoRA / text-handler validation:** validate
  `backend/handlers/ic_lora_handler.py` (I2V/T2V enhancer setting) and
  `backend/handlers/text_handler.py` (GGUF helper decision), plus the relevant
  `backend/tests/test_ic_lora.py` paths.
- **Lane B — Electron logging / graphics-env validation:** validate
  `electron/main.ts`, `electron/python-backend.ts`, and the untracked
  `electron/linux-graphics-env.ts` as one unit (they must commit together).
- **Lane C — docs / staging hygiene:** the working-tree/staging inventory and
  the planning-doc staging decision (see section above).
- **Do not parallelize `ICLoraPanel.tsx` (HDR UI) with HDR backend/UI work** —
  the gate/status decision in task 3 is one logical change across UI +
  scanner/resolver; sequence same-file and cross-cutting HDR edits.

## Stop conditions

- Stop if `typecheck:py` / `typecheck:ts` reports errors not trivially fixable
  in this slice.
- Stop if HDR is not finished to `supported` before the Step 3 commit — i.e.
  do not commit while the handler/UI/scanner/resolver are not all consistent at
  `supported` (the obsolete re-gate is not an acceptable commit state).
- Stop if the GGUF helper decision (task 2) would reintroduce GGUF I2V gating
  or API-key prompting, or if tests do not prove GGUF I2V stays ungated and
  non-prompting.
- Stop if `electron/main.ts` would be committed while
  `electron/linux-graphics-env.ts` is still untracked and unreferenced-safe.
- Stop before any commit and get explicit confirmation (no push without user
  confirmation — repo rule).
