# Model Library, Download & Profile Plan

Status: **Implementation / reference plan (oracle-reviewed)**. Not code.
Audience: future implementers across `electron/`, `frontend/`, and `backend/`.
Scope: read-only scanning, cataloging, resolver/capability engine, transactional downloader, Model Library UI, profile schema, pipeline integration, HDR gating, telemetry, expert controls.

---

## 1. Purpose

Provide a single, user-owned model library that:

- Treats **already-downloaded** models (manual or otherwise) as first-class.
- Knows what is installed, what is missing, and what is broken/misplaced — without ever silently moving, deleting, or "repairing" files.
- Can download missing models **inside the user-selected root only**, with transactional guarantees (atomic promote, never overwrite user data on failure).
- Resolves a *profile* into concrete absolute paths and per-workflow/per-pipeline capabilities, so the rest of the app never hardcodes model paths.
- Exposes an honest UI: missing filename + exact expected path + source URL + wrong-folder usable status + gated/unvalidated support status.
- Keeps HDR, official-original, and distilled-LoRA flows correctly gated even when their files happen to be present.

This plan supersedes ad-hoc model path logic. It is deliberately read-only-first: the scanner ships before the downloader.

---

## 2. Verified Current State (do not re-download)

These were verified against the live install and code; bake the first two into the scanner's golden cases.

### 2.1 Installed HDR assets (already present, no download)

Path: `/mnt/ssd1/LTX_models/adapters/`

- `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`
- `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors`

These are **installed but must remain gated** (see §9 HDR gating). Presence ≠ supported.

### 2.2 Current models root layout

`/mnt/ssd1/LTX_models/` contains:

- `adapters/`
- `diffusion_models/`
- `gguf/`
- `text_encoders/`
- `vae/`
- `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` (at root — **legacy-compatible path**; canonical future location is `latent_upscale_models/`)

Scanner must handle both the root-legacy upscaler and the canonical subfolder layout without "fixing" either.

### 2.3 Current code facts (as of writing)

- Local specs expose **only `fast`**.
- Profiles are loaded **only from `model_profiles.json`** — no DB, no multi-source.
- **No recursive scanner** exists today.
- Distilled LoRA specs exist in code, but: **no download CP IDs**, and they are **not auto-loaded**.
- `fast` pipeline runs with `loras=[]`.
- Retake passes `loras=[]` and **does not wire `upsampler_path`**.
- A2V (image/audio-to-video) **does** use the upscaler.
- **HDR workflow remains gated** regardless of installed files.

Implication: the scanner must discover what exists; the resolver must decide what is *usable* in which pipeline; nothing today wires distilled LoRA or the retake upsampler — those land in Phase 6.

---

## 3. Principles (non-negotiable)

1. **One user-selected models root is source of truth.** Never download, write, or "repair" outside it.
2. **Never re-download installed models.** Scan first; the scanner is authoritative for "already have it."
3. **Scanner is read-only by default.** Never move, delete, rename, copy, or "repair" files automatically. Any mutation requires explicit user action in the UI.
4. **Existing manually downloaded models are first-class.** No "you must re-download from our list" friction.
5. **Wrong-folder models are usable** via their resolved absolute path. Canonical folders are a **recommendation**, not a requirement, unless the user explicitly approves a copy/move.
6. **UI honesty.** For every artifact, surface: missing filename, exact expected path, source URL, wrong-folder-but-usable status, and gated/unvalidated support status.
7. **Official original models are candidate/unvalidated until live-tested.** Never label them "supported" prematurely.
8. **HDR remains gated** even when HDR files are present (see §9).
9. **Distilled LoRA**: only for `dev`/non-distilled base in `fast`/`distilled` mode. **Never** for standalone distilled, Kijai-distilled, or QuantStack-distilled bases.
10. **Canny/depth disabled by default.** Runs only when Union Control is explicitly enabled. Order is **Union Control first, then selected LoRA.** Other LoRAs never get canny/depth preprocessing.

---

## 4. Phase Structure

Phases are ordered; each phase is independently shippable and testable. Do not reorder HDR (Phase 9) ahead of the gate.

### Phase 1 — Read-only scanner + catalog + profile schema migration
- Recursive-but-read-only scanner over the user root.
- Produce a **catalog** (see §5) of discovered artifacts.
- Migrate `model_profiles.json` to the new schema (`schema_version`, `created_by`, `validation_status`, `last_scanned_at`, `problems[]`) with full backward compatibility for existing profiles.
- No downloads, no mutations, no moves.

### Phase 2 — Resolver / capability engine + tests
- Pure function: `(active_profile, catalog, workflow, pipeline) → CapabilityResult`.
- Encode priority chain and support matrix (see §5).
- Heavily unit-tested before any UI/downloader depends on it.

### Phase 3 — Downloader backend (transactional)
- Per-item state machine: `idle → queued → downloading → promoting → installed | failed | cancelled`.
- All guarantees in §6. Fake downloader used in tests (no network).

### Phase 4 — Model Library UI
- Browse catalog, see installed/missing/wrong-folder/gated/unvalidated status.
- Initiate downloads (respecting "no redownload" and "root only").
- Show exact expected path + source URL + size/hash when known.
- Surface scanner `problems[]` per profile/artifact.

### Phase 5 — Guided candidate profile creation + activation validation
- Wizard to create a *candidate* profile from official templates.
- Activation is blocked during active generation and blocked if resolver says required artifacts are missing.
- Candidate profiles stay `validation_status: candidate` until validated.

### Phase 6 — Pipeline integration: distilled LoRA, component wiring, cache keys
- Wire distilled LoRA **only** for dev/non-distilled base in fast/distilled mode.
- Wire `upsampler_path` into retake (currently unwired).
- Component ordering: Union Control first, then selected LoRA.
- **Invalidate pipeline caches** when the active profile (or its resolved paths) change.

### Phase 7 — Local capability specs from resolver
- Replace hardcoded "only fast is exposed locally" with resolver-driven specs.
- Local specs become a function of `(profile, catalog)`.

### Phase 8 — Official model validation
- Live-test official original models. Flip `validation_status` from `candidate` → `validated` only after a successful live run.
- Never advertise official-original as supported before this.

### Phase 9 — HDR research/implementation behind gate
- HDR stays gated through Phases 1–8 even though files exist at `/mnt/ssd1/LTX_models/adapters/`.
- Research, then implement, then ungate via an explicit user-facing toggle — never silently.

### Phase 10 — Generation telemetry + hardware metrics
- Emit per-generation telemetry and hardware metrics (VRAM, time-to-first-frame, step times).
- **Download progress state is separate** from generation progress and hardware metrics (do not conflate channels).

### Phase 11 — Expert OOM / quality controls
- Exposed only in expert mode: resolution/step/batch tradeoffs, offload toggles, HDR/quality knobs (still gated), distilled toggles.

---

## 5. Data Model Notes

### Catalog (scanner output)
- Each entry has:
  - `artifact_kind` (e.g., `diffusion_model`, `vae`, `text_encoder`, `gguf`, `upscaler`, `control_adapter`, `lora`, `scene_embeddings`) — **not everything is LoRA.**
  - `component_role` (semantic role: `union_control`, `hdr_lora`, `hdr_scene_embeddings`, `distilled_lora_384_1_1`, `spatial_upscaler`, …).
  - `absolute_path`, `size_bytes`, `sha256?` (lazy/optional), `filename`.
  - `scanner_confidence`: `exact_catalog_match | filename_match | heuristic_match | unknown`.

### Profile (migrated schema)
- `schema_version`, `created_by` (`user` | `wizard` | `official_template`), `validation_status` (`candidate` | `validated` | `deprecated`), `last_scanned_at`, `problems[]`.
- Stores **resolved absolute paths** + **catalog provenance** (catalog entry id/hash when known) so we can detect moves/changes.
- **Adapter map keyed by semantic role**, not filename:
  - `union_control`, `hdr_lora`, `hdr_scene_embeddings`, `distilled_lora_384_1_1`, `spatial_upscaler`, `base_diffusion_model`, `vae`, `text_encoder`, `gemma`, …

### Support status
- **Per `(workflow, pipeline)` pair**, not only per artifact. An artifact can be "installed" yet "unsupported in retake" or "gated."

### Resolver priority (highest → lowest)
1. Active profile's resolved absolute path (authoritative when present).
2. Scanner catalog match.
3. Legacy root path (e.g., root-level upscaler).
4. `missing` (UI offers download with exact expected path + source URL).

---

## 6. Downloader Guarantees

- **Create canonical folders** under the user root on demand (`adapters/`, `diffusion_models/`, `vae/`, `text_encoders/`, `gguf/`, `latent_upscale_models/`, …).
- **Temp partial file, then atomic promote** (write `.part` → `fsync` → atomic rename). Never expose a half-written file to the scanner or pipeline.
- **Per-item lock file** to prevent concurrent downloads of the same artifact.
- **Never delete a user file on failed download.** A failed/`promote` step leaves the pre-existing file (if any) untouched and the `.part` for optional resume/cleanup.
- **Size/hash are optional and lazy.** Validate when known; missing metadata must not block download.
- **Disk-space preflight** before starting; fail fast with a clear error if insufficient.
- **HF gated-model handling**: detect gated repos, prompt for/refresh token, surface clear errors (401/403/gated) — never silently swallow.
- **Download progress is its own state**, separate from generation progress and hardware metrics (see Phase 10).
- **Root-only enforcement**: every byte is written under the user-selected root. No exceptions for caches/temp.

---

## 7. UI Requirements

For each artifact, the Model Library UI must show:

- **Filename** (and a missing-filename callout when absent).
- **Exact expected path** (canonical, under the user root).
- **Source URL** (HF/CP/etc.) for one-click download.
- **Status chip**: `installed` | `wrong_folder_usable` | `missing` | `downloading` | `failed` | `gated` | `unvalidated_candidate`.
- **Wrong-folder-but-usable** affordance: show the resolved absolute path and offer (not auto-run) a copy/move to the canonical folder.
- **Gated / unvalidated** badges (HDR, official-original candidate) — distinct from "installed."
- Scanner `problems[]` per profile/artifact (e.g., duplicate, unknown kind, partial file, size mismatch).
- Download controls respect: no-redownload, root-only, per-item lock, disk-space preflight, cancel.
- Profile activation UI blocked during active generation and when required artifacts are missing per resolver.

---

## 8. HDR Gating Details

- HDR artifacts are **installed** at `/mnt/ssd1/LTX_models/adapters/` (`ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`, `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors`).
- Despite being installed, HDR is **gated** end-to-end: scanner may list them, resolver marks the HDR workflow `gated`, UI shows a gated badge, and pipelines must not wire HDR components unless the explicit HDR toggle is on.
- The gate is **not** lifted by file presence, profile creation, or validation of *other* workflows. HDR ungate is Phase 9 work and requires its own user-facing toggle.
- Local Gemma/text projection remains the path for prompt enhancement with HDR; HDR presence does **not** introduce API-key prompting (see §10 test).

---

## 9. Test Strategy

Integration-first on the backend (Starlette `TestClient` + fakes, no `unittest.mock` per repo rules), unit tests for the resolver, and (new) frontend component tests for the Library UI status chips.

### Scanner tests
- Handles the **current `/mnt/ssd1/LTX_models` layout** (subfolders + root-legacy upscaler).
- **Root-legacy upscaler** (`ltx-2.3-spatial-upscaler-x2-1.0.safetensors` at root) resolves as usable.
- **Wrong-folder usable**: artifact in non-canonical folder resolves via absolute path.
- **Duplicates** (same file in two folders) → catalog dedupe + `problems[]`.
- **Unknowns** (unrecognized files) → listed, `scanner_confidence: unknown`, never deleted.
- **Partials** (`*.part`, `*.tmp`) → flagged, not promoted, not used.

### Profile tests
- **Migration / backward compatibility**: old `model_profiles.json` loads under new schema with `schema_version`, `validation_status`, etc.
- Round-trip: scan → resolve → serialize → reload.

### Downloader tests
- **No redownload** when artifact already in catalog.
- **Failed download preserves existing user files** (pre-existing file untouched; only `.part` mutated).
- **Fake downloader** performs no network I/O (enforced by test boundary).
- Atomic promote ordering; per-item lock contention; disk-space preflight failure path; cancel mid-download.

### Resolver / capability tests
- Full **capability matrix** across `(workflow, pipeline)` pairs.
- Priority chain: active-profile path > scanner match > legacy root > missing.
- **Distilled LoRA load/no-load matrix**:
  - dev/non-distilled base + fast/distilled → **loads** distilled LoRA.
  - standalone distilled / Kijai-distilled / QuantStack-distilled → **does not load**.
- Canny/depth default-off; Union-Control-first ordering when enabled.

### Gating tests
- **HDR remains gated when HDR files are present** (assert workflow status is `gated`, pipelines do not wire HDR components).
- Local Gemma/text projection **suppresses API-key prompt** for prompt enhancement.

### Lifecycle tests
- **Profile activation blocked during active generation.**
- **Pipeline cache invalidates on active profile changes** (resolved-path change, adapter map change, validation_status change).

---

## 10. Living Checklist

Strike items as they ship. Update this section in the same PR that lands the work.

- [ ] Phase 1: read-only recursive scanner over `/mnt/ssd1/LTX_models` (incl. root-legacy upscaler).
- [ ] Phase 1: catalog with `artifact_kind` + `component_role` + `scanner_confidence`.
- [ ] Phase 1: `model_profiles.json` schema migration (backward-compatible).
- [ ] Phase 2: resolver unit suite (priority chain + capability matrix).
- [ ] Phase 2: distilled LoRA load/no-load matrix green.
- [ ] Phase 3: transactional downloader (atomic promote, per-item lock, no-delete-on-fail).
- [ ] Phase 3: disk-space preflight + HF gated/token handling.
- [ ] Phase 3: fake downloader (no network) wired into tests.
- [ ] Phase 4: Library UI status chips (installed/missing/wrong-folder/gated/unvalidated).
- [ ] Phase 4: exact expected path + source URL + size/hash-when-known displayed.
- [ ] Phase 5: candidate-profile wizard + activation validation.
- [ ] Phase 5: activation blocked during active generation.
- [ ] Phase 6: distilled LoRA wired for dev/non-distilled base in fast/distilled only.
- [ ] Phase 6: `upsampler_path` wired into retake (currently `loras=[]`, no upsampler).
- [ ] Phase 6: Union-Control-first ordering; canny/depth default-off.
- [ ] Phase 6: pipeline cache invalidation on active-profile change.
- [ ] Phase 7: local specs derived from resolver (no longer "only fast" hardcoded).
- [ ] Phase 8: official-original live validation; `candidate → validated` only after live run.
- [ ] Phase 9: HDR researched/implemented behind explicit gate toggle (files already installed).
- [ ] Phase 10: generation telemetry + hardware metrics (separate from download progress).
- [ ] Phase 11: expert OOM/quality controls (gated HDR/quality knobs stay gated until Phase 9).

---

## Appendix — Out of Scope / Explicit Non-Goals

- Auto-move/repair of wrong-folder or partial files.
- Re-downloading the already-installed HDR adapters or any catalog-present artifact.
- Treating "installed" as "supported" for HDR or official-original models.
- Loading distilled LoRA on standalone/Kijai/QuantStack distilled bases.
- Enabling canny/depth by default or running non-Union LoRAs through edge/depth preprocessing.
