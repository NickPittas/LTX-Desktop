# 03 — Live Model Selection (Request-Scoped Profile Switching)

> Step 4 of the [current plan](README.md).

Goal: let the user pick a model variant **per generation** from the prompt box,
without leaving their current profile as the default/advanced fallback. This is
the latest agreed architecture; it supersedes any older "global model switch"
framing in the archived plans.

## Status (2026-06-29) — in-progress: model registry source-of-truth split

Phases 1–4 landed at code/test/build level, but Step 4 is **not closed**. The
current implementation exposed a model-discovery source-of-truth split: live
model options are still derived from selectable/downloadable CP IDs, while
scanner-known local base models such as **Kijai distilled FP8** and
**QuantStack distilled GGUF** are recognized by the scanner but not selectable in
the Fast-family popover. Fix the unified base-video model registry plan below
before any smoke/commit closes this step. Phase 5 (A2V / IC-LoRA / retake)
remains **deferred by design**.

### Completed this session

- **Prior HDR commit `d42f226` pushed to `main`.**
- **Phase 1 — backend contract / options endpoint.** `model_selection` request
  field added; admin-guarded `GET /api/models/model-options` returns
  backend-owned option DTOs (workflow-aware, scanner-derived). OpenAPI
  regenerated.
- **Phase 2 — T2V/I2V request-scoped resolver + cache/text-cache threading.**
  Bad/unsupported selections reject with a clear error; A2V and the API
  profile reject when a selection is present; pipeline cache key includes the
  selection + effective component paths; text prompt cache includes model
  identity so GGUF-vs-full-Gemma text encoders do not leak across selections.
- **Phase 4 — frontend prompt-box Model popover.** Renders backend-owned
  options only (grouped, with disabled reasons shown verbatim — no inference);
  `Auto` / current-profile default; `model_selection` is sent **only** for
  video T2V/I2V; A2V/API and other deferred modes clear/disable the field.

### Oracle backend-review blockers fixed

- Extra fields forbidden on `model_selection`-bearing requests (camelCase
  `modelSelection` now 422s instead of being silently ignored).
- GGUF options are enabled only when **runtime-ready** (not merely installed).
- Distilled override clears split sidecars (no stale sidecar leak).
- Effective distilled LoRA path is part of the cache key.
- Selection-specific rejections run **before** generic spec validation, so
  errors are precise.

### Validation evidence

- `rtk npx pnpm backend:test -- tests/test_generation.py tests/test_ltx_components.py tests/test_models.py` → **175 passed**.
- Full backend suite (run earlier by the fixer) → **987 passed**.
- `rtk npx pnpm typecheck` → **passed** (TypeScript + Pyright, 0 errors).
- `rtk npx pnpm build:frontend` → **passed**.
- `openapi:generate` is **idempotent**. Note: `openapi:check` exits non-zero
  **only while the working tree is dirty** — the generated OpenAPI necessarily
  differs from HEAD until the change is committed; this is expected, not a
  failure.

### Remaining before Step 4 is fully closed

- **Required** model-discovery source-of-truth fix — model options must come from
  the unified base-video registry below, not only from downloadable CP IDs.
- **Required** real Fast-family popover check — `LTX 2.3 Fast` must show official
  distilled, Kijai distilled FP8, and QuantStack distilled GGUF entries with
  correct enabled/disabled states.
- **Required** real Full-family popover check — `LTX 2.3 Full` must show official
  dev/full and dev/full GGUF entries with correct enabled/disabled states.
- **Optional** real T2V/I2V live generation smoke after the option lists are
  correct.
- **Commit / push decision** — Step 4 work is currently **uncommitted**.
- **Phase 5 (A2V / IC-LoRA / retake)** remains deferred by design until the
  core T2V/I2V path + cache-key hardening have held up in real use.

## Required source-of-truth fix — base-video model registry

### Problem statement

The current live model-selection implementation uses CP/download specs as the
main option source. That is wrong for generation model selection because some
valid local base models are scanner-known artifacts, not downloadable CP IDs.
Result: Kijai distilled FP8 and QuantStack distilled GGUF can be recognized in
the model library scan but cannot be selected from the Fast-family Model popover.

### Model Discovery Rules for this slice

- One model registry/source of truth must drive scanner, model-options, resolver,
  downloader metadata, frontend generated types, and frontend model-selection UI.
- Generation model options must not be derived only from downloadable CP IDs.
- Scanner-only artifacts must either be promoted to selectable IDs or the
  model-selection API must explicitly support scanner artifact IDs.
- Kijai and QuantStack distilled models are Fast-family selectable base models.
- Dev/full GGUF models are Full-family selectable base models.

### Endpoint / API contract

No new endpoint.

Existing endpoint changed:

- `GET /api/models/model-options`
  - keep response field: `options`
  - keep option field: `pipeline_family`
  - change option field type: `options[].id` becomes `ModelSelectionID`
  - every option must preserve: `label`, `group`, `section`, `variant_group`,
    `installed`, `pipeline_family`, `disabled_reason`, `repo_id`, `source_url`,
    `canonical_relative_path`, `expected_absolute_path`, `downloadable`

Existing generation request changed:

- `POST /api/generate`
  - keep backend request field: `model_selection`
  - keep frontend request field: `modelSelection`
  - change type from `ModelCheckpointID | None` to `ModelSelectionID | None`

Shared names to preserve end-to-end:

- backend type alias: `ModelSelectionID`
- backend request field: `model_selection`
- frontend request field: `modelSelection`
- backend response field: `pipeline_family`
- generated/frontend response field: `pipelineFamily`
- resolver function: `resolve_base_video_model_selection`
- registry entry type: `BaseVideoModelRegistryEntry`
- registry iterator: `iter_base_video_model_entries`

### Registry ownership contract

The base-video registry is the source of truth for **generation-selectable base
video variants**. It owns the selectable IDs and the metadata used by scanner,
model-options, resolver, and frontend display. Downloadable checkpoint specs may
remain in `model_download_specs.py`, but registry entries must link to them via
`download_cp_id` instead of duplicating downloader state or deriving selection
only from `SELECTABLE_BASE_VIDEO_CP_IDS`.

`ModelSelectionID` is intentionally a runtime string, not a generated `Literal`,
because `/api/models/model-options` is the authoritative runtime source. Required
safety rules:

- backend rejects any unknown string with `UNSUPPORTED_MODEL_SELECTION`;
- frontend never hardcodes selection IDs except storing/rendering the backend's
  returned `option.id` and clearing to `null`/Auto;
- tests cover arbitrary unknown `model_selection` strings;
- missing/disabled options remain enumerable with exact source and placement
  data.

`BaseVideoModelRegistryEntry` fields are exact and must not be invented by a
worker:

| Field | Type / values | Purpose |
|---|---|---|
| `id` | `ModelSelectionID` | Value sent in `model_selection`. |
| `label` | `str` | UI label returned by model-options. |
| `group` | `str` | Model-options group; use `Base video model`. |
| `pipeline_family` | `LTXVideoGenPipelineFamily` (`"fast" | "full"`) | Filters Fast/Full popover. |
| `section` | `CatalogSection` (`"full" | "kijai" | "gguf" | "addons"`) | Existing frontend grouping. |
| `variant_group` | `str` | Stable grouping key. |
| `repo_id` | `str` | HuggingFace repo id. |
| `source_url` | `str` | `https://huggingface.co/{repo_id}` unless explicitly overridden. |
| `canonical_relative_path` | `str` | Required placement under models dir. |
| `expected_absolute_path` | `str` | Derived from models dir + canonical path. |
| `downloadable` | `bool` | Whether backend downloader can fetch it. |
| `download_cp_id` | `ModelCheckpointID | None` | CP spec used for downloader metadata, or `None` for scanner-only/manual placement. |
| `expected_size_bytes` | `int` | From CP spec when `download_cp_id` exists, otherwise registry metadata. |
| `remote_filename` | `str | None` | Remote filename override, if any. |
| `artifact_kind` | `ArtifactKind` | Scanner artifact kind. |
| `component_role` | `str` | Scanner role. For selectable bases use `base_diffusion_model`, `base_diffusion_model_fp8`, or `base_diffusion_model_gguf`. |
| `scanner_status` | derived | Scanner/model-options presence status. |
| `installed` | `bool` | `True` for `installed`, `wrong_folder_usable`, or `duplicate`; `False` for missing. |
| `preferred_path` | `str | None` | Scanner preferred path when installed/wrong-folder/duplicate; canonical expected path when missing is not a runtime path. |
| `transformer_path` | `str | None` | Runtime-selected transformer path; equals `preferred_path` when installed. |
| `transformer_format` | `"safetensors" | "gguf"` | Passed to resolver; do not infer solely from path in downstream handlers. |
| `base_family` | `"distilled" | "dev"` | Passed to resolver; do not infer solely from filename. |
| `runtime_readiness` | `"none" | "requires_active_profile_sidecars"` | Disabled-policy selector. |

Scanner evidence semantics:

- `installed`: use canonical path as `preferred_path`.
- `wrong_folder_usable`: `installed=true`; use scanner `preferred_path` for runtime;
  still return canonical placement path for user cleanup guidance.
- `duplicate`: `installed=true`; use scanner `preferred_path` for runtime; keep
  canonical placement path in response.
- missing: `installed=false`; `transformer_path=None`; option disabled with
  `MODEL_SELECTION_NOT_INSTALLED` / missing-model disabled reason and exact
  `expected_absolute_path`.

Resolver contract:

- `resolve_base_video_model_selection(models_dir, selection_id)` returns a
  `BaseVideoModelRegistryEntry`, not just a path.
- `PipelinesHandler._resolve_selection()` preserves the entry and uses its
  `transformer_path`, `transformer_format`, and `base_family`.
- `resolve_components(...)` receives explicit selection metadata. If the current
  function signature changes, use names:
  - `selected_transformer_path`
  - `selected_model_selection_id`
  - `selected_transformer_format`
  - `selected_base_family`
- Kijai distilled FP8 is a transformer-only safetensors selection but
  `base_family="distilled"`; it must not be treated as an official monolithic
  checkpoint purely because the path ends with `.safetensors`.
- Official dev/full safetensors is `base_family="dev"` and requires profile
  sidecars / runtime readiness if the runtime cannot execute it as a standalone
  monolith.

### Selectable IDs and families

Fast family (`pipeline_family = "fast"`):

| ID | Source | Canonical relative path | Installed behavior |
|---|---|---|---|
| `ltx-2.3-22b-distilled` | `Lightricks/LTX-2.3` | `diffusion_models/ltx-2.3-22b-distilled.safetensors` | Disabled if missing. |
| `ltx-2.3-22b-distilled-fp8-kijai-v3` | `Kijai/LTX2.3_comfy` | `diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors` | Enabled if file exists. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q2-k` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q2_K.gguf` | Enabled if file exists. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q3-k-s` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q3_K_S.gguf` | Enabled if file exists. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q3-k-m` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q3_K_M.gguf` | Enabled if file exists. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q4-k-s` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_S.gguf` | Enabled if file exists. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf` | Enabled if file exists. Current known installed file. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q5-k-s` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q5_K_S.gguf` | Enabled if file exists. |
| `ltx-2.3-22b-distilled-gguf-quantstack-q5-k-m` | `QuantStack/LTX-2.3-GGUF` | `gguf/QuantStack/LTX-2.3-GGUF/LTX-2.3-distilled-1.1/LTX-2.3-22B-distilled-1.1-Q5_K_M.gguf` | Enabled if file exists. |

Full family (`pipeline_family = "full"`):

| ID | Source | Canonical relative path | Installed behavior |
|---|---|---|---|
| `ltx-2.3-22b-dev` | `Lightricks/LTX-2.3` | `diffusion_models/ltx-2.3-22b-dev.safetensors` | Disabled if missing. |
| `ltx-2.3-22b-dev-gguf-q4-k-m` | `unsloth/LTX-2.3-GGUF` | `diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-Q4_K_M.gguf` | Enabled if file exists and runtime sidecars are ready. |
| `ltx-2.3-22b-dev-gguf-ud-q4-k-m` | `unsloth/LTX-2.3-GGUF` | `diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-UD-Q4_K_M.gguf` | Enabled if file exists and runtime sidecars are ready. |
| `ltx-2.3-22b-dev-gguf-q6-k` | `unsloth/LTX-2.3-GGUF` | `diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-Q6_K.gguf` | Enabled if file exists and runtime sidecars are ready. |
| `ltx-2.3-22b-dev-gguf-ud-q5-k-m` | `unsloth/LTX-2.3-GGUF` | `diffusion_models/unsloth/LTX-2.3-GGUF/ltx-2.3-22b-dev-UD-Q5_K_M.gguf` | Enabled if file exists and runtime sidecars are ready. |

Exact non-derived registry metadata:

| id | label | section | variant_group | repo_id | downloadable | download_cp_id | expected_size_bytes | remote_filename | artifact_kind | component_role | transformer_format | base_family | runtime_readiness |
|---|---|---|---|---|---:|---|---:|---|---|---|---|---|---|
| `ltx-2.3-22b-distilled` | `LTX-2.3 22B distilled (full precision)` | `full` | `ltx-2.3-distilled` | `Lightricks/LTX-2.3` | true | `ltx-2.3-22b-distilled` | 43000000000 | null | `diffusion_model` | `base_diffusion_model` | `safetensors` | `distilled` | `none` |
| `ltx-2.3-22b-distilled-fp8-kijai-v3` | `LTX-2.3 22B distilled FP8 (Kijai v3)` | `kijai` | `ltx-2.3-distilled-fp8` | `Kijai/LTX2.3_comfy` | false | null | 0 | null | `diffusion_model` | `base_diffusion_model_fp8` | `safetensors` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q2-k` | `LTX-2.3 22B distilled 1.1 GGUF — Q2_K (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 12408656544 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q3-k-s` | `LTX-2.3 22B distilled 1.1 GGUF — Q3_K_S (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 13959437984 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q3-k-m` | `LTX-2.3 22B distilled 1.1 GGUF — Q3_K_M (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 14702550688 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q4-k-s` | `LTX-2.3 22B distilled 1.1 GGUF — Q4_K_S (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 16706378400 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q4-k-m` | `LTX-2.3 22B distilled 1.1 GGUF — Q4_K_M (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 17763015328 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q5-k-s` | `LTX-2.3 22B distilled 1.1 GGUF — Q5_K_S (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 18542680736 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-distilled-gguf-quantstack-q5-k-m` | `LTX-2.3 22B distilled 1.1 GGUF — Q5_K_M (QuantStack)` | `gguf` | `ltx-2.3-distilled-gguf` | `QuantStack/LTX-2.3-GGUF` | false | null | 19388448416 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `distilled` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-dev` | `LTX-2.3 22B dev (full precision)` | `full` | `ltx-2.3-dev` | `Lightricks/LTX-2.3` | false | null | 43000000000 | null | `diffusion_model` | `base_diffusion_model` | `safetensors` | `dev` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-dev-gguf-q4-k-m` | `LTX-2.3 22B dev GGUF — Q4_K_M` | `gguf` | `ltx-2.3-dev-gguf` | `unsloth/LTX-2.3-GGUF` | true | `ltx-2.3-22b-dev-gguf-q4-k-m` | 14326856736 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `dev` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-dev-gguf-ud-q4-k-m` | `LTX-2.3 22B dev GGUF — UD Q4_K_M` | `gguf` | `ltx-2.3-dev-gguf` | `unsloth/LTX-2.3-GGUF` | true | `ltx-2.3-22b-dev-gguf-ud-q4-k-m` | 16506438688 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `dev` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-dev-gguf-q6-k` | `LTX-2.3 22B dev GGUF — Q6_K` | `gguf` | `ltx-2.3-dev-gguf` | `unsloth/LTX-2.3-GGUF` | true | `ltx-2.3-22b-dev-gguf-q6-k` | 17774906400 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `dev` | `requires_active_profile_sidecars` |
| `ltx-2.3-22b-dev-gguf-ud-q5-k-m` | `LTX-2.3 22B dev GGUF — UD Q5_K_M` | `gguf` | `ltx-2.3-dev-gguf` | `unsloth/LTX-2.3-GGUF` | true | `ltx-2.3-22b-dev-gguf-ud-q5-k-m` | 18274719776 | null | `gguf` | `base_diffusion_model_gguf` | `gguf` | `dev` | `requires_active_profile_sidecars` |

`source_url` is always `https://huggingface.co/{repo_id}`. `group` is always
`Base video model`. `expected_absolute_path`, `scanner_status`, `installed`,
`preferred_path`, and `transformer_path` are derived from `models_dir` plus
scanner evidence using the scanner evidence semantics above.

### Backend files / functions to touch

Only these backend files are in scope:

- `backend/api_types.py`
  - Add `ModelSelectionID: TypeAlias = str`.
  - Change `ModelSelectionOption.id` to `ModelSelectionID`.
  - Change `GenerateVideoRequest.model_selection` to `ModelSelectionID | None`.

- `backend/runtime_config/model_download_specs.py`
  - Keep downloadable CP specs for downloader metadata.
  - Add or expose metadata needed by the base-video registry for official
    downloadable base models.
  - Do **not** make `SELECTABLE_BASE_VIDEO_CP_IDS` the source for generation
    options.

- `backend/services/model_scanner.py`
  - Derive scanner-known Kijai/QuantStack base-video artifacts from the unified
    registry instead of keeping separate scanner-only definitions that cannot be
    selected.
  - Scanner remains read-only; no downloads, moves, repairs, or folder creation.

- `backend/services/base_video_model_registry.py` (new file)
  - Define `BaseVideoModelRegistryEntry`.
  - Define `iter_base_video_model_entries(models_dir: Path) -> list[BaseVideoModelRegistryEntry]`.
  - Define `resolve_base_video_model_selection(models_dir: Path, selection_id: ModelSelectionID) -> BaseVideoModelRegistryEntry`.
  - Entry fields must match the full `BaseVideoModelRegistryEntry` contract
    above exactly: `id`, `label`, `group`, `pipeline_family`, `section`,
    `variant_group`, `repo_id`, `source_url`, `canonical_relative_path`,
    `expected_absolute_path`, `downloadable`, `download_cp_id`,
    `expected_size_bytes`, `remote_filename`, `artifact_kind`,
    `component_role`, `scanner_status`, `installed`, `preferred_path`,
    `transformer_path`, `transformer_format`, `base_family`,
    `runtime_readiness`.

- `backend/handlers/models_handler.py`
  - `get_model_selection_options()` enumerates `iter_base_video_model_entries(...)`.
  - `installed` comes from registry/scanner-compatible canonical path evidence,
    not only `is_cp_downloaded(...)`.
  - Existing dev/full runtime-readiness gate remains for Full GGUF entries.

- `backend/handlers/pipelines_handler.py`
  - `_resolve_selection()` accepts `ModelSelectionID` and delegates to
    `resolve_base_video_model_selection(...)`.
  - It preserves the selected entry metadata, not only the selected path.
  - It passes `selected_transformer_path`, `selected_model_selection_id`,
    `selected_transformer_format`, and `selected_base_family` downstream to
    `resolve_components(...)` or an equivalent explicitly named selection bundle.
  - It passes selection runtime readiness (name: `selected_runtime_readiness`) or
    an equivalent explicit field so `resolve_components(...)` can decide whether
    selected safetensors is a true monolith or a transformer-only/split selection.
  - No downstream filename/path inference is allowed for selected family/format.
  - Unknown selected IDs still reject with `UNSUPPORTED_MODEL_SELECTION`.
  - Missing selected IDs still reject with `MODEL_SELECTION_NOT_INSTALLED` and
    include the exact expected absolute path.

- `backend/services/ltx_components.py`
  - `resolve_components(...)` keeps the selected-transformer override behavior.
  - If `selected_cp_id` is renamed for type correctness, the new name must be
    `selected_model_selection_id` everywhere it is used.
  - For selected safetensors entries, sidecars are cleared only when
    `selected_runtime_readiness == "none"`. Kijai FP8 and official dev/full
    selections have `selected_runtime_readiness == "requires_active_profile_sidecars"`
    and must preserve profile sidecars.

- `backend/handlers/text_handler.py`
  - Text/prompt cache model identity must accept `ModelSelectionID`, not only
    `ModelCheckpointID`.
  - Non-CP registry IDs (Kijai/QuantStack/official dev) must resolve identity via
    `resolve_base_video_model_selection(...)`, not `resolve_model_path(...)`.

- `backend/handlers/video_generation_handler.py`
  - Family mismatch validation uses the registry entry's `pipeline_family`.
  - `model="fast"` rejects Full-family selected IDs.
  - `model="full"` rejects Fast-family selected IDs.

Oracle implementation-review blockers that this slice must satisfy:

- Kijai FP8 and official dev safetensors must not be treated as monolithic only
  because they end in `.safetensors`; runtime readiness controls sidecar
  preservation.
- Runtime readiness must require distilled LoRA only for `base_family == "dev"`,
  not for Fast-family Kijai/QuantStack distilled entries.
- Text-cache identity must be registry-aware for non-CP `ModelSelectionID` values.
- Scanner registry artifact derivation must include every non-CP base-video
  registry entry not covered by CP specs, including `ltx-2.3-22b-dev`.

Follow-up piping correction from live Fast GGUF/Kijai Generate:

- `resolve_components(...)` must use `selected_runtime_readiness` to decide the
  actual builder checkpoint tuple, not only sidecar metadata clearing.
- If `selected_runtime_readiness == "requires_active_profile_sidecars"`, then
  `checkpoint_paths_for_filtered_builders` must be:
  `(selected_transformer_path, text_projection, embeddings_connector, video_vae, audio_vae)`
  with falsey entries filtered, regardless of `selected_base_family` or
  `selected_transformer_format`.
- If `selected_runtime_readiness == "none"`, then builder paths remain the
  single selected transformer path.
- This is required for Fast-family QuantStack distilled GGUF and Kijai distilled
  FP8 selections; both are selectable Fast variants but still need active profile
  sidecars for runtime piping.
- Add at most one focused resolver test in `backend/tests/test_ltx_components.py`
  named `test_selected_fast_sidecar_entries_keep_builder_sidecars`. No broad test
  rewrites and no full test modules.

### Frontend files / functions to touch

Frontend changes are required because the API type changes from `ModelCheckpointID`
to `ModelSelectionID`.

- `frontend/generated/backend-openapi.json`
- `frontend/generated/backend-openapi.ts`
  - Regenerate after backend schema changes.

- `frontend/lib/model-selection.ts`
  - Consume `ModelSelectionID` string ids from generated types.
  - Keep grouping by backend-owned `section`, `variant_group`, and
    `pipelineFamily`; do not infer install paths or support.

- `frontend/hooks/use-model-selection-options.ts`
  - Type-only update if generated response type changes.

- `frontend/views/GenSpace.tsx`
  - Keep current family filtering by selected `settings.model`.
  - Keep clearing incompatible `modelSelection` on family change.
  - Do not invent UI behavior beyond rendering backend-owned options.

### Validation commands for this slice only

Test scope must stay smaller than implementation scope. Tests prove the changed
contract only; they must not become a broad fixture-rewrite project.

Allowed test changes:

- `backend/tests/test_models.py`
  - Add/adjust one focused test named
    `test_model_options_use_base_video_registry_fast_and_full_families`.
  - It should verify model-options include Fast-family Kijai/QuantStack entries
    and Full-family dev entries with correct installed/disabled status from fake
    filesystem evidence.
- `backend/tests/test_generation.py`
  - Add/adjust one focused test named
    `test_generate_rejects_unknown_model_selection_id`.
  - Family-mismatch coverage must be folded into this focused test or replace it;
    it must not create a fourth test for this slice.
- `backend/tests/test_model_scanner.py`
  - Add/adjust one focused test named
    `test_scanner_base_video_registry_artifacts_are_selectable`.
  - It should verify scanner-recognized Kijai/QuantStack artifacts carry the same
    IDs/metadata used by model-options.

Prohibited test work:

- no full backend suite;
- no unrelated test cleanup;
- no broad fake-service rewrites;
- no changing tests to match a wrong implementation;
- if more than the three focused tests above are needed, stop and report a
  revised plan before editing further.

Validation commands:

- `rtk pnpm openapi:generate`
- `rtk pnpm backend:test -- tests/test_models.py -k test_model_options_use_base_video_registry_fast_and_full_families`
- `rtk pnpm backend:test -- tests/test_generation.py -k test_generate_rejects_unknown_model_selection_id`
- `rtk pnpm backend:test -- tests/test_model_scanner.py -k test_scanner_base_video_registry_artifacts_are_selectable`
- `rtk pnpm typecheck:py`
- `rtk pnpm typecheck:ts`

No substitutes are allowed for worker validation. If any listed command fails
because the wrapper misroutes arguments or the environment is inconvenient, the
worker must stop and report the validation blocker. The orchestrator alone may
choose an alternate command. Running whole test files, "affected tests", or a
larger subset such as all touched test modules is prohibited.

### Worker boundaries / prohibitions

- Worker receives only the files/functions above.
- Worker must read `AGENTS.md`, `codemap.md`, and relevant folder codemaps before
  any search.
- Worker must not use broad repo search unless a listed file contradicts this
  plan and the worker stops with a revised plan.
- Worker must not run `git stash`, `git reset`, `git checkout`, `git restore`,
  `git clean`, broad staging, commit, amend, or push.
- Worker must not run the full backend suite.
- Worker must not fix unrelated tests.
- If implementing this requires files outside the scope above, stop and report a
  revised plan before editing.

## Agreed architecture

### Frontend — prompt-box compact Model popover

- A compact **Model** chip/popover anchored in the prompt box.
- Default label form: `Model: Fast · GGUF Q4_K_M` (variant summary), e.g.
  `Model: <profile family> · <variant>`.
- Options are **grouped** (by family / quant / source) and rendered purely from
  the backend-owned options response (see below). The frontend must **not**
  infer which options are supported.
- **Output controls are grouped separately and added later**, not bundled into
  this popover. The popover is model-selection only for now.

### Request scope

- Selection is carried in a request-scoped **`model_selection`** field on
  generation requests.
- **The active profile is the fallback only when `model_selection` is absent.**
  If `model_selection` is omitted (`None`/empty), the request resolves to the
  active profile. Live selection never mutates the user's active profile.
- **No silent fallback on bad/unsupported selection.** If `model_selection` is
  **present but** unknown, malformed, disabled by the backend options
  endpoint, or unsupported for the workflow, the request **must reject with a
  clear error** (not silently fall back to the active profile). Silent
  fallback would mask bugs and let stale/disabled options succeed
  nondeterministically. Only an **absent** `model_selection` falls back.

### Backend-owned model-options endpoint

- A backend endpoint returns the **workflow-aware** model options for the
  current request context (workflow kind + installed assets): each option
  carries its id, label, grouping, and an explicit **disabled reason** when an
  option is not selectable (e.g. missing asset, unsupported for this workflow).
- **Options are derived from the configured models folder / scanner** — i.e.
  the same source of truth used for profile scanning and status. The endpoint
  must not hardcode or speculate options; it reflects what is actually
  installed/resolvable.
- **Each option must include, when relevant:**
  - repo / **source link** (where the asset comes from),
  - canonical **relative path** within the models folder,
  - the **expected absolute placement path** for a missing asset (so a missing
    option can show the user exactly where to install it, consistent with the
    missing-model UI policy),
  - a **downloadable flag** (whether the backend can fetch it).
- **Missing options must still be enumerable** (with disabled reason +
  source/placement info) so the frontend can surface "install this to enable
  X" rather than hiding it.
- The endpoint is **read-only**: no download, no mutation, no state change. It
  describes options only.
- **The frontend must not infer.** It renders exactly what the backend declares
  — options, grouping, and disabled reasons verbatim — so we never fake support
  and never silently block.

### Cache-key hardening (prerequisite for broad live switching)

- The pipeline cache key must include the resolved `model_selection` so that a
  per-request variant switch rebuilds/serves the correct pipeline.
- Harden the cache key **before** enabling broad live switching, otherwise
  stale pipelines get served across switches.

## Implementation phases (in order)

1. **Backend contract / options endpoint.** Define the request/response types
   for `model_selection` and the model-options endpoint. Workflow-aware option
   list with disabled reasons. No frontend yet.
2. **Request-scoped resolver context.** Thread `model_selection` through the
   resolver so a generation resolves components from the selection — falling
   back to the active profile **only when `model_selection` is absent**, and
   rejecting clearly when it is present but bad/unsupported (see Request scope).
   Update cache key to include the selection.
3. **T2V / I2V first.** Ship live model selection for text-to-video and
   image-to-video only. Validate end-to-end with fake-service tests plus a
   scoped live smoke.
4. **Frontend popover.** Build the prompt-box Model popover consuming the
   backend options endpoint. Render grouping + disabled reasons verbatim. Wire
   `model_selection` into the generation request.
5. **A2V / IC-LoRA / retake later.** Extend to audio-to-video, IC-LoRA, and
   retake workflows once the core path is proven and cache-key hardening has
   held up under the T2V/I2V rollout.

## Pitfalls (from oracle review — must avoid)

- **Do not overload the legacy `model` field.** Introduce a distinct
  `model_selection` rather than reusing/repurposing the existing `model` field,
  to avoid ambiguity and silent behaviour changes for callers that still send
  the legacy field.
- **No silent fallback on bad/unsupported selection.** Only an **absent**
  `model_selection` resolves to the active profile. A present-but-unknown,
  malformed, disabled, or workflow-unsupported `model_selection` must **reject
  with a clear error**, never silently fall back (see Request scope).
- **Do not let the frontend infer support.** Support is declared by the backend
  options endpoint only. The frontend must not compute "is this workflow
  supported" locally; it renders the options and disabled reasons as given.
- **Cache correctness.** A model switch that resolves to a different pipeline
  configuration must not be served from a cache entry built for another
  configuration. The cache key must fully capture the resolved selection
  (transformer, format, text encoder, quant, etc.), not just an opaque id.
- **Text-encoder / prompt-cache semantics.** Prompt/text-encoding caches are
  keyed by prompt + enhancer flags; if a model selection changes the text
  encoder (e.g. GGUF vs full Gemma), the text cache must not leak across
  selections. Audit the text-cache key alongside the pipeline cache key.
- **Settings persistence.** `model_selection` is **request-scoped**, not a
  persisted setting. Do not write it into the user's profile/settings. The
  active profile remains the persisted source of truth.

## Parallel lanes

Live model selection has natural parallel lanes once the contract is agreed;
**sequence same-file edits** and respect the phase ordering:

- **Lane A — backend contract / options endpoint:** define `model_selection`
  and the options response in `backend/api_types.py`, and add the models
  routes/handlers for the read-only options endpoint (workflow-aware, scanner
  -derived, with source link / placement path / downloadable flag).
- **Lane B — resolver / cache-key work:** thread `model_selection` through
  `backend/services/ltx_components.py`, `backend/handlers/pipelines_handler.py`,
  `backend/handlers/text_handler.py`, and the generation handlers; update the
  pipeline + text cache keys; implement the reject-on-bad-selection rule.
- **Lane C — frontend popover (design-only until the endpoint exists):** draft
  the prompt-box Model popover UX (grouping, disabled reasons verbatim). **Do
  not wire it** until Lane A's endpoint is live — the frontend must not infer.
- **Lane D — tests / OpenAPI (after the contract stabilizes):** fake-service
  tests for resolver + cache key, OpenAPI consistency check, and (later) the
  scoped T2V/I2V live smoke.
- **Ordering constraints:** Lane A (contract) must stabilize before Lane D's
  OpenAPI/test work finalizes; Lane C may design in parallel but wires only
  after Lane A; Lane B depends on Lane A's type definitions.

## Stop conditions

- Stop if the cache key cannot fully express the resolved selection — resolve
  before enabling live switching, or keep it gated to profiles only.
- Stop if a present-but-bad/unsupported `model_selection` would silently fall
  back to the active profile — it must reject clearly (see Request scope).
- Stop if the frontend would need to infer support for any workflow — the
  backend options endpoint must cover it instead.
- Stop if T2V/I2V smoke shows stale-pipeline or stale-text-cache behaviour on
  switch — do not extend to A2V/IC-LoRA/retake until cache correctness holds.
