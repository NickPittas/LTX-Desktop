# 03 — Live Model Selection (Request-Scoped Profile Switching)

> Step 4 of the [current plan](README.md).

Goal: let the user pick a model variant **per generation** from the prompt box,
without leaving their current profile as the default/advanced fallback. This is
the latest agreed architecture; it supersedes any older "global model switch"
framing in the archived plans.

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
