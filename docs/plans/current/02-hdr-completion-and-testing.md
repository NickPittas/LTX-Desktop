# 02 — HDR Closeout, Validation, and Status Cleanup

> Step 2 of the [current plan](README.md).

Goal: the HDR IC-LoRA workflow is **implemented end-to-end** in the backend/UI
generation path (scene-embedding prompt-encoder swap, LogC3→linear postprocess,
EXR primary output, HDR UI workflow) and is now **validated** — the real-asset
end-to-end smoke **passed** (2026-06-29), so HDR is `supported`. This step is
**complete**; what remains is the Step 3 commit decision. HDR is **not** "just
another LoRA enablement" — it has its own required assets (incl. a
scene-embedding support asset) and its own decode/postprocess path, reusing the
primary EXR output plumbing.

## Current state (HDR implemented, validated — end-to-end smoke PASSED)

HDR is **not** wholly unimplemented. The backend and UI generation path exists,
is wired, and is now **validated end-to-end** (real-asset smoke passed). The
remaining item is the **commit decision** (Step 3), not building or validating
the path.

**Already implemented:**

- **Backend HDR handler.** `_generate_hdr()` exists in
  `backend/handlers/ic_lora_handler.py` and is dispatched when
  `workflow == "hdr"`. **⚠️ Transient working-tree state (obsolete re-gate):**
  as of 2026-06-29 an "option (a) temporary re-gate" briefly put `hdr` back in
  `_UNAVAILABLE_WORKFLOWS`, making the dispatch unreachable. That re-gate is a
  **wrong-turn, superseded by user clarification** — Step 2 removes `hdr` from
  `_UNAVAILABLE_WORKFLOWS` to re-enable the dispatch as part of finishing HDR
  to `supported`. `hdr_scene_embeddings` stays in `_UNAVAILABLE_WORKFLOWS`
  because it is a **support asset**, not a standalone adapter.
- **Scene-embedding prompt-encoder swap + postprocess.**
  `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py` swaps
  `pipeline.prompt_encoder` for an HDR injector that supplies pre-computed
  scene embeddings (video-only; audio context is intentionally dropped), and
  applies the LogC3 → linear HDR decode/postprocess after VAE decode. The
  original encoder is restored in a `finally` block so non-HDR calls are
  unaffected.
- **EXR output plumbing.** HDR forces a linear scene-referred EXR primary
  output (`OutputFormat.EXR_ZIP_HALF`) and passes EXR directory/sequence paths
  through unchanged (no transcoding).
- **Tests.** HDR coverage exists:
  `backend/tests/test_ic_lora.py` (HDR endpoint tests were **temporarily**
  flipped to the unavailable-400 re-gate — this is part of the obsolete re-gate
  and must be **restored to success tests** as HDR reaches `supported`;
  `hdr_scene_embeddings`-as-standalone-adapter → 400 stays),
  `backend/tests/test_hdr_utils.py`, and
  `backend/tests/test_hdr_scene_embeddings.py` (lower-level utility coverage).

**Remaining work (finish HDR to `supported`):**

- **Gate/status — finish to `supported` (option b) is the active goal.** ✅
  Largely landed (2026-06-29): backend handler, UI, `model_scanner.py`, and
  `model_resolver.py` (plus their tests) are flipped to `supported`, with
  `hdr_scene_embeddings` kept as a support asset. The earlier "✅ RESOLVED via
  temporary re-gate (option a)" entry is **obsolete / superseded** by user
  clarification (2026-06-29); the re-gate is not the commit state.
- **Validation / live smoke evidence (prerequisite for `supported`) — ✅ PASSED
  (2026-06-29).** Real HDR generation succeeded (after the audio-context fix):
  the endpoint returned **200**; output was **9 linear EXR frames** at
  `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr`;
  the sidecar SDR proxy
  `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr_proxy.mp4`
  is **H.264, yuv420p, BT.709, 512×512, 9 frames**. Prior validation also
  passed: **238 targeted backend tests**, `pnpm typecheck`, and
  `pnpm build:frontend`. The hard prerequisite for `supported` is satisfied.
- **SDR proxy policy (prerequisite for `supported`).** ✅ Implemented
  (2026-06-29) — the sidecar SDR proxy tonemap is written alongside the linear
  EXR so browser `<video>` playback keeps working while linear EXR frames are
  preserved losslessly. Oracle's **revised** recommendation treated this as a
  hard prerequisite for `supported`; satisfied (and confirmed by the smoke
  proxy above).
- **Commit decision.** Commit is now **unblocked** — HDR is fully `supported`
  (SDR proxy + real smoke both passed). Step 3 may proceed subject to the
  enumerated staging rules and explicit confirmation (no push without it). The
  obsolete re-gate is not a commit state.

## HDR `supported` completion checklist

HDR is `supported` only when **all** of the following are true. This is the gate
for the Step 3 commit; the obsolete "temporary re-gate" is not an acceptable
state.

- [x] **Backend enabled** — `hdr` removed from `_UNAVAILABLE_WORKFLOWS` in
   `backend/handlers/ic_lora_handler.py`; `_generate_hdr()` dispatch reachable;
   `hdr_scene_embeddings` still returns 400 as a standalone adapter.
   ✅ Lane A completed (2026-06-29).
- [x] **SDR proxy generated** — a tonemapped SDR proxy is written alongside the
   linear EXR primary so browser `<video>` playback works; exposure affects the
   proxy only, never the linear EXR. ✅ Lane B completed (2026-06-29).
- [x] **Scanner/resolver `supported`** — `model_scanner.py` (`_GATED_ADAPTER_IDS`)
   and `model_resolver.py` (`_GATED_ROLES`, `hdr_status`) flipped so HDR reads
   `supported`/not-gated; `hdr_scene_embeddings` remains a support asset;
   `tests/test_model_scanner.py` + `tests/test_model_resolver.py` updated to
   assert `supported`. ✅ Lane C completed (2026-06-29).
- [x] **UI previews proxy + reveals EXR** — `frontend/components/ICLoraPanel.tsx`
   `hdr` adapter restored to a real workflow; the UI previews the SDR proxy in
   `<video>` and surfaces/reveals the EXR folder/sequence as the primary HDR
   output. ✅ Lane D completed (2026-06-29).
- [x] **Tests / smoke pass** — targeted tests restored and green
   (`tests/test_ic_lora.py` HDR endpoint success tests restored — no longer
   400-unavailable; `test_hdr_utils.py`, `test_hdr_scene_embeddings.py`,
   `test_media_encoder.py`, `test_model_scanner.py`, `test_model_resolver.py`
   green; `typecheck` and `build:frontend` pass). ✅ Known validation
   (2026-06-29): **238 targeted backend tests passed**; full `typecheck`
   passed; `build:frontend` passed. ✅ **Real-asset HDR smoke PASSED
   (2026-06-29)** — endpoint returned **200**; **9 linear EXR frames** at
   `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr`;
   SDR proxy
   `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr_proxy.mp4`
   is **H.264, yuv420p, BT.709, 512×512, 9 frames** (succeeded after the
   audio-context fix). **All checklist items satisfied — HDR is `supported`.**

   > All five checklist items are `[x]`. HDR is `supported`; Step 2 is complete;
   > Step 3 (commit) is unblocked (still subject to explicit confirmation).

## Closeout concerns (the path exists; validate and reconcile)

HDR cannot ship as a fake standard IC-LoRA run. The pieces below are
**implemented**; each has a specific closeout/validation concern before the
gate/status mismatch can be resolved toward `supported`.

1. **Scene embedding injection — implemented; validate.**
   - The HDR adapter uses **two** assets: the IC-LoRA weights **and** a
     separate `kind="embeddings"` scene embedding file. The pipeline now
     injects these via the prompt-encoder swap (video-only).
   - **Closeout:** confirm profile validation still requires **both**
     `ic_lora_hdr` and `ic_lora_hdr_scene_embeddings`, and that the injector is
     not stacking the embedding as a LoRA. Keep `hdr_scene_embeddings` a
     **support asset** — never selectable as a standalone adapter (the 400 test
     must keep passing).

2. **HDR decode / postprocess — implemented; validate linearity.**
   - The LogC3 → linear HDR decode postprocess runs after VAE decode, before
     output persistence, producing `hdr_linear` (linear HDR float → EXR).
   - **Closeout:** confirm the postprocess emits **linear** HDR tensors (not
     already-tonemapped SDR). Exposure default `~7.1` EV must affect **SDR
     preview only**, never the linear EXR. Half-precision default true for EXR.
     If inspection shows the path only returns already-tonemapped SDR tensors,
     **stop** and fix the deeper decode layer rather than exporting fake HDR.

3. **Linear EXR primary output — plumbed; depends on primary-output work.**
   - HDR forces `OutputFormat.EXR_ZIP_HALF` and passes EXR sequence/directory
     paths through unchanged. The **generic** output-format plumbing
     (output-format field, handler output path, EXR sequence writer, ZIP/folder
     convention) belongs to the primary EXR/MOV output plan; HDR reuses it.
   - **Closeout:** confirm the EXR writer dependency (OpenEXR/pyexr) and the
     generic EXR sequence writer are actually available end-to-end (the
     encode-output chokepoint in `ltx_pipelines.utils.media_io` /
     `backend/services/ltx_pipeline_common.py`). Do **not** build a second
     competing EXR/export framework inside the HDR codepath.

4. **SDR proxy / preview strategy — active prerequisite for `supported`.**
   - EXR is not playable in a browser `<video>`. The SDR proxy tonemapping is
     currently `None` (deferred) to avoid emitting an incorrect proxy. Implement
     the SDR proxy/preview sidecar policy so existing UI playback keeps working
     while the primary HDR EXR frames are preserved losslessly. Oracle's
     **revised** recommendation treats this as a hard prerequisite for
     `supported`.
   - Keep exposure scoped to SDR preview; never bake exposure into linear EXR.

5. **Resolve the gate/status — finish HDR to `supported` (option b).**
   - **The mismatch:** the handler (`ic_lora_handler.py`) and UI
     (`ICLoraPanel.tsx`) HDR generation path is implemented, but the working
     tree currently carries an obsolete "option (a) temporary re-gate" (`hdr`
     back in `_UNAVAILABLE_WORKFLOWS`, UI `hdr` `unavailable`, endpoint tests
     expecting 400), while `model_scanner.py`
     (`_GATED_ADAPTER_IDS = {"hdr", "hdr_scene_embeddings"}`) and
     `model_resolver.py` (`_GATED_ROLES`, `hdr_status="gated"`) still mark HDR
     **gated**. None of these is the target state.
   - **Target state (option b) — fully `supported`:** after the validation/smoke
     strategy below passes and the SDR proxy policy is implemented, update
     `model_scanner.py`, `model_resolver.py`, and the scanner/resolver tests so
     HDR reads `supported`/not-gated; remove `hdr` from
     `_UNAVAILABLE_WORKFLOWS`; restore the UI `hdr` workflow (SDR-proxy preview
     + EXR reveal); restore the HDR endpoint success tests — while keeping
     `hdr_scene_embeddings` as a **support asset** (still returns 400 as a
     standalone adapter).
   - This is the single gating decision shared with
     `01-finish-uncommitted-code.md` task 3; do not resolve it in two places at
     once.
   - **✳️ DECISION (revised 2026-06-29, user clarification): option (b) —
     fully finish HDR to `supported` before commit.** Oracle's **revised**
     recommendation: `supported` is gated on two prerequisites — (i) an SDR
     proxy is generated alongside the linear EXR, and (ii) the real-asset HDR
     smoke passes (linear EXR frame stats + tonemap sanity).
   - **⚠️ OBSOLETE / superseded — kept for history only:** "option (a)
     temporary re-gate" (2026-06-29, Lane D). Its rationale was that the HDR
     path lacked real-smoke evidence and an SDR proxy policy while
     scanner/resolver still marked HDR `gated`, so it re-gated handler/UI to
     match and deferred un-gating. **This is a wrong-turn and is not the active
     direction.** The working-tree re-gate artifacts are to be reverted as part
     of reaching `supported`. Scanner/resolver flip to `supported` happens as
     part of this step (not left `gated`).

## Test / smoke strategy

- **Backend integration tests (exist; restore success path):** IC-LoRA HDR tests
  cover profile validation requiring both HDR assets, scene-embedding injection
  (fake pipeline recording the call), missing-scene-embeddings → 400,
  prompt-required, and `hdr_scene_embeddings`-as-standalone-adapter → 400. The
  HDR **success** tests were temporarily flipped to 400 by the obsolete re-gate
  — **restore them** to the success path (incl. HDR success with a Kijai/GGUF
  profile) as part of reaching `supported`. Keep all green.
- **Gate/status reconciliation tests:** whichever decision is taken in closeout
  concern 5, the scanner/resolver tests must assert the **same** state the
  handler/UI expose. If flipping to `supported`, update
  `tests/test_model_scanner.py` and `tests/test_model_resolver.py` (currently
  asserting `gated`) to assert supported, while keeping the
  `hdr_scene_embeddings`-as-standalone-adapter 400 behaviour.
- **Postprocess unit coverage:** HDR decode/postprocess must produce a linear
  HDR tensor and a tonemapped SDR tensor with exposure affecting only SDR.
- **EXR writer test:** confirm the EXR sequence writer (OpenEXR/pyexr) is
  available end-to-end — exercise it with a small synthetic linear tensor.
- **Live smoke (scoped runner, not parent): ✅ PASSED (2026-06-29)** — a single
  HDR generation against real assets confirmed the output is a linear EXR frame
  sequence (9 frames) plus a tonemapped SDR proxy. Evidence: endpoint **200**;
  EXR dir `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr`;
  SDR proxy `..._exr_proxy.mp4` = H.264 / yuv420p / BT.709 / 512×512 / 9 frames
  (succeeded after the audio-context fix). Tonemap/proxy sanity captured; the
  linear EXR primary is preserved losslessly.
- **Dependency ordering:** the generic EXR primary output plumbing must be
  available and validated **before** the HDR smoke is meaningful. Do not run
  the HDR smoke against an MP4-transcode path.

## Exact validation commands

Run via the RTK-wrapped pnpm pin. Run the full set when pursuing option (b)
(flip to `supported`); run the targeted handler/media set for any HDR touch:

```bash
# Targeted HDR / encoder / scanner / resolver tests
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_ic_lora.py
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_hdr_utils.py
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_hdr_scene_embeddings.py
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_media_encoder.py
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_model_scanner.py
rtk npx --yes pnpm@10.30.3 backend:test -- tests/test_model_resolver.py

# Type checks
rtk npx --yes pnpm@10.30.3 typecheck:py
rtk npx --yes pnpm@10.30.3 typecheck:ts

# Frontend build — only required if ICLoraPanel.tsx / HDR UI changed
rtk npx --yes pnpm@10.30.3 build:frontend
```

CI gate (must pass before commit): `pnpm typecheck` + `pnpm backend:test` +
frontend Vite build (the latter only if UI changed).

## Parallel lanes

Where helpful, run HDR closeout as three parallel lanes; **sequence same-file
edits** (do not edit `ic_lora_handler.py` / `model_scanner.py` /
`model_resolver.py` / `ICLoraPanel.tsx` from two lanes at once):

- **Lane A — HDR backend / status cleanup:** reconcile the gate/status state in
  `ic_lora_handler.py` (only if re-gating), `model_scanner.py`,
  `model_resolver.py`, and the scanner/resolver tests; keep
  `tests/test_ic_lora.py` / `tests/test_hdr_utils.py` green.
- **Lane B — HDR UI validation / messaging:** validate `ICLoraPanel.tsx`
  (EXR-folder output messaging, no-input prompt-only flow) matches whichever
  gate/status decision Lane A lands on.
- **Lane C — live smoke / evidence collection:** the scoped real-asset HDR
  generation + EXR/SDR evidence capture (independent of source edits, but its
  result gates the `supported` decision).
- **Sequence same-file edits:** the gate/status decision touches UI +
  scanner/resolver together — finish Lane A's decision before Lane B reflects
  it, and do not run them as concurrent edits to the same logical change.

### Lane results (2026-06-29)

For this execution the closeout was run as four implementation lanes (A–D) plus
the live-smoke lane. All lanes passed; HDR is `supported`.

- **Lane A — HDR backend / API enabled to `supported` path: ✅ completed
  (2026-06-29).** `hdr` removed from `_UNAVAILABLE_WORKFLOWS`;
  `_generate_hdr()` dispatch reachable; `hdr_scene_embeddings` still 400 as a
  standalone adapter. HDR endpoint success tests restored.
- **Lane B — SDR proxy encoder implemented: ✅ completed (2026-06-29).** The
  tonemapped SDR proxy is written alongside the linear EXR primary; exposure
  affects the proxy only, never the linear EXR. Browser `<video>` playback
  keeps working; linear EXR frames preserved losslessly.
- **Lane C — Scanner/resolver `supported` flip + tests: ✅ completed
  (2026-06-29).** `model_scanner.py` (`_GATED_ADAPTER_IDS`) and
  `model_resolver.py` (`_GATED_ROLES`, `hdr_status`) flipped so HDR reads
  `supported`/not-gated; `hdr_scene_embeddings` remains a support asset;
  `tests/test_model_scanner.py` + `tests/test_model_resolver.py` updated to
  assert `supported`.
- **Lane D — UI enabled with proxy preview + EXR reveal: ✅ completed
  (2026-06-29).** `frontend/components/ICLoraPanel.tsx` `hdr` adapter restored
  to a real workflow; UI previews the SDR proxy in `<video>` and reveals the
  EXR folder/sequence as the primary HDR output.
- **Prior validation (2026-06-29): ✅ passed** — **238 targeted backend tests
  passed**; full `typecheck` passed; `build:frontend` passed.
- **Lane (live smoke / final validation): ✅ PASSED (2026-06-29).** Real HDR
  generation succeeded after the audio-context fix: endpoint returned **200**;
  **9 linear EXR frames** at
  `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr`;
  SDR proxy
  `/home/npittas/.local/share/LTXDesktop/outputs/hdr_20260629_145849_645d0893_exr_proxy.mp4`
  is **H.264, yuv420p, BT.709, 512×512, 9 frames**. This was the last gate for
  `supported`; it is satisfied, Step 2 is complete, and Step 3 (commit) is
  unblocked subject to explicit confirmation.

## Stop conditions

- Stop if inspection shows the postprocess only returns already-tonemapped SDR
  tensors (no linear HDR decode) — do not export fake HDR; fix the deeper
  decode layer.
- Stop if the generic primary EXR output plumbing / EXR writer is not actually
  available end-to-end — HDR EXR output depends on it.
- Do **not** flip scanner/resolver to `supported` until the test/smoke strategy
  above passes and the SDR proxy policy is defined.
- Do **not** commit the half-gated state (handler/UI enabled while
  scanner/resolver mark HDR `gated`) — resolve closeout concern 5 first.
- Do **not** commit the obsolete temporary re-gate state — HDR must be fully
  `supported` (see the completion checklist above) before the Step 3 commit.
- Do **not** broaden HDR to inpainting — HDR is single-stage V2V-style.
