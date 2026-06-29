# HDR IC-LoRA Architecture Plan

## Status
architecture-plan-ready

## Scope
Plan HDR IC-LoRA support only. No implementation packet yet. Source edits should wait until this architecture is reviewed and the primary EXR/MOV output direction is decided.

## Source Evidence Used
- `AGENTS.md`: repo architecture, subagent workflow, backend/frontend constraints.
- `subagent-artifacts/hdr-implementation-locator.md`: HDR IC-LoRA gates, model specs, workflow reference, backend/frontend/test anchors.
- `subagent-artifacts/primary-output-formats-locator.md`: primary MOV/EXR output architecture and encode bottleneck.

## Architecture Summary
HDR IC-LoRA is not just another LoRA enablement. The HDR adapter has two required assets: the IC-LoRA weights and a separate `kind="embeddings"` scene embedding file. Current IC-LoRA infrastructure stacks only LoRA-like adapters into `ICLoraPipeline`; it does not inject conditioning embeddings. HDR also requires a decode/postprocess path equivalent to ComfyUI `LTXVHDRDecodePostprocess`, producing linear HDR EXR frames plus a tonemapped SDR preview. Therefore HDR should be implemented after, or tightly coordinated with, the primary EXR/MOV output work so the app does not build two competing EXR/export paths.

## Non-Goals
- Do not treat `hdr_scene_embeddings` as a normal LoRA.
- Do not remove frontend/backend HDR gates before the pipeline can load both HDR assets and produce valid outputs.
- Do not use ProRes MOV as the true HDR deliverable; MOV ProRes is useful preview/editor output, not a replacement for linear 16-bit EXR.
- Do not broaden to inpainting; locator confirms HDR is single-stage V2V.
- Do not add a second generic output-format framework inside the HDR codepath if the primary output plan owns that abstraction.

## Required Architecture Decisions

### 1. Scene Embedding Injection
Current state from locator:
- `backend/runtime_config/model_download_specs.py:236-254` registers `hdr` as `kind="ic_lora"` and `hdr_scene_embeddings` as `kind="embeddings"`.
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py` currently routes resolved LoRA components into `LoraPathStrengthAndSDOps` / `ICLoraPipeline` LoRA stack.
- ComfyUI workflow feeds `hdr_scene_embeddings` into conditioning, not into the model LoRA stack.

Plan:
1. Inspect installed `ltx_pipelines.ic_lora.ICLoraPipeline` and related conditioning code for a supported HDR/scene-embedding parameter before writing custom loader code.
2. If upstream supports scene embeddings, add the smallest adapter-specific pass-through in `ltx_ic_lora_pipeline.py`.
3. If upstream does not support scene embeddings, stop and design a narrow conditioning injection path against the text encoder/conditioning layer; do not fake it by stacking the embedding file as LoRA.
4. Require profile validation that HDR generation has both:
   - `ModelComponentPaths.ic_lora_hdr`
   - `ModelComponentPaths.ic_lora_hdr_scene_embeddings`

Stop condition: if `ICLoraPipeline` has no stable API for external embeddings and conditioning internals are unclear, block implementation pending upstream/code inspection.

### 2. HDR Decode Postprocess
Current state from locator:
- Current app output is `ICLoraPipeline.pipeline()` tensor/audio -> `encode_video_output()` -> `ltx_pipelines.utils.media_io.encode_video()` -> MP4/H.264.
- ComfyUI HDR workflow uses `LTXVHDRDecodePostprocess` after `VAEDecodeTiled`.
- Postprocess produces:
  - `hdr_linear`: linear HDR float pixels saved as EXR.
  - `tonemapped`: SDR preview image/video.
  - exposure default `7.1` EV affects SDR preview only, not EXR.
  - half precision default true for EXR.

Plan:
1. Add HDR-specific postprocess after VAE decode / pipeline tensor output, before final output persistence.
2. Preserve linear HDR data for EXR sequence output.
3. Generate a tonemapped SDR preview video for existing app playback, using exposure default `7.1` EV and a simple documented tone map first.
4. Keep exposure scoped to SDR preview; never bake exposure into linear EXR.
5. Add a calibration/option hook for exposure because HDR viewing is environment-dependent.

Stop condition: if pipeline only exposes already-tonemapped SDR tensors and not linear HDR decode output, block HDR output work and inspect/modify deeper decode layer rather than exporting fake HDR.

### 3. EXR Primary Output Dependency
HDR depends on primary EXR output support. The primary output locator found:
- `backend/.venv/lib/python3.13/site-packages/ltx_pipelines/utils/media_io.py:encode_video()` hardcodes libx264.
- `backend/services/ltx_pipeline_common.py:encode_video_output()` is the app bridge.
- Handler output paths are hardcoded `.mp4`.
- EXR is not playable in `<video>` and needs proxy/preview handling.
- OpenEXR/pyexr dependency is absent.

Plan relationship:
1. Primary EXR/MOV plan should own generic output-format plumbing:
   - API output format field.
   - handler output path extension/asset metadata.
   - EXR sequence writer dependency choice.
   - ZIP/folder sequence convention.
   - preview/proxy handling for non-video outputs.
2. HDR plan should own HDR-specific content generation:
   - scene embedding injection.
   - HDR decode/postprocess.
   - linear HDR frame production.
   - SDR tonemapped preview policy.
3. HDR implementation should not add its own parallel asset schema or output format enum unless primary output plan is not proceeding.
4. Recommended sequencing:
   - First: primary output plan implements generic `exr_sequence` + preview/proxy contract.
   - Second: HDR plugs linear HDR frames into that writer and returns preview MP4 plus EXR primary artifact.

Stop condition: if primary EXR/MOV output plan is rejected or delayed, create a temporary HDR-only EXR writer only behind a clear `ponytail:` comment and plan to delete it when generic output support lands.

### 4. Gating Removal Phases
Current gates from locator:
- Backend returns 400 unavailable for `hdr` and `hdr_scene_embeddings` in `backend/handlers/ic_lora_handler.py`.
- Tests assert those 400s in `backend/tests/test_ic_lora.py:644-685`.
- Frontend marks HDR unavailable in `frontend/components/ICLoraPanel.tsx:69` and renders in disabled group around `~753-760`.

Phased removal:
1. Keep all gates while only model specs exist.
2. Backend internal phase: allow HDR only in tests/fake pipeline once both assets are resolved and scene embedding injection path is wired; frontend still disabled.
3. Output phase: enable HDR backend only when EXR writer + SDR preview output both exist.
4. UI phase: move frontend HDR option from unavailable to active only after backend integration tests pass.
5. Replace unavailable tests with functional validation:
   - missing HDR LoRA returns clear model/profile requirement.
   - missing scene embeddings returns clear model/profile requirement.
   - both assets present routes through HDR pipeline mode.
   - HDR output response includes primary EXR artifact plus preview path/metadata once primary output contract exists.

Stop condition: do not remove frontend unavailable gate before backend rejects missing models with specific placement/source guidance.

### 5. Model/Profile Requirements
Required assets from locator:
- Base model: `ltx-2.3-22b-dev.safetensors` (~43GB), transformer, `Lightricks/LTX-2.3`.
- Distilled LoRA: `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` (~7GB), `Lightricks/LTX-2.3`.
- HDR adapter: `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors` (~312MB), `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`.
- HDR scene embeddings: `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors` (~12MB), same repo.

Plan:
1. Treat HDR as a compound capability requiring `hdr` + `hdr_scene_embeddings` + existing 2.3 base/distilled components.
2. Surface missing model UI with source link and exact required placement path, per repo instruction.
3. Keep `OFFICIAL_LTX23_ADAPTERS` pairing test; adapt recommendation logic rather than removing it.
4. Do not auto-download or re-download without scanning the configured models folder first.
5. Profile should expose both component paths and validation should fail fast before GPU work.

### 6. Test Plan
Backend tests:
- Keep `tests/test_model_download_specs.py` HDR pairing/spec tests.
- Update `tests/test_models.py` adapter recommendation expectations to require/display both HDR files.
- Replace `tests/test_ic_lora.py` unavailable gate tests with missing/available HDR behavior tests.
- Add focused HDR pipeline tests, likely `backend/tests/test_hdr_ic_lora.py`, using fakes only:
  - rejects missing `ic_lora_hdr`.
  - rejects missing `ic_lora_hdr_scene_embeddings`.
  - passes scene embedding path separately from LoRA stack.
  - invokes HDR postprocess for HDR adapter only.
  - returns/records EXR primary artifact and SDR preview once primary output contract exists.

Output tests aligned with primary plan:
- EXR writer creates expected `.exr` frame sequence or zip using chosen dependency.
- Linear EXR values are not exposure-adjusted.
- SDR preview exposure affects preview only.
- MOV/ProRes tests belong to primary output plan, not HDR-specific tests, except verifying HDR preview/asset relationship.

Frontend tests/manual checks:
- No frontend test framework exists. Use typecheck and manual UI check.
- Manual: HDR option disabled until backend support phase; active only after end-to-end support.
- Manual: missing model UI shows source link and exact placement for both HDR assets.

Validation commands for eventual implementation:
- `rtk pnpm backend:test -- tests/test_model_download_specs.py -v`
- `rtk pnpm backend:test -- tests/test_models.py -v`
- `rtk pnpm backend:test -- tests/test_ic_lora.py -v`
- `rtk pnpm backend:test -- tests/test_hdr_ic_lora.py -v`
- `rtk pnpm typecheck:py`
- `rtk pnpm typecheck:ts`

### 7. Implementation Sequencing Recommendation
1. Primary output architecture decision: choose EXR writer dependency and asset contract.
2. Primary output implementation slice: generic EXR sequence + preview/proxy path.
3. HDR pipeline research slice: inspect `ltx_pipelines` scene embedding API and VAE decode access.
4. HDR backend slice: scene embedding injection + profile validation, gate still on UI.
5. HDR postprocess slice: linear EXR frames + SDR tonemapped preview using primary writer contract.
6. Gate removal slice: backend tests converted, frontend option enabled.
7. End-to-end manual verification with real HDR weights and viewer (DJV or equivalent).

## Risks
- Upstream `ICLoraPipeline` may not expose scene embedding injection; custom conditioning injection may be invasive.
- Current pipeline may only expose SDR decoded tensors; true HDR EXR may require deeper VAE/decode integration.
- EXR dependency choice affects install portability; OpenCV EXR needs environment/system support, while `pyexr`/`OpenEXR` adds new Python dependency.
- EXR sequences are large and not playable in Chromium; preview/proxy contract must be designed before UI enablement.
- Real validation needs large model files not present locally per locator.

## Planner Self-Check
- Locator evidence sufficient: yes; HDR locator confidence high and primary output locator confidence high.
- Allowed edit files minimal and explicit: not applicable; no implementation packet requested.
- Read-only context minimal: yes; used AGENTS plus two locator artifacts only.
- Anchors/lines included: yes where locator provided stable backend/frontend/test locations.
- Validation concrete: yes for eventual implementation phases; no validation run for architecture-only planning.
- Parallelization decision explicit and safe: yes; primary output and HDR research can be separate, but HDR gate removal is sequential after output/postprocess.
- Non-goals and stop conditions sufficient: yes.
- Reviewer findings addressed, if revision: not applicable.
