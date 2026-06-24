# Revised Roadmap — Start from LTX-Desktop

This supersedes the earlier greenfield/hybrid app recommendation. We should use `Lightricks/LTX-Desktop` as the app base and add a model-profile layer for existing local models, Kijai split models, and GGUF.

## Milestone 1 — Preserve LTX-Desktop, Add Manual Official Profiles

Goal: users can install the app without downloading models immediately, then point it at an existing official LTX model folder.

### Backend

- Add `ModelProfile` / `ModelComponentPaths` models.
- Store profiles in `<app_data>/model_profiles.json`.
- Add endpoints:
  - `GET /api/model-profiles`
  - `POST /api/model-profiles`
  - `PATCH /api/model-profiles/{id}`
  - `DELETE /api/model-profiles/{id}`
  - `POST /api/model-profiles/{id}/validate`
  - `POST /api/model-profiles/{id}/activate`
- Update model recommendations so a valid active profile satisfies the first-run gate.

### Frontend / Electron

- Change first-run flow from “download required” to “choose model source”.
- Add model profile wizard:
  - Official monolithic `.safetensors`
  - Spatial upsampler
  - Gemma root or API text encoding
- Reuse existing Electron file/folder dialogs.

### Success criteria

- Fresh install can launch without model download.
- User can select existing official LTX-2.3 files and generate with current `DistilledPipeline` path.
- Existing official download flow still works.

## Milestone 2 — Kijai Split Safetensors

Goal: support `Kijai/LTX2.3_comfy` paths.

### Backend

- Extend profile type: `source = "kijai"`, `transformer_format = "split_safetensors"`.
- Allow component paths:
  - `transformer`
  - `text_projection` / `embeddings_connector`
  - `video_vae`
  - `audio_vae`
  - `upsampler`
- Create `ResolvedLtxComponents` and pass it to pipeline wrappers instead of raw `checkpoint_path`.
- Use `ltx-core` multi-file loading support (`model_path: tuple[str, ...]`) where possible.

### Success criteria

- Fast video pipeline runs using Kijai split safetensors.
- Retake and IC-LoRA either run or fail with clear “unsupported with this profile” messages.

## Milestone 3 — Official LTX-2.3 Adapter Registry

Goal: add all official LTX-2.3 LoRA / IC-LoRA assets as first-class optional model-profile components before adding GGUF-specific LoRA behavior.

### Backend

- Add an adapter registry independent of the current `ModelCheckpointID` literals.
- Include official LTX-2.3 distillation LoRAs and all discovered `Lightricks/LTX-2.3-22b-IC-LoRA-*` repos.
- Add per-adapter fields: `repo_id`, `filename`, `path`, `kind`, `required_for`, `optional_for`, expected size.
- Add adapter status/validation endpoints.
- Make recommendation logic pipeline-aware: base install should not force all adapters, but a selected pipeline should require its adapter.

### Frontend

- Add Models → Official Adapters checklist.
- For each adapter: show `missing / local / downloadable / loaded` status.
- Add “Browse local file” and “Download official adapter” actions.

### Success criteria

- Union Control, HDR, LipDub, Ingredients, Motion Track, In/Outpainting, and restoration/VFX IC-LoRAs can be registered locally without code changes.
- IC-LoRA panel still works with Union Control.
- Missing adapter errors are specific to the selected feature.

## Milestone 4 — GGUF Transformer Profiles

Goal: support `QuantStack/LTX-2.3-GGUF` and Kijai GGUF transformer files.

### Backend

- Add optional dependencies: `gguf` and GGUF loader module.
- Implement `GGUFTransformerBuilder` using ComfyUI-GGUF / Diffusers GGUF as reference.
- Merge optional embeddings connector into GGUF transformer state like KJNodes `GGUFLoaderKJ`.
- Add validation for `.gguf` architecture `ltxv`.

### Success criteria

- App loads `LTX-2.3-dev-Q4_K_M.gguf` transformer plus Kijai VAE/text projection.
- Generation works on <24 GB VRAM with API text encoding first, local text later.

## Milestone 5 — Local Text Encoder Options

Goal: avoid forcing the 25 GB Gemma download.

Options:

1. API text encoding (existing, free, fastest).
2. Existing local Gemma folder selected by user.
3. Future: GGUF / quantized Gemma profile.

### Success criteria

- User can choose API text encoding even with local GGUF transformer.
- User can point to existing Gemma root if they want fully local.

## Milestone 6 — LoRA Compatibility

Goal: support normal user LoRAs, official IC-LoRAs, and Kijai rank-reduced LoRAs across safetensors and GGUF bases.

### Backend

- Add LoRA entries to model profiles or generation request:
  - path
  - strength
  - optional type/profile
- Safetensors base: use official `LoraPathStrengthAndSDOps` + Comfy rename map.
- GGUF base: add runtime patching using ComfyUI-GGUF approach.
- Later: KJNodes-style advanced block/stream strength controls.

### Success criteria

- Official IC-LoRA works on official/split safetensors.
- Regular LoRA works with GGUF transformer path.

## Milestone 7 — UX Polish

- `Models` settings tab.
- “Validate profile” panel.
- Hardware/VRAM preset recommendations:
  - 12 GB: GGUF Q4, API text, low resolution, tiled VAE.
  - 16 GB: GGUF Q4/Q5, API or 8-bit local text.
  - 24 GB: Kijai fp8 or official fp8, local or API text.
- Explicit compatibility warnings.

## First Implementation Slice

Smallest useful PR:

1. Add profile JSON storage and API endpoints.
2. Add first-run “Use existing official model files” path.
3. Make recommendations return OK for a valid active official profile.
4. Wire existing `LTXFastVideoPipeline` to selected profile paths.
5. Add tests for profile validation and recommendation behavior.

No GGUF code in the first PR. The second slice should add the official adapter registry and pipeline-aware adapter validation; GGUF should come after that.
