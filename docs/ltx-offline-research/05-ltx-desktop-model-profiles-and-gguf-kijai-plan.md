# LTX-Desktop as Starting Point — Local Model Profiles, Kijai Models, and GGUF

## 1. Updated Goal

Use `https://github.com/Lightricks/LTX-Desktop` as the application base, but change model management so users do **not** have to download the official model bundle during installation/first launch.

The app should let users configure local paths for model components, including:

- Transformer / diffusion model
- Text encoder / Gemma root
- Text projection / embeddings connector
- Video VAE
- Audio VAE / vocoder
- Spatial upsampler
- Official LTX-2.3 distillation LoRAs
- Official LTX-2.3 IC-LoRAs: Union Control, Motion Track, Ingredients, HDR, LipDub, In/Outpainting, restoration, and VFX adapters
- Regular/user LoRA files
- Optional preprocessing models: depth, pose, person detector

It should support:

- Official Lightricks monolithic `.safetensors`
- Kijai split-component `.safetensors`
- Kijai GGUF-compatible setups
- QuantStack / community GGUF transformer files

## 2. LTX-Desktop Current Architecture

LTX-Desktop is already a good starting point because it has the desktop shell, local backend, settings UI, generation UI, project editor, and model-download flow.

Repo: `/tmp/clones/LTX-Desktop`

### 2.1 Layers

| Layer | Path | Stack | Role |
|------|------|-------|------|
| Frontend | `frontend/` | React + TypeScript + Tailwind | UI, settings, launch gate, editor |
| Electron | `electron/` | Electron main/preload | app lifecycle, IPC, dialogs, backend process |
| Backend | `backend/` | FastAPI + Python | model orchestration, downloads, generation |

### 2.2 Current model root support

LTX-Desktop already supports one configurable **models root**:

- Backend setting: `backend/state/app_settings.py` → `models_dir: str = ""`
- Effective model directory: `backend/handlers/base.py` → `StateHandlerBase.models_dir`
- First-run location dialog: `frontend/components/FirstRunSetup.tsx`
- Electron folder picker: `electron/ipc/app-handlers.ts` → `openModelsDirChangeDialog`
- Backend admin-protected settings update: `POST /api/settings` with `modelsDir`

This means we do **not** need to invent basic model directory support. The missing piece is **component-level model profiles**, because the existing root assumes the official fixed filenames.

## 3. Current Limitation: Fixed Official Checkpoint Specs

Model availability is currently defined by fixed checkpoint IDs in:

- `backend/api_types.py`
- `backend/runtime_config/model_download_specs.py`

Important current checkpoint specs:

| Checkpoint ID | Expected path under `models_dir` | Size | Meaning |
|--------------|----------------------------------|------|---------|
| `ltx-2.3-22b-distilled` | `ltx-2.3-22b-distilled.safetensors` | ~43 GB | Main monolithic model |
| `ltx-2.3-spatial-upscaler-x2-1.0` | `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | ~1.9 GB | Spatial upsampler |
| `gemma-3-12b-it-qat-q4_0-unquantized` | `gemma-3-12b-it-qat-q4_0-unquantized/` | ~25 GB | Local Gemma text encoder |
| `ltx-2.3-22b-ic-lora-union-control-ref0.5` | `.safetensors` | ~0.65 GB | IC-LoRA control |
| `dpt-hybrid-midas` | folder | ~0.5 GB | Depth processor |
| `dw-ll-ucoco-384-bs5` | `.pt` | ~0.13 GB | Pose processor |
| `z-image-turbo` | folder | ~31 GB | Text-to-image helper model |

`is_cp_downloaded()` only checks that those expected files/folders exist. It does not understand arbitrary component paths.

## 4. Current Pipeline Loading Flow

LTX-Desktop wraps official `ltx-pipelines` services:

| LTX-Desktop wrapper | Underlying official pipeline |
|--------------------|------------------------------|
| `services/fast_video_pipeline/ltx_fast_video_pipeline.py` | `ltx_pipelines.distilled.DistilledPipeline` |
| `services/a2v_pipeline/ltx_a2v_pipeline.py` | custom `DistilledA2VPipeline` wrapper |
| `services/ic_lora_pipeline/ltx_ic_lora_pipeline.py` | `ltx_pipelines.ic_lora.ICLoraPipeline` |
| `services/retake_pipeline/ltx_retake_pipeline.py` | forked retake orchestration using `ltx_pipelines.utils.blocks` |

The backend resolves paths in `backend/handlers/pipelines_handler.py`:

```python
checkpoint_path = str(get_existing_cp_path(self.models_dir, model_spec.model_cp))
upsampler_path = str(get_existing_cp_path(self.models_dir, model_spec.upscale_cp))
gemma_root = self._text_handler.resolve_gemma_root()
```

Then wrappers pass those paths into official pipelines.

### 4.1 Critical implication

Most official LTX pipeline blocks currently assume `checkpoint_path` contains multiple component weights:

- Transformer
- Prompt embedding processor / text projection
- Video VAE encoder/decoder
- Audio VAE encoder/decoder
- Vocoder

In `ltx_pipelines.utils.blocks.py`, the same `checkpoint_path` is used by:

- `DiffusionStage`
- `PromptEncoder`
- `ImageConditioner`
- `VideoDecoder`
- `AudioDecoder`
- `AudioConditioner`

So a **Kijai transformer-only** checkpoint will not be enough unless we either:

1. Pass multiple checkpoint paths into the LTX loader, or
2. Change the blocks/wrappers to accept component-specific paths.

`ltx-core` already supports multi-file loading in `load_state_dict(paths: str | tuple[str, ...] | list[str])`, and `SingleGPUModelBuilder` accepts `model_path: str | tuple[str, ...]`. This is the easiest path for split Kijai safetensors, but we must verify metadata/config behavior because `read_model_config()` reads metadata from the **first** file only.

## 5. First-Run Download Gate That Must Change

The current frontend blocks local-mode startup until required official models are downloaded:

- `frontend/App.tsx` checks `ApiClient.getLtxRecommendation()` and `ApiClient.getImgGenRecommendation()`.
- If missing, it renders `<LaunchGate />`.
- `frontend/components/FirstRunSetup.tsx` then pushes users through model license, install location, Hugging Face auth, and downloads.

For our goal, first-run should become:

> “Choose model source” instead of “download official bundle”.

Recommended first-run choices:

1. **Use existing local model components** — open Model Profile setup.
2. **Download official Lightricks bundle** — keep current flow.
3. **API-only mode** — skip local model setup.

The app should no longer force model download as the only path to a usable local install.

## 6. Kijai Model Findings

### 6.1 `Kijai/LTX2.3_comfy`

Kijai provides split LTX-2.3 components for ComfyUI-style loading.

Important files discovered via Hugging Face API:

| Component | Example file | Size |
|----------|--------------|------|
| Transformer bf16 | `diffusion_models/ltx-2.3-22b-dev_transformer_only_bf16.safetensors` | ~39.1 GiB |
| Transformer distilled bf16 | `diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_bf16.safetensors` | ~39.1 GiB |
| Transformer fp8 scaled | `diffusion_models/*_transformer_only_fp8_scaled.safetensors` | ~21.9–23.5 GiB |
| Transformer fp8 input-scaled | `diffusion_models/*_transformer_only_fp8_input_scaled*.safetensors` | ~21.6–23.3 GiB |
| Transformer MXFP8 | `diffusion_models/*_transformer_only_mxfp8_block32.safetensors` | ~22.4 GiB |
| Text projection | `text_encoders/ltx-2.3_text_projection_bf16.safetensors` | ~2.15 GiB |
| Video VAE | `vae/LTX23_video_vae_bf16.safetensors` | ~1.35 GiB |
| Audio VAE | `vae/LTX23_audio_vae_bf16.safetensors` | ~0.34 GiB |
| Tiny preview VAE | `vae/taeltx2_3.safetensors` | ~0.02 GiB |
| LoRAs | `loras/*.safetensors` | ~0.57–2.55 GiB |

Model card notes:

- Split checkpoint is intended as an alternate way to load LTX-2.3 in ComfyUI.
- FP8 input-scaled variants are experimental; `input_scaled_v3` keeps the first two and last two blocks bf16 and is better calibrated.
- Tiny VAE is for previews only, not final quality.

### 6.2 `Kijai/LTXV2_comfy`

Older LTXV2 split set includes both safetensors and GGUF transformer variants.

Important files:

| Component | Example file | Size |
|----------|--------------|------|
| Dev transformer bf16 | `ltx-2-19b-dev_transformer_only_bf16.safetensors` | ~35.17 GiB |
| Dev transformer fp8 | `ltx-2-19b-dev-fp8_transformer_only.safetensors` | ~20.07 GiB |
| Dev transformer fp4 | `ltx-2-19b-dev_fp4_transformer_only.safetensors` | ~13.47 GiB |
| Dev transformer GGUF Q4 | `ltx-2-19b-dev_Q4_K_M.gguf` | ~11.78 GiB |
| Distilled GGUF Q4 | `ltx-2-19b-distilled_Q4_K_M.gguf` | ~11.78 GiB |
| Distilled GGUF Q6 | `ltx-2-19b-distilled_Q6_K.gguf` | ~14.83 GiB |
| Embeddings connector | `text_encoders/ltx-2-19b-embeddings_connector_*_bf16.safetensors` | ~2.67 GiB |
| Rank-reduced LoRAs | `loras/*.safetensors` | ~1.67–4.55 GiB |

Kijai model-card caveat:

- ComfyUI changed model loading so the **embedding connector moved from text encoder to diffusion model**.
- Kijai recommends updated KJNodes loaders, especially for single-file/GGUF setups.

## 7. GGUF Findings Relevant to LTX-Desktop

### 7.1 `QuantStack/LTX-2.3-GGUF`

Model card says this is a direct conversion of `Lightricks/LTX-2.3` and uses GGUF architecture `ltxv`.

Important files:

| File | Size |
|------|------|
| `LTX-2.3-dev/LTX-2.3-dev-Q2_K.gguf` | ~11.56 GiB |
| `LTX-2.3-dev/LTX-2.3-dev-Q3_K_M.gguf` | ~13.69 GiB |
| `LTX-2.3-dev/LTX-2.3-dev-Q4_K_M.gguf` | ~16.54 GiB |
| `LTX-2.3-dev/LTX-2.3-dev-Q5_K_M.gguf` | ~18.06 GiB |
| `LTX-2.3-dev/LTX-2.3-dev-Q6_K.gguf` | ~19.56 GiB |

Model card component layout:

| Component | Location in ComfyUI terms | Source |
|----------|----------------------------|--------|
| Main model / transformer | `ComfyUI/models/unet` | this GGUF repo |
| Gemma 3 text encoder | `ComfyUI/models/text_encoders` | safetensors or GGUF |
| Text projection | `ComfyUI/models/text_encoders` | `Kijai/LTX2.3_comfy` |
| Video/audio VAE | `ComfyUI/models/vae` | `Kijai/LTX2.3_comfy` |

### 7.2 Loader references

`city96/ComfyUI-GGUF`:

- Supports image/diffusion architectures including `ltxv` in `loader.py`.
- Uses `GGMLTensor` / `GGMLOps` to keep weights quantized and dynamically dequantize during forward.
- Uses `GGUFModelPatcher` so ComfyUI LoRA patches can be applied to quantized weights.

`kijai/ComfyUI-KJNodes`:

- Provides `GGUFLoaderKJ`.
- Requires `ComfyUI-GGUF`.
- Can load an `extra_model_name` and merge it into the GGUF state dict.
- Special case: if `extra_model_name` contains `connector`, it loads it from `text_encoders` and strips `model.diffusion_model.` prefix before merging.
- Exposes attention override: `sdpa`, `sageattn`, `xformers`, `flashattn`.
- Has `LTX2LoraLoaderAdvanced` with per-layer/per-block strength control and optional absolute LoRA path.

This is a strong reference for our LTX-Desktop implementation.

## 8. Recommended Data Model: Model Profiles

Do **not** try to force arbitrary model setups into `ModelCheckpointID` literals. Add a new model-profile layer and keep old checkpoint specs for official downloads.

### 8.1 Backend settings addition

Add to `AppSettings`:

```python
active_model_profile_id: str = "official-default"
model_profiles: list[ModelProfile] = []
```

Or keep profiles in a separate JSON file under app data:

```text
<AppData>/model_profiles.json
```

Separate file is cleaner because profiles contain user paths and can grow independently of normal settings.

### 8.2 Suggested schema

```python
class ModelComponentPaths(BaseModel):
    transformer: str
    transformer_format: Literal["official_safetensors", "split_safetensors", "gguf"]
    transformer_quantization: str | None = None

    upsampler: str | None = None
    text_encoder_root: str | None = None
    text_encoder_format: Literal["hf_folder", "safetensors", "gguf", "api"] = "api"
    text_projection: str | None = None
    embeddings_connector: str | None = None

    video_vae: str | None = None
    audio_vae: str | None = None
    vocoder: str | None = None

    ic_lora_union: str | None = None
    ic_lora_motion_track: str | None = None
    ic_lora_ingredients: str | None = None
    ic_lora_hdr: str | None = None
    ic_lora_hdr_scene_embeddings: str | None = None
    ic_lora_lipdub: str | None = None
    ic_lora_in_outpainting: str | None = None
    official_adapters: dict[str, str] = {}

    depth_processor: str | None = None
    pose_processor: str | None = None
    person_detector: str | None = None

class ModelProfile(BaseModel):
    id: str
    display_name: str
    family: Literal["ltx-2", "ltx-2.3", "ltxv2", "custom"]
    source: Literal["official", "kijai", "quantstack", "custom"]
    components: ModelComponentPaths
    capabilities: set[Literal["t2v", "i2v", "a2v", "retake", "ic_lora", "local_text", "gguf"]]
```

### 8.3 Validation rules

At profile save time:

- Path exists.
- Extension matches declared format (`.safetensors`, `.gguf`, folder).
- If transformer is `gguf`, require GGUF loader dependency.
- If transformer is Kijai split, require `text_projection` or `embeddings_connector`.
- If user wants final decode locally, require video/audio VAE paths unless the base checkpoint is monolithic.
- If `text_encoder_format = api`, allow `text_encoder_root` to be empty.
- If an IC-LoRA pipeline/feature is enabled, require that feature’s adapter file and any processor/input dependencies. Do not assume Union Control covers HDR, LipDub, In/Outpainting, Motion Track, Ingredients, or restoration/VFX adapters.

## 9. Recommended Loader Architecture

Add a path-resolution service between handlers and pipeline wrappers.

```text
handlers/pipelines_handler.py
  -> ModelProfileResolver
      -> OfficialMonolithResolver
      -> SplitSafetensorsResolver
      -> GGUFResolver
  -> pipeline wrapper
```

### 9.1 Component bundle object

```python
class ResolvedLtxComponents(BaseModel):
    transformer_paths: tuple[str, ...]
    transformer_format: Literal["safetensors", "gguf"]
    checkpoint_paths_for_filtered_builders: tuple[str, ...]
    upsampler_path: str | None
    gemma_root: str | None
    text_projection_path: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    loras: list[str]
```

For official monolithic:

```python
checkpoint_paths_for_filtered_builders = (official_checkpoint,)
```

For Kijai split safetensors:

```python
checkpoint_paths_for_filtered_builders = (
    transformer_only,
    text_projection,
    video_vae,
    audio_vae,
)
```

This may work because `ltx-core` loaders accept multiple paths and component builders filter keys. Must validate metadata/config handling.

For GGUF:

```python
transformer_format = "gguf"
transformer_paths = (gguf_transformer, maybe_embeddings_connector)
checkpoint_paths_for_filtered_builders = (text_projection, video_vae, audio_vae)
```

GGUF likely needs a custom transformer builder, while VAE/text projection can stay safetensors.

## 10. Pipeline Wrapper Changes

Current wrapper `create(...)` signatures accept `checkpoint_path: str`. Change them to accept a component bundle or resolver.

Current:

```python
LTXFastVideoPipeline.create(checkpoint_path, gemma_root, upsampler_path, device, streaming_prefetch_count)
```

Recommended:

```python
LTXFastVideoPipeline.create(components: ResolvedLtxComponents, device, streaming_prefetch_count)
```

Then internally:

- Official / split safetensors: pass `components.checkpoint_paths_for_filtered_builders` into `ltx-pipelines` or modified block constructors.
- GGUF: replace only the transformer `DiffusionStage` builder with a GGUF-aware builder.

## 11. UI Changes

### 11.1 First run

Replace download-only LaunchGate with:

1. **Use existing model files**
2. **Download official LTX model bundle**
3. **Use API-only mode**

For “Use existing model files”, show a Model Profile wizard:

- Profile name
- Model type: Official / Kijai split / GGUF / Custom
- File pickers for each component
- Validation result panel
- Save and activate profile

### 11.2 Settings modal

Add a `Models` tab:

- Active profile selector
- Add / edit / duplicate / delete profile
- Component path fields
- “Validate profile” button
- “Open containing folder” buttons
- Optional download buttons for official components only

### 11.3 File dialogs

Electron already has general file/directory dialogs in `electron/ipc/file-handlers.ts`, plus a model-dir picker in `app-handlers.ts`. Reuse these for component pickers.

## 12. API Changes

Add backend endpoints:

```text
GET    /api/model-profiles
POST   /api/model-profiles
PATCH  /api/model-profiles/{id}
DELETE /api/model-profiles/{id}
POST   /api/model-profiles/{id}/validate
POST   /api/model-profiles/{id}/activate
```

Keep existing `/api/models/*` download endpoints for official downloads.

Update recommendation endpoints:

- If active profile validates, return OK.
- If no valid profile and local mode, return “profile required” rather than always “download required”.
- If user selects official download mode, keep current behavior.

## 13. Implementation Strategy

### Phase A — Non-invasive model profile support

- Add model profile schema/storage.
- Add UI to choose existing official monolithic paths manually.
- Make first-run skip downloads if an official monolithic profile validates.

This validates the UX without touching model internals.

### Phase B — Kijai split safetensors

- Extend pipeline wrappers to accept tuple paths.
- Use tuple paths for `SingleGPUModelBuilder`:
  - transformer-only
  - text projection / embeddings connector
  - video VAE
  - audio VAE
- Verify each pipeline: fast, A2V, IC-LoRA, retake.
- If metadata problems occur, add a tiny metadata/config sidecar in the profile or ensure the transformer file is first.

### Phase C — GGUF transformer support

- Port/adapt `ComfyUI-GGUF` loader primitives:
  - GGUF reader
  - `GGMLTensor`
  - dynamic dequantized `Linear`
  - LoRA patch behavior
- Or use Diffusers `GGUFQuantizationConfig` as a reference path for transformer-only loading.
- Add `GGUFTransformerBuilder` behind the same component bundle interface.
- Merge Kijai embeddings connector into GGUF state dict like `GGUFLoaderKJ` does.

### Phase D — LoRA profile support

- Add LoRA picker and per-LoRA strength.
- For safetensors base: use official `LoraPathStrengthAndSDOps` + `LTXV_LORA_COMFY_RENAMING_MAP`.
- For GGUF base: apply LoRAs as runtime patches like `GGUFModelPatcher`.
- Optionally support KJNodes-style advanced per-stream/per-block LoRA strengths later.

## 14. Key Risks

| Risk | Why it matters | Mitigation |
|------|----------------|------------|
| Kijai split files may not include metadata needed by `ltx-core` | `read_model_config()` reads first path metadata | Put transformer file first; add sidecar config fallback if needed |
| Official `ltx-pipelines` assumes one checkpoint path | Kijai/GGUF is componentized | Introduce component bundle and modify wrappers/blocks minimally |
| GGUF LoRA fusion cannot use official `fuse_loras.py` directly | GGUF weights are quantized tensor wrappers | Use patch-at-runtime approach from ComfyUI-GGUF |
| First-run gate still forces downloads | Bad UX for local model users | Change recommendations to recognize valid active profiles |
| Kijai loader behavior changes | ComfyUI/KJNodes evolves quickly | Treat Kijai support as profile presets + validation, not hardcoded only paths |

## 15. Bottom Line

LTX-Desktop is a strong base, but the current code is designed around a fixed official monolithic model bundle. The right next step is not to hack arbitrary paths into `ModelCheckpointID`; it is to add a **Model Profile** layer that resolves component paths before pipeline construction.

Recommended order:

1. Add model profiles and first-run “use existing model files”.
2. Support official monolithic profiles manually selected by the user.
3. Support Kijai split safetensors via tuple checkpoint paths.
4. Add GGUF transformer builder using ComfyUI-GGUF / KJNodes as the reference.
5. Add GGUF-safe LoRA patching.
