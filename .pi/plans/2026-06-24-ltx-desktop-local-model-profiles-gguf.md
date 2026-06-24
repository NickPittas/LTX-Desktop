# LTX-Desktop Local Model Profiles + Official Adapters + GGUF Implementation Plan

**Goal:** Turn LTX-Desktop into a local-model-friendly app that can use official LTX-2.3 files, Kijai split components, official LoRA/IC-LoRA adapters, and GGUF transformers without forcing model downloads at install/first launch.
**Architecture:** Add a backend model-profile layer that resolves user-selected component paths into a typed component bundle consumed by existing pipeline wrappers. Keep official downloads as an option, add an official adapter registry, then extend loaders to support split safetensors and GGUF behind the same component contract.
**Tech Stack:** Electron + React + TypeScript frontend; FastAPI + Pydantic + Python backend; `ltx-core`, `ltx-pipelines`, `gguf`, PyTorch; existing `pnpm`/`uv` test gates.
---

## Current Progress — 2026-06-24

**Stage:** Milestone 1 complete; start Milestone 2 next.

Completed commit:

- `85499c0 feat(models): add local model profiles`

Milestone 1 shipped:

- Backend model profile DTOs in `backend/api_types.py`.
- Persistent profile state fields in `backend/state/app_state_types.py`.
- `ModelProfilesHandler` with JSON persistence at `<app_data>/model_profiles.json`.
- Admin-gated model profile routes under `/api/model-profiles`.
- Startup loading via `AppHandler.load_persistent_state()`.
- LTX recommendation now returns `ok` when a valid active official profile exists.
- Integration tests in `backend/tests/test_model_profiles.py`.

Validation run before commit:

```bash
cd /tmp/clones/LTX-Desktop/backend && uv run pyright
cd /tmp/clones/LTX-Desktop/backend && uv run pytest -q --tb=short
cd /tmp/clones/LTX-Desktop && npx --yes pnpm@10.30.3 run typecheck
cd /tmp/clones/LTX-Desktop && npx --yes pnpm@10.30.3 run build:frontend
cd /tmp/clones/LTX-Desktop && npx --yes pnpm@10.30.3 run backend:test
```

Result: `pyright` green, frontend build green, backend suite `273 passed`.

Next action:

- Start **Milestone 2: Official LTX-2.3 Adapter Registry** using `docs/ltx-offline-research/07-official-ltx23-lora-registry.md`.

## Non-Negotiable Constraints

- Do not replace LTX-Desktop; build on `Lightricks/LTX-Desktop`.
- Do not force-download models at install or first launch.
- Keep current official download flow as an optional path.
- Component paths are user-owned local paths, not copied into app data unless user downloads official files.
- Add official LTX-2.3 adapters as known capabilities, not mandatory install payload.
- Implement in small sequential slices; backend API/types first, UI second, loaders last.
- No ComfyUI runtime dependency in LTX-Desktop. Use ComfyUI-GGUF/KJNodes as reference only.

## Quality Gate

Run from `/tmp/clones/LTX-Desktop` after each slice:

```bash
pnpm backend:test -- tests/<targeted>.py
pnpm typecheck:py
pnpm typecheck:ts
pnpm build:frontend
```

When backend API schema changes:

```bash
pnpm openapi:generate
pnpm openapi:check
```

Full final gate:

```bash
pnpm typecheck
pnpm backend:test
pnpm build:frontend
```

---

## Milestone 1: Backend Model Profiles

**Purpose:** Add backend-only profile storage/API so an existing official local model can satisfy startup readiness without downloading.

### Task 1.1: Model profile DTOs

**Files:**
- Modify: `backend/api_types.py`
- Test: `backend/tests/test_model_profiles.py`

**Step 1: Write failing tests**

Create `backend/tests/test_model_profiles.py` covering:

- `GET /api/model-profiles` returns empty profile list and null active profile.
- Creating an official profile accepts transformer/upscaler/Gemma/API text fields.
- Validation reports missing files and bad extensions.

**Step 2: Run test to verify failure**

Run:

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_model_profiles.py
```

Expected: fail because route and DTOs do not exist.

**Step 3: Add DTO contract**

Add these names to `backend/api_types.py`:

- `ModelProfileId`
- `ModelProfileFamily = Literal["ltx-2", "ltx-2.3", "ltxv2", "custom"]`
- `ModelProfileSource = Literal["official", "kijai", "quantstack", "custom"]`
- `ModelProfileTransformerFormat = Literal["official_safetensors", "split_safetensors", "gguf"]`
- `ModelProfileTextEncoderFormat = Literal["hf_folder", "safetensors", "gguf", "api"]`
- `ModelProfileCapability = Literal["t2v", "i2v", "a2v", "retake", "ic_lora", "local_text", "gguf"]`
- `ModelComponentPaths`
- `ModelProfilePayload`
- `ModelProfilePatchPayload`
- `ModelProfilesResponse`
- `ModelProfileValidationIssuePayload`
- `ModelProfileValidationResponse`
- `ModelProfileActivateResponse`

Core contract:

```python
class ModelComponentPaths(BaseModel):
    transformer: str | None = None
    transformer_format: ModelProfileTransformerFormat = "official_safetensors"
    transformer_quantization: str | None = None
    upsampler: str | None = None
    text_encoder_root: str | None = None
    text_encoder_format: ModelProfileTextEncoderFormat = "api"
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
    official_adapters: dict[str, str] = Field(default_factory=dict)
    depth_processor: str | None = None
    pose_processor: str | None = None
    person_detector: str | None = None
```

**Step 4: Run DTO tests**

Run:

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_model_profiles.py
```

Expected: still fails on missing route/handler, DTO import failures gone.

### Task 1.2: Model profile handler + persistence

**Files:**
- Create: `backend/handlers/model_profiles_handler.py`
- Modify: `backend/state/app_state_types.py`
- Modify: `backend/handlers/__init__.py`
- Test: `backend/tests/test_model_profiles.py`

**Step 1: Extend state**

Add to `AppState`:

```python
model_profiles: list[ModelProfilePayload] = field(default_factory=list)
active_model_profile_id: str | None = None
```

**Step 2: Implement handler**

Create `ModelProfilesHandler(StateHandlerBase)` with:

- `load_profiles()`
- `save_profiles()`
- `list_profiles()`
- `create_profile(req)`
- `patch_profile(profile_id, req)`
- `delete_profile(profile_id)`
- `activate_profile(profile_id)`
- `validate_profile_by_id(profile_id)`
- `validate_profile(profile)`
- `has_valid_active_official_profile()`

Storage file:

```text
<app_data_dir>/model_profiles.json
```

JSON shape:

```json
{
  "active_model_profile_id": null,
  "profiles": []
}
```

**Step 3: Validation rules**

- Every non-empty path must exist.
- `official_safetensors` requires `transformer` `.safetensors`.
- `split_safetensors` requires `.safetensors` transformer and component files.
- `gguf` requires `.gguf` transformer.
- VAE/upscaler/projection/LoRA file fields require `.safetensors` when present.
- `text_encoder_format="hf_folder"` requires existing directory.
- `text_encoder_format="api"` requires app LTX API key only when profile must satisfy readiness.
- `has_valid_active_official_profile()` requires source `official`, transformer, upsampler, and either API key or valid local HF text folder.

**Step 4: Run tests**

Run:

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_model_profiles.py
```

Expected: handler unit coverage passes once routes are wired.

### Task 1.3: Model profile API routes + recommendation gate

**Files:**
- Create: `backend/_routes/model_profiles.py`
- Modify: `backend/app_factory.py`
- Modify: `backend/app_handler.py`
- Modify: `backend/handlers/models_handler.py`
- Test: `backend/tests/test_model_profiles.py`
- Test: `backend/tests/test_models.py`

**Step 1: Add route contract**

Endpoints:

```text
GET    /api/model-profiles
POST   /api/model-profiles
PATCH  /api/model-profiles/{profile_id}
DELETE /api/model-profiles/{profile_id}
POST   /api/model-profiles/{profile_id}/validate
POST   /api/model-profiles/{profile_id}/activate
```

Use `guard_admin_permission(request)` for all endpoints. These expose local paths.

**Step 2: Wire AppHandler**

- Construct `self.model_profiles` before `self.models`.
- Pass it into `ModelsHandler`.
- Call `self.model_profiles.load_profiles()` in persistent-state load.

**Step 3: Update readiness**

In `ModelsHandler.get_ltx_recommendation()`:

```python
if self._model_profiles.has_valid_active_official_profile():
    return LtxOkRecommendationResponse(status="ok")
```

Keep existing official download behavior otherwise.

**Step 4: Run tests**

Run:

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_model_profiles.py tests/test_models.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

---

## Milestone 2: Official LTX-2.3 Adapter Registry

**Purpose:** Add all official LTX-2.3 LoRA/IC-LoRA assets as known optional components with pipeline-aware validation.

### Task 2.1: Adapter types and registry

**Files:**
- Modify: `backend/api_types.py`
- Modify: `backend/runtime_config/model_download_specs.py`
- Test: `backend/tests/test_model_download_specs.py`

**Step 1: Write failing tests**

Test registry includes all official adapter IDs and no missing metadata.

**Step 2: Add API types**

Add:

- `AdapterID`
- `AdapterKind`
- `AdapterSource`
- `AdapterPipeline`
- `AdapterComponentPayload`
- `AdapterStatusItem`
- `AdapterRequirementItem`
- `AdapterStatusResponse`
- `AdapterRecommendationResponse`

Adapter IDs:

```python
AdapterID = Literal[
    "distilled_lora_384",
    "distilled_lora_384_1_1",
    "union_control",
    "motion_track_control",
    "ingredients",
    "water_simulation",
    "decompression",
    "deblur",
    "colorization",
    "day_to_night",
    "in_outpainting",
    "instant_shave",
    "cross_eyed",
    "hdr",
    "hdr_scene_embeddings",
    "lipdub",
]
```

**Step 3: Add registry dataclass**

In `backend/runtime_config/model_download_specs.py` add `AdapterComponent` and `OFFICIAL_LTX23_ADAPTERS`.

Registry must include:

| ID | Repo | File |
|---|---|---|
| `distilled_lora_384` | `Lightricks/LTX-2.3` | `ltx-2.3-22b-distilled-lora-384.safetensors` |
| `distilled_lora_384_1_1` | `Lightricks/LTX-2.3` | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` |
| `union_control` | `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control` | `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors` |
| `motion_track_control` | `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control` | `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors` |
| `ingredients` | `Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients` | `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors` |
| `water_simulation` | `Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation` | `ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors` |
| `decompression` | `Lightricks/LTX-2.3-22b-IC-LoRA-Decompression` | `ltx-2.3-22b-ic-lora-decompression-0.9.safetensors` |
| `deblur` | `Lightricks/LTX-2.3-22b-IC-LoRA-Deblur` | `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors` |
| `colorization` | `Lightricks/LTX-2.3-22b-IC-LoRA-Colorization` | `ltx-2.3-22b-ic-lora-colorization-0.9.safetensors` |
| `day_to_night` | `Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night` | `ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors` |
| `in_outpainting` | `Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting` | `ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors` |
| `instant_shave` | `Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave` | `ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors` |
| `cross_eyed` | `Lightricks/LTX-2.3-22b-IC-LoRA-Cross-Eyed` | `ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors` |
| `hdr` | `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` | `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors` |
| `hdr_scene_embeddings` | `Lightricks/LTX-2.3-22b-IC-LoRA-HDR` | `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors` |
| `lipdub` | `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub` | `ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors` |

**Step 4: Run tests**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_model_download_specs.py
```

Expected: pass.

### Task 2.2: Adapter status/recommendation endpoints

**Files:**
- Modify: `backend/state/app_settings.py`
- Modify: `backend/handlers/models_handler.py`
- Modify: `backend/_routes/models.py`
- Test: `backend/tests/test_models.py`

**Step 1: Add local path overrides**

Add to `AppSettings`:

```python
adapter_paths: dict[str, str] = Field(default_factory=dict)
```

Expose in `SettingsResponse` only if UI needs it; otherwise keep status endpoints as source of truth.

**Step 2: Add handler methods**

In `ModelsHandler`:

- `get_adapter_status(pipeline: AdapterPipeline | None = None)`
- `get_adapter_recommendation(pipeline: AdapterPipeline)`
- `resolve_adapter_path(adapter_id: AdapterID) -> Path | None`

Source priority:

1. `app_settings.adapter_paths[adapter_id]`
2. file under `models_dir` by expected filename
3. missing/downloadable official repo

**Step 3: Add routes**

```text
GET /api/models/adapters/status
GET /api/models/adapters/recommendation?pipeline=<pipeline>
```

Keep legacy `GET /api/models/ltx-ic-lora-recommendation` backed by Union Control recommendation.

**Step 4: Run tests**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_models.py tests/test_ic_lora.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

---

## Milestone 3: Frontend Model Source + Adapter UX

**Purpose:** Expose model profiles and adapter status in first-run and Settings.

### Task 3.1: Generated API + hooks

**Files:**
- Modify: `frontend/lib/api-client.ts`
- Create: `frontend/types/model-profile.ts`
- Create: `frontend/hooks/use-model-profiles.ts`
- Create: `frontend/hooks/use-official-adapters.ts`
- Generated: `frontend/generated/backend-openapi.json`
- Generated: `frontend/generated/backend-openapi.ts`

**Step 1: Generate OpenAPI**

```bash
cd /tmp/clones/LTX-Desktop && pnpm openapi:generate
```

**Step 2: Add client helpers**

Add methods for:

- list/create/patch/delete/validate/activate model profile
- get adapter status/recommendation

Use encoded path segments for `{profile_id}`.

**Step 3: Add hooks**

- `useModelProfiles()` — load, refresh, create, patch, validate, activate, delete.
- `useOfficialAdapters()` — load status/recommendations.

**Step 4: Run checks**

```bash
cd /tmp/clones/LTX-Desktop && pnpm typecheck:ts
```

Expected: pass.

### Task 3.2: Reusable component pickers + profile wizard

**Files:**
- Create: `frontend/components/ModelComponentPicker.tsx`
- Create: `frontend/components/ModelProfileWizard.tsx`
- Modify: `frontend/types/model-profile.ts`
- Modify: `frontend/hooks/use-model-profiles.ts`

**Step 1: Picker contract**

Use existing Electron APIs:

- `window.electronAPI.showOpenFileDialog`
- `window.electronAPI.showOpenDirectoryDialog`

Filters:

- safetensors: `['safetensors']`
- GGUF: `['gguf']`
- model files: `['safetensors', 'gguf', 'pt', 'bin']`

**Step 2: Wizard flow**

Wizard supports:

1. Official monolithic
2. Kijai split safetensors
3. GGUF transformer profile
4. Custom

Fields shown depend on source/format. Validation happens via backend before activation.

**Step 3: Run checks**

```bash
cd /tmp/clones/LTX-Desktop && pnpm typecheck:ts && pnpm build:frontend
```

Expected: pass.

### Task 3.3: First-run choose source + Settings Models tab

**Files:**
- Modify: `frontend/App.tsx`
- Modify: `frontend/components/FirstRunSetup.tsx`
- Modify: `frontend/components/FirstRunSetup.css`
- Modify: `frontend/components/SettingsModal.tsx`
- Create: `frontend/components/AdapterChecklist.tsx`

**Step 1: First-run choices**

Replace download-only gate with:

1. Use existing local model components
2. Download official Lightricks bundle
3. API-only mode

**Step 2: Model readiness**

Rename readiness logic from “downloaded” to “ready”. Ready if:

- official download recommendation OK, or
- active profile validates, or
- force API/API-only path selected.

**Step 3: Settings Models tab**

Add tab ID `models`. Include:

- active profile card
- profile create/edit wizard
- profile validate/activate/delete actions
- official adapter checklist

**Step 4: Run checks**

```bash
cd /tmp/clones/LTX-Desktop && pnpm typecheck:ts && pnpm build:frontend
```

Expected: pass.

---

## Milestone 4: Component Bundle Seam

**Purpose:** Route pipeline construction through a resolved component bundle instead of fixed `checkpoint_path` strings.

### Task 4.1: Component bundle service

**Files:**
- Create: `backend/services/ltx_components.py`
- Modify: `backend/handlers/pipelines_handler.py`
- Modify: `backend/state/app_state_types.py`
- Modify: pipeline protocol files under `backend/services/*_pipeline/*.py`
- Test: `backend/tests/test_ltx_components.py`

**Step 1: Define component contract**

Add:

```python
TransformerFormat = Literal["safetensors", "gguf"]

@dataclass(frozen=True, slots=True)
class ResolvedLtxComponents:
    profile_id: str
    transformer_format: TransformerFormat
    transformer_path: str
    checkpoint_paths_for_filtered_builders: tuple[str, ...]
    upsampler_path: str
    gemma_root: str | None
    text_projection_path: str | None
    embeddings_connector_path: str | None
    video_vae_path: str | None
    audio_vae_path: str | None
    cache_key: tuple[str, ...]
```

Helper:

```python
def checkpoint_path_arg(components: ResolvedLtxComponents) -> str | tuple[str, ...]:
    paths = components.checkpoint_paths_for_filtered_builders
    return paths[0] if len(paths) == 1 else paths
```

**Step 2: Resolve components**

- Official profile: one monolithic checkpoint path.
- Kijai split: tuple path ordered transformer first, then text projection/connector, video VAE, audio VAE.
- GGUF: transformer path is `.gguf`; non-transformer safetensors go in filtered builder paths.

**Step 3: Cache key**

Pipeline cache must include component paths. Switching active profile evicts/recreates GPU pipeline.

**Step 4: Run tests**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_ltx_components.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

---

## Milestone 5: Kijai Split Safetensors

**Purpose:** Use Kijai transformer-only + component safetensors with existing `ltx-core` tuple-path loading.

### Task 5.1: Pass tuple paths through pipeline wrappers

**Files:**
- Modify: `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
- Modify: `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
- Modify: `backend/services/a2v_pipeline/distilled_a2v_pipeline.py`
- Modify: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- Modify: `backend/services/retake_pipeline/ltx_retake_pipeline.py`
- Modify: `backend/services/ltx_pipeline_common.py`
- Test: `backend/tests/test_ltx_split_safetensors.py`

**Step 1: Wrapper constructors accept `ResolvedLtxComponents`**

Replace raw params:

```python
checkpoint_path: str, gemma_root: str | None, upsampler_path: str
```

with:

```python
components: ResolvedLtxComponents
```

**Step 2: Use `checkpoint_path_arg(components)`**

For all `PromptEncoder`, `DiffusionStage`, `ImageConditioner`, `VideoDecoder`, `AudioDecoder`, pass single path or tuple.

**Step 3: Add split validation tests**

Fake component files. Assert tuple ordering:

1. transformer
2. text projection / connector
3. video VAE
4. audio VAE

**Step 4: Run tests**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_ltx_split_safetensors.py tests/test_generation.py tests/test_ic_lora.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

**Manual smoke:** with real Kijai files, run one 540p/5s fast T2V using API text encoding.

---

## Milestone 6: GGUF Transformer Loading

**Purpose:** Add GGUF transformer support for `ltxv` files while keeping VAE/projection components in safetensors.

### Task 6.1: Add dependency + GGUF loader shell

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/uv.lock`
- Create: `backend/services/gguf_dequant.py`
- Create: `backend/services/gguf_transformer_loader.py`
- Test: `backend/tests/test_gguf_transformer_loader.py`

**Step 1: Add dependency**

Add:

```toml
"gguf>=0.13.0"
```

Run:

```bash
cd /tmp/clones/LTX-Desktop/backend && uv lock
```

**Step 2: Implement loader gates**

`GGUFStateDictLoader` must:

- import `gguf`
- require file suffix `.gguf`
- require metadata `general.architecture == "ltxv"`
- require usable config metadata initially; if missing, return deterministic `GGUF_CONFIG_METADATA_MISSING`
- strip `model.diffusion_model.` prefix if present

**Step 3: Implement tensor/linear minimal path**

- `GGUFQuantizedTensor`
- `GGUFLinear`
- `replace_linear_with_gguf(module)`
- `build_gguf_transformer_builder(components)`

Ponytail simplification: first version dequantizes per forward; add caching only after measurement.

**Step 4: Run tests**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_gguf_transformer_loader.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

### Task 6.2: Use GGUF builder in fast pipeline

**Files:**
- Modify: `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
- Modify: `backend/handlers/pipelines_handler.py`
- Test: `backend/tests/test_gguf_transformer_loader.py`
- Test: `backend/tests/test_generation.py`

**Step 1: Transformer selection**

If `components.transformer_format == "gguf"`, use custom transformer builder in `DiffusionStage`.

**Step 2: Disable unsupported combos**

For first GGUF pass:

- no `torch.compile`
- no LoRA/IC-LoRA fusion
- no GGUF LoRA patching

Raise clear errors:

- `GGUF_LORA_UNSUPPORTED`
- `GGUF_COMPILE_UNSUPPORTED`

**Step 3: Run tests**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_gguf_transformer_loader.py tests/test_generation.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

**Manual smoke:** activate QuantStack Q4_K_M profile plus Kijai VAE/text projection; generate short low-res video with API text encoding.

---

## Milestone 7: GGUF-Safe LoRA Patching

**Purpose:** Let official/user LoRAs work with GGUF transformer profiles.

### Task 7.1: Runtime LoRA patcher for GGUF

**Files:**
- Create: `backend/services/gguf_lora_patcher.py`
- Modify: `backend/services/gguf_transformer_loader.py`
- Modify: `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- Test: `backend/tests/test_gguf_lora_patcher.py`

**Contract:**

- Base GGUF weights stay quantized.
- LoRA delta applies on dequantized weight during forward.
- Use official `LTXV_LORA_COMFY_RENAMING_MAP` for key compatibility.
- Support adapter path + strength.

Reference behavior: `city96/ComfyUI-GGUF` `GGUFModelPatcher.patch_weight_to_device`.

**Validation:**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_gguf_lora_patcher.py tests/test_ic_lora.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

---

## Milestone 8: Local Text Encoder Options

**Purpose:** Avoid forcing 25GB Gemma download while preserving fully-local option.

### Task 8.1: Text encoder profile support

**Files:**
- Modify: `backend/handlers/text_handler.py`
- Modify: `backend/services/text_encoder/ltx_text_encoder.py`
- Modify: `backend/tests/test_generation.py`
- Modify: `backend/tests/test_models.py`

**Supported modes:**

1. API text encoding — current recommended default.
2. Existing local Gemma HF folder — user-selected profile path.
3. Future GGUF/quantized text encoder — schema only unless loader proven.

**Validation:**

```bash
cd /tmp/clones/LTX-Desktop && pnpm backend:test -- tests/test_generation.py tests/test_models.py
cd /tmp/clones/LTX-Desktop && pnpm typecheck:py
```

Expected: pass.

---

## Milestone 9: Final Integration and Release Readiness

### Task 9.1: End-to-end profile matrix

**Files:**
- Create: `backend/tests/test_model_profile_integration.py`
- Update docs/research only if needed.

**Matrix:**

| Profile | Expected status |
|---|---|
| official monolith + API text | ready |
| official monolith + local Gemma | ready |
| Kijai split + API text | ready if component files exist |
| GGUF + Kijai components + API text | experimental ready if GGUF loader passes |
| GGUF + IC-LoRA before patcher | explicit unsupported |
| Missing adapter for selected pipeline | explicit missing adapter |

**Validation:**

```bash
cd /tmp/clones/LTX-Desktop && pnpm typecheck
cd /tmp/clones/LTX-Desktop && pnpm backend:test
cd /tmp/clones/LTX-Desktop && pnpm build:frontend
```

Expected: green.

---

## Recommended Implementation Order

1. Backend model profiles.
2. Official adapter registry.
3. Frontend source/profile/adapters UX.
4. Component bundle seam.
5. Kijai split safetensors.
6. GGUF transformer.
7. GGUF LoRA patching.
8. Text encoder profile refinements.
9. Full validation and manual GPU smoke tests.

## Manual GPU Smoke Tests

Run only when model files are available.

### Official profile

- Full distilled checkpoint
- Spatial upsampler
- API text encoding
- Generate 540p/5s T2V

### Kijai split profile

- `ltx-2.3-22b-distilled*_transformer_only_fp8_*`
- `ltx-2.3_text_projection_bf16.safetensors`
- `LTX23_video_vae_bf16.safetensors`
- `LTX23_audio_vae_bf16.safetensors`
- upsampler
- Generate 540p/5s T2V

### GGUF profile

- `LTX-2.3-dev-Q4_K_M.gguf`
- Kijai text projection + VAEs
- API text encoding
- Generate smallest supported video first

## Open Decisions

- Whether profile path responses need redaction in non-admin APIs. Current plan admin-gates profile APIs.
- Whether adapter downloads should be separate endpoints or reuse checkpoint downloader. Current plan starts with status/browse; download can follow.
- Whether Kijai split supports all A2V/retake audio components cleanly. Validate before exposing those capabilities.
- Whether QuantStack GGUF includes config metadata. If not, add explicit config sidecar instead of guessing.
