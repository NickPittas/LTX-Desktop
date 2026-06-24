# Planner Report

## Status
ready

## Rationale
This slice is limited to backend profile schema, JSON persistence, CRUD/validation API, and the existing LTX recommendation gate. It avoids pipeline-loader changes, frontend wiring, GGUF loading, and adapter registry work; active profiles only satisfy recommendations when they describe an official monolithic local setup that current pipelines can already support.

# Task Packet

## User Goal
Plan slice A for LTX-Desktop fork: backend model profiles and local component path support, scoped to backend profile storage/API/recommendation gate only.

## Mode
general-coding

## Relevant Locations
- file: `/tmp/clones/LTX-Desktop/backend/api_types.py`
  symbol: `ModelCheckpointID`, recommendation response DTOs, request models
  approximate lines: 11-19, 226-257, 333-355
  stable anchor: `class LtxDownloadRecommendationResponse(BaseModel):`
  reason: Add model-profile DTO/schema names beside existing backend API models.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/_routes/models.py`
  symbol: model recommendation/download routes
  approximate lines: 24-72
  stable anchor: `router = APIRouter(prefix="/api", tags=["models"])`
  reason: Existing endpoint style and recommendation route names to preserve.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/app_factory.py`
  symbol: route imports and `app.include_router(...)`
  approximate lines: 16-25, 154-163
  stable anchor: `app.include_router(models_router)`
  reason: New model-profile router must be imported and registered.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/app_handler.py`
  symbol: `AppHandler.__init__`, `load_persistent_state`
  approximate lines: 39-214
  stable anchor: `# Handlers (wired in dependency order)`
  reason: Wire `ModelProfilesHandler`, inject it into `ModelsHandler`, and load persistent profile store.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/handlers/__init__.py`
  symbol: handler exports
  approximate lines: 3-31
  stable anchor: `__all__ = [`
  reason: Export new handler following existing handler package convention.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/handlers/models_handler.py`
  symbol: `ModelsHandler.get_ltx_recommendation`, `get_text_encoder_recommendation`
  approximate lines: 35-170
  stable anchor: `def get_ltx_recommendation(self) -> LtxRecommendationResponse:`
  reason: Existing recommendation gate must return OK when active official profile validates.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/state/app_state_types.py`
  symbol: `AppState`
  approximate lines: 207-220
  stable anchor: `class AppState:`
  reason: Store loaded profiles and active profile id in centralized runtime state.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/handlers/settings_handler.py`
  symbol: JSON persistence pattern
  approximate lines: 19-67
  stable anchor: `def load_settings(self, default_settings: AppSettings) -> AppSettings:`
  reason: Reuse JSON load/save error-handling style for `<app_data>/model_profiles.json`.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/tests/conftest.py`
  symbol: `test_state` fixture
  approximate lines: 26-83
  stable anchor: `config = RuntimeConfig(`
  reason: Tests get isolated `app_data_dir`; profile store should live there.
  confidence: high
- file: `/tmp/clones/LTX-Desktop/backend/tests/test_models.py`
  symbol: `TestRecommendations`
  approximate lines: 23-86
  stable anchor: `class TestRecommendations:`
  reason: Add/adjust recommendation assertions for active valid profile behavior.
  confidence: high
- file: `/home/npittas/ltx_offline/research/05-ltx-desktop-model-profiles-and-gguf-kijai-plan.md`
  symbol: sections 8, 12, 13
  approximate lines: 8.1-13
  stable anchor: `## 8. Recommended Data Model: Model Profiles`
  reason: Source schema, validation, endpoint, and phase-A scope requirements.
  confidence: high
- file: `/home/npittas/ltx_offline/research/06-revised-ltx-desktop-implementation-roadmap.md`
  symbol: Milestone 1 / First Implementation Slice
  approximate lines: 5-31, 112-119
  stable anchor: `## First Implementation Slice`
  reason: Confirms backend slice scope and excludes GGUF for first PR.
  confidence: high

## Allowed Edit Files
- `/tmp/clones/LTX-Desktop/backend/api_types.py`
- `/tmp/clones/LTX-Desktop/backend/state/app_state_types.py`
- `/tmp/clones/LTX-Desktop/backend/handlers/model_profiles_handler.py` (create)
- `/tmp/clones/LTX-Desktop/backend/handlers/models_handler.py`
- `/tmp/clones/LTX-Desktop/backend/handlers/__init__.py`
- `/tmp/clones/LTX-Desktop/backend/_routes/model_profiles.py` (create)
- `/tmp/clones/LTX-Desktop/backend/app_handler.py`
- `/tmp/clones/LTX-Desktop/backend/app_factory.py`
- `/tmp/clones/LTX-Desktop/backend/tests/test_model_profiles.py` (create)
- `/tmp/clones/LTX-Desktop/backend/tests/test_models.py`

## Read-Only Context Files
- `/tmp/clones/LTX-Desktop/AGENTS.md`
- `/tmp/clones/LTX-Desktop/CLAUDE.md`
- `/tmp/clones/LTX-Desktop/backend/_routes/models.py`
- `/tmp/clones/LTX-Desktop/backend/_routes/_admin_guard.py`
- `/tmp/clones/LTX-Desktop/backend/handlers/settings_handler.py`
- `/tmp/clones/LTX-Desktop/backend/handlers/base.py`
- `/tmp/clones/LTX-Desktop/backend/tests/conftest.py`
- `/tmp/clones/LTX-Desktop/backend/tests/test_settings.py`
- `/tmp/clones/LTX-Desktop/backend/tests/test_response_models.py`
- `/home/npittas/ltx_offline/research/05-ltx-desktop-model-profiles-and-gguf-kijai-plan.md`
- `/home/npittas/ltx_offline/research/06-revised-ltx-desktop-implementation-roadmap.md`

## Required Change
Implement one backend slice:

1. Add DTO/schema names in `backend/api_types.py`:
   - `ModelProfileId = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=80, pattern=r"^[A-Za-z0-9_.-]+$")]`
   - `ModelProfileFamily = Literal["ltx-2", "ltx-2.3", "ltxv2", "custom"]`
   - `ModelProfileSource = Literal["official", "kijai", "quantstack", "custom"]`
   - `ModelProfileTransformerFormat = Literal["official_safetensors", "split_safetensors", "gguf"]`
   - `ModelProfileTextEncoderFormat = Literal["hf_folder", "safetensors", "gguf", "api"]`
   - `ModelProfileCapability = Literal["t2v", "i2v", "a2v", "retake", "ic_lora", "local_text", "gguf"]`
   - `ModelComponentPaths(BaseModel)` with fields: `transformer: str | None = None`, `transformer_format: ModelProfileTransformerFormat = "official_safetensors"`, `transformer_quantization: str | None = None`, `upsampler: str | None = None`, `text_encoder_root: str | None = None`, `text_encoder_format: ModelProfileTextEncoderFormat = "api"`, `text_projection: str | None = None`, `embeddings_connector: str | None = None`, `video_vae: str | None = None`, `audio_vae: str | None = None`, `vocoder: str | None = None`, `ic_lora_union: str | None = None`, `ic_lora_motion_track: str | None = None`, `ic_lora_ingredients: str | None = None`, `ic_lora_hdr: str | None = None`, `ic_lora_hdr_scene_embeddings: str | None = None`, `ic_lora_lipdub: str | None = None`, `ic_lora_in_outpainting: str | None = None`, `official_adapters: dict[str, str] = Field(default_factory=dict)`, `depth_processor: str | None = None`, `pose_processor: str | None = None`, `person_detector: str | None = None`.
   - `ModelProfilePayload(BaseModel)` with `id`, `display_name`, `family`, `source`, `components`, `capabilities: set[ModelProfileCapability] = Field(default_factory=set)`.
   - `ModelProfilePatchPayload(BaseModel)` with optional `display_name`, `family`, `source`, `components`, `capabilities`; forbid extra fields.
   - `ModelProfilesResponse(BaseModel)` with `active_model_profile_id: str | None`, `profiles: list[ModelProfilePayload]`.
   - `ModelProfileValidationIssuePayload(BaseModel)` with `field: str`, `message: str`.
   - `ModelProfileValidationResponse(BaseModel)` with `profile_id: str`, `valid: bool`, `issues: list[ModelProfileValidationIssuePayload]`.
   - `ModelProfileActivateResponse(BaseModel)` with `status: Literal["ok"]`, `active_model_profile_id: str`.

2. Add state in `backend/state/app_state_types.py`:
   - Import `ModelProfilePayload` under existing API type imports.
   - Add `model_profiles: list[ModelProfilePayload] = field(default_factory=list)` and `active_model_profile_id: str | None = None` to `AppState`.

3. Create `backend/handlers/model_profiles_handler.py`:
   - Subclass `StateHandlerBase`.
   - Persist to `self.config.app_data_dir / "model_profiles.json"` using JSON shape `{"active_model_profile_id": str | null, "profiles": [...]}`.
   - Methods: `load_profiles()`, `save_profiles()`, `list_profiles()`, `create_profile(req: ModelProfilePayload)`, `patch_profile(profile_id: str, req: ModelProfilePatchPayload)`, `delete_profile(profile_id: str)`, `activate_profile(profile_id: str)`, `validate_profile_by_id(profile_id: str)`, `validate_profile(profile: ModelProfilePayload)`, `has_valid_active_official_profile()`.
   - Error codes via `HTTPError`: `MODEL_PROFILE_NOT_FOUND` (404), `MODEL_PROFILE_ALREADY_EXISTS` (409), `MODEL_PROFILE_INVALID` (409), `ACTIVE_MODEL_PROFILE_DELETE_FORBIDDEN` (409).
   - Validation rules: every non-empty component path must exist; `transformer_format="official_safetensors"` requires `components.transformer` exists and suffix `.safetensors`; `split_safetensors` requires `.safetensors`; `gguf` requires `.gguf`; `upsampler`, LoRA, VAE, projection/connector file fields require `.safetensors` when present; `text_encoder_format="hf_folder"` requires `text_encoder_root` exists and is a directory; `text_encoder_format="api"` allows empty `text_encoder_root` but recommendation-gate validity requires `ltx_api_key` to be present; `official` source used for gate must include existing `transformer` and `upsampler`, plus either API text with key or local HF folder.
   - Keep validation path-only; do not import torch, pipelines, or model loaders.

4. Wire handler:
   - `backend/handlers/__init__.py`: export `ModelProfilesHandler`.
   - `backend/app_handler.py`: construct `self.model_profiles = ModelProfilesHandler(...)` before `self.models`; pass it to `ModelsHandler`; call `self.model_profiles.load_profiles()` inside `load_persistent_state()` after settings load and before HF token load.
   - `backend/handlers/models_handler.py`: update constructor to accept `model_profiles_handler`; in `get_ltx_recommendation()`, after `_ensure_local_model_mode()` and before official checkpoint checks, return `LtxOkRecommendationResponse(status="ok")` when `model_profiles_handler.has_valid_active_official_profile()` is true. Do not change download/upgrade semantics when no valid active profile exists. Leave image-gen recommendation unchanged.

5. Add API route `backend/_routes/model_profiles.py`:
   - `router = APIRouter(prefix="/api/model-profiles", tags=["model-profiles"])`.
   - Endpoints:
     - `GET /api/model-profiles` -> `ModelProfilesResponse`
     - `POST /api/model-profiles` -> `ModelProfilePayload`
     - `PATCH /api/model-profiles/{profile_id}` -> `ModelProfilePayload`
     - `DELETE /api/model-profiles/{profile_id}` -> `StatusResponse`
     - `POST /api/model-profiles/{profile_id}/validate` -> `ModelProfileValidationResponse`
     - `POST /api/model-profiles/{profile_id}/activate` -> `ModelProfileActivateResponse`
   - Use `guard_admin_permission(request)` for every endpoint because responses and validation expose arbitrary local filesystem paths.
   - `backend/app_factory.py`: import router and include it.

6. Tests:
   - Create `backend/tests/test_model_profiles.py` covering: empty list, admin guard, create/list persistence to `<app_data>/model_profiles.json`, duplicate create 409, patch, delete, activate valid profile, invalid activation 409, validate missing path reports `valid=false`, validate bad extension reports issue, API text encoder without key does not satisfy activation/gate, local HF folder satisfies gate.
   - Update `backend/tests/test_models.py` `TestRecommendations` with one active official profile case: create real temp `.safetensors` transformer/upscaler and Gemma folder, activate profile, assert `GET /api/models/ltx-recommendation` returns `{"status": "ok"}` even without official checkpoint files under `default_models_dir`.
   - Preserve existing recommendation tests for no profile and downloaded official bundle.

## Non-Goals
- No frontend, Electron, preload, or first-run UI changes.
- No pipeline wrapper changes and no generation request routing through profiles.
- No GGUF loader/dependency work.
- No Kijai split safetensors loading beyond schema storage and path validation.
- No official adapter registry.
- No changes to existing official model download endpoints.
- No broad settings schema migration; profiles live in separate `model_profiles.json`.

## Validation
Commands:
- `cd /tmp/clones/LTX-Desktop && rtk pnpm backend:test -- tests/test_model_profiles.py tests/test_models.py`
- `cd /tmp/clones/LTX-Desktop && rtk pnpm typecheck:py`

Expected result:
- New model-profile tests pass.
- Existing model recommendation tests still pass.
- Pyright strict check passes with new DTOs, state fields, handler wiring, and route imports.

## Stop Conditions
Stop and report if:
- target symbol is missing
- required fix exceeds allowed files
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet
- adding profile storage requires frontend/electron edits to make backend tests pass
- implementing generation from profiles requires pipeline-loader changes
- security review rejects exposing local paths without admin guard

## Planner Self-Check
- locator evidence sufficient: yes; backend route/handler/state/test anchors and research slice anchors inspected.
- allowed edit files minimal and explicit: yes; ten files, all directly needed for schema, state, handler, route registration, recommendation gate, and tests.
- read-only context minimal: yes; only repo instructions, backend patterns, current routes/handlers/tests, and relevant research files.
- anchors/lines included: yes; each relevant location includes path, symbol, approximate lines, stable anchor, reason, confidence.
- validation concrete: yes; targeted backend pytest command plus Python typecheck.
- parallelization decision explicit and safe: yes; single task because route, handler, AppHandler wiring, state, and recommendation tests share files and must land atomically.
- non-goals and stop conditions sufficient: yes; excludes frontend, loaders, GGUF, adapters, and pipeline use.
- reviewer findings addressed, if revision: not applicable; no prior reviewer findings supplied.

## Required Return Contract
Return only a task-focused summary. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks.
