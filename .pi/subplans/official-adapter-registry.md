# Planner Report

## Status
split-required

## Why Split / Parallelize
Slice B touches backend config/API/tests plus optional frontend consumption. Backend registry and validation must land first because OpenAPI and UI types depend on new response models. Keep UI work second; do not implement new HDR/LipDub/inpaint pipelines in this slice.

## Interference Check
- parallel safe: no
- shared files or generated outputs: `backend/api_types.py`, `backend/_routes/models.py`, `backend/handlers/models_handler.py`, `frontend/generated/backend-openapi.json`, `frontend/generated/backend-openapi.ts`
- shared validation state: OpenAPI generation and backend tests share backend schema/fakes
- worktree isolation required: no, if tasks run sequentially
- rationale: generated OpenAPI and frontend API client depend on backend schema shape, so backend task must complete first.

## Proposed Task Sequence Or Parallel Batch
1. Task name: backend official adapter registry + pipeline-aware validation
   - purpose: Add official LTX-2.3 adapter registry, local-path/model-root resolution, status/recommendation endpoints, and tests without forcing adapter downloads at first-run.
   - allowed files:
     - `backend/api_types.py`
     - `backend/runtime_config/model_download_specs.py`
     - `backend/handlers/models_handler.py`
     - `backend/_routes/models.py`
     - `backend/state/app_settings.py`
     - `backend/handlers/ic_lora_handler.py`
     - `backend/tests/conftest.py`
     - `backend/tests/test_model_download_specs.py`
     - `backend/tests/test_models.py`
     - `backend/tests/test_ic_lora.py`
     - `frontend/generated/backend-openapi.json`
     - `frontend/generated/backend-openapi.ts`
   - API endpoints:
     - `GET /api/models/adapters/status?pipeline=<AdapterPipeline?>`
       - response: `AdapterStatusResponse { adapters: list[AdapterStatusItem] }`
     - `GET /api/models/adapters/recommendation?pipeline=<AdapterPipeline>`
       - response: `AdapterRecommendationResponse { pipeline, ready, adapters_to_download, processor_cps_to_download, missing }`
     - keep `GET /api/models/ltx-ic-lora-recommendation` as legacy compatibility, internally backed by adapter validation for current IC-LoRA panel needs.
   - data structures:
     - `AdapterID = Literal["distilled_lora_384", "distilled_lora_384_1_1", "union_control", "motion_track_control", "ingredients", "water_simulation", "decompression", "deblur", "colorization", "day_to_night", "in_outpainting", "instant_shave", "cross_eyed", "hdr", "hdr_scene_embeddings", "lipdub"]`
     - `AdapterKind = Literal["lora", "ic_lora", "distilled_lora", "embeddings"]`
     - `AdapterSource = Literal["official", "kijai", "custom"]`
     - `AdapterPipeline = Literal["fast", "dev_distilled_lora", "ic_lora_canny", "ic_lora_depth", "ic_lora_pose", "motion_track", "ingredients", "hdr", "lipdub", "in_outpainting", "decompression", "deblur", "colorization", "water_simulation", "day_to_night", "instant_shave", "cross_eyed"]`
     - runtime dataclass `AdapterComponent(id, display_name, kind, source, repo_id, filename, relative_path, required_for=frozenset(), optional_for=frozenset(), expected_size_bytes=None)`; use `frozenset` to avoid mutable defaults.
     - API models `AdapterComponentPayload`, `AdapterStatusItem`, `AdapterRequirementItem`, `AdapterStatusResponse`, `AdapterRecommendationResponse`.
     - `AppSettings.adapter_paths: dict[AdapterID, str] = {}` for source-priority item 1; no UI required in task 1.
   - registry entries:
     - `distilled_lora_384`: repo `Lightricks/LTX-2.3`, file `ltx-2.3-22b-distilled-lora-384.safetensors`, kind `distilled_lora`, optional/required only for future `dev_distilled_lora`.
     - `distilled_lora_384_1_1`: repo `Lightricks/LTX-2.3`, file `ltx-2.3-22b-distilled-lora-384-1.1.safetensors`, kind `distilled_lora`, preferred for `dev_distilled_lora`.
     - `union_control`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control`, file `ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors`, required for `ic_lora_canny`, `ic_lora_depth`, `ic_lora_pose`.
     - `motion_track_control`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control`, file `ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors`, required for `motion_track`.
     - `ingredients`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Ingredients`, file `ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors`, required for `ingredients`.
     - `water_simulation`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Water-Simulation`, file `ltx-2.3-22b-ic-lora-water-simulation-0.9.safetensors`, required for `water_simulation`.
     - `decompression`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Decompression`, file `ltx-2.3-22b-ic-lora-decompression-0.9.safetensors`, required for `decompression`.
     - `deblur`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Deblur`, file `ltx-2.3-22b-ic-lora-deblur-0.9.safetensors`, required for `deblur`.
     - `colorization`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Colorization`, file `ltx-2.3-22b-ic-lora-colorization-0.9.safetensors`, required for `colorization`.
     - `day_to_night`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Day-To-Night`, file `ltx-2.3-22b-ic-lora-day-to-night-0.9.safetensors`, required for `day_to_night`.
     - `in_outpainting`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-In-Outpainting`, file `ltx-2.3-22b-ic-lora-in-outpainting-0.9.safetensors`, required for `in_outpainting`.
     - `instant_shave`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Instant-Shave`, file `ltx-2.3-22b-ic-lora-instant-shave-0.9.safetensors`, required for `instant_shave`.
     - `cross_eyed`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-Cross-Eyed`, file `ltx-2.3-22b-ic-lora-cross-eyed-0.9.safetensors`, required for `cross_eyed`.
     - `hdr`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`, file `ltx-2.3-22b-ic-lora-hdr-0.9.safetensors`, required for `hdr`.
     - `hdr_scene_embeddings`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`, file `ltx-2.3-22b-ic-lora-hdr-scene-emb.safetensors`, kind `embeddings`, required for `hdr`.
     - `lipdub`: repo `Lightricks/LTX-2.3-22b-IC-LoRA-LipDub`, file `ltx-2.3-22b-ic-lora-lipdub-0.9.safetensors`, required for `lipdub`.
   - validation rules:
     - Source priority: `app_settings.adapter_paths[adapter_id]` existing file, else models root expected filename, else missing.
     - `fast` returns ready with no adapter requirements; do not force distillation LoRAs for current full-distilled fast pipeline.
     - `dev_distilled_lora` reports `distilled_lora_384_1_1`; no pipeline implementation.
     - `ic_lora_depth` requires `union_control` plus `dpt-hybrid-midas` processor.
     - `ic_lora_canny` requires `union_control`; preserve legacy `/ltx-ic-lora-recommendation` behavior if existing runtime still requires `dpt-hybrid-midas` to instantiate IC-LoRA pipeline.
     - `ic_lora_pose` reports `union_control`, `yolox-l-torchscript`, and `dw-ll-ucoco-384-bs5`; no pose UI/pipeline implementation.
     - `hdr` requires both `hdr` and `hdr_scene_embeddings`.
   - tests:
     - `test_model_download_specs.py`: registry covers every `AdapterID`; filenames unique except deliberate HDR same repo; legacy `LtxIcLorasSpec` aliases all point to `union_control`/existing union checkpoint; `fast` has no adapter requirements.
     - `test_models.py`: status endpoint lists official adapters; pipeline recommendation for `ic_lora_depth`, `ic_lora_canny`, `ic_lora_pose`, `hdr`, `lipdub`; app setting local path makes adapter downloaded; model-root file makes adapter downloaded; legacy ltx IC-LoRA recommendation remains compatible.
     - `test_ic_lora.py`: existing depth/canny generation/extraction still passes; missing union adapter returns scoped missing-component error if handler validation is changed.
   - validation:
     - `rtk pnpm backend:test -- tests/test_model_download_specs.py tests/test_models.py tests/test_ic_lora.py`
     - `rtk pnpm typecheck:py`
     - `rtk pnpm openapi:generate`
     - `rtk pnpm typecheck:ts`
   - can run in parallel with: none
2. Task name: frontend consume pipeline-aware adapter status
   - purpose: Use backend adapter recommendation in current IC-LoRA panel and show adapter names instead of raw checkpoint IDs.
   - allowed files:
     - `frontend/components/ICLoraPanel.tsx`
     - `frontend/hooks/use-hf-model-access.ts` only if backend still needs HF access checks for adapter repos
     - `frontend/lib/api-client.ts` only if generated endpoint client is not enough
   - validation:
     - `rtk pnpm typecheck:ts`
     - manual: switch Canny/Depth in IC-LoRA panel and confirm missing-resource copy changes by selected pipeline
   - can run in parallel with: none; depends on task 1 OpenAPI/types.
3. Task name: official adapters settings checklist
   - purpose: Optional follow-up UI for Settings → Models → Official Adapters status list.
   - allowed files:
     - `frontend/components/SettingsModal.tsx`
     - `frontend/contexts/AppSettingsContext.tsx` if surfacing `adapterPaths`
   - validation:
     - `rtk pnpm typecheck:ts`
     - manual: open Settings and verify all known adapters display status.
   - can run in parallel with: none; depends on task 1.

## Task Packets

# Task Packet

## User Goal
Add official LTX-2.3 adapter/LoRA registry and pipeline-aware adapter validation for LTX-Desktop without forced first-run downloads or new generation pipelines.

## Mode
general-coding

## Relevant Locations
- file: `backend/runtime_config/model_download_specs.py`
  symbol: `LtxIcLorasSpec`, `LTXLocalModelSpec`, `get_model_cp_spec`, `get_ic_loras_cp_ids`, path helpers
  approximate lines: 53-305
  stable anchor: `class LtxIcLorasSpec:`
  reason: current IC-LoRA model config and model-root path resolution live here.
  confidence: high
- file: `backend/api_types.py`
  symbol: `ModelCheckpointID`, `LtxIcLoraRecommendationResponse`, `ConditioningType`
  approximate lines: 10-250, 400-430
  stable anchor: `ModelCheckpointID = Literal[`
  reason: add adapter IDs, pipeline literals, response models, and endpoint schemas.
  confidence: high
- file: `backend/handlers/models_handler.py`
  symbol: `ModelsHandler.get_ltx_ic_lora_recommendation`
  approximate lines: 190-201
  stable anchor: `def get_ltx_ic_lora_recommendation`
  reason: add adapter status/recommendation methods and preserve legacy recommendation behavior.
  confidence: high
- file: `backend/_routes/models.py`
  symbol: models router
  approximate lines: 1-83
  stable anchor: `router = APIRouter(prefix="/api", tags=["models"])`
  reason: add `GET /api/models/adapters/status` and `GET /api/models/adapters/recommendation`.
  confidence: high
- file: `backend/state/app_settings.py`
  symbol: `AppSettings`, `SettingsResponse`
  approximate lines: 37-116
  stable anchor: `class AppSettings(SettingsBaseModel):`
  reason: add optional per-adapter local path overrides for validation source priority.
  confidence: high
- file: `backend/handlers/ic_lora_handler.py`
  symbol: `_require_ic_lora_model_paths`
  approximate lines: 80-94
  stable anchor: `def _require_ic_lora_model_paths`
  reason: optionally route current depth/canny checks through registry helpers while keeping runtime behavior compatible.
  confidence: high
- file: `backend/tests/test_models.py`
  symbol: `TestRecommendations`
  approximate lines: 21-115
  stable anchor: `def test_ic_lora_recommendation`
  reason: add endpoint and compatibility coverage.
  confidence: high
- file: `backend/tests/test_model_download_specs.py`
  symbol: model spec helper tests
  approximate lines: 1-100
  stable anchor: `def test_ic_lora_cp_ids_are_deduped`
  reason: add pure registry consistency tests.
  confidence: high
- file: `backend/tests/conftest.py`
  symbol: `create_fake_ic_lora_files`
  approximate lines: 144-165
  stable anchor: `def create_fake_ic_lora_files`
  reason: extend/add fake adapter file fixture for status tests.
  confidence: high
- file: `frontend/generated/backend-openapi.json`
  symbol: generated schema
  approximate lines: generated
  stable anchor: generated by `pnpm openapi:generate`
  reason: API schema must be regenerated after backend model changes.
  confidence: high
- file: `frontend/generated/backend-openapi.ts`
  symbol: generated TypeScript API types
  approximate lines: generated
  stable anchor: generated by `pnpm openapi:generate`
  reason: frontend typecheck depends on generated API types.
  confidence: high

## Allowed Edit Files
- `backend/api_types.py`
- `backend/runtime_config/model_download_specs.py`
- `backend/handlers/models_handler.py`
- `backend/_routes/models.py`
- `backend/state/app_settings.py`
- `backend/handlers/ic_lora_handler.py`
- `backend/tests/conftest.py`
- `backend/tests/test_model_download_specs.py`
- `backend/tests/test_models.py`
- `backend/tests/test_ic_lora.py`
- `frontend/generated/backend-openapi.json`
- `frontend/generated/backend-openapi.ts`

## Read-Only Context Files
- `/home/npittas/ltx_offline/research/07-official-ltx23-lora-registry.md`
- `AGENTS.md`
- `CLAUDE.md`
- `package.json`
- `frontend/components/ICLoraPanel.tsx`
- `frontend/lib/api-client.ts`

## Required Change
Implement task 1 from sequence above only. Add registry helpers, API response models, models handler methods, routes, app setting local-path support, tests, and generated OpenAPI. Keep old `LtxIcLorasSpec`/`get_ic_loras_cp_ids` behavior as compatibility alias for Union Control. Do not add actual HDR/LipDub/inpainting/motion-track generation.

## Non-Goals
- No new adapter download endpoint in this task.
- No forced first-run download of optional adapters.
- No new frontend settings checklist.
- No new generation pipelines, preprocessors, or UI workflows for HDR/LipDub/inpainting/motion tracking.
- No broad model download refactor from checkpoint IDs to adapter IDs.

## Validation
Commands, from `/tmp/clones/LTX-Desktop`:
- `rtk pnpm backend:test -- tests/test_model_download_specs.py tests/test_models.py tests/test_ic_lora.py`
- `rtk pnpm typecheck:py`
- `rtk pnpm openapi:generate`
- `rtk pnpm typecheck:ts`

Expected result:
Backend tests pass, pyright passes, OpenAPI generated files updated without manual edits, TypeScript typecheck passes.

## Stop Conditions
Stop and report if:
- target symbol is missing
- required fix exceeds allowed files
- validation cannot run
- existing architecture contradicts the requested change
- task requires product/design judgment not in packet
- implementing adapter download requires changing download session state beyond allowed scope
- supporting a listed pipeline requires adding actual generation/preprocessing code

## Required Return Contract
Return only a task-focused summary. Do not include transcript, tool logs, raw file dumps, large code blocks, or broad unrelated issues. Include status, files inspected/changed, validation evidence, blockers, and task-specific risks.
