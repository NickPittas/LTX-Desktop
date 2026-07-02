# backend/state

## Responsibility

Defines the canonical in-memory runtime state of the backend and the persisted
settings schema. This folder holds pure data models and one tiny singleton
accessor — it contains **no** business logic and **no** I/O of its own (the
single exception is `conditioning_cache.cleanup` deleting cached control-video
files). All mutation of `AppState` is performed by `handlers/` under the shared
`RLock`; `state/` only declares the shapes and the registry that exposes the
`AppHandler` owning the live state.

Files:
- `app_state_types.py` — `AppState` and every discriminated-union state machine.
- `app_settings.py` — pydantic `AppSettings` schema, patch/response models, API-routing helper.
- `conditioning_cache.py` — IC-LoRA control-video cache.
- `deps.py` — module-level `AppHandler` singleton accessor.

(Note: `state/__init__.py` re-exports `AppState`, `AppHandler`,
`build_initial_state`, `RuntimeConfig`, and the `deps` accessors, but
`AppHandler`/`build_initial_state` actually live in the backend root
`app_handler.py`.)

## Design Patterns

**Discriminated unions via `@dataclass` variants.** Each state machine is a
union of frozen/mutable dataclasses matched exhaustively by callers:
- `GenerationState = GenerationRunning | GenerationComplete | GenerationError |
  GenerationCancelled` (each carrying `id`; `GenerationRunning` also holds
  `GenerationProgress(phase, progress, current_step, total_steps)`,
  `GenerationComplete` carries `result: str | list[str]`).
- `ActiveGeneration = GpuGeneration | ApiGeneration`, each wrapping a
  `GenerationState` — separates GPU-pipeline jobs from remote-API jobs.
- `HfAuthState = HfNotAuthenticated | HfOAuthPending | HfAuthenticated`
  (the pending/authenticated variants are `frozen=True`).
- `DownloadSessionResult = DownloadSessionComplete | DownloadSessionError`.
- `GpuSlot.active_pipeline` is itself a union: `VideoPipelineState |
  ICLoraState | A2VPipelineState | RetakePipelineState |
  ImageGenerationPipeline` (the last imported as a Protocol from `services`).
- `LTXLocalModelRelevance = LTXLocalModelDeprecated | LTXLocalModelRelevant`
  lives in `runtime_config/model_download_specs.py` but is matched from
  `models_handler`.

**`NewType` for opaque IDs.** `DownloadSessionId = NewType("DownloadSessionId",
str)` keys `completed_download_sessions` and is constructed in
`DownloadHandler.start_download`.

**Pydantic v2 settings models with alias generation.** `app_settings.py`:
`SettingsBaseModel` uses `ConfigDict(alias_generator=_to_camel_case,
populate_by_name=True, validate_assignment=True, extra="ignore")`; the patch
subclass sets `extra="forbid"`. `_to_camel_case` has explicit special-case
aliases for `prompt_enhancer_enabled_t2v/i2v` (`...EnabledT2V`/`I2V`).
`field_validator`s clamp `prompt_cache_size` to 0–1000 and `locked_seed` to
0–2_147_483_647 (via `_clamp_int`).

**Dynamically generated patch model.** `make_partial_model(AppSettings)`
recursively wraps each field annotation in `T | None` (default `None`) and
caches the result in `_PARTIAL_MODEL_CACHE`; `UpdateSettingsRequest =
AppSettingsPatch` is this generated model.

**Not-thread-safe cache with caller-owned locking.** `ConditioningCache`
explicitly documents "caller is expected to hold the state lock"; it owns file
cleanup in `cleanup()` and `__del__`.

**Module-global singleton.** `deps.py` holds `_app_handler` and exposes
`init_state_service`/`get_state_service` (asserts initialised) /
`set_state_service_for_tests`. Route layers resolve the `AppHandler` (and thus
`AppState`) through `get_state_service()`.

## Data & Control Flow

**`app_state_types.AppState`** (top-level `@dataclass`) aggregates every runtime
facet:
- `downloading_session: DownloadingSession | None` — live download
  (`DownloadingSession` holds `id`, `current_running_file:
  FileDownloadRunning | None`, `files_to_download`/`completed_files` sets,
  `completed_bytes`).
- `completed_download_sessions: dict[DownloadSessionId, DownloadSessionResult]`
  (default factory) — terminal download outcomes for polling.
- `gpu_slot: GpuSlot | None` and `cpu_slot: CpuSlot | None` — device slots.
  `GpuSlot.active_pipeline` is the union above; `CpuSlot.active_pipeline` is
  restricted to `ImageGenerationPipeline` (parking target).
- `active_generation: ActiveGeneration | None` — current job.
- `text_encoder: TextEncoderState | None` — wraps `service: TextEncoder`,
  `prompt_cache: dict[tuple[str, bool], TextEncodingResult]` (keyed by
  `(prompt.strip(), enhance_prompt)`), `api_embeddings:
  TextEncodingResult | None`, `cached_encoder`. `TextEncodingResult` carries
  `video_context: torch.Tensor` and optional `audio_context`.
- `app_settings: AppSettings`.
- `hf_auth_state: HfAuthState` (default `HfNotAuthenticated()`).
- `model_profiles: list[ModelProfilePayload]` and
  `active_model_profile_id: str | None`.

Pipeline-state dataclasses: `VideoPipelineState(pipeline, is_compiled,
cache_key: tuple[str,...]=())`; `ICLoraState(pipeline, lora_paths: list[str],
lora_strength: float = 1.0, depth_pipeline=None, depth_model_path=None,
adapter_path=None, pose_resources: PoseResources | None = None,
conditioning_cache=ConditioningCache())` — **`lora_strength` defaults to `1.0`**
and is part of the cache-match key in `PipelinesHandler.load_ic_lora`
(epsilon `0.001`); `A2VPipelineState(pipeline)`; `RetakePipelineState(pipeline,
distilled, quantized)`; `PoseResources(pipeline, person_detector_model_path,
pose_model_path)`.

`TextEncodingResult` and `CachedTextEncoder` reference `torch.Tensor` /
`torch.device` under `TYPE_CHECKING` (no runtime torch import in this module).

**`app_settings.py`** control flow: `AppSettings` is the persisted schema
(`use_torch_compile`, `ltx_api_key`, `user_prefers_ltx_api_video_generations`,
`fal_api_key`, `use_local_text_encoder`, `prompt_cache_size`, prompt-enhancer
flags, `gemini_api_key`, `seed_locked`/`locked_seed`, `models_dir`,
`adapter_paths`). `SettingsResponse` mirrors it but replaces the three API keys
with `has_*_api_key` booleans; `to_settings_response(settings)` pops the keys
and sets the booleans (`models_dir` passes through). `should_video_generate_with_ltx_api(
force_api_generations, settings)` returns `force_api_generations or
(settings.user_prefers_ltx_api_video_generations and bool(ltx_api_key.strip()))`
— the single decision point consumed by `video_generation_handler` and
`retake_handler`.

**`conditioning_cache.py`** — `ConditioningCacheKey = NamedTuple(video_path:
str, conditioning_type: ConditioningType)`; `ConditioningCacheEntry =
NamedTuple(control_video_path: str, frame_count: int, fps: float)`.
`ConditioningCache` is a plain dict wrapper with `get`/`put`/`cleanup`
(unlinks each `control_video_path`, swallows errors, logs warning) and a
`__del__` finaliser. One instance lives on each `ICLoraState`, so the cache is
rebuilt whenever the IC-LoRA pipeline is evicted.

**`deps.py`** — `init_state_service(AppHandler)` stores the singleton,
`get_state_service()` asserts and returns it, `set_state_service_for_tests`
aliases `init_state_service` for test wiring.

## Integration Points

- **`handlers/`** — the sole mutator of `AppState`. `StateHandlerBase` holds the
  `state`/`lock` references; every discriminated union declared here is matched
  in handlers (e.g. `GenerationHandler`, `HuggingFaceAuthHandler`,
  `DownloadHandler`, `PipelinesHandler`).
- **`app_handler.py`** (backend root) — `build_initial_state` constructs the
  default `AppState` and `AppHandler` wires it into the singleton via
  `state/deps.init_state_service`.
- **`state/__init__.py`** re-exports `AppState`, `AppHandler`,
  `build_initial_state`, `RuntimeConfig`, and the `deps` accessors for
  convenient imports.
- **`api_types.py`** — supplies `ModelCheckpointID`, `ModelProfilePayload`,
  `ConditioningType` used in state definitions.
- **`services/interfaces.py`** — pipeline/encoder Protocols
  (`FastVideoPipeline`, `IcLoraPipeline`, `A2VPipeline`, `RetakePipeline`,
  `ImageGenerationPipeline`, `DepthProcessorPipeline`, `PoseProcessorPipeline`,
  `TextEncoder`) referenced under `TYPE_CHECKING` by `app_state_types.py`.
- **`runtime_config/model_download_specs.py`** — `ModelCheckpointID` consumers
  and adapter specs; `state` itself does not import it.
- **`state/app_settings.py` → `handlers/settings_handler.py`** +
  `_settings_utils.py` for load/merge/patch; `should_video_generate_with_ltx_api`
  → `video_generation_handler` / `retake_handler`.
- **`state/conditioning_cache.py` → `handlers/ic_lora_handler.py`** (cache
  populated/read on `ICLoraState`).
