# backend/handlers

## Responsibility

Business-logic layer for the FastAPI backend. Each domain handler owns one
concern and mutates the shared `AppState` under a single process-wide `RLock`.
Handlers sit between the thin route functions in `_routes/` (which only parse
HTTP input and serialise output) and the side-effect `services/` (GPU, network,
disk) plus `state/` (mutations). `app_handler.AppHandler` is the composition root
that constructs every handler with the same `(state, lock, config)` triple and
wires inter-handler dependencies.

Handlers in this folder:
- `base.py` — shared `StateHandlerBase` + `with_state_lock` decorator.
- `generation_handler.py` — generation lifecycle / progress state machine (GPU + API slots).
- `video_generation_handler.py` — local T2V/I2V/A2V and forced-API video generation.
- `ic_lora_handler.py` — IC-LoRA adapter orchestration and workflow dispatch.
- `retake_handler.py` — local + API video retake.
- `image_generation_handler.py` — local + API text-to-image (Z-Image / fal).
- `pipelines_handler.py` — GPU/CPU pipeline lifecycle and cache.
- `text_handler.py` — text-encoding routing (local Gemma vs LTX API embeddings + prompt cache).
- `models_handler.py` — checkpoint recommendation / upgrade / adapter resolution.
- `model_profiles_handler.py` — model profile CRUD, validation, activation, persistence.
- `download_handler.py` — checkpoint download sessions (staging + atomic commit).
- `settings_handler.py` + `_settings_utils.py` — settings load/save/patch + JSON merge helpers.
- `health_handler.py` — `/health` and GPU telemetry.
- `hf_auth_handler.py` + `hf_auth_utils.py` — HuggingFace PKCE OAuth + token requirement helper.
- `runtime_policy_handler.py` — exposes server-side forced-API policy.
- `suggest_gap_prompt_handler.py` — Gemini-powered timeline gap prompt suggestion.

## Design Patterns

**Lock-aware base (`base.py`).** `StateHandlerBase.__init__(state, lock, config)`
stores the shared `AppState`, `RLock`, and `RuntimeConfig`. Exposes `state`,
`lock`, `config` properties and a derived `models_dir` property
(`state.app_settings.models_dir` if set, else `config.default_models_dir`).
`with_state_lock` is a generic decorator (`ParamSpec`/`TypeVar` bound to
`StateHandlerBase`) that wraps a method in `with self.lock:`.

**Lock discipline (lock → read/validate → unlock → heavy work → lock → write).**
Short, fully-locked mutations use `@with_state_lock` (e.g. every method in
`GenerationHandler`, `SettingsHandler.update_settings`,
`DownloadHandler.start_file`/`finish_download`/`fail_download`).
Long-running handlers acquire the lock only for state checks/edits and release
it during heavy compute/IO:
- `PipelinesHandler.load_gpu_pipeline`/`load_ic_lora`/`load_a2v_pipeline`/
  `load_retake_pipeline`: lock to test cache hit, release, build pipeline
  (`FastVideoPipeline.create`, etc.), re-lock to install into `state.gpu_slot`.
  Heavy `create` and `gpu_cleaner.cleanup()` run **outside** the lock.
- `HuggingFaceAuthHandler._exchange_code`: locks to read/validate `HfOAuthPending`,
  releases for the `requests.post` token exchange, re-locks to store
  `HfAuthenticated`.
- `DownloadHandler.start_model_download`: briefly locks to reject
  `DOWNLOAD_ALREADY_RUNNING`, then dispatches `_download_worker` on
  `task_runner.run_background`; the worker uses locked helpers
  (`start_file`/`update_file_progress`/`finish_download`/`fail_download`) between
  unlocked download IO.

**Discriminated-union state machines** (types from `state/app_state_types.py`).
`GenerationHandler` is the canonical example: it operates on
`GenerationState = GenerationRunning | GenerationComplete | GenerationError |
GenerationCancelled`, wrapped in `ActiveGeneration = GpuGeneration |
ApiGeneration`. Internal matchers (`_gpu_generation`, `_api_generation`,
`_active_generation_state`, `_running_generation`, `_cancelled_generation`)
narrow the union; mutators transition states via `_set_generation_state(slot,
new_state)`. `HuggingFaceAuthHandler` mirrors this on
`HfAuthState = HfNotAuthenticated | HfOAuthPending | HfAuthenticated` via
`_set_hf_auth_state` (which also persists/clears `hf_auth_token.json`).
`DownloadHandler.get_download_progress` matches `DownloadSessionResult =
DownloadSessionComplete | DownloadSessionError`.

**HTTPError-with-`from exc` chaining.** `HTTPError(status_code, detail, code=...)`
(defined in `_routes/_errors.py`) carries `status_code`, `detail`, `code`, and a
pre-built `response`. Convention:
- Validation refusals raise `HTTPError` directly with no chaining
  (`raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")`,
  `HTTPError(400, "INVALID_VIDEO_GENERATION_SPEC")`).
- Wrapping an unexpected exception uses `raise HTTPError(500, str(e)) from e`
  (see `video_generation_handler.generate`, `ic_lora_handler.generate`/
  `_generate_ingredients`, `retake_handler._run_local_retake`,
  `image_generation_handler.generate`/`_generate_via_api`,
  `download_handler.check_model_access`→`HTTPError(exc.status_code, exc.detail) from exc`,
  `suggest_gap_prompt_handler.suggest_gap`→`HTTPError(504, ...) from exc`).
- Re-raising a caught `HTTPError` is bare (`except HTTPError: raise`), optionally
  after `fail_generation(e.detail)`. Boundary logging is owned by
  `app_factory.py`, so handlers do not `logger.exception(...)` then rethrow.

**Generation try/except/finally skeleton** is shared across
`video_generation_handler`, `ic_lora_handler`, `retake_handler`,
`image_generation_handler`:
`start_generation` → `update_progress` phases → pipeline call → on
`is_generation_cancelled()` unlink output and `raise RuntimeError("...cancelled")`
→ `complete_generation(result)`; `except HTTPError: fail_generation; raise`;
`except Exception: fail_generation; if "cancelled" → return *CancelledResponse
else raise HTTPError(500) from exc`; `finally: clear_api_embeddings()`.

**Inter-handler composition.** Handlers receive sibling handlers in their
constructor (e.g. `VideoGenerationHandler(state, lock, generation_handler,
pipelines_handler, text_handler, ltx_api_client, config)`) rather than importing
globals. Stateful side effects go through injected service protocols
(`LTXAPIClient`, `ZitAPIClient`, `ModelDownloader`, `TaskRunner`, `VideoProcessor`,
`GpuInfo`, `HTTPClient`, `GpuCleaner`).

## Data & Control Flow

**`generation_handler.GenerationHandler`** — pure state, no services. Owns the
`active_generation` slot. `start_generation(id)` (requires `gpu_slot` set) /
`start_api_generation(id)` create `GpuGeneration`/`ApiGeneration` wrapping
`GenerationRunning(progress=GenerationProgress(...))`; both reject if
`is_generation_running()`. `update_progress(phase, progress, current_step,
total_steps)` mutates the running progress. `cancel_generation()` returns
`CancelCancellingResponse` (Running→Cancelled) or
`CancelNoActiveGenerationResponse`. `complete_generation(result)` and
`fail_generation(error)` (Running→Complete/Error; `fail_generation` no-ops if
already Cancelled). `get_generation_progress()` maps the union to
`GenerationProgressResponse` (status `running`/`complete`/`cancelled`/`error`/`idle`).

**`video_generation_handler.VideoGenerationHandler`** — `generate(req)` first
calls `should_video_generate_with_ltx_api(force_api_generations,
state.app_settings)`; if API, delegates to `_generate_forced_api`. Local path:
validates via `validate_generate_video_request`, rejects 409 if running,
resolves resolution from `RESOLUTION_MAP_16_9` (`540p`→(960,544),
`720p`→(1280,704), `1080p`→(1920,1088); `9:16` swaps w/h), computes
`num_frames = _compute_num_frames(duration, fps) = max(((duration*fps)//8)*8+1, 9)`.
A2V branch (`_generate_a2v`) uses its own `RESOLUTION_MAP`
(`540p`→(960,576), `720p`→(1280,704), `1080p`→(1920,1088)) and validates audio
via `validate_audio_file`. `generate_video()` snaps height/width to multiples of
64, optionally writes a temp PNG for image conditioning, calls
`pipeline_state.pipeline.generate(...)` with `enhance_prompt` resolved from
`use_api_encoding and settings.prompt_enhancer_enabled_{i2v,t2v}`.
**Output path is hardcoded `.mp4`:**
`_make_output_path() = config.outputs_dir / f"ltx2_video_{timestamp}_{gen_id}.mp4"`
(relevant to upcoming EXR/MOV work).
`_resolve_seed()` honours `settings.seed_locked`/`locked_seed`, else `1000` in
`dev_mode`, else `int(time.time()) % 2147483647`.
`_generate_forced_api` maps `req.model` via `FORCED_API_MODEL_MAP`
(`fast`→`ltx-2-3-fast`, `pro`→`ltx-2-3-pro`) and resolution via
`FORCED_API_RESOLUTION_MAP` (1080p/1440p/2160p × 16:9/9:16), requires
`ltx_api_key`, then branches on audio/image presence to
`ltx_api_client.generate_audio_to_video`/`generate_image_to_video`/
`generate_text_to_video`, writes bytes via `_write_forced_api_video`.
`_map_ltx_api_generation_error` maps 402 `insufficient_funds_error` →
`HTTPError(402, ..., code="LTX_INSUFFICIENT_FUNDS")`.

**`ic_lora_handler.IcLoraHandler`** — `generate(req)` resolves `workflow` from
`_ADAPTER_WORKFLOW[adapter_id]` (`WorkflowType` literal). Workflows in
`_UNAVAILABLE_WORKFLOWS` (`motion_track_control`, `hdr_scene_embeddings`,
`lipdub`) are rejected with `_UNAVAILABLE_MESSAGES` before any path access.
`hdr` is a **supported** V2V workflow (no longer unavailable). `ingredients`
is dispatched **before** `video_path` validation to `_generate_ingredients`
(T2V: no video, no conditioning, uses `req` dims aligned via `_align_up(*, 64)`
and `_snap_frame_count` (min 9, 1+8k), loads only the adapter LoRA with
`lora_strength=req.lora_strength`). The `hdr` branch (`_generate_hdr`)
**ignores** prompt/audio/`conditioning_type`/`images` (an empty prompt is
fine), requires a source video, **never rejects by frame count** (the wrapper
decodes all frames and pads in memory to `8n+1`), gates to the **official
distilled safetensors** base only (`load_hdr_ic_lora` rejects dev/full/GGUF/
split/Kijai/QuantStack with `UNSUPPORTED_MODEL_BASE_FAMILY`/
`UNSUPPORTED_MODEL_FORMAT`), forwards `scene_embeddings_path` into the HDR
pipeline `create()` (and includes it in `_hdr_cache_key`), and returns
`video_path` = EXR dir + `proxy_path` = SDR MP4. All other non-ingredients
workflows **require** `req.video_path` (existence-checked) plus workflow-
specific guards: `union_control` requires `conditioning_type`, `in_outpainting`
requires `mask_path` (and is the only non-HDR workflow allowing an empty
prompt). `_resolve_base_lora_path` prefers the `union_control` adapter
path (profile→typed field→`models_dir`), falling back to the legacy
`_require_ic_lora_model_paths(conditioning_type, require_lora=True)` checkpoint.
`lora_paths` stacks base + adapter; `lora_strength` is forwarded into
`pipelines.load_ic_lora(..., lora_strength=req.lora_strength)`. For
`in_outpainting` it calls `ic_state.pipeline.generate_inpaint(...)`; otherwise
`pipeline.generate(...)` with `video_conditioning=[(control_video_path,
conditioning_strength)]`. Canny/depth control videos are built frame-by-frame
through `_build_conditioning_frame` (depth requires `ic_state.depth_pipeline`)
and cached in `ic_state.conditioning_cache` keyed by `ConditioningCacheKey(
video_path, conditioning_type)`. Alignment is 128 when conditioning (union
ref_downscale=2) else 64. **Output hardcoded `.mp4`:**
`config.outputs_dir / f"ic_lora_{timestamp}_{uuid}.mp4"`. `extract_conditioning`
reads a single frame, builds canny/depth, returns base64 JPEG data URIs.

**`retake_handler.RetakeHandler`** — `run(req)` validates `video_path`/
`duration>=2`/prompt, then branches on
`should_video_generate_with_ltx_api` to `_run_api_retake`
(`ltx_api_client.retake`, writes bytes) or `_run_local_retake`.
`_run_local_retake` validates 32-multiple dimensions
(`_validate_video_metadata` via `ltx_pipelines.utils.media_io.get_videostream_metadata`),
prepares text encoding, loads `load_retake_pipeline(distilled=True)`, calls
`pipeline.generate(..., distilled=True, regenerate_video/regenerate_audio from
_resolve_retake_mode(mode))`. **Output hardcoded `.mp4`:**
`config.outputs_dir / f"retake_{timestamp}_{gen_id}.mp4"`.

**`image_generation_handler.ImageGenerationHandler`** — `generate(req)` snaps
width/height to multiples of 16, clamps `numImages` to 1–12. If
`config.force_api_generations` → `_generate_via_api` (fal/Zit). Local path loads
`load_image_generation_pipeline_to_gpu`, loops `num_images` calling
`pipeline.generate(guidance_scale=0.0, seed=seed+i)`, saves PNGs
(`zit_image_{ts}_{uuid}.png`). API path writes `zit_api_image_{ts}_{uuid}.png`.

**`pipelines_handler.PipelinesHandler`** — owns `state.gpu_slot`/`cpu_slot`.
Cache-keyed reuse: `VideoPipelineState.cache_key` comes from
`_resolve_active_components().cache_key` (active profile) else `(model_id,)`;
`_pipeline_matches_model_type` compares `pipeline_kind` **and** `cache_key`.
`ICLoraState` reuse matches `lora_paths` + `depth_model_path` + `adapter_path`
**and** `abs(current_lora_strength - lora_strength) < 0.001`. `RetakePipelineState`
matches `distilled` + `quantized` (`device_supports_fp8`). `A2VPipelineState` has
no key (rebuilt if not already A2V). Image-gen pipelines park on CPU
(`park_image_generation_pipeline_on_cpu` sets `state.cpu_slot`) and reload to GPU
(`load_image_generation_pipeline_to_gpu`). Every load calls
`_install_text_patches_if_needed`, `_evict_gpu_pipeline_for_swap` (which runs
`gpu_cleaner.cleanup()` outside the lock), `_resolve_checkpoint_paths`, and
`_compile_if_enabled` (skips `torch.compile` on MPS).
`_assert_invariants` forbids an image-gen pipeline being in both slots.

**`text_handler.TextHandler`** — `should_use_local_encoding()` returns True if
the active profile provides a local encoder, else if both API+local are
available defers to `settings.use_local_text_encoder`, else whichever is
available. `prepare_text_encoding(prompt, enhance_prompt)` raises
`RuntimeError("TEXT_ENCODING_NOT_CONFIGURED...")` if neither available; calls
`_prepare_api_embeddings` which hits `te.service.encode_via_api(...)` (cached in
`te.prompt_cache` keyed by `(prompt.strip(), enhance_prompt)`) and stores into
`te.api_embeddings`. `clear_api_embeddings()` resets it (called in generation
`finally` blocks). `resolve_gemma_root()` returns the profile
`text_encoder_root` or the downloaded Gemma CP path.

**`models_handler.ModelsHandler`** — filesystem state + recommendations.
`get_ltx_recommendation()` short-circuits to `LtxOkRecommendationResponse` if
`has_valid_active_official_profile()`, else inspects
`get_downloaded_ltx_model_id(models_dir)` and emits
`LtxDownloadRecommendationResponse`/`LtxUpgradeRecommendationResponse` (computing
`_get_upgrade_dependency_downloads`/`_get_upgrade_delete_cp_ids` against the
latest model). `resolve_adapter_path(adapter_id)` precedence: settings
`adapter_paths` → active profile (`official_adapters`/typed field) →
`models_dir / adapter.filename`. `resolve_upgrade_download` re-validates the
upgrade request against the recommendation. `delete_checkpoints` refuses
`get_protected_cp_ids()` (current model's CPs). `_ensure_local_model_mode()`
refuses in `force_api_generations`.

**`model_profiles_handler.ModelProfilesHandler`** — persists
`config.app_data_dir/"model_profiles.json"`. `load_profiles` reads/repairs blank
IDs and re-saves. `validate_profile` walks `_PATH_FIELDS` (existence) and
`_SAFE_TENSORS_FIELDS` (`.safetensors` ext), plus transformer ext by
`transformer_format` and text-encoder dir/ext by `text_encoder_format`.
`activate_profile` requires `validate_profile().valid`. `has_valid_active_*`
helpers feed `models_handler`.

**`download_handler.DownloadHandler`** — `start_model_download(download_type,
cp_ids)`: rejects in force-api mode and on concurrent session, then for
`"upgrade"` calls `models_handler.resolve_upgrade_download` (atomic commit) and
for `"download"` computes missing CPs (`_discover_download_cp_ids`,
non-atomic). `_download_worker` stages to `resolve_downloading_dir` (`.downloading`)
then commits via `_commit_staged_checkpoint` (`rename` file/`shutil.rmtree`+rename
folder); atomic mode rolls back committed CPs on failure.
`_make_progress_callback` reports EMA-smoothed speed.
`get_download_progress(session_id)` matches the live session or the
`completed_download_sessions` map. `check_model_access` HEADs HF repos when
`config.hf_gating_enabled`.

**`settings_handler.SettingsHandler`** — `load_settings(default)` merges
`settings_file` over defaults via `migrate_legacy_settings` +
`deep_merge_dicts`. `update_settings(patch)` strips `None`, drops empty API-key
fields, deep-merges into `app_settings`, trims `text_encoder.prompt_cache` if
`prompt_cache_size` changed, returns `(before, after, changed_paths)` and
persists. `get_settings_snapshot` returns a deep copy.

**`_settings_utils.py`** — JSON type guards (`_is_json_value`/`_is_json_object`),
`ensure_json_object`, `deep_merge_dicts`, `strip_none_values`,
`collect_changed_paths` (dotted-path diff), `migrate_legacy_settings`
(`prompt_enhancer_enabled` → `_t2v`/`_i2v`).

**`health_handler.HealthHandler`** — `get_health` matches `state.gpu_slot` for
`active_model`/`models_loaded`, pulls `gpu_info.get_gpu_info()`, lists
downloaded checkpoints; `get_gpu_info` returns cuda/mps availability + VRAM.

**`hf_auth_handler.HuggingFaceAuthHandler`** — PKCE OAuth. `start_login`
generates `state`/`code_verifier`/`code_challenge` (S256), stores
`HfOAuthPending`, returns OAuth params. `handle_callback` → `_exchange_code`
validates state (`hmac.compare_digest`) + 10-min timeout, POSTs
`https://huggingface.co/oauth/token`, stores `HfAuthenticated` and persists
`hf_auth_token.json`. `load_token` rehydrates on startup (clears if expired).
`get_auth_status`/`logout` transition state. Redirect URI uses
`config.backend_port`.

**`hf_auth_utils.require_hf_token(state, lock)`** — locks, matches
`HfAuthenticated` with unexpired `expires_at`, else raises
`HTTPError(403, "HuggingFace authentication required for gated models")`.
Consumed by `DownloadHandler`.

**`runtime_policy_handler.RuntimePolicyHandler`** — stateless; returns
`RuntimePolicyResponse(force_api_generations=config.force_api_generations)`.

**`suggest_gap_prompt_handler.SuggestGapPromptHandler`** — base64-encodes
before/after/input frames, builds a Gemini 2.0 Flash prompt, POSTs via
`http_client`, parses `_GeminiResponsePayload` → `SuggestGapPromptResponse`.
Requires `gemini_api_key`.

## Integration Points

- **`app_handler.AppHandler`** (backend root) constructs every handler here with
  the shared `(AppState, RLock, RuntimeConfig)` plus sibling handlers/services;
  it is the only place handlers are wired together.
- **`state/deps.py`** exposes the `AppHandler` singleton (`get_state_service`);
  route functions resolve handlers through it.
- **`state/app_state_types.py`** supplies all discriminated-union types
  (`AppState`, `GenerationState`, `ActiveGeneration`, `HfAuthState`,
  `DownloadSessionResult`, `GpuSlot`/`CpuSlot`, `ICLoraState`,
  `VideoPipelineState`, `A2VPipelineState`, `RetakePipelineState`,
  `TextEncoderState`). `ICLoraState.lora_strength` (default `1.0`) is matched by
  `PipelinesHandler.load_ic_lora`.
- **`state/app_settings.py`** provides `AppSettings`, `UpdateSettingsRequest`
  (generated `AppSettingsPatch`), `SettingsResponse`, and
  `should_video_generate_with_ltx_api`; consumed by `video_generation_handler`,
  `retake_handler`, `image_generation_handler`, `text_handler`,
  `settings_handler`.
- **`state/conditioning_cache.py`** — `ConditioningCache`/`ConditioningCacheKey`/
  `ConditioningCacheEntry` used by `ic_lora_handler` (lives on `ICLoraState`).
- **`runtime_config/runtime_config.py`** — `RuntimeConfig` (device, dirs, ports,
  `force_api_generations`, `camera_motion_prompts`, `default_negative_prompt`,
  `dev_mode`, `local_generations_mode`, HF OAuth fields) consumed by every
  handler via `self.config`.
- **`runtime_config/runtime_policy.py`** — `streaming_prefetch_count_for_mode`
  called by `pipelines_handler` when creating local pipelines.
- **`runtime_config/model_download_specs.py`** — checkpoint/adapter specs and
  path resolvers (`get_model_cp_spec`, `get_ltx_model_spec`,
  `OFFICIAL_LTX23_ADAPTERS`, `ADAPTER_TO_CP_ID`, `resolve_*`, `is_cp_downloaded`,
  `get_existing_cp_path`, `get_downloaded_ltx_model_id`, `delete_cp_path`)
  consumed by `models_handler`, `download_handler`, `ic_lora_handler`,
  `text_handler`, `pipelines_handler`.
- **`_routes/_errors.py`** — `HTTPError` raised throughout; `code=` is
  snake-uppercase machine code (e.g. `LTX_INSUFFICIENT_FUNDS`,
  `INVALID_VIDEO_GENERATION_SPEC`, `NO_DOWNLOADED_LTX_MODEL`,
  `DOWNLOAD_ALREADY_RUNNING`).
- **`services/interfaces.py`** + impls — `LTXAPIClient`, `ZitAPIClient`,
  `ModelDownloader`, `TaskRunner`, `VideoProcessor`, `GpuInfo`, `HTTPClient`,
  `GpuCleaner`, and pipeline classes (`FastVideoPipeline`, `IcLoraPipeline`,
  `A2VPipeline`, `RetakePipeline`, `ImageGenerationPipeline`,
  `DepthProcessorPipeline`).
- **`services/ltx_components.py`** — `resolve_components`/`ResolvedLtxComponents`
  used by `pipelines_handler._resolve_active_components` to derive
  `cache_key`/paths from the active profile.
- **`api_types.py`** / **`api_model_specs.py`** / **`server_utils/media_validation.py`**
  — request/response DTOs, video request validation, and
  `validate_image_file`/`validate_audio_file`/`normalize_optional_path`.
- **`ltx_pipelines/utils/media_io.py`** and **`ltx_core/quantization.py`** —
  imported lazily by `retake_handler` and `pipelines_handler` respectively.
