# backend/_routes/

## Responsibility

Thin HTTP plumbing layer. Each module declares one `APIRouter`, exposes
endpoints that parse/validate the request body, obtain the `AppHandler` via
`Depends(get_state_service)`, and delegate to a single handler method.
**These modules contain no business logic** — no state mutation, no GPU/network
calls, no policy decisions (the only exceptions are the cross-cutting guards in
`_admin_guard.py`, the error type in `_errors.py`, and the localhost check in
`health.route_shutdown`). `app_factory.create_app()` imports every `router`
here and calls `app.include_router(...)` on it.

Shared helpers (private modules, not routers):
- `_errors.py` — `class HTTPError(Exception)` (carries `status_code`,
  `response: HTTPErrorResponse`, `detail`, `code`) and
  `build_http_error_response(status_code, detail, *, code=None)` which derives a
  machine `code` (`HTTP_{status}` unless `detail` already matches
  `^[A-Z0-9]+(_[A-Z0-9]+)*$`). Raised by handlers/routes; converted to JSON by
  the `HTTPError` exception handler in `app_factory.py`.
- `_admin_guard.py` — `guard_admin_permission(request)`: reads
  `request.app.state.admin_token` (set by `create_app`), compares the
  `X-Admin-Token` header with `hmac.compare_digest`, raises
  `HTTPError(403, "Admin token required")` on mismatch/missing.
- `__init__.py` — empty package marker.

## Design Patterns

- **One router per domain file**, constructed as `APIRouter(prefix=...,
  tags=[...])` and imported as `router` (e.g. `from _routes.generation import
  router as generation_router`).
- **Uniform handler access** — every endpoint signature is
  `def route_x(req, handler: AppHandler = Depends(get_state_service))`.
  Pydantic request models from `api_types.py` (or `state.app_settings`) provide
  validation; failures surface as 422 via the `RequestValidationError` handler.
- **Guard-first ordering** — admin-gated endpoints call
  `guard_admin_permission(request)` as the first statement so authorization
  is enforced before any handler work.
- **Typed responses** — every decorator sets `response_model=` (or
  `response_class=HTMLResponse` for the HF callback), making the OpenAPI
  schema and frontend types deterministic.

## Data & Control Flow

Per-route delegation table (method, full path, AppHandler target, notes):

### `generation.py` — `prefix="/api"`, tag `generation`
- `POST /api/generate` → `handler.video_generation.generate(req)`.
  Response `GenerateVideoResponse`; documents extra `402` →
  `LtxInsufficientFundsErrorResponse`.
- `GET /api/generate/models-specs` → `handler.video_generation.get_model_specs()`.
- `POST /api/generate/cancel` → `handler.generation.cancel_generation()`.
- `GET /api/generation/progress` → `handler.generation.get_generation_progress()`.

### `health.py` — no prefix, tag `health`
- `GET /health` → `handler.health.get_health()`.
- `GET /api/gpu-info` → `handler.health.get_gpu_info()`.
- `POST /api/system/shutdown` → **not handler-delegated.** Verifies
  `request.client.host` is in `{127.0.0.1, ::1, localhost}` else raises
  `HTTPException(403, "Forbidden")`; schedules `_shutdown_process()`
  (`os.kill(os.getpid(), signal.SIGTERM)`) via `BackgroundTasks`; returns
  `{"status": "shutting_down"}`.

### `hf_auth.py` — `prefix="/api/auth/huggingface"`, tag `hf_auth`
- `POST /login` → `handler.hf_auth.start_login()`.
- `GET /callback` (`response_class=HTMLResponse`, query `code`/`state`/`error`)
  → `HTMLResponse(handler.hf_auth.handle_callback(code, state, error))`.
  (Exempted from auth middleware in `app_factory`.)
- `GET /status` → `handler.hf_auth.get_auth_status()`.
- `POST /logout` → `handler.hf_auth.logout()`.

### `ic_lora.py` — `prefix="/api/ic-lora"`, tag `ic-lora`
- `POST /extract-conditioning` → `handler.ic_lora.extract_conditioning(req)`.
- `POST /generate` → `handler.ic_lora.generate(req)`.

### `image_gen.py` — `prefix="/api"`, tag `image`
- `POST /api/generate-image` → `handler.image_generation.generate(req)`.

### `model_profiles.py` — `prefix="/api"`, tag `model-profiles`
  (every route calls `guard_admin_permission(request)` first)
- `GET /api/model-profiles` → `handler.model_profiles.list_profiles()`.
- `POST /api/model-profiles` → `handler.model_profiles.create_profile(req)`.
- `PATCH /api/model-profiles/{profile_id}` →
  `handler.model_profiles.patch_profile(profile_id, req)`.
- `DELETE /api/model-profiles/{profile_id}` →
  `handler.model_profiles.delete_profile(profile_id)`; returns
  `StatusResponse(status="ok")`.
- `POST /api/model-profiles/{profile_id}/validate` →
  `handler.model_profiles.validate_profile_by_id(profile_id)`.
- `POST /api/model-profiles/{profile_id}/activate` →
  `handler.model_profiles.activate_profile(profile_id)`.

### `models.py` — `prefix="/api"`, tag `models`
- `GET /api/models/ltx-recommendation` →
  `handler.models.get_ltx_recommendation()`.
- `GET /api/models/img-gen-recommendation` →
  `handler.models.get_img_gen_recommendation()`.
- `GET /api/models/ltx-ic-lora-recommendation` →
  `handler.models.get_ltx_ic_lora_recommendation()`.
- `GET /api/models/adapters/status` (admin-guarded, optional query `pipeline:
  AdapterPipeline | None`) → `handler.models.get_adapter_status(pipeline)`.
- `GET /api/models/adapters/recommendation` (admin-guarded, required query
  `pipeline: AdapterPipeline`) →
  `handler.models.get_adapter_recommendation(pipeline)`.
- `GET /api/models/text-encoder-recommendation` →
  `handler.models.get_text_encoder_recommendation()`.
- `GET /api/models/download/progress` (query `sessionId`) →
  `handler.downloads.get_download_progress(sessionId)`; maps `ValueError` to
  `HTTPError(404, "UNKNOWN_DOWNLOAD_SESSION")`.
- `POST /api/models/check-access` →
  `handler.downloads.check_model_access(req.cp_ids)`.
- `POST /api/models/download` →
  `handler.downloads.start_model_download(download_type=req.type,
  cp_ids=req.cp_ids)`, wrapped in `ModelDownloadStartResponse(status="started",
  sessionId=...)`.
- `DELETE /api/models/delete` → raises `HTTPError(409,
  "DOWNLOAD_ALREADY_RUNNING")` if `handler.downloads.is_download_running()`,
  else `handler.models.delete_checkpoints(req.cp_ids)` → `StatusResponse("ok")`.

### `retake.py` — `prefix="/api"`, tag `retake`
- `POST /api/retake` → `handler.retake.run(req)`.

### `runtime_policy.py` — `prefix="/api"`, tag `runtime-policy`
- `GET /api/runtime-policy` → `handler.runtime_policy.get_runtime_policy()`.

### `settings.py` — `prefix="/api"`, tag `settings`
- `GET /api/settings` → calls
  `to_settings_response(handler.settings.get_settings_snapshot())` then
  overwrites `response.models_dir = str(handler.settings.models_dir)`.
- `POST /api/settings` → `guard_admin_permission(request)` only when
  `models_dir`/`modelsDir` is present in the partial patch
  (`req.model_dump(exclude_unset=True)`); then
  `handler.settings.update_settings(req)`, logs changed root keys, returns
  `StatusResponse(status="ok")`.

### `suggest_gap_prompt.py` — `prefix="/api"`, tag `prompt`
- `POST /api/suggest-gap-prompt` → `handler.suggest_gap_prompt.suggest_gap(req)`.

## Integration Points

- **`app_factory.py`** — imports each `router` and registers it; owns the
  `HTTPError`/`RequestValidationError`/`StarletteHTTPException`/`Exception`
  exception handlers that turn `_errors.HTTPError` into `HTTPErrorResponse`
  JSON, plus the auth middleware that gates all these paths (except the HF
  callback).
- **`app_handler.AppHandler`** — the single delegation target; `get_state_service`
  (from `state/deps.py`, seeded by `init_state_service`) is how routes obtain it.
- **`api_types.py`** and **`state/app_settings.py`** — request/response models
  (`GenerateVideoRequest`, `IcLoraGenerateRequest`, `RetakeRequest`,
  `SuggestGapPromptRequest`, `ModelProfilePayload`, `ModelDownloadRequest`,
  `UpdateSettingsRequest`, etc.) and `response_model` targets.
- **`server_utils/media_validation.py`** is *not* referenced here; handlers
  invoke it.
