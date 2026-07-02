# backend/

## Responsibility

Root-level Python modules that bootstrap, configure, and compose the LTX FastAPI
server, plus the request/response contract shared with the frontend.

- `ltx2_server.py` — process entrypoint. Enables `faulthandler`, optionally
  starts `debugpy` (env `BACKEND_DEBUG=1`), imports ltx-core/safetensors monkey
  patches from `services/patches/`, patches `F.scaled_dot_product_attention`
  with SageAttention when `USE_SAGE_ATTENTION=1`, resolves the device/dtype and
  `LTX_APP_DATA_DIR`, builds a `RuntimeConfig`, calls `build_initial_state(...)`,
  then constructs the app via `create_app(...)` and runs it under `uvicorn`.
- `app_factory.py` — `create_app()` constructs a `FastAPI` instance decoupled
  from runtime side effects: installs CORS, the bearer/Basic/WebSocket auth
  middleware, the four exception handlers, and `include_router(...)` for every
  router in `_routes/`. When `LTX_MEMORY_TRACE_PATH` is set, also installs an
  HTTP trace middleware (`http_start` on entry, `http_end`/`http_error` on exit;
  terminal events deduped via a request-state flag that survives
  `BaseHTTPMiddleware` task isolation) and each exception handler records one
  terminal `http_error` via `services.memory_trace`. No-op when trace env unset.
- `app_handler.py` — `AppHandler`, the single composition root. Owns the shared
  `threading.RLock`, the `AppState` dataclass, and all domain sub-handlers.
  Also defines `ServiceBundle`, `build_default_service_bundle(config)`, and
  `build_initial_state(config, default_settings, service_bundle=None)`.
- `api_types.py` — Pydantic v2 request/response DTOs and `Literal` type aliases
  that form the HTTP contract.
- `api_model_specs.py` — canonical resolution/fps/duration capability matrices
  for local + API video pipelines, and `validate_generate_video_request(...)`.
- `logging_policy.py` — boundary-owned logging helpers: `log_http_error`,
  `log_unhandled_exception`, `log_background_exception`.
- `export_openapi_schema.py` — boots the app with `FakeServices` (no GPU) and
  writes `frontend/generated/backend-openapi.json` for frontend codegen.
- `generate_api_docs.py` — static HTML API reference generator that regex-scans
  `_routes/*.py` and `api_types.py`.
- `scripts/live_workflow_test.py` — Phase 1 live harness (out-of-band,
  stdlib-only). **Spawn** (`--backend auto`, owning `report.json`/`summary.md`/
  `backend.log`/trace truncation exactly once, per-case restarts append) or
  **attach** (`--backend http://…`, requires explicit `--trace-path` + tokens,
  never truncates). Per case: preflights `GET /api/settings` + admin
  `GET /api/model-profiles` + selected-model/HDR-adapter install, then runs one
  atomic (fast/Kijai/GGUF/IC-LoRA ingredients/HDR EXR/retake) and validates
  output (mp4 via cv2, EXR via OpenImageIO/OpenEXR + proxy mp4) **and** memory-trace
  ownership (`http_start` + terminal event for this run/case/pid; HDR also
  requires a pipeline-create phase event). `harness:selftest` is a no-generation
  `GET /health` case. Writes `report.json` + `summary.md`; exit 0 only if every
  case passed / expected-failed / allowed-gate. Every request carries
  `X-LTX-Memory-Run-Id`/`X-LTX-Memory-Case-Id` correlation headers.

## Design Patterns

- **Thin routes → composition root.** Request flow is fixed:
  `_routes/* (parse/validate) → AppHandler → handlers/* → services/* + state/*`.
  Routes contain no business logic; they delegate to one handler method.
- **Handler injection via app state, not FastAPI deps.** `create_app()` calls
  `init_state_service(handler)` (from `state/deps.py`); every route reads it
  back through `handler: AppHandler = Depends(get_state_service)`.
  `state.build_initial_state` is a re-export of `app_handler.build_initial_state`.
- **Composition over inheritance.** `AppHandler.__init__` instantiates each
  sub-handler (`ModelsHandler`, `PipelinesHandler`, `VideoGenerationHandler`,
  `IcLoraHandler`, `RetakeHandler`, `HealthHandler`, `RuntimePolicyHandler`,
  `SuggestGapPromptHandler`, etc.), passing the same `self._lock`, `self.state`,
  and `config`. Heavier handlers receive collaborators (e.g.
  `VideoGenerationHandler` gets `generation_handler`, `pipelines_handler`,
  `text_handler`, `ltx_api_client`).
- **Service swapping for tests.** `ServiceBundle` (a `@dataclass`) holds all
  Protocol-typed services and pipeline *classes*. `build_default_service_bundle`
  performs the lazy real imports; tests inject a `FakeServices`-backed bundle
  into `build_initial_state(..., service_bundle=bundle)`. No `unittest.mock`.
- **Concurrency invariant.** One `threading.RLock` is shared by every handler.
  Pattern is lock→read/validate→unlock→heavy work→lock→write; the lock is never
  held during GPU/network work. `TaskRunner` (`ThreadingRunner`) fans the heavy
  work into the background thread pool.
- **Discriminated-union state machines.** `AppState.active_generation` holds a
  `GenerationState` union (`GenerationRunning | GenerationComplete |
  GenerationError | GenerationCancelled`); response DTOs mirror this with
  `status` literal discriminants (e.g. `GenerateVideoResponse`,
  `DownloadProgressResponse`, `CancelResponse`, `RetakeResponse`,
  `IcLoraGenerateResponse`).
- **Boundary-owned traceback policy.** Handlers raise `HTTPError` (from
  `_routes/_errors.py`) with `from exc` chaining. `app_factory.py` registers
  exception handlers that delegate to `logging_policy.py`; 5xx `HTTPError` and
  unhandled `Exception` log with `exc_info`, 4xx `HTTPError` logs at WARNING
  without traceback. Handlers must not `logger.exception(...)` then re-raise.
- **Side-effect-free bootstrap.** `ltx2_server.py` runs module-level work
  (patches, device probe, `migrate_legacy_models_layout(APP_DATA_DIR)`,
  `_resolve_local_generations_mode()`), then hands off to the side-effect-free
  `create_app()` so tests and `export_openapi_schema.py` can reuse it.

## Data & Control Flow

1. **Startup.** `ltx2_server.py` imports `services/patches/*` → resolves
   `DEVICE`/`DTYPE`/`APP_DATA_DIR`/`DEFAULT_MODELS_DIR`/`SETTINGS_FILE` →
   builds `DEFAULT_APP_SETTINGS = AppSettings()` → constructs `runtime_config`
   (`RuntimeConfig(...)` carrying `local_generations_mode`, `backend_port`,
   `use_sage_attention`, `camera_motion_prompts`, `hf_oauth_client_id`, …) →
   `handler = build_initial_state(runtime_config, DEFAULT_APP_SETTINGS)`.
   `build_initial_state` calls `build_default_service_bundle(config)` then
   `AppHandler(...)`, whose constructor ends with
   `downloads.cleanup_downloading_dir()` and `load_persistent_state(...)`
   (`settings.load_settings`, `hf_auth.load_token`, `model_profiles.load_profiles`).
2. **App assembly.** `create_app(handler, allowed_origins, auth_token,
   admin_token)` → `init_state_service(handler)` → builds `FastAPI` with
   `DEFAULT_ERROR_RESPONSES` (4XX/5XX → `HTTPErrorResponse`) → adds CORS,
   `_auth_middleware` (skips when `auth_token` empty, for OPTIONS, and for
   `/api/auth/huggingface/callback`; accepts Bearer, Basic, or `?token=` for
   WebSocket), and the four exception handlers → `include_router` for all 11
   routers.
3. **Serving.** `__main__` builds a `uvicorn.Config(app, host="127.0.0.1",
   port=runtime_config.backend_port, access_log=False, log_config=...)`,
   monkey-wraps `server.startup` to print the machine-parseable
   `"Server running on http://127.0.0.1:{port}"` line that Electron awaits,
   then `asyncio.run(server.serve())`.
4. **Request path.** `auth middleware → router (Depends(get_state_service)) →
   handler.<domain>.<method>(req) → sub-handler → services + state (under the
   shared RLock) → typed response model`. Validation failures become
   `RequestValidationError` → 422; business failures raise `HTTPError`.
5. **Spec validation.** `api_model_specs.validate_generate_video_request(req,
   use_api_specs=...)` selects local vs API capability specs and returns a
   human-readable error string (or `None`) for unsupported pipeline/resolution/
   fps/duration combinations.

## Integration Points

- **`_routes/`** — all routers registered by `app_factory.create_app`;
  `_routes/_errors.HTTPError` is the exception type produced by handlers and
  consumed by the `HTTPError` exception handler + `logging_policy`.
- **`handlers/`** — the domain logic classes instantiated by `AppHandler`
  (`GenerationHandler`, `VideoGenerationHandler`, `PipelinesHandler`,
  `IcLoraHandler`, `RetakeHandler`, `HealthHandler`, `RuntimePolicyHandler`,
  `SuggestGapPromptHandler`, `ModelsHandler`, `ModelProfilesHandler`,
  `SettingsHandler`, `TextHandler`, `ImageGenerationHandler`,
  `DownloadHandler`, `HuggingFaceAuthHandler`).
- **`services/`** — Protocol interfaces (`services/interfaces.py`) and real
  implementations wired in `build_default_service_bundle`; `services/patches/`
  applied at top of `ltx2_server.py`.
- **`state/`** — `AppState`/`TextEncoderState` (discriminated-union state
  machines), `RuntimeConfig`, `init_state_service`/`get_state_service`
  (`state/deps.py`), and `build_initial_state` (re-exported here).
- **`runtime_config/`** — `PORT` constant, `runtime_policy.decide_local_
  generation_mode` (called in `_resolve_local_generations_mode`),
  `model_download_specs.get_latest_ltx_model_id`/`get_ltx_model_spec` consumed
  by `api_model_specs`.
- **`server_utils/`** — `model_layout_migration.migrate_legacy_models_layout`
  run at startup; `media_validation` imported by handlers.
- **`tests/fakes/`** — `FakeServices` / `FAKE_CAMERA_MOTION_PROMPTS` drive
  `export_openapi_schema.py` and the integration tests.
- **Frontend** — `frontend/generated/backend-openapi.json` (from
  `export_openapi_schema.py`) feeds client type generation; the
  `"Server running on http://127.0.0.1:{port}"` line is Electron's readiness
  signal.
