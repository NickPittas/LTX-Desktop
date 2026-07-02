# backend/services/ltx_api_client/

## Responsibility
LTX Studio REST API client for cloud video generation. Wraps the four generation
endpoints (`text-to-video`, `image-to-video`, `audio-to-video`, `retake`) plus the
two-step media `upload` flow, normalizing their varied response shapes (inline
binary, JSON with direct URL, JSON with nested `result.video_url`) into raw
`bytes` (generation) or an `LTXRetakeResult`. Owns the structured error type
(`LTXAPIClientError`) that handlers translate into HTTP errors.

## Design Patterns
- **Protocol-first service**: `ltx_api_client.py` defines the `LTXAPIClient`
  Protocol, the frozen `LTXRetakeResult` dataclass (`video_bytes`, `result_payload`)
  and `LTXAPIClientError(RuntimeError)` carrying `status_code`, `detail`, `stage`,
  `provider_error_type`, `provider_message`, `request_id`. `ltx_api_client_impl.py`
  provides `LTXAPIClientImpl`. Re-exported via `services/interfaces.py` /
  `services/__init__.py`.
- **Stages on errors**: `LTXAPIClientError.stage` tags where a failure happened
  (`upload_init`, `upload_parse`, `upload_put`, `generation`) so `retake()` can
  re-map upload-stage errors into human-readable messages (e.g.
  `"Failed to get upload URL: ..."`).
- **Pydantic response parsing**: private models `_RetakeNestedPayload`,
  `_RetakeResponsePayload` (`extract_video_url()`), `_LTXErrorDetailPayload`,
  `_LTXErrorPayload` (`extra="ignore"`/`"allow"`) validate unpredictable provider
  JSON; parse failures degrade gracefully to raw `response.text[:500]`.
- **Value-mapping adapters**: `_CAMERA_MOTION_TO_LTX` maps the app
  `VideoCameraMotion` enum to the LTX `LTXCameraMotion` Literal (`"none"` →
  `None`, which omits the field entirely).
- **Injected HTTPClient**: all network goes through `services/http_client.http_client.HTTPClient`
  (`post`/`get`/`put`); the client holds only `_http` and `_base_url`.
- **Test seam**: `tests/fakes/services.py` `FakeLTXAPIClient` records
  `*_calls`/returns/raise-on flags; `tests/test_ltx_api_client.py` exercises
  `LTXAPIClientImpl` against a fake `HTTPClient`.

## Data & Control Flow
**Generation** (`generate_text_to_video` / `_image_to_video` / `_audio_to_video`):
1. Build JSON payload (`prompt`, `model`, `resolution`, `duration`, `fps`,
   `generate_audio`, plus `image_uri`/`audio_uri` as relevant); append
   `camera_motion` only when `_map_camera_motion(...)` is non-`None`.
2. POST to `/v1/{text|image|audio}-to-video` with `Authorization: Bearer {api_key}`,
   `timeout=1200`.
3. `_extract_video_bytes(response, api_key)`:
   - non-200 → `_extract_generation_error` (parses `_LTXErrorPayload` for
     `provider_error_type`/`provider_message`), raises `LTXAPIClientError(stage="generation")`;
   - Content-Type `video/*` or `octet-stream` → return `response.content`;
   - JSON → `_extract_video_url` (checks `video_url`, `output_video`,
     `output_video_url`, `output_url`, `url`, and the same keys nested under
     `result`); if found, GET-download with `Bearer`, return bytes; otherwise raise.

**Retake** (`retake(api_key, video_path, start_time, duration, prompt, mode)`):
1. `storage_uri = upload_file(api_key=..., file_path=video_path)`; stage-tagged
   `LTXAPIClientError`s are caught and re-mapped to friendly messages.
2. POST `/v1/retake` with `{video_uri, start_time, duration, mode}` (+ optional
   `prompt`), `timeout=600`.
3. On 200: inline video body → `LTXRetakeResult(video_bytes=..., result_payload=None)`;
   JSON with `extract_video_url()` → GET-download; otherwise return
   `LTXRetakeResult(video_bytes=None, result_payload=<dump>)`. 422 →
   `"Content rejected by safety filters"`. Other → `LTXAPIClientError`.

**Upload** (`upload_file(api_key, file_path)`):
1. POST `/v1/upload` → JSON `{upload_url, storage_uri, required_headers}`.
2. Resolve MIME via `mimetypes`, `open(path, "rb")`, PUT the file to `upload_url`
   with `Content-Type` + `required_headers`, `timeout=300`. Returns `storage_uri`.

Every error path appends the provider's `x-request-id` via `_fmt_request_id` /
`_request_id` for traceability.

## Integration Points
- **`api_types`**: `RetakeMode`, `VideoCameraMotion` shared enums.
- **`services/http_client/http_client.py`**: `HTTPClient` injected at construction.
- **`services/services_utils.py`**: `JSONValue` for payload typing.
- **`app_handler.py`**: constructs `LTXAPIClientImpl(http=http,
  ltx_api_base_url=config.ltx_api_base_url)` and injects it into the handler bundle.
- **`handlers/video_generation_handler.py`**: calls `upload_file`,
  `generate_audio_to_video`, `generate_image_to_video`, `generate_text_to_video`;
  `_map_ltx_api_generation_error(exc: LTXAPIClientError)` converts the structured
  client error into an `HTTPError`.
- **`handlers/retake_handler.py`**: calls `.retake(...)` and catches
  `LTXAPIClientError` to surface upload/generation/safety (422) failures.
- **`text_encoder/ltx_text_encoder.py`**: independent caller of the same base URL's
  `/v1/prompt-embedding` (not via this client).
- **Tests**: `tests/test_ltx_api_client.py`, `tests/test_generation.py`,
  `tests/test_api_calls.py`, `tests/fakes/services.py` `FakeLTXAPIClient`.
