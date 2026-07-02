# backend/services/zit_api_client/

## Responsibility
Z-Image Turbo text-to-image client. Submits a prompt to the FAL-hosted
`fal-ai/z-image/turbo` endpoint, extracts the returned image URL, downloads the
PNG, and returns the raw image bytes. Sole image-generation backend used by the
image generation flow (e.g. generating start frames for image-to-video).

## Design Patterns
- **Protocol-first service**: `zit_api_client.py` defines the `ZitAPIClient`
  Protocol (`generate_text_to_image(*, api_key, prompt, width, height, seed,
  num_inference_steps) -> bytes`); `zit_api_client_impl.py` provides
  `ZitAPIClientImpl`. Both are exported from `__init__.py` and re-exported via
  `services/interfaces.py` / `services/__init__.py`.
- **Module-level constants for FAL contract**: `FAL_API_BASE_URL = "https://fal.run"`,
  `FAL_TEXT_TO_IMAGE_ENDPOINT = "/fal-ai/z-image/turbo"`, plus defaults
  `DEFAULT_OUTPUT_FORMAT = "png"`, `DEFAULT_ACCELERATION = "regular"`,
  `DEFAULT_ENABLE_SAFETY_CHECKER = True`.
- **Injected HTTPClient, base-url overridable**: constructed as
  `ZitAPIClientImpl(http, *, fal_api_base_url=FAL_API_BASE_URL)`; `_base_url` is
  `rstrip("/")`-ed and joined with the endpoint path.
- **Submit-then-download helper**: `_submit_and_download(endpoint, api_key, payload)`
  centralizes the two-step FAL flow (POST JSON → parse → GET image) reused by the
  single public method.
- **Header convention difference from LTX**: FAL uses `Authorization: Key {api_key}`
  (via `_json_headers`), **not** `Bearer`.
- **Defensive response parsing**: `_json_object` ensures a dict; `_extract_image_url`
  tries `images[0].url` (or `images[0]` if it is a string) then falls back to
  `image_url` / `imageUrl` / `url`; raises `RuntimeError("FAL response missing
  image url")` if none. Non-200 responses raise `RuntimeError` with truncated body.
- **Test seam**: `tests/fakes/services.py` `FakeZitAPIClient` records
  `text_to_image_calls` and supports `raise_on_text_to_image`.

## Data & Control Flow
1. Caller invokes `generate_text_to_image(api_key=..., prompt=..., width=...,
   height=..., seed=..., num_inference_steps=...)`.
2. The method builds the payload: `{prompt, image_size:{width,height},
   num_inference_steps, seed, num_images:1, output_format="png",
   acceleration="regular", enable_safety_checker=True}`.
3. `_submit_and_download` POSTs to `{fal_api_base_url}/fal-ai/z-image/turbo` with
   `Authorization: Key {api_key}`, `Content-Type: application/json`,
   `timeout=180`.
4. Non-200 → `RuntimeError("FAL submit failed ({status}): {detail}")`.
5. Response body is parsed via `_json_object` → `_extract_image_url` resolves the
   download URL.
6. `HTTPClient.get(image_url, timeout=120)` downloads the PNG; non-200 or empty body
   → `RuntimeError`. Returns the raw `download.content` (`bytes`).

## Integration Points
- **`services/http_client/http_client.py`**: `HTTPClient` injected at construction
  for both `post` and `get`.
- **`services/services_utils.py`**: `JSONValue` used for the request payload typing.
- **`app_handler.py`**: constructs `ZitAPIClientImpl(http=http)` in the composition
  root and injects it into the handler bundle.
- **`handlers/image_generation_handler.py`**: sole production caller —
  `self._zit_api_client.generate_text_to_image(...)` (~line 171) to produce
  generated image bytes (e.g. for downstream image-to-video / IC-LoRA input).
- **Tests**: `tests/test_generation.py` (image-generation scenarios),
  `tests/fakes/services.py` `FakeZitAPIClient`.
