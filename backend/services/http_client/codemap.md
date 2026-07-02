# backend/services/http_client/

## Responsibility

External HTTP transport for the backend. Defines the `HTTPClient` `Protocol` (the dependency-inversion surface every API-client service programs against) and the real `requests`-backed implementation. Owns timeout→exception translation so callers can catch a single `HttpTimeoutError` regardless of transport.

## Design Patterns

- **Protocol + Impl split.** `http_client.py` is import-safe (no `requests` import) and holds only `Protocol`s (`HTTPClient`, `HttpResponseLike`) and `HttpTimeoutError`. `http_client_impl.py` carries the `requests` dependency and the concrete `HTTPClientImpl`.
- **Structural response typing.** `HttpResponseLike` (Protocol: `status_code`, `text`, `headers`, `content`, `json()`) lets fakes (`tests/fakes/services.py::FakeResponse`) and the real `requests.Response` satisfy the same interface without inheritance.
- **Uniform timeout→`HttpTimeoutError`.** Every method (`post`/`get`/`put`) wraps its `requests.*` call in `try/except requests.exceptions.Timeout` and re-raises as `HttpTimeoutError(str(exc)) from exc`, logging the URL first.

## Data & Control Flow

`HTTPClientImpl.post(url, headers, json_payload, data, timeout=30)` → `requests.post(url, headers=headers, json=json_payload, data=data, timeout=timeout)` → `requests.Response` (or `HttpTimeoutError`).

`HTTPClientImpl.get(url, headers, timeout=30)` → `requests.get(...)`.

`HTTPClientImpl.put(url, data, headers, timeout=300)` → `requests.put(...)` (longer default timeout for uploads).

`json_payload: Mapping[str, JSONValue] | None` and `data: RequestData` (`bytes | str | Mapping | BinaryIO | None`, from `services/services_utils.py`) map directly onto `requests` kwargs. No retry, no auth injection here — callers add `Authorization` headers explicitly.

## Integration Points

- **`app_handler.build_default_service_bundle()`** instantiates a single `HTTPClientImpl()` and injects it into `LTXAPIClientImpl(http=http, ...)`, `ZitAPIClientImpl(http=http)`, and `LTXTextEncoder(http=http, ...)` — one shared transport per app.
- **`services/interfaces.py`** re-exports `HTTPClient`, `HttpResponseLike`, `HttpTimeoutError` as the public boundary.
- **API-client services** (`services/ltx_api_client/`, `services/zit_api_client/`) and `services/text_encoder/ltx_text_encoder.py` consume the Protocol; they never import `requests` directly.
- **Tests:** `tests/fakes/services.py::FakeHTTPClient` (+ `FakeResponse`) implement the Protocol; `tests/test_ltx_api_client.py` drives the LTX API client entirely through `FakeHTTPClient` — no real network.
