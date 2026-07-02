# backend/services/text_encoder/

## Responsibility
Own text-encoding orchestration for generation. Two jobs: (1) install idempotent
monkey-patches into the upstream LTX pipeline (`ltx_pipelines.*`,
`ltx_core.text_encoders.gemma.*`) so a prompt can be encoded either by a local
Gemma/text encoder or by precomputed API embeddings fetched from the LTX API; and
(2) call the LTX API `/v1/prompt-embedding` endpoint to produce those embeddings.
This folder is the local-vs-API switching seam — when a local profile (e.g. local
GGUF Gemma) is active it short-circuits and **no API key is requested**.

## Design Patterns
- **Protocol-first service**: `text_encoder.py` defines `TextEncoder` Protocol
  (`install_patches(state_getter)`, `encode_via_api(...)`); `ltx_text_encoder.py`
  provides `LTXTextEncoder` (real) and a `DummyTextEncoder`. Re-exported via
  `services/interfaces.py` / `services/__init__.py`.
- **Stateless service + idempotent patching**: `LTXTextEncoder` holds only config
  (`device`, `http`, `ltx_api_base_url`) and two `_patched` bools. `install_patches`
  is safe to call repeatedly; each installer guards with `_prompt_encoder_patched` /
  `_cleanup_memory_patched` and wraps the whole body in try/except (logs a warning,
  never raises).
- **Lazy imports of upstream symbols**: `PromptEncoder`, `EmbeddingsProcessorOutput`,
  `ltx_pipelines.utils.helpers.cleanup_memory` are imported inside installer methods
  so importing this module never requires the heavy ML stack.
- **Discriminated runtime behavior via AppState**: the patched `PromptEncoder.__call__`
  reads `state.text_encoder.api_embeddings` on each invocation — `None` means run the
  local encoder; non-None means return the API embeddings (first prompt gets the real
  tensors, subsequent prompts get zero-filled copies) so the pipeline's multi-prompt
  contract is satisfied.
- **Test seam**: `tests/fakes/services.py` provides `FakeTextEncoder` whose
  `install_patches`/`encode_via_api` match the Protocol for handler tests.

## Data & Control Flow
**Patch install path** (`handlers/pipelines_handler.py` line ~118:
`te.service.install_patches(lambda: self.state)`):
1. `_install_prompt_encoder_init_patch` → rebinds
   `ltx_pipelines.utils.blocks.PromptEncoder.__init__`. When `gemma_root` is falsy
   (API-only mode) it short-circuits, setting `_dtype`/`_device` and nulling the
   builders so upstream never eagerly resolves Gemma file paths.
2. `_install_prompt_encoder_patch` → rebinds `PromptEncoder.__call__`. On call it
   reads `state.text_encoder`; if `api_embeddings is not None` it builds an
   `EmbeddingsProcessorOutput(video_encoding=video_context, audio_encoding=audio_context,
   attention_mask=dummy_mask)` per prompt (real for index 0, zeros for the rest);
   otherwise delegates to the original `__call__` (local encoder).
3. `_install_cleanup_memory_patch` → rebinds `ltx_pipelines.utils.helpers.cleanup_memory`
   **and** re-points the same name in `ltx_pipelines.distilled`,
   `ti2vid_one_stage`, `ti2vid_two_stages`, `ic_lora`, `a2vid_two_stage`, `retake`,
   `retake_pipeline`. The wrapper moves `te_state.cached_encoder` to CPU before
   calling the original cleanup.

**API encode path** (`handlers/text_handler.py` →
`te.service.encode_via_api(prompt, api_key, checkpoint_path, enhance_prompt)`):
1. `get_model_id_from_checkpoint` opens the safetensors file and reads metadata key
   `encrypted_wandb_properties` (the model id). Returns `None` → caller skips API
   encoding.
2. `encode_via_api` POSTs `{ltx_api_base_url}/v1/prompt-embedding` with
   `Authorization: Bearer {api_key}` and JSON `{prompt, model_id, enhance_prompt}`,
   `timeout=60`.
3. On 200 it `pickle.loads(response.content)` (trusted source), takes
   `conditioning[0][0]`, and splits at feature dim 4096: `[..., :4096]` →
   `video_context`, `[..., 4096:]` → `audio_context` (or `None` if absent). Both are
   cast to `bfloat16` and moved to `self.device`. Returns a `TextEncodingResult`.

**Local-suppression path** (relevant to "local profile must not prompt for API key"):
`TextHandler.should_use_local_encoding()` returns `True` as soon as
`_active_profile_provides_local_encoder()` is true — evaluated **before** any
`ltx_api_key` check. `_prepare_api_embeddings` then calls `clear_api_embeddings()`
and returns `None`, so `encode_via_api` is never invoked and no API key is needed.

## Integration Points
- **`state/app_state_types.py`**: `TextEncoderState` (fields `service`,
  `api_embeddings: TextEncodingResult | None`, `cached_encoder`, `prompt_cache`)
  read by every patch; `TextEncodingResult` (`video_context`, `audio_context`) is
  the return type of `encode_via_api`.
- **`handlers/text_handler.py`**: decides local-vs-API (`should_use_local_encoding`,
  `resolve_gemma_root`, `prepare_text_encoding`), caches embeddings in
  `te.prompt_cache`, and is the only caller of `encode_via_api` /
  `clear_api_embeddings`.
- **`handlers/pipelines_handler.py`**: calls `install_patches(lambda: self.state)`
  once at pipeline setup.
- **`handlers/retake_handler.py`, `ic_lora_handler.py`,
  `video_generation_handler.py`**: call `clear_api_embeddings()` before/after runs.
- **`services/http_client/http_client.py`**: `HTTPClient` injected into
  `LTXTextEncoder` for the `/v1/prompt-embedding` POST.
- **`services/patches/gguf_loader_fix.py`** (sibling patch layer, not this folder):
  rebinds `GemmaTextEncoder._enhance` and `GemmaTextEncoder.encode` to use
  llama.cpp for local GGUF Gemma — the local-encoder counterpart that makes the
  API-key-suppression path viable.
- **`services/patches/safetensors_metadata_fix.py`**: re-patches
  `LTXTextEncoder.get_model_id_from_checkpoint` to also accept multi-checkpoint
  inputs.
- **`app_handler.py`**: constructs `LTXTextEncoder(device, http, ltx_api_base_url)`
  and injects it into `AppHandler`.
- **Tests**: `tests/fakes/services.py` `FakeTextEncoder`.
