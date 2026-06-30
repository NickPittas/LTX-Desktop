# backend/services/

## Responsibility

Root of the service layer. Holds the cross-cutting primitives every pipeline/service subpackage shares, plus the public service-interface re-export surface consumed by `app_handler.py` and tests:

- `interfaces.py` — single import location for all service `Protocol` types and DTOs (the dependency-inversion boundary).
- `__init__.py` — re-exports `interfaces.py` symbols so callers can `from services import HTTPClient, GpuCleaner, ...`.
- `services_utils.py` — shared type aliases, device helpers, and structural `Protocol`s used across services.
- `ltx_components.py` — pure resolver turning a `ModelProfilePayload` into a typed `ResolvedLtxComponents` bundle (no heavy imports; GPU-free/testable).
- `base_video_model_registry.py` — **unified source of truth for generation-selectable base video transformer variants** (`BaseVideoModelRegistryEntry`, `iter_base_video_model_entries`, `resolve_base_video_model_selection`). Drives `GET /api/models/model-options`, the scanner's Kijai/QuantStack artifact recognition, the request resolver, and the family-mismatch guard. `ModelSelectionID` is a runtime `str`; unknown ids raise `KeyError` (handlers translate to `UNSUPPORTED_MODEL_SELECTION`).
- `ltx_pipeline_common.py` — shared LTX pipeline helpers, including `encode_video_output()` (the **central video-encode wrapper**) and the `DistilledNativePipeline` fast-path.

## Design Patterns

- **Interface-first / Protocol-based DI.** Each subpackage defines a `Protocol` (in `<subpkg>/<subpkg>.py`) and a concrete `*Impl`. `interfaces.py` aggregates the Protocols; `app_handler.ServiceBundle` wires real impls (`build_default_service_bundle`), tests wire fakes (`tests/fakes/services.py`). `HTTPClient`, `GpuCleaner`, `GpuInfo`, `ModelDownloader`, etc. are all structural Protocols — duck-typed, no inheritance.
- **Lazy heavy imports.** `ltx_pipeline_common.py` imports `ltx_core.*` / `ltx_pipelines.*` inside function bodies (`encode_video_output`, `default_tiling_config`, `default_guiders`, `video_chunks_number`) and inside `DistilledNativePipeline.__init__`/`__call__`. Keeps `services` importable without torch/GPU.
- **TYPE_CHECKING-only type aliases.** `services_utils.py` exposes `TilingConfigType`, `FrameArray`, `AudioType` as `object` at runtime but real types under `TYPE_CHECKING` — avoids importing `ltx_core`/`numpy` at module import.
- **Pure resolver for profile data.** `ltx_components.resolve_components()` is a side-effect-free mapping from `ModelProfilePayload` → `ResolvedLtxComponents`; `cache_key` tuple enables downstream pipeline caching. When a live selection is present, the caller passes explicit `selected_transformer_format` and `selected_base_family` (from the registry entry) so no filename/path-only inference happens for the selected family/format.
- **Unified base-video registry (source of truth).** `base_video_model_registry.iter_base_video_model_entries(models_dir)` enumerates every selectable base video variant with read-only filesystem evidence (installed/missing/wrong-folder/duplicate). `models_handler.get_model_selection_options()` consumes it; `model_scanner` derives Fast-family Kijai/QuantStack artifacts from its static table; `pipelines_handler._resolve_selection()` resolves a selection id to an entry; `video_generation_handler` reads `pipeline_family` for the family-mismatch guard.

## Data & Control Flow

### Encode path — `encode_video_output()` (EXR/MOV plan focal point)

`ltx_pipeline_common.py:35`:

```python
def encode_video_output(video, audio, fps, output_path, video_chunks_number_value) -> None:
    from ltx_pipelines.utils.media_io import encode_video
    encode_video(video=video, fps=fps, audio=audio, output_path=output_path,
                 video_chunks_number=video_chunks_number_value)
```

- **Inputs:** `video: torch.Tensor | Iterator[torch.Tensor]` (decoded RGB frames straight from the VAE `VideoDecoder` — the PRIMARY generation output, *not* a transcoded MP4); `audio: AudioOrNone`; `fps: int` (cast from request `frame_rate`); `output_path: str` (final `.mp4` path); `video_chunks_number_value: int` (computed via `video_chunks_number()` → `ltx_core.model.video_vae.get_video_chunks_number(num_frames, tiling_config)` — controls VAE chunked decoding, unrelated to encode container).
- **Dims/pixel-format handling:** the wrapper passes *no* dimensions or codec args. `encode_video()` derives H/W from the tensor shape and **hardcodes H.264 / yuv420p** (orchestrator-confirmed bottleneck). **`ltx_pipelines` is an EXTERNAL git dependency** (`Lightricks/LTX-2`, rev `a2c3f240…`, `backend/pyproject.toml:48`), **not vendored in this repo** — so `media_io.encode_video` cannot be edited here. No codec, pixel-format, container, or bit-depth parameter is threaded through this signature.
- **Callers (uniform signature):**
  - `services/fast_video_pipeline/ltx_fast_video_pipeline.py:180` and `:200` (fps=8 fallback path)
  - `services/a2v_pipeline/ltx_a2v_pipeline.py:172`
  - `services/ic_lora_pipeline/ltx_ic_lora_pipeline.py:436` and `:830`
  - `services/retake_pipeline/ltx_retake_pipeline.py:361` calls `encode_video(...)` **directly** (bypasses the wrapper) with the same 5 kwargs — must be updated in lockstep with any encode-path change.
- **EXR/MOV implication:** the decoded-frame tensor/iterator is the natural ProRes/EXR input. Because `ltx_pipelines` is an external (non-vendored) dependency, the correct interception point is **this wrapper** (`encode_video_output`) — branch to a new encoder service (ProRes via `ffmpeg` subprocess, EXR via a writer) using the decoded-frame tensor *before* the `encode_video` import/call, rather than editing `media_io` itself. The direct retake call must be re-routed through the same new encoder.

### `DistilledNativePipeline` (`ltx_pipeline_common.py:53`)

Fast native T2V/A2V path. `__init__` builds (lazy imports from `ltx_pipelines.utils.blocks`): `PromptEncoder`, `ImageConditioner`, `DiffusionStage` (with `build_policy(checkpoint_path)` from `ltx_core.quantization.fp8_cast` when `fp8transformer and device_supports_fp8(device)`), `VideoDecoder`, `AudioDecoder`; all bf16 on `device` (`get_device()` if None). `__call__` (decorated `@torch.inference_mode()`) returns `(decoded_video, decoded_audio)` — i.e. the exact `(video, audio)` tuple handed to `encode_video_output()`.

### `services_utils.py` device helpers

`get_device_type()` normalizes `str | torch.device | object` → `"cuda"|"mps"|"cpu"`. `device_supports_fp8()` is true only for `cuda`. `sync_device()` and `empty_device_cache()` dispatch to `torch.cuda.*` / `torch.mps.*` with logged fallbacks — consumed by `TorchCleaner` and pipeline teardown.

## Integration Points

- **`app_handler.py`** — `ServiceBundle` (line 230) types every field against the Protocols re-exported here; `build_default_service_bundle()` instantiates the real `*Impl` classes from subpackages. `services/__init__.py` is the import source for those type hints.
- **`ltx_components.resolve_components()`** → consumed by pipeline constructors (fast/a2v/ic-lora/retake) to drive `DistilledPipeline`/`DistilledNativePipeline` builder wiring and to key the pipeline cache (`cache_key`).
- **`ltx_pipeline_common.encode_video_output()`** → **the single chokepoint for the EXR/MOV-Primary-Output work.** All four generation pipelines funnel decoded frames through it (retake bypasses via a direct `encode_video` call that must change in tandem). `ltx_pipelines` is an external dependency, so the new ProRes/EXR encoder must be inserted here (before the `encode_video` call), not inside `media_io`.
- **HDR pipeline (`hdr_ic_lora_pipeline/`)** — `LTXHdrIcLoraPipeline(HDRICLoraPipeline)` is the dedicated HDR V2V path. It overrides only `_create_conditionings` (decode all source frames + in-memory duplicate-final-frame pad to `8n+1`), delegates stage 1 / upsampler / stage 2 / decode to upstream unchanged, and writes the linear HDR tensor as a primary EXR sequence via `ltx_pipelines.utils.media_io.save_exr_tensor` (no EOTF/tonemap/clamp) then an SDR proxy via `encode_exr_sequence_to_mp4`. **It does NOT use `encode_video_output`** — HDR is the one pipeline with its own EXR writer.
- **`services/utils` device helpers** → `services/gpu_cleaner/torch_cleaner.py` (`empty_device_cache`), pipeline `__del__`/teardown paths (`sync_device`).
- **`tests/fakes/services.py`** — fakes implement the Protocols aggregated in `interfaces.py`; `conftest.py` wires a fresh `AppHandler` per test with the fake bundle.
- **Sibling subpackages** (`http_client/`, `gpu_info/`, `gpu_cleaner/`, `model_downloader/`, `patches/`) each own their own Protocol + Impl and are documented in their own `codemap.md`.
