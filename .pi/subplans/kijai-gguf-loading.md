# Slice D Plan — Kijai Split Safetensors + GGUF Transformer Loading

## Status
split-required

## Why Split
This slice touches shared backend model-resolution and pipeline-construction seams. Implement sequentially: first pass a resolved component bundle through existing pipelines, then enable Kijai split safetensors, then add GGUF transformer support behind the same bundle. Parallel work would collide in pipeline protocol files and `PipelinesHandler` cache/state logic.

## Scope
- Backend loading path for local LTX video pipelines.
- Kijai split-component `.safetensors` support for transformer + text projection/embeddings connector + video VAE + audio VAE + spatial upsampler.
- GGUF transformer support for `ltxv` architecture using Kijai/QuantStack-style GGUF plus safetensors non-transformer components.

## Non-Goals
- No frontend wizard redesign in this slice.
- No official adapter registry expansion beyond existing IC-LoRA path plumbing.
- No GGUF LoRA patching in first GGUF pass; fail clearly if LoRAs are requested with GGUF.
- No new ComfyUI runtime dependency.
- No changes to source repos under `/tmp/clones/LTX-2`, `/tmp/clones/ComfyUI-GGUF`, or `/tmp/clones/ComfyUI-KJNodes`.

## Preconditions / Stop Gate
Stop before source edits if the fork does not already have model-profile storage/API from prior slices. This slice should consume an active profile with component paths; it should not recreate profile CRUD/UI. On the inspected `/tmp/clones/LTX-Desktop` snapshot, profile code is not present yet, so worker must confirm prior slice exists in target branch/worktree.

## Evidence Used
- `/tmp/clones/LTX-Desktop/backend/handlers/pipelines_handler.py`: resolves fixed official checkpoint paths and constructs fast, IC-LoRA, A2V, retake pipelines.
- `/tmp/clones/LTX-Desktop/backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`: wraps `ltx_pipelines.distilled.DistilledPipeline` with single `checkpoint_path`.
- `/tmp/clones/LTX-Desktop/backend/services/a2v_pipeline/distilled_a2v_pipeline.py`: custom block-based A2V pipeline; all component builders use one checkpoint path.
- `/tmp/clones/LTX-Desktop/backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`: IC-LoRA wrapper passes one checkpoint to official `ICLoraPipeline`.
- `/tmp/clones/LTX-Desktop/backend/services/retake_pipeline/ltx_retake_pipeline.py`: forked retake pipeline directly builds `PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoDecoder`, `AudioDecoder` from one checkpoint.
- `/tmp/clones/LTX-Desktop/backend/services/ltx_pipeline_common.py`: `DistilledNativePipeline` direct block-based path uses one checkpoint.
- `/tmp/clones/LTX-Desktop/backend/runtime_config/model_download_specs.py`: official checkpoint literals and fixed download path mapping; do not force Kijai/GGUF into these literals.
- `/tmp/clones/LTX-2/packages/ltx-core/src/ltx_core/loader/helpers.py`: `load_state_dict()` accepts `str | tuple[str, ...] | list[str]`; config metadata is read from first path only.
- `/tmp/clones/LTX-2/packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py`: `SingleGPUModelBuilder(model_path: str | tuple[str, ...])` supports tuple checkpoint paths.
- `/tmp/clones/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py`: `DiffusionStage` accepts custom `transformer_builder`; other blocks internally create builders from `checkpoint_path`.
- `/tmp/clones/ComfyUI-GGUF/loader.py`: `gguf_sd_loader()` handles `ltxv`, strips `model.diffusion_model.` prefix, reads metadata and original shapes.
- `/tmp/clones/ComfyUI-GGUF/ops.py` + `/tmp/clones/ComfyUI-GGUF/dequant.py`: reference dynamic dequantized linear layer behavior.
- `/tmp/clones/ComfyUI-KJNodes/nodes/model_optimization_nodes.py`: `GGUFLoaderKJ` merges extra connector state dict into GGUF state dict and strips `model.diffusion_model.` prefix.
- `/home/npittas/ltx_offline/research/05-ltx-desktop-model-profiles-and-gguf-kijai-plan.md`: recommends component bundle and tuple safetensors path.
- `/home/npittas/ltx_offline/research/06-revised-ltx-desktop-implementation-roadmap.md`: places Kijai split before GGUF; GGUF depends on component profiles.

## Dependency Choice
- Add direct backend dependency: `gguf>=0.13.0` in `backend/pyproject.toml`; update `backend/uv.lock`.
- Do not add `ComfyUI-GGUF` or `ComfyUI-KJNodes` as runtime deps; they pull ComfyUI-only APIs.
- Do not use Diffusers GGUF path for this slice; LTX-Desktop pipelines use `ltx-core` `X0Model`/`DiffusionStage`, and Diffusers transformer objects are not drop-in compatible.
- Reuse existing `torch`, `ltx-core`, `ltx-pipelines`, and transitive `safetensors`.

## Interference Check
- parallel safe: no
- shared files: `backend/handlers/pipelines_handler.py`, pipeline protocol files, concrete pipeline wrappers, backend dependency lockfile
- shared generated outputs: `backend/uv.lock`
- shared validation state: backend pytest/pyright
- worktree isolation required: recommended if another worker touches model profiles, downloads, or pipeline handlers

# Implementation Breakdown

## D1 — Resolved component bundle seam

### Allowed Edit Files
- `backend/services/ltx_components.py` (new)
- `backend/handlers/pipelines_handler.py`
- `backend/state/app_state_types.py`
- `backend/services/fast_video_pipeline/fast_video_pipeline.py`
- `backend/services/a2v_pipeline/a2v_pipeline.py`
- `backend/services/ic_lora_pipeline/ic_lora_pipeline.py`
- `backend/services/retake_pipeline/retake_pipeline.py`
- `backend/tests/test_ltx_components.py` (new)

### Required Classes / Functions
Introduce `backend/services/ltx_components.py`:
- `TransformerFormat = Literal["safetensors", "gguf"]`
- `ResolvedLtxComponents` dataclass with:
  - `profile_id: str`
  - `transformer_format: TransformerFormat`
  - `transformer_path: str`
  - `checkpoint_paths_for_filtered_builders: tuple[str, ...]`
  - `upsampler_path: str`
  - `gemma_root: str | None`
  - `text_projection_path: str | None`
  - `embeddings_connector_path: str | None`
  - `video_vae_path: str | None`
  - `audio_vae_path: str | None`
  - `cache_key: tuple[str, ...]`
- `resolve_official_ltx_components(models_dir, model_spec, gemma_root) -> ResolvedLtxComponents`
- `resolve_profile_ltx_components(active_profile, fallback_official_spec, gemma_root) -> ResolvedLtxComponents`
- `checkpoint_path_arg(components) -> str | tuple[str, ...]`; one-line helper returns single string for length 1, tuple otherwise.

Alter protocol `create(...)` signatures to accept `components: ResolvedLtxComponents` instead of raw `checkpoint_path`, `gemma_root`, and `upsampler_path` for:
- `FastVideoPipeline.create`
- `A2VPipeline.create`
- `IcLoraPipeline.create`
- `RetakePipeline.create`

Alter `backend/state/app_state_types.py`:
- Add `components_cache_key: tuple[str, ...]` to `VideoPipelineState`, `A2VPipelineState`, `ICLoraState`, and `RetakePipelineState` or equivalent small cache identity field.

Alter `PipelinesHandler`:
- Add `_resolve_ltx_components()` that asks the prior-slice profile service for active profile if present, otherwise returns official model components using existing `get_existing_cp_path()`.
- Cache-match active GPU pipelines by both pipeline kind and `components.cache_key`; profile switch must evict/recreate.
- Keep `_install_text_patches_if_needed()` behavior.
- For unsupported profile/features, raise `HTTPError(409, "PROFILE_UNSUPPORTED_<FEATURE>")` with specific feature name.

### Stop Conditions
- Stop if prior-slice active profile object/service is absent.
- Stop if adding this seam requires frontend or endpoint design not already specified.
- Stop if pyright cannot represent tuple checkpoint paths without broad `Any`; use one local `cast(Any, checkpoint_path_arg(...))` at third-party constructor boundary only.

### Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_components.py`
- `rtk pnpm typecheck:py`

Expected:
- Official profile resolves to existing single checkpoint path.
- Split/GGUF profiles produce stable `cache_key` including every component path.
- Pipeline cache mismatch on profile change is tested.

## D2 — Kijai split safetensors path

### Allowed Edit Files
- `backend/services/ltx_components.py`
- `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
- `backend/services/a2v_pipeline/ltx_a2v_pipeline.py`
- `backend/services/a2v_pipeline/distilled_a2v_pipeline.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/services/retake_pipeline/ltx_retake_pipeline.py`
- `backend/services/ltx_pipeline_common.py`
- `backend/tests/test_ltx_split_safetensors.py` (new)

### Required Changes
- For `transformer_format == "safetensors"`, pass `checkpoint_path_arg(components)` to all ltx-core/ltx-pipelines builders that previously received one monolithic checkpoint path.
- For official monolith: `checkpoint_paths_for_filtered_builders == (official_checkpoint,)`.
- For Kijai split safetensors: `checkpoint_paths_for_filtered_builders == (transformer, text_projection_or_connector, video_vae, audio_vae)` with transformer first because `read_model_config()` reads metadata from first path.
- In `LTXFastVideoPipeline`, replace stored `_checkpoint_path`, `_gemma_root`, `_upsampler_path` with `_components`; construct `DistilledPipeline(... distilled_checkpoint_path=checkpoint_path_arg(components), gemma_root=components.gemma_root, spatial_upsampler_path=components.upsampler_path, ...)`.
- In `DistilledA2VPipeline.__init__`, rename `distilled_checkpoint_path: str` to `checkpoint_paths: str | tuple[str, ...]`; pass it to `PromptEncoder`, `ImageConditioner`, `AudioConditioner`, `DiffusionStage`, `VideoUpsampler`, and `VideoDecoder`.
- In `LTXIcLoraPipeline`, pass `checkpoint_path_arg(components)` and `components.upsampler_path`; keep LoRA mapping `LTXV_LORA_COMFY_RENAMING_MAP`.
- In `LTXRetakePipeline`, pass `checkpoint_path_arg(components)` to all block builders.
- In `DistilledNativePipeline`, accept `components` or `checkpoint_paths: str | tuple[str, ...]`; use it for `PromptEncoder`, `ImageConditioner`, `DiffusionStage`, `VideoDecoder`, `AudioDecoder`.
- Profile validation must require split safetensors component paths to exist and end with `.safetensors`; require `upsampler_path` too.

### Risk Gates
- Metadata: if Kijai transformer-only file lacks `config` metadata, stop and add explicit `config_source_path`/sidecar handling in profile layer before continuing.
- Key coverage: if any builder logs uninitialized parameters for split profile, stop and list missing key prefixes rather than falling back silently.
- A2V/retake: if audio VAE/vocoder keys are absent from Kijai files, mark A2V/retake unsupported for that profile with clear 409; do not fake audio output.

### Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_split_safetensors.py`
- `rtk pnpm backend:test -- tests/test_generation.py tests/test_ic_lora.py`
- `rtk pnpm typecheck:py`

Manual GPU smoke, when real Kijai files are available:
- `rtk pnpm dev` then create/activate Kijai split profile and run one 540p/5s fast T2V with API text encoding.

Expected:
- Existing official monolithic tests still pass.
- Split profile constructs pipeline with tuple checkpoint paths.
- Missing split component fails at validation/profile activation, not during generation.

## D3 — GGUF transformer loader path

### Allowed Edit Files
- `backend/pyproject.toml`
- `backend/uv.lock`
- `backend/services/gguf_dequant.py` (new)
- `backend/services/gguf_transformer_loader.py` (new)
- `backend/services/ltx_components.py`
- `backend/services/fast_video_pipeline/ltx_fast_video_pipeline.py`
- `backend/services/a2v_pipeline/distilled_a2v_pipeline.py`
- `backend/services/ic_lora_pipeline/ltx_ic_lora_pipeline.py`
- `backend/services/retake_pipeline/ltx_retake_pipeline.py`
- `backend/services/ltx_pipeline_common.py`
- `backend/tests/test_gguf_transformer_loader.py` (new)

### Required Classes / Functions
Add `backend/services/gguf_dequant.py`:
- Minimal adapted Apache-2.0 code from `ComfyUI-GGUF/dequant.py`; include attribution header.
- Functions: `is_torch_compatible()`, `is_quantized()`, `dequantize_tensor()`.
- Start with qtypes needed by target LTX GGUF: `Q4_K`, `Q5_K`, `Q6_K`, `Q8_0`, `BF16`, `F16`, `F32`; fallback to `gguf.quants.dequantize()` with warning for other qtypes.

Add `backend/services/gguf_transformer_loader.py`:
- `GGUFLoadError(RuntimeError)`
- `GGUFQuantizedTensor(torch.Tensor)` carrying `tensor_type` and `tensor_shape` metadata; can mirror `ComfyUI-GGUF.ops.GGMLTensor` without Comfy imports.
- `GGUFLinear(torch.nn.Module)`:
  - same constructor shape as `torch.nn.Linear`
  - `_load_from_state_dict()` assigns quantized `weight`/`bias` as frozen parameters
  - `forward()` dequantizes weight/bias to input dtype/device and calls `torch.nn.functional.linear()`
  - ponytail: first pass dequantizes per call; add cache only after measured bottleneck/OOM behavior is known.
- `replace_linear_with_gguf(module: torch.nn.Module) -> torch.nn.Module` recursively swaps `torch.nn.Linear` to `GGUFLinear` while preserving dimensions/bias.
- `GGUF_LINEAR_MODULE_OP = ModuleOps(name="gguf_linear", matcher=lambda m: True, mutator=replace_linear_with_gguf)`.
- `GGUFStateDictLoader(StateDictLoader)`:
  - `metadata(path)` reads `general.architecture`; require `ltxv`; parse `config` metadata as JSON; if absent, raise `GGUFLoadError("GGUF_CONFIG_METADATA_MISSING")`.
  - `load(path, sd_ops, device)` accepts one `.gguf` path plus optional safetensors connector state merged later only if needed; strip `model.diffusion_model.` prefix if present; apply `sd_ops.apply_to_key()`; build `StateDict` with quantized tensors.
- `build_gguf_transformer_builder(components, registry=None)`:
  - returns `SingleGPUModelBuilder(model_class_configurator=LTXModelConfigurator, model_path=components.transformer_path, model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP, model_loader=GGUFStateDictLoader(), module_ops=(GGUF_LINEAR_MODULE_OP,), registry=...)`.

Alter pipeline wrappers:
- When `components.transformer_format == "gguf"`, construct `DiffusionStage(... transformer_builder=build_gguf_transformer_builder(components))` instead of letting `DiffusionStage` build transformer from safetensors.
- Non-transformer blocks still use `components.checkpoint_paths_for_filtered_builders` containing Kijai text projection/connector + video VAE + audio VAE safetensors.
- Disable `compile_transformer()` for GGUF profiles in `PipelinesHandler._compile_if_enabled()` or inside pipeline wrappers; custom tensor subclass + dynamic dequant path is not a safe initial compile target.
- Disable LoRA fusion for GGUF in this slice. If LoRA/IC-LoRA requested with GGUF, raise `HTTPError(409, "GGUF_LORA_UNSUPPORTED")` unless future slice adds runtime patching.

### Risk Gates
- Import gate: if `gguf` import fails, profile validation must fail with `GGUF_DEPENDENCY_MISSING` before generation.
- Architecture gate: require `general.architecture == "ltxv"`.
- Metadata gate: require GGUF `config` metadata initially. If QuantStack/Kijai target files lack it, stop and plan explicit LTX-2.3 config sidecar instead of guessing dimensions.
- Compile gate: torch compile disabled for GGUF until dedicated tests prove stability.
- Performance gate: first implementation may be slower due per-call dequant. Do not add global cache until memory profile is measured.

### Validation
Commands:
- `rtk pnpm backend:test -- tests/test_gguf_transformer_loader.py`
- `rtk pnpm backend:test -- tests/test_ltx_split_safetensors.py`
- `rtk pnpm typecheck:py`

Manual GPU smoke, when real files are available:
- Activate GGUF profile using `LTX-2.3-dev-Q4_K_M.gguf` + Kijai text projection/video VAE/audio VAE + official/Kijai upsampler.
- Generate 540p/5s fast T2V with API text encoding.

Expected:
- GGUF validation rejects missing dependency, non-`.gguf`, non-`ltxv`, missing config metadata.
- Loader test verifies prefix stripping, shape reversal/original-shape handling, and qtype preservation.
- Existing safetensors path unaffected.

## D4 — Tests and integration guardrails

### Allowed Edit Files
- `backend/tests/test_ltx_components.py`
- `backend/tests/test_ltx_split_safetensors.py`
- `backend/tests/test_gguf_transformer_loader.py`
- `backend/tests/test_generation.py`
- `backend/tests/test_ic_lora.py`
- `backend/tests/test_pyright.py` only if import surface needs explicit inclusion

### Required Tests
- Official monolith profile preserves current path behavior.
- Split safetensors profile yields tuple path in stable order: transformer first.
- Profile cache key changes when any component path changes.
- `PipelinesHandler` reloads GPU pipeline on component cache key change.
- Split missing component returns validation error before generation.
- GGUF dependency/architecture/config gates return deterministic errors.
- GGUF + LoRA/IC-LoRA returns `GGUF_LORA_UNSUPPORTED`.
- No `unittest.mock` usage; use fake profile service / fake pipeline classes like existing backend tests.

### Validation
Commands:
- `rtk pnpm backend:test -- tests/test_ltx_components.py tests/test_ltx_split_safetensors.py tests/test_gguf_transformer_loader.py tests/test_generation.py tests/test_ic_lora.py`
- `rtk pnpm typecheck:py`
- `rtk pnpm backend:test`

Expected:
- Full backend tests pass.
- Pyright strict pass.

# Final Risk Gates Before Merge
- Real Kijai split safetensors smoke passes or split support remains behind explicit experimental flag/profile validation warning.
- Real GGUF smoke passes on one target file or GGUF profile activation remains blocked with exact reason.
- Existing official download/recommendation path unchanged.
- API text encoding still works when local Gemma root is absent.
- No ComfyUI imports in backend runtime.
- `backend/uv.lock` updated in same change as `backend/pyproject.toml`.

# Recommended Validation Commands
```bash
rtk pnpm typecheck:py
rtk pnpm backend:test -- tests/test_ltx_components.py tests/test_ltx_split_safetensors.py tests/test_gguf_transformer_loader.py tests/test_generation.py tests/test_ic_lora.py
rtk pnpm backend:test
```

# Worker Return Contract
Return status, files changed, tests run with pass/fail, exact blocker if a risk gate trips, and whether real Kijai/GGUF smoke was run. Do not include broad refactors or unrelated model-profile/UI work.
