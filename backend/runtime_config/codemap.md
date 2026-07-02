# backend/runtime_config

## Responsibility

Static, process-startup configuration: the `RuntimeConfig` value object, the
policy decisions that derive hardware capability from the host, the canonical
download/checkpoint/adapter spec catalog, and the default backend port. Nothing
here mutates at runtime (the lone exception is `model_download_specs` performing
import-time self-validation). Handlers read these values via
`StateHandlerBase.config` and the spec lookup functions; they never write back.

Files:
- `runtime_config.py` — `RuntimeConfig` dataclass.
- `runtime_policy.py` — `LocalGenerationMode` + mode/prefetch decisions.
- `model_download_specs.py` — checkpoint & LTX model & adapter spec catalog and path resolvers.
- `port_constant.py` — default backend port constant.
- `__init__.py` — re-exports `RuntimeConfig`.

## Design Patterns

**Frozen dataclass value object.** `RuntimeConfig` is a plain `@dataclass`
populated once at startup; derived flag `force_api_generations` is a `@property`
(`local_generations_mode == "unsupported"`), not stored state.

**Literal + total function for policy.** `LocalGenerationMode = Literal[
"full_models_loading", "streaming_models_loading", "unsupported"]`.
`decide_local_generation_mode(system, cuda_available, vram_gb)` is a total,
side-effect-free function that fails closed (returns `"unsupported"` for Darwin,
non-CUDA, unknown VRAM, or <15 GB). `streaming_prefetch_count_for_mode(mode)`
maps modes to `None` (full) / `2` (streaming) and raises `AssertionError` for
`"unsupported"` — forcing callers to route to the API rather than build a local
pipeline.

**Exhaustive spec catalogs via `match` + `assert_never`.**
`model_download_specs.get_model_cp_spec` and `get_ltx_model_spec` are total
matches over `ModelCheckpointID` / `LTXLocalModelId` literals (re-exported from
`api_types`); the `case _: assert_never(...)` makes a missing spec a startup
crash. `ALL_MODEL_CP_IDS` / `ALL_LTX_LOCAL_MODEL_IDS` are derived with
`get_args(...)` so the literals stay the single source of truth.

**Frozen, slotted dataclasses for specs.** `ModelCheckpointSpec`,
`LTXLocalModelSpec`, `LtxIcLorasSpec`, `AdapterComponent`,
`LTXLocalModelRelevant`/`LTXLocalModelDeprecated` are all
`@dataclass(frozen=True, slots=True)`.

**Import-time invariant validation.** At module load,
`_validate_model_cp_specs()` asserts no two CP ids map to the same normalised
relative path, and `_validate_ltx_specs()` asserts LTX primary checkpoints map
1:1 with model ids and that `get_latest_ltx_model_id()` resolves to exactly one
relevant model. A misconfiguration therefore aborts startup, not a later
request.

**Path normalisation as the security boundary.** `_normalized_relative_path`
rejects absolute paths, empty paths, and any `..` traversal before joining to
`models_dir`; every public resolver (`resolve_model_path`,
`resolve_downloading_*`) goes through it.

## Data & Control Flow

**`runtime_config.RuntimeConfig`** fields: `device: torch.device`,
`app_data_dir`, `default_models_dir`, `outputs_dir`, `settings_file` (all
`pathlib.Path`), `ltx_api_base_url: str`, `local_generations_mode:
LocalGenerationMode`, `use_sage_attention: bool`, `camera_motion_prompts:
dict[str, str]`, `default_negative_prompt: str`, `dev_mode: bool`,
`backend_port: int`, `hf_oauth_client_id: str = ""`, `hf_gating_enabled: bool =
False`. `force_api_generations` derives from `local_generations_mode`. This
object threads through every `StateHandlerBase` subclass and is the single
source for device, directories, ports, prompts, and capability flags.

**`runtime_policy.py`** — `decide_local_generation_mode` gating:
Darwin→`"unsupported"`; Windows/Linux require `cuda_available`, known `vram_gb`,
and ≥15 GB; 15–30 GB→`"streaming_models_loading"`, ≥31 GB→
`"full_models_loading"`. `streaming_prefetch_count_for_mode` → `None` (full),
`2` (streaming). Consumed by `pipelines_handler` (prefetch count passed into
every `*.create(...)`) and indirectly by every handler that checks
`config.force_api_generations`.

**`model_download_specs.py`**:
- Constants: `ALL_MODEL_CP_IDS`, `ALL_LTX_LOCAL_MODEL_IDS`,
  `IMG_GEN_MODEL_CP_ID = "z-image-turbo"`,
  `DEPTH_PROCESSOR_CP_ID = "dpt-hybrid-midas"`,
  `PERSON_DETECTOR_CP_ID = "yolox-l-torchscript"`,
  `POSE_PROCESSOR_CP_ID = "dw-ll-ucoco-384-bs5"`.
- `OFFICIAL_LTX23_ADAPTERS: dict[AdapterID, AdapterComponent]` enumerates every
  downloadable LTX-2.3 adapter with `id`, `display_name`, `kind`
  (`distilled_lora`/`ic_lora`/`embeddings`), `source`, `repo_id`, `filename`,
  `expected_size_bytes`, and `required_for`/`optional_for` pipeline tuples.
- `ADAPTER_TO_CP_ID: dict[AdapterID, ModelCheckpointID]` maps each adapter to
  its downloadable checkpoint (e.g. `union_control`→
  `ltx-2.3-22b-ic-lora-union-control-ref0.5`, `hdr_scene_embeddings`→
  `ltx-2.3-22b-ic-lora-hdr-scene-emb`).
- `get_model_cp_spec(cp_id) → ModelCheckpointSpec(relative_path,
  expected_size_bytes, is_folder, repo_id, description)` (`.name` property).
  Covers the distilled transformer, upscaler, every IC-LoRA, the depth/pose/person
  processors, `gemma-3-12b-it-qat-q4_0-unquantized`, and `z-image-turbo`.
- `get_ltx_model_spec(model_id) → LTXLocalModelSpec(model_cp, upscale_cp,
  text_encoder_cp, ic_loras_spec: LtxIcLorasSpec(depth_cp, canny_cp, pose_cp),
  relevance, supported_pipelines)`. For `ltx-2.3-22b-distilled` all three
  IC-LoRA CPs point at the union-control checkpoint.
- Relationship helpers: `get_ltx_cps()`, `get_latest_ltx_model_id()` (exactly
  one relevant), `get_ltx_model_id_for_cp(cp_id)`,
  `get_ic_loras_cp_ids(spec)`, `get_ltx_model_cp_ids(model_id)`.
- Filesystem resolvers: `resolve_model_path`, `resolve_downloading_dir`
  (`models_dir/.downloading`), `resolve_downloading_target_path`,
  `resolve_downloading_path`; predicates `is_cp_downloaded` (folder must be
  non-empty), `get_existing_cp_path` (raises `FileNotFoundError`),
  `delete_cp_path` (rmtree for folders), `get_downloaded_ltx_model_id` (returns
  the single downloaded model or warns/selects among multiples).

**`port_constant.py`** — `PORT = 41954`. This is the one hard-coded port
constant in the folder. The actual served port is carried at runtime by
`RuntimeConfig.backend_port` (populated at startup and consumed by
`HuggingFaceAuthHandler._redirect_uri` as
`http://127.0.0.1:{backend_port}/api/auth/huggingface/callback`). Project docs
(`AGENTS.md`) describe the FastAPI dev server on port **8000**; the binding
itself is established outside this folder (Electron spawn / `app_factory`),
which is why the constant (41954) and the documented default (8000) can differ —
`port_constant.PORT` is the canonical numeric reference, while
`RuntimeConfig.backend_port` is the value actually embedded in URLs.

**`__init__.py`** — exports `RuntimeConfig` only.

## Integration Points

- **`handlers/base.StateHandlerBase`** stores `RuntimeConfig` as `self.config`
  and exposes it via the `config` property; `models_dir` falls back to
  `config.default_models_dir` when settings omit it.
- **`handlers/pipelines_handler`** imports
  `streaming_prefetch_count_for_mode`, `IMG_GEN_MODEL_CP_ID`,
  `get_downloaded_ltx_model_id`, `get_existing_cp_path`, `get_ltx_model_spec`.
- **`handlers/models_handler`** imports the spec catalog, adapter map, and
  `LTXLocalModelRelevant`/`AdapterComponent`/`OFFICIAL_LTX23_ADAPTERS`/
  `ADAPTER_TO_CP_ID`/`get_ltx_*`.
- **`handlers/download_handler`** imports `ALL_MODEL_CP_IDS`,
  `get_model_cp_spec`, `is_cp_downloaded`, `resolve_downloading_*`,
  `resolve_model_path`.
- **`handlers/ic_lora_handler`** imports `DEPTH_PROCESSOR_CP_ID`,
  `OFFICIAL_LTX23_ADAPTERS`, `get_downloaded_ltx_model_id`,
  `get_existing_cp_path`, `get_latest_ltx_model_id`, `get_ltx_model_spec`.
- **`handlers/text_handler`** imports `get_downloaded_ltx_model_id`,
  `get_existing_cp_path`, `get_ltx_model_spec`, `is_cp_downloaded`.
- **`state/app_state_types.py`** references `ModelCheckpointID` (from
  `api_types`, mirrored here) inside `FileDownloadRunning`.
- **`api_types.py`** is the upstream source of the `ModelCheckpointID`,
  `LTXLocalModelId`, `AdapterID`/`AdapterKind`/`AdapterPipeline`/`AdapterSource`,
  and `LTXVideoGen*` literals that this catalog matches against; the
  `assert_never` arms keep the two in lock-step.
- **`services/ltx_components.py`** consumes profile component paths that are
  validated against `OFFICIAL_LTX23_ADAPTERS` filenames via
  `model_profiles_handler`.
