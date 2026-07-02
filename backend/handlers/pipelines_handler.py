"""Pipeline lifecycle handler."""

from __future__ import annotations

import logging
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from _routes._errors import HTTPError
from api_types import LTXLocalModelId, ModelSelectionID
from handlers.base import StateHandlerBase
from handlers.text_handler import TextHandler
from runtime_config.model_download_specs import (
    IMG_GEN_MODEL_CP_ID,
    OFFICIAL_LTX23_ADAPTERS,
    UPSAMPLER_CP_ID,
    get_downloaded_ltx_model_id,
    get_existing_cp_path,
    get_ltx_model_spec,
    resolve_model_path,
)
from runtime_config.runtime_policy import offload_mode_value_for_mode
from services.base_video_model_registry import (
    BaseVideoModelRegistryEntry,
    resolve_base_video_model_selection,
)
from services.ltx_components import (
    CheckpointPath,
    ResolvedLtxComponents,
    checkpoint_path_arg,
    resolve_components,
)
from services.interfaces import (
    A2VPipeline,
    DepthProcessorPipeline,
    FastVideoPipeline,
    HdrIcLoraPipeline,
    ImageGenerationPipeline,
    GpuCleaner,
    IcLoraPipeline,
    PoseProcessorPipeline,
    RetakePipeline,
    VideoPipelineModelType,
)
from services import memory_trace
from services.local_memory_plan import (
    LocalMemoryPlan,
    Workflow,
    detect_vram_gb,
    plan_for_transformer,
)
from services.services_utils import device_supports_fp8, get_device_type
from state.app_state_types import (
    A2VPipelineState,
    AppState,
    CpuSlot,
    GpuGeneration,
    GenerationRunning,
    GpuSlot,
    HdrICLoraState,
    ICLoraState,
    RetakePipelineState,
    VideoPipelineState,
)

if TYPE_CHECKING:
    from runtime_config.runtime_config import RuntimeConfig

logger = logging.getLogger(__name__)


class PipelinesHandler(StateHandlerBase):
    def __init__(
        self,
        state: AppState,
        lock: RLock,
        text_handler: TextHandler,
        gpu_cleaner: GpuCleaner,
        fast_video_pipeline_class: type[FastVideoPipeline],
        image_generation_pipeline_class: type[ImageGenerationPipeline],
        ic_lora_pipeline_class: type[IcLoraPipeline],
        hdr_ic_lora_pipeline_class: type[HdrIcLoraPipeline],
        depth_processor_pipeline_class: type[DepthProcessorPipeline],
        pose_processor_pipeline_class: type[PoseProcessorPipeline],
        a2v_pipeline_class: type[A2VPipeline],
        retake_pipeline_class: type[RetakePipeline],
        config: RuntimeConfig,
    ) -> None:
        super().__init__(state, lock, config)
        self._text_handler = text_handler
        self._gpu_cleaner = gpu_cleaner
        self._fast_video_pipeline_class = fast_video_pipeline_class
        self._image_generation_pipeline_class = image_generation_pipeline_class
        self._ic_lora_pipeline_class = ic_lora_pipeline_class
        self._hdr_ic_lora_pipeline_class = hdr_ic_lora_pipeline_class
        self._depth_processor_pipeline_class = depth_processor_pipeline_class
        self._pose_processor_pipeline_class = pose_processor_pipeline_class
        self._a2v_pipeline_class = a2v_pipeline_class
        self._retake_pipeline_class = retake_pipeline_class
        self._runtime_device = get_device_type(self.config.device)

    def _ensure_no_running_generation(self) -> None:
        match self.state.active_generation:
            case GpuGeneration(state=GenerationRunning()) if self.state.gpu_slot is not None:
                raise RuntimeError("Generation already running; cannot swap pipelines")
            case _:
                return

    def _resolve_selection(self, model_selection: ModelSelectionID) -> BaseVideoModelRegistryEntry:
        """Validate a present ``model_selection`` and return its registry entry.

        Delegates to the unified base-video registry
        (:func:`resolve_base_video_model_selection`). Raises clear, actionable
        HTTP errors (never silent fallback) when the selection is unknown or not
        installed:

        - ``UNSUPPORTED_MODEL_SELECTION`` (422): the id is not a registered
          selectable base video transformer (e.g. an upscaler/adapter id or an
          arbitrary unknown string). The registry raises ``KeyError`` for
          unknown ids; it is translated here to the HTTP error so the services
          layer stays free of route imports.
        - ``MODEL_SELECTION_NOT_INSTALLED`` (409): the candidate is known but not
          present under the effective models dir; the message names the exact
          canonical placement path from the registry entry.

        Called only when ``model_selection`` is present; absent/None selection
        always falls back to active/current behavior. Returns the full registry
        entry so the caller can pass explicit ``transformer_path``,
        ``transformer_format``, and ``base_family`` downstream — no filename/path
        inference for the selected family/format.
        """
        try:
            entry = resolve_base_video_model_selection(self.models_dir, model_selection)
        except KeyError:
            raise HTTPError(
                422,
                (
                    f"Model selection '{model_selection}' is not a selectable base video "
                    "transformer. Live model selection supports registered base video "
                    "models only; see GET /api/models/model-options for the list."
                ),
                code="UNSUPPORTED_MODEL_SELECTION",
            )
        if not entry.installed:
            raise HTTPError(
                409,
                (
                    f"Selected model '{model_selection}' is not installed. Install it at the "
                    f"canonical placement path: {entry.expected_absolute_path}"
                ),
                code="MODEL_SELECTION_NOT_INSTALLED",
            )
        return entry

    def _pipeline_matches_model_type(
        self,
        model_type: VideoPipelineModelType,
        model_selection: ModelSelectionID | None = None,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> bool:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=VideoPipelineState(pipeline=pipeline, cache_key=cached_key)):
                # Local "fast" (distilled) and "full" (dev/full GGUF) families
                # both run on the FastVideoPipeline (pipeline_kind == "fast").
                # The cache_key (model_selection + effective distilled LoRA path)
                # differentiates the two builds, so either family is kind-
                # compatible with a cached fast video pipeline.
                if not (
                    pipeline.pipeline_kind == "fast" and model_type in ("fast", "full")
                ) and pipeline.pipeline_kind != model_type:
                    return False
                # ponytail: cache_key comparison only; richer invalidation lands with split/GGUF
                expected_key = self._current_cache_key(model_selection, memory_plan)
                return cached_key == expected_key
            case _:
                return False

    def _video_cache_key_for_components(
        self,
        components: ResolvedLtxComponents | None,
        model_selection: ModelSelectionID | None,
    ) -> tuple[str, ...]:
        """Effective fast-video cache key for resolved components.

        For dev base families the ACTUAL effective distilled LoRA path is
        included in the key (explicit-existing preferred, else canonical
        fallback). The effective path is appended whenever it differs from the
        explicit path already baked into ``components.cache_key`` — e.g. when
        the explicit path is stale/missing and the handler falls back to
        canonical — so the key always reflects the real runtime path and a
        second ``load_gpu_pipeline`` with the same selection/profile cache-hits.
        """
        if components is None:
            model_id = get_downloaded_ltx_model_id(self.models_dir)
            if model_id is None:
                return ()
            if model_selection is not None:
                return (model_id, "model_selection", model_selection)
            return (model_id,)
        cache_key = components.cache_key
        if components.base_family == "dev":
            effective_lora = self._resolve_distilled_lora_path(components)
            explicit_lora = components.distilled_lora_path
            if effective_lora is not None and effective_lora != explicit_lora:
                cache_key = (*cache_key, effective_lora)
        return cache_key

    def _current_cache_key(
        self,
        model_selection: ModelSelectionID | None = None,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> tuple[str, ...]:
        components = self._resolve_active_components(model_selection)
        key = self._video_cache_key_for_components(components, model_selection)
        if memory_plan is not None:
            key = (*key, *memory_plan.cache_key_parts)
        return key

    def _assert_invariants(self) -> None:
        match self.state.gpu_slot:
            case GpuSlot(active_pipeline=active_pipeline):
                gpu_has_image_generation_pipeline = isinstance(active_pipeline, ImageGenerationPipeline)
            case _:
                gpu_has_image_generation_pipeline = False

        if gpu_has_image_generation_pipeline and self.state.cpu_slot is not None:
            raise RuntimeError("Invariant violation: image generation pipeline cannot be in both GPU and CPU slots")

    def _install_text_patches_if_needed(self) -> None:
        te = self.state.text_encoder
        if te is None:
            return
        te.service.install_patches(lambda: self.state)

    def _resolve_active_components(
        self, model_selection: ModelSelectionID | None = None
    ) -> ResolvedLtxComponents | None:
        profile_id = self.state.active_model_profile_id
        profile = (
            next((p for p in self.state.model_profiles if p.id == profile_id), None)
            if profile_id is not None
            else None
        )
        if profile is not None:
            if model_selection is None:
                return resolve_components(profile)
            entry = self._resolve_selection(model_selection)
            # Pass explicit selection metadata (transformer path, format, base
            # family, runtime readiness) so downstream code never infers
            # family/format/readiness from the selected path/filename (plan: no
            # path-only inference). Runtime readiness drives whether sidecar
            # metadata is cleared (only ``runtime_readiness == "none"`` is a
            # true monolith).
            return resolve_components(
                profile,
                selected_transformer_path=entry.transformer_path,
                selected_cp_id=model_selection,
                selected_transformer_format=entry.transformer_format,
                selected_base_family=entry.base_family,
                selected_runtime_readiness=entry.runtime_readiness,
                selected_quantization_kind=entry.quantization_kind,
            )

        # No active profile (legacy downloaded-model path).
        if model_selection is not None:
            # Validate (unsupported / not installed) before the profile check.
            entry = self._resolve_selection(model_selection)
            # Only selections whose runtime needs no profile sidecars (the
            # official distilled monolith, ``runtime_readiness == "none"``) can
            # run without an active profile — they reuse the legacy downloaded
            # bundle (upsampler + text encoder). Entries that require split
            # sidecar components need an active profile; reject clearly rather
            # than falling through to a deep pipeline failure.
            if entry.runtime_readiness == "requires_active_profile_sidecars":
                raise HTTPError(
                    409,
                    (
                        f"Live model selection for '{model_selection}' requires an active model profile "
                        "with split components (text projection, VAEs). "
                        "Activate a profile that provides these components and retry."
                    ),
                    code="MODEL_SELECTION_REQUIRES_PROFILE",
                )
        return None

    def _require_downloaded_ltx_model_id(self) -> LTXLocalModelId:
        model_id = get_downloaded_ltx_model_id(self.models_dir)
        if model_id is None:
            raise HTTPError(409, "NO_DOWNLOADED_LTX_MODEL")
        return model_id

    def _compile_if_enabled(
        self,
        state: VideoPipelineState,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> VideoPipelineState:
        if memory_plan is not None and memory_plan.disable_compile:
            logger.info(
                "Skipping torch.compile() - disabled by local memory plan (strategy=%s)",
                memory_plan.strategy,
            )
            return state
        if not self.state.app_settings.use_torch_compile:
            return state
        if state.is_compiled:
            return state
        if self._runtime_device == "mps":
            logger.info("Skipping torch.compile() for %s - not supported on MPS", state.pipeline.pipeline_kind)
            return state
        # GGUF transformers use lazy per-forward dequant that torch.compile
        # cannot trace. Skip silently (info, no traceback) instead of calling
        # compile_transformer() and relying on its RuntimeError guard.
        if not state.pipeline.supports_torch_compile():
            logger.info(
                "Skipping torch.compile() for %s - unsupported transformer format",
                state.pipeline.pipeline_kind,
            )
            return state

        try:
            state.pipeline.compile_transformer()
            state.is_compiled = True
        except Exception as exc:
            logger.warning("Failed to compile transformer: %s", exc, exc_info=True)
        return state

    def _resolve_profile_upsampler_path(self) -> str:
        """Resolve a usable upscaler path for an active profile.

        Prefers the profile's explicit ``components.upsampler`` path when it
        exists on disk. If that explicit path is stale/missing AND the
        canonical upscaler (``latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.0.safetensors``
        under the effective models root) exists, returns the canonical path.
        Otherwise returns an empty string so callers can decide how to surface
        the missing artifact (e.g. fast video fails fast with HTTP 409).
        """
        components = self._resolve_active_components()
        if components is None:
            return ""
        explicit = components.upsampler_path or ""
        if explicit and Path(explicit).exists():
            return explicit
        canonical = resolve_model_path(self.models_dir, UPSAMPLER_CP_ID)
        if canonical.exists():
            return str(canonical)
        return ""

    def _canonical_distilled_lora_candidates(self) -> list[tuple[str, Path]]:
        """Canonical models-dir distilled LoRA paths in preference order.

        Returns ``(adapter_id, path)`` tuples for the newest-then-older
        distilled LoRA filenames declared in ``OFFICIAL_LTX23_ADAPTERS``.
        Adapter canonical placement is ``<models_dir>/adapters/<filename>``
        (matches the scanner's canonical subfolder).
        """
        candidates: list[tuple[str, Path]] = []
        for role in ("distilled_lora_384_1_1", "distilled_lora_384"):
            adapter = OFFICIAL_LTX23_ADAPTERS.get(role)  # type: ignore[arg-type]
            if adapter is None:
                continue
            candidates.append((role, self.models_dir / "adapters" / adapter.filename))
        return candidates

    def _resolve_distilled_lora_path(
        self,
        components: ResolvedLtxComponents | None,
    ) -> str | None:
        """Resolve the effective distilled LoRA path for a dev base profile.

        Preference order:
        1. explicit profile path (``components.distilled_lora_path``)
        2. canonical models-dir fallback using ``OFFICIAL_LTX23_ADAPTERS``
           filenames for ``distilled_lora_384_1_1`` then ``distilled_lora_384``.

        Returns ``None`` when neither exists on disk.
        """
        explicit = components.distilled_lora_path if components is not None else None
        if explicit and Path(explicit).exists():
            return explicit

        for _role, path in self._canonical_distilled_lora_candidates():
            if path.exists():
                return str(path)
        return None

    def _resolve_checkpoint_paths(
        self, model_selection: ModelSelectionID | None = None
    ) -> tuple[CheckpointPath, str | None, str, tuple[str, ...]]:
        """Return (checkpoint_path, gemma_root, upsampler_path, cache_key)."""
        components = self._resolve_active_components(model_selection)
        # TextHandler.resolve_gemma_root currently ignores the selection (the
        # active profile's Gemma root is used regardless); the value is threaded
        # for future per-selection text-encoder routing.
        gemma_root = self._text_handler.resolve_gemma_root(model_selection)
        if components is not None:
            return (
                checkpoint_path_arg(components),
                components.gemma_root or gemma_root,
                self._resolve_profile_upsampler_path(),
                components.cache_key,
            )
        model_id = self._require_downloaded_ltx_model_id()
        spec = get_ltx_model_spec(model_id)
        if model_selection is not None:
            # No-profile path is only reachable for ``runtime_readiness == "none"``
            # selections (the official distilled monolith) — see
            # ``_resolve_active_components``. Resolve its runtime path via the
            # registry (already validated as installed there) instead of the
            # CP-only ``resolve_model_path`` so non-CP distilled-family ids are
            # not routed through the CP catalog.
            entry = resolve_base_video_model_selection(self.models_dir, model_selection)
            selected_path = entry.transformer_path or entry.expected_absolute_path
            cache_key: tuple[str, ...] = (model_id, "model_selection", model_selection)
            return (
                selected_path,
                gemma_root,
                str(get_existing_cp_path(self.models_dir, spec.upscale_cp)),
                cache_key,
            )
        return (
            str(get_existing_cp_path(self.models_dir, spec.model_cp)),
            gemma_root,
            str(get_existing_cp_path(self.models_dir, spec.upscale_cp)),
            (model_id,),
        )

    def _local_offload_mode(self):
        """Resolve the upstream ``OffloadMode`` for the active local generation mode.

        Maps the runtime policy to the upstream offload enum (``NONE`` for full
        residency, ``CPU`` for streaming; never ``DISK``). The policy function
        raises for ``unsupported`` — callers must not build a local pipeline then.
        """
        from ltx_pipelines.utils.types import OffloadMode

        return OffloadMode(offload_mode_value_for_mode(self.config.local_generations_mode))

    def _block_offload_available(self) -> bool:
        """Whether the block-offload residency strategy is wired and usable.

        Phase 3B: delegates to
        :func:`services.block_offload.block_offload_available` (the core service
        + the DiffusionStage ``_build_transformer`` patch imported at server
        boot). The memory planner maps low-VRAM transformers to ``block_offload``;
        such plans always carry ``requires_block_offload=True`` — when this
        returns ``False`` the handler gates construction with a 409 (see
        ``_enforce_memory_plan_gate``).
        """
        from services.block_offload import block_offload_available

        return block_offload_available()

    def _memory_plan_for_components(
        self,
        components: ResolvedLtxComponents | None,
        workflow: Workflow,
    ) -> LocalMemoryPlan:
        """Compute the local memory plan for a resolved transformer.

        Derives (transformer format, base family, componentized-split-ness,
        quantization kind) from ``components`` with safe defaults for the
        no-profile legacy downloaded-model path, probes VRAM via
        :func:`detect_vram_gb` (no GPU injection), and delegates to
        :func:`plan_for_transformer`. ``workflow`` distinguishes HDR
        (``"hdr"``) from the standard non-HDR pipelines (``"standard"``).
        """
        if components is not None:
            transformer_format = components.transformer_format
            base_family = components.base_family
            # ponytail: a multi-path builder tuple is the split/componentized
            # signal (split-safetensors + sidecars, GGUF + sidecars, Kijai).
            is_componentized_split = len(components.checkpoint_paths_for_filtered_builders) > 1
            quantization_kind = components.quantization_kind
        else:
            # No active profile (legacy downloaded-bundle path): mirror the
            # distilled safetensors monolith shape assumed elsewhere here.
            transformer_format = "safetensors"
            base_family = "distilled"
            is_componentized_split = False
            quantization_kind = "bf16"
        return plan_for_transformer(
            transformer_format,
            base_family,
            is_componentized_split,
            quantization_kind,
            detect_vram_gb(),
            workflow,
            block_offload_available=self._block_offload_available(),
        )

    def _enforce_memory_plan_gate(self, memory_plan: LocalMemoryPlan) -> None:
        """Gate heavy load on the memory plan before pipeline construction.

        When the plan requires block offload but the strategy is unavailable,
        emit a ``memory_strategy_gate`` trace event and raise 409 so the client
        sees an actionable error rather than an OOM deep inside the GPU build.
        Phase 3B: block offload is wired, so a ``requires_block_offload`` plan
        proceeds when :meth:`_block_offload_available` is True.
        """
        if not memory_plan.requires_block_offload:
            return
        if self._block_offload_available():
            return
        memory_trace.write_event(
            "memory_strategy_gate",
            "local_memory_strategy",
            strategy=memory_plan.strategy,
            code="LOCAL_MEMORY_STRATEGY_UNAVAILABLE",
            reason=memory_plan.reason,
        )
        raise HTTPError(
            409,
            f"Required local memory strategy is not available: {memory_plan.strategy}",
            code="LOCAL_MEMORY_STRATEGY_UNAVAILABLE",
        )

    def _create_video_pipeline(
        self,
        model_type: VideoPipelineModelType,
        model_selection: ModelSelectionID | None = None,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> VideoPipelineState:
        checkpoint_path, gemma_root, upsampler_path, _resolved_cache_key = self._resolve_checkpoint_paths(model_selection)
        # Fast video pipeline always invokes the spatial upscaler during
        # inference. Fail early with an actionable error instead of letting a
        # FileNotFoundError surface deep inside the diffusers pipeline.
        if not upsampler_path:
            canonical = resolve_model_path(self.models_dir, UPSAMPLER_CP_ID)
            raise HTTPError(
                409,
                (
                    "Spatial upscaler is required for fast video generation but was not found. "
                    "The active profile's upsampler path is missing or stale, and no canonical "
                    f"upscaler is installed at {canonical}. "
                    "Install 'ltx-2.3-spatial-upscaler-x2-1.0' or update the profile's upsampler path."
                ),
                code="UPSCALER_REQUIRED",
            )
        components = self._resolve_active_components(model_selection)
        transformer_format = components.transformer_format if components is not None else "safetensors"

        # Phase 2: compute exactly one local memory plan before construction
        # (caller may pass it from the cache-matching path; otherwise derive it
        # here from the resolved components).
        if memory_plan is None:
            memory_plan = self._memory_plan_for_components(components, "standard")

        # Phase 3D (plan §12): route dev/distilled pipeline selection via
        # base_family. Unknown base family fails fast with an actionable error
        # before any heavy GPU work — never silently guess.
        base_family = components.base_family if components is not None else "distilled"
        if base_family == "unknown":
            raise HTTPError(
                409,
                (
                    "Active model profile has an unrecognized base family. The fast video "
                    "pipeline supports 'dev' and 'distilled' LTX-2.3 base models only; "
                    "the transformer path/filename did not contain a 'dev' or 'distilled' "
                    "signal. Choose an official LTX-2.3 dev or distilled transformer "
                    "(the filename must contain 'dev' or 'distilled'; note that "
                    "'distilled-lora' / 'distilled_lora' is an adapter name and does not "
                    "imply a distilled base)."
                ),
                code="UNSUPPORTED_MODEL_BASE_FAMILY",
            )

        # Dev route requires a distilled LoRA. Resolve explicit → canonical
        # fallback; if neither exists, fail before pipeline creation with the
        # exact canonical placement path(s) the user needs.
        distilled_lora_path: str | None = None
        if base_family == "dev":
            distilled_lora_path = self._resolve_distilled_lora_path(components)
            if not distilled_lora_path:
                canonical_paths = ", ".join(
                    str(p) for _role, p in self._canonical_distilled_lora_candidates()
                )
                raise HTTPError(
                    409,
                    (
                        "Dev base model requires a distilled LoRA for the fast video "
                        "pipeline, but none was found. Install one of the official "
                        f"distilled LoRAs at: {canonical_paths}."
                    ),
                    code="DISTILLED_LORA_REQUIRED",
                )

        # Gate heavy load on the memory plan (raises 409 if block offload is
        # required but unavailable).
        self._enforce_memory_plan_gate(memory_plan)

        with memory_trace.phase("pipeline_create:video"):
            pipeline = self._fast_video_pipeline_class.create(
                checkpoint_path,
                gemma_root,
                upsampler_path,
                self.config.device,
                self._local_offload_mode(),
                components=components,
                transformer_format=transformer_format,
                distilled_lora_path=distilled_lora_path,
                memory_plan=memory_plan,
            )

        # Cache key must reflect the effective distilled LoRA path so a dev
        # profile that toggles between explicit and canonical fallback (or
        # whose fallback appears/disappears on disk) invalidates correctly.
        # Computed via the same helper as ``_current_cache_key`` so a second
        # ``load_gpu_pipeline`` (e.g. inside ``generate_video``) cache-hits.
        # Phase 2: append the memory-plan cache-key parts so a strategy/
        # quantization change invalidates the pipeline cache.
        effective_cache_key = self._video_cache_key_for_components(components, model_selection)
        effective_cache_key = (*effective_cache_key, *memory_plan.cache_key_parts)

        state = VideoPipelineState(
            pipeline=pipeline,
            is_compiled=False,
            cache_key=effective_cache_key,
        )
        return self._compile_if_enabled(state, memory_plan)

    def unload_gpu_pipeline(self) -> None:
        with self._lock:
            self._ensure_no_running_generation()
            self.state.gpu_slot = None
            self._assert_invariants()
        self._gpu_cleaner.cleanup()

    def park_image_generation_pipeline_on_cpu(self) -> None:
        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if not isinstance(active, ImageGenerationPipeline):
                return

            if isinstance(self.state.active_generation, GpuGeneration) and isinstance(
                self.state.active_generation.state, GenerationRunning
            ):
                raise RuntimeError("Cannot park image generation pipeline while generation is running")

            image_generation_pipeline = active
            self.state.gpu_slot = None

        assert image_generation_pipeline is not None
        image_generation_pipeline.to("cpu")
        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.cpu_slot = CpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

    def load_image_generation_pipeline_to_gpu(self) -> ImageGenerationPipeline:
        with self._lock:
            if self.state.gpu_slot is not None:
                active = self.state.gpu_slot.active_pipeline
                if isinstance(active, ImageGenerationPipeline):
                    return active
                self._ensure_no_running_generation()

        image_generation_pipeline: ImageGenerationPipeline | None = None

        with self._lock:
            match self.state.cpu_slot:
                case CpuSlot(active_pipeline=stored):
                    image_generation_pipeline = stored
                    self.state.cpu_slot = None
                case _:
                    image_generation_pipeline = None

        if image_generation_pipeline is None:
            zit_path = get_existing_cp_path(self.models_dir, IMG_GEN_MODEL_CP_ID)
            image_generation_pipeline = self._image_generation_pipeline_class.create(str(zit_path), self._runtime_device)
        else:
            image_generation_pipeline.to(self._runtime_device)

        self._gpu_cleaner.cleanup()

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=image_generation_pipeline)
            self._assert_invariants()

        return image_generation_pipeline

    def _evict_gpu_pipeline_for_swap(self) -> None:
        should_park_image_generation_pipeline = False
        should_cleanup = False

        with self._lock:
            self._ensure_no_running_generation()
            if self.state.gpu_slot is None:
                return

            active = self.state.gpu_slot.active_pipeline
            if isinstance(active, ImageGenerationPipeline):
                should_park_image_generation_pipeline = True
            else:
                self.state.gpu_slot = None
                self._assert_invariants()
                should_cleanup = True

        if should_park_image_generation_pipeline:
            with memory_trace.phase("pipeline_evict:gpu_swap"):
                self.park_image_generation_pipeline_on_cpu()
        elif should_cleanup:
            with memory_trace.phase("pipeline_evict:gpu_swap"):
                self._gpu_cleaner.cleanup()

    def load_gpu_pipeline(
        self,
        model_type: VideoPipelineModelType,
        model_selection: ModelSelectionID | None = None,
    ) -> VideoPipelineState:
        self._install_text_patches_if_needed()

        # Phase 2: compute exactly one memory plan for both cache matching and
        # construction (the VRAM probe happens outside the lock).
        components = self._resolve_active_components(model_selection)
        memory_plan = self._memory_plan_for_components(components, "standard")

        state: VideoPipelineState | None = None
        with self._lock:
            if self._pipeline_matches_model_type(model_type, model_selection, memory_plan):
                match self.state.gpu_slot:
                    case GpuSlot(active_pipeline=VideoPipelineState() as existing_state):
                        state = existing_state
                    case _:
                        pass

        if state is None:
            with memory_trace.phase("pipeline_load:video"):
                self._evict_gpu_pipeline_for_swap()
                state = self._create_video_pipeline(model_type, model_selection, memory_plan)
        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    # ------------------------------------------------------------------
    # HDR IC-LoRA (dedicated two-stage video/sequence-input workflow)
    # ------------------------------------------------------------------

    def _hdr_cache_key(
        self,
        components: ResolvedLtxComponents | None,
        model_selection: ModelSelectionID | None,
        hdr_lora_path: str,
        scene_embeddings_path: str,
        effective_distilled_lora_path: str | None,
        memory_plan: LocalMemoryPlan | None = None,
    ) -> tuple[str, ...]:
        """Cache key for the HDR IC-LoRA pipeline state.

        Includes the active component cache key (which already keys on
        selection/profile/transformer path/format/base family), the literal
        ``"hdr_ic_lora_official_v1"`` discriminator (official-parity rebuild
        generation tag), the model selection id (or empty), the HDR LoRA
        path, the scene-embeddings path (so changing scene embeddings
        invalidates the cache), and the effective distilled LoRA path (or
        empty) so toggling any of these invalidates the cache. Phase 2:
        appends the memory-plan cache-key parts so a strategy/quantization
        change invalidates the HDR cache.
        """
        if components is None:
            component_key: tuple[str, ...] = ()
        else:
            component_key = components.cache_key
        key = (
            *component_key,
            "hdr_ic_lora_official_v1",
            model_selection or "",
            hdr_lora_path,
            scene_embeddings_path,
            effective_distilled_lora_path or "",
        )
        if memory_plan is not None:
            key = (*key, *memory_plan.cache_key_parts)
        return key

    def load_hdr_ic_lora(
        self,
        model_selection: ModelSelectionID | None,
        hdr_lora_path: str,
        scene_embeddings_path: str,
    ) -> HdrICLoraState:
        """Load (or cache-hit) the dedicated HDR IC-LoRA two-stage pipeline.

        HDR accepts dev, distilled, split, Kijai, and GGUF component builds.
        ``checkpoint_path`` passed to ``create`` is the effective builder path
        (single for a monolith, tuple for split/GGUF) via
        ``checkpoint_path_arg(components)``. LoRA selection and post-build
        component patching are owned by ``LTXHdrIcLoraPipeline.create``. This
        handler surfaces actionable error codes:

        - ``UNSUPPORTED_MODEL_BASE_FAMILY`` (409) when the base family is
          ``unknown`` (dev/distilled are supported; distilled LoRA is required
          for dev).
        - ``DISTILLED_LORA_REQUIRED`` (409) when the dev base has no usable
          distilled LoRA (explicit profile path or canonical fallback).
        - ``UPSCALER_REQUIRED`` (409) when no usable spatial upscaler exists.

        ``scene_embeddings_path`` is forwarded into the pipeline ``create()``
        and included in the cache key. Cache hit returns the existing
        ``HdrICLoraState`` when the computed cache key matches.
        """
        self._install_text_patches_if_needed()

        components = self._resolve_active_components(model_selection)
        # ``_resolve_active_components`` already validates a live selection
        # (``UNSUPPORTED_MODEL_SELECTION`` / ``MODEL_SELECTION_NOT_INSTALLED`` /
        # ``MODEL_SELECTION_REQUIRES_PROFILE``). HDR ``create`` owns LoRA
        # selection, quantization, and post-build component patching, so the
        # handler no longer forces a single selected transformer path.

        # Validate base family before any heavy work. Dev and distilled are
        # supported; reject unknown families explicitly (never silently guess).
        base_family = components.base_family if components is not None else "distilled"
        if base_family == "unknown":
            raise HTTPError(
                409,
                (
                    "HDR IC-LoRA supports 'dev' and 'distilled' LTX-2.3 base models. "
                    "The active model's base family could not be recognized "
                    f"(base_family={base_family!r}). Choose an official LTX-2.3 "
                    "dev or distilled transformer."
                ),
                code="UNSUPPORTED_MODEL_BASE_FAMILY",
            )

        # Dev base requires a distilled LoRA (applied first @ 0.5 by the
        # pipeline); distilled base skips it. Resolve explicit → canonical
        # fallback; if neither exists, fail before pipeline creation.
        effective_distilled_lora_path: str | None = None
        if base_family == "dev":
            effective_distilled_lora_path = self._resolve_distilled_lora_path(components)
            if not effective_distilled_lora_path:
                canonical_paths = ", ".join(
                    str(p) for _role, p in self._canonical_distilled_lora_candidates()
                )
                raise HTTPError(
                    409,
                    (
                        "Dev base model requires a distilled LoRA for HDR IC-LoRA, "
                        "but none was found. Install one of the official distilled "
                        f"LoRAs at: {canonical_paths}."
                    ),
                    code="DISTILLED_LORA_REQUIRED",
                )

        # Phase 2: compute exactly one memory plan for the HDR workflow and
        # build the cache key with its parts so a strategy/quantization change
        # invalidates the HDR cache. (components is resolved above; base_family
        # is dev/distilled here.)
        memory_plan = self._memory_plan_for_components(components, "hdr")
        cache_key = self._hdr_cache_key(
            components,
            model_selection,
            hdr_lora_path,
            scene_embeddings_path,
            effective_distilled_lora_path,
            memory_plan,
        )

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=HdrICLoraState(cache_key=cached_key) as state) if cached_key == cache_key:
                    return state
                case _:
                    pass

        # Gate heavy load on the memory plan before any GPU work.
        self._enforce_memory_plan_gate(memory_plan)

        # Upsampler is required for the HDR two-stage spatial upsample step.
        # ``_resolve_profile_upsampler_path`` short-circuits to "" when there is
        # no active profile (components is None); fall back to the canonical
        # models-dir upscaler so the no-profile downloaded-bundle path (which
        # ``create_fake_model_files`` installs) works for HDR too. Mirror the
        # fast-pipeline UPSCALER_REQUIRED guard so the error is actionable and
        # surfaces before any GPU work.
        upsampler_path = self._resolve_profile_upsampler_path()
        if not upsampler_path:
            canonical = resolve_model_path(self.models_dir, UPSAMPLER_CP_ID)
            if canonical.exists():
                upsampler_path = str(canonical)
        if not upsampler_path:
            canonical = resolve_model_path(self.models_dir, UPSAMPLER_CP_ID)
            raise HTTPError(
                409,
                (
                    "Spatial upscaler is required for HDR IC-LoRA generation but was not found. "
                    "The active profile's upsampler path is missing or stale, and no canonical "
                    f"upscaler is installed at {canonical}. "
                    "Install 'ltx-2.3-spatial-upscaler-x2-1.0' or update the profile's upsampler path."
                ),
                code="UPSCALER_REQUIRED",
            )

        # Resolve checkpoint paths (threads model_selection) and derive the
        # transformer format for the HDR loader path. ``checkpoint_path`` is
        # already ``checkpoint_path_arg(components)`` when components is not
        # None (single path for a monolith, tuple for split/GGUF); for the
        # no-profile downloaded-bundle path it is a single model CP path.
        checkpoint_path, gemma_root, _upsampler, _resolved_cache_key = self._resolve_checkpoint_paths(model_selection)
        transformer_format = components.transformer_format if components is not None else "safetensors"

        with memory_trace.phase("pipeline_load:hdr_ic_lora"):
            self._evict_gpu_pipeline_for_swap()

            with memory_trace.phase("pipeline_create:hdr_ic_lora"):
                pipeline = self._hdr_ic_lora_pipeline_class.create(
                    checkpoint_path=checkpoint_path,
                    upsampler_path=upsampler_path,
                    hdr_lora_path=hdr_lora_path,
                    scene_embeddings_path=scene_embeddings_path,
                    device=self.config.device,
                    components=components,
                    transformer_format=transformer_format,
                    base_family=base_family,
                    distilled_lora_path=effective_distilled_lora_path,
                    gemma_root=gemma_root,
                    offload_mode=self._local_offload_mode(),
                    memory_plan=memory_plan,
                )

            state = HdrICLoraState(pipeline=pipeline, cache_key=cache_key)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_ic_lora(
        self,
        lora_paths: list[str],
        depth_model_path: str | None,
        adapter_path: str | None = None,
        lora_strength: float = 1.0,
    ) -> ICLoraState:
        self._install_text_patches_if_needed()

        # Phase 2: compute exactly one memory plan for the (standard) IC-LoRA
        # workflow and derive the cache key from it.
        components = self._resolve_active_components()
        memory_plan = self._memory_plan_for_components(components, "standard")
        cache_key = (
            *self._current_cache_key(None, memory_plan),
            "ic_lora",
            adapter_path or "",
            depth_model_path or "",
            f"lora_strength={lora_strength:.4f}",
            *lora_paths,
        )

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=ICLoraState(cache_key=cached_key) as state) if cached_key == cache_key:
                    return state
                case _:
                    pass

        # Gate heavy load on the memory plan before any GPU work.
        self._enforce_memory_plan_gate(memory_plan)

        with memory_trace.phase("pipeline_load:ic_lora"):
            self._evict_gpu_pipeline_for_swap()
            checkpoint_path, gemma_root, upsampler_path, _cache_key = self._resolve_checkpoint_paths()

            with memory_trace.phase("pipeline_create:ic_lora"):
                pipeline = self._ic_lora_pipeline_class.create(
                    checkpoint_path,
                    gemma_root,
                    upsampler_path,
                    lora_paths,
                    self.config.device,
                    self._local_offload_mode(),
                    components=components,
                    lora_strength=lora_strength,
                    memory_plan=memory_plan,
                )
            depth_pipeline: DepthProcessorPipeline | None = None
            if depth_model_path is not None:
                depth_pipeline = self._depth_processor_pipeline_class.create(depth_model_path, self.config.device)
            state = ICLoraState(
                pipeline=pipeline,
                lora_paths=lora_paths,
                lora_strength=lora_strength,
                depth_pipeline=depth_pipeline,
                depth_model_path=depth_model_path,
                adapter_path=adapter_path,
                cache_key=cache_key,
            )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_a2v_pipeline(self) -> A2VPipelineState:
        self._install_text_patches_if_needed()

        # Phase 2: compute exactly one memory plan for the (standard) A2V
        # workflow and derive the cache key from it.
        components = self._resolve_active_components()
        memory_plan = self._memory_plan_for_components(components, "standard")
        cache_key = (*self._current_cache_key(None, memory_plan), "a2v")

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=A2VPipelineState(cache_key=cached_key) as state) if cached_key == cache_key:
                    return state
                case _:
                    pass

        # Gate heavy load on the memory plan before any GPU work.
        self._enforce_memory_plan_gate(memory_plan)

        with memory_trace.phase("pipeline_load:a2v"):
            self._evict_gpu_pipeline_for_swap()
            checkpoint_path, gemma_root, upsampler_path, _cache_key = self._resolve_checkpoint_paths()

            with memory_trace.phase("pipeline_create:a2v"):
                pipeline = self._a2v_pipeline_class.create(
                    checkpoint_path,
                    gemma_root,
                    upsampler_path,
                    self.config.device,
                    self._local_offload_mode(),
                    components=components,
                    memory_plan=memory_plan,
                )
            state = A2VPipelineState(pipeline=pipeline, cache_key=cache_key)

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state

    def load_retake_pipeline(self, *, distilled: bool = True) -> RetakePipelineState:
        self._install_text_patches_if_needed()

        quantized = device_supports_fp8(self.config.device)

        # Phase 2: compute exactly one memory plan for the (standard) retake
        # workflow and derive the cache key from it.
        components = self._resolve_active_components()
        memory_plan = self._memory_plan_for_components(components, "standard")
        cache_key = (
            *self._current_cache_key(None, memory_plan),
            "retake",
            *memory_plan.cache_key_parts,
        )

        with self._lock:
            match self.state.gpu_slot:
                case GpuSlot(active_pipeline=RetakePipelineState(cache_key=cached_key) as state) if cached_key == cache_key:
                    return state
                case _:
                    pass

        # Gate heavy load on the memory plan before any GPU work.
        self._enforce_memory_plan_gate(memory_plan)

        with memory_trace.phase("pipeline_load:retake"):
            self._evict_gpu_pipeline_for_swap()

            checkpoint_path, gemma_root, _upsampler_path, _cache_key = self._resolve_checkpoint_paths()
            # build_policy needs a single checkpoint path; split-safetensors
            # checkpoints arrive as a tuple of shards — read from the first shard
            # (for non-prequant checkpoints this is equivalent to the old path-less
            # fp8_cast(), and the retake pipeline overrides split+fp8 with its own
            # kijai guard regardless). Net gating is unchanged.
            if quantized:
                from ltx_core.quantization.fp8_cast import build_policy

                cp = checkpoint_path[0] if isinstance(checkpoint_path, tuple) else checkpoint_path
                quantization = build_policy(cp)
            else:
                quantization = None
            with memory_trace.phase("pipeline_create:retake"):
                pipeline = self._retake_pipeline_class.create(
                    checkpoint_path=checkpoint_path,
                    gemma_root=gemma_root,
                    device=self.config.device,
                    offload_mode=self._local_offload_mode(),
                    components=components,
                    loras=[],
                    quantization=quantization,
                    memory_plan=memory_plan,
                )
            state = RetakePipelineState(
                pipeline=pipeline,
                distilled=distilled,
                quantized=quantized,
                cache_key=cache_key,
            )

        with self._lock:
            self.state.gpu_slot = GpuSlot(active_pipeline=state)
            self._assert_invariants()
        return state
